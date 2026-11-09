"""按真实root姿态审计目标几何可达性和名义静态负载。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def origin_pose(element: ET.Element | None) -> tuple[np.ndarray, np.ndarray]:
    """Read a URDF origin in parent coordinates.

    Args:
        element: Optional origin with meter xyz and radian fixed-axis rpy.

    Returns:
        Rotation [3,3] and translation [3], identity/zero when absent.
    """
    if element is None:
        return np.eye(3), np.zeros(3)
    return (Rotation.from_euler('xyz',np.fromstring(element.get('rpy','0 0 0'),sep=' ')).as_matrix(),
            np.fromstring(element.get('xyz','0 0 0'),sep=' '))


class ArmGeometry:
    """Evaluate the active six-joint URDF chain without simulator dependencies."""

    def __init__(self, urdf_path: Path) -> None:
        """Load actual joint transforms, hard limits, inertials and fixed children.

        Args:
            urdf_path: V6 active URDF; rotations in radians, geometry in meters.

        Raises:
            KeyError: The expected base-to-arm_seg6 chain is absent.
        """
        robot = ET.parse(urdf_path).getroot()
        self.links = {link.get('name'):link for link in robot.findall('link')}
        parents = {joint.find('child').get('link'):joint for joint in robot.findall('joint')}
        chain, child = [], 'arm_seg6'
        while child != 'base_link':
            joint = parents[child]
            chain.append(joint)
            child = joint.find('parent').get('link')
        self.chain = list(reversed(chain))
        self.arm_names = [f'arm_joint{i}' for i in range(1,7)]
        self.limits = np.array([[float(joint.find('limit').get(key)) for key in ('lower','upper')]
            for name in self.arm_names for joint in self.chain if joint.get('name')==name])
        self.efforts = np.array([float(joint.find('limit').get('effort'))
            for name in self.arm_names for joint in self.chain if joint.get('name')==name])
        self.transforms = [(joint,*origin_pose(joint.find('origin'))) for joint in self.chain]
        self.fixed_children = []
        included = {'arm_seg6'}
        pending = [joint for joint in robot.findall('joint') if joint.get('type')=='fixed']
        while True:
            found = [joint for joint in pending if joint.find('parent').get('link') in included]
            if not found:
                break
            for joint in found:
                self.fixed_children.append((joint,*origin_pose(joint.find('origin'))))
                included.add(joint.find('child').get('link'));pending.remove(joint)
        self.tcp_offset = np.array([0.,0.,.07961206])
        # Triangle inequality from joint1's fixed anchor is a necessary reach bound.
        first_index = next(i for i,(joint,_,_) in enumerate(self.transforms) if joint.get('name')=='arm_joint1')
        self.reach_bound_m = sum(np.linalg.norm(t) for _,_,t in self.transforms[first_index+1:]) + np.linalg.norm(self.tcp_offset)
        _,_,self.anchor = self.forward(np.zeros((1,6)))
        self.anchor = self.anchor[0,0]

    def forward(self, joint_positions: np.ndarray) -> tuple[dict[str, tuple[np.ndarray,np.ndarray,int]], np.ndarray, np.ndarray]:
        """Compute base-frame link poses and world-in-base joint axes/origins.

        Args:
            joint_positions: Six arm positions [N,6] in radians.

        Returns:
            Link pose/count mapping, joint axes [N,6,3], joint origins [N,6,3].
        """
        count=len(joint_positions)
        rotation=np.broadcast_to(np.eye(3),(count,3,3)).copy()
        translation=np.zeros((count,3))
        poses: dict[str,tuple[np.ndarray,np.ndarray,int]]={}
        axes,origins=[],[]
        active_count=0
        for joint,local_rotation,local_translation in self.transforms:
            translation=translation+np.einsum('nij,j->ni',rotation,local_translation)
            rotation=rotation@local_rotation
            if joint.get('name') in self.arm_names:
                index=self.arm_names.index(joint.get('name'))
                axis=np.fromstring(joint.find('axis').get('xyz'),sep=' ')
                axis/=np.linalg.norm(axis)
                origins.append(translation.copy());axes.append(np.einsum('nij,j->ni',rotation,axis))
                rotation=rotation@Rotation.from_rotvec(joint_positions[:,index,None]*axis).as_matrix()
                active_count+=1
            poses[joint.find('child').get('link')]=(rotation.copy(),translation.copy(),active_count)
        for joint,local_rotation,local_translation in self.fixed_children:
            parent_rotation,parent_translation,parent_count=poses[joint.find('parent').get('link')]
            poses[joint.find('child').get('link')]=(parent_rotation@local_rotation,
                parent_translation+np.einsum('nij,j->ni',parent_rotation,local_translation),parent_count)
        return poses,np.stack(axes,1),np.stack(origins,1)

    def tcp_and_jacobian(self, positions: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
        """Evaluate one TCP and its six-column translational Jacobian.

        Args:
            positions: Arm joint positions [6] in radians.

        Returns:
            TCP [3] meters and Jacobian [3,6] meters/radian in base coordinates.
        """
        poses,axes,origins=self.forward(positions[None])
        rotation,translation,_=poses['arm_seg6']
        point=translation[0]+rotation[0]@self.tcp_offset
        jacobian=np.cross(axes[0],point-origins[0]).T
        return point,jacobian

    def static_loads(self, positions: np.ndarray, root_rotations: np.ndarray,
                     force_world: np.ndarray, tool_mass_delta: np.ndarray) -> np.ndarray:
        """Estimate fixed-root contact-free equilibrium torques at actual poses.

        Args:
            positions: Actual arm positions [N,6], radians.
            root_rotations: Base-to-world rotations [N,3,3].
            force_world: Actual injected tool-COM force [N,3], newtons.
            tool_mass_delta: Saved arm_seg6 mass adjustments [N], kilograms.

        Returns:
            Required static arm torques [N,6], N m; excludes inertia and contacts.

        Note:
            Uses URDF mass/COM and saved tool mass delta. This is a diagnostic
            static proxy, never a necessary condition for every dynamic maneuver.
        """
        poses,axes,origins=self.forward(positions)
        gravity_base=np.einsum('nji,j->ni',root_rotations,np.array([0.,0.,-9.81]))
        torques=np.zeros((len(positions),6))
        for name,(rotation,translation,active_count) in poses.items():
            inertial=self.links[name].find('inertial')
            if not active_count or inertial is None:
                continue
            _,local_com=origin_pose(inertial.find('origin'))
            com=translation+np.einsum('nij,j->ni',rotation,local_com)
            mass=np.full(len(positions),float(inertial.find('mass').get('value')))
            if name=='arm_seg6':
                mass+=tool_mass_delta
                external_base=np.einsum('nji,nj->ni',root_rotations,force_world)
                external_jacobian=np.cross(axes,com[:,None]-origins)
                torques+=np.einsum('nkj,nj->nk',external_jacobian,external_base)
            jacobian=np.cross(axes[:,:active_count],com[:,None]-origins[:,:active_count])
            torques[:,:active_count]+=np.einsum('nkj,nj->nk',jacobian,gravity_base*mass[:,None])
        return -torques


def main() -> None:
    """Audit all snapshots by a reach bound and bounded multi-start position IK.

    Raises:
        OSError: Evidence, properties or active URDF cannot be read.
        AssertionError: Analytic Jacobian does not match finite differences.

    Note:
        IK uses hard limits and position only, no self-collision or velocity check.
        Failed numerical search is reported as unresolved, not unreachable.
    """
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--urdf',type=Path,required=True)
    args=parser.parse_args()
    arm=ArmGeometry(args.urdf)
    nominal=np.array([0.,1.,-1.5,0.,0.,0.])
    point,jacobian=arm.tcp_and_jacobian(nominal)
    numeric=np.column_stack([(arm.tcp_and_jacobian(nominal+np.eye(6)[i]*1e-6)[0]-arm.tcp_and_jacobian(nominal-np.eye(6)[i]*1e-6)[0])/2e-6 for i in range(6)])
    assert np.max(np.abs(jacobian-numeric))<1e-7
    properties=json.loads((args.root/'creation_properties.json').read_text())
    reports=[]
    for directory in sorted(args.root.glob('check_*')):
        if not directory.is_dir() or not (directory/'summary.json').exists():
            continue
        if (directory/'workspace_result.json').exists():
            reports.append(json.loads((directory/'workspace_result.json').read_text()))
            continue
        summary=json.loads((directory/'summary.json').read_text())
        arrays=np.load(directory/'measurements.npz')
        roots=arrays['root'].reshape(-1,13)
        rotations=Rotation.from_quat(roots[:,3:7]).as_matrix()
        joints=arrays['joint_position'].reshape(-1,14)
        arm_indices=[summary['dof_names'].index(name) for name in arm.arm_names]
        arm_positions=joints[:,arm_indices]
        mass_delta=np.tile(np.array(properties[str(summary['iteration'])])[:,4],len(roots)//128)
        static_torques=arm.static_loads(arm_positions,rotations,arrays['external_force_world'].reshape(-1,3),mass_delta)
        actual_saturation=arrays['snapshot_saturated_fraction'].reshape(-1,14)[:,8:]
        report: dict[str,Any]={'iteration':summary['iteration'],'snapshot_count':len(roots),
            'static_required_over_limit_fraction_by_arm_joint':(np.abs(static_torques)>arm.efforts).mean(0).tolist(),
            'static_required_over_limit_any_fraction':float((np.abs(static_torques)>arm.efforts).any(1).mean()),
            'actual_substep_saturation_mean_in_same_snapshots':actual_saturation.mean(0).tolist(),
            'targets':{}}
        for target_name in ('geometric_target','compensated_target'):
            targets_world=arrays[target_name].reshape(-1,3)
            targets=np.einsum('nji,nj->ni',rotations,targets_world-roots[:,:3])
            distances=np.linalg.norm(targets-arm.anchor,axis=1)
            impossible=distances>arm.reach_bound_m+1e-6
            selected=np.linspace(0,len(roots)-1,96,dtype=int)
            rng=np.random.default_rng(20261108)
            solutions=[]
            for index in selected:
                row={'snapshot_index':int(index),'outside_chain_bound':bool(impossible[index]),'distance_from_joint1_m':float(distances[index])}
                if not impossible[index]:
                    starts=[np.clip(arm_positions[index],arm.limits[:,0]+1e-7,arm.limits[:,1]-1e-7),nominal,arm.limits.mean(1)]
                    starts += list(rng.uniform(arm.limits[:,0],arm.limits[:,1],(3,6)))
                    best_error,best_q=float('inf'),None
                    for initial in starts:
                        result=least_squares(lambda q:arm.tcp_and_jacobian(q)[0]-targets[index],initial,
                            jac=lambda q:arm.tcp_and_jacobian(q)[1],bounds=(arm.limits[:,0],arm.limits[:,1]),
                            max_nfev=100,ftol=1e-9,xtol=1e-9,gtol=1e-9)
                        error=np.linalg.norm(result.fun)
                        if error<best_error:
                            best_error,best_q=float(error),result.x.tolist()
                        if error<1e-3:
                            break
                    row.update(residual_m=best_error,position_solution_within_1mm=best_error<1e-3,joint_solution_rad=best_q)
                solutions.append(row)
            report['targets'][target_name]={'outside_chain_bound_count':int(impossible.sum()),
                'outside_chain_bound_fraction':float(impossible.mean()),'distance_p95_m':float(np.quantile(distances,.95)),
                'ik_sample_count':len(selected),'ik_outside_bound':sum(row['outside_chain_bound'] for row in solutions),
                'ik_found_position_solution':sum(row.get('position_solution_within_1mm',False) for row in solutions),
                'ik_unresolved_inside_bound':sum(not row['outside_chain_bound'] and not row.get('position_solution_within_1mm',False) for row in solutions),
                'ik_samples':solutions}
        reports.append(report)
        (directory/'workspace_result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'iteration':summary['iteration'],'targets':{k:{a:b for a,b in v.items() if a!='ik_samples'} for k,v in report['targets'].items()}}),flush=True)
    (args.root/'workspace_check.json').write_text(json.dumps({'reach_bound_m':arm.reach_bound_m,
        'joint1_anchor_base_m':arm.anchor.tolist(),'jacobian_max_abs_error':float(np.max(np.abs(jacobian-numeric))),
        'reports':reports,'说明':'固定实际root的位置必要界与数值位置IK；不证明动态／无碰撞可达，不把IK未收敛等同不可达。静力代理不含惯性、接触和控制余量。'},
        ensure_ascii=False,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()

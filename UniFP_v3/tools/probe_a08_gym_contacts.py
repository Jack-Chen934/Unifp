"""Check discrete A08 IK poses for contacts in the active desktop Gym asset."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

from isaacgym import gymapi, gymtorch
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'legged_gym/utils'))
from v3_asset_collision import apply_desktop_assembly_filters


def probe(audit_path: Path, asset_path: Path, manifest_path: Path,
          sample_start: int, sample_count: int, contact_offset: float) -> Dict[str, Any]:
    """Run one-step contact checks for saved six-axis IK poses.

    Args:
        audit_path: Offline A08 JSON containing spherical points and joint solutions.
        asset_path: Active desktop URDF path loaded into Isaac Gym.
        manifest_path: Manifest describing the local approved collision filters.
        sample_start: First zero-based IK solution index to inspect.
        sample_count: Number of consecutive saved IK solutions to inspect.
        contact_offset: PhysX contact shell size in meters for this diagnostic.

    Returns:
        Per-sample state vectors, reported body pairs, and their maximum overlap.

    Raises:
        RuntimeError: Isaac Gym cannot create the simulation or import the URDF.
        IndexError: The requested solution range exceeds the saved audit data.

    Note:
        Uses a fixed base, zero gravity, one physics step per independently
        reset pose, and no learning or controller. The IK points are not treated
        as a continuous trajectory. Imported collision-shape thickness remains
        fixed at 0.01 m while ``contact_offset`` is varied.
    """
    audit = json.loads(audit_path.read_text())
    points = audit['samples_spherical_r_m_elevation_rad_azimuth_rad']
    solutions = audit['joint_solutions_rad']
    sample_stop = sample_start + sample_count
    if sample_stop > len(solutions):
        raise IndexError(f'Requested {sample_start}:{sample_stop} from {len(solutions)} IK poses')

    gym = gymapi.acquire_gym()
    parameters = gymapi.SimParams()
    parameters.dt = 0.005
    parameters.up_axis = gymapi.UP_AXIS_Z
    parameters.gravity = gymapi.Vec3(0, 0, 0)
    parameters.use_gpu_pipeline = False
    parameters.physx.use_gpu = True
    parameters.physx.num_position_iterations = 4
    parameters.physx.num_velocity_iterations = 1
    parameters.physx.contact_offset = contact_offset
    parameters.physx.rest_offset = 0.0
    sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, parameters)
    if sim is None:
        raise RuntimeError('Gym create_sim failed')
    try:
        options = gymapi.AssetOptions()
        options.fix_base_link = True
        options.collapse_fixed_joints = False
        options.replace_cylinder_with_capsule = False
        options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        options.thickness = 0.01
        asset = gym.load_asset(sim, str(asset_path.parent), asset_path.name, options)
        if asset is None:
            raise RuntimeError('Gym asset import failed')
        env = gym.create_env(sim, gymapi.Vec3(-2, -2, 0), gymapi.Vec3(2, 2, 2), 1)
        root_pose = gymapi.Transform()
        root_pose.p.z = 0.34
        actor = gym.create_actor(env, asset, root_pose, 'a08_contact_probe', 0, 0, 0)
        body_names = gym.get_actor_rigid_body_names(env, actor)
        dof_names = gym.get_actor_dof_names(env, actor)
        dof_properties = gym.get_actor_dof_properties(env, actor)
        dof_properties['stiffness'] = 0.0
        gym.set_actor_dof_properties(env, actor, dof_properties)
        filter_rules = apply_desktop_assembly_filters(gym, env, actor, manifest_path)
        gym.prepare_sim(sim)
        zero_torque = torch.zeros(len(dof_names), dtype=torch.float32)
        samples: List[Dict[str, Any]] = []
        for sample_index in range(sample_start, sample_stop):
            state = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
            joint_positions = np.zeros(len(dof_names), dtype=np.float32)
            for joint_index in range(6):
                joint_positions[dof_names.index(f'arm_joint{joint_index + 1}')] = solutions[sample_index][joint_index]
            state['pos'] = joint_positions
            state['vel'] = 0
            gym.set_actor_dof_states(env, actor, state, gymapi.STATE_ALL)
            gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(zero_torque))
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            contacts = []
            for contact in gym.get_env_rigid_contacts(env):
                body0 = int(contact['body0'])
                body1 = int(contact['body1'])
                contacts.append({
                    'body0': body_names[body0] if body0 >= 0 else 'ground',
                    'body1': body_names[body1] if body1 >= 0 else 'ground',
                    'initial_overlap_m': float(contact['initialOverlap']),
                })
            pair_groups: Dict[str, Dict[str, Any]] = {}
            for contact in contacts:
                pair = ' <-> '.join(sorted((contact['body0'], contact['body1'])))
                summary = pair_groups.setdefault(pair, {'count': 0, 'max_overlap_m': 0.0})
                summary['count'] += 1
                summary['max_overlap_m'] = max(summary['max_overlap_m'], contact['initial_overlap_m'])
            samples.append({
                'sample_index': sample_index,
                'target_spherical_m_rad_rad': points[sample_index],
                'arm_joint_positions_rad': solutions[sample_index],
                'reported_contact_count': len(contacts),
                'contact_pairs': pair_groups,
            })
        return {
            'audit': str(audit_path),
            'active_asset': str(asset_path),
            'active_asset_sha256': hashlib.sha256(asset_path.read_bytes()).hexdigest(),
            'sample_start': sample_start,
            'sample_count': sample_count,
            'contact_offset_m': contact_offset,
            'collision_shape_thickness_m': options.thickness,
            'local_collision_filter_rules': filter_rules,
            'samples': samples,
            'limitations': [
                '仅检测保存IK姿态的离散Gym接触候选，不检查姿态间连续运动或关节分支可执行性',
                '固定底座、零重力和单步静态检查不代表自由底盘动态稳定性',
                'initialOverlap受PhysX接触偏移影响，不单独等同于实际碰撞力',
            ],
        }
    finally:
        gym.destroy_sim(sim)


def main() -> None:
    """Parse batch contact settings and write the diagnostic report.

    Returns:
        None.

    Note:
        Writes the requested JSON output and does not start training.
    """
    parser = argparse.ArgumentParser(description='A08离散IK姿态Gym接触抽查')
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--asset', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--sample-start', type=int, default=173)
    parser.add_argument('--sample-count', type=int, default=132)
    parser.add_argument('--contact-offset', type=float, default=0.001)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = probe(args.audit, args.asset, args.manifest, args.sample_start,
                   args.sample_count, args.contact_offset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    candidates = sum(sample['reported_contact_count'] > 0 for sample in report['samples'])
    print(json.dumps({'姿态数': len(report['samples']), '存在接触候选的姿态数': candidates,
                      '报告': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()

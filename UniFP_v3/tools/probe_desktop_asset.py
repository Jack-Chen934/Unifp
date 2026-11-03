"""Probe the real Gym import without changing training or source assets."""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

from isaacgym import gymapi, gymtorch
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'legged_gym/utils'))
from v3_asset_collision import apply_desktop_assembly_filters


ASSET_ROOT = Path(__file__).resolve().parents[1] / 'resources/robots/v3_a1_desktop'
DEFAULT_ASSET = ASSET_ROOT / 'urdf/a1arm_rgm3_desktop_fixed_gripper.urdf'


def body_properties(gym: Any, env: Any, actor: int) -> List[Dict[str, Any]]:
    """Read physical properties indexed by actor body name.

    Args:
        gym: Isaac Gym interface.
        env: Environment containing the actor.
        actor: Actor handle within the environment.

    Returns:
        Mass (kg), COM (m), inertia and inverse inertia as engine matrices.
    """
    result = []
    names = gym.get_actor_rigid_body_names(env, actor)
    for name, prop in zip(names, gym.get_actor_rigid_body_properties(env, actor)):
        matrix = prop.inertia
        result.append({'name': name, 'mass': prop.mass,
                       'com': [prop.com.x, prop.com.y, prop.com.z],
                       'inertia': [[matrix.x.x, matrix.x.y, matrix.x.z],
                                   [matrix.y.x, matrix.y.y, matrix.y.z],
                                   [matrix.z.x, matrix.z.y, matrix.z.z]]})
    return result


def probe(args: argparse.Namespace) -> Dict[str, Any]:
    """Import, step and inspect one isolated robot in GPU PhysX.

    Args:
        args: Asset, root height (m), duration (s), ground/gravity/fixed-root
            switches, collision filter, movement, TCP obstacle and recompute controls.

    Returns:
        JSON-compatible import, physical readback and contact diagnostic.

    Raises:
        RuntimeError: Simulator or URDF import fails.

    Note:
        Runs only the diagnostic simulation; destroys it on completion.
    """
    gym = gymapi.acquire_gym()
    parameters = gymapi.SimParams()
    parameters.dt = 0.005
    parameters.up_axis = gymapi.UP_AXIS_Z
    parameters.gravity = gymapi.Vec3(0, 0, -9.81 if args.gravity else 0)
    parameters.use_gpu_pipeline = False
    parameters.physx.use_gpu = True
    parameters.physx.num_position_iterations = 4
    parameters.physx.num_velocity_iterations = 1
    parameters.physx.contact_offset = args.contact_offset
    parameters.physx.rest_offset = 0.0
    sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, parameters)
    if sim is None:
        raise RuntimeError('Gym create_sim failed')
    try:
        if args.ground:
            plane = gymapi.PlaneParams()
            plane.normal = gymapi.Vec3(0, 0, 1)
            plane.static_friction = plane.dynamic_friction = 1.0
            gym.add_ground(sim, plane)
        options = gymapi.AssetOptions()
        options.fix_base_link = args.fixed_base
        options.collapse_fixed_joints = False
        options.replace_cylinder_with_capsule = False
        options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        options.thickness = 0.01
        asset = gym.load_asset(sim, str(args.asset.parent), args.asset.name, options)
        if asset is None:
            raise RuntimeError('Gym asset import failed')
        env = gym.create_env(sim, gymapi.Vec3(-2, -2, 0), gymapi.Vec3(2, 2, 2), 1)
        pose = gymapi.Transform()
        pose.p.z = args.height
        actor = gym.create_actor(env, asset, pose, 'desktop_probe', 0, args.collision_filter, 0)
        names = gym.get_actor_rigid_body_names(env, actor)
        contact_names = list(names)
        dof_names = gym.get_actor_dof_names(env, actor)
        shape_props = gym.get_actor_rigid_shape_properties(env, actor)
        initial_filters = [prop.filter for prop in shape_props]
        spans = gym.get_actor_rigid_body_shape_indices(env, actor)
        if args.exclude_pairs:
            # This is a named-pair diagnostic, never an actor-wide exemption.
            pairs = json.loads(args.exclude_pairs.read_text())
            for bit, pair in enumerate(pairs):
                for body in pair:
                    span = spans[names.index(body)]
                    for index in range(span.start, span.start + span.count):
                        shape_props[index].filter |= 1 << bit
            gym.set_actor_rigid_shape_properties(env, actor, shape_props)
        rules = []
        if args.assembly_filters:
            rules = apply_desktop_assembly_filters(gym, env, actor, args.asset.parent.parent / 'asset_manifest.json')
        before = body_properties(gym, env, actor)
        if args.recompute:
            props = gym.get_actor_rigid_body_properties(env, actor)
            gym.set_actor_rigid_body_properties(env, actor, props, recomputeInertia=True)
        after = body_properties(gym, env, actor)
        dof_properties = gym.get_actor_dof_properties(env, actor)
        if args.zero_imported_stiffness:
            dof_properties['stiffness'] = 0.0
            gym.set_actor_dof_properties(env, actor, dof_properties)
        state = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
        targets = np.zeros(len(dof_names), dtype=np.float32)
        for name, angle in [('arm_joint2', 1.0), ('arm_joint3', -1.5)]:
            if name in dof_names:
                targets[dof_names.index(name)] = angle
        if args.arm_angles:
            for index, angle in enumerate(args.arm_angles, 1):
                targets[dof_names.index('arm_joint' + str(index))] = angle
        state['pos'] = targets
        state['vel'] = 0
        gym.set_actor_dof_states(env, actor, state, gymapi.STATE_ALL)
        if args.obstacle_at_tcp:
            from compare_desktop_fk import forward_kinematics

            # Independent FK places a test object before prepare_sim; the robot
            # state above uses the same URDF joint coordinates and base height.
            base_pose = np.eye(4)
            base_pose[2, 3] = args.height
            poses = forward_kinematics(args.asset, dict(zip(dof_names, targets.tolist())), base_pose)
            tcp_world = poses['arm_seg6'] @ np.array([0.0, 0.0, 0.07961206, 1.0])
            obstacle_options = gymapi.AssetOptions()
            obstacle_options.fix_base_link = True
            obstacle = gym.create_box(sim, 0.08, 0.08, 0.08, obstacle_options)
            obstacle_pose = gymapi.Transform()
            obstacle_pose.p = gymapi.Vec3(*tcp_world[:3])
            gym.create_actor(env, obstacle, obstacle_pose, 'tcp_test_box', 0, 0, 0)
            contact_names.append('tcp_test_box')
        gym.prepare_sim(sim)
        net_forces = gymtorch.wrap_tensor(gym.acquire_net_contact_force_tensor(sim))
        dof_force = torch.zeros(len(dof_names), dtype=torch.float32)
        peaks = np.zeros(len(names))
        first_forces = None
        first_contacts = []
        peak_base_contacts = []
        for step in range(max(1, round(args.seconds / parameters.dt))):
            current_state = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
            # Hold the nominal pose while applying small wheel group targets.
            torques = 50.0 * (targets - current_state['pos']) - 2.0 * current_state['vel']
            for index, name in enumerate(dof_names):
                if 'wheel_drive' in name:
                    torques[index] = 2.0 * ((0.5 if args.move else 0.0) - current_state['vel'][index])
                elif 'steering' in name and args.move:
                    torques[index] = 10.0 * (0.1 - current_state['pos'][index]) - current_state['vel'][index]
            limits = gym.get_actor_dof_properties(env, actor)['effort']
            torques = np.clip(torques, -limits, limits).astype(np.float32)
            dof_force[:] = torch.as_tensor(torques, device=dof_force.device)
            gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(dof_force))
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.refresh_net_contact_force_tensor(sim)
            force_values = net_forces[:len(names)].cpu().numpy().copy()
            base_index = names.index('base_link')
            if np.linalg.norm(force_values[base_index]) > peaks[base_index]:
                peak_base_contacts = [
                    {'step': step,
                     'body0': contact_names[int(contact['body0'])] if int(contact['body0']) >= 0 else 'ground',
                     'body1': contact_names[int(contact['body1'])] if int(contact['body1']) >= 0 else 'ground',
                     'local_position0_m': [float(value) for value in contact['localPos0']],
                     'local_position1_m': [float(value) for value in contact['localPos1']],
                     'overlap': float(contact['initialOverlap']), 'lambda': float(contact['lambda'])}
                    for contact in gym.get_env_rigid_contacts(env)
                    if base_index in (int(contact['body0']), int(contact['body1']))]
            peaks = np.maximum(peaks, np.linalg.norm(force_values, axis=1))
            if step == 0:
                first_forces = force_values.tolist()
                for contact in gym.get_env_rigid_contacts(env):
                    body0 = int(contact['body0'])
                    body1 = int(contact['body1'])
                    first_contacts.append({'body0': contact_names[body0] if body0 >= 0 else 'ground',
                                           'body1': contact_names[body1] if body1 >= 0 else 'ground',
                                           'local_position0_m': [float(value) for value in contact['localPos0']],
                                           'local_position1_m': [float(value) for value in contact['localPos1']],
                                           'overlap': float(contact['initialOverlap']),
                                           'lambda': float(contact['lambda'])})
        final_state = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_ALL)
        final_dofs = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
        return {'asset': str(args.asset), 'parameters': vars(args), 'dofs': dof_names,
                'dof_properties': {name: {field: dof_properties[field][index].item() for field in dof_properties.dtype.names}
                                   for index, name in enumerate(dof_names)},
                'last_applied_torques_Nm': dict(zip(dof_names, torques.tolist())),
                'properties_before': before, 'properties_after': after,
                'shape_filters_before': initial_filters,
                'shape_filters_after': [prop.filter for prop in gym.get_actor_rigid_shape_properties(env, actor)],
                'local_collision_rules': rules,
                'shape_spans': [{'body': name, 'start': span.start, 'count': span.count}
                                for name, span in zip(names, spans)],
                'first_forces_N': dict(zip(names, first_forces)),
                'final_forces_N': dict(zip(names, force_values.tolist())),
                'peak_forces_N': dict(zip(names, peaks.tolist())), 'first_contacts': first_contacts,
                'peak_base_contacts': peak_base_contacts,
                'final_positions_m': {name: [float(final_state['pose']['p'][axis][i]) for axis in 'xyz']
                                      for i, name in enumerate(names)},
                'final_orientations_xyzw': {
                    name: [float(final_state['pose']['r'][axis][i]) for axis in 'xyzw']
                    for i, name in enumerate(names)},
                'final_dof_positions_rad': dict(zip(dof_names, final_dofs['pos'].tolist())),
                'final_dof_velocities_rad_s': dict(zip(dof_names, final_dofs['vel'].tolist())),
                'finite': bool(np.isfinite(final_dofs['pos']).all() and np.isfinite(net_forces.numpy()).all())}
    finally:
        gym.destroy_sim(sim)


def main() -> None:
    """Parse the diagnostic controls and write a machine-readable report.

    Returns:
        None.

    Note:
        Writes the requested report and prints a compact Chinese summary.
    """
    parser = argparse.ArgumentParser(description='桌面基准候选资产Gym短测')
    parser.add_argument('--asset', type=Path, default=DEFAULT_ASSET)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=0.005)
    parser.add_argument('--height', type=float, default=0.34)
    parser.add_argument('--collision-filter', type=int, default=0)
    parser.add_argument('--contact-offset', type=float, default=0.01)
    parser.add_argument('--exclude-pairs', type=Path)
    parser.add_argument('--assembly-filters', action='store_true')
    parser.add_argument('--ground', action='store_true')
    parser.add_argument('--gravity', action='store_true')
    parser.add_argument('--fixed-base', action='store_true')
    parser.add_argument('--recompute', action='store_true')
    parser.add_argument('--move', action='store_true')
    parser.add_argument('--arm-angles', type=float, nargs=6)
    parser.add_argument('--zero-imported-stiffness', action='store_true')
    parser.add_argument('--obstacle-at-tcp', action='store_true')
    args = parser.parse_args()
    report = probe(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + '\n')
    print(json.dumps({'报告': str(args.output), 'DOF': len(report['dofs']),
                      '质量kg': sum(p['mass'] for p in report['properties_before']),
                      'base峰值N': report['peak_forces_N'].get('base_link'), '有限值': report['finite']}, ensure_ascii=False))


if __name__ == '__main__':
    main()

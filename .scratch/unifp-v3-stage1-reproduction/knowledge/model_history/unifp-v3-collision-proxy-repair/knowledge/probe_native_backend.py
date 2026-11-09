"""Locate native import failures independently of Torch CUDA kernels."""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

from isaacgym import gymapi
import numpy as np


def probe(arguments: argparse.Namespace) -> Dict[str, Any]:
    """Exercise actor property writes and physics with CPU-side state arrays.

    Args:
        arguments: Asset/source paths, PhysX backend, filter/aggregate switches,
            initial pose timing, and number of native 2.5 ms simulation steps.

    Returns:
        Import identity, backend, DOF positions in radians, and finite-state verdict.

    Raises:
        RuntimeError: Simulation creation or finite-state validation fails.
        OSError: The asset or collision manifest cannot be read.

    Note:
        Prints stage boundaries with flushing to locate native aborts. Writes
        no training state and always attempts native simulator destruction.
    """
    sys.path.insert(0, str(arguments.source_root / 'legged_gym/utils'))
    from v3_asset_collision import apply_desktop_assembly_filters

    gym = gymapi.acquire_gym()
    parameters = gymapi.SimParams()
    parameters.dt = 0.0025
    parameters.up_axis = gymapi.UP_AXIS_Z
    parameters.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
    parameters.use_gpu_pipeline = False
    parameters.physx.use_gpu = arguments.backend == 'gpu'
    parameters.physx.num_threads = 4
    parameters.physx.max_gpu_contact_pairs = 2 ** 25
    print('[native-probe] create_sim', flush=True)
    simulation = gym.create_sim(0, -1, gymapi.SIM_PHYSX, parameters)
    if simulation is None:
        raise RuntimeError('Unable to create PhysX simulation')
    try:
        options = gymapi.AssetOptions()
        options.collapse_fixed_joints = False
        options.replace_cylinder_with_capsule = False
        options.flip_visual_attachments = False
        options.default_dof_drive_mode = int(gymapi.DOF_MODE_EFFORT)
        print('[native-probe] load_asset', flush=True)
        asset = gym.load_asset(simulation, str(arguments.asset.parent), arguments.asset.name, options)
        environment = gym.create_env(simulation, gymapi.Vec3(-2, -2, 0), gymapi.Vec3(2, 2, 2), 1)
        root_pose = gymapi.Transform()
        root_pose.p.z = 0.34
        if arguments.aggregate:
            gym.begin_aggregate(environment, gym.get_asset_rigid_body_count(asset),
                                gym.get_asset_rigid_shape_count(asset), True)
        actor = gym.create_actor(environment, asset, root_pose, 'native_probe', 0, 0, 0)
        if arguments.aggregate:
            gym.end_aggregate(environment)
        joint_names = gym.get_actor_dof_names(environment, actor)
        body_names = gym.get_actor_rigid_body_names(environment, actor)
        spans = gym.get_actor_rigid_body_shape_indices(environment, actor)
        shape_counts = {name: span.count for name, span in zip(body_names, spans)}
        print('[native-probe] actor ' + json.dumps(shape_counts), flush=True)
        if arguments.filters:
            print('[native-probe] apply_filters', flush=True)
            apply_desktop_assembly_filters(gym, environment, actor, arguments.asset.parents[1] / 'asset_manifest.json')
            print('[native-probe] filters_complete', flush=True)
        dof_properties = gym.get_actor_dof_properties(environment, actor)
        dof_properties['stiffness'][:] = 0.0
        dof_properties['driveMode'][:] = int(gymapi.DOF_MODE_EFFORT)
        gym.set_actor_dof_properties(environment, actor, dof_properties)
        joint_states = gym.get_actor_dof_states(environment, actor, gymapi.STATE_ALL)
        joint_states['pos'][:] = 0.0
        joint_states['vel'][:] = 0.0
        if arguments.pose != 'zero':
            joint_states['pos'][joint_names.index('arm_joint2')] = 1.0
            joint_states['pos'][joint_names.index('arm_joint3')] = -1.5
        if arguments.pose == 'before':
            print('[native-probe] set_dof_before_prepare', flush=True)
            gym.set_actor_dof_states(environment, actor, joint_states, gymapi.STATE_ALL)
        print('[native-probe] prepare_sim', flush=True)
        gym.prepare_sim(simulation)
        if arguments.pose == 'after':
            print('[native-probe] set_dof_after_prepare', flush=True)
            gym.set_actor_dof_states(environment, actor, joint_states, gymapi.STATE_ALL)
        print('[native-probe] simulate', flush=True)
        for _ in range(arguments.steps):
            gym.simulate(simulation)
            gym.fetch_results(simulation, True)
        final_states = gym.get_actor_dof_states(environment, actor, gymapi.STATE_ALL)
        finite = bool(np.isfinite(final_states['pos']).all() and np.isfinite(final_states['vel']).all())
        if not finite:
            raise RuntimeError('Non-finite native DOF state')
        print('[native-probe] simulation_complete', flush=True)
        return {'asset': str(arguments.asset), 'backend': arguments.backend,
                'torch_state_device': 'not_used', 'gpu_pipeline': False,
                'filters': arguments.filters, 'aggregate': arguments.aggregate,
                'pose': arguments.pose,
                'steps': arguments.steps, 'dof_names': joint_names,
                'body_shape_counts': shape_counts, 'finite': finite,
                'final_dof_positions_rad': final_states['pos'].tolist()}
    finally:
        print('[native-probe] destroy_sim', flush=True)
        gym.destroy_sim(simulation)
        print('[native-probe] destroyed', flush=True)


def main() -> None:
    """Parse diagnostic options and persist a report only after clean destruction.

    Returns:
        None.

    Raises:
        OSError: The requested report cannot be written.

    Note:
        Creates the report parent directory and writes JSON with native identity.
    """
    parser = argparse.ArgumentParser(description='CPU状态数组下的原生PhysX后端对照')
    parser.add_argument('--asset', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--backend', choices=('cpu', 'gpu'), required=True)
    parser.add_argument('--pose', choices=('zero', 'before', 'after'), default='before')
    parser.add_argument('--filters', action='store_true')
    parser.add_argument('--aggregate', action='store_true')
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    report = probe(arguments)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print('[native-probe] report=' + str(arguments.output), flush=True)


if __name__ == '__main__':
    main()

"""Check the actual V3 step and indexed reset using CPU tensors."""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

from isaacgym import gymapi, gymutil
import torch


def probe(arguments: argparse.Namespace) -> Dict[str, Any]:
    """Run nominal V3 diagnostics with CPU tensors and the selected PhysX backend.

    Args:
        arguments: V3 source root, isolated asset path, CPU/GPU PhysX backend,
            and number of 20 ms policy steps in each motion segment.

    Returns:
        Device identity, physical failures, finite-state, physical property,
        imported shape settings and indexed-reset checks for three segments.

    Raises:
        RuntimeError: V3 creates GPU tensors, non-finite states, or changes
            unselected environments during indexed reset.

    Note:
        Disables randomization and push only in this diagnostic configuration.
        Runs no policy optimizer and destroys the simulator before returning.
    """
    sys.path.insert(0, str(arguments.source_root))
    from legged_gym.envs.v3.legged_robot_v3 import V3PosForce
    from legged_gym.envs.v3.v3_pos_force_config import V3PosForceCfg
    from legged_gym.utils.helpers import class_to_dict, set_seed

    set_seed(1)
    torch.set_num_threads(4)
    configuration = V3PosForceCfg()
    configuration.asset.file = str(arguments.asset.resolve())
    configuration.env.num_envs = 4
    configuration.noise.add_noise = False
    for setting in ('randomize_base_mass', 'randomize_base_com', 'randomize_tool_mass',
                    'randomize_friction', 'randomize_execution_scale', 'push_robots'):
        setattr(configuration.domain_rand, setting, False)
    parameters = gymapi.SimParams()
    gymutil.parse_sim_config(class_to_dict(configuration.sim), parameters)
    parameters.use_gpu_pipeline = False
    parameters.physx.use_gpu = arguments.backend == 'gpu'
    device = 'cuda:0' if arguments.backend == 'gpu' else 'cpu'
    print('[cpu-tensor-probe] constructing', flush=True)
    environment = V3PosForce(configuration, parameters, gymapi.SIM_PHYSX, device, True)
    try:
        if environment.device != 'cpu':
            raise RuntimeError('CPU tensor pipeline was not selected')
        actions = torch.zeros(4, 14)
        segments = []
        for segment_name in ('stationary', 'wheel_motion', 'arm_motion'):
            actions.zero_()
            if segment_name == 'wheel_motion':
                actions[:, 4:8] = 0.1
            if segment_name == 'arm_motion':
                environment.reset_idx(torch.arange(4, dtype=torch.long))
                actions[:, 8:] = 0.1
            failures = 0
            peak_base_contact_n = 0.0
            for _ in range(arguments.steps):
                _, _, _, information = environment.step(actions)
                failures += int(information['terminated'].sum())
                peak_base_contact_n = max(peak_base_contact_n, float(torch.linalg.vector_norm(
                    environment.contact_forces[:, environment.base_body_index], dim=-1).max()))
            segments.append({'name': segment_name, 'steps': arguments.steps,
                             'physical_failures': failures, 'peak_base_contact_n': peak_base_contact_n,
                             'mean_drive_velocity_rad_s': environment.dof_vel[:, environment.wheel_dof_indices].mean(0).tolist()})
        # Indexed reset must preserve task rows and the flattened DOF rows of
        # the other three environments, including after native CPU readback.
        preserved = {name: getattr(environment, name)[1:].clone()
                     for name in ('root_states', 'commands', 'obs_history', 'episode_length_buf')}
        preserved_dof_state = environment.dof_state.view(4, 14, 2)[1:].clone()
        environment.reset_idx(torch.tensor([0], dtype=torch.long))
        differences = {name: float((previous - getattr(environment, name)[1:]).abs().max())
                       for name, previous in preserved.items()}
        differences['dof_state'] = float((preserved_dof_state - environment.dof_state.view(4, 14, 2)[1:]).abs().max())
        if max(differences.values()) > 1e-5:
            raise RuntimeError('Indexed reset changed other environment rows')
        environment._validate_simulation_state()
        # Read real actor properties after creation, rather than inferring them
        # from XML; this exposes unintended inertia recomputation during import.
        actor_properties = environment.gym.get_actor_rigid_body_properties(
            environment.envs[0], environment.actor_handles[0])
        body_names = environment.gym.get_actor_rigid_body_names(
            environment.envs[0], environment.actor_handles[0])
        properties = [{'name': name, 'mass': prop.mass,
                       'com': [prop.com.x, prop.com.y, prop.com.z],
                       'inertia': [[getattr(getattr(prop.inertia, row), column)
                                    for column in 'xyz'] for row in 'xyz']}
                      for name, prop in zip(body_names, actor_properties)]
        shape_properties = environment.gym.get_actor_rigid_shape_properties(
            environment.envs[0], environment.actor_handles[0])
        return {'asset': str(arguments.asset), 'backend': arguments.backend,
                'tensor_device': environment.device, 'num_envs': 4,
                'shape_count': len(environment.gym.get_actor_rigid_shape_properties(
                    environment.envs[0], environment.actor_handles[0])),
                'segments': segments, 'properties': properties,
                'shape_contact_offsets_m': [prop.contact_offset for prop in shape_properties],
                'shape_rest_offsets_m': [prop.rest_offset for prop in shape_properties],
                'shape_filters': [prop.filter for prop in shape_properties],
                'local_reset_other_rows_max_difference': differences,
                'finite': True, 'note_zh': '名义短测，无PPO；不代替长时间运动与策略验收。'}
    finally:
        print('[cpu-tensor-probe] destroying', flush=True)
        environment.gym.destroy_sim(environment.sim)
        print('[cpu-tensor-probe] destroyed', flush=True)


def main() -> None:
    """Parse the isolated probe inputs and write evidence after clean destruction.

    Returns:
        None.

    Raises:
        OSError: The evidence path cannot be written.

    Note:
        Writes JSON and creates its parent directory without changing run files.
    """
    parser = argparse.ArgumentParser(description='V3 CPU tensor与CPU/GPU PhysX短测')
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--asset', type=Path, required=True)
    parser.add_argument('--backend', choices=('cpu', 'gpu'), required=True)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    report = probe(arguments)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

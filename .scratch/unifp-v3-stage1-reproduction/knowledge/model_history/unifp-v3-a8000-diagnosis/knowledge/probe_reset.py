"""Inspect real reset boundaries using URDF FK and the frozen GPU environment."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

from isaacgym import gymapi, gymtorch, gymutil
import numpy as np
import torch
from scipy.spatial.transform import Rotation


def snapshot(environment: Any, label: str) -> Dict[str, Any]:
    """Compare tensor readbacks with independent FK at one reset boundary.

    Args:
        environment: Frozen V3 environment; first actor is inspected.
        label: Human-readable boundary identifier stored in the JSON report.

    Returns:
        TCP and initial target positions in world meters plus cache/FK error.
    """
    from compare_desktop_fk import forward_kinematics

    root = environment.root_states[0].detach().cpu().numpy()
    positions = environment.dof_pos[0].detach().cpu().tolist()
    base_pose = np.eye(4)
    base_pose[:3, :3] = Rotation.from_quat(root[3:7]).as_matrix()
    base_pose[:3, 3] = root[:3]
    poses = forward_kinematics(Path(environment.cfg.asset.file),
                               dict(zip(environment.dof_names, positions)), base_pose)
    expected_tcp = (poses['arm_seg6'] @ np.asarray([0.0, 0.0, 0.07961206, 1.0]))[:3]
    cached_tcp = environment._tcp_position()[0].detach().cpu().numpy()
    target = environment._geometric_target_world()[0].detach().cpu().numpy()
    return {'boundary': label, 'tcp_fk_world_m': expected_tcp.tolist(),
            'tcp_cache_world_m': cached_tcp.tolist(), 'target_world_m': target.tolist(),
            'cache_fk_error_m': float(np.linalg.norm(cached_tcp - expected_tcp)),
            'target_fk_error_m': float(np.linalg.norm(target - expected_tcp)),
            'target_cache_error_m': float(np.linalg.norm(target - cached_tcp))}


def simulate_policy_interval(environment: Any, actions: torch.Tensor) -> None:
    """Advance 20ms through existing force control without task/reset mutation.

    Args:
        environment: GPU environment whose native control function is used.
        actions: Dimensionless policy-order joint actions with shape [N,14].

    Returns:
        None.

    Note:
        Only this isolated simulator changes; no optimizer or checkpoint exists.
    """
    for _ in range(environment.cfg.control.decimation):
        torques = environment._compute_v3_torques(actions)
        environment.gym.set_dof_actuation_force_tensor(environment.sim, gymtorch.unwrap_tensor(torques))
        environment.gym.simulate(environment.sim)
        environment.gym.refresh_dof_state_tensor(environment.sim)
    environment.gym.refresh_actor_root_state_tensor(environment.sim)
    environment.gym.refresh_rigid_body_state_tensor(environment.sim)
    environment.gym.refresh_net_contact_force_tensor(environment.sim)
    environment._update_base_state()


def main() -> None:
    """Run two reset boundaries and persist diagnostic evidence.

    Raises:
        RuntimeError: Gym creation or environment validation fails.
        OSError: The output file cannot be written.

    Note:
        Uses four nominal actors with noise/push disabled only in this probe.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root))
    sys.path.insert(0, str(args.source_root / 'tools'))
    from legged_gym.envs.v3.legged_robot_v3 import V3PosForce
    from legged_gym.envs.v3.v3_pos_force_config import V3PosForceCfg
    from legged_gym.utils.helpers import class_to_dict, set_seed

    set_seed(1)
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    cfg = V3PosForceCfg()
    cfg.env.num_envs = 4
    cfg.noise.add_noise = False
    for field in ('randomize_base_mass', 'randomize_base_com', 'randomize_tool_mass',
                  'randomize_friction', 'randomize_execution_scale', 'push_robots'):
        setattr(cfg.domain_rand, field, False)
    parameters = gymapi.SimParams()
    gymutil.parse_sim_config(class_to_dict(cfg.sim), parameters)
    parameters.use_gpu_pipeline = True
    parameters.physx.use_gpu = True
    environment = V3PosForce(cfg, parameters, gymapi.SIM_PHYSX, 'cuda:0', True)
    try:
        actions = torch.zeros(4, 14, device=environment.device)
        records = [snapshot(environment, 'constructor_reset')]
        for _ in range(5):
            environment.gym.refresh_rigid_body_state_tensor(environment.sim)
        records.append(snapshot(environment, 'repeat_refresh_without_simulate'))
        simulate_policy_interval(environment, actions)
        records.append(snapshot(environment, 'first_physics_interval'))
        for _ in range(15):
            simulate_policy_interval(environment, actions)
        environment.root_states[0, 0] += 0.7
        actor_ids = torch.tensor([0], device=environment.device, dtype=torch.int32)
        environment.gym.set_actor_root_state_tensor_indexed(
            environment.sim, gymtorch.unwrap_tensor(environment.root_states),
            gymtorch.unwrap_tensor(actor_ids), 1)
        simulate_policy_interval(environment, actions)
        records.append(snapshot(environment, 'before_local_reset_after_displacement'))
        other_states = environment.root_states[1:].clone()
        environment.reset_idx(actor_ids.long())
        records.append(snapshot(environment, 'after_local_reset'))
        other_difference = float((environment.root_states[1:] - other_states).abs().max())
        simulate_policy_interval(environment, actions)
        records.append(snapshot(environment, 'physics_after_local_reset'))
        report = {'source_root': str(args.source_root), 'seed': 1, 'num_envs': 4,
                  'records': records, 'other_root_rows_reset_difference': other_difference,
                  'reset_cache_consistent': all(row['cache_fk_error_m'] < 1e-4 for row in records
                                                 if row['boundary'] in ('constructor_reset', 'after_local_reset'))}
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps(report, allow_nan=False), flush=True)
    finally:
        environment.gym.destroy_sim(environment.sim)


if __name__ == '__main__':
    main()

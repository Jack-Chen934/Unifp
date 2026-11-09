"""Check actual GPU reset targets and labels against independent URDF FK."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from isaacgym import gymapi, gymtorch, gymutil
from isaacgym.torch_utils import quat_apply
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from compare_desktop_fk import forward_kinematics
from legged_gym.envs.v3.legged_robot_v3 import V3PosForce
from legged_gym.envs.v3.v3_pos_force_config import V3PosForceCfg
from legged_gym.utils.helpers import class_to_dict, set_seed


def measure_boundary(environment: Any, boundary: str) -> Dict[str, Any]:
    """Read TCP, target and labels at a real simulator lifecycle boundary.

    Args:
        environment: Four-actor V3 environment, with actor zero inspected.
        boundary: Boundary name preserved in the diagnostic report.

    Returns:
        Independent FK errors in meters and label consistency in scaled units.
    """
    root_state = environment.root_states[0].cpu().numpy()
    root_pose = np.eye(4)
    root_pose[:3, :3] = Rotation.from_quat(root_state[3:7]).as_matrix()
    root_pose[:3, 3] = root_state[:3]
    joint_positions = dict(zip(environment.dof_names, environment.dof_pos[0].cpu().tolist()))
    poses = forward_kinematics(Path(environment.cfg.asset.file), joint_positions, root_pose)
    expected_tcp_m = (poses['arm_seg6'] @ np.array([0, 0, 0.07961206, 1]))[:3]
    actual_tcp_m = environment._tcp_position()[0].cpu().numpy()
    body = environment.rigid_state[0, environment.tcp_body_index]
    cached_tcp_m = (body[:3] + quat_apply(body[3:7].unsqueeze(0),
                    body.new_tensor([[0, 0, 0.07961206]]))[0]).cpu().numpy()
    _, critic, labels = environment._build_frames(torch.tensor([0], device=environment.device))
    scaled_tcp = environment._tcp_spherical_position()[0] * body.new_tensor([0.5, 1, 1.3])
    return {'boundary': boundary, 'tcp_world_m': actual_tcp_m.tolist(),
            'tcp_fk_error_m': float(np.linalg.norm(actual_tcp_m - expected_tcp_m)),
            'raw_cache_fk_error_m': float(np.linalg.norm(cached_tcp_m - expected_tcp_m)),
            'target_fk_error_m': float(np.linalg.norm(environment._geometric_target_world()[0].cpu().numpy() - expected_tcp_m)),
            'label_error': float((labels[0, 3:6] - scaled_tcp).abs().max()),
            'critic_label_error': float((critic[0, :12] - labels[0]).abs().max())}


def advance_physics(environment: Any, actions: torch.Tensor) -> None:
    """Advance one native policy interval without task counters or automatic resets.

    Args:
        environment: Isolated GPU V3 environment.
        actions: Dimensionless policy-order actions [4,14].

    Note:
        Mutates only this simulation's state; no optimizer or training run exists.
    """
    for _ in range(environment.cfg.control.decimation):
        torques = environment._compute_v3_torques(actions)
        environment.gym.set_dof_actuation_force_tensor(environment.sim, gymtorch.unwrap_tensor(torques))
        environment.gym.simulate(environment.sim)
        environment.gym.refresh_dof_state_tensor(environment.sim)
    environment.gym.refresh_actor_root_state_tensor(environment.sim)
    environment.gym.refresh_rigid_body_state_tensor(environment.sim)
    environment._update_base_state()


def main() -> None:
    """Validate constructor/local resets and write evidence before asserting success.

    Raises:
        AssertionError: Reset targets/labels differ from current state or leak to other rows.
        OSError: The specified output report cannot be written.

    Note:
        Creates four nominal actors on the GPU selected by CUDA_VISIBLE_DEVICES,
        then destroys the simulation. Does not modify source assets or checkpoints.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--asset', type=Path)
    args = parser.parse_args()
    set_seed(1)
    torch.set_num_threads(4)
    cfg = V3PosForceCfg()
    cfg.env.num_envs = 4
    if args.asset is not None:
        cfg.asset.file = str(args.asset)
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
        records = [measure_boundary(environment, 'constructor_reset')]
        actions = torch.zeros(4, 14, device=environment.device)
        for _ in range(15):
            advance_physics(environment, actions)
        environment.root_states[0, 0] += 0.7
        actor_ids = torch.tensor([0], device=environment.device, dtype=torch.int32)
        environment.gym.set_actor_root_state_tensor_indexed(
            environment.sim, gymtorch.unwrap_tensor(environment.root_states),
            gymtorch.unwrap_tensor(actor_ids), 1)
        advance_physics(environment, actions)
        before = {name: getattr(environment, name)[1:].clone() for name in
                  ('root_states', 'dof_pos', 'dof_vel', 'obs_history', 'critic_history', 'episode_length_buf')}
        counters = (environment.completed_iterations, environment.training_policy_steps, environment.training_transitions)
        environment.reset_idx(actor_ids.long())
        records.append(measure_boundary(environment, 'local_reset'))
        differences = {name: float((getattr(environment, name)[1:] - value).abs().max())
                       for name, value in before.items()}
        counters_unchanged = counters == (
            environment.completed_iterations, environment.training_policy_steps, environment.training_transitions)
        advance_physics(environment, actions)
        records.append(measure_boundary(environment, 'after_physics'))
        passed = (all(row['tcp_fk_error_m'] <= 1e-4 and row['label_error'] <= 1e-6
                      and row['critic_label_error'] <= 1e-6 for row in records)
                  and all(row['target_fk_error_m'] <= 1e-4 for row in records[:2])
                  and all(value == 0 for value in differences.values()) and counters_unchanged)
        report = {'passed': passed, 'num_envs': 4, 'seed': 1, 'records': records,
                  'other_row_differences': differences, 'training_counters_unchanged': counters_unchanged}
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps(report), flush=True)
        assert passed, 'Reset TCP, target, labels or local isolation failed'
    finally:
        environment.gym.destroy_sim(environment.sim)


if __name__ == '__main__':
    main()

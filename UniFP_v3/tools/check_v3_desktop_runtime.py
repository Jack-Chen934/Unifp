"""Check the production V3 creation chain and an indexed reset on four robots."""

import json
from pathlib import Path
import sys
from typing import Any, Dict

from isaacgym import gymapi
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import legged_gym.envs
from legged_gym.envs.v3.v3_pos_force_config import V3PosForceCfg
from legged_gym.envs.v3.legged_robot_v3 import V3PosForce


def check_runtime() -> Dict[str, Any]:
    """Exercise 32 real V3 actors using only small deterministic actions.

    Returns:
        Runtime shape, termination, physics and local-reset evidence.

    Raises:
        AssertionError: A required model or indexed-reset contract fails.

    Note:
        Disables observation noise and nominal-property randomization for this
        check only; no learner is created and no training is started.
    """
    cfg = V3PosForceCfg()
    num_envs = 32
    cfg.env.num_envs = num_envs
    cfg.domain_rand.randomize_base_mass = False
    cfg.domain_rand.randomize_friction = False
    cfg.domain_rand.randomize_base_com = False
    cfg.domain_rand.randomize_tool_mass = False
    cfg.domain_rand.randomize_execution_scale = False
    cfg.domain_rand.push_robots = False
    cfg.noise.add_noise = False
    params = gymapi.SimParams()
    params.dt = cfg.sim.dt
    params.up_axis = gymapi.UP_AXIS_Z
    params.gravity = gymapi.Vec3(0, 0, -9.81)
    params.use_gpu_pipeline = False
    params.physx.use_gpu = True
    params.physx.num_position_iterations = 4
    params.physx.num_velocity_iterations = 1
    params.physx.contact_offset = 0.01
    params.physx.rest_offset = 0.0
    env = V3PosForce(cfg, params, gymapi.SIM_PHYSX, 'cuda:0', True)
    try:
        assert env.max_episode_length == 1000
        initial_common_step_counter = env.common_step_counter
        initial_episode_lengths = env.episode_length_buf.clone()
        initial_commands = env.commands.clone()
        env.set_training_progress(0, 0)
        env.set_training_progress(0, 23)
        env.set_training_progress(1, 24)
        assert env.completed_iterations == 1
        assert env.training_policy_steps == 24
        assert env.training_transitions == 24 * num_envs
        assert env.common_step_counter == initial_common_step_counter
        assert torch.equal(env.episode_length_buf, initial_episode_lengths)
        assert torch.equal(env.commands, initial_commands)
        for invalid_progress in ((2, 24), (1, 23)):
            try:
                env.set_training_progress(*invalid_progress)
            except ValueError:
                pass
            else:
                raise AssertionError(f"Accepted inconsistent progress {invalid_progress}")
        initial_target = env.commands[:, 3:6].clone()
        assert torch.allclose(initial_target, env.target_start)
        assert torch.allclose(initial_target, env._tcp_spherical_position(), atol=1e-6)
        assert torch.count_nonzero(env.commands[:, 9:15]) == 0
        assert torch.all((env.target_end[:, 0] >= 0.45) & (env.target_end[:, 0] <= 0.70))
        angle_limit = torch.deg2rad(torch.tensor(60.0, device=env.device))
        assert torch.all((env.target_end[:, 1] >= 0.0) & (env.target_end[:, 1] <= angle_limit))
        assert torch.all((env.target_end[:, 2] >= -angle_limit) & (env.target_end[:, 2] <= angle_limit))
        assert cfg.goal_ee.num_path_check_samples == 10
        screen_ids = torch.tensor([0, 1], device=env.device)
        valid_path, ground_rejected = env._screen_target_paths(
            screen_ids, env.target_start[screen_ids], env.target_end[screen_ids]
        )
        assert torch.all(valid_path) and not torch.any(ground_rejected)
        below_ground_end = env.target_start[screen_ids].clone()
        below_ground_end[:, 0] = 0.70
        below_ground_end[:, 1] = -torch.pi / 2
        rejected_path, ground_rejected = env._screen_target_paths(
            screen_ids, env.target_start[screen_ids], below_ground_end
        )
        assert torch.all(~rejected_path) and torch.all(ground_rejected)
        done_count = 0
        base_force_peak = 0.0
        for step in range(24):
            actions = torch.zeros((num_envs, 14), device=env.device)
            actions[:, :4] = 0.05
            actions[:, 4:8] = 0.05
            obs, _, dones, _ = env.step(actions)
            done_count += int(dones.sum())
            base_force_peak = max(base_force_peak, float(torch.linalg.vector_norm(
                env.contact_forces[:, env.base_body_index], dim=1).max()))
            assert all(torch.isfinite(value).all() for value in obs.values())
        assert torch.all(env.target_elapsed_steps == 24)
        expected_progress = 24.0 / env.target_move_steps.float()
        expected_goal = env.target_start + expected_progress[:, None] * (
            env.target_end - env.target_start
        )
        expected_azimuth_delta = torch.atan2(
            torch.sin(env.target_end[:, 2] - env.target_start[:, 2]),
            torch.cos(env.target_end[:, 2] - env.target_start[:, 2]),
        )
        expected_goal[:, 2] = env.target_start[:, 2] + expected_progress * expected_azimuth_delta
        assert torch.allclose(env.commands[:, 3:6], expected_goal, atol=1e-6)
        assert env.training_policy_steps == 24
        assert env.training_transitions == 24 * num_envs

        scheduled_env = torch.tensor([3], device=env.device)
        previous_command = env.commands[3, :3].clone()
        previous_goal_end = env.target_end[3].clone()
        env.episode_length_buf[3] = 250
        env.target_elapsed_steps[3] = env.target_move_steps[3] - 1
        env._advance_task_commands(scheduled_env)
        assert torch.allclose(env.commands[3, 3:6], previous_goal_end, atol=1e-6)
        assert not torch.equal(env.commands[3, :3], previous_command)
        old_goal_end = env.target_end[3].clone()
        env.target_elapsed_steps[3] = env.target_move_steps[3] + env.target_hold_steps[3] - 1
        env._advance_task_commands(scheduled_env)
        assert torch.allclose(env.target_start[3], old_goal_end)
        assert torch.allclose(env.commands[3, 3:6], old_goal_end)
        assert int(env.target_elapsed_steps[3]) == 0

        timeout_env = torch.tensor([2], device=env.device)
        timeout_elapsed = int(env.target_elapsed_steps[2]) + 1
        timeout_move_steps = int(env.target_move_steps[2])
        timeout_start = env.target_start[2].clone()
        timeout_end = env.target_end[2].clone()
        timeout_old_base_command = torch.tensor((11.0, -12.0, 13.0), device=env.device)
        env.commands[2, :3] = timeout_old_base_command
        env.episode_length_buf[2] = env.max_episode_length - 1
        _, _, timeout_dones, timeout_infos = env.step(torch.zeros((num_envs, 14), device=env.device))
        assert bool(timeout_dones[2])
        assert bool(timeout_infos['time_outs'][2])
        assert not bool(timeout_infos['terminated'][2])
        assert torch.equal(timeout_infos['terminal_env_ids'], timeout_env)
        assert timeout_infos['terminal_privileged_obs'].shape == (1, 294)
        terminal_frame = timeout_infos['terminal_privileged_obs'].view(1, 3, 98)[0, -1]
        terminal_commands = terminal_frame[80:95]
        terminal_base_command = terminal_commands[:3] / torch.tensor((2.0, 2.0, 0.25), device=env.device)
        assert not torch.equal(terminal_base_command, timeout_old_base_command), (
            f"base command did not resample: old={timeout_old_base_command.tolist()}, "
            f"terminal={terminal_base_command.tolist()}, dt={env.dt}, "
            f"interval={env.cfg.commands.resampling_time / env.dt}"
        )
        expected_terminal_goal = timeout_start + min(timeout_elapsed / timeout_move_steps, 1.0) * (
            timeout_end - timeout_start
        )
        expected_terminal_goal[2] = timeout_start[2] + min(
            timeout_elapsed / timeout_move_steps, 1.0
        ) * torch.atan2(
            torch.sin(timeout_end[2] - timeout_start[2]),
            torch.cos(timeout_end[2] - timeout_start[2]),
        )
        assert torch.allclose(
            terminal_commands[3:6] / torch.tensor((0.5, 1.0, 1.3), device=env.device),
            expected_terminal_goal,
            atol=1e-5,
        )
        assert torch.count_nonzero(terminal_commands[9:15]) == 0
        assert int(env.episode_length_buf[2]) == 0
        before_roots = env.root_states.clone()
        before_dofs = env.dof_state.view(num_envs, len(env.dof_names), 2).clone()
        before_lengths = env.episode_length_buf.clone()
        env.reset_idx(torch.tensor([1], device=env.device))
        assert env.completed_iterations == 1
        assert env.training_policy_steps == 24
        assert env.training_transitions == 24 * num_envs
        all_env_ids = torch.arange(num_envs, device=env.device)
        other_envs = all_env_ids[all_env_ids != 1]
        assert torch.equal(env.root_states[other_envs], before_roots[other_envs])
        assert torch.equal(
            env.dof_state.view(num_envs, len(env.dof_names), 2)[other_envs], before_dofs[other_envs]
        )
        assert torch.equal(env.episode_length_buf[other_envs], before_lengths[other_envs])
        assert int(env.episode_length_buf[1]) == 0
        props = env.gym.get_actor_rigid_body_properties(env.envs[0], env.actor_handles[0])
        names = env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0])
        base = props[names.index('base_link')]
        assert abs(base.mass - 65.8456) < 1e-4
        assert abs(base.inertia.x.x - 1.6878422133333333) < 1e-5
        assert done_count == 0, f'Unexpected physical terminations: {done_count}'
        return {'环境数': num_envs, '策略步': 25, 'DOF': len(env.dof_names),
                'body_count': env.num_bodies, '非超时终止': done_count,
                '初始命令从实测TCP开始': True, '零末端力命令': True,
                '官方10点路径筛查及地面拒绝': True,
                '训练时钟发布无仿真副作用': True,
                '目标插值与段切换': True, '250步底盘命令重采': True,
                '纯timeout终止快照及局部reset': True,
                'base接触峰值N': base_force_peak, '局部reset不影响其他环境': True,
                '最终episode_lengths': env.episode_length_buf.tolist(),
                '实际actor总质量kg': sum(p.mass for p in props),
                '实际base惯量对角': [base.inertia.x.x, base.inertia.y.y, base.inertia.z.z],
                '轮系实际位置rad': env.dof_pos[:, env.action_dof_indices[:8]].tolist()}
    finally:
        env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    report = check_runtime()
    output = Path(__file__).resolve().parents[2] / '.scratch/unifp-v3-stage1-reproduction/knowledge/model_history/v3-desktop-model-repair/knowledge/v3_runtime_check.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False))

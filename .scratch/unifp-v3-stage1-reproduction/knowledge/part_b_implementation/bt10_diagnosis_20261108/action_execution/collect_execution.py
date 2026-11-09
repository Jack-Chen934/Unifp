"""独立V6受力回放：同状态动作反事实与reset前TCP响应。"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import isaacgym  # noqa: F401
from isaacgym import gymapi, gymutil
import numpy as np
import torch

import legged_gym
import legged_gym.envs  # noqa: F401
from legged_gym.b2_gym_learn.ppo_cse_pf.v3_actor_critic import V3ActorCritic
from legged_gym.b2_gym_learn.ppo_cse_pf.v3_run_identity import build_run_identity, file_sha256
from legged_gym.envs.v3.legged_robot_v3 import V3PosForce
from legged_gym.utils.helpers import class_to_dict, set_seed
from legged_gym.utils.v3_config import resolve_v3_configs

from torque_requests import decompose_requests


class ExecutionEnvironment(V3PosForce):
    """Delegate original physics while measuring unexecuted candidate requests."""

    def _compute_v3_torques(self, actions: torch.Tensor) -> torch.Tensor:
        """Compare candidate requests at every actual physical substep state.

        Args:
            actions: Original clipped actions [N,14], dimensionless policy order.

        Returns:
            Original controller torque [N,14], in native DOF order and N m.

        Note:
            Only the superclass result is executed. Candidate evaluation does not
            mutate state, sample randomness, or advance the simulator.
        """
        actual_torque = super()._compute_v3_torques(actions)
        if not getattr(self, 'measurement_ready', False):
            return actual_torque
        order = self.action_dof_indices
        parts = decompose_requests(self.candidate_actions, self.dof_pos[:, order],
            self.dof_vel[:, order], self.default_dof_pos[:, order], self.dof_pos_limits[order],
            self.execution_scale_delta, self.p_gains[order], self.d_gains[order],
            self.torque_limits[order], self.cfg.normalization.clip_actions,
            self.cfg.control.action_scale, self.cfg.control.wheel_action_scale,
            self.cfg.control.wheel_velocity_limit)
        difference = (actual_torque[:, order] - parts['torque'][self.actual_variant]).abs().max()
        self.max_controller_difference = torch.maximum(self.max_controller_difference, difference)
        if not self.initial_parity_checked:
            for variant in range(3):
                original = V3PosForce._compute_v3_torques(self, parts['clipped_actions'][variant])
                torch.testing.assert_close(original[:, order], parts['torque'][variant], rtol=1e-5, atol=1e-5)
            self.initial_parity_checked = True
        for key in self.substep_sums:
            self.substep_sums[key] += parts[key]
        mean_saturated, sample_saturated = parts['saturated'][:2]
        self.overlap_counts += torch.stack((
            ~mean_saturated & ~sample_saturated,
            ~mean_saturated & sample_saturated,
            mean_saturated & ~sample_saturated,
            mean_saturated & sample_saturated,
        ))
        self.current_targets = {key: parts[key] for key in ('raw_target', 'target')}
        return actual_torque

    def _accumulate_episode_metrics(
        self, done_ids: torch.Tensor, terminated: torch.Tensor
    ) -> list[dict[str, Any]]:
        """Capture terminal-inclusive physical data before command advance/reset.

        Args:
            done_ids: Ending environment indices [K].
            terminated: Original physical termination flags [N].

        Returns:
            Original unmodified episode statistics.

        Note:
            Measurement dictionaries receive CPU copies, never live state views.
        """
        if getattr(self, 'measurement_ready', False):
            row = dict(self.pre_step)
            row.update({
                'post_tcp': self._tcp_position(),
                'post_geometric': self._geometric_target_world(),
                'post_target': self._compensated_target_world(),
                'post_root': self.root_states,
                'terminated': terminated,
                'done': terminated | (self.episode_length_buf >= self.max_episode_length),
                'external_world': self.applied_ee_force_world,
                'command_yaw': self.commands[:, 9:12],
                'base_command': self.commands[:, :3],
                'target_command': self.commands[:, 3:6],
                'episode_id': self.diagnostic_episode_id,
                'local_step': self.episode_length_buf,
                'raw_target': self.current_targets['raw_target'],
                'target': self.current_targets['target'],
                'saturation_overlap': self.overlap_counts / self.cfg.control.decimation,
            })
            row.update({key: value / self.cfg.control.decimation for key, value in self.substep_sums.items()})
            self.measurements.append({key: value.detach().cpu().numpy().copy() for key, value in row.items()})
            self.diagnostic_episode_id[done_ids] += 1
        return super()._accumulate_episode_metrics(done_ids, terminated)


def upcoming_target(environment: ExecutionEnvironment) -> torch.Tensor:
    """Compute the next action interval target using pending, not stale force.

    Args:
        environment: Live environment before step; root/yaw and commands are current.

    Returns:
        World XYZ compensated target [N,3], in meters.
    """
    yaw = environment.rpy[:, 2]
    command = environment.commands[:, 9:12]
    command_world = torch.stack((torch.cos(yaw) * command[:, 0] - torch.sin(yaw) * command[:, 1],
        torch.sin(yaw) * command[:, 0] + torch.cos(yaw) * command[:, 1], command[:, 2]), -1)
    return environment._geometric_target_world() + (environment.pending_ee_force_world + command_world) / environment.gripper_force_kps


def main() -> None:
    """Run one isolated mean/sample rollout with same-state counterfactuals.

    Raises:
        OSError: A checkpoint, frozen source, or independent output is unavailable.
        AssertionError: Source/config identity or controller parity does not match.

    Note:
        Saves diagnostics only; no optimizer is constructed or policy updated.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('mean', 'sample'), required=True)
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    source = Path(legged_gym.__file__).resolve().parents[1]
    snapshot = copy.deepcopy(payload['config_snapshot'])
    asset_name = Path(snapshot['environment']['asset']['file']).name
    relative_asset = next(name for name in payload['identity']['asset_manifest'] if name.endswith('/' + asset_name))
    snapshot['environment']['asset']['file'] = str(source / relative_asset)
    assert build_run_identity(source, snapshot, payload['identity']['runtime']) == payload['identity']
    torch.set_num_threads(4)
    set_seed(1001)
    cfg, _ = resolve_v3_configs(argparse.Namespace(reproduction_part='B', run_purpose='debug',
        seed=1001, num_envs=128, max_iterations=1), purpose='debug')
    actual, expected = class_to_dict(cfg), copy.deepcopy(snapshot['environment'])
    for config in (actual, expected):
        config.pop('seed')
        config.pop('run')
        config['env'].pop('num_envs')
    assert actual == expected
    args.output.mkdir(parents=True, exist_ok=False)
    params = gymapi.SimParams()
    gymutil.parse_sim_config(class_to_dict(cfg.sim), params)
    params.use_gpu_pipeline = params.physx.use_gpu = True
    params.physx.num_threads = 4
    properties = {key: payload['environment_training_state'][key][:128] for key in ('mass_com_delta', 'friction')}
    environment = ExecutionEnvironment(cfg, params, gymapi.SIM_PHYSX, 'cuda:0', True, creation_properties=properties)
    try:
        model = V3ActorCritic(1856, 294, 12, 58, 14, **snapshot['training']['policy']).cuda()
        model.load_state_dict(payload['model_state_dict'], strict=True)
        model.eval()
        earlier = torch.load(args.checkpoint.parent / 'model_009000.pt', map_location='cpu', weights_only=False)
        earlier_std = earlier['model_state_dict']['std'].cuda()
        noise_generator = torch.Generator(device='cuda:0').manual_seed(3001)
        environment.set_evaluation_push_seed(2001)
        progress = payload['progress']
        environment.set_training_progress(progress['completed_iterations'], progress['training_policy_steps'])
        environment.measurements = []
        environment.actual_variant = 0 if args.mode == 'mean' else 1
        environment.max_controller_difference = torch.zeros((), device='cuda:0')
        environment.initial_parity_checked = False
        environment.diagnostic_episode_id = torch.zeros(128, dtype=torch.long, device='cuda:0')
        environment.substep_sums = {key: torch.zeros(3, 128, 14, device='cuda:0')
            for key in ('position_p', 'velocity_term', 'raw_torque', 'torque', 'saturated')}
        environment.overlap_counts = torch.zeros(4, 128, 14, device='cuda:0')
        environment.measurement_ready = True
        observations = environment.get_observations()
        push_rows = []
        for step in range(1000):
            environment.set_evaluation_policy_step(progress['training_policy_steps'] + step)
            with torch.inference_mode():
                model.update_distribution(observations['obs'])
                mean = model.action_mean.clone()
                noise = torch.randn(mean.shape, generator=noise_generator, device='cuda:0')
                candidates = torch.stack((mean, mean + noise * model.std, mean + noise * earlier_std))
                environment.candidate_actions = candidates
                environment.pre_step = {key: value.detach().clone() for key, value in {
                    'pre_tcp': environment._tcp_position(), 'pre_geometric': environment._geometric_target_world(),
                    'pre_target': upcoming_target(environment), 'pre_root': environment.root_states,
                    'position': environment.dof_pos[:, environment.action_dof_indices],
                    'velocity': environment.dof_vel[:, environment.action_dof_indices],
                    'execution_delta': environment.execution_scale_delta,
                    'action_mean': mean, 'noise': noise,
                }.items()}
                for accumulator in environment.substep_sums.values():
                    accumulator.zero_()
                environment.overlap_counts.zero_()
                observations, _, _, infos = environment.step(candidates[environment.actual_variant])
                push_rows.append(infos['base_push_applied'])
            if (step + 1) % 250 == 0:
                print(json.dumps({'mode': args.mode, 'step': step + 1}), flush=True)
        difference = float(environment.max_controller_difference)
        assert difference < 1e-4, difference
        arrays = {key: np.stack([row[key] for row in environment.measurements]) for key in environment.measurements[0]}
        arrays['base_push'] = np.asarray(push_rows)
        assert all(np.isfinite(value).all() for value in arrays.values())
        np.savez_compressed(args.output / 'trace.npz', **arrays)
        summary = {
            'iteration': progress['completed_iterations'], 'mode': args.mode,
            'checkpoint_sha256': file_sha256(args.checkpoint), 'policy_updates': 0,
            'num_envs': 128, 'steps': 1000, 'decimation': cfg.control.decimation,
            'policy_dt_s': environment.dt, 'controller_max_abs_difference_nm': difference,
            'all_variants_initial_parity_checked': environment.initial_parity_checked,
            'physical_failures': int(arrays['terminated'].sum()),
            'tcp_l2_mean_m': float(np.linalg.norm(arrays['post_target'] - arrays['post_tcp'], axis=-1).mean()),
            'variant_names': ['mean', 'sample', 'sample_std9000'],
            'saturation_by_variant_action': arrays['saturated'].mean((0, 2)).tolist(),
            'overlap_names': ['neither', 'sample_only', 'mean_only', 'both'],
            'saturation_overlap_by_action': arrays['saturation_overlap'].mean((0, 2)).tolist(),
            'action_joint_names': [environment.dof_names[index] for index in environment.action_dof_indices.tolist()],
            'std': model.std.detach().cpu().tolist(), 'earlier_std': earlier_std.cpu().tolist(),
            'effort_limits_nm': environment.torque_limits[environment.action_dof_indices].cpu().tolist(),
            '说明': '同状态反事实不执行；实际轨迹为原mean/sample策略。均值与采样轨迹在reset分歧后不配对。',
        }
        (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    finally:
        environment.gym.destroy_sim(environment.sim)


if __name__ == '__main__':
    main()

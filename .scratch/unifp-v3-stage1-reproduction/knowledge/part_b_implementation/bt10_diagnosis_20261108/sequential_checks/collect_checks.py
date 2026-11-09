"""独立采样奖励／子步限幅，并观察内存副本的一轮原PPO更新。"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any

import isaacgym  # noqa: F401
from isaacgym import gymapi, gymutil
import numpy as np
import torch

import legged_gym
import legged_gym.envs  # noqa: F401
from legged_gym.b2_gym_learn.ppo_cse_pf.v3_actor_critic import V3ActorCritic
from legged_gym.b2_gym_learn.ppo_cse_pf.v3_ppo import V3PPO
from legged_gym.b2_gym_learn.ppo_cse_pf.v3_run_identity import build_run_identity, file_sha256
from legged_gym.utils.helpers import class_to_dict, set_seed
from legged_gym.utils.v3_config import resolve_v3_configs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probe_policy import DiagnosticEnvironment


class MeasuredEnvironment(DiagnosticEnvironment):
    """Observe every torque substep and selected pre-reset policy boundaries."""

    def _compute_v3_torques(self, actions: torch.Tensor) -> torch.Tensor:
        """Read original clipped torques and accumulate their saturation durations.

        Args:
            actions: Original dimensionless policy actions [N,14].

        Returns:
            Unmodified native-DOF torques [N,14], in N m.

        Note:
            Measurement starts only after simulator initialization; no target,
            force, drive property or termination is modified.
        """
        torques = super()._compute_v3_torques(actions)
        if getattr(self, 'measurement_ready', False):
            saturated = torques[:, self.action_dof_indices].abs() >= self.torque_limits[self.action_dof_indices] * .999
            self.step_saturated_counts += saturated
            self.continuous_saturated_steps = torch.where(saturated, self.continuous_saturated_steps + 1, 0)
            self.longest_saturated_steps = torch.maximum(self.longest_saturated_steps, self.continuous_saturated_steps)
        return torques

    def _accumulate_episode_metrics(self, done_ids: torch.Tensor, terminated: torch.Tensor) -> list[dict[str, Any]]:
        """Save original reward and state evidence before any local reset.

        Args:
            done_ids: Ending environment indices [K].
            terminated: Physical-failure flags [N].

        Returns:
            Original episode metrics including terminal transitions.

        Note:
            Records all substep saturation fractions. Geometry snapshots are every
            ten policy steps; their root is fixed only during later offline IK.
        """
        self.reward_rows.append(self.reward_raw.detach().cpu().numpy())
        self.saturation_rows.append((self.step_saturated_counts.float() / self.cfg.control.decimation).cpu().numpy())
        self.error_rows.append(self._physical_tracking_errors().detach().cpu().numpy())
        if self.inspection_step % 10 == 0:
            self.geometry_rows.append({
                'step': self.inspection_step,
                'root': self.root_states.detach().cpu().numpy(),
                'joint_position': self.dof_pos.detach().cpu().numpy(),
                'geometric_target': self._geometric_target_world().detach().cpu().numpy(),
                'compensated_target': self._compensated_target_world().detach().cpu().numpy(),
                'external_force_world': self.applied_ee_force_world.detach().cpu().numpy(),
                'command_force_yaw': self.commands[:, 9:12].detach().cpu().numpy(),
                'snapshot_saturated_fraction': self.saturation_rows[-1],
            })
        records = super()._accumulate_episode_metrics(done_ids, terminated)
        self.continuous_saturated_steps[done_ids] = 0
        return records


class UpdateObserver:
    """Measure the real optimizer steps on an isolated learner copy."""

    def __init__(self, algorithm: V3PPO, observations: torch.Tensor, labels: torch.Tensor) -> None:
        """Attach observers to the two existing Adam instances.

        Args:
            algorithm: Private learner with checkpoint-restored Adam moments.
            observations: Fixed held-out histories [H,1856], outside rollout times.
            labels: Same-boundary scaled auxiliary targets [H,12].

        Note:
            Replaces instance step callables only in this private process. Each
            wrapper calls the original step exactly once and changes no gradient.
        """
        self.algorithm, self.observations, self.labels = algorithm, observations, labels
        self.original_main_step = algorithm.optimizer.step
        self.original_auxiliary_step = algorithm.adaptation_module_optimizer.step
        self.encoder_parameters = list(algorithm.actor_critic.adaptation_encoder_module.parameters())
        self.events: list[dict[str, Any]] = []
        self.main_gradient: torch.Tensor | None = None
        algorithm.optimizer.step = self.main_step
        algorithm.adaptation_module_optimizer.step = self.auxiliary_step

    def snapshot(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Read fixed-input latent, action mean, predictions and Gaussian scale.

        Returns:
            Detached tensors with widths 64,14,12,14 respectively, on CUDA.
        """
        model = self.algorithm.actor_critic
        with torch.no_grad():
            latent = model.adaptation_encoder_module(self.observations)
            mean = model.actor_body(torch.cat((self.observations[:, -58:], latent), -1))
            prediction = model.adaptation_decoder_module(latent)
        return latent, mean, prediction, model.std.detach().clone()

    def record(self, kind: str, before: tuple[torch.Tensor, ...], after: tuple[torch.Tensor, ...]) -> None:
        """Save direct policy/estimator changes caused by one optimizer step.

        Args:
            kind: Main PPO or auxiliary update identifier.
            before: Snapshot immediately before the original Adam step.
            after: Snapshot immediately after the same step.
        """
        old_z, old_mean, old_prediction, old_std = before
        new_z, new_mean, new_prediction, new_std = after
        kl = (torch.log(new_std / old_std) + (old_std.square() + (old_mean-new_mean).square()) / (2*new_std.square()) - .5).sum(-1)
        row = {'kind': kind, 'mean_action_rms_change': float((new_mean-old_mean).square().mean().sqrt()),
               'latent_rms_change': float((new_z-old_z).square().mean().sqrt()),
               'gaussian_kl_mean': float(kl.mean()),
               'force_mae_before_n': ((old_prediction[:, 6:9]-self.labels[:, 6:9]).abs().mean(0)*100).tolist(),
               'force_mae_after_n': ((new_prediction[:, 6:9]-self.labels[:, 6:9]).abs().mean(0)*100).tolist()}
        if kind == 'auxiliary':
            auxiliary_gradient = torch.cat([parameter.grad.flatten() for parameter in self.encoder_parameters])
            row['encoder_gradient_cosine_with_preceding_main'] = float(torch.nn.functional.cosine_similarity(self.main_gradient, auxiliary_gradient, dim=0))
        self.events.append(row)

    def main_step(self) -> None:
        """Observe and execute exactly one original main Adam step.

        Note:
            Stores the already-clipped encoder gradient; clipping preserves its
            direction. Later cosine is between adjacent, slightly different points.
        """
        self.main_gradient = torch.cat([parameter.grad.detach().flatten().clone() for parameter in self.encoder_parameters])
        before = self.snapshot()
        self.original_main_step()
        self.record('main', before, self.snapshot())

    def auxiliary_step(self) -> None:
        """Observe and execute exactly one original auxiliary Adam step."""
        before = self.snapshot()
        self.original_auxiliary_step()
        self.record('auxiliary', before, self.snapshot())


def main() -> None:
    """Collect bounded diagnostics, destroy simulation, then update a private copy.

    Raises:
        OSError: Checkpoint, source or independent evidence cannot be accessed.
        AssertionError: Source identity, effective config or reward accounting differs.

    Note:
        Formal training is never touched. No modified model or optimizer is saved.
        The rollout update is diagnostic, not another formal training iteration.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    source = Path(legged_gym.__file__).resolve().parents[1]
    snapshot = copy.deepcopy(payload['config_snapshot'])
    asset_name = Path(snapshot['environment']['asset']['file']).name
    relative = next(name for name in payload['identity']['asset_manifest'] if name.endswith('/'+asset_name))
    snapshot['environment']['asset']['file'] = str(source / relative)
    assert build_run_identity(source, snapshot, payload['identity']['runtime']) == payload['identity']
    torch.set_num_threads(4)
    set_seed(1001)
    cfg, _ = resolve_v3_configs(argparse.Namespace(reproduction_part='B', run_purpose='debug', seed=1001, num_envs=128, max_iterations=1), purpose='debug')
    actual, expected = class_to_dict(cfg), copy.deepcopy(snapshot['environment'])
    for value in (actual, expected):
        value.pop('seed'); value.pop('run'); value['env'].pop('num_envs')
    assert actual == expected
    params = gymapi.SimParams()
    gymutil.parse_sim_config(class_to_dict(cfg.sim), params)
    params.use_gpu_pipeline = params.physx.use_gpu = True
    params.physx.num_threads = 4
    properties = {key: payload['environment_training_state'][key][:128] for key in ('mass_com_delta', 'friction')}
    environment = MeasuredEnvironment(cfg, params, gymapi.SIM_PHYSX, 'cuda:0', True, creation_properties=properties)
    model = V3ActorCritic(1856, 294, 12, 58, 14, **snapshot['training']['policy']).cuda()
    model.load_state_dict(payload['model_state_dict'], strict=True)
    algorithm = V3PPO(model, device='cuda:0', **snapshot['training']['algorithm'])
    assert payload['main_parameter_names'] == [algorithm.main_parameter_names]
    assert payload['auxiliary_parameter_names'] == [algorithm.auxiliary_parameter_names]
    algorithm.optimizer.load_state_dict(payload['main_optimizer_state_dict'])
    algorithm.adaptation_module_optimizer.load_state_dict(payload['auxiliary_optimizer_state_dict'])
    algorithm.learning_rate = payload['learning_rate_state']['main']
    algorithm.init_storage(128, 24, [1856], [294], [12], [14])
    held_observations, held_labels = [], []
    try:
        environment.failure_events, environment.ended_records = [], []
        environment.body_names = environment.gym.get_actor_rigid_body_names(environment.envs[0], environment.actor_handles[0])
        environment.set_evaluation_push_seed(2001)
        progress = payload['progress']
        environment.set_training_progress(progress['completed_iterations'], progress['training_policy_steps'])
        environment.reward_rows, environment.saturation_rows, environment.error_rows, environment.geometry_rows = [], [], [], []
        environment.step_saturated_counts = torch.zeros(128, 14, dtype=torch.int64, device='cuda:0')
        environment.continuous_saturated_steps = torch.zeros_like(environment.step_saturated_counts)
        environment.longest_saturated_steps = torch.zeros_like(environment.step_saturated_counts)
        environment.measurement_ready = True
        observations = environment.get_observations()
        for step in range(1000):
            environment.inspection_step = step
            environment.set_evaluation_policy_step(progress['training_policy_steps'] + step)
            environment.step_saturated_counts.zero_()
            in_rollout = 400 <= step < 424
            if step % 25 == 0 and not in_rollout:
                held_observations.append(observations['obs'][:8].clone())
                held_labels.append(observations['obs_pred'][:8].clone())
            with torch.no_grad():
                actions = algorithm.act(observations['obs'], observations['privileged_obs'], observations['obs_pred']) if in_rollout else model.act(observations['obs'])
                observations, rewards, dones, infos = environment.step(actions)
                weighted_reward = (environment.reward_raw * environment.reward_raw.new_tensor(
                    [environment.reward_scales[name] for name in environment.reward_names])).sum(-1)
                assert torch.allclose(weighted_reward, rewards, atol=1e-6)
                if in_rollout:
                    algorithm.process_env_step(rewards, dones, infos)
                    if step == 423:
                        algorithm.compute_returns(observations['privileged_obs'])
            if (step+1) % 250 == 0:
                print(json.dumps({'step':step+1,'failures':len(environment.failure_events)}), flush=True)
        raw = np.stack(environment.reward_rows)
        weights = np.array([environment.configured_reward_weights[name] for name in environment.reward_names])
        geometry_rows = environment.geometry_rows
        arrays = {key: np.stack([row[key] for row in geometry_rows]) for key in geometry_rows[0]}
        np.savez_compressed(args.output/'measurements.npz', raw_rewards=raw,
            saturated_fraction=np.stack(environment.saturation_rows), errors=np.stack(environment.error_rows), **arrays)
        summary = {'iteration':progress['completed_iterations'], 'checkpoint_sha256':file_sha256(args.checkpoint),
            'policy_dt_s':environment.dt,'physics_dt_s':cfg.sim.dt,'num_envs':128,'steps':1000,
            'reward_names':list(environment.reward_names),'reward_weights':weights.tolist(),
            'mean_reward_contributions_per_second':(raw.mean((0,1))*weights).tolist(),
            'saturation_fraction_by_action':np.stack(environment.saturation_rows).mean((0,1)).tolist(),
            'longest_saturation_s_by_action':(environment.longest_saturated_steps.max(0).values.float()*cfg.sim.dt).cpu().tolist(),
            'dof_names':environment.dof_names,'action_dof_indices':environment.action_dof_indices.cpu().tolist(),
            'torque_limits_nm':environment.torque_limits.cpu().tolist(),
            'joint_limits_rad':environment.dof_pos_limits.cpu().tolist(),
            'tcp_l2_mean_m':float(np.stack(environment.error_rows)[:,:,5].mean()),
            'physical_failures':len(environment.failure_events),'inference_policy_updates':0,
            'reward_sum_verified_every_step':True,
            '说明':'独立新episode、原采样动作；与上一轮独立噪声Generator探针不构成同轨迹配对。'}
        (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
        torch.save({'observations':torch.cat(held_observations).cpu(),'labels':torch.cat(held_labels).cpu()}, args.output/'held_inputs.pt')
        torch.save({name:getattr(algorithm.storage,name).cpu() for name in ('observations','privileged_observations','observation_preds','actions','mu','sigma','values','advantages','returns','actions_log_prob','rewards','dones')}, args.output/'rollout.pt')
    finally:
        environment.gym.destroy_sim(environment.sim)
    observer = UpdateObserver(algorithm, torch.cat(held_observations), torch.cat(held_labels))
    update_result = algorithm.update()
    assert len(observer.events) == 40
    (args.output/'update_observation.json').write_text(json.dumps({'events':observer.events,
        'losses':update_result,'diagnostic_iterations':1,'formal_policy_updates':0,
        '说明':'只在内存副本执行一轮原PPO，恢复双Adam；128环境的小rollout不是正式4096更新的效应量。'},ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'completed':True,'iteration':summary['iteration']},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()

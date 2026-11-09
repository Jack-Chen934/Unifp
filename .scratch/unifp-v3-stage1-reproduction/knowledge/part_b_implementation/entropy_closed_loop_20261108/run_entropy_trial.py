"""按已确认方案运行独立B熵消融，复用冻结控制器与PPO。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterator

import isaacgym  # noqa: F401
from isaacgym import gymapi, gymutil
import numpy as np
import torch
import legged_gym
import legged_gym.envs  # noqa: F401
from legged_gym.envs.v3.legged_robot_v3 import V3PosForce
from legged_gym.b2_gym_learn.ppo_cse_pf.v3_on_policy_runner import V3OnPolicyRunner
from legged_gym.b2_gym_learn.ppo_cse_pf.v3_run_identity import build_run_identity, file_sha256
from legged_gym.utils.helpers import class_to_dict, set_seed
from legged_gym.utils.v3_config import resolve_v3_configs

from experiment_contract import experiment_snapshot, validate_runtime_move

ORIGIN_SHA256 = 'd0c64674c852300776662b389e14047e998f8444486f74db5fabea684f089b1f'


def assert_equal_state(actual: Any, expected: Any) -> None:
    """Check all restored tensors and optimizer scalars against their checkpoint.

    Args:
        actual: Live model or optimizer state; tensors may reside on CUDA.
        expected: Matching CPU checkpoint state, including Adam steps and slots.

    Raises:
        AssertionError: A nested value or parameter association differs.
    """
    if isinstance(actual, torch.Tensor):
        assert torch.equal(actual.detach().cpu(), expected.detach().cpu())
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_equal_state(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_equal_state(left, right)
    else:
        assert actual == expected


def tensor_digest(tensor: torch.Tensor) -> str:
    """Fingerprint a startup tensor without changing its state or RNG.

    Args:
        tensor: Contiguous-convertible observation, property or RNG tensor.

    Returns:
        SHA256 of the CPU value bytes, retaining dtype and shape in the caller.
    """
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class MeasuredEnvironment(V3PosForce):
    """Add read-only device counters after the unchanged original controller."""

    def start_measurement(self) -> None:
        """Allocate empty per-actor counters after restoration and reset.

        Note:
            Counters are never part of training state or learning observations.
        """
        self.saturated_counts = torch.zeros(self.num_envs, 14, device=self.device, dtype=torch.long)
        self.position_clip_counts = torch.zeros(self.num_envs, 10, device=self.device, dtype=torch.long)
        self.wheel_clip_counts = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.long)
        self.measured_substeps = 0
        self.measurement_ready = True

    def _compute_v3_torques(self, actions: torch.Tensor) -> torch.Tensor:
        """Execute original control once and observe all eight physical substeps.

        Args:
            actions: Executed dimensionless policy-order actions [4096,14].

        Returns:
            Unchanged native-DOF torques [4096,14], in N m.

        Note:
            Target clipping is measured at each policy step's first substep;
            torque saturation covers every physical substep, before local reset.
        """
        torques = super()._compute_v3_torques(actions)
        if getattr(self, 'measurement_ready', False):
            self.saturated_counts += torques[:, self.action_dof_indices].abs() >= .999 * self.torque_limits[self.action_dof_indices]
            if self.measured_substeps % self.cfg.control.decimation == 0:
                scaled_actions = actions * (1 + self.execution_scale_delta)
                position_actions = torch.cat((scaled_actions[:, :4], scaled_actions[:, 8:]), dim=-1)
                requested_position = self.default_dof_pos[:, self.position_dof_indices] + self.cfg.control.action_scale * position_actions
                limits = self.dof_pos_limits[self.position_dof_indices]
                self.position_clip_counts += (requested_position < limits[:, 0]) | (requested_position > limits[:, 1])
                requested_wheel_speed = self.cfg.control.wheel_action_scale * scaled_actions[:, 4:8]
                self.wheel_clip_counts += requested_wheel_speed.abs() > self.cfg.control.wheel_velocity_limit
            self.measured_substeps += 1
        return torques


class MeasuredRunner(V3OnPolicyRunner):
    """Observe unchanged sampling/minibatches/Adam and log complete boundaries."""

    def attach_measurements(self) -> None:
        """Wrap only this learner's original callables after strict state restoration.

        Note:
            Calls original methods exactly once, consumes no extra random values,
            and never changes gradients. New counters begin at empty boundaries.
        """
        self.original_act = self.alg.act
        self.original_batches = self.alg.storage.mini_batch_generator
        self.original_step = self.alg.optimizer.step
        self.action_square_sum = torch.zeros(14, device=self.device)
        self.mean_square_sum = torch.zeros_like(self.action_square_sum)
        self.raw_clip_count = torch.zeros_like(self.action_square_sum)
        self.kl_sum = torch.zeros((), device=self.device)
        self.sample_steps = 0
        self.main_steps = 0
        self.alg.act = self.measured_act
        self.alg.storage.mini_batch_generator = self.measured_batches
        self.alg.optimizer.step = self.measured_main_step
        self.env.start_measurement()

    def measured_act(self, obs: torch.Tensor, critic_obs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Pass through raw samples while accumulating same-boundary action statistics.

        Args:
            obs: Actor histories [4096,1856].
            critic_obs: Privileged histories [4096,294].
            labels: Environment supervision [4096,12].

        Returns:
            The exact original raw action samples [4096,14].
        """
        actions = self.original_act(obs, critic_obs, labels)
        self.action_square_sum += actions.square().sum(0)
        self.mean_square_sum += self.alg.actor_critic.action_mean.detach().square().sum(0)
        self.raw_clip_count += (actions.abs() > self.env.cfg.normalization.clip_actions).sum(0)
        self.sample_steps += 1
        return actions

    def measured_batches(self, num_mini_batches: int, num_epochs: int = 5) -> Iterator[tuple[Any, ...]]:
        """Preserve permutation, RNG consumption and the original eleven-field batches.

        Args:
            num_mini_batches: Original mini-batch count, unchanged at four.
            num_epochs: Original passes over one rollout, unchanged at five.

        Yields:
            Each original batch without cloning or modifying any tensor.
        """
        for batch in self.original_batches(num_mini_batches, num_epochs):
            self.old_mu, self.old_sigma = batch[8], batch[9]
            yield batch

    def measured_main_step(self) -> None:
        """Observe the same pre-step KL used by the original adaptive scheduler.

        Note:
            Reads detached tensors after backward/clipping, then executes original
            Adam exactly once without modifying gradients, lr, moments or samples.
        """
        with torch.no_grad():
            model = self.alg.actor_critic
            sigma, mean = model.action_std, model.action_mean
            self.kl_sum += (torch.log(sigma / self.old_sigma + 1e-5)
                + (self.old_sigma.square() + (self.old_mu - mean).square()) / (2 * sigma.square()) - .5).sum(-1).mean()
        self.main_steps += 1
        self.original_step()

    def _append_json(self, filename: str, record: dict[str, Any]) -> None:
        """Augment only experiment iteration records and reset completed counters.

        Args:
            filename: Original runner log basename.
            record: Original finite learning/checkpoint metrics.

        Raises:
            AssertionError: A supposedly complete rollout omits policy or substeps.
        """
        if filename != 'iterations.jsonl':
            super()._append_json(filename, record)
            return
        assert self.sample_steps == 24 and self.main_steps == 20 and self.env.measured_substeps == 192
        samples = self.env.num_envs * 24
        augmented = {**record, 'diagnostic/std_by_action': self.alg.actor_critic.std.detach().cpu().tolist(),
            'diagnostic/action_rms': (self.action_square_sum / samples).sqrt().cpu().tolist(),
            'diagnostic/mean_rms': (self.mean_square_sum / samples).sqrt().cpu().tolist(),
            'diagnostic/raw_action_clip_fraction': (self.raw_clip_count / samples).cpu().tolist(),
            'diagnostic/torque_saturation_fraction': (self.env.saturated_counts.float().sum(0) / (self.env.num_envs * 192)).cpu().tolist(),
            'diagnostic/position_target_clip_fraction': (self.env.position_clip_counts.float().sum(0) / samples).cpu().tolist(),
            'diagnostic/wheel_target_clip_fraction': (self.env.wheel_clip_counts.float().sum(0) / samples).cpu().tolist(),
            'diagnostic/kl_mean': float(self.kl_sum / self.main_steps), 'diagnostic/physical_substeps': 192}
        super()._append_json(filename, augmented)
        for counter in (self.action_square_sum, self.mean_square_sum, self.raw_clip_count, self.kl_sum,
                        self.env.saturated_counts, self.env.position_clip_counts, self.env.wheel_clip_counts):
            counter.zero_()
        self.sample_steps = self.main_steps = self.env.measured_substeps = 0


def actual_runtime(parent: dict[str, Any]) -> dict[str, Any]:
    """Independently measure the existing runtime fields on physical GPU3.

    Args:
        parent: Saved runtime used only to require the exact same field set.

    Returns:
        Actual interpreter, library, simulator, driver and logical-device identity.

    Raises:
        AssertionError: The runtime field schema unexpectedly changes.
    """
    runtime = {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
        'cuda_runtime': torch.version.cuda, 'isaac_gym': '1.0.preview4', 'device': 'cuda:0',
        'sim_device': 'cuda:0', 'physics_engine': str(gymapi.SIM_PHYSX), 'use_gpu_pipeline': True,
        'physx_use_gpu': True, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'gpu_name': torch.cuda.get_device_name(0),
        'gpu_driver': subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).splitlines()[0],
        'torch_num_threads': torch.get_num_threads(), 'platform': platform.platform(),
        'python_executable': sys.executable, 'gym_import_path': isaacgym.__file__}
    assert runtime.keys() == parent.keys()
    return runtime


def main() -> None:
    """Restore strict parent state, publish explicit intervention identity and train.

    Raises:
        AssertionError: Parent SHA, source, config, restored learning state or budget differs.
        ValueError: An unapproved runtime difference or entropy setting is requested.
        OSError: Source, independent output or complete checkpoints cannot be accessed.

    Note:
        Creates one independent run and destroys its simulator on exit. It does not
        mutate original files or the formal training process. Parent restoration
        is complete-state; unfinished episodes and PhysX caches are not restored.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--origin', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, required=True)
    parser.add_argument('--entropy-coef', type=float, choices=(0., .01), required=True)
    args = parser.parse_args()
    assert 1 <= args.iterations <= 200 and os.environ['CUDA_VISIBLE_DEVICES'] == '3'
    assert file_sha256(args.origin) == ORIGIN_SHA256
    source = Path(legged_gym.__file__).resolve().parents[1]
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    is_origin = args.checkpoint.resolve() == args.origin.resolve()
    if is_origin:
        assert payload['progress']['completed_iterations'] == 9000
    else:
        experiment = payload['lineage']['entropy_experiment']
        assert experiment['origin_sha256'] == ORIGIN_SHA256
        assert experiment['entropy_coef'] == args.entropy_coef
        assert experiment['launcher_sha256'] == file_sha256(Path(__file__))
        assert experiment['contract_sha256'] == file_sha256(Path(__file__).with_name('experiment_contract.py'))
    parent_snapshot = copy.deepcopy(payload['config_snapshot'])
    asset_relative = next(name for name in payload['identity']['asset_manifest']
        if name.endswith('/' + Path(parent_snapshot['environment']['asset']['file']).name))
    parent_snapshot['environment']['asset']['file'] = str(source / asset_relative)
    assert build_run_identity(source, parent_snapshot, payload['identity']['runtime']) == payload['identity']
    assert parent_snapshot['environment']['env']['num_envs'] == 4096
    torch.set_num_threads(payload['identity']['runtime']['torch_num_threads'])
    torch.cuda.set_device(0)
    runtime = actual_runtime(payload['identity']['runtime'])
    runtime_changes = validate_runtime_move(payload['identity']['runtime'], runtime)
    target = experiment_snapshot(parent_snapshot, source, args.entropy_coef)
    target_identity = build_run_identity(source, target, runtime)
    if not is_origin:
        assert target_identity == payload['identity']
    cfg, _ = resolve_v3_configs(argparse.Namespace(reproduction_part='B',
        run_purpose=parent_snapshot['environment']['run']['run_purpose'], seed=1, num_envs=4096, max_iterations=1),
        purpose=parent_snapshot['environment']['run']['run_purpose'])
    cfg.asset.file = parent_snapshot['environment']['asset']['file']
    assert class_to_dict(cfg) == parent_snapshot['environment']
    params = gymapi.SimParams()
    gymutil.parse_sim_config(class_to_dict(cfg.sim), params)
    params.use_gpu_pipeline = params.physx.use_gpu = True
    params.physx.num_threads = 4
    set_seed(1)
    properties = {key: payload['environment_training_state'][key] for key in ('mass_com_delta', 'friction')}
    environment = MeasuredEnvironment(cfg, params, gymapi.SIM_PHYSX, 'cuda:0', True, creation_properties=properties)
    try:
        runner = MeasuredRunner(environment, parent_snapshot['training'], args.output,
                                payload['identity'], parent_snapshot, 'cuda:0')
        runner.load(args.checkpoint)
        assert_equal_state(runner.alg.actor_critic.state_dict(), payload['model_state_dict'])
        assert_equal_state(runner.alg.optimizer.state_dict(), payload['main_optimizer_state_dict'])
        assert_equal_state(runner.alg.adaptation_module_optimizer.state_dict(), payload['auxiliary_optimizer_state_dict'])
        assert runner.alg.learning_rate == payload['learning_rate_state']['main']
        runner.identity, runner.config_snapshot, runner.cfg = target_identity, target, target['training']['runner']
        runner.env.cfg.run.run_purpose = 'development'
        runner.alg.entropy_coef = args.entropy_coef
        runner.lineage.update(restore_mode='diagnostic_intervention' if is_origin else 'same_stage',
            parent_identity=payload['identity'], entropy_experiment={
                'origin_path': str(args.origin), 'origin_sha256': ORIGIN_SHA256, 'entropy_coef': args.entropy_coef,
                'launcher_sha256': file_sha256(Path(__file__)),
                'contract_sha256': file_sha256(Path(__file__).with_name('experiment_contract.py')),
                'runtime_changes': runtime_changes, 'learning_change': {'entropy_coef': [.01, args.entropy_coef]}})
        startup = {'completed_iterations': runner.completed_iterations, 'restored_learning_state_equal': True,
            'identity': target_identity, 'lineage': runner.lineage, 'std': runner.alg.actor_critic.std.detach().cpu().tolist(),
            'main_lr': runner.alg.learning_rate, 'initial_state_digests': {
                **{name: tensor_digest(tensor) for name, tensor in environment.get_observations().items()},
                'root_states': tensor_digest(environment.root_states), 'dof_pos': tensor_digest(environment.dof_pos),
                'execution_scale_delta': tensor_digest(environment.execution_scale_delta),
                'mass_com_delta': tensor_digest(environment.mass_com_delta), 'friction': tensor_digest(environment.friction),
                'cpu_rng': tensor_digest(torch.get_rng_state()), 'cuda_rng': tensor_digest(torch.cuda.get_rng_state(0))}}
        (args.output / 'startup.json').write_text(json.dumps(startup, ensure_ascii=False, indent=2) + '\n')
        runner.attach_measurements()
        runner.learn(args.iterations)
        (args.output / 'completion.json').write_text(json.dumps({'completed_iterations': runner.completed_iterations,
            'additional_iterations': args.iterations, 'last_saved_iteration': runner.last_saved_iteration,
            'entropy_coef': args.entropy_coef, 'origin_sha256': ORIGIN_SHA256}, ensure_ascii=False, indent=2) + '\n')
    finally:
        environment.gym.destroy_sim(environment.sim)


if __name__ == '__main__':
    main()

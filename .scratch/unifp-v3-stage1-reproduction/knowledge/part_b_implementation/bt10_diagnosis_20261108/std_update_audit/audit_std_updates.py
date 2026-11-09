"""离线测量原PPO的std梯度、全局裁剪与Adam历史贡献。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch


EXPECTED_HASHES = {
    9000: 'd0c64674c852300776662b389e14047e998f8444486f74db5fabea684f089b1f',
    10000: '7e7ec14c8d772d9343ed3a9f9918e477657383980e9d747cd209f79006f6fd4d',
    12000: '273d4269bc37e95f879d21b178bc0380e20d2539e2d9cfd7488296f049c3146c',
}


def sha256(path: Path) -> str:
    """Hash one immutable input or diagnostic script.

    Args:
        path: Existing file to read as bytes.

    Returns:
        Hexadecimal SHA256 digest.

    Raises:
        OSError: The file cannot be read.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def restore_copy(payload: dict[str, Any], rollout: dict[str, torch.Tensor]) -> Any:
    """Restore an isolated CPU learner and an already collected real rollout.

    Args:
        payload: Original complete checkpoint including both Adam states and lr.
        rollout: Stored tensors [24,128,width], including normalized advantages.

    Returns:
        Learner with original hyperparameters, moments, and ready-to-update storage.

    Raises:
        AssertionError: Parameter ordering, actual lr, or auxiliary ownership differs.

    Note:
        Allocates private models and buffers; does not construct a simulation or
        claim restoration of the formal environment/RNG trajectory.
    """
    from legged_gym.b2_gym_learn.ppo_cse_pf.v3_actor_critic import V3ActorCritic
    from legged_gym.b2_gym_learn.ppo_cse_pf.v3_ppo import V3PPO

    training = payload['config_snapshot']['training']
    model = V3ActorCritic(1856, 294, 12, 58, 14, **training['policy'])
    model.load_state_dict(payload['model_state_dict'], strict=True)
    algorithm = V3PPO(model, device='cpu', **training['algorithm'])
    assert payload['main_parameter_names'] == [algorithm.main_parameter_names]
    assert payload['auxiliary_parameter_names'] == [algorithm.auxiliary_parameter_names]
    assert 'std' not in algorithm.auxiliary_parameter_names
    # Deepcopy is essential: load_state_dict may otherwise alias CPU Adam slots.
    algorithm.optimizer.load_state_dict(copy.deepcopy(payload['main_optimizer_state_dict']))
    algorithm.adaptation_module_optimizer.load_state_dict(copy.deepcopy(payload['auxiliary_optimizer_state_dict']))
    algorithm.learning_rate = payload['learning_rate_state']['main']
    assert algorithm.optimizer.param_groups[0]['lr'] == algorithm.learning_rate
    algorithm.init_storage(128, 24, [1856], [294], [12], [14])
    for name, tensor in rollout.items():
        getattr(algorithm.storage, name).copy_(tensor)
    algorithm.storage.step = 24
    algorithm.storage.returns_ready = True
    return algorithm


def same_point_gradients(algorithm: Any, rollout: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Separate exact std gradients at the checkpoint before any update.

    Args:
        algorithm: CPU learner with checkpoint parameters, not yet updated.
        rollout: Real old-policy samples and advantages [24,128,width].

    Returns:
        Per-dimension surrogate/entropy gradients, score check and old logp error.

    Raises:
        AssertionError: Old likelihood, analytic entropy/score, or value isolation fails.
    """
    model = algorithm.actor_critic
    model.update_distribution(rollout['observations'].flatten(0, 1))
    actions = rollout['actions'].flatten(0, 1)
    advantages = rollout['advantages'].flatten()
    logp = model.get_actions_log_prob(actions)
    old_logp = rollout['actions_log_prob'].flatten()
    difference = float((logp - old_logp).abs().max())
    assert difference < 1e-3
    ratio = (logp - old_logp).exp()
    surrogate = torch.maximum(-advantages * ratio, -advantages * ratio.clamp(
        1 - algorithm.clip_param, 1 + algorithm.clip_param)).mean()
    surrogate_gradient = torch.autograd.grad(surrogate, model.std, retain_graph=True)[0]
    entropy_gradient = torch.autograd.grad(-algorithm.entropy_coef * model.entropy.mean(), model.std)[0]
    analytic_entropy = -algorithm.entropy_coef / model.std.detach()
    # Near the old policy all ratios lie strictly within PPO's clipping window.
    score = (actions - model.action_mean.detach()).square() / model.std.detach().pow(3) - 1 / model.std.detach()
    analytic_surrogate = (-advantages[:, None] * ratio.detach()[:, None] * score).mean(0)
    value = model.evaluate(rollout['privileged_observations'].flatten(0, 1))
    value_gradient = torch.autograd.grad(value.square().mean(), model.std, allow_unused=True)[0]
    assert value_gradient is None
    assert torch.allclose(entropy_gradient, analytic_entropy, atol=1e-6, rtol=1e-5)
    assert torch.allclose(surrogate_gradient, analytic_surrogate, atol=1e-6, rtol=1e-4)
    return {'max_old_logp_error': difference, 'surrogate_gradient': surrogate_gradient.tolist(),
            'entropy_gradient': entropy_gradient.tolist(), 'std': model.std.detach().tolist(),
            'value_has_std_gradient': False,
            'analytic_entropy_max_error': float((entropy_gradient - analytic_entropy).abs().max()),
            'analytic_score_max_error': float((surrogate_gradient - analytic_surrogate).abs().max())}


class StdObserver:
    """Record unchanged std updates with a conditional Adam numerator decomposition."""

    def __init__(self, algorithm: Any) -> None:
        """Attach a read-only gradient hook and original-step wrapper.

        Args:
            algorithm: Private learner whose main optimizer is observed.

        Note:
            Hooks return no replacement gradient. Only this in-memory instance
            receives a wrapper, which executes its original step exactly once.
        """
        self.algorithm = algorithm
        self.original_step = algorithm.optimizer.step
        self.raw_gradient = torch.zeros(14)
        self.events: list[dict[str, Any]] = []
        initial_moment = algorithm.optimizer.state[algorithm.actor_critic.std]['exp_avg'].double().clone()
        self.moment_sources = {'restored_momentum': initial_moment,
                               'rollout_surrogate': torch.zeros_like(initial_moment),
                               'rollout_entropy': torch.zeros_like(initial_moment)}
        self.hook = algorithm.actor_critic.std.register_hook(self.capture_gradient)
        algorithm.optimizer.step = self.step

    def capture_gradient(self, gradient: torch.Tensor) -> None:
        """Observe the total main-loss std gradient before global clipping.

        Args:
            gradient: Unclipped gradient [14] in dimensionless action units.
        """
        self.raw_gradient = gradient.detach().clone()

    def step(self) -> None:
        """Execute original Adam once and verify the measured std increment.

        Raises:
            AssertionError: Clipping is not common-scalar or Adam reconstruction fails.

        Note:
            Attribution holds the realized second-moment denominator fixed. Old
            first moment contains mixed past causes and is not an entropy-only
            history. This is not a counterfactual entropy-free training update.
        """
        model = self.algorithm.actor_critic
        before = model.std.detach().double().clone()
        raw_gradient = self.raw_gradient.double()
        clipped_gradient = model.std.grad.detach().double().clone()
        clip_factor = float(torch.dot(clipped_gradient, raw_gradient) / raw_gradient.square().sum())
        clip_error = float((clipped_gradient - clip_factor * raw_gradient).abs().max())
        assert clip_error < 1e-7 and 0 < clip_factor <= 1.00001
        entropy_gradient = -self.algorithm.entropy_coef / before
        surrogate_gradient = raw_gradient - entropy_gradient
        group = self.algorithm.optimizer.param_groups[0]
        beta1, beta2 = group['betas']
        state = self.algorithm.optimizer.state[model.std]
        old_moment = state['exp_avg'].double().clone()
        old_variance = state['exp_avg_sq'].double().clone()
        step_number = int(state['step']) + 1
        next_variance = beta2 * old_variance + (1 - beta2) * clipped_gradient.square()
        denominator = (next_variance / (1 - beta2 ** step_number)).sqrt() + group['eps']
        multiplier = -group['lr'] / (1 - beta1 ** step_number) / denominator
        contributions = {
            'history': multiplier * beta1 * old_moment,
            'surrogate': multiplier * (1 - beta1) * clip_factor * surrogate_gradient,
            'entropy': multiplier * (1 - beta1) * clip_factor * entropy_gradient,
        }
        # Carry each source through all 20 main steps, avoiding double-counting
        # this rollout's early gradients as unexplained checkpoint history.
        self.moment_sources['restored_momentum'] *= beta1
        self.moment_sources['rollout_surrogate'] = beta1 * self.moment_sources['rollout_surrogate'] + (1 - beta1) * clip_factor * surrogate_gradient
        self.moment_sources['rollout_entropy'] = beta1 * self.moment_sources['rollout_entropy'] + (1 - beta1) * clip_factor * entropy_gradient
        unrolled_contributions = {name: multiplier * moment for name, moment in self.moment_sources.items()}
        self.original_step()
        actual_delta = model.std.detach().double() - before
        reconstruction_error = float((actual_delta - sum(contributions.values())).abs().max())
        assert reconstruction_error < 1e-7
        unrolled_error = float((actual_delta - sum(unrolled_contributions.values())).abs().max())
        assert unrolled_error < 1e-7
        self.events.append({'step': len(self.events) + 1, 'lr': group['lr'],
            'adam_step': step_number, 'clip_factor': clip_factor, 'clip_max_error': clip_error,
            'std_before': before.tolist(), 'raw_gradient': raw_gradient.tolist(),
            'surrogate_gradient': surrogate_gradient.tolist(), 'entropy_gradient': entropy_gradient.tolist(),
            'actual_delta': actual_delta.tolist(),
            'conditional_delta': {key: value.tolist() for key, value in contributions.items()},
            'unrolled_conditional_delta': {key: value.tolist() for key, value in unrolled_contributions.items()},
            'reconstruction_max_error': reconstruction_error, 'unrolled_reconstruction_max_error': unrolled_error})


def assert_equal_state(first: Any, second: Any) -> None:
    """Verify bitwise observer neutrality for nested model/optimizer states.

    Args:
        first: State from a learner using the observation wrapper.
        second: State from the original unobserved update using the same RNG seed.

    Raises:
        AssertionError: Any nested tensor, key, scalar or sequence differs.
    """
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_equal_state(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for left, right in zip(first, second):
            assert_equal_state(left, right)
    else:
        assert first == second


def main() -> None:
    """Audit original updates, paired zero-entropy copies and observer neutrality.

    Raises:
        OSError: A frozen source, checkpoint, rollout or result cannot be accessed.
        AssertionError: Input identity, restoration, math or observer checks fail.

    Note:
        Saves only diagnostic JSON. Executes 9 observed original updates, 9 paired
        zero-entropy copies and 1 neutrality replay. Never saves models or touches
        the formal process; fixed old-policy data are not a new closed-loop run.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--checkpoints', type=Path, required=True)
    parser.add_argument('--rollouts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source))
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for iteration in EXPECTED_HASHES:
        checkpoint_path = args.checkpoints / f'model_{iteration:06d}.pt'
        rollout_path = args.rollouts / f'check_{iteration}' / 'rollout.pt'
        checkpoint_hash, rollout_hash = sha256(checkpoint_path), sha256(rollout_path)
        assert checkpoint_hash == EXPECTED_HASHES[iteration]
        payload = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        rollout = torch.load(rollout_path, map_location='cpu', weights_only=False)
        point = same_point_gradients(restore_copy(payload, rollout), rollout)
        trials = []
        for shuffle_seed in (3101, 3102, 3103):
            algorithm = restore_copy(payload, rollout)
            observer = StdObserver(algorithm)
            torch.manual_seed(shuffle_seed)
            losses = algorithm.update()
            assert len(observer.events) == 20
            trials.append({'shuffle_seed': shuffle_seed, 'events': observer.events,
                           'final_std': algorithm.actor_critic.std.detach().tolist(), 'losses': losses})
            observer.hook.remove()
            if iteration == 9000 and shuffle_seed == 3101:
                baseline = restore_copy(payload, rollout)
                torch.manual_seed(shuffle_seed)
                baseline_losses = baseline.update()
                assert_equal_state(algorithm.actor_critic.state_dict(), baseline.actor_critic.state_dict())
                assert_equal_state(algorithm.optimizer.state_dict(), baseline.optimizer.state_dict())
                assert_equal_state(algorithm.adaptation_module_optimizer.state_dict(), baseline.adaptation_module_optimizer.state_dict())
                assert_equal_state(losses, baseline_losses)
            # A paired one-rollout intervention recomputes the full original
            # optimizer dynamics, rather than subtracting a fixed-denominator term.
            ablation = restore_copy(payload, rollout)
            ablation.entropy_coef = 0.0
            torch.manual_seed(shuffle_seed)
            ablation_losses = ablation.update()
            trials[-1]['zero_entropy_copy'] = {
                'final_std': ablation.actor_critic.std.detach().tolist(),
                'losses': ablation_losses, 'final_main_lr': ablation.learning_rate,
                'changed_setting': {'entropy_coef': [algorithm.entropy_coef, 0.0]}}
        assert sha256(checkpoint_path) == checkpoint_hash and sha256(rollout_path) == rollout_hash
        results.append({'iteration': iteration, 'checkpoint_sha256': checkpoint_hash,
                        'rollout_sha256': rollout_hash, 'point': point, 'trials': trials,
                        'restored_lr': payload['learning_rate_state']['main'],
                        'checkpoint_progress': payload['progress'],
                        'std_adam_step_before': trials[0]['events'][0]['adam_step'] - 1,
                        'entropy_coef': payload['config_snapshot']['training']['algorithm']['entropy_coef']})
        print(json.dumps({'iteration': iteration, 'trials': 3, 'completed': True}), flush=True)
    source_hashes = {name: sha256(args.source / 'legged_gym/b2_gym_learn/ppo_cse_pf' / name)
                     for name in ('v3_actor_critic.py', 'v3_ppo.py', 'v3_rollout_storage.py')}
    report = {'results': results, 'source_sha256': source_hashes,
              'script_sha256': sha256(Path(__file__)), 'observer_bitwise_neutrality': True,
              'private_rollout_updates': 19, 'formal_policy_updates': 0,
              '说明': '三种shuffle仅检验同一小rollout的更新方向稳健性，不是独立训练seed。Adam贡献固定实际分母；展开后分别追踪存档动量与本rollout的两类梯度。另做原PPO零熵内存副本，重算其自身分母／KL／裁剪与辅助更新。两者均不能替代闭环重新训练或解释累计因果。'}
    (args.output / 'std_updates.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()

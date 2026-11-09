"""Exercise the confirmed V3 PPO interfaces with real models and offline tensors."""

from typing import Dict, Tuple

import pytest
import torch

from legged_gym.b2_gym_learn.ppo_cse_pf.v3_actor_critic import V3ActorCritic
from legged_gym.b2_gym_learn.ppo_cse_pf.v3_ppo import V3PPO


def make_ppo(num_envs: int = 3) -> V3PPO:
    """Construct the actual V3 network with the official effective PPO parameters.

    Args:
        num_envs: Number of offline synchronous trajectories.

    Returns:
        CPU PPO with allocated T=24 storage and both independent Adam optimizers.
    """
    torch.set_num_threads(2)
    model = V3ActorCritic(1856, 294, 12, 58, 14)
    algorithm = V3PPO(model, device="cpu")
    algorithm.init_storage(num_envs, 24, [1856], [294], [12], [14])
    return algorithm


def empty_infos(num_envs: int) -> Dict[str, object]:
    """Provide the environment's valid no-termination result at the simulator seam.

    Args:
        num_envs: Number of trajectories represented by the observation batch.

    Returns:
        Empty terminal rows and explicit boolean failure/timeout flags.
    """
    return {"terminated": torch.zeros(num_envs, dtype=torch.bool),
            "time_outs": torch.zeros(num_envs, dtype=torch.bool),
            "terminal_env_ids": torch.empty(0, dtype=torch.long),
            "terminal_privileged_obs": torch.empty(0, 294), "episode_records": []}


def test_timeout_uses_unordered_terminal_successor_once_without_mutating_reward() -> None:
    """Distinguish old V=2, terminal V=3 and reset V=100 for mixed done rows.

    Raises:
        AssertionError: A timeout uses the wrong row/value, leaks or mutates raw reward.
    """
    algorithm = make_ppo()
    with torch.no_grad():
        for parameter in algorithm.actor_critic.critic_body.parameters():
            parameter.zero_()
        for layer in algorithm.actor_critic.critic_body:
            if isinstance(layer, torch.nn.Linear):
                layer.weight[0, 0] = 1.0
    observations = torch.zeros(3, 1856)
    critic_observations = torch.zeros(3, 294)
    critic_observations[:, 0] = 2.0
    algorithm.act(observations, critic_observations, torch.zeros(3, 12))
    terminal_observations = torch.zeros(2, 294)
    terminal_observations[:, 0] = torch.tensor([7.0, 3.0])
    records = [{"env_id": 0, "timeout_bootstrap_bonus": None},
               {"env_id": 1, "timeout_bootstrap_bonus": None}]
    infos = {"terminated": torch.tensor([False, True, False]),
             "time_outs": torch.tensor([True, False, False]),
             "terminal_env_ids": torch.tensor([1, 0]),
             "terminal_privileged_obs": terminal_observations, "episode_records": records}
    raw_rewards = torch.ones(3)
    algorithm.process_env_step(raw_rewards, torch.tensor([1, 1, 0]), infos)
    torch.testing.assert_close(raw_rewards, torch.ones(3))
    torch.testing.assert_close(algorithm.storage.rewards[0, :, 0], torch.tensor([3.97, 1.0, 1.0]))
    assert abs(records[0]["timeout_bootstrap_bonus"] - 2.97) < 1e-5
    assert records[1]["timeout_bootstrap_bonus"] == 0.0
    assert algorithm.storage.step == 1


@pytest.mark.parametrize("force_labels", [(0., 0., 0., 0., 0., 0.), (.2, -.1, .05, 0., 0., 0.)])
def test_full_update_uses_twenty_main_and_auxiliary_steps_with_separate_momenta(
    force_labels: Tuple[float, ...]
) -> None:
    """Run a real offline rollout/update and check public optimizer ownership/state.

    Args:
        force_labels: Six scaled EE/base force labels, covering A zero and B nonzero.

    Raises:
        AssertionError: Main/auxiliary Adam states overlap incorrectly or skip updates.
    """
    algorithm = make_ppo(1)
    observations = torch.randn(1, 1856) * 0.02
    critic = torch.randn(1, 294) * 0.02
    labels = torch.tensor([[0.1] * 6 + list(force_labels)])
    decoder_before = algorithm.actor_critic.adaptation_decoder_module[-1].weight.detach().clone()
    for _ in range(24):
        algorithm.act(observations, critic, labels)
        algorithm.process_env_step(torch.tensor([0.1]), torch.zeros(1, dtype=torch.long), empty_infos(1))
    algorithm.compute_returns(critic)
    losses = algorithm.update()
    assert all(torch.isfinite(torch.tensor(value)) for value in losses[:3])
    assert abs(losses[2] - sum(losses[3].values())) < 1e-6
    assert algorithm.storage.step == 0 and not algorithm.storage.returns_ready
    model_parameters = dict(algorithm.actor_critic.named_parameters())
    for name, parameter in model_parameters.items():
        if name.startswith("adaptation_decoder_module"):
            assert parameter not in algorithm.optimizer.state
            assert int(algorithm.adaptation_module_optimizer.state[parameter]["step"]) == 20
        elif name.startswith("adaptation_encoder_module"):
            assert int(algorithm.optimizer.state[parameter]["step"]) == 20
            assert int(algorithm.adaptation_module_optimizer.state[parameter]["step"]) == 20
            assert algorithm.optimizer.state[parameter]["exp_avg"].data_ptr() != algorithm.adaptation_module_optimizer.state[parameter]["exp_avg"].data_ptr()
        else:
            assert int(algorithm.optimizer.state[parameter]["step"]) == 20
            assert parameter not in algorithm.adaptation_module_optimizer.state
    assert all(parameter.grad is None for parameter in model_parameters.values())
    assert algorithm.adaptation_module_optimizer.param_groups[0]["lr"] == 1e-5
    assert not torch.equal(decoder_before, algorithm.actor_critic.adaptation_decoder_module[-1].weight)

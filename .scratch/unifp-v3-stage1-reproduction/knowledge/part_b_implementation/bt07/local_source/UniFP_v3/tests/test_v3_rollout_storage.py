"""Verify A07 storage ownership, terminal GAE and complete sample coverage."""

from typing import List

import pytest
import torch

from legged_gym.b2_gym_learn.ppo_cse_pf.v3_rollout_storage import V3RolloutStorage


def make_storage() -> V3RolloutStorage:
    """Build a CPU rollout at the public storage seam.

    Returns:
        One environment with 24 transitions and compact independent field widths.
    """
    return V3RolloutStorage(1, 24, [1], [1], [1], [1], device="cpu")


def fill_storage(storage: V3RolloutStorage, rewards: List[float], dones: List[bool]) -> None:
    """Insert transitions whose field offsets reveal accidental index misalignment.

    Args:
        storage: Empty CPU storage receiving exactly 24 independent transitions.
        rewards: Scalar rewards in time order; timeout bonuses are already included.
        dones: True episode endings, independent of the rollout boundary.
    """
    for index in range(24):
        transition = storage.Transition()
        transition.observations = torch.tensor([[float(index)]], requires_grad=True)
        transition.critic_observations = torch.tensor([[float(index + 100)]])
        transition.observation_preds = torch.tensor([[float(index + 200)]])
        transition.actions = torch.tensor([[float(index + 300)]])
        transition.values = torch.tensor([[2.0]])
        transition.action_mean = torch.zeros(1, 1)
        transition.action_sigma = torch.ones(1, 1)
        transition.actions_log_prob = torch.zeros(1)
        transition.rewards = torch.tensor([rewards[index]])
        transition.dones = torch.tensor([dones[index]])
        storage.add_transitions(transition)
        transition.observations.detach().fill_(-999.0)


def test_returns_use_raw_gae_and_do_not_cross_done() -> None:
    """Check hand-worked returns, bootstrap isolation and rollout normalization.

    Raises:
        AssertionError: Returns leak into another episode or use normalized advantages.
    """
    storage = make_storage()
    rewards = [0.0] * 22 + [1.0, 3.97]
    fill_storage(storage, rewards, [True] * 22 + [False, True])
    storage.compute_returns(torch.tensor([[100.0]]), gamma=0.99, lam=0.95)
    torch.testing.assert_close(storage.returns[-2:, 0, 0], torch.tensor([4.832785, 3.97]))
    assert storage.dones.dtype == torch.bool
    assert not storage.observations.requires_grad
    assert float(storage.observations[23, 0, 0]) == 23.0
    assert abs(float(storage.advantages.mean())) < 1e-6
    assert abs(float(storage.advantages.std()) - 1.0) < 1e-6
    before_statistics = storage.dones.clone()
    storage.get_statistics()
    torch.testing.assert_close(storage.dones, before_statistics)


def test_minibatches_cover_every_sample_with_paired_fields_and_reused_order() -> None:
    """Check one permutation per update and reject incomplete or stale returns.

    Raises:
        AssertionError: Samples are missing, mismatched or reused after clear.
    """
    storage = make_storage()
    with pytest.raises(RuntimeError):
        storage.compute_returns(torch.zeros(1, 1), 0.99, 0.95)
    fill_storage(storage, [0.0] * 24, [False] * 24)
    with pytest.raises(RuntimeError):
        list(storage.mini_batch_generator(4, 2))
    storage.compute_returns(torch.tensor([[3.0]]), 0.99, 0.95)
    batches = list(storage.mini_batch_generator(4, 2))
    first_order = torch.cat([batch[0][:, 0] for batch in batches[:4]])
    second_order = torch.cat([batch[0][:, 0] for batch in batches[4:]])
    torch.testing.assert_close(first_order, second_order)
    torch.testing.assert_close(first_order.sort().values, torch.arange(24).float())
    for batch in batches:
        torch.testing.assert_close(batch[1], batch[0] + 100)
        torch.testing.assert_close(batch[2], batch[0] + 200)
        torch.testing.assert_close(batch[3], batch[0] + 300)
        assert batch[4].ndim == 2 and batch[-1] is None
    with pytest.raises(ValueError):
        list(storage.mini_batch_generator(5, 1))
    storage.clear()
    assert storage.step == 0 and not storage.returns_ready
    with pytest.raises(RuntimeError):
        list(storage.mini_batch_generator(4, 1))

"""Preallocated feed-forward UniFP storage with explicit A07 boundary semantics."""

from dataclasses import dataclass
from typing import Iterator, Optional, Sequence, Tuple

import torch


class V3RolloutStorage:
    """Own one rollout and preserve the official eleven-field minibatch interface."""

    @dataclass
    class Transition:
        """Hold detached action-boundary fields until their single storage copy."""

        observations: Optional[torch.Tensor] = None
        critic_observations: Optional[torch.Tensor] = None
        observation_preds: Optional[torch.Tensor] = None
        actions: Optional[torch.Tensor] = None
        rewards: Optional[torch.Tensor] = None
        dones: Optional[torch.Tensor] = None
        values: Optional[torch.Tensor] = None
        actions_log_prob: Optional[torch.Tensor] = None
        action_mean: Optional[torch.Tensor] = None
        action_sigma: Optional[torch.Tensor] = None

        def clear(self) -> None:
            """Release references after a complete transition has been copied.

            Note:
                Mutates only the temporary transition, never rollout tensors.
            """
            for name in self.__dataclass_fields__:
                setattr(self, name, None)

    def __init__(self, num_envs: int, num_transitions_per_env: int,
                 obs_shape: Sequence[int], privileged_obs_shape: Sequence[int],
                 obs_pred_shape: Sequence[int], actions_shape: Sequence[int],
                 device: str = "cpu") -> None:
        """Allocate independent float32 fields and boolean episode boundaries.

        Args:
            num_envs: Number of synchronous environments N.
            num_transitions_per_env: Rollout length T, fixed to 24 for V3 training.
            obs_shape: Actor history shape, [1856] in the real V3 task.
            privileged_obs_shape: Critic history shape, [294] in V3.
            obs_pred_shape: Environment supervision shape, [12] in V3.
            actions_shape: Raw policy action shape, [14] in V3.
            device: Explicit tensor device; all input fields must match it.

        Raises:
            ValueError: N or T is not a positive integer.
        """
        if type(num_envs) is not int or num_envs < 1 or type(num_transitions_per_env) is not int or num_transitions_per_env < 1:
            raise ValueError("Rollout dimensions must be positive integers")
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env
        self.device = torch.device(device)
        field_widths = {"observations": obs_shape, "privileged_observations": privileged_obs_shape,
                        "observation_preds": obs_pred_shape, "actions": actions_shape,
                        "mu": actions_shape, "sigma": actions_shape,
                        "rewards": [1], "values": [1], "actions_log_prob": [1],
                        "returns": [1], "advantages": [1]}
        for name, shape in field_widths.items():
            setattr(self, name, torch.zeros(num_transitions_per_env, num_envs, *shape, device=self.device))
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, dtype=torch.bool, device=self.device)
        self.step = 0
        self.returns_ready = False

    @torch.no_grad()
    def add_transitions(self, transition: Transition) -> None:
        """Validate a complete transition before copying its paired fields.

        Args:
            transition: Same-boundary observations, raw actions and old distribution,
                followed by reward [N] and bool or 0/1 integer dones [N].

        Raises:
            RuntimeError: The rollout is full or a required field is missing.
            ValueError: A field has a wrong shape, dtype, device or non-finite value.

        Note:
            Copies detached data and advances the valid length only after all fields.
        """
        if self.step >= self.num_transitions_per_env:
            raise RuntimeError("Rollout buffer overflow")
        pairs = {"observations": transition.observations,
                 "privileged_observations": transition.critic_observations,
                 "observation_preds": transition.observation_preds, "actions": transition.actions,
                 "values": transition.values, "mu": transition.action_mean, "sigma": transition.action_sigma,
                 "actions_log_prob": transition.actions_log_prob, "rewards": transition.rewards,
                 "dones": transition.dones}
        validated = {}
        for name, source in pairs.items():
            if source is None:
                raise RuntimeError(f"Missing transition field: {name}")
            destination = getattr(self, name)[self.step]
            if name in {"rewards", "dones", "actions_log_prob"} and source.shape == (self.num_envs,):
                source = source.unsqueeze(-1)
            if source.shape != destination.shape or source.device != self.device:
                raise ValueError(f"Invalid transition shape/device: {name}")
            if name == "dones":
                if source.dtype not in (torch.bool, torch.int32, torch.int64) or not ((source == 0) | (source == 1)).all():
                    raise ValueError("dones must contain boolean episode boundaries")
            elif source.dtype != torch.float32 or not torch.isfinite(source).all():
                raise ValueError(f"Invalid transition dtype/finite values: {name}")
            validated[name] = source.detach()
        if not (validated["sigma"] > 0.0).all():
            raise ValueError("Stored action standard deviations must be positive")
        for name, source in validated.items():
            getattr(self, name)[self.step].copy_(source)
        self.step += 1
        self.returns_ready = False

    def clear(self) -> None:
        """Invalidate the previous rollout without clearing its allocated memory.

        Note:
            Resets valid length and returns readiness; next collection must refill it.
        """
        self.step = 0
        self.returns_ready = False

    @torch.no_grad()
    def compute_returns(self, last_values: torch.Tensor, gamma: float, lam: float) -> None:
        """Compute done-masked GAE, raw returns and whole-rollout normalized advantages.

        Args:
            last_values: True surviving successor values [N,1], zero for done rows.
            gamma: Reward discount per policy step.
            lam: GAE decay per policy step.

        Raises:
            RuntimeError: Collection is incomplete or targets are non-finite.
            ValueError: Successor values have an incompatible shape or device.

        Note:
            Timeout reward compensation has already occurred exactly once in PPO.
        """
        if self.step != self.num_transitions_per_env:
            raise RuntimeError("Returns require a complete rollout")
        if last_values.shape != (self.num_envs, 1) or last_values.device != self.device or last_values.dtype != torch.float32:
            raise ValueError("Invalid rollout successor values")
        advantage = torch.zeros_like(last_values)
        for index in reversed(range(self.step)):
            next_values = last_values if index == self.step - 1 else self.values[index + 1]
            continuation = (~self.dones[index]).float()
            delta = self.rewards[index] + continuation * gamma * next_values - self.values[index]
            advantage = delta + continuation * gamma * lam * advantage
            self.returns[index].copy_(advantage + self.values[index])
        raw_advantages = self.returns - self.values
        self.advantages.copy_((raw_advantages - raw_advantages.mean()) / (raw_advantages.std(unbiased=True) + 1e-8))
        if not torch.isfinite(self.returns).all() or not torch.isfinite(self.advantages).all():
            raise RuntimeError("GAE produced non-finite targets")
        self.returns_ready = True

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 5) -> Iterator[Tuple[Optional[torch.Tensor], ...]]:
        """Yield aligned official-order batches, reusing one permutation across epochs.

        Args:
            num_mini_batches: Positive divisor of T*N; samples are never dropped.
            num_epochs: Positive number of passes over the same rollout.

        Yields:
            Actor, Critic, labels, actions, old V, advantages, returns, old logp,
            old mu, old sigma, and the non-recurrent None mask placeholder.

        Raises:
            RuntimeError: Collection or return calculation is incomplete.
            ValueError: Epoch count or minibatch partition is invalid.
        """
        if self.step != self.num_transitions_per_env or not self.returns_ready:
            raise RuntimeError("Minibatches require a complete rollout and fresh returns")
        batch_size = self.num_envs * self.num_transitions_per_env
        if type(num_mini_batches) is not int or num_mini_batches < 1 or batch_size % num_mini_batches or type(num_epochs) is not int or num_epochs < 1:
            raise ValueError("Minibatches must divide the full rollout and epochs must be positive")
        indices = torch.randperm(batch_size, device=self.device)
        width = batch_size // num_mini_batches
        field_names = ("observations", "privileged_observations", "observation_preds", "actions", "values",
                       "advantages", "returns", "actions_log_prob", "mu", "sigma")
        flat_fields = [getattr(self, name).flatten(0, 1) for name in field_names]
        for _ in range(num_epochs):
            for batch_index in range(num_mini_batches):
                selected = indices[batch_index * width:(batch_index + 1) * width]
                yield (*[field[selected] for field in flat_fields], None)

    def get_statistics(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Describe collected trajectory segments without changing true episode ends.

        Returns:
            Mean collected segment length in policy steps and mean training reward.

        Raises:
            RuntimeError: No transitions have been collected.
        """
        if self.step == 0:
            raise RuntimeError("Statistics require collected transitions")
        ends = self.dones[:self.step].clone()
        ends[-1] = True
        flat_ends = ends.permute(1, 0, 2).reshape(-1)
        boundaries = torch.cat((torch.tensor([-1], device=self.device), flat_ends.nonzero().flatten()))
        return (boundaries[1:] - boundaries[:-1]).float().mean(), self.rewards[:self.step].mean()

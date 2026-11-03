"""V3 Actor-Critic model matching the confirmed reproduction schemas."""

import math
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn
from torch.distributions import Normal


ACTOR_OBS_DIM = 1856
CRITIC_OBS_DIM = 294
PREDICTION_DIM = 12
SINGLE_OBS_DIM = 58
ACTION_DIM = 14
LATENT_DIM = 64
ACTOR_HIDDEN_DIMS = (512, 256, 128)
CRITIC_HIDDEN_DIMS = (512, 256, 128)
ENCODER_HIDDEN_DIMS = (512, 256, 128)
DECODER_HIDDEN_DIMS = (128, 64)
ADAPTATION_LABELS = (
    "base_velocity_loss",
    "tcp_position_loss",
    "force_ee_loss",
    "force_base_loss",
)
ADAPTATION_DIMS = (3, 3, 3, 3)
ADAPTATION_WEIGHTS = (0.2, 0.2, 1.0, 1.0)


class V3ActorCritic(nn.Module):
    """Implement the V3 policy, value function, and shared-latent estimator."""

    is_recurrent = False

    def __init__(
        self,
        num_obs: int,
        num_privileged_obs: int,
        num_obs_pred: int,
        num_single_obs: int,
        num_actions: int,
        actor_hidden_dims: Sequence[int] = ACTOR_HIDDEN_DIMS,
        critic_hidden_dims: Sequence[int] = CRITIC_HIDDEN_DIMS,
        activation: str = "elu",
        init_noise_std: float = 1.0,
    ) -> None:
        """Build fixed-schema V3 networks and a state-independent Gaussian scale.

        Args:
            num_obs: Flattened Actor history width; must be 1856 (32 x 58).
            num_privileged_obs: Flattened Critic history width; must be 294.
            num_obs_pred: Auxiliary target width; must be 12.
            num_single_obs: Actor frame width; must be 58.
            num_actions: Policy action width; must be 14.
            actor_hidden_dims: Actor hidden widths; the confirmed value is (512, 256, 128).
            critic_hidden_dims: Critic hidden widths; the confirmed value is (512, 256, 128).
            activation: Hidden-layer activation; the confirmed value is ``"elu"``.
            init_noise_std: Initial dimensionless standard deviation for each action.

        Raises:
            ValueError: A schema dimension, network width, activation, or initial scale
                differs from the confirmed V3 model contract.
        """
        super().__init__()
        schema = (num_obs, num_privileged_obs, num_obs_pred, num_single_obs, num_actions)
        expected_schema = (ACTOR_OBS_DIM, CRITIC_OBS_DIM, PREDICTION_DIM, SINGLE_OBS_DIM, ACTION_DIM)
        if schema != expected_schema:
            raise ValueError(f"V3 model schema mismatch: expected {expected_schema}, got {schema}")
        if tuple(actor_hidden_dims) != ACTOR_HIDDEN_DIMS:
            raise ValueError(f"V3 actor hidden widths must be {ACTOR_HIDDEN_DIMS}")
        if tuple(critic_hidden_dims) != CRITIC_HIDDEN_DIMS:
            raise ValueError(f"V3 critic hidden widths must be {CRITIC_HIDDEN_DIMS}")
        if activation != "elu":
            raise ValueError("V3 model uses ELU hidden activations")
        if not math.isfinite(init_noise_std) or init_noise_std <= 0.0:
            raise ValueError("Initial action standard deviation must be finite and positive")

        self.num_obs = num_obs
        self.num_privileged_obs = num_privileged_obs
        self.num_obs_pred = num_obs_pred
        self.num_obs_now = num_single_obs
        self.num_latent_dim = LATENT_DIM
        self.adaptation_labels = ADAPTATION_LABELS
        self.adaptation_dims = ADAPTATION_DIMS
        self.adaptation_weights = ADAPTATION_WEIGHTS

        self.adaptation_encoder_module = self._build_mlp(
            ACTOR_OBS_DIM, ENCODER_HIDDEN_DIMS, LATENT_DIM
        )
        self.adaptation_decoder_module = self._build_mlp(
            LATENT_DIM, DECODER_HIDDEN_DIMS, PREDICTION_DIM
        )
        self.actor_body = self._build_mlp(
            SINGLE_OBS_DIM + LATENT_DIM, ACTOR_HIDDEN_DIMS, ACTION_DIM
        )
        self.critic_body = self._build_mlp(
            CRITIC_OBS_DIM, CRITIC_HIDDEN_DIMS, 1
        )
        self.std = nn.Parameter(init_noise_std * torch.ones(ACTION_DIM))
        self.distribution: Optional[Normal] = None

    @staticmethod
    def _build_mlp(
        input_dim: int, hidden_dims: Sequence[int], output_dim: int
    ) -> nn.Sequential:
        """Build a default-initialized ELU MLP with an unactivated output layer.

        Args:
            input_dim: Width of the input vector.
            hidden_dims: Ordered widths of hidden layers.
            output_dim: Width of the final linear layer.

        Returns:
            Sequential network with ELU after each hidden linear layer.
        """
        widths = (input_dim, *hidden_dims, output_dim)
        layers = []
        for layer_index, (layer_input_dim, layer_output_dim) in enumerate(
            zip(widths, widths[1:])
        ):
            layers.append(nn.Linear(layer_input_dim, layer_output_dim))
            if layer_index < len(widths) - 2:
                layers.append(nn.ELU())
        return nn.Sequential(*layers)

    def _encode_observations(self, observations: torch.Tensor) -> torch.Tensor:
        """Validate and encode one batch of flattened Actor histories.

        Args:
            observations: Float32 Actor histories with shape ``[batch, 1856]``.

        Returns:
            Shared latent tensor with shape ``[batch, 64]``.

        Raises:
            ValueError: The input shape, dtype, or device is incompatible.
            FloatingPointError: The observations or encoded latent are non-finite.
        """
        if observations.ndim != 2 or observations.shape[1] != ACTOR_OBS_DIM:
            raise ValueError(f"Expected Actor observations [batch, 1856], got {tuple(observations.shape)}")
        if observations.dtype != torch.float32 or observations.device != self.std.device:
            raise ValueError("Actor observations must be float32 on the model device")
        if not torch.isfinite(observations).all():
            raise FloatingPointError("Actor observations contain non-finite values")
        latent = self.adaptation_encoder_module(observations)
        if not torch.isfinite(latent).all():
            raise FloatingPointError("V3 encoder produced a non-finite latent")
        return latent

    def update_distribution(self, observations: torch.Tensor) -> None:
        """Update the cached diagonal Gaussian from an Actor history batch.

        Args:
            observations: Float32 Actor histories with shape ``[batch, 1856]``.

        Raises:
            ValueError: The learned action standard deviation is not strictly positive.
            FloatingPointError: The action mean or standard deviation is non-finite.

        Note:
            Replaces the cached distribution; it does not sample an action.
        """
        latent = self._encode_observations(observations)
        current_frame = observations[:, -self.num_obs_now :]
        mean = self.actor_body(torch.cat((current_frame, latent), dim=-1))
        if not torch.isfinite(mean).all() or not torch.isfinite(self.std).all():
            raise FloatingPointError("V3 policy distribution parameters are non-finite")
        if torch.any(self.std <= 0.0):
            raise ValueError("V3 action standard deviation must remain positive")
        scale = torch.ones_like(mean) * self.std
        self.distribution = Normal(mean, scale)

    @property
    def action_mean(self) -> torch.Tensor:
        """Return the cached deterministic policy mean.

        Returns:
            Action means with shape ``[batch, 14]``.

        Raises:
            RuntimeError: No policy distribution has been cached.
        """
        if self.distribution is None:
            raise RuntimeError("Call update_distribution before reading action_mean")
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        """Return the cached broadcast policy standard deviation.

        Returns:
            Standard deviations with shape ``[batch, 14]``.

        Raises:
            RuntimeError: No policy distribution has been cached.
        """
        if self.distribution is None:
            raise RuntimeError("Call update_distribution before reading action_std")
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        """Return joint entropy summed across the 14 action dimensions.

        Returns:
            Entropy values with shape ``[batch]``.

        Raises:
            RuntimeError: No policy distribution has been cached.
        """
        if self.distribution is None:
            raise RuntimeError("Call update_distribution before reading entropy")
        return self.distribution.entropy().sum(dim=-1)

    def act(self, observations: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Sample the raw 14-dimensional action used by PPO.

        Args:
            observations: Float32 Actor histories with shape ``[batch, 1856]``.
            **kwargs: Unused recurrent-policy arguments accepted by the PPO interface.

        Returns:
            Unclipped action samples with shape ``[batch, 14]``.

        Note:
            Sampling consumes the active Torch random generator state.
        """
        self.update_distribution(observations)
        if self.distribution is None:
            raise RuntimeError("Policy distribution was not initialized")
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Return joint log probability for raw actions under the cached policy.

        Args:
            actions: Raw action samples with shape ``[batch, 14]``.

        Returns:
            Joint action log probabilities with shape ``[batch]``.

        Raises:
            RuntimeError: No distribution is cached.
            ValueError: Action shape is incompatible.
            FloatingPointError: An action sample is non-finite.
        """
        if self.distribution is None:
            raise RuntimeError("Call update_distribution before evaluating action log probability")
        if actions.shape != self.distribution.mean.shape:
            raise ValueError("Action batch shape does not match the cached distribution")
        if not torch.isfinite(actions).all():
            raise FloatingPointError("Action samples contain non-finite values")
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_student_latent(self, observations: torch.Tensor) -> torch.Tensor:
        """Predict the 12-dimensional auxiliary state/force target from history.

        Args:
            observations: Float32 Actor histories with shape ``[batch, 1856]``.

        Returns:
            Differentiable prediction tensor with shape ``[batch, 12]``.

        Raises:
            FloatingPointError: The decoder produces non-finite predictions.
        """
        latent = self._encode_observations(observations)
        predictions = self.adaptation_decoder_module(latent)
        if not torch.isfinite(predictions).all():
            raise FloatingPointError("V3 estimator produced a non-finite prediction")
        return predictions

    def act_inference(
        self,
        observations: Dict[str, torch.Tensor],
        policy_info: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Return deterministic actions and optionally same-device state estimates.

        Args:
            observations: Observation package containing ``obs[N, 1856]``.
            policy_info: Optional output dictionary; when provided, receives
                detached ``state_estimates[N, 12]`` without device transfer.

        Returns:
            Deterministic action means with shape ``[N, 14]``.

        Raises:
            KeyError: The observation package does not contain ``obs``.
        """
        return self.act_student(observations["obs"], policy_info)

    def act_student(
        self,
        observations: torch.Tensor,
        policy_info: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Run the deterministic V3 policy in inference mode.

        Args:
            observations: Float32 Actor histories with shape ``[batch, 1856]``.
            policy_info: Optional dictionary for detached ``state_estimates`` output.

        Returns:
            Deterministic action means with shape ``[batch, 14]``.

        Note:
            When ``policy_info`` is provided, adds detached predictions on the
            model device under the ``state_estimates`` key.
        """
        with torch.inference_mode():
            latent = self._encode_observations(observations)
            current_frame = observations[:, -self.num_obs_now :]
            action_mean = self.actor_body(torch.cat((current_frame, latent), dim=-1))
            if not torch.isfinite(action_mean).all():
                raise FloatingPointError("V3 inference policy produced a non-finite action")
            if policy_info is not None:
                state_estimates = self.adaptation_decoder_module(latent)
                if not torch.isfinite(state_estimates).all():
                    raise FloatingPointError("V3 inference estimator produced a non-finite prediction")
                policy_info["state_estimates"] = state_estimates.detach()
            return action_mean

    def evaluate(self, critic_observations: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Evaluate one scalar value from the clean privileged Critic history.

        Args:
            critic_observations: Float32 Critic histories with shape ``[batch, 294]``.
            **kwargs: Unused recurrent-policy arguments accepted by the PPO interface.

        Returns:
            Scalar values with shape ``[batch, 1]``.

        Raises:
            ValueError: The input shape, dtype, or device is incompatible.
            FloatingPointError: The input or predicted value is non-finite.
        """
        if critic_observations.ndim != 2 or critic_observations.shape[1] != CRITIC_OBS_DIM:
            raise ValueError(
                f"Expected Critic observations [batch, 294], got {tuple(critic_observations.shape)}"
            )
        if critic_observations.dtype != torch.float32 or critic_observations.device != self.std.device:
            raise ValueError("Critic observations must be float32 on the model device")
        if not torch.isfinite(critic_observations).all():
            raise FloatingPointError("Critic observations contain non-finite values")
        values = self.critic_body(critic_observations)
        if not torch.isfinite(values).all():
            raise FloatingPointError("V3 Critic produced a non-finite value")
        return values

    def reset(self, dones: Optional[torch.Tensor] = None) -> None:
        """Keep the stateless feedforward model unchanged across environment resets.

        Args:
            dones: Optional done flags accepted for compatibility with PPO callers.

        Returns:
            None.

        Note:
            This feedforward model carries no recurrent state to reset.
        """
        return None

    def forward(self) -> None:
        """Reject implicit module forwarding in favor of named model interfaces.

        Raises:
            NotImplementedError: Actor, Critic, or estimator input must be selected.

        Returns:
            No value; implicit forward calls are unsupported.
        """
        raise NotImplementedError("Call act, evaluate, or get_student_latent explicitly")

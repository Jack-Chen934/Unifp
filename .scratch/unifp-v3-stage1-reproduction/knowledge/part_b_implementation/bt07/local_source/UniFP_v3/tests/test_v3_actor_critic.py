"""Unit tests for the fixed-schema V3 Actor-Critic model."""

import pytest
import torch
from torch import nn

from legged_gym.b2_gym_learn.ppo_cse_pf.v3_actor_critic import V3ActorCritic


def _new_model() -> V3ActorCritic:
    """Construct the confirmed V3 network dimensions for focused unit tests.

    Returns:
        A CPU model with the fixed A05/A06 input and output schema.
    """
    return V3ActorCritic(
        num_obs=1856,
        num_privileged_obs=294,
        num_obs_pred=12,
        num_single_obs=58,
        num_actions=14,
    )


def test_v3_model_shapes_distribution_and_inference() -> None:
    """Check confirmed linear widths, Gaussian actions, value, and estimator shapes.

    Args:
        None.

    Returns:
        None.
    """
    model = _new_model()
    actor_observations = torch.randn(3, 1856)
    critic_observations = torch.randn(3, 294)
    prediction_targets = torch.randn(3, 12)

    assert [(layer.in_features, layer.out_features) for layer in model.adaptation_encoder_module
            if isinstance(layer, nn.Linear)] == [(1856, 512), (512, 256), (256, 128), (128, 64)]
    assert [(layer.in_features, layer.out_features) for layer in model.actor_body
            if isinstance(layer, nn.Linear)] == [(122, 512), (512, 256), (256, 128), (128, 14)]
    assert [(layer.in_features, layer.out_features) for layer in model.adaptation_decoder_module
            if isinstance(layer, nn.Linear)] == [(64, 128), (128, 64), (64, 12)]
    assert [(layer.in_features, layer.out_features) for layer in model.critic_body
            if isinstance(layer, nn.Linear)] == [(294, 512), (512, 256), (256, 128), (128, 1)]

    sampled_actions = model.act(actor_observations)
    action_log_probabilities = model.get_actions_log_prob(sampled_actions)
    predicted_values = model.evaluate(critic_observations)
    auxiliary_predictions = model.get_student_latent(actor_observations)
    inference_info = {}
    deterministic_actions = model.act_inference({"obs": actor_observations}, inference_info)

    assert sampled_actions.shape == (3, 14)
    assert action_log_probabilities.shape == (3,)
    assert model.action_mean.shape == (3, 14)
    assert model.action_std.shape == (3, 14)
    assert model.entropy.shape == (3,)
    assert predicted_values.shape == (3, 1)
    assert auxiliary_predictions.shape == prediction_targets.shape
    assert deterministic_actions.shape == (3, 14)
    assert not deterministic_actions.requires_grad
    assert set(inference_info) == {"state_estimates"}
    assert inference_info["state_estimates"].shape == (3, 12)
    assert inference_info["state_estimates"].device == actor_observations.device
    assert all(torch.isfinite(tensor).all() for tensor in (
        sampled_actions, action_log_probabilities, predicted_values, auxiliary_predictions
    ))


def test_v3_model_gradient_paths_are_separated() -> None:
    """Check that policy, value, and auxiliary losses reach only confirmed modules.

    Args:
        None.

    Returns:
        None.
    """
    model = _new_model()
    actor_observations = torch.randn(4, 1856)
    critic_observations = torch.randn(4, 294)
    prediction_targets = torch.randn(4, 12)

    sampled_actions = model.act(actor_observations)
    model.get_actions_log_prob(sampled_actions.detach()).mean().backward()
    assert model.adaptation_encoder_module[0].weight.grad is not None
    assert model.actor_body[0].weight.grad is not None
    assert model.std.grad is not None
    assert model.critic_body[0].weight.grad is None
    assert model.adaptation_decoder_module[0].weight.grad is None

    model.zero_grad(set_to_none=True)
    model.evaluate(critic_observations).mean().backward()
    assert model.critic_body[0].weight.grad is not None
    assert model.adaptation_encoder_module[0].weight.grad is None
    assert model.actor_body[0].weight.grad is None
    assert model.adaptation_decoder_module[0].weight.grad is None

    model.zero_grad(set_to_none=True)
    auxiliary_loss = (model.get_student_latent(actor_observations) - prediction_targets).square().mean()
    auxiliary_loss.backward()
    assert model.adaptation_encoder_module[0].weight.grad is not None
    assert model.adaptation_decoder_module[0].weight.grad is not None
    assert model.actor_body[0].weight.grad is None
    assert model.critic_body[0].weight.grad is None
    assert model.std.grad is None


def test_v3_model_rejects_schema_and_invalid_standard_deviation() -> None:
    """Fail promptly for incompatible model shapes and invalid Gaussian scales.

    Args:
        None.

    Returns:
        None.
    """
    with pytest.raises(ValueError, match="schema mismatch"):
        V3ActorCritic(1856, 447, 12, 58, 14)

    model = _new_model()
    observations = torch.randn(2, 1856)
    with pytest.raises(ValueError, match="standard deviation"):
        model.std.data[0] = 0.0
        model.update_distribution(observations)

    with pytest.raises(ValueError, match="Expected Actor observations"):
        model.get_student_latent(torch.zeros(2, 58))

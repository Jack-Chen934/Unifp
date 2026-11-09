"""Check actual V3 force scheduling at the agreed tensor and clock seams."""

import math
from unittest.mock import patch

import isaacgym  # noqa: F401 -- Native bindings must precede Torch.
import torch

from legged_gym.envs.v3.legged_robot_v3 import V3PosForce
from test_v3_rewards import make_reward_environment


def make_force_environment() -> V3PosForce:
    """Add B event state to the real CPU reward/frame fixture.

    Returns:
        Two robots with 20 ms clocks and real V3 force-event methods, without Gym.
    """
    environment = make_reward_environment()
    environment.cfg.run.reproduction_part = "B"
    environment.episode_length_buf = torch.zeros(2, dtype=torch.long)
    environment.num_bodies = 23
    environment.tcp_body_index = 8
    environment._initialize_force_events()
    return environment


def test_trapezoids_have_exact_zero_endpoints_and_independent_force_frames() -> None:
    """Match a hand-worked one-second rise, hold and fall without world drift.

    Raises:
        AssertionError: Events skip zero endpoints or confuse command and external axes.
    """
    environment = make_force_environment()
    ids = torch.arange(2)
    for stream, peak in (("command", (0.0, 10.0, 0.0)), ("external", (20.0, 0.0, 0.0))):
        state = environment.force_state[stream]
        state["peak"][:] = torch.tensor(peak)
        state["enabled"][:] = True
        state["start_tick"][:] = 0
        state["next_start_tick"][:] = 200
        state["ramp_ticks"][:] = 50.0
    for tick, expected_external in ((0, 0.0), (25, 10.0), (50, 20.0), (100, 20.0), (125, 10.0), (150, 0.0), (175, 0.0)):
        environment.episode_length_buf.fill_(tick)
        environment._advance_force_events(ids)
        torch.testing.assert_close(environment.pending_ee_force_world[:, 0], torch.full((2,), expected_external))
        torch.testing.assert_close(environment.commands[:, 10], torch.full((2,), expected_external / 2.0))
        assert torch.count_nonzero(environment.applied_ee_force_world) == 0
    environment.rpy[:] = torch.tensor((0.2, -0.3, math.pi / 2))
    environment.applied_ee_force_world[:] = torch.tensor((20.0, 0.0, 0.0))
    environment._update_applied_force_label()
    torch.testing.assert_close(environment.external_ee_force_y, torch.tensor([[0.0, -20.0, 0.0]] * 2), atol=1e-5, rtol=1e-5)
    assert torch.count_nonzero(environment.external_base_force_y) == 0


def test_force_reset_and_inactive_cycle_resampling_are_local_and_boundary_driven() -> None:
    """Restart selected clocks and resample inactive cycles only at their next start.

    Raises:
        AssertionError: Reset contaminates another row or a disabled stream never restarts.
    """
    environment = make_force_environment()
    environment.cfg.force_events.enabled_probability = 0.0
    environment._reset_force_events(torch.arange(2))
    untouched = {name: {key: value[1].clone() for key, value in state.items()}
                 for name, state in environment.force_state.items()}
    environment._reset_force_events(torch.tensor([0]))
    for name, state in environment.force_state.items():
        for key, value in state.items():
            torch.testing.assert_close(value[1], untouched[name][key], rtol=0.0, atol=0.0)
    for state in environment.force_state.values():
        state["next_start_tick"][0] = 3
    environment.episode_length_buf[0] = 2
    with patch("torch.randint", wraps=torch.randint) as draw_interval:
        environment._advance_force_events(torch.tensor([0]))
        assert draw_interval.call_count == 0
        environment.episode_length_buf[0] = 3
        environment._advance_force_events(torch.tensor([0]))
        assert draw_interval.call_count == 2
    for state in environment.force_state.values():
        assert not state["enabled"][0]
        assert 178 <= int(state["next_start_tick"][0]) <= 452
        assert state["ramp_ticks"][0] <= (state["next_start_tick"][0] - 3 - 50) / 2
    assert torch.count_nonzero(environment.pending_ee_force_world) == 0


def test_zero_base_command_probability_preserves_a_rng_and_zeros_whole_b_rows() -> None:
    """Keep A random consumption exact and add the explicit B row-wise mixture.

    Raises:
        AssertionError: A consumes an extra random draw or B zeroes individual axes.
    """
    environment = make_reward_environment()
    environment.cfg.commands.zero_vel_cmd_prob = 0.0
    rng = torch.get_rng_state().clone()
    environment._sample_base_velocity_commands(8)
    actual_rng = torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    torch.rand(8, 3)
    torch.testing.assert_close(torch.get_rng_state(), actual_rng, rtol=0.0, atol=0.0)
    environment.cfg.commands.zero_vel_cmd_prob = 1.0
    assert torch.count_nonzero(environment._sample_base_velocity_commands(8)) == 0

"""Check B physics submission and transition/reset ordering through real V3 steps."""

from unittest.mock import Mock, patch

import isaacgym  # noqa: F401 -- Native bindings must precede Torch.
from isaacgym import gymapi
import torch
import pytest

from legged_gym.envs.v3.legged_robot_v3 import V3PosForce
from test_v3_force_events import make_force_environment


def make_step_environment() -> V3PosForce:
    """Attach recording-only Gym boundaries to the real two-robot tensor fixture.

    Returns:
        CPU environment using actual control, rewards, targets, frames and reset;
        native Gym calls record submissions without fabricating physical motion.
    """
    environment = make_force_environment()
    environment.gym = Mock()
    environment.sim = object()
    environment.headless = True
    environment.num_dof = 14
    environment.dof_state = torch.zeros(2, 14, 2)
    environment.dof_pos = environment.dof_state[:, :, 0]
    environment.dof_vel = environment.dof_state[:, :, 1]
    environment.default_dof_pos = torch.tensor([list(environment.cfg.init_state.default_joint_angles.values())])
    environment.wheel_dof_indices = torch.arange(4, 8)
    environment.p_gains = torch.ones(14)
    environment.d_gains = torch.ones(14)
    environment.base_body_index = 0
    environment.wheel_body_indices = torch.arange(19, 23)
    environment.rigid_state = torch.zeros(2, 23, 13)
    environment.base_init_state = torch.tensor([0., 0., .34, 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.])
    environment.obs_history = torch.zeros(2, 32, 58)
    environment.critic_history = torch.zeros(2, 3, 98)
    environment.obs_pred = torch.zeros(2, 12)
    environment.contact_valid = torch.zeros(2, dtype=torch.bool)
    environment.mass_com_delta = torch.zeros(2, 5)
    environment.friction = torch.ones(2, 1)
    environment.execution_scale_delta = torch.zeros(2, 14)
    environment.reset_buf = torch.zeros(2, dtype=torch.long)
    environment.max_episode_length = 1000
    environment.completed_iterations = 8000
    environment.training_policy_steps = 192000
    environment.training_transitions = 384000
    environment.last_push_policy_step = 192000
    environment.last_push_velocity_world = torch.zeros(2, 2)
    environment.evaluation_policy_step = None
    environment.push_generator = None
    environment.policy_trace_enabled = False
    environment.cfg.domain_rand.push_robots = False
    environment.cfg.noise.add_noise = False
    for name in ("target_start", "target_end"):
        setattr(environment, name, torch.zeros(2, 3))
    for name in ("target_move_steps", "target_hold_steps", "target_elapsed_steps"):
        setattr(environment, name, torch.zeros(2, dtype=torch.long))
    environment._initialize_episode_metrics()
    with patch("isaacgym.gymtorch.unwrap_tensor", side_effect=torch.clone):
        environment.reset()
    environment.gym.reset_mock()
    return environment


def test_step_submits_only_applied_world_com_force_and_publishes_previous_interval_label() -> None:
    """Separate applied +20 N, next pending -10 N and untouched reset labels.

    Raises:
        AssertionError: A substep submits commands, torque or next-interval forces,
            or publication uses a future force instead of the completed interval.
    """
    environment = make_step_environment()
    environment.episode_length_buf.fill_(24)
    environment.pending_ee_force_world[:, 0] = 20.0
    for name, state in environment.force_state.items():
        state["start_tick"].zero_()
        state["next_start_tick"].fill_(200)
        state["ramp_ticks"].fill_(50.)
        state["enabled"].fill_(True)
        state["peak"][:] = torch.tensor([-20., 0., 0.] if name == "external" else [0., 10., 0.])
    # A deliberately different command force must never be submitted to PhysX.
    environment.commands[:, 9:12] = torch.tensor([0., 30., 0.])
    with patch("isaacgym.gymtorch.unwrap_tensor", side_effect=torch.clone):
        observations, _, dones, _ = environment.step(torch.zeros(2, 14))
    calls = environment.gym.apply_rigid_body_force_tensors.call_args_list
    assert len(calls) == 8
    for call in calls:
        _, forces, torques, frame = call.args
        assert torques is None and frame == gymapi.GLOBAL_SPACE
        assert forces.shape == (46, 3)
        expected = torch.zeros(46, 3)
        expected[[8, 31], 0] = 20.
        torch.testing.assert_close(forces, expected)
    assert not dones.any()
    torch.testing.assert_close(environment.pending_ee_force_world[:, 0], torch.full((2,), -10.))
    torch.testing.assert_close(observations["obs_pred"][:, 6:9], torch.tensor([[.2, 0., 0.]] * 2))
    torch.testing.assert_close(observations["obs"].view(2, 32, 58)[:, -1, 52:55], torch.tensor([[0., .05, 0.]] * 2))


def test_done_clones_true_successor_before_local_reset_and_failure_overrides_timeout() -> None:
    """Preserve applied force in terminal 294 frames while new episodes start at zero.

    Raises:
        AssertionError: Timeout/failure masks overlap, terminal history is replaced
            by reset data, or a new episode inherits previous actions or forces.
    """
    environment = make_step_environment()
    environment.max_episode_length = 25
    environment.episode_length_buf.fill_(24)
    environment.contact_forces[0, 0, 0] = 2.
    environment.critic_history.fill_(7.)
    environment.obs_history.fill_(8.)
    environment.pending_ee_force_world[:, 0] = 20.
    with patch("isaacgym.gymtorch.unwrap_tensor", side_effect=torch.clone):
        observations, _, dones, infos = environment.step(torch.full((2, 14), .1))
    assert dones.tolist() == [1, 1]
    assert infos["terminated"].tolist() == [True, False]
    assert infos["time_outs"].tolist() == [False, True]
    terminal = infos["terminal_privileged_obs"].view(2, 3, 98)
    torch.testing.assert_close(terminal[:, :2], torch.full((2, 2, 98), 7.))
    torch.testing.assert_close(terminal[:, -1, 6:9], torch.tensor([[.2, 0., 0.]] * 2))
    assert torch.count_nonzero(observations["obs_pred"][:, 6:12]) == 0
    assert torch.count_nonzero(environment.applied_ee_force_world) == 0
    assert torch.count_nonzero(environment.pending_ee_force_world) == 0
    assert torch.count_nonzero(environment.obs_history[:, :-1]) == 0
    assert torch.count_nonzero(environment.critic_history[:, :-1]) == 0
    assert torch.count_nonzero(environment.last_actions) == 0
    assert [row["end_type"] for row in infos["episode_records"]] == ["physical_failure", "timeout"]
    # Terminal rows are owned clones, unaffected by later reset or history writes.
    environment.critic_history.zero_()
    torch.testing.assert_close(terminal[:, -1, 6], torch.full((2,), .2))


def test_b_snapshot_is_detached_and_restore_restarts_local_force_events() -> None:
    """Keep diagnostic snapshots independent while restoring properties before reset.

    Raises:
        AssertionError: Snapshot aliases live tensors, B silently accepts missing
            fields, or reset continues a saved half-event on a new robot state.
    """
    environment = make_step_environment()
    environment.pending_ee_force_world[:, 0] = -10.
    environment.applied_ee_force_world[:, 0] = 20.
    snapshot = environment.get_training_state()
    environment.applied_ee_force_world.zero_()
    torch.testing.assert_close(snapshot["force_state"]["applied_ee_force_world"][:, 0], torch.full((2,), 20.))
    environment.load_training_state(snapshot)
    with patch("isaacgym.gymtorch.unwrap_tensor", side_effect=torch.clone):
        observations, _ = environment.reset()
    assert torch.count_nonzero(environment.pending_ee_force_world) == 0
    assert torch.count_nonzero(environment.applied_ee_force_world) == 0
    assert torch.count_nonzero(observations["obs_pred"][:, 6:12]) == 0
    del snapshot["force_state"]
    with pytest.raises(ValueError, match="force"):
        environment.load_training_state(snapshot)
    environment.load_training_state(snapshot, transition_from_a=True)

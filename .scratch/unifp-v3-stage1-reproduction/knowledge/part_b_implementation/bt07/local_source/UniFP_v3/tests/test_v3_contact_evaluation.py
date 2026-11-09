"""Check real contact-environment mapping at the native tensor/setter boundary."""

from unittest.mock import Mock, patch

import isaacgym  # noqa: F401
import torch
import pytest

from legged_gym.envs.v3.v3_contact_evaluation import V3ContactEvaluation
from legged_gym.envs.v3.legged_robot_v3 import V3PosForce


def test_wall_is_created_in_the_current_environment_before_mapping() -> None:
    """Create a wall through the existing per-robot native callback.

    Raises:
        AssertionError: The callback changes robot properties or creates the wall
            in a previous environment, with a different collision group or pose.
    """
    environment = V3ContactEvaluation.__new__(V3ContactEvaluation)
    environment.gym, environment.sim = Mock(), object()
    environment.wall_asset = object()
    environment.wall_handles = []
    environment.contact_manifest = {"scene": {"wall_center_relative_m": [.656, 0., .779]},
                                    "nominal_friction": 1., "contact_offset_m": .01, "rest_offset_m": 0.}
    environment.env_origins = torch.tensor([[4., 6., 0.]])
    environment.gym.get_actor_rigid_shape_properties.return_value = [Mock()]
    properties = [object()]
    assert environment._process_rigid_body_props(properties, 0) is properties
    environment.gym.get_env.assert_called_once_with(environment.sim, 0)
    arguments = environment.gym.create_actor.call_args.args
    assert arguments[0] is environment.gym.get_env.return_value
    assert arguments[1] is environment.wall_asset
    assert [arguments[2].p.x, arguments[2].p.y, arguments[2].p.z] == pytest.approx([4.656, 6., .779])
    assert arguments[3:6] == ("contact_wall", 0, 0)
    assert environment.wall_handles == [environment.gym.create_actor.return_value]


def native_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Represent an already wrapped native buffer in a CPU-only boundary test.

    Args:
        tensor: Test-owned raw simulator storage with its full global layout.

    Returns:
        The same storage, preserving the real native aliasing contract.
    """
    return tensor


def test_interleaved_actor_mapping_commits_full_raw_buffers_and_preserves_walls() -> None:
    """Reset logical env0/global actor2 without overwriting either wall or actor0.

    Raises:
        AssertionError: Logical advanced-index copies are mistaken for raw views,
            a setter receives only robot rows, or env_id substitutes for actor_id.
    """
    environment = V3ContactEvaluation.__new__(V3ContactEvaluation)
    environment.device, environment.sim = "cpu", object()
    environment.num_envs, environment.num_dof, environment.num_bodies = 2, 14, 3
    environment.robot_actor_indices = torch.tensor([2, 0])
    environment.robot_body_indices = torch.tensor([[5, 1, 7], [0, 6, 3]])
    environment.wall_body_indices = torch.tensor([2, 4])
    environment.gym = Mock()
    roots = torch.arange(52, dtype=torch.float32).view(4, 13)
    dofs = torch.zeros(28, 2)
    contacts = torch.arange(24, dtype=torch.float32).view(8, 3)
    rigid = torch.arange(104, dtype=torch.float32).view(8, 13)
    environment.gym.acquire_actor_root_state_tensor.return_value = roots
    environment.gym.acquire_dof_state_tensor.return_value = dofs
    environment.gym.acquire_net_contact_force_tensor.return_value = contacts
    environment.gym.acquire_rigid_body_state_tensor.return_value = rigid
    environment.envs, environment.actor_handles = [object()], [object()]
    environment.gym.get_actor_rigid_body_names.return_value = ["base_link", "gripper1", "gripper2"]
    environment.env_origins = torch.zeros(2, 3)
    environment.contact_manifest = {"scene": {"reference_tcp_m": [.63, 0., .78],
                                              "wall_center_relative_m": [.65, 0., .78],
                                              "wall_size_m": [.05, .4, .4]}}
    environment.num_bodies = 3
    with patch("isaacgym.gymtorch.wrap_tensor", side_effect=native_tensor):
        environment._bind_simulation_tensors()
        environment._bind_robot_rigid_state()
    torch.testing.assert_close(environment.root_states, roots[[2, 0]])
    torch.testing.assert_close(environment.rigid_state, rigid[environment.robot_body_indices])
    torch.testing.assert_close(environment.contact_forces, contacts[environment.robot_body_indices])
    old_roots = roots.clone()
    environment.root_states[0, 0] = -100.
    assert roots[2, 0] != -100.  # Advanced indexing deliberately returned a copy.
    with patch("isaacgym.gymtorch.unwrap_tensor", side_effect=torch.clone):
        environment._commit_robot_reset(torch.tensor([0]))
    args = environment.gym.set_actor_root_state_tensor_indexed.call_args.args
    assert args[1].shape == (4, 13) and args[2].tolist() == [2]
    assert args[3] == 1 and args[1][2, 0] == -100.
    torch.testing.assert_close(roots[[0, 1, 3]], old_roots[[0, 1, 3]])
    dof_args = environment.gym.set_dof_state_tensor_indexed.call_args.args
    assert dof_args[1].shape == (28, 2) and dof_args[2].tolist() == [2]


def test_contact_substeps_keep_eight_raw_samples_and_last_reaction_label() -> None:
    """Keep 1..8 N wall samples, their 4.5 N mean, and -8 N latest reaction.

    Raises:
        AssertionError: Samples collapse to the last point, change sign or become
            an artificial-force label, or collecting a sample advances physics.
    """
    environment = V3ContactEvaluation.__new__(V3ContactEvaluation)
    environment.device, environment.sim = "cpu", object()
    environment.num_envs = 2
    environment.gym = Mock()
    environment.raw_contact_forces = torch.zeros(8, 3)
    environment.wall_body_indices = torch.tensor([2, 4])
    environment.finger_body_indices = torch.tensor([[5, 1], [0, 6]])
    environment.wall_force_samples = torch.zeros(2, 8, 3)
    environment.finger_force_samples = torch.zeros(2, 8, 3)
    environment.contact_samples_written = torch.zeros(2, dtype=torch.long)
    environment.measurement_valid = torch.zeros(2, dtype=torch.bool)
    environment.last_contact_force_world = torch.zeros(2, 3)
    environment.robot_body_indices = torch.tensor([[5, 1, 7], [0, 6, 3]])
    environment.finger_local_indices = torch.tensor([0, 1])
    environment.raw_rigid_state = torch.zeros(8, 13)
    environment.raw_rigid_state[:, 6] = 1.
    environment.collision_local_corners = torch.zeros(3, 8, 3)
    environment.collision_bounds_present = torch.ones(3, dtype=torch.bool)
    environment.wall_lower_world = torch.tensor([[2., -1., -1.], [2., -1., -1.]])
    environment.wall_upper_world = torch.tensor([[3., 1., 1.], [3., 1., 1.]])
    environment.env_origins = torch.zeros(2, 3)
    environment.wall_proximity_samples = torch.zeros(2, 8, 3)
    environment.finger_ground_proximity_samples = torch.zeros(2, 8, dtype=torch.bool)
    environment.robot_contact_samples = torch.zeros(2, 8, 3, 3)
    for substep in range(8):
        environment.raw_contact_forces[[2, 4], 0] = substep + 1.
        environment.raw_contact_forces[[5, 0], 0] = -(substep + 1.)
        environment._after_physics_substep(substep)
    torch.testing.assert_close(environment.wall_force_samples[:, :, 0], torch.arange(1., 9.).repeat(2, 1))
    torch.testing.assert_close(environment.wall_force_samples.mean(dim=1)[:, 0], torch.full((2,), 4.5))
    torch.testing.assert_close(environment.last_contact_force_world[:, 0], torch.full((2,), -8.))
    assert environment.measurement_valid.all()
    environment.gym.simulate.assert_not_called()
    environment.gym.refresh_net_contact_force_tensor.assert_called()


def test_contact_world_script_keeps_target_fixed_and_rotates_command_once() -> None:
    """Keep an absolute target after base translation/yaw and release by 18 s.

    Raises:
        AssertionError: The goal follows the base, force is double-rotated, or
            the 0 N control uses a different geometric trajectory.
    """
    from test_v3_force_transitions import make_step_environment

    environment = make_step_environment()
    environment.__class__ = V3ContactEvaluation
    environment.contact_reference_world = torch.tensor([[.63, 0., .78], [.63, 0., .78]])
    environment.contact_start_world = torch.tensor([[.4, 0., .7], [.4, 0., .7]])
    environment.contact_target_world = environment.contact_start_world.clone()
    environment.contact_peak_n = torch.tensor([10., 20.])
    environment.episode_length_buf.fill_(400)  # 8 s plateau.
    environment.root_states[:, :2] = torch.tensor([.2, -.1])
    environment.rpy[:, 2] = torch.pi / 2
    environment._advance_task_commands(torch.arange(2))
    torch.testing.assert_close(environment._geometric_target_world(), environment.contact_reference_world)
    torch.testing.assert_close(V3PosForce._geometric_target_world(environment), environment.contact_reference_world,
                               atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(environment.commands[:, 9:12], torch.tensor([[0., -10., 0.], [0., -20., 0.]]),
                               atol=1e-5, rtol=1e-5)
    assert torch.count_nonzero(environment.pending_ee_force_world) == 0
    environment.episode_length_buf.fill_(900)
    environment._advance_task_commands(torch.arange(2))
    torch.testing.assert_close(environment.contact_target_world[:, 0], torch.tensor([.58, .58]))
    assert torch.count_nonzero(environment.commands[:, 9:12]) == 0


def test_contact_summary_keeps_failed_and_ambiguous_episodes_in_the_denominator() -> None:
    """Count one clean success, one missing failure and one polluted full episode.

    Raises:
        AssertionError: Filtering invalid samples inflates success or a zero force
            from a valid sensor is treated as missing measurement.
    """
    import json
    from pathlib import Path
    from legged_gym.b2_gym_learn.ppo_cse_pf.v3_evaluation import summarize_contact_evaluation

    manifest = json.loads(Path("configs/v3_b_contact_v1.json").read_text())
    manifest.update(num_envs=1, episodes_per_env=3)
    rows, episodes = [], []
    for episode_index in range(3):
        peak_n = 10. if episode_index % 2 == 0 else 20.
        for step in range(1000 if episode_index < 2 else 1):
            normal_n = peak_n if 450 <= step < 550 else 0.
            rows.append({"env_id": 0, "episode_index": episode_index,
                         "policy_input_time_s": step * .019999999552965164,
                         "contact_peak_n": peak_n, "contact_measurement_valid": episode_index < 2,
                         "contact_status": "ambiguous_contact" if episode_index == 1 else
                                           "clean_expected_contact" if normal_n else "no_contact",
                         "wall_force_mean_world_n": [normal_n, 0., 0.]})
        episodes.append({"env_id": 0, "episode_index": episode_index,
                         "end_type": "physical_failure" if episode_index == 2 else "timeout"})
    summary = summarize_contact_evaluation(rows, episodes, manifest, zero_control=False)
    assert summary["amplitudes"]["half"]["expected_episodes"] == 2
    assert summary["amplitudes"]["half"]["success_rate"] == .5
    assert summary["amplitudes"]["half"]["physical_failure_rate"] == .5
    assert summary["amplitudes"]["full"]["success_rate"] == 0.
    assert summary["episodes"][2]["missing_measurement_samples"] == 1
    assert summary["episodes"][0]["release_abs_normal_mean_n"] == 0.
    assert summary["assessment"] != "pass"


def test_contact_classification_requires_geometry_and_force_consistency() -> None:
    """Distinguish clean two-sided contact, nearby non-fingers and invalid data.

    Raises:
        AssertionError: A low residual alone certifies clean contact, an unmeasured
            interval becomes zero contact, or a reversed wall force passes.
    """
    from legged_gym.envs.v3.v3_contact_evaluation import classify_contact_interval

    wall = torch.tensor([[10., 0., 0.]] * 8)
    fingers = -wall
    assert classify_contact_interval(wall, fingers, False, True, False, True) == "clean_expected_contact"
    assert classify_contact_interval(wall, fingers, True, True, False, True) == "ambiguous_contact"
    assert classify_contact_interval(-wall, -fingers, False, True, False, True) == "ambiguous_contact"
    assert classify_contact_interval(wall, fingers, False, True, False, False) == "invalid"
    assert classify_contact_interval(wall * 0., fingers * 0., False, False, False, True) == "no_contact"

"""Check A08 reward contracts with physical tensors and no simulator steps."""

import unittest

import isaacgym  # noqa: F401 -- Gym must precede Torch.
import torch

from legged_gym.envs.v3.legged_robot_v3 import V3PosForce
from legged_gym.envs.v3.v3_pos_force_config import V3PosForceCfg
from legged_gym.utils.v3_kinematics import V3TcpKinematics
from pathlib import Path


def make_reward_environment() -> V3PosForce:
    """Build a tensor-only fixture at the confirmed reward computation seam.

    Returns:
        Two stationary robots with zero TCP/velocity tracking errors, in SI units.
    """
    environment = V3PosForce.__new__(V3PosForce)
    environment.cfg = V3PosForceCfg()
    environment.device = "cpu"
    environment.num_envs = 2
    environment.dt = 0.02
    environment.rew_buf = torch.zeros(2)
    environment.action_dof_indices = torch.arange(14)
    environment.position_dof_indices = torch.tensor([0, 1, 2, 3, 8, 9, 10, 11, 12, 13])
    environment.dof_pos = torch.zeros(2, 14)
    environment.dof_vel = torch.zeros(2, 14)
    environment.velocity_before_step = torch.zeros(2, 14)
    environment.dof_pos_limits = torch.tensor([[-2.0, 2.0]] * 14)
    environment.soft_position_limits = torch.tensor([[-1.9, 1.9]] * 10)
    environment.torque_limits = torch.full((14,), 40.0)
    environment.torques = torch.zeros(2, 14)
    environment.actions = torch.zeros(2, 14)
    environment.last_actions = torch.zeros(2, 14)
    environment.base_lin_vel = torch.zeros(2, 3)
    environment.base_ang_vel = torch.zeros(2, 3)
    environment.env_origins = torch.zeros(2, 3)
    environment.root_states = torch.zeros(2, 13)
    environment.root_states[:, 2] = 0.32
    environment.root_states[:, 6] = 1.0
    environment.rpy = torch.zeros(2, 3)
    environment.tcp_kinematics = V3TcpKinematics(
        Path(environment.cfg.asset.file),
        [f'{corner}_{kind}_joint' for kind in ('steering', 'wheel_drive')
         for corner in ('fl', 'fr', 'bl', 'br')] + [f'arm_joint{index}' for index in range(1, 7)],
        'cpu')
    environment.commands = torch.zeros(2, 15)
    # Keep perfect tracking using the real FK seam rather than an injected body cache.
    environment.commands[:, 3:6] = environment._tcp_spherical_position()
    environment.external_ee_force_y = torch.zeros(2, 3)
    environment.external_base_force_y = torch.zeros(2, 3)
    environment.gripper_force_kps = torch.full((2, 3), 200.0)
    environment.contact_forces = torch.zeros(2, 23, 3)
    environment.collision_body_indices = torch.arange(19)
    environment._prepare_reward_function()
    return environment


class V3RewardTests(unittest.TestCase):
    """Verify agreed physical formulas, signed sums, and single time scaling."""

    def test_b_compensation_changes_only_privileged_target_and_separates_error_metrics(self) -> None:
        """Check +20 N gives +0.1 m in world X, and opposite commands cancel it.

        Raises:
            AssertionError: The privileged target stays geometric, physical errors
                share the wrong samples, or artificial truth leaks into the Actor.
        """
        environment = make_reward_environment()
        environment.cfg.run.reproduction_part = "B"
        environment.default_dof_pos = torch.zeros(1, 14)
        environment.last_actions = torch.zeros(2, 14)
        environment.mass_com_delta = torch.zeros(2, 5)
        environment.friction = torch.ones(2, 1)
        environment.execution_scale_delta = torch.zeros(2, 14)
        environment.wheel_body_indices = torch.arange(19, 23)
        environment.contact_valid = torch.ones(2, dtype=torch.bool)
        ids = torch.arange(2)
        actor_before, critic_before, _ = environment._build_frames(ids)
        environment.external_ee_force_y[:, 0] = 20.
        environment.commands[1, 9] = -20.
        actor, critic, labels = environment._build_frames(ids)
        # At yaw zero the independent worked displacement is exactly +0.1 m.
        target_world = environment._geometric_target_world()
        expected_world = target_world.clone()
        expected_world[0, 0] += .1
        torch.testing.assert_close(environment._compensated_target_world(), expected_world)
        self.assertFalse(torch.allclose(critic[0, 95:98], critic_before[0, 95:98]))
        torch.testing.assert_close(critic[1, 95:98], critic_before[1, 95:98])
        torch.testing.assert_close(actor[0], actor_before[0])
        torch.testing.assert_close(labels[:, 6:9], torch.tensor([[.2, 0., 0.]] * 2))
        # Decode the privileged spherical point independently of the task helper.
        spherical = critic[:, 95:98] / torch.tensor([.5, 1., 1.3])
        radius, elevation, azimuth = spherical.unbind(dim=1)
        recovered_world = torch.stack((.174 + radius * elevation.cos() * azimuth.cos(),
                                       radius * elevation.cos() * azimuth.sin(),
                                       .32 + .09535 + radius * elevation.sin()), dim=1)
        torch.testing.assert_close(recovered_world, expected_world)
        environment.max_episode_length = 1000
        environment.completed_iterations = 8000
        environment.training_policy_steps = 192000
        environment.training_transitions = 384000
        environment._initialize_episode_metrics()
        environment.compute_reward()
        rows = environment._accumulate_episode_metrics(ids, torch.zeros(2, dtype=torch.bool))
        self.assertAlmostEqual(rows[0]["errors"]["tcp_geometric_l2_m"]["mean"], 0., places=5)
        self.assertAlmostEqual(rows[0]["errors"]["tcp_compensated_l2_m"]["mean"], .1, places=5)
        self.assertAlmostEqual(rows[1]["errors"]["tcp_compensated_l2_m"]["mean"], 0., places=5)

    def test_load_verifies_creation_properties_without_runtime_setters(self) -> None:
        """Restore mutable state without rewriting already prepared actor properties.

        Raises:
            AssertionError: Loading needs native setters or accepts different physical values.
        """
        environment = make_reward_environment()
        environment.completed_iterations = 3
        environment.training_policy_steps = 72
        environment.training_transitions = 144
        environment.mass_com_delta = torch.zeros(2, 5)
        environment.friction = torch.full((2, 1), 0.75)
        environment.execution_scale_delta = torch.zeros(2, 14)
        environment.last_push_policy_step = 0
        environment.last_push_velocity_world = torch.zeros(2, 2)
        snapshot = environment.get_training_state()
        environment.execution_scale_delta.fill_(0.1)
        environment.load_training_state(snapshot)
        torch.testing.assert_close(environment.execution_scale_delta, torch.zeros(2, 14))
        torch.testing.assert_close(environment.friction, torch.full((2, 1), 0.75))
        snapshot["mass_com_delta"][0, 0] = 1.0
        with self.assertRaisesRegex(ValueError, "creation"):
            environment.load_training_state(snapshot)

    def test_perfect_tracking_and_signed_collision_reward(self) -> None:
        """Check the worked A08 examples: perfect 0.13 and two-body -0.07.

        Returns:
            None.
        """
        environment = make_reward_environment()
        self.assertEqual(len(environment.reward_names), 23)
        environment.compute_reward()
        torch.testing.assert_close(environment.rew_buf, torch.full((2,), 0.13))
        environment.contact_forces[0, :2, 0] = 1.0
        environment.compute_reward()
        torch.testing.assert_close(environment.rew_buf, torch.tensor([-0.07, 0.13]))

    def test_episode_statistics_use_real_samples_and_clear_only_done_rows(self) -> None:
        """Check exact two-sample P95, actual duration, and local-reset isolation.

        Returns:
            None.
        """
        environment = make_reward_environment()
        environment.max_episode_length = 1000
        environment.completed_iterations = 1
        environment.training_policy_steps = 25
        environment.training_transitions = 50
        environment._initialize_episode_metrics()
        environment.compute_reward()
        environment._accumulate_episode_metrics(torch.empty(0, dtype=torch.long), torch.zeros(2, dtype=torch.bool))
        environment.commands[0, 3] += 0.1
        environment.compute_reward()
        records = environment._accumulate_episode_metrics(torch.tensor([0]), torch.zeros(2, dtype=torch.bool))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["steps"], 2)
        self.assertAlmostEqual(records[0]["duration_s"], 0.04)
        self.assertAlmostEqual(records[0]["errors"]["tcp_geometric_l2_m"]["mean"], 0.05, places=6)
        self.assertAlmostEqual(records[0]["errors"]["tcp_geometric_l2_m"]["p95"], 0.095, places=6)
        self.assertAlmostEqual(records[0]["reward_rate"], records[0]["reward_return"] / 0.04)
        self.assertIsNone(records[0]["timeout_bootstrap_bonus"])
        environment._clear_episode_metrics(torch.tensor([0]))
        self.assertEqual(environment.episode_metric_steps.tolist(), [0, 2])

    def test_grouped_acceleration_and_hard_soft_limits_remain_distinct(self) -> None:
        """Check a 20 ms acceleration, legal soft-limit excess, and wheel-speed exemption.

        Returns:
            None.
        """
        environment = make_reward_environment()
        environment.dof_vel[:, 4:8] = 10.0
        environment.velocity_before_step[:, 4:8] = 10.0
        environment.compute_reward()
        torch.testing.assert_close(environment.rew_buf, torch.full((2,), 0.13))
        environment.dof_vel[0, 4] += 0.2
        environment.dof_pos[1, 0] = 1.95
        environment.compute_reward()
        # 10 rad/s² on one wheel costs 2.5e-7 * .02 * 100 = .0000005.
        torch.testing.assert_close(environment.rew_buf, torch.tensor([0.1299995, 0.12]))
        self.assertEqual(environment.dof_pos_limits[0, 1], 2.0)
        environment._prepare_reward_function()
        environment.compute_reward()
        torch.testing.assert_close(environment.rew_buf, torch.tensor([0.1299995, 0.12]))

    def test_execution_scale_changes_targets_before_hard_clipping(self) -> None:
        """Check the same coefficient scales angular residuals and wheel-speed targets.

        Returns:
            None.
        """
        environment = make_reward_environment()
        environment.default_dof_pos = torch.zeros(1, 14)
        environment.wheel_dof_indices = torch.arange(4, 8)
        environment.p_gains = torch.ones(14)
        environment.d_gains = torch.ones(14)
        environment.execution_scale_delta = torch.full((2, 14), 0.1)
        torque = environment._compute_v3_torques(torch.ones(2, 14))
        torch.testing.assert_close(torque[:, :4], torch.full((2, 4), 0.275))
        torch.testing.assert_close(torque[:, 4:8], torch.full((2, 4), 5.5))
        torch.testing.assert_close(torque[:, 8:], torch.full((2, 6), 0.275))
        saturated = environment._compute_v3_torques(torch.full((2, 14), 100.0))
        torch.testing.assert_close(saturated[:, :4], torch.full((2, 4), 2.0))
        torch.testing.assert_close(saturated[:, 4:8], torch.full((2, 4), 20.0))


if __name__ == "__main__":
    unittest.main()

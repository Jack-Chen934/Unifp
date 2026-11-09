"""Check the calibrated V3 wheel velocity feedback configuration."""

import unittest

from legged_gym.envs.v3.v3_pos_force_config import V3PosForceCfg


class V3ControlConfigTests(unittest.TestCase):
    """Verify the confirmed drive wheel feedback coefficient."""

    def test_drive_feedback_matches_stable_timing_candidate(self) -> None:
        """Use the physically validated drive feedback with the refined control step.

        Raises:
            AssertionError: A wheel is missing or its damping differs from the validated
                7.73109 Nm*s/rad simulation coefficient.
        """
        damping = V3PosForceCfg.control.damping
        wheel_joints = (
            "fl_wheel_drive_joint",
            "fr_wheel_drive_joint",
            "bl_wheel_drive_joint",
            "br_wheel_drive_joint",
        )

        for joint_name in wheel_joints:
            self.assertEqual(damping[joint_name], 7.73109)

    def test_inner_loop_preserves_policy_and_episode_time(self) -> None:
        """Refine native integration without changing policy or episode duration.

        Raises:
            AssertionError: Native timing, policy interval, or episode time differs
                from 2.5 ms, eight substeps, 20 ms, or 20 s respectively.
        """
        self.assertEqual(V3PosForceCfg.sim.dt, 0.0025)
        self.assertEqual(V3PosForceCfg.control.decimation, 8)
        self.assertAlmostEqual(V3PosForceCfg.sim.dt * V3PosForceCfg.control.decimation, 0.02)
        self.assertEqual(V3PosForceCfg.env.episode_length_s, 20)

    def test_steering_stiffness_matches_loaded_motion_candidate(self) -> None:
        """Keep the validated steering feedback and retain its original damping.

        Raises:
            AssertionError: A steering joint has an unvalidated stiffness or damping.
        """
        for corner in ("fl", "fr", "bl", "br"):
            joint_name = f"{corner}_steering_joint"
            self.assertEqual(V3PosForceCfg.control.stiffness[joint_name], 49.8436)
            self.assertEqual(V3PosForceCfg.control.damping[joint_name], 0.664582)

    def test_training_uses_the_independent_coplanar_asset(self) -> None:
        """Select the validated geometry candidate for subsequent runtime checks.

        Raises:
            AssertionError: The default asset points to an older collision model.
        """
        self.assertIn("/v3_a1_desktop_train_v6_coplanar_r2/", V3PosForceCfg.asset.file)


if __name__ == "__main__":
    unittest.main()

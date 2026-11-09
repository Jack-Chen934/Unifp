"""以可手算的PD／轮速／裁剪工况核验独立分解器。"""

import unittest

import torch

from torque_requests import decompose_requests


class TorqueRequestTests(unittest.TestCase):
    """Check controller stages against independently calculated scalar results."""

    def test_position_velocity_and_limits(self) -> None:
        """Check execution scaling, hard target bounds, PD and final clipping.

        Note:
            Uses two candidates at the same two states; mutating any input fails.
        """
        actions = torch.zeros(2, 2, 14, dtype=torch.float64)
        actions[0, :, 9] = 2.0
        actions[1, :, 9] = -2.0
        actions[:, :, 4] = 1000.0
        position = torch.full((2, 14), 0.1, dtype=torch.float64)
        velocity = torch.full_like(position, 0.2)
        limits = torch.tensor([[-0.5, 0.5]] * 14, dtype=torch.float64)
        originals = [value.clone() for value in (actions, position, velocity, limits)]
        result = decompose_requests(actions, position, velocity, torch.zeros_like(position),
            limits, torch.full_like(position, 0.2), torch.full((14,), 300.),
            torch.full((14,), 2.), torch.full((14,), 40.), 100., 0.25, 5., 12.)
        # Joint2: raw targets +/-0.6, bounded +/-0.5; q=0.1, qdot=0.2.
        torch.testing.assert_close(result['raw_target'][:, 0, 9], torch.tensor([0.6, -0.6], dtype=torch.float64))
        torch.testing.assert_close(result['raw_torque'][:, 0, 9], torch.tensor([119.6, -180.4], dtype=torch.float64))
        torch.testing.assert_close(result['torque'][:, 0, 9], torch.tensor([40., -40.], dtype=torch.float64))
        self.assertTrue(result['saturated'][:, :, 9].all())
        self.assertEqual(float(result['clipped_actions'][0, 0, 4]), 100.)
        self.assertEqual(float(result['target'][0, 0, 4]), 12.)
        self.assertAlmostEqual(float(result['torque'][0, 0, 4]), 23.6)
        for value, original in zip((actions, position, velocity, limits), originals):
            torch.testing.assert_close(value, original, rtol=0., atol=0.)

    def test_zero_request_at_nominal_rest(self) -> None:
        """Verify zero residual and zero velocity generate zero torque at rest."""
        state = torch.zeros(2, 14)
        result = decompose_requests(torch.zeros(3, 2, 14), state, state, state,
            torch.tensor([[-1., 1.]] * 14), state, torch.ones(14), torch.ones(14),
            torch.full((14,), 40.), 100., 0.25, 5., 12.)
        self.assertEqual(int(torch.count_nonzero(result['torque'])), 0)
        self.assertFalse(bool(result['saturated'].any()))


if __name__ == '__main__':
    unittest.main()

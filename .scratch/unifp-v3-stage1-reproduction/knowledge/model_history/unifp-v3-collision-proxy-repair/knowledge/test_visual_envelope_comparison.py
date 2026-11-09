"""Check counterforce attribution against actual saved single/multiple contacts."""

import json
import unittest

from summarize_visual_envelope_comparison import OUTPUT_ROOT, identify_counterparts


class CounterforceAttributionTests(unittest.TestCase):
    """Use saved native forces so wheel-ground forces remain in the evidence."""

    def test_single_and_dual_gripper_counterforces(self) -> None:
        """Identify the two grippers jointly and two isolated camera contacts.

        Returns:
            None. Literal body sets come from the independently inspected forces.
        """
        events = json.loads((OUTPUT_ROOT / 'v8_seed1001_20s/failures.json').read_text())
        expected = {296: {'gripper1', 'gripper2'}, 479: {'wrist_camera'}, 838: {'wrist_camera'}}
        for event in events:
            bodies, residual_n = identify_counterparts(event)
            self.assertEqual(set(bodies), expected[event['policy_step']])
            self.assertLess(residual_n, 1e-4)


if __name__ == '__main__':
    unittest.main()

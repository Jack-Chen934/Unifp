"""Check the V8 candidate on saved native poses and real visual contact controls."""

import unittest

from audit_visual_envelope_regression import audit_geometry


class VisualEnvelopeRegressionTests(unittest.TestCase):
    """Exercise actual meshes without importing or modifying the simulator."""

    @classmethod
    def setUpClass(cls) -> None:
        """Audit the frozen saved poses and three controlled local contacts.

        Note:
            Reads existing assets and pose evidence; keeps the report in memory.
        """
        cls.report = audit_geometry()

    def test_saved_false_contacts_and_substeps(self) -> None:
        """Remove the specified false intersections without regressing separation.

        Returns:
            None. All 14 saved poses and 80 native substeps must be measured.
        """
        self.assertEqual(len(self.report['saved_poses']), 14)
        self.assertEqual(len(self.report['substeps']), 80)
        self.assertTrue(self.report['checks']['target_false_intersections_removed'])
        self.assertTrue(self.report['checks']['step454_remains_separate'])
        self.assertTrue(self.report['checks']['no_new_intersections_on_separate_poses'])
        self.assertTrue(self.report['checks']['historical_baseline_reproduced'])

    def test_real_visual_contacts_remain_detectable(self) -> None:
        """Keep actual surface collision detection at all three repaired regions.

        Returns:
            None. Each local visual must intersect before judging collision output.
        """
        controls = self.report['positive_controls']
        self.assertEqual({control['source_component'] for control in controls}, {8, 99, 43})
        for control in controls:
            with self.subTest(component=control['source_component']):
                self.assertTrue(control['visual']['surface_intersection'])
                self.assertTrue(control['collision']['surface_intersection'])


if __name__ == '__main__':
    unittest.main()

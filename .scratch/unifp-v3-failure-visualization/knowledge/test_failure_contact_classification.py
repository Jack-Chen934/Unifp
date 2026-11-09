"""Verify terminal contact categories against all saved native failure records."""

from collections import Counter
import json
from pathlib import Path
import unittest

import numpy as np

from visualize_failure_contacts import classify_terminal_event, triangle_intersection_anchor


class FailureContactClassificationTests(unittest.TestCase):
    """Keep real visual contact distinct from proxy-only and unresolved timing."""

    def test_intersection_anchor_lies_on_both_triangles(self) -> None:
        """Locate the known shared segment without using FCL contact positions.

        Returns:
            None. The anchor must be inside both triangles and on their planes.
        """
        horizontal_m = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]])
        vertical_m = np.array([[.5, -1., -1.], [.5, 1., 1.], [.5, 1., -1.]])
        anchor_m = triangle_intersection_anchor(horizontal_m, vertical_m)
        np.testing.assert_allclose(anchor_m[[0, 2]], [.5, 0.], atol=1e-12)
        self.assertGreaterEqual(anchor_m[1], 0.)
        self.assertLessEqual(anchor_m[1], 1.)

    def test_saved_three_variant_contact_counts(self) -> None:
        """Retain all 71 events and known per-variant geometry categories.

        Returns:
            None. Literal expected counts come from the saved independent audit.
        """
        project_root = Path(__file__).resolve().parents[3]
        path = project_root / '.scratch/unifp-v3-stage1-reproduction/knowledge/model_history/unifp-v3-collision-proxy-repair/artifacts/visual_envelope_v8_20261106/comparison_summary.json'
        summary = json.loads(path.read_text())
        expected = {'v6': (0, 20, 5), 'r3': (1, 22, 4), 'v8': (2, 11, 6)}
        for variant, counts in expected.items():
            categories = Counter(classify_terminal_event(event) for run in summary['runs']
                                 if run['variant'] == variant for event in run['events'])
            self.assertEqual(tuple(categories[key] for key in
                             ('visual_surface_intersection', 'proxy_only_surface_intersection',
                              'unresolved_terminal_separation')), counts)


if __name__ == '__main__':
    unittest.main()

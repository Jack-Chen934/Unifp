"""Regressions for collision budgets, source preservation and local filters."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from build_training_collision_asset import (DEFAULT_SOURCE_ROOT, URDF_NAME,
                                          build_training_collision_asset, collision_points)


def noncollision_tree(element: ET.Element) -> tuple:
    """Canonicalize XML while excluding only collision subtrees and whitespace.

    Args:
        element: Source or candidate XML element.

    Returns:
        Recursive tuple of tag, attributes, text and preserved children.
    """
    return (element.tag, tuple(sorted(element.attrib.items())), (element.text or '').strip(),
            tuple(noncollision_tree(child) for child in element if child.tag != 'collision'))


class TrainingCollisionAssetTests(unittest.TestCase):
    """Exercise the real offline generation and production filter interface."""

    @classmethod
    def setUpClass(cls) -> None:
        """Build a temporary candidate shared by the geometry regressions.

        Returns:
            None.

        Note:
            Creates test-owned files removed by tearDownClass.
        """
        cls.temporary = tempfile.TemporaryDirectory(prefix='unifp_collision_test_')
        cls.source_path = DEFAULT_SOURCE_ROOT / 'urdf' / URDF_NAME
        cls.path = build_training_collision_asset(DEFAULT_SOURCE_ROOT, Path(cls.temporary.name) / 'candidate')
        cls.root = cls.path.parents[1]
        cls.robot = ET.parse(cls.path).getroot()
        cls.source = ET.parse(cls.source_path).getroot()
        cls.manifest = json.loads((cls.root / 'asset_manifest.json').read_text())

    @classmethod
    def tearDownClass(cls) -> None:
        """Remove only files created by this test fixture.

        Returns:
            None.
        """
        cls.temporary.cleanup()

    def test_preserved_visuals_joints_and_physical_properties(self) -> None:
        """Reject unexpected model, visual-byte and physical-contract changes.

        Returns:
            None.
        """
        self.assertEqual(noncollision_tree(self.source), noncollision_tree(self.robot))
        self.assertEqual(len(self.robot.findall('link')), 28)
        self.assertEqual(len([joint for joint in self.robot.findall('joint')
                              if joint.get('type') != 'fixed']), 14)
        for mesh in self.source.findall('link/visual/geometry/mesh'):
            reference = mesh.get('filename')
            self.assertEqual(hashlib.sha256((self.source_path.parent / reference).read_bytes()).digest(),
                             hashlib.sha256((self.path.parent / reference).read_bytes()).digest())

    def test_shape_budget_hulls_and_semantic_indices(self) -> None:
        """Reject dense proxies, missing sources and overlapping filter groups.

        Returns:
            None.
        """
        count = len(self.robot.findall('link/collision'))
        self.assertGreaterEqual(count, 40)
        self.assertLessEqual(count, 80)
        self.assertEqual(count, self.manifest['collision_shape_count'])
        source_manifest = json.loads((DEFAULT_SOURCE_ROOT / 'asset_manifest.json').read_text())
        for link in self.robot.findall('link'):
            name = link.get('name')
            entry = self.manifest['collision_solid_counts'][name]
            self.assertEqual(entry['count'], len(link.findall('collision')))
            group_indices = [index for indices in entry['source_groups'].values() for index in indices]
            self.assertEqual(sorted(group_indices), list(range(entry['count'])))
        for record in self.manifest['proxy_records']:
            name = record['link']
            self.assertLessEqual(record['vertices'], 64)
            tolerance_m = 0.001 if name.startswith('gripper') or name.endswith('_Link') else 0.003
            self.assertLessEqual(record['support_error_m'], tolerance_m)
            source_count = source_manifest['collision_solid_counts'][name]['count']
            self.assertTrue(all(0 <= index < source_count for index in record['source_components']))
        # Every original component belongs to exactly one proxy; this detects
        # omitted frame rails and duplicated CAD mappings after manual grouping.
        records = self.manifest['proxy_records'] + self.manifest['retained_collision_records']
        for record in self.manifest['retained_collision_records']:
            name = record['link']
            original = self.source.find(f"link[@name='{name}']").findall('collision')[record['source_components'][0]]
            retained = self.robot.find(f"link[@name='{name}']").findall('collision')[record['local_shape']]
            self.assertEqual(ET.tostring(original).strip(), ET.tostring(retained).strip())
            reference = original.find('geometry/mesh').get('filename')
            self.assertEqual((self.source_path.parent / reference).read_bytes(),
                             (self.path.parent / reference).read_bytes())
        for name in {record['link'] for record in records}:
            indices = [index for record in records if record['link'] == name
                       for index in record['source_components']]
            self.assertEqual(sorted(indices), list(range(source_manifest['collision_solid_counts'][name]['count'])), name)

    def test_wheel_collision_envelope_and_bytes_unchanged(self) -> None:
        """Preserve tire bytes and wheel extrema while simplifying recessed hubs.

        Returns:
            None.
        """
        for corner in ('fl', 'fr', 'bl', 'br'):
            source_wheel = self.source.find(f"link[@name='{corner}_Link']")
            candidate_wheel = self.robot.find(f"link[@name='{corner}_Link']")
            source_tire = source_wheel.findall('collision')[4]
            candidate_tire = candidate_wheel.findall('collision')[1]
            self.assertEqual(ET.tostring(source_tire).strip(), ET.tostring(candidate_tire).strip())
            reference = source_tire.find('geometry/mesh').get('filename')
            self.assertEqual((self.source_path.parent / reference).read_bytes(),
                             (self.path.parent / reference).read_bytes())
            source_points = np.concatenate([collision_points(self.source_path, collision)
                                            for collision in source_wheel.findall('collision')])
            candidate_points = np.concatenate([collision_points(self.path, collision)
                                               for collision in candidate_wheel.findall('collision')])
            np.testing.assert_allclose(source_points.min(axis=0), candidate_points.min(axis=0), atol=0.0005)
            np.testing.assert_allclose(source_points.max(axis=0), candidate_points.max(axis=0), atol=0.0005)
            hub_points = collision_points(self.path, candidate_wheel.findall('collision')[0])
            # Wheel-local y is the axle. Hubs remain at least 35 mm inside the
            # tire radius, so they cannot replace its rolling ground surface.
            self.assertLess(np.linalg.norm(hub_points[:, [0, 2]], axis=1).max(), 0.066)

    def test_frame_openings_remain_empty(self) -> None:
        """Catch the failed single-box frame proxy that occupied moving-arm space.

        Returns:
            None.
        """
        frame = self.robot.find("link[@name='aluminum_profiles']")
        # Link-local metres: above the low mounting platform, ahead of the
        # vertical rear rails, including the arm's default working space.
        free_points = np.array([[0.17, 0, 0.55], [0.0, 0, 0.9], [-0.1, 0, 1.1]])
        for collision in frame.findall('collision'):
            hull = ConvexHull(collision_points(self.path, collision))
            signed_planes = free_points @ hull.equations[:, :3].T + hull.equations[:, 3]
            self.assertTrue(np.all(signed_planes.max(axis=1) > 0.005))

    def test_left_rear_wheel_arch_clearance(self) -> None:
        """Catch the convex fill that caused repeated wheel/chassis contacts.

        Returns:
            None.
        """
        # Base-local metres at the first failing left-rear wheel pose. The
        # approved CAD has >1 mm clearance here; the failed merged hull fills it.
        clearance_points = np.array([[-0.24846, 0.19281, -0.1215],
                                     [0.223535, -0.209643, -0.119768]])
        base = self.robot.find("link[@name='base_link']")
        indices = self.manifest['collision_solid_counts']['base_link']['source_groups']['base_link_collision']
        for index in indices:
            hull = ConvexHull(collision_points(self.path, base.findall('collision')[index]))
            distances = (clearance_points @ hull.equations[:, :3].T + hull.equations[:, 3]).max(axis=1)
            self.assertTrue(np.all(distances > 0.001), distances)

    def test_existing_output_cannot_be_overwritten(self) -> None:
        """Protect both candidate identities and the immutable approved asset.

        Returns:
            None.
        """
        with self.assertRaises(FileExistsError):
            build_training_collision_asset(DEFAULT_SOURCE_ROOT, self.root)
        with self.assertRaises(FileExistsError):
            build_training_collision_asset(DEFAULT_SOURCE_ROOT, DEFAULT_SOURCE_ROOT)


if __name__ == '__main__':
    unittest.main()

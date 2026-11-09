"""Check approved visual-envelope geometry through its public interfaces."""

import unittest
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

import build_visual_envelope_candidate as builder
from build_visual_envelope_candidate import clip_visual_triangles
from test_local_base_proxy_candidate import normalized_element

PROJECT_ROOT = Path(__file__).resolve().parents[6]
SOURCE_ROOT = PROJECT_ROOT / 'UniFP_v3/resources/robots/v3_a1_desktop_train_v7_local_base_proxy_r3'
URDF_NAME = 'a1arm_rgm3_desktop_fixed_gripper.urdf'
TARGET_NAMES = {'base_link_local_1_group_0', 'base_link_local_1_group_5', 'base_link_proxy_13'}
DESKTOP_ROOT = PROJECT_ROOT / 'UniFP_v3/resources/robots/v3_a1_desktop'


class VisualEnvelopeGeometryTests(unittest.TestCase):
    """Use analytic shapes to check clipping and conservative geometry."""

    def test_clipping_retains_crossing_triangle(self) -> None:
        """Keep the unit square crossed by a triangle whose vertices lie outside.

        Returns:
            None. The expected intersection is the four literal square corners.
        """
        triangle = trimesh.Trimesh(vertices=[[-2, -2, 0], [2, -2, 0], [0, 2, 0]],
                                   faces=[[0, 1, 2]], process=False)
        region_planes = np.array([[1, 0, 0, -0.5], [-1, 0, 0, -0.5],
                                  [0, 1, 0, -0.5], [0, -1, 0, -0.5],
                                  [0, 0, 1, -0.1], [0, 0, -1, -0.1]])
        clipped_points_m = clip_visual_triangles(triangle, region_planes)
        expected_corners_m = {(-0.5, -0.5, 0.0), (-0.5, 0.5, 0.0),
                              (0.5, -0.5, 0.0), (0.5, 0.5, 0.0)}
        self.assertEqual({tuple(point) for point in clipped_points_m}, expected_corners_m)

    def test_hull_covers_reference_with_half_millimeter_margin(self) -> None:
        """Cover a 20 mm cube with the known 21 mm axis-aligned envelope.

        Returns:
            None. Bounds are literal meter values independent of hull construction.
        """
        reference_points_m = trimesh.creation.box(extents=[0.02, 0.02, 0.02]).vertices
        hull, _ = builder.rebuild_local_hull(reference_points_m, 0.0005)
        np.testing.assert_allclose(hull.bounds, [[-0.0105] * 3, [0.0105] * 3], atol=1e-8)
        self.assertTrue(hull.contains(reference_points_m).all())
        self.assertLessEqual(len(hull.vertices), 64)

    def test_detailed_reference_is_covered_within_vertex_budget(self) -> None:
        """Keep all vertices of a detailed sphere while using at most 64 vertices.

        Returns:
            None. Independent ray containment verifies coverage of reference points.
        """
        reference_points_m = trimesh.creation.icosphere(subdivisions=2, radius=0.01).vertices
        hull, measurements = builder.rebuild_local_hull(reference_points_m, 0.0005)
        self.assertTrue(hull.contains(reference_points_m).all())
        self.assertLessEqual(len(hull.vertices), 64)
        self.assertLessEqual(measurements['max_reference_outside_m'], 1e-5)


class VisualEnvelopeAssetTests(unittest.TestCase):
    """Validate one real generated asset through its files and public manifest."""

    @classmethod
    def setUpClass(cls) -> None:
        """Generate a temporary candidate shared by the real asset contracts.

        Note:
            Writes only a new temporary directory and removes it after this class.
        """
        cls.temporary_directory = tempfile.TemporaryDirectory(prefix='unifp-v8-contract-')
        cls.addClassCleanup(cls.temporary_directory.cleanup)
        cls.candidate_root = Path(cls.temporary_directory.name) / 'candidate'
        cls.candidate_urdf = builder.build_candidate(cls.candidate_root)
        cls.robot = ET.parse(cls.candidate_urdf).getroot()
        cls.manifest = json.loads((cls.candidate_root / 'asset_manifest.json').read_text())

    def test_non_target_and_physical_contract_preserved(self) -> None:
        """Preserve every non-target geometry, visual, joint and inertial property.

        Returns:
            None. The independent r3 source supplies the expected XML and meshes.
        """
        source_robot = ET.parse(SOURCE_ROOT / 'urdf' / URDF_NAME).getroot()
        roots = (deepcopy(source_robot), deepcopy(self.robot))
        for root in roots:
            base = root.find("link[@name='base_link']")
            for collision in list(base.findall('collision')):
                if collision.get('name') in TARGET_NAMES:
                    base.remove(collision)
        self.assertEqual(normalized_element(roots[0], SOURCE_ROOT),
                         normalized_element(roots[1], self.candidate_root))
        self.assertEqual(sum(len(link.findall('collision')) for link in self.robot.findall('link')), 168)
        self.assertEqual(len(self.robot.find("link[@name='base_link']").findall('collision')), 60)

    def test_serialized_mesh_covers_independent_visual_intersection(self) -> None:
        """Cover all reference surfaces clipped independently by Trimesh.

        Returns:
            None. Final GLB planes must cover every independent clipped vertex and
            every vertex of the five retained proxy13 source components, within 10 um.
        """
        desktop_urdf = DESKTOP_ROOT / 'urdf' / URDF_NAME
        base = ET.parse(desktop_urdf).getroot().find("link[@name='base_link']")
        visual = base.find('visual')
        reference = visual.find('geometry/mesh')
        visual_mesh = trimesh.load((desktop_urdf.parent / reference.get('filename')).resolve(),
                                   force='scene').to_geometry()
        # Derive the visual transform independently from the builder's loader.
        origin = visual.find('origin')
        rotation = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy'), sep=' ')).as_matrix()
        scale = np.fromstring(reference.get('scale', '1 1 1'), sep=' ')
        translation_m = np.fromstring(origin.get('xyz'), sep=' ')
        visual_mesh.vertices = (visual_mesh.vertices * scale) @ rotation.T + translation_m
        for record in self.manifest['visual_envelope_refinements']:
            clipped_reference = visual_mesh.copy()
            for plane in np.asarray(record['region_planes_in_base_m']):
                clipped_reference = trimesh.intersections.slice_mesh_plane(
                    clipped_reference, -plane[:3], -plane[3] * plane[:3], cap=False)
            serialized_mesh = trimesh.load(self.candidate_root / record['mesh'], force='scene').to_geometry()
            planes = ConvexHull(serialized_mesh.vertices).equations
            reference_points_m = np.asarray(clipped_reference.vertices)
            self.assertGreater(len(reference_points_m), 3)
            outside_m = reference_points_m @ planes[:, :3].T + planes[:, 3]
            self.assertLessEqual(float(outside_m.max()), 1e-5)
            for component in record['retained_source_components']:
                points_m = builder.collision_points(desktop_urdf, base.findall('collision')[component])
                self.assertLessEqual(float((points_m @ planes[:, :3].T + planes[:, 3]).max()), 1e-5)

    def test_candidate_identity_and_source_indices_are_traceable(self) -> None:
        """Verify real referenced bytes, direct provenance and preserved index rules.

        Returns:
            None. Source records and SHA256 bytes provide the expected identities.
        """
        source = json.loads((SOURCE_ROOT / 'asset_manifest.json').read_text())
        for key in ('collision_solid_counts', 'retained_collision_records', 'visual_clearance_records',
                    'rolling_collision_records', 'local_base_shape_map', 'local_base_proxy_replacements'):
            self.assertEqual(self.manifest[key], source[key])
        self.assertEqual(self.manifest['source_asset'], str(SOURCE_ROOT.relative_to(PROJECT_ROOT)))
        self.assertEqual(self.manifest['source_manifest_sha256'],
                         hashlib.sha256((SOURCE_ROOT / 'asset_manifest.json').read_bytes()).hexdigest())
        self.assertEqual(self.manifest['derived_sha256'], hashlib.sha256(self.candidate_urdf.read_bytes()).hexdigest())
        actual_targets = [(record['collision_name'], record['local_shape'], record['rebuilt_source_component'])
                           for record in self.manifest['visual_envelope_refinements']]
        self.assertEqual(actual_targets, [('base_link_local_1_group_0', 1, 8),
                                         ('base_link_local_1_group_5', 6, 99), ('base_link_proxy_13', 26, 43)])
        for record in self.manifest['mesh_files']:
            mesh_path = (self.candidate_root / record['path']).resolve()
            self.assertTrue(mesh_path.is_relative_to(self.candidate_root))
            self.assertEqual(hashlib.sha256(mesh_path.read_bytes()).hexdigest(), record['sha256'])
        with self.assertRaises(FileExistsError):
            builder.build_candidate(self.candidate_root)


if __name__ == '__main__':
    unittest.main()

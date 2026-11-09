"""Check source coverage, geometry and serialized contracts of the grouped asset."""

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

TASK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TASK_ROOT.parents[4]
SOURCE_ROOT = PROJECT_ROOT / 'UniFP_v3/resources/robots/v3_a1_desktop_train_v6_coplanar_r2'
URDF_NAME = 'a1arm_rgm3_desktop_fixed_gripper.urdf'
EXPECTED_GROUPS = ((8,), (10,), (23,), (25,), (57,), (99,), (92, 118),
                   (26, 72), (9, 94), (42, 65, 40, 93, 114), (18, 91))


def normalized_element(element: ET.Element, asset_root: Path) -> bytes:
    """Canonicalize XML and compare referenced mesh contents across asset roots.

    Args:
        element: URDF element whose content must remain unchanged.
        asset_root: Asset directory used to resolve relative URDF mesh paths.

    Returns:
        XML bytes without whitespace and with mesh URIs replaced by SHA256 values.

    Raises:
        OSError: A referenced mesh cannot be read.
    """
    canonical = deepcopy(element)
    for node in canonical.iter():
        node.text = node.tail = None
        if node.tag == 'mesh':
            path = (asset_root / 'urdf' / node.attrib['filename']).resolve()
            node.set('filename', hashlib.sha256(path.read_bytes()).hexdigest())
    return ET.tostring(canonical)


class GroupedCandidateTests(unittest.TestCase):
    """Validate one real generated asset without loading Gym or an optimizer."""

    @classmethod
    def setUpClass(cls) -> None:
        """Build one candidate in a fresh temporary directory for all contracts.

        Returns:
            None.

        Note:
            Generates temporary meshes; class cleanup removes only this directory.
        """
        cls.temporary_directory = tempfile.TemporaryDirectory(prefix='unifp-grouped-contract-')
        cls.addClassCleanup(cls.temporary_directory.cleanup)
        cls.candidate_root = Path(cls.temporary_directory.name) / 'candidate'
        specification = importlib.util.spec_from_file_location(
            'grouped_builder', TASK_ROOT / 'knowledge/build_local_base_proxy_candidate.py')
        builder = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(builder)
        builder.build_candidate(cls.candidate_root)
        cls.manifest = json.loads((cls.candidate_root / 'asset_manifest.json').read_text())
        cls.robot = ET.parse(cls.candidate_root / 'urdf' / URDF_NAME).getroot()
        cls.source_robot = ET.parse(SOURCE_ROOT / 'urdf' / URDF_NAME).getroot()

    def test_fixed_groups_cover_every_source_component_once(self) -> None:
        """Require the approved 11 groups and the complete 19-part source mapping.

        Returns:
            None.
        """
        replacements = self.manifest['local_base_proxy_replacements']
        self.assertEqual(len(replacements), 1)
        replacement = replacements[0]
        actual_groups = tuple(tuple(record['source_components']) for record in replacement['groups'])
        self.assertEqual(actual_groups, EXPECTED_GROUPS)
        covered = [component for group in actual_groups for component in group]
        self.assertEqual(len(covered), len(set(covered)))
        self.assertCountEqual(covered, replacement['old_components'])
        self.assertEqual(self.manifest['collision_shape_count'], 168)
        self.assertEqual(len(self.robot.find("link[@name='base_link']").findall('collision')), 60)

    def test_only_target_collision_changes(self) -> None:
        """Preserve visual, inertial, joint and every non-target collision element.

        Returns:
            None.
        """
        roots = (deepcopy(self.source_robot), deepcopy(self.robot))
        for root in roots:
            for link in root.findall('link'):
                for collision in link.findall('collision'):
                    link.remove(collision)
        self.assertEqual(normalized_element(roots[0], SOURCE_ROOT),
                         normalized_element(roots[1], self.candidate_root))
        candidate_links = {link.attrib['name']: link for link in self.robot.findall('link')}
        mapping = self.manifest['local_base_shape_map']
        for source_link in self.source_robot.findall('link'):
            name = source_link.attrib['name']
            collisions = candidate_links[name].findall('collision')
            for old_index, collision in enumerate(source_link.findall('collision')):
                if name == 'base_link' and old_index == 1:
                    continue
                new_index = mapping[str(old_index)][0] if name == 'base_link' else old_index
                self.assertEqual(normalized_element(collision, SOURCE_ROOT),
                                 normalized_element(collisions[new_index], self.candidate_root))

    def test_meshes_are_self_contained_and_hashes_match(self) -> None:
        """Require a movable asset with valid source identity and all mesh digests.

        Returns:
            None.
        """
        digest = hashlib.sha256((self.candidate_root / 'urdf' / URDF_NAME).read_bytes()).hexdigest()
        self.assertEqual(digest, self.manifest['derived_sha256'])
        source_manifest_path = SOURCE_ROOT / 'asset_manifest.json'
        self.assertEqual(hashlib.sha256(source_manifest_path.read_bytes()).hexdigest(),
                         self.manifest['source_manifest_sha256'])
        source_manifest = json.loads(source_manifest_path.read_text())
        for key in ('source_asset', 'source_urdf_sha256', 'source_manifest_sha256'):
            self.assertEqual(source_manifest[key], self.manifest['upstream_provenance'][key])
        for key in ('base_refinement_records', 'rolling_collision_records'):
            self.assertEqual(source_manifest[key], self.manifest[key])
        for reference in self.robot.findall('.//mesh'):
            mesh_path = (self.candidate_root / 'urdf' / reference.attrib['filename']).resolve()
            self.assertTrue(mesh_path.is_relative_to(self.candidate_root))
            self.assertTrue(mesh_path.is_file())
        for record in self.manifest['mesh_files']:
            digest = hashlib.sha256((self.candidate_root / record['path']).read_bytes()).hexdigest()
            self.assertEqual(digest, record['sha256'])

    def test_group_geometry_matches_approved_fixture(self) -> None:
        """Keep grouped convex support within two micrometers of the tested fixture.

        Returns:
            None.

        Note:
            This compares convex support, not concavity or physical contact safety.
        """
        directions = np.random.default_rng(1).normal(size=(256, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        for index, record in enumerate(self.manifest['local_base_proxy_replacements'][0]['groups']):
            mesh = trimesh.load(self.candidate_root / 'meshes' / record['mesh'], force='scene').to_geometry()
            fixture = trimesh.load(TASK_ROOT / f'artifacts/grouped11_diagnostic/meshes/group_{index}.glb',
                                   force='scene').to_geometry()
            self.assertLessEqual(len(mesh.vertices), 64)
            self.assertTrue(mesh.is_watertight and mesh.is_volume)
            difference_m = np.abs((mesh.vertices @ directions.T).max(axis=0)
                                  - (fixture.vertices @ directions.T).max(axis=0))
            self.assertLessEqual(float(difference_m.max()), 2e-6)

    def test_all_shape_index_rules_preserve_source_identity(self) -> None:
        """Keep mounting, retained and clearance rules attached to their original solids.

        Returns:
            None.
        """
        source = json.loads((SOURCE_ROOT / 'asset_manifest.json').read_text())
        mapping = self.manifest['local_base_shape_map']
        expected_groups = {name: [new for old in indices for new in mapping[str(old)]]
                           for name, indices in source['collision_solid_counts']['base_link']['source_groups'].items()}
        self.assertEqual(expected_groups, self.manifest['collision_solid_counts']['base_link']['source_groups'])
        for key in ('retained_collision_records', 'visual_clearance_records'):
            expected = deepcopy(source[key])
            for record in expected:
                if key == 'visual_clearance_records' or record['link'] == 'base_link':
                    record['local_shape'] = mapping[str(record['local_shape'])][0]
            self.assertEqual(expected, self.manifest[key])


if __name__ == '__main__':
    unittest.main()

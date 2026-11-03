"""Regression checks against the actual desktop assembly and generated URDF."""

import hashlib
import sys
from pathlib import Path
from typing import Dict
import unittest
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from scipy.spatial import cKDTree
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from build_desktop_asset import (CORNERS, DESKTOP_ARCHIVE, OUTPUT_ROOT,
                                 SOURCE_MEMBER, load_archive_mesh, origin_transform)


def link_poses(robot: ET.Element) -> Dict[str, np.ndarray]:
    """Compute zero-joint link poses relative to the physical base.

    Args:
        robot: Parsed source or candidate robot XML.

    Returns:
        Mapping from reachable link names to 4-by-4 base-relative poses, meters.
    """
    poses = {'base_link': np.eye(4)}
    pending = list(robot.findall('joint'))
    while pending:
        reachable = [joint for joint in pending if joint.find('parent').get('link') in poses]
        if not reachable:
            break  # Source-only virtual roots are deliberately above base_link.
        for joint in reachable:
            parent = joint.find('parent').get('link')
            poses[joint.find('child').get('link')] = poses[parent] @ origin_transform(joint.find('origin'))
            pending.remove(joint)
    return poses


class DesktopAssetTests(unittest.TestCase):
    """Validate shape, physical contracts and preserved source world geometry."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load immutable desktop input and the generated candidate.

        Returns:
            None.

        Raises:
            FileNotFoundError: Candidate has not yet been generated.

        Note:
            Caches parsed input and link poses for all regression checks.
        """
        with zipfile.ZipFile(DESKTOP_ARCHIVE) as archive:
            cls.source = ET.fromstring(archive.read(SOURCE_MEMBER))
        cls.candidate_path = OUTPUT_ROOT / 'urdf/a1arm_rgm3_desktop_fixed_gripper.urdf'
        cls.candidate = ET.parse(cls.candidate_path).getroot()
        cls.poses = link_poses(cls.candidate)

    def test_dofs_and_mass(self) -> None:
        """Check free root, fixed fingers, eight wheel axes and mass budget.

        Returns:
            None.
        """
        joints = [joint for joint in self.candidate.findall('joint') if joint.get('type') != 'fixed']
        expected = {c + suffix for c in CORNERS for suffix in ('_steering_joint', '_wheel_drive_joint')}
        expected.update('arm_joint' + str(i) for i in range(1, 7))
        self.assertEqual({joint.get('name') for joint in joints}, expected)
        self.assertEqual(len(joints), 14)
        self.assertFalse(any(joint.get('name').startswith('root_') for joint in self.candidate.findall('joint')))
        mass = sum(float(node.get('value')) for node in self.candidate.findall('link/inertial/mass'))
        self.assertAlmostEqual(mass, 83.3179153728508, places=7)

    def test_nonwheel_data_preserved(self) -> None:
        """Check source non-wheel visuals, installation and payload inertials.

        Returns:
            None.
        """
        for link in self.source.findall('link'):
            name = link.get('name')
            if name.startswith('root_link') or name in {c + '_Link' for c in CORNERS}:
                continue
            candidate = self.candidate.find(f"link[@name='{name}']")
            # Ignore formatting whitespace, compare actual XML attributes/children.
            for tag in ('visual', 'inertial'):
                if tag == 'inertial' and name == 'base_link':
                    continue
                source_elements = link.findall(tag)
                candidate_elements = candidate.findall(tag)[:len(source_elements)]
                self.assertEqual([[n.tag, n.attrib] for e in source_elements for n in e.iter()],
                                 [[n.tag, n.attrib] for e in candidate_elements for n in e.iter()], name)
        source_poses = link_poses(self.source)
        for name, pose in source_poses.items():
            if name not in {c + '_Link' for c in CORNERS}:
                np.testing.assert_allclose(self.poses[name], pose, atol=1e-10, rtol=0)

    def test_wheel_zero_geometry(self) -> None:
        """Compare complete source and split wheel assemblies in base space.

        Returns:
            None.
        """
        source_poses = link_poses(self.source)
        with zipfile.ZipFile(DESKTOP_ARCHIVE) as archive:
            for corner in CORNERS:
                visual = self.source.find(f"link[@name='{corner}_Link']/visual")
                original = load_archive_mesh(archive, visual)
                original.apply_transform(source_poses[corner + '_Link'] @ origin_transform(visual.find('origin')))
                parts = []
                faces = 0
                for link in self.candidate.findall('link'):
                    for candidate_visual in link.findall('visual'):
                        if not candidate_visual.get('name', '').startswith(corner + '_'):
                            continue
                        reference = candidate_visual.find('geometry/mesh').get('filename')
                        mesh = trimesh.load(self.candidate_path.parent / reference, force='scene').to_geometry()
                        mesh.apply_transform(self.poses[link.get('name')] @ origin_transform(candidate_visual.find('origin')))
                        parts.append(mesh.vertices)
                        faces += len(mesh.faces)
                candidate_vertices = np.concatenate(parts)
                self.assertEqual(faces, len(original.faces), corner)
                # Both directions detect lost source pieces and stray added geometry.
                forward_error = cKDTree(candidate_vertices).query(original.vertices)[0].max()
                backward_error = cKDTree(original.vertices).query(candidate_vertices)[0].max()
                self.assertLessEqual(max(forward_error, backward_error), 1e-5, corner)

    def test_mass_inertia_and_source_hash(self) -> None:
        """Check approved wheel inertia, finite physical tensors and source ZIP.

        Returns:
            None.
        """
        self.assertEqual(hashlib.sha256(DESKTOP_ARCHIVE.read_bytes()).hexdigest(),
                         '9638c2794cfa6874b57a8def527899b0352a29997e203b3a3b51dbc0a38346d8')
        for corner in CORNERS:
            inertial = self.candidate.find(f"link[@name='{corner}_Link']/inertial")
            np.testing.assert_allclose(np.fromstring(inertial.find('origin').get('xyz'), sep=' '), [0, 0, 0])
            self.assertAlmostEqual(float(inertial.find('inertia').get('iyy')), 0.011393)
            self.assertAlmostEqual(float(inertial.find('mass').get('value')), 2.2786)
        for inertial in self.candidate.findall('link/inertial'):
            values = inertial.find('inertia').attrib
            tensor = np.array([
                [float(values['ixx']), float(values['ixy']), float(values['ixz'])],
                [float(values['ixy']), float(values['iyy']), float(values['iyz'])],
                [float(values['ixz']), float(values['iyz']), float(values['izz'])]])
            eigenvalues = np.linalg.eigvalsh(tensor)
            self.assertTrue(np.isfinite(eigenvalues).all())
            self.assertGreater(eigenvalues.min(), 0)
            self.assertLessEqual(eigenvalues[-1], eigenvalues[:2].sum() + 1e-7)

    def test_collision_meshes_are_three_dimensional_hulls(self) -> None:
        """Reject mesh collisions that cannot define a volumetric convex hull.

        Returns:
            None.
        """
        collision_meshes = self.candidate.findall('link/collision/geometry/mesh')
        self.assertGreater(len(collision_meshes), 0)
        for mesh_element in collision_meshes:
            mesh_path = self.candidate_path.parent / mesh_element.get('filename')
            mesh = trimesh.load(mesh_path, force='scene').to_geometry()
            unique_vertices = np.unique(mesh.faces)
            points = mesh.vertices[unique_vertices]
            singular_values = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
            rank = int(np.count_nonzero(singular_values > singular_values[0] * 1e-8))
            self.assertGreaterEqual(len(unique_vertices), 4, mesh_path.name)
            self.assertEqual(rank, 3, mesh_path.name)


if __name__ == '__main__':
    unittest.main()

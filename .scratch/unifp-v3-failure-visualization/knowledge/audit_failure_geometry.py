"""Measure saved V3 terminal-frame surface distances using independent FCL meshes."""

import argparse
import json
from pathlib import Path
import sys
from typing import Any
import xml.etree.ElementTree as ET

import fcl
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / 'UniFP_v3/tools'))
from compare_desktop_fk import forward_kinematics, pose_from_origin


def load_surfaces(urdf_path: Path, link: ET.Element, kind: str) -> list[tuple[str, Any]]:
    """Build one FCL BVH per URDF surface in its link-local frame.

    Args:
        urdf_path: Source URDF whose relative mesh references are resolved.
        link: Link element containing the requested geometries.
        kind: Either visual or collision; all matching mesh entries are loaded.

    Returns:
        Named BVH models with meter-valued, scaled and origin-transformed vertices.

    Raises:
        OSError: A referenced mesh cannot be read.
        ValueError: A surface is not a supported triangular mesh.
    """
    surfaces = []
    for index, element in enumerate(link.findall(kind)):
        reference = element.find('geometry/mesh')
        mesh = trimesh.load((urdf_path.parent / reference.get('filename')).resolve(),
                            force='scene').to_geometry()
        mesh.apply_scale(np.fromstring(reference.get('scale', '1 1 1'), sep=' '))
        if element.find('origin') is not None:
            mesh.apply_transform(pose_from_origin(element.find('origin')))
        model = fcl.BVHModel()
        model.beginModel(len(mesh.faces), len(mesh.vertices))
        model.addSubModel(np.asarray(mesh.vertices), np.asarray(mesh.faces, dtype=np.int32))
        model.endModel()
        surfaces.append((element.get('name', f'{link.get("name")}_{kind}_{index}'), model))
    return surfaces


def measure_pair(base_surfaces: list[tuple[str, Any]], other_surfaces: list[tuple[str, Any]],
                 other_pose: np.ndarray) -> dict[str, Any]:
    """Measure all surface pairs after placing the other body in the base frame.

    Args:
        base_surfaces: Named local surfaces belonging to base_link.
        other_surfaces: Named local surfaces belonging to the contacting body.
        other_pose: Other body's [4,4] pose in base_link coordinates, in meters.

    Returns:
        Minimum unsigned surface separation, intersecting pair names, and nearest
        points for the closest disjoint pair. Intersection is not penetration depth.

    Note:
        BVH surface tests do not establish solid containment. No PhysX contact
        margins, solver history, or prior simulation substeps are reconstructed.
    """
    intersections = []
    closest = None
    for base_name, base_model in base_surfaces:
        base_object = fcl.CollisionObject(base_model, fcl.Transform())
        for other_name, other_model in other_surfaces:
            other_object = fcl.CollisionObject(other_model, fcl.Transform(
                other_pose[:3, :3], other_pose[:3, 3]))
            contact = fcl.CollisionResult()
            fcl.collide(base_object, other_object, fcl.CollisionRequest(), contact)
            if contact.is_collision:
                intersections.append([base_name, other_name])
                continue
            distance_result = fcl.DistanceResult()
            distance_m = float(fcl.distance(base_object, other_object,
                                           fcl.DistanceRequest(enable_nearest_points=True), distance_result))
            if closest is None or distance_m < closest['distance_m']:
                closest = {'distance_m': distance_m, 'base_surface': base_name,
                           'other_surface': other_name,
                           'nearest_points_in_base_m': [point.tolist() for point in distance_result.nearest_points]}
    return {'surface_intersection': bool(intersections), 'intersecting_pairs': intersections,
            'minimum_surface_distance_m': 0.0 if intersections else closest['distance_m'],
            'closest_disjoint_pair': closest}


def main() -> None:
    """Audit captured no-push failures with actual Gym DOF ordering.

    Raises:
        OSError: Input evidence or referenced meshes cannot be read.
        ValueError: Saved positions do not match the native DOF count.

    Note:
        Reads formal assets only and writes independent JSON; never imports Gym,
        changes training assets, or interprets triangle separation as PhysX force.
    """
    parser = argparse.ArgumentParser(description='终止姿态真实视觉与碰撞代理表面核验')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--failures', type=Path)
    parser.add_argument('--urdf', type=Path)
    args = parser.parse_args()
    task_root = Path(__file__).resolve().parents[1]
    urdf_path = args.urdf or PROJECT_ROOT / 'UniFP_v3/resources/robots/v3_a1_desktop_train_v6_coplanar_r2/urdf/a1arm_rgm3_desktop_fixed_gripper.urdf'
    robot = ET.parse(urdf_path).getroot()
    links = {link.get('name'): link for link in robot.findall('link')}
    native = json.loads((task_root / 'artifacts/static_flip_probe/summary.json').read_text())
    dof_names = native['variants'][0]['dof_names']
    failures_path = args.failures or task_root / 'artifacts/early_sample_no_push_failures.json'
    events = json.loads(failures_path.read_text())
    body_names = ('base_link', 'wrist_camera', 'wrist_camera_connector', 'arm_seg4')
    surfaces = {(name, kind): load_surfaces(urdf_path, links[name], kind)
                for name in body_names for kind in ('visual', 'collision')}
    desktop_urdf = PROJECT_ROOT / 'UniFP_v3/resources/robots/v3_a1_desktop/urdf' / urdf_path.name
    desktop_links = {link.get('name'): link for link in ET.parse(desktop_urdf).getroot().findall('link')}
    desktop_surfaces = {name: load_surfaces(desktop_urdf, desktop_links[name], 'collision')
                        for name in body_names}
    results = []
    for event in events:
        # Native DOF ordering differs from the policy's steering/drive/arm ordering.
        assert len(event['dof_position_rad']) == len(dof_names)
        positions = dict(zip(dof_names, event['dof_position_rad']))
        body_poses = forward_kinematics(urdf_path, positions, np.eye(4))
        contact_vector = np.asarray(event['base_contact_vector_n'])
        opposite = next(item['body'] for item in event['largest_contacts']
                        if item['body'] != 'base_link' and np.linalg.norm(
                            contact_vector + item['vector_n']) < 1e-5)
        # Prefer native terminal poses when available; FK-only evidence is labeled.
        pair_pose = body_poses[opposite]
        pose_source = 'urdf_fk'
        fk_position_error_m = None
        if 'body_pose_world_m_xyzw' in event:
            native_poses = {}
            for name in ('base_link', opposite):
                values = event['body_pose_world_m_xyzw'][name]
                matrix = np.eye(4)
                matrix[:3, :3] = Rotation.from_quat(values[3:7]).as_matrix()
                matrix[:3, 3] = values[:3]
                native_poses[name] = matrix
            pair_pose = np.linalg.inv(native_poses['base_link']) @ native_poses[opposite]
            pose_source = 'native_terminal_body_pose'
            fk_position_error_m = float(np.linalg.norm(pair_pose[:3, 3] - body_poses[opposite][:3, 3]))
        result = {'policy_step': event['policy_step'], 'env_id': event['env_id'],
                  'other_body': opposite, 'joint_positions_rad': positions,
                  'pose_source': pose_source, 'fk_position_error_m': fk_position_error_m,
                  'other_pose_in_base_m': pair_pose.tolist(),
                  'base_contact_norm_n': event['base_contact_norm_n']}
        for kind in ('visual', 'collision'):
            result[kind] = measure_pair(surfaces['base_link', kind],
                                        surfaces[opposite, kind], pair_pose)
        result['desktop_collision'] = measure_pair(desktop_surfaces['base_link'],
                                                   desktop_surfaces[opposite], pair_pose)
        result['train_base_desktop_other'] = measure_pair(surfaces['base_link', 'collision'],
                                                         desktop_surfaces[opposite], pair_pose)
        result['desktop_base_train_other'] = measure_pair(desktop_surfaces['base_link'],
                                                         surfaces[opposite, 'collision'], pair_pose)
        results.append(result)
        print(json.dumps({'step': result['policy_step'], 'body': opposite,
                          'visual_mm': result['visual']['minimum_surface_distance_m'] * 1000,
                          'collision_mm': result['collision']['minimum_surface_distance_m'] * 1000,
                          'desktop_collision_mm': result['desktop_collision']['minimum_surface_distance_m'] * 1000}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'urdf': str(urdf_path), 'dof_names': dof_names,
        'python_fcl_version': fcl.__version__, 'events': results,
        'note_zh': '终止策略边界的独立三角表面核验；不重建历史子步／PhysX接触余量，不排除实体包含。'},
        ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()

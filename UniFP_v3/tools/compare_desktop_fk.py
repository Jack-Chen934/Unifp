"""Compare an independent URDF FK calculation with a Gym probe report."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


def pose_from_origin(element: ET.Element) -> np.ndarray:
    """Build a homogeneous transform from one URDF origin element.

    Args:
        element: URDF origin element; missing xyz/rpy values default to zero.

    Returns:
        A 4-by-4 transform with translation in meters and URDF fixed-axis xyz
        rotation in radians.
    """
    pose = np.eye(4)
    pose[:3, 3] = np.fromstring(element.get('xyz', '0 0 0'), sep=' ')
    pose[:3, :3] = Rotation.from_euler(
        'xyz', np.fromstring(element.get('rpy', '0 0 0'), sep=' ')
    ).as_matrix()
    return pose


def axis_motion(joint: ET.Element, position: float) -> np.ndarray:
    """Build the motion transform for one revolute or prismatic joint.

    Args:
        joint: URDF joint element containing type and optional axis.
        position: Joint angle in radians or linear displacement in meters.

    Returns:
        A 4-by-4 joint-motion transform in the joint frame.
    """
    motion = np.eye(4)
    joint_type = joint.get('type')
    if joint_type == 'fixed':
        return motion
    axis_element = joint.find('axis')
    axis = np.fromstring(axis_element.get('xyz', '1 0 0'), sep=' ')
    axis = axis / np.linalg.norm(axis)
    if joint_type in ('revolute', 'continuous'):
        motion[:3, :3] = Rotation.from_rotvec(axis * position).as_matrix()
    elif joint_type == 'prismatic':
        motion[:3, 3] = axis * position
    else:
        raise ValueError(f'Unsupported URDF joint type: {joint_type}')
    return motion


def forward_kinematics(urdf_path: Path, dof_positions: Dict[str, float],
                       base_pose: np.ndarray) -> Dict[str, np.ndarray]:
    """Evaluate every URDF link pose from ``base_link``.

    Args:
        urdf_path: Active URDF whose parent/child joint graph is evaluated.
        dof_positions: Joint positions keyed by URDF joint name, in radians or
            meters according to the joint type; omitted joints use zero.
        base_pose: 4-by-4 pose of ``base_link`` in the world frame.

    Returns:
        Homogeneous world poses keyed by link name.

    Raises:
        ValueError: The joint graph cannot be traversed from ``base_link``.
    """
    joints = ET.parse(urdf_path).getroot().findall('joint')
    body_poses: Dict[str, np.ndarray] = {'base_link': base_pose.copy()}
    remaining = joints.copy()
    while remaining:
        progressed = False
        for joint in remaining[:]:
            parent = joint.find('parent').get('link')
            child = joint.find('child').get('link')
            if parent not in body_poses:
                continue
            origin = joint.find('origin')
            origin_pose = pose_from_origin(origin) if origin is not None else np.eye(4)
            joint_position = dof_positions.get(joint.get('name'), 0.0)
            body_poses[child] = body_poses[parent] @ origin_pose @ axis_motion(
                joint, joint_position
            )
            remaining.remove(joint)
            progressed = True
        if not progressed:
            missing = sorted(joint.find('parent').get('link') for joint in remaining)
            raise ValueError(f'URDF joint graph is disconnected from base_link: {missing}')
    return body_poses


def compare(probe: Dict[str, Any], urdf_path: Path, tcp_offset_m: List[float]) -> Dict[str, Any]:
    """Compute active-asset link FK and compare it with Gym body poses.

    Args:
        probe: JSON-compatible result from ``probe_desktop_asset.py``.
        urdf_path: Active URDF path used by the Gym probe.
        tcp_offset_m: Candidate TCP position in ``arm_seg6`` coordinates.

    Returns:
        Per-body position/orientation differences and independent TCP position.

    Raises:
        ValueError: The URDF joint graph cannot be traversed from ``base_link``.
    """
    dof_positions = probe['final_dof_positions_rad']
    base_pose = np.eye(4)
    base_pose[:3, 3] = probe['final_positions_m']['base_link']
    base_pose[:3, :3] = Rotation.from_quat(
        probe['final_orientations_xyzw']['base_link']
    ).as_matrix()
    body_poses = forward_kinematics(urdf_path, dof_positions, base_pose)

    differences = {}
    for body_name, expected_pose in body_poses.items():
        if body_name not in probe['final_positions_m']:
            continue
        gym_position = np.asarray(probe['final_positions_m'][body_name])
        gym_rotation = Rotation.from_quat(
            probe['final_orientations_xyzw'][body_name]
        ).as_matrix()
        rotation_delta = expected_pose[:3, :3].T @ gym_rotation
        differences[body_name] = {
            'position_error_m': float(np.linalg.norm(expected_pose[:3, 3] - gym_position)),
            'orientation_error_rad': float(Rotation.from_matrix(rotation_delta).magnitude()),
        }

    tcp_world_m = body_poses['arm_seg6'][:3, 3] + body_poses['arm_seg6'][:3, :3] @ np.asarray(tcp_offset_m)
    return {
        'urdf': str(urdf_path),
        'probe_asset': probe['asset'],
        'arm_joint_positions_rad': [dof_positions[f'arm_joint{index}'] for index in range(1, 7)],
        'tcp_offset_in_arm_seg6_m': tcp_offset_m,
        'tcp_world_position_m': tcp_world_m.tolist(),
        'gym_to_urdf_link_differences': differences,
        'max_link_position_error_m': max(item['position_error_m'] for item in differences.values()),
        'max_link_orientation_error_rad': max(item['orientation_error_rad'] for item in differences.values()),
    }


def audit_saved_ik(audit: Dict[str, Any], urdf_path: Path,
                   tcp_offset_m: List[float]) -> Dict[str, Any]:
    """Recompute saved A08 target residuals and inspect sampled joint continuity.

    Args:
        audit: Offline A08 JSON with spherical targets and six-joint solutions.
        urdf_path: Active desktop URDF used for independent forward kinematics.
        tcp_offset_m: Candidate TCP position in ``arm_seg6`` coordinates.

    Returns:
        Position-residual statistics and adjacent joint-step statistics for
        the 12 sampled first-transition paths.
    """
    solutions = audit['joint_solutions_rad']
    spherical_points = audit['samples_spherical_r_m_elevation_rad_azimuth_rad']
    mount = np.asarray(audit['reference_mount_in_base_m'])
    tcp_offset = np.asarray(tcp_offset_m)
    residuals = []
    for spherical_point, arm_positions in zip(spherical_points, solutions):
        radius_m, elevation_rad, azimuth_rad = spherical_point
        target_relative_to_mount = radius_m * np.asarray([
            np.cos(elevation_rad) * np.cos(azimuth_rad),
            np.cos(elevation_rad) * np.sin(azimuth_rad),
            np.sin(elevation_rad),
        ])
        joint_positions = {f'arm_joint{index + 1}': value
                           for index, value in enumerate(arm_positions)}
        link_poses = forward_kinematics(urdf_path, joint_positions, np.eye(4))
        tcp_position = link_poses['arm_seg6'][:3, 3] + link_poses['arm_seg6'][:3, :3] @ tcp_offset
        residuals.append(float(np.linalg.norm(tcp_position - mount - target_relative_to_mount)))

    transition_start = (
        audit['summary']['endpoint_grid']['samples']
        + audit['summary']['random_endpoints']['samples']
    )
    transition_count = audit['summary']['first_transition_samples']['samples']
    samples_per_path = transition_count // 12
    path_steps = []
    for path_index in range(12):
        offset = transition_start + path_index * samples_per_path
        path_solutions = np.asarray(solutions[offset:offset + samples_per_path])
        joint_step_max = np.max(np.abs(np.diff(path_solutions, axis=0)), axis=0)
        path_steps.append({
            'path_index': path_index,
            'max_joint_step_rad': float(joint_step_max.max()),
            'max_step_by_joint_rad': joint_step_max.tolist(),
        })
    return {
        'sample_count': len(residuals),
        'position_residual_p95_m': float(np.percentile(residuals, 95)),
        'position_residual_max_m': max(residuals),
        'position_residual_below_0_1mm_count': sum(value < 1e-4 for value in residuals),
        'first_transition_path_count': len(path_steps),
        'first_transition_path_max_joint_step_rad': max(
            path['max_joint_step_rad'] for path in path_steps
        ),
        'first_transition_paths': path_steps,
    }


def main() -> None:
    """Load diagnostic inputs and write a machine-readable FK comparison.

    Returns:
        None.

    Note:
        Writes only the requested report; it does not create a simulator or
        modify robot assets.
    """
    parser = argparse.ArgumentParser(description='活动桌面URDF与Gym刚体FK对照')
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--urdf', type=Path, required=True)
    parser.add_argument('--tcp-offset', type=Path, required=True)
    parser.add_argument('--saved-audit', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    probe = json.loads(args.probe.read_text())
    tcp_source = json.loads(args.tcp_offset.read_text())['tcp_surface_candidate']
    report = compare(probe, args.urdf, tcp_source['point_in_arm_seg6_m'])
    if args.saved_audit:
        saved_audit = json.loads(args.saved_audit.read_text())
        report['saved_ik_audit'] = audit_saved_ik(
            saved_audit, args.urdf, tcp_source['point_in_arm_seg6_m']
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({
        'max_link_position_error_m': report['max_link_position_error_m'],
        'max_link_orientation_error_rad': report['max_link_orientation_error_rad'],
        'tcp_world_position_m': report['tcp_world_position_m'],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()

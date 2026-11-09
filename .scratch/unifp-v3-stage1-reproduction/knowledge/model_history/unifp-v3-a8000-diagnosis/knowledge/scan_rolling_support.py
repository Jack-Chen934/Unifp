"""Estimate geometric support-height variation of the retained wheel collisions."""

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh


parser = argparse.ArgumentParser()
parser.add_argument('--asset-root', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
urdf_path = args.asset_root / 'urdf/a1arm_rgm3_desktop_fixed_gripper.urdf'
robot = ET.parse(urdf_path).getroot()
angles_rad = np.linspace(0.0, 2.0 * np.pi, 1441)
assumed_normal_load_n = 400.0
report = {
    'scope_zh': 'URDF轮轴局部Y轴旋转时的顶点支撑包络；几何量级估算，不是PhysX阻力矩实测。',
    'urdf_sha256': hashlib.sha256(urdf_path.read_bytes()).hexdigest(),
    'angle_samples': int(angles_rad.size),
    'assumed_normal_load_N': assumed_normal_load_n,
    'wheels': {},
}
for corner in ('fl', 'fr', 'bl', 'br'):
    link_name = corner + '_Link'
    link = robot.find(f"link[@name='{link_name}']")
    vertices_by_shape = []
    mesh_hashes = {}
    for collision in link.findall('collision'):
        mesh_element = collision.find('geometry/mesh')
        mesh_path = (urdf_path.parent / mesh_element.get('filename')).resolve()
        geometry = trimesh.load(mesh_path, force='mesh')
        mesh_vertices_m = np.asarray(geometry.vertices).copy()
        mesh_vertices_m *= np.fromstring(mesh_element.get('scale', '1 1 1'), sep=' ')
        origin = collision.find('origin')
        if origin is not None:
            origin_rpy_rad = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')
            origin_xyz_m = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
            mesh_vertices_m = Rotation.from_euler('xyz', origin_rpy_rad).apply(mesh_vertices_m) + origin_xyz_m
        vertices_by_shape.append(mesh_vertices_m)
        mesh_hashes[mesh_path.name] = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
    # A convex hull has the same directional support as its source vertices.
    vertices_m = np.concatenate(vertices_by_shape)
    downward_support_m = np.max(
        vertices_m[:, 0, None] * np.sin(angles_rad)
        - vertices_m[:, 2, None] * np.cos(angles_rad), axis=0
    )
    support_slope_m_per_rad = np.gradient(downward_support_m, angles_rad)
    peak_slope_m_per_rad = float(np.max(np.abs(support_slope_m_per_rad)))
    report['wheels'][link_name] = {
        'mesh_sha256': mesh_hashes,
        'support_min_m': float(downward_support_m.min()),
        'support_max_m': float(downward_support_m.max()),
        'support_variation_m': float(np.ptp(downward_support_m)),
        'max_abs_support_slope_m_per_rad': peak_slope_m_per_rad,
        'quasistatic_lift_torque_scale_Nm': assumed_normal_load_n * peak_slope_m_per_rad,
    }
report['nominal_zero_speed_feedback_Nm'] = {
    'target_2_9527559_rad_s': 0.227385 * 2.9527559,
    'target_20_rad_s': 0.227385 * 20.0,
}
args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
print(json.dumps(report, ensure_ascii=False, indent=2))

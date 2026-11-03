"""Render preserved visuals and local proxies in representative joint poses."""

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import trimesh

from build_desktop_asset import origin_transform
from build_training_collision_asset import DEFAULT_OUTPUT_ROOT, URDF_NAME
from compare_desktop_fk import forward_kinematics


def render_overlay(urdf_path: Path, output_path: Path) -> None:
    """Render four poses with exact visual points and translucent proxy hulls.

    Args:
        urdf_path: Candidate URDF resolving visual and collision meshes.
        output_path: PNG path for a standalone inspection artifact.

    Returns:
        None.

    Note:
        Writes an image only. The overlay is qualitative; it does not certify
        surface tolerances, continuous joint clearance or runtime contacts.
    """
    robot = ET.parse(urdf_path).getroot()
    nominal = {'arm_joint2': 1.0, 'arm_joint3': -1.5}
    steering = dict(nominal, **{corner + '_steering_joint': 0.7 for corner in ('fl', 'fr', 'bl', 'br')})
    folded = dict(nominal, arm_joint2=1.6, arm_joint3=-2.2)
    cases = [('零位', {}, 25, -55), ('默认臂位', nominal, 10, -90),
             ('转向0.7rad', steering, 65, -70), ('折叠臂示例', folded, 15, 30)]
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'DejaVu Sans']
    figure = plt.figure(figsize=(16, 12))
    # Load each local mesh once; preserve its declared scale and scene transform.
    geometries = []
    for link in robot.findall('link'):
        for kind in ('visual', 'collision'):
            for element in link.findall(kind):
                reference = element.find('geometry/mesh')
                mesh = trimesh.load(urdf_path.parent / reference.get('filename'), force='scene').to_geometry()
                mesh.apply_scale(np.fromstring(reference.get('scale', '1 1 1'), sep=' '))
                mesh.apply_transform(origin_transform(element.find('origin')))
                geometries.append((link.get('name'), kind, mesh))
    for panel, (title, positions, elevation, azimuth) in enumerate(cases, 1):
        axes = figure.add_subplot(2, 2, panel, projection='3d')
        poses = forward_kinematics(urdf_path, positions, np.eye(4))
        for name, kind, local_mesh in geometries:
            mesh = local_mesh.copy()
            mesh.apply_transform(poses[name])
            if kind == 'visual':
                points = mesh.vertices[::max(1, len(mesh.vertices) // 1200)]
                axes.scatter(*points.T, s=0.1, alpha=0.25, color='0.25', rasterized=True)
            else:
                hull = mesh.convex_hull
                axes.add_collection3d(Poly3DCollection(
                    hull.triangles, facecolor='tab:orange', edgecolor='none', alpha=0.12))
        axes.set(xlim=(-0.8, 0.8), ylim=(-0.45, 0.45), zlim=(-0.4, 1.3),
                 xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)', title=title)
        axes.set_box_aspect((1.6, 0.9, 1.7))
        axes.view_init(elev=elevation, azim=azimuth)
    figure.suptitle('灰色：原始视觉顶点；橙色：候选碰撞凸包（含保留轮体）')
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    """Write a qualitative collision overlay from command-line file paths.

    Returns:
        None.

    Note:
        Creates the specified PNG and its parent directory.
    """
    parser = argparse.ArgumentParser(description='候选碰撞与桌面视觉叠图')
    parser.add_argument('--asset', type=Path, default=DEFAULT_OUTPUT_ROOT / 'urdf' / URDF_NAME)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    render_overlay(arguments.asset, arguments.output)


if __name__ == '__main__':
    main()

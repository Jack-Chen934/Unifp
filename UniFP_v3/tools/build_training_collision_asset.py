"""Build local convex proxies without changing the approved desktop model."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Dict, List
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull
import trimesh

from build_desktop_asset import origin_transform

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / 'UniFP_v3/resources/robots/v3_a1_desktop'
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / 'UniFP_v3/resources/robots/v3_a1_desktop_train_v1'
URDF_NAME = 'a1arm_rgm3_desktop_fixed_gripper.urdf'
CORNERS = ('fl', 'fr', 'bl', 'br')
# These component indices refer to the frozen desktop source manifest, not faces.
PROFILE_GROUPS = (
    [0, 3, 4, 9], [2, 5, 6, 7, 8, 10, 12, 14, 17],
    [19, 21], [11, 15, 22], [1, 16], [13, 18], [20], [23], [24])
SOURCE_URDF_SHA256 = '1dc2f47cf62baf707d9f1e7595b9df6b4455193feabedbd0b46e940ddffb9c0d'


def collision_points(urdf_path: Path, collision: ET.Element) -> np.ndarray:
    """Read one source mesh collision in its link frame.

    Args:
        urdf_path: Desktop URDF path resolving relative mesh references.
        collision: Source mesh collision including origin and optional scale.

    Returns:
        Vertex array of shape [V, 3], in link-local metres.

    Raises:
        FileNotFoundError: A referenced source mesh is missing.
    """
    reference = collision.find('geometry/mesh')
    mesh = trimesh.load(urdf_path.parent / reference.get('filename'), force='scene').to_geometry()
    mesh.apply_scale(np.fromstring(reference.get('scale', '1 1 1'), sep=' '))
    mesh.apply_transform(origin_transform(collision.find('origin')))
    return np.asarray(mesh.vertices)


def compact_hull(points: np.ndarray, tolerance_m: float) -> tuple[trimesh.Trimesh, float]:
    """Approximate a local convex envelope with at most 64 original vertices.

    Args:
        points: Source vertices of shape [V, 3], in link-local metres.
        tolerance_m: Desired maximum source-to-proxy supporting-plane distance.

    Returns:
        Convex mesh and measured maximum outside-plane distance in metres.

    Raises:
        scipy.spatial.QhullError: Source vertices cannot enclose a 3D volume.
    """
    source_hull = ConvexHull(points)
    hull_points = points[source_hull.vertices]
    # Start with a full-volume hull; then refine the largest omitted support.
    sample_indices = list(dict.fromkeys(
        [int(np.argmin(hull_points[:, axis])) for axis in range(3)] +
        [int(np.argmax(hull_points[:, axis])) for axis in range(3)]))
    # A tetrahedron chosen from the source hull avoids coplanar axis extrema.
    if len(sample_indices) < 4 or np.linalg.matrix_rank(
            hull_points[sample_indices] - hull_points[sample_indices[0]]) < 3:
        sample_indices = list(range(min(64, len(hull_points))))
    while True:
        proxy_hull = ConvexHull(hull_points[sample_indices])
        plane_distances = (hull_points @ proxy_hull.equations[:, :3].T +
                           proxy_hull.equations[:, 3]).max(axis=1)
        maximum_error = max(0.0, float(plane_distances.max()))
        if maximum_error <= tolerance_m or len(sample_indices) >= 64:
            break
        sample_indices.append(int(plane_distances.argmax()))
    proxy_points = hull_points[sample_indices]
    mesh = trimesh.Trimesh(vertices=proxy_points, faces=proxy_hull.simplices, process=False).convex_hull
    return mesh, maximum_error


def component_groups(link_name: str, parts: List[np.ndarray],
                     source_groups: Dict[str, List[int]]) -> Dict[str, List[List[int]]]:
    """Choose functional component groups without crossing moving link frames.

    Args:
        link_name: Approved URDF link name.
        parts: Ordered source component vertices, each in link-local metres.
        source_groups: Source manifest collision-group to component-index mapping.

    Returns:
        Named semantic groups, each containing groups of source components.
    """
    all_indices = list(range(len(parts)))
    if link_name == 'base_link':
        buckets: Dict[int, List[int]] = {}
        for index in source_groups['base_link_collision']:
            center = (parts[index].min(axis=0) + parts[index].max(axis=0)) / 2
            if center[0] > 0.15:
                bucket = 2 + int(center[0] > 0.30) * 2 + int(center[1] > 0)
            elif center[0] < -0.18:
                bucket = 6 + int(center[1] > 0)
            else:
                bucket = int(center[2] > -0.11)
            buckets.setdefault(bucket, []).append(index)
        shell_groups = []
        for key in sorted(buckets):
            indices = buckets[key]
            if key < 4:
                shell_groups.append(indices)
                continue
            # Separate the low central bumper from the high outer wheel arch.
            lateral_cut_m = 0.14 if key < 6 else 0.11
            for outside in (False, True):
                group = [index for index in indices if
                         (abs((parts[index].min(axis=0)[1] + parts[index].max(axis=0)[1]) / 2)
                          > lateral_cut_m) == outside]
                if group:
                    shell_groups.append(group)
        groups = {'base_link_collision': shell_groups}
        # Lower wheel-arch CAD pieces must not be joined across their recess.
        # The failed 80-shape hull filled a 1.3 mm left-rear clearance and
        # repeatedly contacted the tire after its first small rotation.
        refined_groups = []
        for indices in shell_groups:
            centers = np.array([(parts[index].min(axis=0) + parts[index].max(axis=0)) / 2
                                for index in indices])
            near_wheel = abs(centers[:, 0].mean()) > 0.15 and abs(centers[:, 1].mean()) > 0.10
            lower = [index for index in indices if near_wheel and parts[index][:, 2].min() < -0.11]
            upper = [index for index in indices if index not in lower]
            if upper:
                refined_groups.append(upper)
            refined_groups.extend([[index] for index in lower])
        groups['base_link_collision'] = refined_groups
        groups.update({corner + '_fixed_collision': [source_groups[corner + '_fixed_collision']]
                       for corner in CORNERS})
        return groups
    if link_name in {corner + '_Link' for corner in CORNERS}:
        # Preserve the source tire (component 4); only the recessed hub solids
        # share a hull, whose radial extent remains well inside the tire.
        return {next(iter(source_groups)): [[index for index in all_indices if index != 4], [4]]}
    if link_name == 'aluminum_profiles':
        return {next(iter(source_groups)): list(PROFILE_GROUPS)}
    if link_name.endswith('_steering_link'):
        # The upper bearing and rear fork are separate; neither spans the tire.
        upper, lower = [], []
        for index, points in enumerate(parts):
            center = (points.min(axis=0) + points.max(axis=0)) / 2
            (upper if center[2] > -0.032 else lower).append(index)
        return {next(iter(source_groups)): [upper, lower]}
    if link_name in ('arm_base', 'arm_seg1', 'arm_seg6') or 'camera' in link_name:
        return {next(iter(source_groups)): [all_indices]}
    # Split remaining shells at physical length axes, keeping each CAD solid
    # intact. This retains the elbow's bend instead of enclosing the whole link.
    axis = {'arm_base': 2, 'arm_seg2': 0, 'arm_seg3': 1,
            'arm_seg4': 2, 'arm_seg5': 1, 'gripper1': 0, 'gripper2': 0}[link_name]
    centers = np.array([(points.min(axis=0) + points.max(axis=0)) / 2 for points in parts])
    threshold = (centers[:, axis].min() + centers[:, axis].max()) / 2
    groups = [[index for index in all_indices if centers[index, axis] <= threshold],
              [index for index in all_indices if centers[index, axis] > threshold]]
    return {next(iter(source_groups)): [group for group in groups if group]}


def build_training_collision_asset(source_root: Path = DEFAULT_SOURCE_ROOT,
                                   output_root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    """Write an independent candidate with traced local convex proxies.

    Args:
        source_root: Frozen approved desktop asset root; never modified.
        output_root: New, non-existing candidate directory.

    Returns:
        Generated candidate URDF path, suitable for cfg.asset.file.

    Raises:
        FileExistsError: Output already exists, including the source directory.
        ValueError: Source identity differs or a local hull exceeds its tolerance.
        FileNotFoundError: Required desktop files are absent.

    Note:
        Copies visual bytes, joints and inertials unchanged. Tires and the
        separated lower wheel arches retain original collision meshes; recessed
        hub solids share a separate convex proxy.
        Does not modify any training configuration or running process.
    """
    source_path = source_root / 'urdf' / URDF_NAME
    source_bytes = source_path.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != SOURCE_URDF_SHA256:
        raise ValueError('The desktop source identity differs from the reviewed component mapping')
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / 'urdf').mkdir()
    (output_root / 'meshes').mkdir()
    robot = ET.fromstring(source_bytes)
    source_manifest = json.loads((source_root / 'asset_manifest.json').read_text())
    counts: Dict[str, Any] = {}
    proxy_records: List[dict] = []
    retained_collision_records: List[dict] = []
    for link in robot.findall('link'):
        link_name = link.get('name')
        original_collisions = list(link.findall('collision'))
        source_groups = source_manifest['collision_solid_counts'][link_name]['source_groups']
        if not original_collisions:
            counts[link_name] = {'count': 0, 'source_groups': {}}
            continue
        parts = [collision_points(source_path, collision) for collision in original_collisions]
        for collision in original_collisions:
            link.remove(collision)
        local_count = 0
        new_source_groups = {}
        is_wheel = link_name in {corner + '_Link' for corner in CORNERS}
        tolerance_m = 0.001 if link_name.startswith('gripper') or is_wheel else 0.003
        for group_name, component_sets in component_groups(link_name, parts, source_groups).items():
            local_indices = []
            for indices in component_sets:
                retain_tire = is_wheel and indices == [4]
                retain_arch = link_name == 'base_link' and len(indices) == 1
                if retain_tire or retain_arch:
                    # The real GPU pipeline contact regression passes only
                    # when these near-contact CAD pieces keep their original
                    # representation, even after the false convex fill is split.
                    link.append(deepcopy(original_collisions[indices[0]]))
                    retained_collision_records.append({'link': link_name, 'local_shape': local_count,
                                                       'source_components': indices})
                    local_indices.append(local_count)
                    local_count += 1
                    continue
                points = np.concatenate([parts[index] for index in indices])
                mesh, support_error_m = compact_hull(points, tolerance_m)
                if support_error_m > tolerance_m:
                    raise ValueError(f'{link_name}: proxy support error {support_error_m} > {tolerance_m}')
                filename = f'{link_name}_proxy_{local_count}.glb'
                mesh.export(output_root / 'meshes' / filename)
                collision = ET.SubElement(link, 'collision', name=f'{link_name}_proxy_{local_count}')
                ET.SubElement(collision, 'origin', xyz='0 0 0', rpy='0 0 0')
                ET.SubElement(ET.SubElement(collision, 'geometry'), 'mesh', filename='../meshes/' + filename)
                local_indices.append(local_count)
                proxy_records.append({'link': link_name, 'local_shape': local_count,
                                      'source_group': group_name, 'source_components': indices,
                                      'vertices': len(mesh.vertices), 'support_error_m': support_error_m})
                local_count += 1
            new_source_groups[group_name] = local_indices
        counts[link_name] = {'count': local_count, 'source_groups': new_source_groups}

    # Preserve relative references and bytes for visuals and retained wheels.
    mesh_files = []
    for reference in sorted({node.get('filename') for node in robot.findall('.//mesh')}):
        target = (output_root / 'urdf' / reference).resolve()
        if not target.exists():
            source_mesh = (source_path.parent / reference).resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_mesh, target)
        mesh_files.append({'path': str(target.relative_to(output_root.resolve())),
                           'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
    urdf_path = output_root / 'urdf' / URDF_NAME
    ET.indent(robot, space='  ')
    ET.ElementTree(robot).write(urdf_path, encoding='utf-8', xml_declaration=True)
    manifest = {
        'asset_status': 'independent_training_collision_candidate',
        'asset_status_zh': '独立候选，未完成静态／物理／性能验收；不可替换正式训练。',
        'source_asset': str(source_root), 'source_urdf_sha256': SOURCE_URDF_SHA256,
        'source_manifest_sha256': hashlib.sha256((source_root / 'asset_manifest.json').read_bytes()).hexdigest(),
        'derived_urdf': str(urdf_path.relative_to(output_root)),
        'derived_sha256': hashlib.sha256(urdf_path.read_bytes()).hexdigest(),
        'expected_dofs': 14, 'collision_shape_count': sum(item['count'] for item in counts.values()),
        'collision_solid_counts': counts, 'proxy_records': proxy_records,
        'retained_collision_records': retained_collision_records, 'mesh_files': mesh_files,
        'geometry_note_zh': '新增局部凸代理最多64顶点；轮拱下缘及四轮胎碰撞逐字节保留原表示，轮毂单独近似。凸包支持面误差不等于凹陷填充误差，仍需间隙与接触对照。',
    }
    (output_root / 'asset_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    (output_root / 'README.md').write_text(
        '# V3 训练碰撞独立候选\n\n'
        '视觉／关节／惯性保留桌面基准，局部CAD凸代理降低碰撞数量；轮胎保留原始碰撞，轮毂单独近似。'
        '本目录未完成物理与性能验收，不能替换正式资产。见任务实验记录。\n')
    return urdf_path


def main() -> None:
    """Parse paths and write a candidate without overwriting existing assets.

    Returns:
        None.

    Raises:
        FileExistsError: Candidate output directory already exists.

    Note:
        Writes candidate files only under the requested output root.
    """
    parser = argparse.ArgumentParser(description='生成V3独立局部凸代理资产')
    parser.add_argument('--source-root', type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT_ROOT)
    arguments = parser.parse_args()
    print(build_training_collision_asset(arguments.source_root, arguments.output_root))


if __name__ == '__main__':
    main()

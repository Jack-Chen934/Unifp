"""Build the approved V3 candidate from clipped desktop visual surfaces."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from scipy.spatial import ConvexHull, HalfspaceIntersection

PROJECT_ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(PROJECT_ROOT / 'UniFP_v3/tools'))
from build_training_collision_asset import collision_points
from compare_desktop_fk import pose_from_origin

SOURCE_ROOT = PROJECT_ROOT / 'UniFP_v3/resources/robots/v3_a1_desktop_train_v7_local_base_proxy_r3'
DESKTOP_ROOT = PROJECT_ROOT / 'UniFP_v3/resources/robots/v3_a1_desktop'
DEFAULT_OUTPUT = PROJECT_ROOT / 'UniFP_v3/resources/robots/v3_a1_desktop_train_v8_visual_envelope_r1'
URDF_NAME = 'a1arm_rgm3_desktop_fixed_gripper.urdf'
SOURCE_URDF_SHA256 = '0f94cea2bad8b3cf2fa7c5b422348adb2e0345b899df2fddebffaa8d6d7a5e45'
SOURCE_MANIFEST_SHA256 = '3174b486bb12e7a103bb0e18815af329b11a7308b6f7074e8e5f50728e6db1c2'
MARGIN_M = 0.0005
COVERAGE_TOLERANCE_M = 1e-5
MAX_HULL_VERTICES = 64
MAX_COVERING_PLANES = (MAX_HULL_VERTICES + 4) // 2
TARGET_COMPONENTS = {'base_link_local_1_group_0': 8,
                     'base_link_local_1_group_5': 99, 'base_link_proxy_13': 43}


def clip_visual_triangles(visual_mesh: trimesh.Trimesh,
                          region_planes: np.ndarray) -> np.ndarray:
    """Clip full triangles against a convex region without discarding crossing faces.

    Args:
        visual_mesh: Triangular visual surface in base_link coordinates, in meters.
        region_planes: Normalized [K,4] plane coefficients with interior n*x+d<=0.

    Returns:
        Unique [P,3] vertices in meters, including newly created edge crossings.
        Returns an empty [0,3] array when no visual surface intersects the region.
    """
    triangles_m = np.asarray(visual_mesh.triangles)
    # A face is outside only if all three vertices violate the same half-space.
    # This rejects distant faces while retaining triangles crossing the whole ROI.
    distances_m = triangles_m @ region_planes[:, :3].T + region_planes[:, 3]
    candidate_faces = ~np.any(np.all(distances_m > 1e-10, axis=1), axis=1)
    clipped_polygons = []
    for triangle_m in triangles_m[candidate_faces]:
        polygon_m = list(triangle_m)
        for plane in region_planes:
            if not polygon_m:
                break
            clipped_polygon_m = []
            previous_point_m = polygon_m[-1]
            previous_distance_m = float(previous_point_m @ plane[:3] + plane[3])
            for point_m in polygon_m:
                distance_m = float(point_m @ plane[:3] + plane[3])
                if (distance_m <= 0) != (previous_distance_m <= 0):
                    fraction = previous_distance_m / (previous_distance_m - distance_m)
                    clipped_polygon_m.append(previous_point_m + fraction * (point_m - previous_point_m))
                if distance_m <= 0:
                    clipped_polygon_m.append(point_m)
                previous_point_m, previous_distance_m = point_m, distance_m
            polygon_m = clipped_polygon_m
        if polygon_m:
            clipped_polygons.extend(polygon_m)
    if not clipped_polygons:
        return np.empty((0, 3))
    return np.unique(np.asarray(clipped_polygons), axis=0)


def rebuild_local_hull(local_points_m: np.ndarray,
                       margin_m: float) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Offset local visual support planes outward and retain complete coverage.

    Args:
        local_points_m: Full clipped visual point set, shape [P,3], in base meters.
        margin_m: Outward distance along each unit support normal, in meters.

    Returns:
        Watertight convex mesh and reference-plane coverage/expansion measurements.
        Expansion is measured along original hull normals, not every ray or cavity.

    Raises:
        scipy.spatial.QhullError: Reference points do not enclose a 3D volume.
        ValueError: The covering mesh exceeds the approved 64-vertex budget.
    """
    reference_hull = ConvexHull(local_points_m)
    reference_planes = np.unique(np.round(reference_hull.equations, 12), axis=0)
    reference_normals = reference_planes[:, :3]
    reference_support_m = (local_points_m @ reference_normals.T).max(axis=0)
    # Start with a bounded box, then add the plane that removes the largest excess.
    # Every added half-space covers the full reference; no omitted vertex is cut off.
    covering_normals = np.concatenate((np.eye(3), -np.eye(3)))
    covering_support_m = (local_points_m @ covering_normals.T).max(axis=0) + margin_m
    interior_point_m = local_points_m[reference_hull.vertices].mean(axis=0)
    # A convex polyhedron with at most 34 faces has at most 64 vertices (Euler).
    # This bounds topology without using an inscribed approximation of the visual.
    while True:
        covering_planes = np.column_stack((covering_normals, -covering_support_m))
        intersections = HalfspaceIntersection(covering_planes, interior_point_m)
        mesh = trimesh.convex.convex_hull(intersections.intersections)
        expansion_m = (mesh.vertices @ reference_normals.T).max(axis=0) - reference_support_m
        largest_expansion = int(np.argmax(expansion_m))
        if len(covering_normals) >= MAX_COVERING_PLANES or expansion_m[largest_expansion] <= margin_m + 1e-10:
            break
        covering_normals = np.vstack((covering_normals, reference_normals[largest_expansion]))
        covering_support_m = np.append(covering_support_m, reference_support_m[largest_expansion] + margin_m)
    if len(mesh.vertices) > MAX_HULL_VERTICES:
        raise ValueError(f'Covering hull has {len(mesh.vertices)} vertices, budget is {MAX_HULL_VERTICES}')
    final_planes = ConvexHull(mesh.vertices).equations
    outside_distance_m = float(np.maximum(
        local_points_m @ final_planes[:, :3].T + final_planes[:, 3], 0).max())
    return mesh, {'reference_hull_vertices': int(len(reference_hull.vertices)),
                  'hull_vertices': int(len(mesh.vertices)), 'target_margin_m': margin_m,
                  'support_plane_count': int(len(covering_normals)),
                  'max_reference_outside_m': outside_distance_m,
                  'reference_normal_expansion_min_m': float(expansion_m.min()),
                  'reference_normal_expansion_max_m': float(expansion_m.max())}


def file_sha256(path: Path) -> str:
    """Read the SHA256 identity of one input or generated file.

    Args:
        path: Existing file whose bytes define its identity.

    Returns:
        Hexadecimal SHA256 digest.

    Raises:
        OSError: The file cannot be read.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_visual_mesh(urdf_path: Path, visual: ET.Element) -> trimesh.Trimesh:
    """Load one visual surface and apply its URDF transform exactly once.

    Args:
        urdf_path: Desktop URDF resolving the relative mesh reference.
        visual: Visual element with mesh scale and link-local origin.

    Returns:
        Triangle mesh in base_link coordinates, in meters.

    Raises:
        OSError: The referenced mesh cannot be read.
    """
    reference = visual.find('geometry/mesh')
    mesh = trimesh.load((urdf_path.parent / reference.get('filename')).resolve(),
                        force='scene').to_geometry()
    mesh.apply_scale(np.fromstring(reference.get('scale', '1 1 1'), sep=' '))
    mesh.apply_transform(pose_from_origin(visual.find('origin')))
    return mesh


def build_candidate(output_root: Path) -> Path:
    """Build the three approved replacements in a new self-contained V8 directory.

    Args:
        output_root: New candidate asset root; existing directories are rejected.

    Returns:
        Generated URDF path, with the original shape ordering and source coverage.

    Raises:
        FileExistsError: The output directory already exists.
        ValueError: Frozen source identity, visual geometry or coverage is invalid.
        scipy.spatial.QhullError: A clipped reference cannot form a 3D hull.
        OSError: Input reading or new candidate writing fails.

    Note:
        Writes only output_root. Keeps production configuration and source assets
        unchanged; successful generation does not grant runtime or training admission.
    """
    if output_root.exists():
        raise FileExistsError(output_root)
    source_urdf = SOURCE_ROOT / 'urdf' / URDF_NAME
    source_manifest_path = SOURCE_ROOT / 'asset_manifest.json'
    if (file_sha256(source_urdf) != SOURCE_URDF_SHA256
            or file_sha256(source_manifest_path) != SOURCE_MANIFEST_SHA256):
        raise ValueError('Frozen r3 source identity differs from the approved asset')
    source_manifest = json.loads(source_manifest_path.read_text())
    for record in source_manifest['mesh_files']:
        if file_sha256(SOURCE_ROOT / record['path']) != record['sha256']:
            raise ValueError(f"r3 mesh identity differs: {record['path']}")
    desktop_urdf = DESKTOP_ROOT / 'urdf' / URDF_NAME
    desktop_base = ET.parse(desktop_urdf).getroot().find("link[@name='base_link']")
    desktop_collisions = desktop_base.findall('collision')
    visual_element = desktop_base.find('visual')
    visual_path = (desktop_urdf.parent / visual_element.find('geometry/mesh').get('filename')).resolve()
    visual_mesh = load_visual_mesh(desktop_urdf, visual_element)
    robot = ET.parse(source_urdf).getroot()
    base = robot.find("link[@name='base_link']")
    input_hashes = {str(path.relative_to(PROJECT_ROOT)): file_sha256(path)
                    for path in (desktop_urdf, visual_path, source_urdf, source_manifest_path)}

    # Finish all local geometry before creating a distributable output directory.
    # Source convex regions localize full visual triangles; triangle crossings are
    # retained even when no original vertex belongs to that CAD component's region.
    generated = []
    for local_shape, collision in enumerate(base.findall('collision')):
        name = collision.get('name')
        if name not in TARGET_COMPONENTS:
            continue
        component = TARGET_COMPONENTS[name]
        source_record = next(record for record in source_manifest['proxy_records']
                             if record['link'] == 'base_link' and record['local_shape'] == local_shape)
        region_points_m = collision_points(desktop_urdf, desktop_collisions[component])
        region_planes = ConvexHull(region_points_m).equations
        local_points_m = clip_visual_triangles(visual_mesh, region_planes)
        if len(local_points_m) < 4:
            raise ValueError(f'{name}: no full-volume local visual reference')
        local_hull, local_measurements = rebuild_local_hull(local_points_m, MARGIN_M)
        retained_components = [index for index in source_record['source_components'] if index != component]
        retained_points_m = [collision_points(desktop_urdf, desktop_collisions[index])
                             for index in retained_components]
        combined_reference_m = np.concatenate([local_hull.vertices, *retained_points_m])
        if retained_components:
            final_hull, final_measurements = rebuild_local_hull(combined_reference_m, 0.0)
        else:
            final_hull, final_measurements = local_hull, local_measurements
        final_planes = ConvexHull(final_hull.vertices).equations
        visual_outside_m = float(np.maximum(local_points_m @ final_planes[:, :3].T + final_planes[:, 3], 0).max())
        if (visual_outside_m > COVERAGE_TOLERANCE_M
                or final_measurements['max_reference_outside_m'] > COVERAGE_TOLERANCE_M):
            raise ValueError(f'{name}: final convex hull loses reference coverage')
        for index in source_record['source_components']:
            reference = desktop_collisions[index].find('geometry/mesh')
            path = (desktop_urdf.parent / reference.get('filename')).resolve()
            input_hashes[str(path.relative_to(PROJECT_ROOT))] = file_sha256(path)
        generated.append((collision, final_hull, {
            'collision_name': name, 'local_shape': local_shape,
            'source_components': source_record['source_components'],
            'rebuilt_source_component': component, 'retained_source_components': retained_components,
            'target_margin_m': MARGIN_M, 'region_planes_in_base_m': region_planes.tolist(),
            'clipped_point_count': int(len(local_points_m)),
            'clipped_bounds_in_base_m': [local_points_m.min(axis=0).tolist(), local_points_m.max(axis=0).tolist()],
            'local_hull_measurements': local_measurements, 'final_hull_measurements': final_measurements,
            'final_max_visual_outside_m': visual_outside_m}))
    if len(generated) != len(TARGET_COMPONENTS):
        raise ValueError('Expected exactly the three approved r3 collisions')

    # Preserve shape names and order, changing only the target mesh references.
    shutil.copytree(SOURCE_ROOT, output_root)
    # A failed export must not leave the copied r3 manifest looking like a valid V8.
    (output_root / 'asset_manifest.json').unlink()
    for collision, hull, record in generated:
        old_reference = collision.find('geometry/mesh').get('filename')
        mesh_name = f"base_link_visual_envelope_source_{record['rebuilt_source_component']}.glb"
        hull.export(output_root / 'meshes' / mesh_name)
        serialized_mesh = trimesh.load(output_root / 'meshes' / mesh_name, force='scene').to_geometry()
        serialized_planes = ConvexHull(serialized_mesh.vertices).equations
        clipped_points_m = clip_visual_triangles(visual_mesh, np.asarray(record['region_planes_in_base_m']))
        record['final_max_visual_outside_m'] = float(np.maximum(
            clipped_points_m @ serialized_planes[:, :3].T + serialized_planes[:, 3], 0).max())
        record['serialized_hull_vertices'] = int(len(serialized_mesh.vertices))
        retained_outside_m = 0.0
        for component in record['retained_source_components']:
            points_m = collision_points(desktop_urdf, desktop_collisions[component])
            retained_outside_m = max(retained_outside_m, float(np.maximum(
                points_m @ serialized_planes[:, :3].T + serialized_planes[:, 3], 0).max()))
        record['serialized_max_retained_source_outside_m'] = retained_outside_m
        if max(record['final_max_visual_outside_m'], retained_outside_m) > COVERAGE_TOLERANCE_M:
            raise ValueError(f"{record['collision_name']}: serialized mesh loses reference coverage")
        collision.find('geometry/mesh').set('filename', '../meshes/' + mesh_name)
        collision.find('origin').set('xyz', '0 0 0')
        collision.find('origin').set('rpy', '0 0 0')
        record['mesh'] = 'meshes/' + mesh_name
        old_mesh_path = (output_root / 'urdf' / old_reference).resolve()
        if not any((output_root / 'urdf' / node.get('filename')).resolve() == old_mesh_path
                   for node in robot.findall('.//mesh')):
            old_mesh_path.unlink()
    candidate_urdf = output_root / 'urdf' / URDF_NAME
    ET.indent(robot, space='  ')
    ET.ElementTree(robot).write(candidate_urdf, encoding='utf-8', xml_declaration=True)

    # Keep historical V6-to-r3 maps unchanged, and identify this direct r3 derivation.
    manifest = deepcopy(source_manifest)
    manifest['upstream_provenance'] = {key: source_manifest[key] for key in
        ('source_asset', 'source_urdf_sha256', 'source_manifest_sha256', 'upstream_provenance')}
    manifest['source_asset'] = str(SOURCE_ROOT.relative_to(PROJECT_ROOT))
    manifest['source_urdf_sha256'] = SOURCE_URDF_SHA256
    manifest['source_manifest_sha256'] = SOURCE_MANIFEST_SHA256
    manifest['derived_sha256'] = file_sha256(candidate_urdf)
    manifest['generator_sha256'] = file_sha256(Path(__file__))
    manifest['visual_envelope_input_sha256'] = input_hashes
    manifest['visual_envelope_refinements'] = [record for _, _, record in generated]
    for _, hull, record in generated:
        proxy = next(item for item in manifest['proxy_records'] if item['link'] == 'base_link'
                     and item['local_shape'] == record['local_shape'])
        proxy.pop('support_error_m', None)
        proxy.update({'vertices': len(hull.vertices), 'geometry_reference': 'desktop_local_visual_with_retained_sources',
                      'max_reference_outside_m': record['final_hull_measurements']['max_reference_outside_m']})
    referenced_paths = sorted({(output_root / 'urdf' / node.get('filename')).resolve()
                               for node in robot.findall('.//mesh')})
    manifest['mesh_files'] = [{'path': str(path.relative_to(output_root.resolve())),
                               'sha256': file_sha256(path)} for path in referenced_paths]
    manifest['historical_records_note_zh'] += ' 本轮保留local_base_shape_map及local_base_proxy_replacements作为V6到r3历史；当前shape顺序不变，V8重建以visual_envelope_refinements为准。'
    manifest['grouping_note_zh'] = 'r3历史分组说明：' + source_manifest['grouping_note_zh'] + ' V8三处当前精度与覆盖以visual_envelope_refinements为准，历史3mm指标不适用于当前视觉拟合。'
    manifest['geometry_note_zh'] = '上游轮区历史说明：' + source_manifest['geometry_note_zh'] + ' V8另按本清单重建三处底盘视觉包络，不能将历史“底盘碰撞保持”表述解释为本轮未修改底盘。'
    manifest['repair_scope_zh'] = '从r3仅替换source8、source99及proxy13内source43的局部视觉包络；proxy13另外五源组件保留。目标外扩0.5mm、168 shapes，外观／关节／物性／其他collision不变。'
    manifest['asset_status_zh'] = '独立V8局部视觉包络候选；尚未通过几何与原生运行验收，未切换正式配置。'
    (output_root / 'README.md').write_text(
        '# V8局部视觉包络候选\n\n以r3为来源，仅重建三处已确认底盘局部，目标几何外扩0.5mm。'
        '全部来源、区域、实际误差和哈希见asset_manifest.json；几何／原生／策略对照通过前不得视为正式训练资产。\n')
    (output_root / 'asset_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return candidate_urdf


def main() -> None:
    """Generate one independent candidate using the approved fixed source recipe.

    Raises:
        FileExistsError: The requested candidate directory exists.
        ValueError: Source or geometric contracts cannot be satisfied.
        OSError: Source reading or candidate writing fails.

    Note:
        Exposes only an output-directory argument and prints the resulting URDF.
    """
    parser = argparse.ArgumentParser(description='生成独立V8局部视觉包络候选')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(build_candidate(args.output.resolve()))


if __name__ == '__main__':
    main()

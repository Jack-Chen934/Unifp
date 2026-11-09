"""Replace the evidenced V3 base proxy with eleven approved local convex groups."""

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

PROJECT_ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(PROJECT_ROOT / 'UniFP_v3/tools'))
from build_training_collision_asset import collision_points, compact_hull

SOURCE_ASSET = PROJECT_ROOT / 'UniFP_v3/resources/robots/v3_a1_desktop_train_v6_coplanar_r2'
DESKTOP_ASSET = PROJECT_ROOT / 'UniFP_v3/resources/robots/v3_a1_desktop'
DEFAULT_OUTPUT = PROJECT_ROOT / 'UniFP_v3/resources/robots/v3_a1_desktop_train_v7_local_base_proxy_r3'
URDF_NAME = 'a1arm_rgm3_desktop_fixed_gripper.urdf'
TARGET_SHAPES = (1,)
MAX_SUPPORT_ERROR_M = 0.003
COMPONENT_GROUPS = ((8,), (10,), (23,), (25,), (57,), (99,), (92, 118),
                    (26, 72), (9, 94), (42, 65, 40, 93, 114), (18, 91))


def group_hull(source_urdf: Path, source_collisions: list[ET.Element],
               component_indices: tuple[int, ...], output_root: Path,
               mesh_name: str) -> tuple[ET.Element, dict[str, Any]]:
    """Combine compact desktop part hulls into one approved link-local convex group.

    Args:
        source_urdf: Frozen desktop URDF used to resolve source mesh and origin.
        source_collisions: Original ordered base_link collision elements.
        component_indices: Desktop component indices included in this fixed group.
        output_root: New candidate root receiving the generated mesh file.
        mesh_name: Unique generated GLB basename.

    Returns:
        Collision element and group/part provenance in base_link-local meters.

    Raises:
        OSError: A source mesh cannot be read or the candidate mesh cannot be written.
        ValueError: A part's support approximation exceeds 3 mm or the final
            group exceeds 64 vertices. This support bound does not bound concavity.
    """
    part_vertices_m = []
    component_records = []
    for component_index in component_indices:
        source_collision = source_collisions[component_index]
        points_m = collision_points(source_urdf, source_collision)
        part_hull, support_error_m = compact_hull(points_m, MAX_SUPPORT_ERROR_M)
        if support_error_m > MAX_SUPPORT_ERROR_M:
            raise ValueError(f'Component {component_index}: support error exceeds 3 mm')
        # The validated fixture combined GLB part hulls. Reproduce that float32
        # serialization boundary while rebuilding directly from desktop sources.
        part_vertices_m.append(np.asarray(part_hull.vertices, dtype=np.float32))
        component_records.append({'source_component': component_index,
            'source_collision': source_collision.get('name'),
            'source_vertices': int(len(points_m)), 'hull_vertices': int(len(part_hull.vertices)),
            'support_error_m': support_error_m})
    hull = trimesh.convex.convex_hull(np.concatenate(part_vertices_m).astype(np.float64))
    if len(hull.vertices) > 64:
        raise ValueError(f'{mesh_name}: grouped hull exceeds 64 vertices')
    mesh_path = output_root / 'meshes' / mesh_name
    hull.export(mesh_path)
    collision_name = mesh_name.removesuffix('.glb')
    collision = ET.Element('collision', name=collision_name)
    ET.SubElement(collision, 'origin', xyz='0 0 0', rpy='0 0 0')
    geometry = ET.SubElement(collision, 'geometry')
    ET.SubElement(geometry, 'mesh', filename='../meshes/' + mesh_name)
    record = {'source_components': list(component_indices), 'mesh': mesh_name,
              'hull_vertices': int(len(hull.vertices)), 'components': component_records,
              'max_component_support_error_m': max(part['support_error_m'] for part in component_records)}
    return collision, record


def build_candidate(output_root: Path) -> Path:
    """Create a provenance-tracked candidate without modifying source assets.

    Args:
        output_root: New asset directory. It must not exist before the call.

    Returns:
        Candidate URDF path after replacing base local shape 1.

    Raises:
        FileExistsError: Candidate directory already exists.
        FileNotFoundError: An expected source URDF or mesh is absent.
        ValueError: Source identity, fixed grouping, hull or component coverage fails.

    Note:
        Visual, inertial, joint, non-target collision, controller and training files
        are not changed. The candidate is not registered as an active V3 asset.
    """
    source_urdf = SOURCE_ASSET / 'urdf' / URDF_NAME
    desktop_urdf = DESKTOP_ASSET / 'urdf' / URDF_NAME
    source_manifest_path = SOURCE_ASSET / 'asset_manifest.json'
    source_manifest = json.loads(source_manifest_path.read_text())
    if hashlib.sha256(source_urdf.read_bytes()).hexdigest() != source_manifest['derived_sha256']:
        raise ValueError('V6 source URDF identity does not match its manifest')
    output_root.mkdir(parents=True, exist_ok=False)
    shutil.copytree(SOURCE_ASSET, output_root, dirs_exist_ok=True)
    candidate_urdf = output_root / 'urdf' / URDF_NAME
    candidate_manifest_path = output_root / 'asset_manifest.json'
    robot = ET.parse(candidate_urdf).getroot()
    desktop_robot = ET.parse(desktop_urdf).getroot()
    desktop_base = desktop_robot.find("link[@name='base_link']")
    if desktop_base is None:
        raise ValueError('Desktop source has no base_link')
    desktop_collisions = desktop_base.findall('collision')
    active_base = robot.find("link[@name='base_link']")
    if active_base is None:
        raise ValueError('V6 source has no base_link')
    active_collisions = list(active_base.findall('collision'))
    target_records = {record['local_shape']: record for record in source_manifest['proxy_records']
                      if record['link'] == 'base_link' and record['local_shape'] in TARGET_SHAPES}
    if set(target_records) != set(TARGET_SHAPES):
        raise ValueError('Expected V6 base proxy 1 source mapping')
    shape_map: dict[int, list[int]] = {}
    new_elements: list[ET.Element] = []
    replacements = []
    for old_index, old_collision in enumerate(active_collisions):
        shape_map[old_index] = []
        if old_index not in target_records:
            shape_map[old_index].append(len(new_elements))
            new_elements.append(deepcopy(old_collision))
            continue
        component_indices = target_records[old_index]['source_components']
        if len(set(component_indices)) != len(component_indices):
            raise ValueError(f'base proxy {old_index} contains duplicate source components')
        grouped_components = [component for group in COMPONENT_GROUPS for component in group]
        if sorted(grouped_components) != sorted(component_indices):
            raise ValueError('Approved groups must cover the original 19 source components once')
        generated = []
        for group_index, group in enumerate(COMPONENT_GROUPS):
            new_index = len(new_elements)
            collision, record = group_hull(
                desktop_urdf, desktop_collisions, group, output_root,
                f'base_link_local_{old_index}_group_{group_index}.glb')
            new_elements.append(collision)
            shape_map[old_index].append(new_index)
            generated.append({'local_shape': new_index, **record})
        replacements.append({'old_shape': old_index, 'old_components': component_indices,
                             'new_shapes': shape_map[old_index], 'groups': generated})
    for collision in active_collisions:
        active_base.remove(collision)
    for collision in new_elements:
        active_base.append(collision)
    manifest = source_manifest
    base_counts = manifest['collision_solid_counts']['base_link']
    base_counts['count'] = len(new_elements)
    base_counts['source_groups'] = {
        group: [new for old in old_indices for new in shape_map[old]]
        for group, old_indices in base_counts['source_groups'].items()}
    new_proxies = []
    for record in manifest['proxy_records']:
        if record['link'] != 'base_link':
            new_proxies.append(record)
        elif record['local_shape'] not in TARGET_SHAPES:
            new_proxies.append({**record, 'local_shape': shape_map[record['local_shape']][0]})
    for replacement in replacements:
        for group in replacement['groups']:
            new_proxies.append({'link': 'base_link', 'local_shape': group['local_shape'],
                'source_group': 'base_link_collision_local_group',
                'source_components': group['source_components'],
                'vertices': group['hull_vertices'],
                'support_error_m': group['max_component_support_error_m']})
    manifest['proxy_records'] = new_proxies
    manifest['retained_collision_records'] = [
        {**record, 'local_shape': shape_map[record['local_shape']][0]}
        if record['link'] == 'base_link' else record
        for record in manifest['retained_collision_records']]
    manifest['visual_clearance_records'] = [
        {**record, 'local_shape': shape_map[record['local_shape']][0]}
        for record in manifest.get('visual_clearance_records', [])]
    referenced_meshes = {node.get('filename') for node in robot.findall('.//mesh')}
    stale_target_meshes = [output_root / 'urdf' / reference for reference in (
        f'../meshes/base_link_proxy_{shape}.glb' for shape in TARGET_SHAPES)]
    for stale_mesh in stale_target_meshes:
        if not any((output_root / 'urdf' / reference).resolve() == stale_mesh.resolve()
                   for reference in referenced_meshes):
            stale_mesh.unlink(missing_ok=True)
    ET.indent(robot, space='  ')
    ET.ElementTree(robot).write(candidate_urdf, encoding='utf-8', xml_declaration=True)
    manifest['asset_status'] = 'independent_local_base_proxy_repair_candidate'
    manifest['asset_status_zh'] = '独立局部碰撞修复候选；未切换配置，须完成几何及PhysX短测验收。'
    manifest['upstream_provenance'] = {
        key: source_manifest[key] for key in
        ('source_asset', 'source_urdf_sha256', 'source_manifest_sha256')}
    manifest['source_asset'] = str(SOURCE_ASSET.relative_to(PROJECT_ROOT))
    manifest['source_urdf_sha256'] = hashlib.sha256(source_urdf.read_bytes()).hexdigest()
    manifest['source_manifest_sha256'] = hashlib.sha256(source_manifest_path.read_bytes()).hexdigest()
    manifest['historical_records_note_zh'] = 'base_refinement_records及rolling_collision_records沿用V6来源清单中的历史变更记录，其old_shape/new_shapes属于当时资产；当前V6到本候选的索引以local_base_shape_map为准。upstream_provenance保留V6的上游身份。'
    manifest['derived_urdf'] = str(candidate_urdf.relative_to(output_root))
    manifest['derived_sha256'] = hashlib.sha256(candidate_urdf.read_bytes()).hexdigest()
    manifest['collision_shape_count'] = sum(entry['count'] for entry in manifest['collision_solid_counts'].values())
    manifest['local_base_proxy_replacements'] = replacements
    manifest['local_base_shape_map'] = {str(old): new for old, new in shape_map.items()}
    manifest['repair_scope_zh'] = '只将V6 base_link原shape 1的19个CAD部件完整保留为用户确认的11组局部凸包；所有shape索引引用同步重映射；其余URDF内容及训练配置保持V6 r2。'
    manifest['grouping_note_zh'] = '各部件支持平面近似误差不超过3mm；该指标不限制组内凹隙填补，须结合捕获姿态和原生PhysX回归。'
    manifest['generator_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest['mesh_files'] = [record for record in manifest['mesh_files']
        if (output_root / record['path']).is_file()]
    for mesh_path in sorted((output_root / 'meshes').glob('base_link_local_*.glb')):
        manifest['mesh_files'].append({'path': str(mesh_path.relative_to(output_root)),
                                       'sha256': hashlib.sha256(mesh_path.read_bytes()).hexdigest()})
    candidate_manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    (output_root / 'README.md').write_text(
        '# V3局部底盘代理修复候选\n\n'
        '从冻结V6 r2与桌面源重建，仅将base_link proxy 1的19个源组件组合为已确认的11组局部凸包；总shape为168。'
        '视觉、关节、惯性、其他collision和训练配置保持不变。未经独立短测验收，禁止正式训练接入。\n')
    return candidate_urdf


def main() -> None:
    """Generate a new asset directory and print its URDF path.

    Raises:
        FileExistsError: The candidate output directory already exists.
        OSError: Source files cannot be read or output files cannot be written.
        ValueError: V6 identities or component mapping are inconsistent.
    """
    parser = argparse.ArgumentParser(description='生成独立V3底盘局部碰撞代理候选')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(build_candidate(args.output))


if __name__ == '__main__':
    main()

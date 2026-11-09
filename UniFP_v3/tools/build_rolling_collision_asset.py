"""Build independently validated rolling collisions without altering robot geometry."""

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any, Dict, List
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

from build_desktop_asset import origin_transform
from build_training_collision_asset import collision_points, compact_hull

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / 'resources/robots/v3_a1_desktop_train_v1'
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / 'resources/robots/v3_a1_desktop_train_v5'
DEFAULT_COPLANAR_SOURCE_ROOT = DEFAULT_OUTPUT_ROOT
URDF_NAME = 'a1arm_rgm3_desktop_fixed_gripper.urdf'
TIRE_SECTORS = 16
VISUAL_TREAD_MIN_RADIUS_M = 0.08
MAX_COPLANAR_RADIUS_DELTA_M = 0.0023
WHEEL_CLEARANCE_COMPONENTS = {61: 'br', 71: 'bl'}


def load_link_visual(urdf_path: Path, visual: ET.Element) -> trimesh.Trimesh:
    """Load one URDF visual mesh in its link frame, in meters.

    Args:
        urdf_path: URDF path resolving the visual mesh filename.
        visual: Link visual element containing geometry and origin.

    Returns:
        A transformed triangle mesh in link-local coordinates.
    """
    reference = visual.find('geometry/mesh')
    mesh = trimesh.load(urdf_path.parent / reference.get('filename'), force='scene').to_geometry()
    mesh.apply_scale(np.fromstring(reference.get('scale', '1 1 1'), sep=' '))
    mesh.apply_transform(origin_transform(visual.find('origin')))
    return mesh


def rounded_tire_sectors(tire_points_m: np.ndarray) -> List[trimesh.Trimesh]:
    """Construct convex sectors using the original CAD tire's shoulder profile.

    Args:
        tire_points_m: Original tire vertices [V,3], in wheel-local meters.

    Returns:
        Sixteen convex meshes, each below 64 vertices, forming an 80-sided rolling
        envelope with nine CAD profile samples and the nominal 0.1016 m radius.
    """
    visual_tread_points_m = tire_points_m[
        np.linalg.norm(tire_points_m[:, [0, 2]], axis=1) > VISUAL_TREAD_MIN_RADIUS_M]
    axial_positions_m = np.linspace(visual_tread_points_m[:, 1].min(),
                                    visual_tread_points_m[:, 1].max(), 9)
    radial_distances_m = np.linalg.norm(tire_points_m[:, [0, 2]], axis=1)
    # Neighboring profile bands overlap slightly. Their outermost visual points
    # make the collision conservative while excluding hub surfaces below the tread.
    profile_half_width_m = float(np.ptp(visual_tread_points_m[:, 1]) / 16)
    radii_m = np.asarray([radial_distances_m[
        (np.abs(tire_points_m[:, 1] - axial_m) <= profile_half_width_m)
        & (radial_distances_m > VISUAL_TREAD_MIN_RADIUS_M)].max()
        for axial_m in axial_positions_m])
    sectors = []
    for sector in range(TIRE_SECTORS):
        angles_rad = np.linspace(2 * math.pi * sector / TIRE_SECTORS,
                                 2 * math.pi * (sector + 1) / TIRE_SECTORS, 6)
        vertices_m = [[radius * math.sin(angle), axial, -radius * math.cos(angle)]
                      for axial, radius in zip(axial_positions_m, radii_m) for angle in angles_rad]
        vertices_m.extend([[0, axial_positions_m[index], 0] for index in (0, -1)])
        sectors.append(trimesh.Trimesh(vertices=np.asarray(vertices_m), process=False).convex_hull)
    return sectors


def refine_wheel_clearances(robot: ET.Element, source_root: Path, output_root: Path,
                            manifest: Dict[str, Any]) -> None:
    """Split low wheel-region chassis proxies along their original CAD components.

    Args:
        robot: Candidate XML tree, updated only in base_link collision elements.
        source_root: Frozen train_v1 asset; its sibling desktop asset supplies CAD.
        output_root: New candidate root receiving compact upper-piece hulls.
        manifest: Candidate metadata updated with exact shape/group remapping.

    Raises:
        OSError: Referenced source CAD files cannot be read.

    Note:
        Retains individual low CAD pieces rather than filtering real collisions.
        Changes no visual, mass, inertia or joint XML and preserves all filter groups.
    """
    desktop_path = source_root.parent / 'v3_a1_desktop/urdf' / URDF_NAME
    desktop_base = ET.parse(desktop_path).getroot().find("link[@name='base_link']")
    desktop_collisions = desktop_base.findall('collision')
    base_visual_mesh = load_link_visual(desktop_path, desktop_base.find('visual'))
    base = robot.find("link[@name='base_link']")
    original_collisions = list(base.findall('collision'))
    proxies = {record['local_shape']: record for record in manifest['proxy_records'] if record['link'] == 'base_link'}
    retained = {record['local_shape']: record for record in manifest['retained_collision_records'] if record['link'] == 'base_link'}
    manifest['proxy_records'] = [record for record in manifest['proxy_records'] if record['link'] != 'base_link']
    manifest['retained_collision_records'] = [record for record in manifest['retained_collision_records'] if record['link'] != 'base_link']
    for collision in original_collisions:
        base.remove(collision)
    old_to_new = {}
    refinements = []
    for old_index, collision in enumerate(original_collisions):
        record = proxies.get(old_index, retained.get(old_index))
        components = record['source_components']
        source_parts = [collision_points(desktop_path, desktop_collisions[index]) for index in components]
        merged_m = np.concatenate(source_parts)
        in_wheel_region = abs((merged_m[:, 0].min() + merged_m[:, 0].max()) / 2) > 0.15
        lower = [index for index, points_m in zip(components, source_parts) if points_m[:, 2].min() < -0.11]
        split = old_index in proxies and len(components) > 1 and in_wheel_region and lower
        old_to_new[old_index] = []
        if not split:
            new_index = len(base.findall('collision'))
            base.append(deepcopy(collision))
            old_to_new[old_index].append(new_index)
            target_records = manifest['proxy_records'] if old_index in proxies else manifest['retained_collision_records']
            target_records.append({**record, 'local_shape': new_index})
            continue
        upper = [index for index in components if index not in lower]
        for group in ([upper] if upper else []) + [[index] for index in lower]:
            new_index = len(base.findall('collision'))
            old_to_new[old_index].append(new_index)
            if len(group) == 1:
                source_component = group[0]
                if source_component in WHEEL_CLEARANCE_COMPONENTS:
                    points_m = collision_points(desktop_path, desktop_collisions[source_component])
                    wheel_region = (points_m[:, 2] < -0.11) & (np.abs(points_m[:, 1]) > 0.07)
                    projected_points_m, projection_distances_m, _ = trimesh.proximity.closest_point(
                        base_visual_mesh, points_m[wheel_region])
                    points_m[wheel_region] = projected_points_m
                    mesh = trimesh.Trimesh(vertices=points_m, process=False).convex_hull
                    filename = f'base_link_visual_clearance_{new_index}.glb'
                    mesh.export(output_root / 'meshes' / filename)
                    element = ET.SubElement(base, 'collision', name=f'base_link_visual_clearance_{new_index}')
                    ET.SubElement(element, 'origin', xyz='0 0 0', rpy='0 0 0')
                    ET.SubElement(ET.SubElement(element, 'geometry'), 'mesh', filename='../meshes/' + filename)
                    manifest['proxy_records'].append({'link': 'base_link', 'local_shape': new_index,
                        'source_group': 'visual_wheel_clearance', 'source_components': group,
                        'vertices': len(mesh.vertices), 'support_error_m': 0.0})
                    manifest.setdefault('visual_clearance_records', []).append({
                        'wheel': WHEEL_CLEARANCE_COMPONENTS[source_component],
                        'source_component': source_component, 'local_shape': new_index,
                        'projected_vertices': int(wheel_region.sum()),
                        'max_projection_m': float(projection_distances_m.max())})
                else:
                    base.append(deepcopy(desktop_collisions[source_component]))
                    manifest['retained_collision_records'].append(
                        {'link': 'base_link', 'local_shape': new_index, 'source_components': group})
            else:
                points_m = np.concatenate([collision_points(desktop_path, desktop_collisions[index]) for index in group])
                mesh, error_m = compact_hull(points_m, 0.003)
                filename = f'base_link_clearance_{new_index}.glb'
                mesh.export(output_root / 'meshes' / filename)
                element = ET.SubElement(base, 'collision', name=f'base_link_clearance_{new_index}')
                ET.SubElement(element, 'origin', xyz='0 0 0', rpy='0 0 0')
                ET.SubElement(ET.SubElement(element, 'geometry'), 'mesh', filename='../meshes/' + filename)
                manifest['proxy_records'].append({'link': 'base_link', 'local_shape': new_index,
                    'source_group': record['source_group'], 'source_components': group,
                    'vertices': len(mesh.vertices), 'support_error_m': error_m})
        refinements.append({'old_shape': old_index, 'source_components': components,
                            'new_shapes': old_to_new[old_index]})
    groups = manifest['collision_solid_counts']['base_link']['source_groups']
    manifest['collision_solid_counts']['base_link'] = {
        'count': len(base.findall('collision')),
        'source_groups': {name: [new for old in indices for new in old_to_new[old]] for name, indices in groups.items()}}
    manifest['base_refinement_records'] = refinements


def build_rolling_collision_asset(source_root: Path, output_root: Path) -> Path:
    """Build an independent round-shouldered tire and locally refined chassis asset.

    Args:
        source_root: Frozen train_v1 asset beside its original desktop CAD asset.
        output_root: New independent candidate directory; must not already exist.

    Returns:
        Candidate URDF; real GPU import, loaded motion and regression remain required.

    Raises:
        FileExistsError: The output directory already exists.
        ValueError: A reviewed wheel tire mapping is absent or ambiguous.
        OSError: Source or candidate files cannot be accessed.

    Note:
        Writes only output_root. Preserves all visual, inertial and joint XML;
        non-tire collisions change only by separating evidenced wheel-region proxies.
    """
    source_path = source_root / 'urdf' / URDF_NAME
    robot = ET.parse(source_path).getroot()
    manifest = json.loads((source_root / 'asset_manifest.json').read_text())
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / 'urdf').mkdir()
    (output_root / 'meshes').mkdir()
    refine_wheel_clearances(robot, source_root, output_root, manifest)
    rolling_records = []
    for corner in ('fl', 'fr', 'bl', 'br'):
        link_name = corner + '_Link'
        link = robot.find(f"link[@name='{link_name}']")
        tires = [collision for collision in link.findall('collision')
                 if collision.find('geometry/mesh') is not None
                 and collision.find('geometry/mesh').get('filename').endswith('_solid_4.glb')]
        if len(tires) != 1:
            raise ValueError(f'Expected one retained CAD tire for {link_name}')
        visual_points_m = collision_points(source_path, link.find('visual'))
        visual_tread_mask = np.linalg.norm(visual_points_m[:, [0, 2]], axis=1) > VISUAL_TREAD_MIN_RADIUS_M
        tire_points_m = visual_points_m[visual_tread_mask]
        width_m = float(np.ptp(tire_points_m[:, 1]))
        center_y_m = float((tire_points_m[:, 1].max() + tire_points_m[:, 1].min()) / 2)
        old_tire_index = list(link.findall('collision')).index(tires[0])
        link.remove(tires[0])
        shape_indices = []
        for sector, mesh in enumerate(rounded_tire_sectors(tire_points_m)):
            filename = f'{corner}_rolling_tire_{sector}.glb'
            mesh.export(output_root / 'meshes' / filename)
            shape_indices.append(len(link.findall('collision')))
            collision = ET.SubElement(link, 'collision', name=f'{corner}_rolling_tire_{sector}')
            ET.SubElement(collision, 'origin', xyz='0 0 0', rpy='0 0 0')
            ET.SubElement(ET.SubElement(collision, 'geometry'), 'mesh', filename='../meshes/' + filename)
        # train_v1 has one recessed hub followed by its tire; filter groups cover both.
        manifest['retained_collision_records'] = [record for record in manifest['retained_collision_records']
            if not (record['link'] == link_name and record['local_shape'] == old_tire_index)]
        group_name = next(iter(manifest['collision_solid_counts'][link_name]['source_groups']))
        count = len(link.findall('collision'))
        manifest['collision_solid_counts'][link_name] = {'count': count, 'source_groups': {group_name: list(range(count))}}
        rolling_records.append({'link': link_name, 'shape_indices': shape_indices,
                                'radius_m': float(np.linalg.norm(tire_points_m[:, [0, 2]], axis=1).max()),
                                'width_m': width_m, 'center_y_m': center_y_m,
                                'angular_segments': TIRE_SECTORS * 5, 'axial_profile_samples': 9})
    mesh_files = []
    for reference in sorted({mesh.get('filename') for mesh in robot.findall('.//mesh')}):
        destination = (output_root / 'urdf' / reference).resolve()
        if not destination.exists():
            source_mesh = (source_path.parent / reference).resolve()
            if not source_mesh.exists():
                source_mesh = (source_root.parent / 'v3_a1_desktop/urdf' / reference).resolve()
            shutil.copyfile(source_mesh, destination)
        mesh_files.append({'path': str(destination.relative_to(output_root.resolve())),
                           'sha256': hashlib.sha256(destination.read_bytes()).hexdigest()})
    output_path = output_root / 'urdf' / URDF_NAME
    ET.indent(robot, space='  ')
    ET.ElementTree(robot).write(output_path, encoding='utf-8', xml_declaration=True)
    manifest.update({'asset_status': 'independent_rolling_collision_candidate',
        'asset_status_zh': '独立滚动碰撞候选；尚需带载运动和整体回归。',
        'source_asset': str(source_root), 'source_urdf_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
        'source_manifest_sha256': hashlib.sha256((source_root / 'asset_manifest.json').read_bytes()).hexdigest(),
        'derived_urdf': str(output_path.relative_to(output_root)),
        'derived_sha256': hashlib.sha256(output_path.read_bytes()).hexdigest(),
        'collision_shape_count': sum(record['count'] for record in manifest['collision_solid_counts'].values()),
        'rolling_collision_records': rolling_records, 'mesh_files': mesh_files,
        'geometry_note_zh': '胎面轴向包络从对应visual径向大于80mm的顶点采样，最大半径／宽度／轮心记录在manifest；仅后轮接触来源组件61、71的低位轮区碰撞顶点投影回base visual表面。视觉／关节／物性不变，需真实GPU验证。'})
    (output_root / 'asset_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    (output_root / 'README.md').write_text('# V3独立滚动碰撞候选\n\n'
        '四胎碰撞从桌面visual胎面包络生成，后轮局部碰撞外扩投影回base visual表面。视觉、质量、惯量及关节不变。'
        '物性仍为仿真估算，非实机标定。真实带载与整体回归通过前不可用于正式训练。\n')
    return output_path


def build_coplanar_rolling_asset(source_root: Path, output_root: Path) -> Path:
    """Copy a rolling asset and compensate tire collision radii for coplanar support.

    Args:
        source_root: Validated rolling candidate containing the URDF and manifest.
        output_root: New independent output directory; must not already exist.

    Returns:
        The copied URDF path after updating only the four tires' collision sectors.

    Raises:
        FileExistsError: The output directory already exists.
        ValueError: A wheel mapping is ambiguous or compensation exceeds 2.3 mm.
        OSError: Source files cannot be read or output files cannot be written.

    Note:
        Wheel geometry is evaluated at zero steering and drive angles. Visual,
        joint, inertial and chassis collision data are copied without changes.
    """
    from compare_desktop_fk import forward_kinematics

    source_path = source_root / 'urdf' / URDF_NAME
    output_root.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source_root, output_root, dirs_exist_ok=True)
    output_path = output_root / 'urdf' / URDF_NAME
    robot = ET.parse(source_path).getroot()
    manifest_path = source_root / 'asset_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    body_poses = forward_kinematics(source_path, {}, np.eye(4))
    wheel_names = ('fl_Link', 'fr_Link', 'bl_Link', 'br_Link')
    bottom_points_world_m = []
    bottom_radial_world_z = []
    visual_bottom_points_world_m = []
    wheel_mesh_paths = {}

    # Gather the actual support points from the imported sector meshes, not the
    # visual maxima, so the fit describes the collision geometry PhysX receives.
    for link_name in wheel_names:
        link = robot.find(f"link[@name='{link_name}']")
        visual_mesh = load_link_visual(source_path, link.find('visual'))
        visual_tread = visual_mesh.vertices[
            np.linalg.norm(visual_mesh.vertices[:, [0, 2]], axis=1)
            > VISUAL_TREAD_MIN_RADIUS_M]
        visual_bottom = visual_tread[int(np.argmin(visual_tread[:, 2]))]
        visual_bottom_points_world_m.append(
            (body_poses[link_name] @ np.r_[visual_bottom, 1.0])[:3])
        collisions = [element for element in link.findall('collision')
                      if element.get('name', '').startswith(link_name[:2] + '_rolling_tire_')]
        if len(collisions) != TIRE_SECTORS:
            raise ValueError(f'Expected {TIRE_SECTORS} rolling sectors for {link_name}')
        points_by_sector = []
        mesh_paths = []
        for collision in collisions:
            mesh_element = collision.find('geometry/mesh')
            mesh_path = (output_path.parent / mesh_element.get('filename')).resolve()
            mesh = trimesh.load(mesh_path, force='mesh')
            points_by_sector.append(np.asarray(mesh.vertices))
            mesh_paths.append(mesh_path)
        wheel_mesh_paths[link_name] = mesh_paths
        wheel_points_m = np.concatenate(points_by_sector)
        bottom_index = int(np.argmin(wheel_points_m[:, 2]))
        local_bottom_m = wheel_points_m[bottom_index]
        bottom_points_world_m.append((body_poses[link_name] @ np.r_[local_bottom_m, 1.0])[:3])
        radial_local = local_bottom_m[[0, 2]] / np.linalg.norm(local_bottom_m[[0, 2]])
        radial_world = body_poses[link_name][:3, :3] @ np.array([radial_local[0], 0.0, radial_local[1]])
        bottom_radial_world_z.append(float(radial_world[2]))

    # The least-squares plane is the support plane minimizing the four bottom
    # height residuals; radial deltas convert each residual through wheel tilt.
    bottom_points_world_m = np.asarray(bottom_points_world_m)
    plane_matrix = np.column_stack((bottom_points_world_m[:, 0],
                                    bottom_points_world_m[:, 1],
                                    np.ones(len(wheel_names))))
    plane_coefficients = np.linalg.lstsq(
        plane_matrix, bottom_points_world_m[:, 2], rcond=None
    )[0]
    plane_height_m = plane_matrix @ plane_coefficients
    residual_m = bottom_points_world_m[:, 2] - plane_height_m
    visual_bottom_points_world_m = np.asarray(visual_bottom_points_world_m)
    visual_plane_matrix = np.column_stack((visual_bottom_points_world_m[:, 0],
                                           visual_bottom_points_world_m[:, 1],
                                           np.ones(len(wheel_names))))
    visual_plane_height_m = visual_plane_matrix @ np.linalg.lstsq(
        visual_plane_matrix, visual_bottom_points_world_m[:, 2], rcond=None
    )[0]
    visual_residual_m = visual_bottom_points_world_m[:, 2] - visual_plane_height_m
    radial_delta_m = -residual_m / np.asarray(bottom_radial_world_z)
    radial_delta_m[np.abs(radial_delta_m) < 1e-6] = 0.0
    if np.max(np.abs(radial_delta_m)) > MAX_COPLANAR_RADIUS_DELTA_M:
        raise ValueError(
            f'Coplanar tire compensation exceeds 2.3 mm: {radial_delta_m.tolist()}')

    adjustments = []
    for index, link_name in enumerate(wheel_names):
        original_radius_m = next(
            record['radius_m'] for record in manifest['rolling_collision_records']
            if record['link'] == link_name)
        radius_delta_m = float(radial_delta_m[index])
        for mesh_path in wheel_mesh_paths[link_name]:
            mesh = trimesh.load(mesh_path, force='mesh')
            vertices_m = np.asarray(mesh.vertices).copy()
            radial = vertices_m[:, [0, 2]]
            radial_norm_m = np.linalg.norm(radial, axis=1)
            tread_vertices = radial_norm_m > 1e-9
            vertices_m[tread_vertices, 0] += (
                radius_delta_m * radial[tread_vertices, 0] / radial_norm_m[tread_vertices])
            vertices_m[tread_vertices, 2] += (
                radius_delta_m * radial[tread_vertices, 1] / radial_norm_m[tread_vertices])
            adjusted_mesh = trimesh.Trimesh(vertices=vertices_m, faces=mesh.faces, process=False)
            adjusted_mesh.export(mesh_path)
        candidate_radius_m = original_radius_m + radius_delta_m
        manifest_record = next(
            record for record in manifest['rolling_collision_records']
            if record['link'] == link_name)
        manifest_record['source_radius_m'] = original_radius_m
        manifest_record['radius_compensation_m'] = radius_delta_m
        manifest_record['radius_m'] = candidate_radius_m
        manifest_record['support_residual_before_m'] = float(residual_m[index])
        manifest_record['support_residual_after_m'] = float(
            residual_m[index] + radius_delta_m * bottom_radial_world_z[index])
        manifest_record['visual_radius_difference_m'] = float(
            candidate_radius_m - original_radius_m)
        adjustments.append({
            'link': link_name,
            'source_radius_m': original_radius_m,
            'candidate_radius_m': candidate_radius_m,
            'radius_compensation_m': radius_delta_m,
            'support_residual_before_m': float(residual_m[index]),
            'support_residual_after_m': manifest_record['support_residual_after_m'],
        })

    mesh_records = []
    for record in manifest['mesh_files']:
        mesh_path = output_root / record['path']
        record['sha256'] = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
        mesh_records.append(record)
    manifest.update({
        'asset_status': 'independent_coplanar_rolling_collision_candidate',
        'asset_status_zh': '独立共面滚动碰撞候选；尚需静置、带载运动和整体回归。',
        'source_asset': str(source_root),
        'source_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'source_urdf_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
        'support_plane_world_z_ax_by_c_m': plane_coefficients.tolist(),
        'visual_support_residual_peak_to_peak_m': float(np.ptp(visual_residual_m)),
        'support_residual_peak_to_peak_before_m': float(np.ptp(residual_m)),
        'support_residual_peak_to_peak_after_m': float(np.ptp(
            [record['support_residual_after_m'] for record in adjustments])),
        'rolling_radius_adjustments': adjustments,
        'mesh_files': mesh_records,
        'geometry_note_zh': (
            '由train_v5独立复制，仅对四轮滚动碰撞胎面做最佳支撑平面径向补偿，'
            '单轮变化不超过2.3mm。visual、关节、物性、底盘碰撞及14DOF保持不变；'
            '该毫米级碰撞包络差仍需GPU PhysX动态验收。'),
    })
    (output_root / 'asset_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    (output_root / 'README.md').write_text(
        '# V3独立共面滚动碰撞候选\n\n'
        '从train_v5独立复制，仅对四轮滚动碰撞胎面作不超过2.3mm的径向补偿。'
        '视觉、关节、物性、底盘碰撞与正式训练配置不变。'
        '须通过静置、正反偏航及整体回归后才能评估训练准入。\n')
    return output_path


def main() -> None:
    """Parse paths and generate a new candidate without overwriting source assets.

    Returns:
        None.

    Raises:
        FileExistsError: The requested output directory already exists.

    Note:
        Writes only the independent output directory; never starts simulation.
    """
    parser = argparse.ArgumentParser(description='生成V3圆滑肩部滚动碰撞独立候选')
    parser.add_argument('--source-root', type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--coplanar', action='store_true')
    args = parser.parse_args()
    if args.coplanar:
        print(build_coplanar_rolling_asset(args.source_root, args.output_root))
    else:
        print(build_rolling_collision_asset(args.source_root, args.output_root))


if __name__ == '__main__':
    main()

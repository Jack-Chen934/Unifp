"""Build the desktop geometry with the approved V3 simulation properties."""

import argparse
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import posixpath
from typing import Any, Dict
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DESKTOP_ARCHIVE = Path('/home/sorry/桌面/a1arm_description.zip')
SOURCE_MEMBER = 'robot_description/a1arm_description/urdf/a1arm_rgm3.urdf'
REFERENCE_URDF = PROJECT_ROOT / 'UniFP_v3/resources/robots/v3_a1/urdf/a1arm_rgm3_v3_fixed_gripper.urdf'
OUTPUT_ROOT = PROJECT_ROOT / 'UniFP_v3/resources/robots/v3_a1_desktop'
CORNERS = ('fl', 'fr', 'bl', 'br')


def origin_transform(origin: ET.Element) -> np.ndarray:
    """Convert a URDF origin to a homogeneous parent-to-child pose.

    Args:
        origin: Origin element; absent xyz/rpy fields default to zero m/rad.

    Returns:
        A 4-by-4 pose matrix with a translation in meters.
    """
    pose = np.eye(4)
    pose[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
    pose[:3, :3] = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
    return pose


def write_origin(parent: ET.Element, pose: np.ndarray) -> None:
    """Append a URDF origin representing a parent-relative rigid pose.

    Args:
        parent: XML element receiving the origin as a child.
        pose: 4-by-4 pose with translation in meters.

    Returns:
        None.

    Note:
        Mutates the supplied XML parent.
    """
    angles = Rotation.from_matrix(pose[:3, :3]).as_euler('xyz')
    ET.SubElement(parent, 'origin', xyz=' '.join(f'{v:.12g}' for v in pose[:3, 3]),
                  rpy=' '.join(f'{v:.12g}' for v in angles))


def load_archive_mesh(archive: zipfile.ZipFile, geometry: ET.Element) -> trimesh.Trimesh:
    """Read a referenced GLB including its scene-node transforms.

    Args:
        archive: Open desktop source ZIP.
        geometry: URDF visual or collision element containing geometry/mesh.

    Returns:
        Source triangle mesh in the GLB scene frame, in meters.

    Raises:
        KeyError: A referenced mesh is absent from the source archive.
    """
    reference = geometry.find('geometry/mesh').get('filename')
    member = posixpath.normpath(posixpath.join(posixpath.dirname(SOURCE_MEMBER), reference))
    scene = trimesh.load(io.BytesIO(archive.read(member)), file_type='glb', force='scene')
    mesh = scene.to_geometry()
    mesh.apply_scale(np.fromstring(geometry.find('geometry/mesh').get('scale', '1 1 1'), sep=' '))
    return mesh


def tire_frame(mesh: trimesh.Trimesh) -> np.ndarray:
    """Recover the rolling center and axle from a source wheel visual mesh.

    Args:
        mesh: Entire wheel assembly in the base_link frame, in meters.

    Returns:
        Base-relative wheel pose; local y is the axle and local z points up.

    Raises:
        ValueError: The source does not contain one identifiable 0.2 m tire.
    """
    welded = mesh.copy()
    welded.merge_vertices()
    # The tread is the only connected component spanning the 0.2 m diameter.
    tires = [part for part in welded.split(only_watertight=False)
             if 0.195 < part.extents[0] < 0.205 and 0.195 < part.extents[2] < 0.205]
    if len(tires) != 1:
        raise ValueError(f'Expected one tire, found {len(tires)}')
    tire = tires[0]
    # Unique surface vertices prevent unwelded CAD triangle duplication bias.
    _, eigenvectors = np.linalg.eigh(np.cov(tire.vertices.T))
    approximate_axle = eigenvectors[:, 0]
    # Sidewall normals measure the actual axle; PCA alone is biased by
    # asymmetric tread tessellation, notably on the right-hand wheel mesh.
    normals = tire.face_normals
    alignment = normals @ approximate_axle
    sidewalls = np.abs(alignment) > 0.999
    axle = np.median(normals[sidewalls] * np.sign(alignment[sidewalls, None]), axis=0)
    axle /= np.linalg.norm(axle)
    if axle[1] < 0:
        axle = -axle
    forward_axis = np.cross(axle, [0.0, 0.0, 1.0])
    forward_axis /= np.linalg.norm(forward_axis)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack((forward_axis, axle, np.cross(forward_axis, axle)))
    pose[:3, 3] = tire.bounds.mean(axis=0)
    return pose


def steering_frame(mesh: trimesh.Trimesh, wheel_pose: np.ndarray) -> np.ndarray:
    """Recover the steering-bearing center from flat circular CAD components.

    Args:
        mesh: Complete wheel visual assembly in base coordinates, meters.
        wheel_pose: Base-relative rolling center used to identify upper parts.

    Returns:
        Base-relative steering pose, with its local z along the vertical bearing.

    Raises:
        ValueError: No upper flat circular bearing is present in the source.
    """
    welded = mesh.copy()
    welded.merge_vertices()
    bearings = [part for part in welded.split(only_watertight=False)
                if 0.05 < part.extents[0] < 0.08 and 0.05 < part.extents[1] < 0.08
                and abs(part.extents[0] - part.extents[1]) < 0.001
                and part.extents[2] < 0.03
                and part.bounds.mean(axis=0)[2] > wheel_pose[2, 3] + 0.08]
    if not bearings:
        raise ValueError('No source steering-bearing geometry found')
    bearing = max(bearings, key=lambda part: part.extents[0] * part.extents[1])
    pose = np.eye(4)
    pose[:3, 3] = bearing.bounds.mean(axis=0)
    return pose


def split_assembly(mesh: trimesh.Trimesh, wheel_pose: np.ndarray) -> Dict[str, np.ndarray]:
    """Assign intact CAD components to rolling, steering and fixed bodies.

    Args:
        mesh: Assembly triangle mesh in base coordinates, meters.
        wheel_pose: Wheel center and axle pose in base coordinates.

    Returns:
        Mapping from motion group to original triangle indices.

    Raises:
        ValueError: The partition duplicates or loses source triangles.
    """
    welded = mesh.copy()
    welded.merge_vertices()
    components = trimesh.graph.connected_components(
        welded.face_adjacency, nodes=np.arange(len(mesh.faces)), min_len=1)
    groups = {'rolling': [], 'steering': [], 'fixed': []}
    inverse_wheel = np.linalg.inv(wheel_pose)
    for face_indices in components:
        points = trimesh.transform_points(mesh.vertices[mesh.faces[face_indices]].reshape(-1, 3), inverse_wheel)
        radius = np.linalg.norm(points[:, [0, 2]], axis=1).max()
        # Include tire/hub pieces together; upper suspension stays on the base.
        if radius <= 0.101 and np.abs(points[:, 1]).max() <= 0.074:
            group = 'rolling'
        elif points[:, 2].min() >= 0.09:
            group = 'fixed'
        else:
            group = 'steering'
        groups[group].extend(face_indices)
    result = {name: np.asarray(indices, dtype=int) for name, indices in groups.items()}
    if not np.array_equal(np.sort(np.concatenate(list(result.values()))), np.arange(len(mesh.faces))):
        raise ValueError('Source face partition is incomplete')
    return result


def collision_component_groups(mesh: trimesh.Trimesh, reference_visual: trimesh.Trimesh,
                               visual_groups: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Assign each intact collision component to its nearest visual motion group.

    Args:
        mesh: Simplified wheel collision mesh in base coordinates, meters.
        reference_visual: Full wheel visual mesh in base coordinates, meters.
        visual_groups: Source face indices grouped by rolling, steering or fixed motion.

    Returns:
        Mapping from motion group to collision face indices; connected components
        remain intact within each group.
    """
    welded = mesh.copy()
    welded.merge_vertices()
    components = trimesh.graph.connected_components(
        welded.face_adjacency, nodes=np.arange(len(mesh.faces)), min_len=1)
    visual_labels = np.empty(len(reference_visual.faces), dtype=np.int32)
    group_names = list(visual_groups)
    for index, name in enumerate(group_names):
        visual_labels[visual_groups[name]] = index

    nearest_visual_faces = cKDTree(reference_visual.triangles_center).query(mesh.triangles_center)[1]
    grouped_faces = {name: [] for name in group_names}
    for component_faces in components:
        # A whole collision shell may bridge a CAD seam; assign it by its
        # nearest source surface majority instead of cutting it triangle by triangle.
        group_index = np.bincount(
            visual_labels[nearest_visual_faces[component_faces]], minlength=len(group_names)).argmax()
        grouped_faces[group_names[group_index]].extend(component_faces)
    return {name: np.asarray(indices, dtype=int) for name, indices in grouped_faces.items()}


def export_groups(mesh: trimesh.Trimesh, frames: Dict[str, np.ndarray], corner: str,
                  kind: str, output: Path, links: Dict[str, ET.Element],
                  reference_visual: trimesh.Trimesh) -> Dict[str, int]:
    """Export source triangles in each moving body's local coordinate frame.

    Args:
        mesh: Source geometry already transformed to base coordinates, meters.
        frames: Base-relative pose for each fixed/steering/rolling group.
        corner: Wheel identifier fl, fr, bl or br.
        kind: URDF visual or collision tag.
        output: Candidate asset directory receiving generated GLB files.
        links: Mapping from motion groups to XML links to update.
        reference_visual: Source wheel visual used to associate independently
            simplified collision faces with their real moving CAD parts.

    Returns:
        Triangle counts in each exported group.

    Note:
        Writes GLB files and appends XML geometry elements; no registration
        translation or geometric scaling is applied to source world geometry.
    """
    visual_groups = split_assembly(reference_visual, frames['rolling'])
    groups = visual_groups
    if kind == 'collision':
        groups = collision_component_groups(mesh, reference_visual, visual_groups)
    counts = {}
    for group, face_indices in groups.items():
        counts[group] = len(face_indices)
        if not len(face_indices):
            continue
        part = mesh.submesh([face_indices], append=True, repair=False)
        part.apply_transform(np.linalg.inv(frames[group]))
        filename = f'{corner}_{group}_{kind}.glb'
        part.export(output / 'meshes' / filename)
        element = ET.SubElement(links[group], kind, name=f'{corner}_{group}_{kind}')
        write_origin(element, np.eye(4))
        ET.SubElement(ET.SubElement(element, 'geometry'), 'mesh', filename='../meshes/' + filename)
        if kind == 'visual':
            ET.SubElement(element, 'material', name='white')
    return counts


def separate_collision_parts(robot: ET.Element, output: Path) -> Dict[str, Any]:
    """Expose disconnected source solids as separate engine convex shapes.

    Args:
        robot: Candidate robot with mesh collision definitions.
        output: Asset root containing the referenced collision meshes.

    Returns:
        Counts and source collision group indices by link; geometry stays in
        its original frame so filters can identify exact mounting solids.

    Note:
        Replaces collision XML only and writes per-solid GLBs. Visuals, mass,
        COM, joints and world geometry do not change.
    """
    counts = {}
    for link in robot.findall('link'):
        count = 0
        omitted_components = 0
        omitted_faces = 0
        sources = {}
        for collision in list(link.findall('collision')):
            first_index = count
            mesh_reference = collision.find('geometry/mesh')
            if mesh_reference is None:
                count += 1
                sources[collision.get('name')] = [first_index]
                continue
            mesh_path = output / 'urdf' / mesh_reference.get('filename')
            mesh = trimesh.load(mesh_path, force='scene').to_geometry()
            welded = mesh.copy()
            welded.merge_vertices()
            components = trimesh.graph.connected_components(
                welded.face_adjacency, nodes=np.arange(len(mesh.faces)), min_len=1)
            link.remove(collision)
            for face_indices in components:
                part = mesh.submesh([face_indices], append=True, repair=False)
                unique_vertices = np.unique(part.faces)
                singular_values = np.linalg.svd(part.vertices[unique_vertices] - part.vertices[unique_vertices].mean(axis=0),
                                                compute_uv=False)
                rank = int(np.count_nonzero(singular_values > singular_values[0] * 1e-8))
                if len(unique_vertices) < 4 or rank < 3:
                    # Zero-volume CAD scraps cannot contribute a valid contact hull.
                    omitted_components += 1
                    omitted_faces += len(face_indices)
                    continue
                filename = f'{link.get("name")}_solid_{count}.glb'
                part.export(output / 'meshes' / filename)
                element = deepcopy(collision)
                element.set('name', f'{link.get("name")}_solid_{count}')
                element.find('geometry/mesh').set('filename', '../meshes/' + filename)
                link.append(element)
                count += 1
            sources[collision.get('name', mesh_path.stem)] = list(range(first_index, count))
        counts[link.get('name')] = {
            'count': count,
            'source_groups': sources,
            'omitted_zero_volume_components': omitted_components,
            'omitted_zero_volume_faces': omitted_faces,
        }
    return counts


def build_asset(archive_path: Path = DESKTOP_ARCHIVE, output: Path = OUTPUT_ROOT,
                split_collisions: bool = True) -> Path:
    """Generate an independent desktop-based simulation asset and manifest.

    Args:
        archive_path: Read-only source model ZIP; defaults to the desktop file.
        output: Candidate asset directory, separate from all existing assets.
        split_collisions: Whether to expose individual source CAD collision
            solids to the importer instead of enclosing an assembly in one hull.

    Returns:
        Path to the generated 14-DOF URDF.

    Raises:
        ValueError: Tire recovery or triangle partition is ambiguous.
        KeyError: A source mesh or required XML link is absent.

    Note:
        Writes only under output. Desktop ZIP, V3 reference and official code
        remain unchanged. Physical values are simulation estimates, not measured.
    """
    (output / 'meshes').mkdir(parents=True, exist_ok=True)
    (output / 'urdf').mkdir(exist_ok=True)
    reference = ET.parse(REFERENCE_URDF).getroot()
    with zipfile.ZipFile(archive_path) as archive:
        source_bytes = archive.read(SOURCE_MEMBER)
        robot = ET.fromstring(source_bytes)
        source = deepcopy(robot)
        links = {link.get('name'): link for link in robot.findall('link')}
        # Copy source references as regular files; keep non-wheel visual XML intact.
        for mesh in robot.findall('.//mesh'):
            name = mesh.get('filename')
            member = posixpath.normpath(posixpath.join(posixpath.dirname(SOURCE_MEMBER), name))
            (output / 'meshes' / Path(name).name).write_bytes(archive.read(member))
        for joint in list(robot.findall('joint')):
            if joint.get('name').startswith('root_') or joint.get('name') in [c + '_joint' for c in CORNERS]:
                robot.remove(joint)
            elif joint.get('name') in ('gripper1_axis', 'gripper2_axis'):
                joint.set('type', 'fixed')
                for field in ('axis', 'limit', 'mimic'):
                    node = joint.find(field)
                    if node is not None:
                        joint.remove(node)
        for name, link in links.items():
            if name.startswith('root_link'):
                robot.remove(link)
        base = links['base_link']
        base.remove(base.find('inertial'))
        base.append(deepcopy(reference.find("link[@name='base_link']/inertial")))
        wheel_audit = {}
        for corner in CORNERS:
            wheel = links[corner + '_Link']
            source_joint = source.find(f"joint[@name='{corner}_joint']/origin")
            joint_pose = origin_transform(source_joint)
            visual = wheel.find('visual')
            world_visual = load_archive_mesh(archive, visual)
            world_visual.apply_transform(joint_pose @ origin_transform(visual.find('origin')))
            rolling_pose = tire_frame(world_visual)
            steering_pose = steering_frame(world_visual, rolling_pose)
            steering = deepcopy(reference.find(f"link[@name='{corner}_steering_link']"))
            for element in list(steering.findall('visual')):
                steering.remove(element)
            robot.append(steering)
            for name, parent, child, pose in (
                (corner + '_steering_joint', 'base_link', corner + '_steering_link', steering_pose),
                (corner + '_wheel_drive_joint', corner + '_steering_link', corner + '_Link', np.linalg.inv(steering_pose) @ rolling_pose),
            ):
                joint = deepcopy(reference.find(f"joint[@name='{name}']"))
                joint.find('parent').set('link', parent)
                joint.find('child').set('link', child)
                joint.remove(joint.find('origin'))
                write_origin(joint, pose)
                robot.append(joint)
            frames = {'fixed': np.eye(4), 'steering': steering_pose, 'rolling': rolling_pose}
            targets = {'fixed': base, 'steering': steering, 'rolling': wheel}
            counts = {}
            for kind in ('visual', 'collision'):
                element = wheel.find(kind)
                geometry = world_visual if kind == 'visual' else load_archive_mesh(archive, element)
                if kind == 'collision':
                    geometry.apply_transform(joint_pose @ origin_transform(element.find('origin')))
                wheel.remove(element)
                counts[kind] = export_groups(geometry, frames, corner, kind, output, targets, world_visual)
            wheel.remove(wheel.find('inertial'))
            wheel.append(deepcopy(reference.find(f"link[@name='{corner}_Link']/inertial")))
            wheel_audit[corner] = {'steering_origin_m': steering_pose[:3, 3].tolist(),
                                   'wheel_center_m': rolling_pose[:3, 3].tolist(),
                                   'wheel_axis_in_base': rolling_pose[:3, 1].tolist(), 'faces': counts}
        collision_parts = separate_collision_parts(robot, output) if split_collisions else {}
        robot.set('name', 'v3_a1_desktop')
        urdf_path = output / 'urdf/a1arm_rgm3_desktop_fixed_gripper.urdf'
        ET.indent(robot, space='  ')
        ET.ElementTree(robot).write(urdf_path, encoding='utf-8', xml_declaration=True)
        referenced_meshes = {
            (output / 'urdf' / mesh.get('filename')).resolve()
            for mesh in robot.findall('.//mesh')
        }
        manifest: Dict[str, Any] = {
            'source_zip': str(archive_path), 'source_zip_sha256': hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            'source_urdf_member': SOURCE_MEMBER, 'source_urdf_sha256': hashlib.sha256(source_bytes).hexdigest(),
            'derived_urdf': str(urdf_path.relative_to(output)),
            'derived_sha256': hashlib.sha256(urdf_path.read_bytes()).hexdigest(),
            '物理来源': '采用用户确认的V3仿真估算；非实测，非厂家逐部件惯性标定。',
            '几何约束': '桌面模型零位世界几何保留，轮组只分配运动归属，不施加旧V3配准平移。',
            'wheel_geometry': wheel_audit, 'expected_dofs': 14,
            'collision_solid_counts': collision_parts,
            'mesh_files': [{'path': str(path.relative_to(output)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                           for path in sorted(referenced_meshes)],
            '验收状态': '静态资产、Gym物理短测和V3四环境reset验收通过；未启动训练。',
            '碰撞几何处理': '按完整CAD连通组件生成凸包；无法形成凸包的零体积组件及其三角面按link计数省略，视觉几何未改。',
        }
        (output / 'asset_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return urdf_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='从桌面源模型生成独立的V3仿真候选资产')
    parser.add_argument('--output', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--unsplit-collisions', action='store_true', help='只用于整体凸包故障对照')
    arguments = parser.parse_args()
    print(build_asset(output=arguments.output, split_collisions=not arguments.unsplit_collisions))

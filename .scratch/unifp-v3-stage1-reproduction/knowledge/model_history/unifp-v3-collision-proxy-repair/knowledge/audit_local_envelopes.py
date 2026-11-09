"""Audit saved terminal poses with one-side-at-a-time mesh substitutions."""

import hashlib
import json
from pathlib import Path
import sys
from typing import Any
import xml.etree.ElementTree as ET

import fcl
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager, pyplot as plt
import numpy as np
import trimesh

PROJECT_ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(PROJECT_ROOT / '.scratch/unifp-v3-failure-visualization/knowledge'))
from audit_failure_geometry import load_surfaces, measure_pair, pose_from_origin

TASK_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = PROJECT_ROOT / 'UniFP_v3/resources/robots'
URDF_NAME = 'urdf/a1arm_rgm3_desktop_fixed_gripper.urdf'
R3_URDF = ASSET_ROOT / 'v3_a1_desktop_train_v7_local_base_proxy_r3' / URDF_NAME
DESKTOP_URDF = ASSET_ROOT / 'v3_a1_desktop' / URDF_NAME
OUTPUT_ROOT = TASK_ROOT / 'artifacts/envelope_audit_20261106'
LABELS = ('visual', 'desktop', 'r3')
LABELS_ZH = ('视觉', '桌面源碰撞', 'r3碰撞')
INPUT_HASHES: dict[str, str] = {}


def record_hash(path: Path) -> None:
    """Record the SHA256 identity of an input file before reading it.

    Args:
        path: Existing input path, resolved before recording.

    Note:
        Updates INPUT_HASHES; does not modify the referenced file.
    """
    resolved_path = path.resolve()
    INPUT_HASHES[str(resolved_path.relative_to(PROJECT_ROOT))] = hashlib.sha256(
        resolved_path.read_bytes()).hexdigest()


def load_variant(urdf_path: Path, body: str, kind: str,
                 names: list[str] | None = None) -> tuple[list[tuple[str, Any]], trimesh.Trimesh]:
    """Load matching FCL surfaces and a concatenated mesh in link coordinates.

    Args:
        urdf_path: URDF whose origins and mesh scales define meter coordinates.
        body: Link name to load.
        kind: visual or collision geometry.
        names: Collision names to retain; None selects all geometries.

    Returns:
        Named FCL surfaces and the same triangles as a Trimesh, in link meters.

    Raises:
        AssertionError: A requested collision name is missing or duplicated.
    """
    record_hash(urdf_path)
    link = ET.parse(urdf_path).getroot().find(f"link[@name='{body}']")
    if names is not None:
        for element in list(link.findall(kind)):
            if element.get('name') not in names:
                link.remove(element)
        assert sorted(element.get('name') for element in link.findall(kind)) == sorted(names)
    meshes = []
    for element in link.findall(kind):
        reference = element.find('geometry/mesh')
        mesh_path = (urdf_path.parent / reference.get('filename')).resolve()
        record_hash(mesh_path)
        mesh = trimesh.load(mesh_path, force='scene').to_geometry()
        mesh.apply_scale(np.fromstring(reference.get('scale', '1 1 1'), sep=' '))
        if element.find('origin') is not None:
            mesh.apply_transform(pose_from_origin(element.find('origin')))
        meshes.append(mesh)
    return load_surfaces(urdf_path, link, kind), trimesh.util.concatenate(meshes)


def source_mapping(body: str, collision_name: str,
                   manifest: dict[str, Any]) -> dict[str, Any]:
    """Resolve a collision name through its current local index to CAD sources.

    Args:
        body: R3 link containing the target collision.
        collision_name: Exact XML collision name, not a shape index.
        manifest: Current r3 manifest with remapped local_shape records.

    Returns:
        Current shape index and source component indices with XML names.
    """
    link = ET.parse(R3_URDF).getroot().find(f"link[@name='{body}']")
    local_shape = next(index for index, element in enumerate(link.findall('collision'))
                       if element.get('name') == collision_name)
    records = manifest['proxy_records'] + manifest['retained_collision_records']
    record = next(item for item in records if item['link'] == body
                  and item['local_shape'] == local_shape)
    desktop_link = ET.parse(DESKTOP_URDF).getroot().find(f"link[@name='{body}']")
    source_names = [desktop_link.findall('collision')[index].get('name')
                    for index in record['source_components']]
    return {'collision_name': collision_name, 'local_shape': local_shape,
            'source_components': record['source_components'], 'source_names': source_names}


def ray_crossings(mesh: trimesh.Trimesh, anchor_m: np.ndarray,
                  normal: np.ndarray) -> list[float]:
    """Measure shell crossings on a fixed local normal line, without solid inference.

    Args:
        mesh: Triangles already expressed in base_link meters.
        anchor_m: Base visual nearest point, shape [3], in meters.
        normal: Unit direction toward the other visual nearest point, shape [3].

    Returns:
        Unique signed crossings in millimeters within ±30 mm of anchor_m.
        Empty means this line has no crossing in that window, not no collision.
    """
    locations, _, _ = mesh.ray.intersects_location(
        [anchor_m - normal * 0.030], [normal], multiple_hits=True)
    offsets_mm = (locations - anchor_m) @ normal * 1000
    offsets_mm = offsets_mm[np.abs(offsets_mm) <= 30.0001]
    return np.unique(np.round(offsets_mm, 6)).tolist()


def plot_case(result: dict[str, Any], meshes: dict[str, dict[str, trimesh.Trimesh]],
              anchor_m: np.ndarray, normal: np.ndarray, tangent: np.ndarray) -> None:
    """Plot the substitution matrix and local normal sections as PNG and PDF.

    Args:
        result: Case measurements, target mapping and saved pose identity.
        meshes: Base-frame meshes indexed by side and representation.
        anchor_m: Base visual closest point, shape [3], in meters.
        normal: Unit visual separation direction in base_link coordinates.
        tangent: Unit section tangent perpendicular to normal.

    Note:
        Writes figures to OUTPUT_ROOT; asset geometry is not changed.
    """
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    distance_matrix = np.array([[result['matrix'][base][other]['minimum_surface_distance_m']
                                 * 1000 for other in LABELS] for base in LABELS])
    axes[0].imshow(distance_matrix, cmap='YlGnBu', vmin=0,
                   vmax=max(float(distance_matrix.max()), 0.1))
    axes[0].set(xticks=range(3), xticklabels=LABELS_ZH,
                yticks=range(3), yticklabels=LABELS_ZH,
                xlabel=f"对侧：{result['other_body']}", ylabel='底盘侧',
                title='只替换一侧：三角表面间距（mm）')
    for base_index, base_label in enumerate(LABELS):
        for other_index, other_label in enumerate(LABELS):
            cell = result['matrix'][base_label][other_label]
            label = '相交' if cell['surface_intersection'] else f'{distance_matrix[base_index, other_index]:.3f}'
            axes[0].text(other_index, base_index, label, ha='center', va='center',
                         color='white' if distance_matrix[base_index, other_index] > distance_matrix.max() * 0.55 else 'black')

    # The plane includes the separation normal and a deterministic tangent.
    # Axis clipping preserves crossing faces whose vertices lie outside the window.
    plane_normal = np.cross(normal, tangent)
    colors = ('#333333', '#df7b00', '#2474b5')
    for side in ('base', 'other'):
        for label, color in zip(LABELS, colors):
            section = meshes[side][label].section(plane_origin=anchor_m,
                                                 plane_normal=plane_normal)
            plotted_label = False
            if section is not None:
                for points_m in section.discrete:
                    relative_m = points_m - anchor_m
                    axes[1].plot(relative_m @ tangent * 1000, relative_m @ normal * 1000,
                                 color=color, linestyle='-' if side == 'base' else '--',
                                 linewidth=1.4, label=f"{'底盘' if side == 'base' else '对侧'} {LABELS_ZH[LABELS.index(label)]}"
                                 if not plotted_label else None)
                    plotted_label = True
    visual_gap_mm = result['matrix']['visual']['visual']['minimum_surface_distance_m'] * 1000
    axes[1].plot([0, 0], [0, visual_gap_mm], 'r.-', label='视觉最近点连线')
    axes[1].set(xlim=(-22, 22), ylim=(-15, 18), aspect='equal',
                xlabel='局部切向坐标（mm）', ylabel='视觉分离法向坐标（mm）',
                title='最近点局部截面：实线底盘，虚线对侧')
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8, loc='lower right')
    figure.suptitle(f"step {result['policy_step']} / env {result['env_id']}：{result['base_mapping']['collision_name']}")
    stem = OUTPUT_ROOT / f"step{result['policy_step']}_envelopes"
    figure.savefig(stem.with_suffix('.png'), dpi=180)
    figure.savefig(stem.with_suffix('.pdf'))
    plt.close(figure)


def main() -> None:
    """Audit seven saved poses and persist traceable side-substitution evidence.

    Note:
        Reads frozen URDF/mesh/pose files, verifies their hashes remain unchanged,
        and writes diagnostics only. Does not import Gym or run policy updates.

    Raises:
        AssertionError: A saved baseline changes or an input is modified during audit.
    """
    font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams['font.family'] = ['Noto Sans CJK JP']
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = R3_URDF.parent.parent / 'asset_manifest.json'
    record_hash(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    cases = (
        ('new_failures_geometry.json', 423, 'base_link_local_1_group_0', 'wrist_camera_proxy_0'),
        ('new_failures_geometry.json', 480, 'base_link_local_1_group_0', 'wrist_camera_proxy_0'),
        ('new_failures_geometry.json', 766, 'base_link_local_1_group_0', 'wrist_camera_proxy_0'),
        ('nine_geometry.json', 1268, 'base_link_local_1_group_5', 'wrist_camera_proxy_0'),
        ('nine_geometry.json', 2066, 'base_link_proxy_13', 'arm_seg4_proxy_1'),
        ('native_geometry.json', 158, 'base_link_local_1_group_0', 'wrist_camera_connector_proxy_0'),
        ('native_geometry.json', 454, 'base_link_local_1_group_0', 'wrist_camera_proxy_0'),
    )
    results = []
    for evidence_name, step, base_name, other_name in cases:
        evidence_path = TASK_ROOT / 'artifacts/grouped_r3_acceptance' / evidence_name
        record_hash(evidence_path)
        event = next(item for item in json.loads(evidence_path.read_text())['events']
                     if item['policy_step'] == step)
        pose = np.asarray(event['other_pose_in_base_m'])
        mappings = {'base': source_mapping('base_link', base_name, manifest),
                    'other': source_mapping(event['other_body'], other_name, manifest)}
        surfaces = {}
        meshes = {}
        for side, body in (('base', 'base_link'), ('other', event['other_body'])):
            variants = {'visual': load_variant(DESKTOP_URDF, body, 'visual'),
                        'desktop': load_variant(DESKTOP_URDF, body, 'collision', mappings[side]['source_names']),
                        'r3': load_variant(R3_URDF, body, 'collision', [mappings[side]['collision_name']])}
            surfaces[side] = {label: variant[0] for label, variant in variants.items()}
            meshes[side] = {label: variant[1] for label, variant in variants.items()}
            if side == 'other':
                for mesh in meshes[side].values():
                    mesh.apply_transform(pose)
        matrix = {base: {other: measure_pair(surfaces['base'][base], surfaces['other'][other], pose)
                         for other in LABELS} for base in LABELS}
        # The visual baseline must reproduce the independent full-pair audit.
        assert abs(matrix['visual']['visual']['minimum_surface_distance_m']
                   - event['visual']['minimum_surface_distance_m']) < 1e-8
        assert matrix['r3']['r3']['surface_intersection'] == event['collision']['surface_intersection']
        assert abs(matrix['r3']['r3']['minimum_surface_distance_m']
                   - event['collision']['minimum_surface_distance_m']) < 1e-8
        visual_points_m = np.asarray(matrix['visual']['visual']['closest_disjoint_pair']['nearest_points_in_base_m'])
        normal = visual_points_m[1] - visual_points_m[0]
        normal /= np.linalg.norm(normal)
        reference_axis = np.eye(3)[np.argmin(np.abs(normal))]
        tangent = reference_axis - normal * np.dot(reference_axis, normal)
        tangent /= np.linalg.norm(tangent)
        ray_sections = {side: {label: ray_crossings(mesh, visual_points_m[0], normal)
                               for label, mesh in variants.items()} for side, variants in meshes.items()}
        result = {'policy_step': step, 'env_id': event['env_id'], 'other_body': event['other_body'],
                  'pose_source': event['pose_source'], 'pose_file': str(evidence_path.relative_to(PROJECT_ROOT)),
                  'other_pose_in_base_m': pose.tolist(), 'base_mapping': mappings['base'],
                  'other_mapping': mappings['other'], 'matrix': matrix,
                  'section_anchor_in_base_m': visual_points_m[0].tolist(),
                  'section_normal_in_base': normal.tolist(), 'section_tangent_in_base': tangent.tolist(),
                  'normal_ray_crossings_mm': ray_sections}
        plot_case(result, meshes, visual_points_m[0], normal, tangent)
        results.append(result)
        print(json.dumps({'step': step, 'base_sources': mappings['base']['source_components'],
                          'distances_mm': [[round(matrix[b][o]['minimum_surface_distance_m'] * 1000, 6)
                                            for o in LABELS] for b in LABELS],
                          'ray_crossings_mm': ray_sections}, ensure_ascii=False), flush=True)
    for path in (Path(__file__), PROJECT_ROOT / '.scratch/unifp-v3-failure-visualization/knowledge/audit_failure_geometry.py',
                 PROJECT_ROOT / 'UniFP_v3/tools/compare_desktop_fk.py'):
        record_hash(path)
    assert all(hashlib.sha256((PROJECT_ROOT / path).read_bytes()).hexdigest() == digest
               for path, digest in INPUT_HASHES.items())
    report = {'matrix_order': list(LABELS), 'coordinate_frame': 'base_link', 'length_unit': 'm',
              'ray_length_unit': 'mm', 'events': results, 'input_sha256': INPUT_HASHES,
              'python_fcl_version': fcl.__version__, 'trimesh_version': trimesh.__version__,
              'baseline_distance_tolerance_m': 1e-8,
              'inputs_unchanged_after_audit': True,
              'note_zh': 'visual使用完整link；desktop仅使用清单映射的目标源组件；r3仅使用目标shape。表面相交不是穿透深度；正间距不排除实体包含、此前子步接触或PhysX接触余量。法线射线为单截面壳面位置，不是全局膨胀量。'}
    (OUTPUT_ROOT / 'envelope_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()

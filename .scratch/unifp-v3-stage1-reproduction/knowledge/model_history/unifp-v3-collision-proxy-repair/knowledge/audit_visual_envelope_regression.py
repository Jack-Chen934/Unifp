"""Audit the fixed V8 geometry on historical poses, native substeps and contacts."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any
import xml.etree.ElementTree as ET

import fcl
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from build_visual_envelope_candidate import DEFAULT_OUTPUT, URDF_NAME, file_sha256

PROJECT_ROOT = Path(__file__).resolve().parents[6]
TASK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / '.scratch/unifp-v3-failure-visualization/knowledge'))
from audit_failure_geometry import load_surfaces, measure_pair, pose_from_origin

EVIDENCE_ROOT = TASK_ROOT / 'artifacts/grouped_r3_acceptance'
SUBSTEP_ROOT = TASK_ROOT / 'artifacts/envelope_audit_20261106/contact_substeps'
OUTPUT_ROOT = TASK_ROOT / 'artifacts/visual_envelope_v8_20261106'
ASSETS = {'v6': 'v3_a1_desktop_train_v6_coplanar_r2',
          'r3': 'v3_a1_desktop_train_v7_local_base_proxy_r3',
          'v8': DEFAULT_OUTPUT.name}
BODY_NAMES = ('base_link', 'wrist_camera', 'wrist_camera_connector', 'arm_seg4')


def native_pose_matrix(pose_m_xyzw: list[float]) -> np.ndarray:
    """Convert one saved native rigid-body pose to a homogeneous transform.

    Args:
        pose_m_xyzw: World position in meters and quaternion in xyzw order, [7].

    Returns:
        Body-to-world transform with shape [4,4].
    """
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(pose_m_xyzw[3:7]).as_matrix()
    matrix[:3, 3] = pose_m_xyzw[:3]
    return matrix


def convex_solids(urdf_path: Path, link: ET.Element) -> list[tuple[str, Any]]:
    """Load collision convex volumes to distinguish containment from surface tests.

    Args:
        urdf_path: Actual asset URDF resolving collision references.
        link: Body whose existing collision shapes are already convex proxies.

    Returns:
        Named FCL convex solids in link-local meter coordinates.

    Raises:
        OSError: A referenced mesh cannot be loaded.
    """
    solids = []
    for collision in link.findall('collision'):
        reference = collision.find('geometry/mesh')
        mesh = trimesh.load((urdf_path.parent / reference.get('filename')).resolve(),
                            force='scene').to_geometry()
        mesh.apply_scale(np.fromstring(reference.get('scale', '1 1 1'), sep=' '))
        mesh.apply_transform(pose_from_origin(collision.find('origin')))
        # Use consistently oriented convex faces, preserving the proxy's volume.
        hull = mesh.convex_hull
        polygon_indices = np.column_stack((np.full(len(hull.faces), 3), hull.faces))
        solids.append((collision.get('name'), fcl.Convex(
            np.asarray(hull.vertices), len(hull.faces), polygon_indices.reshape(-1).astype(np.int32))))
    return solids


def positive_control(record: dict[str, Any], event: dict[str, Any],
                     surfaces: dict[tuple[str, str, str], list[tuple[str, Any]]]
                     ) -> dict[str, Any]:
    """Translate a saved body toward one clipped real visual until it intersects.

    Args:
        record: V8 refinement record identifying the local region and collision.
        event: Saved pose selecting the original wrist or arm counterpart.
        surfaces: Cached actual meshes keyed by asset, body and geometry kind.

    Returns:
        First verified local visual intersection and the corresponding targeted
        V8 collision test, with the constructed base-frame pose and displacement.

    Raises:
        AssertionError: A local real visual contact cannot be constructed within
            20 mm past its independently measured nearest-surface separation.
    """
    urdf_path = DEFAULT_OUTPUT / 'urdf' / URDF_NAME
    visual = ET.parse(urdf_path).getroot().find("link[@name='base_link']/visual")
    reference = visual.find('geometry/mesh')
    local_visual = trimesh.load((urdf_path.parent / reference.get('filename')).resolve(),
                                force='scene').to_geometry()
    local_visual.apply_scale(np.fromstring(reference.get('scale', '1 1 1'), sep=' '))
    local_visual.apply_transform(pose_from_origin(visual.find('origin')))
    # Independent triangle clipping keeps a real visual surface, not its convex hull.
    for plane in np.asarray(record['region_planes_in_base_m']):
        local_visual = trimesh.intersections.slice_mesh_plane(
            local_visual, -plane[:3], -plane[3] * plane[:3], cap=False)
    local_model = fcl.BVHModel()
    local_model.beginModel(len(local_visual.faces), len(local_visual.vertices))
    local_model.addSubModel(np.asarray(local_visual.vertices), np.asarray(local_visual.faces, dtype=np.int32))
    local_model.endModel()
    local_surface = [('clipped_source_visual', local_model)]
    other_body = event['other_body']
    original_pose = np.asarray(event['other_pose_in_base_m'])
    original = measure_pair(local_surface, surfaces['v8', other_body, 'visual'], original_pose)
    assert not original['surface_intersection']
    nearest_m = np.asarray(original['closest_disjoint_pair']['nearest_points_in_base_m'])
    approach_direction = nearest_m[0] - nearest_m[1]
    approach_direction /= np.linalg.norm(approach_direction)
    target_collision = [item for item in surfaces['v8', 'base_link', 'collision']
                        if item[0] == record['collision_name']]
    # Preserve orientation. Move in 0.1 mm increments past the local visual gap;
    # require an actual triangle intersection so containment cannot fake a pass.
    for displacement_m in np.arange(original['minimum_surface_distance_m'] + 0.0001,
                                    original['minimum_surface_distance_m'] + 0.02, 0.0001):
        pose = original_pose.copy()
        pose[:3, 3] += approach_direction * displacement_m
        visual_result = measure_pair(local_surface, surfaces['v8', other_body, 'visual'], pose)
        if visual_result['surface_intersection']:
            return {'source_component': record['rebuilt_source_component'],
                    'collision_name': record['collision_name'], 'other_body': other_body,
                    'source_policy_step': event['policy_step'], 'translation_m': float(displacement_m),
                    'approach_direction_in_base': approach_direction.tolist(),
                    'other_pose_in_base_m': pose.tolist(), 'visual': visual_result,
                    'collision': measure_pair(target_collision, surfaces['v8', other_body, 'collision'], pose)}
    raise AssertionError(f"Cannot construct local visual contact for {record['collision_name']}")


def audit_geometry() -> dict[str, Any]:
    """Measure all approved saved poses and real contacts on the final V8 identity.

    Returns:
        Complete surface/solid measurements, regression checks, positive controls
        and input SHA256 identities. Distances are meters in base_link coordinates.

    Raises:
        OSError: An asset or frozen evidence file cannot be read.
        AssertionError: A controlled real contact cannot be constructed.

    Note:
        Reads geometry only. Does not import Gym, modify assets or write output.
    """
    input_hashes = {}
    surfaces = {}
    for variant, asset_name in ASSETS.items():
        asset_root = PROJECT_ROOT / 'UniFP_v3/resources/robots' / asset_name
        urdf_path = asset_root / 'urdf' / URDF_NAME
        robot = ET.parse(urdf_path).getroot()
        paths = {urdf_path, asset_root / 'asset_manifest.json'}
        paths.update((urdf_path.parent / node.get('filename')).resolve() for node in robot.findall('.//mesh'))
        for path in paths:
            input_hashes[str(path.relative_to(PROJECT_ROOT))] = file_sha256(path)
        for body in BODY_NAMES:
            link = robot.find(f"link[@name='{body}']")
            for kind in ('visual', 'collision'):
                surfaces[variant, body, kind] = load_surfaces(urdf_path, link, kind)
            surfaces[variant, body, 'solid'] = convex_solids(urdf_path, link)

    # First retain terminal evidence without replacing FK-derived poses with native.
    saved_poses = []
    saved_events = []
    baseline_matches = []
    for filename in ('nine_geometry.json', 'native_geometry.json', 'new_failures_geometry.json'):
        evidence_path = EVIDENCE_ROOT / filename
        input_hashes[str(evidence_path.relative_to(PROJECT_ROOT))] = file_sha256(evidence_path)
        for event in json.loads(evidence_path.read_text())['events']:
            saved_events.append(event)
            pose = np.asarray(event['other_pose_in_base_m'])
            measurements = {variant: measure_pair(surfaces[variant, 'base_link', 'collision'],
                            surfaces[variant, event['other_body'], 'collision'], pose) for variant in ASSETS}
            solid = measure_pair(surfaces['v8', 'base_link', 'solid'],
                                 surfaces['v8', event['other_body'], 'solid'], pose)
            visual = measure_pair(surfaces['v8', 'base_link', 'visual'],
                                  surfaces['v8', event['other_body'], 'visual'], pose)
            baseline_matches.append(measurements['r3']['surface_intersection'] == event['collision']['surface_intersection']
                and abs(measurements['r3']['minimum_surface_distance_m'] - event['collision']['minimum_surface_distance_m']) < 1e-8)
            saved_poses.append({'source': filename, 'policy_step': event['policy_step'], 'env_id': event['env_id'],
                'other_body': event['other_body'], 'pose_source': event['pose_source'],
                'other_pose_in_base_m': pose.tolist(), 'visual': visual, 'variants': measurements,
                'v8_solid_intersection': solid['surface_intersection'], 'v8_solid_pairs': solid['intersecting_pairs']})

    # Reconstruct relative transforms solely from each recorded native substep.
    substeps = []
    old_analysis_path = SUBSTEP_ROOT / 'substep_analysis.json'
    input_hashes[str(old_analysis_path.relative_to(PROJECT_ROOT))] = file_sha256(old_analysis_path)
    old_analysis = json.loads(old_analysis_path.read_text())
    for source_variant in ('v6', 'r3'):
        path = SUBSTEP_ROOT / f'{source_variant}_S1_20s/substeps.json'
        input_hashes[str(path.relative_to(PROJECT_ROOT))] = file_sha256(path)
        for event in json.loads(path.read_text())['events']:
            original_event = next(item for item in old_analysis['variants'][source_variant]['events']
                                  if item['policy_step'] == event['policy_step'] and item['env_id'] == event['env_id'])
            other_body = original_event['other_body']
            for frame, original_frame in zip(event['frames'], original_event['frames'], strict=True):
                native = frame['body_pose_world_m_xyzw']
                pose = np.linalg.inv(native_pose_matrix(native['base_link'])) @ native_pose_matrix(native[other_body])
                measurements = {variant: measure_pair(surfaces[variant, 'base_link', 'collision'],
                                surfaces[variant, other_body, 'collision'], pose) for variant in ASSETS}
                baseline = measurements[source_variant]
                baseline_matches.append(baseline['surface_intersection'] == original_frame['collision']['surface_intersection']
                    and abs(baseline['minimum_surface_distance_m'] - original_frame['collision']['minimum_surface_distance_m']) < 1e-8)
                solid = measure_pair(surfaces['v8', 'base_link', 'solid'], surfaces['v8', other_body, 'solid'], pose)
                substeps.append({'source_variant': source_variant, 'event_policy_step': event['policy_step'],
                    'env_id': event['env_id'], 'policy_step': frame['policy_step'], 'substep': frame['substep'],
                    'time_before_failure_ms': original_frame['time_before_failure_ms'],
                    'other_body': other_body, 'pose_source': 'native_substep_body_pose',
                    'other_pose_in_base_m': pose.tolist(), 'variants': measurements,
                    'visual': original_frame['visual'], 'v8_solid_intersection': solid['surface_intersection'],
                    'v8_solid_pairs': solid['intersecting_pairs']})

    manifest = json.loads((DEFAULT_OUTPUT / 'asset_manifest.json').read_text())
    step_by_component = {8: 423, 99: 1268, 43: 2066}
    controls = [positive_control(record, next(event for event in saved_events
                if event['policy_step'] == step_by_component[record['rebuilt_source_component']]), surfaces)
                for record in manifest['visual_envelope_refinements']]
    all_poses = saved_poses + substeps
    new_intersections = [item for item in all_poses if item['v8_solid_intersection']
                         and not item['variants']['r3']['surface_intersection']]
    checks = {'historical_baseline_reproduced': all(baseline_matches),
        'target_false_intersections_removed': all(not item['v8_solid_intersection'] for item in saved_poses
                                                 if item['policy_step'] in (423, 480, 1268, 2066)),
        'step454_remains_separate': all(not item['v8_solid_intersection'] for item in all_poses
                                       if item.get('event_policy_step', item['policy_step']) == 454),
        'no_new_intersections_on_separate_poses': not new_intersections,
        'three_real_contacts_detected': all(item['visual']['surface_intersection']
                                          and item['collision']['surface_intersection'] for item in controls)}
    for path in (Path(__file__), PROJECT_ROOT / '.scratch/unifp-v3-failure-visualization/knowledge/audit_failure_geometry.py'):
        input_hashes[str(path.relative_to(PROJECT_ROOT))] = file_sha256(path)
    return {'length_unit': 'm', 'coordinate_frame': 'base_link', 'checks': checks,
            'saved_poses': saved_poses, 'substeps': substeps, 'positive_controls': controls,
            'input_sha256': input_hashes, 'python_fcl_version': fcl.__version__,
            'note_zh': '同一保存姿态跨资产离线核对，不是V8仿真轨迹。BVH表面与凸体实体碰撞分别记录；不含PhysX接触余量或求解历史。'}


def main() -> None:
    """Write the fixed regression evidence and a five-event substep distance plot.

    Raises:
        OSError: Input reading or artifact writing fails.
        AssertionError: A confirmed geometry regression check fails.

    Note:
        Writes only the V8 artifact directory; does not change formal training.
    """
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager, pyplot as plt

    report = audit_geometry()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / 'geometry_regression.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams['font.family'] = ['Noto Sans CJK JP']
    figure, axes = plt.subplots(5, 1, figsize=(9, 12), sharex=True)
    event_keys = list(dict.fromkeys((item['source_variant'], item['event_policy_step']) for item in report['substeps']))
    for axis, event_key in zip(axes, event_keys, strict=True):
        rows = [item for item in report['substeps'] if (item['source_variant'], item['event_policy_step']) == event_key]
        times_ms = [item['time_before_failure_ms'] for item in rows]
        for variant in ASSETS:
            axis.plot(times_ms, [item['variants'][variant]['minimum_surface_distance_m'] * 1000 for item in rows],
                      marker='.', label=variant)
        axis.plot(times_ms, [item['visual']['minimum_surface_distance_m'] * 1000 for item in rows],
                  color='black', linestyle='--', label='实际视觉')
        axis.set_title(f'{event_key[0]}原生轨迹，事件{event_key[1]}：{rows[0]["other_body"]}')
        axis.set_ylabel('表面间距（mm）')
        axis.grid(alpha=0.3)
    axes[0].legend(ncol=4)
    axes[-1].set_xlabel('相对原终止边界的时间（ms）；0间距表示表面相交，非穿透深度')
    figure.tight_layout()
    figure.savefig(OUTPUT_ROOT / 'substep_geometry.png', dpi=160)
    plt.close(figure)
    print(json.dumps(report['checks'], ensure_ascii=False), flush=True)
    assert all(report['checks'].values()), 'Geometry regression failed; inspect saved evidence before GPU tests'


if __name__ == '__main__':
    main()

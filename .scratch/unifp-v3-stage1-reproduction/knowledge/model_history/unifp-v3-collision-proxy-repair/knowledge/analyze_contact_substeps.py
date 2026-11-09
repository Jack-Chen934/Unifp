"""Measure saved substep geometry and compare instrumented replay boundaries."""

import json
from pathlib import Path
import sys
from typing import Any
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager, pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(PROJECT_ROOT / '.scratch/unifp-v3-failure-visualization/knowledge'))
from audit_failure_geometry import load_surfaces, measure_pair
from audit_local_envelopes import record_hash, INPUT_HASHES

TASK_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = TASK_ROOT / 'artifacts/envelope_audit_20261106/contact_substeps'
URDF_NAME = 'urdf/a1arm_rgm3_desktop_fixed_gripper.urdf'


def relative_pose(poses: dict[str, list[float]], other_body: str) -> np.ndarray:
    """Convert two native world poses to a relative base-link transform.

    Args:
        poses: World xyz in meters and xyzw quaternions for each observed body.
        other_body: Body to express in base_link coordinates.

    Returns:
        Homogeneous [4,4] transform from the other link to base_link, in meters.
    """
    matrices = {}
    for body in ('base_link', other_body):
        values = poses[body]
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_quat(values[3:7]).as_matrix()
        matrix[:3, 3] = values[:3]
        matrices[body] = matrix
    return np.linalg.inv(matrices['base_link']) @ matrices[other_body]


def analyze_variant(label: str, asset: str, baseline: Path) -> dict[str, Any]:
    """Measure each failed event's preceding 16 simulation boundaries.

    Args:
        label: Downloaded replay directory name.
        asset: Frozen robot resource directory matching that replay.
        baseline: Earlier native failure records for trajectory comparison.

    Returns:
        Event geometry, force traces, boundary comparisons and file identities.

    Raises:
        AssertionError: Frame indexing or the final readback disagrees with replay.

    Note:
        Writes one force/distance PNG per event, without changing any input.
    """
    folder = EVIDENCE_ROOT / label
    failures_path, substeps_path = folder / 'failures.json', folder / 'substeps.json'
    summary_path = folder / 'summary.json'
    for path in (failures_path, substeps_path, summary_path, baseline):
        record_hash(path)
    failures = json.loads(failures_path.read_text())
    previous = {(event['policy_step'], event['env_id']): event
                for event in json.loads(baseline.read_text())}
    observed = json.loads(substeps_path.read_text())
    summary = json.loads(summary_path.read_text())
    assert summary['source_identity_verified'] and summary['policy_updates'] == 0
    assert len(failures) == len(observed['events'])
    urdf_path = PROJECT_ROOT / 'UniFP_v3/resources/robots' / asset / URDF_NAME
    record_hash(urdf_path)
    robot = ET.parse(urdf_path).getroot()
    bodies = ('base_link', 'wrist_camera', 'wrist_camera_connector', 'arm_seg4')
    surfaces = {}
    for body in bodies:
        link = robot.find(f"link[@name='{body}']")
        for kind in ('visual', 'collision'):
            for element in link.findall(kind):
                record_hash((urdf_path.parent / element.find('geometry/mesh').get('filename')).resolve())
            surfaces[body, kind] = load_surfaces(urdf_path, link, kind)
    events = []
    for failure, window in zip(failures, observed['events']):
        key = failure['policy_step'], failure['env_id']
        assert key == (window['policy_step'], window['env_id'])
        base_force = np.asarray(failure['base_contact_vector_n'])
        other_body = min((item for item in failure['largest_contacts'] if item['body'] != 'base_link'),
                         key=lambda item: np.linalg.norm(base_force + item['vector_n']))['body']
        assert other_body in bodies
        frames = []
        for frame in window['frames']:
            pose = relative_pose(frame['body_pose_world_m_xyzw'], other_body)
            force = frame['contact_force_world_n'][window['body_names'].index('base_link')]
            geometry = {kind: measure_pair(surfaces['base_link', kind], surfaces[other_body, kind], pose)
                        for kind in ('visual', 'collision')}
            frames.append({'policy_step': frame['policy_step'], 'substep': frame['substep'],
                           'time_before_failure_ms': ((frame['policy_step'] - key[0]) * 8 + frame['substep'] - 8) * 2.5,
                           'base_contact_norm_n': float(np.linalg.norm(force)), **geometry})
        assert [frame['substep'] for frame in frames[-8:]] == list(range(1, 9))
        assert abs(frames[-1]['base_contact_norm_n'] - failure['base_contact_norm_n']) < 1e-6
        old = previous.get(key)
        comparison = {'same_event_key_as_baseline': old is not None}
        if old is not None:
            comparison['maximum_terminal_pose_component_difference'] = max(
                abs(np.asarray(old['body_pose_world_m_xyzw'][body])
                    - np.asarray(failure['body_pose_world_m_xyzw'][body])).max().item()
                for body in failure['body_pose_world_m_xyzw'])
            comparison['terminal_base_force_vector_difference_n'] = float(np.linalg.norm(
                np.asarray(old['base_contact_vector_n']) - base_force))
        events.append({'policy_step': key[0], 'env_id': key[1], 'other_body': other_body,
                       'baseline_comparison': comparison, 'frames': frames})

        figure, axis = plt.subplots(figsize=(8, 4), layout='constrained')
        times_ms = [frame['time_before_failure_ms'] for frame in frames]
        axis.plot(times_ms, [frame['base_contact_norm_n'] for frame in frames], 'o-', color='#bf4b15', label='底盘净接触力')
        axis.axhline(1, color='#bf4b15', linestyle=':', label='局部reset阈值1N')
        axis.set(xlabel='相对终止策略边界的时间（ms）', ylabel='底盘净力（N）')
        distance_axis = axis.twinx()
        for kind, color, name in [('visual', '#555555', '视觉间距'), ('collision', '#2474b5', '碰撞表面间距')]:
            distance_axis.plot(times_ms, [frame[kind]['minimum_surface_distance_m'] * 1000 for frame in frames],
                               '.--', color=color, label=name)
        intersecting = [frame['time_before_failure_ms'] for frame in frames if frame['collision']['surface_intersection']]
        distance_axis.scatter(intersecting, np.zeros(len(intersecting)), color='red', marker='x', label='碰撞表面相交')
        distance_axis.set_ylabel('表面间距（mm，相交记0，非穿透深度）')
        axis.grid(alpha=0.25)
        axis.legend(loc='upper left', fontsize=8)
        distance_axis.legend(loc='upper right', fontsize=8)
        axis.set_title(f'{label}：step {key[0]} / env {key[1]} / {other_body}')
        figure.savefig(folder / f'step{key[0]}_substeps.png', dpi=180)
        plt.close(figure)
    return {'asset': asset, 'checkpoint_sha256': summary['checkpoint_sha256'],
            'baseline_event_keys': [list(key) for key in previous if key[0] < 1000],
            'observed_event_keys': [[event['policy_step'], event['env_id']] for event in failures],
            'events': events}


def main() -> None:
    """Analyze the V6 and r3 independent substep readback evidence.

    Note:
        Writes a single machine report and per-event figures into the new evidence
        directory; uses only native saved poses and independent FCL measurements.
    """
    font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams['font.family'] = ['Noto Sans CJK JP']
    variants = {
        'v6': analyze_variant('v6_S1_20s', 'v3_a1_desktop_train_v6_coplanar_r2',
                             PROJECT_ROOT / '.scratch/unifp-v3-failure-visualization/artifacts/terminal_pose_probe/failures.json'),
        'r3': analyze_variant('r3_S1_20s', 'v3_a1_desktop_train_v7_local_base_proxy_r3',
                             TASK_ROOT / 'artifacts/grouped_r3_acceptance/sample_S1_20s/failures.json')}
    record_hash(Path(__file__))
    record_hash(Path(__file__).with_name('observe_contact_substeps.py'))
    (EVIDENCE_ROOT / 'substep_analysis.json').write_text(json.dumps(
        {'variants': variants, 'input_sha256': INPUT_HASHES,
         'note_zh': '每帧为2.5ms simulate结束后的姿态与净力；未读取求解器内部接触点/法向距离。正间距与力可并存，不能据此唯一归因于contact_offset。'},
        ensure_ascii=False, indent=2) + '\n')
    for label, variant in variants.items():
        for event in variant['events']:
            print(json.dumps({'variant': label, 'step': event['policy_step'],
                              'frames': [{'ms': f['time_before_failure_ms'],
                                          'force_n': round(f['base_contact_norm_n'], 6),
                                          'collision_mm': round(f['collision']['minimum_surface_distance_m'] * 1000, 6),
                                          'intersection': f['collision']['surface_intersection']} for f in event['frames']]}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

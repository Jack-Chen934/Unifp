"""Summarize the approved frozen-policy comparison without changing thresholds."""

from __future__ import annotations

from collections import Counter
from itertools import combinations
import json
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

from audit_visual_envelope_regression import ASSETS, PROJECT_ROOT, OUTPUT_ROOT, URDF_NAME
from audit_visual_envelope_regression import load_surfaces, measure_pair, native_pose_matrix
from build_visual_envelope_candidate import file_sha256

CHECKPOINT_SHA256 = '83c2784e52d0a89d5fc4b2e725a8df4c7c8e48de51add860e77e3186413d7102'
TARGET_BODIES = {'wrist_camera', 'wrist_camera_connector', 'arm_seg4'}


def identify_counterparts(event: dict[str, Any]) -> tuple[list[str], float]:
    """Find the smallest recorded body set opposing the net base contact force.

    Args:
        event: Native pre-reset failure with base force and up to five largest
            body forces, all in world coordinates and newtons.

    Returns:
        Body names and vector-sum residual in N. Empty names indicate that no
        recorded non-base force combination balances the base within 1e-4 N.
        Net force balance is evidence of counterparts, not contact-point data.
    """
    contacts = [item for item in event['largest_contacts']
                if item['body'] != 'base_link' and item['norm_n'] > 0]
    base_force_n = np.asarray(event['base_contact_vector_n'])
    minimum_residual_n = float(np.linalg.norm(base_force_n))
    for count in range(1, len(contacts) + 1):
        for subset in combinations(contacts, count):
            residual_n = float(np.linalg.norm(base_force_n + np.sum([item['vector_n'] for item in subset], axis=0)))
            minimum_residual_n = min(minimum_residual_n, residual_n)
            if residual_n < 1e-4:
                return sorted(item['body'] for item in subset), residual_n
    return [], minimum_residual_n


def summarize_run(directory: Path, variant: str, seconds: int,
                  surfaces: dict[tuple[str, str], list[tuple[str, Any]]]) -> dict[str, Any]:
    """Validate one fixed run and classify actual pre-reset failure evidence.

    Args:
        directory: Completed replay output containing summary and failures JSON.
        variant: One of v6, r3 or v8, selecting the actual asset identity.
        seconds: Approved per-environment simulation budget, 20 or 60 seconds.
        surfaces: Actual BVHs keyed by link name and visual/collision kind.

    Returns:
        Failure predicates, counterforce attribution, exposure and terminal
        geometry. A nearest terminal shape is reported without causal attribution.

    Raises:
        AssertionError: Run budget, checkpoint, asset or evaluation semantics differ.
        OSError: A completed replay result cannot be read.
    """
    summary_path, failures_path = directory / 'summary.json', directory / 'failures.json'
    summary = json.loads(summary_path.read_text())
    failures = json.loads(failures_path.read_text())
    urdf_path = PROJECT_ROOT / 'UniFP_v3/resources/robots' / ASSETS[variant] / 'urdf' / URDF_NAME
    assert summary['checkpoint_sha256'] == CHECKPOINT_SHA256
    assert summary['source_identity_verified'] and summary['policy_updates'] == 0
    assert summary['scenario'] == 'S1' and summary['action_mode'] == 'sample'
    assert summary['num_envs'] == 128 and not summary['self_collision_disabled']
    assert abs(summary['seconds_per_environment'] - seconds) < 1e-4
    assert summary['policy_steps'] == seconds * 50
    assert np.allclose(summary['shape_contact_offsets_m'], 0.01)
    assert np.allclose(summary['shape_rest_offsets_m'], 0)
    if variant != 'v6':
        assert summary['asset_override_urdf_sha256'] == file_sha256(urdf_path)
    else:
        assert summary['asset_override'] is None
    assert len(failures) == summary['completed_episodes'].get('physical_failure', 0)
    counterpart_counts = Counter()
    target_counterpart_failures = 0
    event_details = []
    for event in failures:
        detail = {'policy_step': event['policy_step'], 'env_id': event['env_id'],
                  'predicates': event['predicates'], 'base_contact_norm_n': event['base_contact_norm_n']}
        if event['predicates']['base_contact']:
            # Retain multi-point contacts: two grippers may jointly balance the
            # base while neither individual gripper has an opposing net force.
            bodies, residual_n = identify_counterparts(event)
            detail['counterforce_residual_n'] = residual_n
            if bodies:
                counterpart_counts['+'.join(bodies)] += 1
                target_counterpart_failures += int(bool(TARGET_BODIES.intersection(bodies)))
                detail['counterpart_bodies'] = bodies
                native = event['body_pose_world_m_xyzw']
                base_inverse = np.linalg.inv(native_pose_matrix(native['base_link']))
                detail['counterpart_geometry'] = {}
                for other_body in bodies:
                    pose = base_inverse @ native_pose_matrix(native[other_body])
                    detail['counterpart_geometry'][other_body] = {'other_pose_in_base_m': pose.tolist(),
                        'collision': measure_pair(surfaces['base_link', 'collision'], surfaces[other_body, 'collision'], pose),
                        'visual': measure_pair(surfaces['base_link', 'visual'], surfaces[other_body, 'visual'], pose)}
            else:
                counterpart_counts['unresolved_multiple_contacts'] += 1
        event_details.append(detail)
    exposure_env_s = summary['num_envs'] * summary['seconds_per_environment']
    return {'variant': variant, 'seed': summary['seed'], 'directory': str(directory.relative_to(PROJECT_ROOT)),
            'physical_failures': len(failures), 'completed_episodes': summary['completed_episodes'],
            'failure_predicates_overlapping': summary['failure_predicates_overlapping'],
            'counterpart_counts': dict(counterpart_counts),
            'target_counterpart_failures': target_counterpart_failures,
            'exposure_environment_seconds': exposure_env_s,
            'failures_per_1000_environment_seconds': len(failures) / exposure_env_s * 1000,
            'events': event_details,
            'input_sha256': {str(path.relative_to(PROJECT_ROOT)): file_sha256(path)
                             for path in (summary_path, failures_path, urdf_path)}}


def main() -> None:
    """Write descriptive per-seed comparisons and a failure-count figure.

    Raises:
        AssertionError: Any run differs from the approved fixed comparison protocol.
        OSError: A required run is incomplete or artifacts cannot be written.

    Note:
        Writes only comparison_summary.json and comparison_failures.png; does not
        adjust training configuration, promote assets or execute policy updates.
    """
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager, pyplot as plt

    surfaces = {}
    for variant, asset_name in ASSETS.items():
        urdf_path = PROJECT_ROOT / 'UniFP_v3/resources/robots' / asset_name / 'urdf' / URDF_NAME
        robot = ET.parse(urdf_path).getroot()
        surfaces[variant] = {(link.get('name'), kind): load_surfaces(urdf_path, link, kind)
                             for link in robot.findall('link') for kind in ('visual', 'collision')}
    initial = summarize_run(OUTPUT_ROOT / 'v8_seed1001_20s', 'v8', 20, surfaces['v8'])
    runs = [summarize_run(OUTPUT_ROOT / f'{variant}_seed{seed}_60s', variant, 60, surfaces[variant])
            for seed in (1001, 1002, 1003) for variant in ASSETS]
    totals = {}
    for variant in ASSETS:
        rows = [row for row in runs if row['variant'] == variant]
        predicates, counterparts, episodes = Counter(), Counter(), Counter()
        for row in rows:
            predicates.update(row['failure_predicates_overlapping'])
            counterparts.update(row['counterpart_counts'])
            episodes.update(row['completed_episodes'])
        failures = sum(row['physical_failures'] for row in rows)
        exposure = sum(row['exposure_environment_seconds'] for row in rows)
        totals[variant] = {'physical_failures': failures, 'completed_episodes': dict(episodes),
            'failure_predicates_overlapping': dict(predicates), 'counterpart_counts': dict(counterparts),
            'target_counterpart_failures': sum(row['target_counterpart_failures'] for row in rows),
            'exposure_environment_seconds': exposure, 'failures_per_1000_environment_seconds': failures / exposure * 1000}
    report = {'initial_v8_20s': initial, 'runs': runs, 'totals': totals,
              'seed_physical_gpu': {'1001': 0, '1002': 1, '1003': 3},
              'checkpoint_sha256': CHECKPOINT_SHA256, 'policy_updates': 0,
              'v8_failure_count_change_vs_v6': totals['v8']['physical_failures'] - totals['v6']['physical_failures'],
              'v8_failure_count_change_vs_r3': totals['v8']['physical_failures'] - totals['r3']['physical_failures'],
              'evidence_sha256': {path.name: file_sha256(path) for path in
                  (Path(__file__), OUTPUT_ROOT / 'asset_files_sha256.json', OUTPUT_ROOT / 'remote_identity.json',
                   OUTPUT_ROOT / 'comparison_execution_identity.json', OUTPUT_ROOT / 'remote_tool_sha256.json')},
              'note_zh': '各资产使用相同策略、预算与物性行，资产变化后轨迹分叉。仅描述本次三个seed的诊断，非新资产训练验收。counterpart依据单体或组合净力抵消，终止边界最近shape不证明全子步接触成因；target_counterpart仅指相机、相机支架、arm_seg4，夹爪另列。总episode含物理失败和超时，未结束episode仍计入环境秒。'}
    (OUTPUT_ROOT / 'comparison_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams['font.family'] = ['Noto Sans CJK JP']
    figure, axis = plt.subplots(figsize=(9, 5))
    positions = np.arange(4)
    for index, variant in enumerate(ASSETS):
        counts = [next(row['physical_failures'] for row in runs if row['variant'] == variant and row['seed'] == seed)
                  for seed in (1001, 1002, 1003)] + [totals[variant]['physical_failures']]
        bars = axis.bar(positions + (index - 1) * 0.25, counts, width=0.25, label=variant)
        axis.bar_label(bars, padding=3)
    axis.set_xticks(positions, ['seed1001', 'seed1002', 'seed1003', '三seed合计'])
    axis.set_ylabel('物理终止次数（每seed：128环境×60秒）')
    axis.set_title('相同1400轮冻结策略，S1／高斯采样，零更新')
    axis.legend()
    axis.grid(axis='y', alpha=0.3)
    axis.margins(y=0.15)
    figure.tight_layout()
    figure.savefig(OUTPUT_ROOT / 'comparison_failures.png', dpi=160)
    plt.close(figure)
    print(json.dumps(totals, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()

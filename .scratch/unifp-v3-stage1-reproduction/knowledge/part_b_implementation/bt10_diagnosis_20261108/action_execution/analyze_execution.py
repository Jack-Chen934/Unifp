"""复算同状态限幅分类、受力事件响应，并生成中文诊断图。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
font_manager.fontManager.addfont(FONT)
plt.rcParams['font.family'] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams['axes.unicode_minus'] = False


def analyze_run(folder: Path) -> dict[str, Any]:
    """Summarize one physical trajectory and its unexecuted candidate requests.

    Args:
        folder: Completed diagnostic directory containing summary.json/trace.npz.

    Returns:
        JSON-compatible statistics, reset-aware event windows, and representative
        traces. Saturation percentages use all physical substeps; force is N,
        distance is m, and controller torque is N m.

    Raises:
        AssertionError: Shapes, finite values, exact identities or counts differ.
        OSError: A completed diagnostic archive is unavailable.
    """
    summary = json.loads((folder / 'summary.json').read_text())
    with np.load(folder / 'trace.npz') as archive:
        data = {key: archive[key] for key in archive.files}
    assert all(np.isfinite(value).all() for value in data.values())
    assert data['saturated'].shape == (1000, 3, 128, 14)
    saturation = data['saturated']
    overlap = data['saturation_overlap']
    assert np.allclose(overlap.sum(1), 1.)
    assert np.allclose(overlap[:, 2] + overlap[:, 3], saturation[:, 0])
    assert np.allclose(overlap[:, 1] + overlap[:, 3], saturation[:, 1])
    assert np.allclose(saturation.mean((0, 2)), summary['saturation_by_variant_action'])
    torque_identity_error = float(np.max(np.abs(data['position_p'] + data['velocity_term'] - data['raw_torque'])))
    # Averaging eight float32 substeps can introduce small addition-order error.
    assert torque_identity_error < 1e-3
    names = summary['action_joint_names']
    joint2 = names.index('arm_joint2')
    result: dict[str, Any] = {
        'run': folder.name, 'iteration': summary['iteration'], 'mode': summary['mode'],
        'tcp_l2_cm': summary['tcp_l2_mean_m'] * 100.,
        'physical_failures': summary['physical_failures'],
        'torque_identity_max_abs_difference_nm': torque_identity_error,
        'saturation_by_variant_action_percent': (saturation.mean((0, 2)) * 100.).tolist(),
        'overlap_by_action_percent': (overlap.mean((0, 2)) * 100.).tolist(),
        'target_clip_by_variant_action_percent': ((np.abs(data['raw_target'] - data['target']) > 1e-6).mean((0, 2)) * 100.).tolist(),
        'joint2_interval_average_request_abs_nm': np.abs(data['raw_torque'][..., joint2]).mean((0, 2)).tolist(),
        'joint2_position_p_abs_nm': np.abs(data['position_p'][..., joint2]).mean((0, 2)).tolist(),
        'joint2_velocity_term_abs_nm': np.abs(data['velocity_term'][..., joint2]).mean((0, 2)).tolist(),
        'joint2_index': joint2, 'events': [],
    }
    result['action_clip_fraction_by_variant'] = [
        float((np.abs(data['action_mean']) >= 100.).mean()),
        float((np.abs(data['action_mean'] + data['noise'] * np.asarray(summary['std'])) >= 100.).mean()),
        float((np.abs(data['action_mean'] + data['noise'] * np.asarray(summary['earlier_std'])) >= 100.).mean()),
    ]
    actual_variant = 0 if summary['mode'] == 'mean' else 1
    actual_arm_saturation = saturation[:, actual_variant, :, 8:].max(-1) > 0
    tcp_error = np.linalg.norm(data['post_target'] - data['post_tcp'], axis=-1)
    for label, mask in (('any_arm_saturated', actual_arm_saturation), ('no_arm_saturation', ~actual_arm_saturation)):
        result[label] = {'policy_interval_count': int(mask.sum()),
            'policy_interval_fraction': float(mask.mean()),
            'tcp_l2_mean_cm': float(tcp_error[mask].mean() * 100.) if mask.any() else None}
    # Separate actual TCP motion from movement of the world-frame task target.
    start_error = data['pre_target'] - data['pre_tcp']
    tcp_delta = data['post_tcp'] - data['pre_tcp']
    target_delta = data['post_target'] - data['pre_target']
    error_delta = target_delta - tcp_delta
    error_identity = (np.square(data['post_target'] - data['post_tcp']).sum(-1)
        - np.square(start_error).sum(-1))
    error_decomposition = (2. * (start_error * target_delta).sum(-1)
        - 2. * (start_error * tcp_delta).sum(-1) + np.square(error_delta).sum(-1))
    result['squared_error_identity_max_abs_difference_m2'] = float(np.max(np.abs(error_identity - error_decomposition)))
    assert result['squared_error_identity_max_abs_difference_m2'] < 1e-4
    valid_error = np.linalg.norm(start_error, axis=-1) > 0.001
    toward = (tcp_delta * start_error).sum(-1)
    closing = ((tcp_delta - target_delta) * start_error).sum(-1)
    result['tcp_moves_toward_interval_start_target_percent'] = float((toward[valid_error] > 0).mean() * 100.)
    result['target_relative_closing_projection_percent'] = float((closing[valid_error] > 0).mean() * 100.)
    # Onsets retain the original independent external/command trapezoids.
    force_vectors = {'external': data['external_world'], 'command': data['command_yaw']}
    onsets: dict[str, np.ndarray] = {}
    same_episode = data['episode_id'][1:] == data['episode_id'][:-1]
    for stream, vectors in force_vectors.items():
        active = np.linalg.norm(vectors, axis=-1) > 1e-4
        mask = np.zeros_like(active)
        mask[1:] = active[1:] & ~active[:-1] & same_episode
        onsets[stream] = mask
    for stream in ('external', 'command'):
        other = 'command' if stream == 'external' else 'external'
        for step, env_id in np.argwhere(onsets[stream]):
            event = {'stream': stream, 'step': int(step), 'env_id': int(env_id),
                'local_step': int(data['local_step'][step, env_id]), 'windows': []}
            for horizon_s in (0.1, 0.5, 1.0):
                end = int(step + round(horizon_s / summary['policy_dt_s']) - 1)
                row: dict[str, Any] = {'horizon_s': horizon_s, 'available': end < 1000}
                if end < 1000:
                    selection = slice(step, end + 1)
                    crossed_reset = bool(data['done'][selection, env_id].any())
                    row.update({'crossed_reset': crossed_reset,
                        'contains_physical_failure': bool(data['terminated'][selection, env_id].any()),
                        'base_push': bool(data['base_push'][selection].any()),
                        'other_force_onset': bool(onsets[other][selection, env_id].any()),
                        'startup': event['local_step'] < 26})
                    if not crossed_reset:
                        # Baseline precedes the force onset; target movement includes
                        # both the task trajectory and the force compensation shift.
                        initial_tcp = data['pre_tcp'][step, env_id]
                        initial_target = data['post_target'][step - 1, env_id]
                        initial_error = initial_target - initial_tcp
                        onset_target = data['pre_target'][step, env_id]
                        onset_error = onset_target - initial_tcp
                        tcp_motion = data['post_tcp'][end, env_id] - initial_tcp
                        target_motion = data['post_target'][end, env_id] - onset_target
                        final_error = data['post_target'][end, env_id] - data['post_tcp'][end, env_id]
                        event_vectors = force_vectors[stream][selection, env_id]
                        peak_vector = event_vectors[np.argmax(np.linalg.norm(event_vectors, axis=-1))]
                        dominant_axis = int(np.argmax(np.abs(peak_vector)))
                        row.update({'initial_error_m': float(np.linalg.norm(initial_error)),
                            'final_error_m': float(np.linalg.norm(final_error)),
                            'error_improvement_m': float(np.linalg.norm(initial_error) - np.linalg.norm(final_error)),
                            'tcp_displacement_m': float(np.linalg.norm(data['post_tcp'][end, env_id] - initial_tcp)),
                            'target_displacement_m': float(np.linalg.norm(data['post_target'][end, env_id] - initial_target)),
                            'geometric_displacement_m': float(np.linalg.norm(data['post_geometric'][end, env_id] - data['post_geometric'][step - 1, env_id])),
                            'joint2_saturation_fraction': float(saturation[selection, 0 if summary['mode'] == 'mean' else 1, env_id, joint2].mean()),
                            'tcp_projection_toward_onset_target_m': float(np.dot(tcp_motion, onset_error) / max(np.linalg.norm(onset_error), 1e-8)),
                            'relative_projection_toward_onset_target_m': float(np.dot(tcp_motion - target_motion, onset_error) / max(np.linalg.norm(onset_error), 1e-8)),
                            'fixed_onset_target_error_improvement_m': float(np.linalg.norm(onset_error) - np.linalg.norm(onset_target - data['post_tcp'][end, env_id])),
                            'event_peak_norm_n': float(np.linalg.norm(force_vectors[stream][selection, env_id], axis=-1).max()),
                            'event_dominant_axis': dominant_axis,
                            'event_dominant_sign': 1 if peak_vector[dominant_axis] >= 0 else -1,
                            'other_force_active': bool((np.linalg.norm(force_vectors[other][selection, env_id], axis=-1) > 1e-4).any())})
                event['windows'].append(row)
            result['events'].append(event)
    result['event_summary'] = []
    result['isolated_direction_magnitude_groups'] = []
    result['representatives'] = []
    for stream in ('external', 'command'):
        for horizon_s in (0.1, 0.5, 1.0):
            windows = [window for event in result['events'] if event['stream'] == stream
                for window in event['windows'] if window['horizon_s'] == horizon_s]
            available = [window for window in windows if window['available']]
            survivors = [window for window in available if not window['crossed_reset'] and not window['startup']]
            clean = [window for window in survivors if not window['base_push'] and not window['other_force_onset']]
            report: dict[str, Any] = {'stream': stream, 'horizon_s': horizon_s, 'onsets': len(windows),
                'available': len(available), 'reset_crossing': sum(window['crossed_reset'] for window in available),
                'physical_failure_windows': sum(window['contains_physical_failure'] for window in available),
                'nonstartup_survivors': len(survivors), 'no_push_or_other_onset_survivors': len(clean)}
            if survivors:
                for name in ('initial_error_m', 'final_error_m', 'error_improvement_m',
                    'target_displacement_m', 'geometric_displacement_m', 'joint2_saturation_fraction',
                    'tcp_projection_toward_onset_target_m', 'relative_projection_toward_onset_target_m',
                    'fixed_onset_target_error_improvement_m'):
                    report[name + '_quartiles'] = np.quantile([window[name] for window in survivors], [.25, .5, .75]).tolist()
            isolated = [window for window in clean if not window['other_force_active']]
            report['isolated_force_survivors'] = len(isolated)
            if isolated:
                report['isolated_final_error_m_median'] = float(np.median([window['final_error_m'] for window in isolated]))
                report['isolated_fixed_onset_target_improvement_m_median'] = float(np.median([window['fixed_onset_target_error_improvement_m'] for window in isolated]))
                if horizon_s == 1.:
                    for axis in range(3):
                        for sign in (-1, 1):
                            for lower, upper in ((0., 30.), (30., 60.), (60., 110.)):
                                group = [window for window in isolated if window['event_dominant_axis'] == axis
                                    and window['event_dominant_sign'] == sign and lower <= window['event_peak_norm_n'] < upper]
                                if group:
                                    result['isolated_direction_magnitude_groups'].append({'stream': stream,
                                        'coordinate_frame': 'world' if stream == 'external' else 'yaw',
                                        'dominant_axis': 'XYZ'[axis], 'sign': sign,
                                        'force_norm_range_n': [lower, upper], 'count': len(group),
                                        'final_error_m_median': float(np.median([window['final_error_m'] for window in group])),
                                        'joint2_saturation_fraction_median': float(np.median([window['joint2_saturation_fraction'] for window in group]))})
            result['event_summary'].append(report)
        candidates = [(event, next(window for window in event['windows'] if window['horizon_s'] == 1.))
            for event in result['events'] if event['stream'] == stream]
        candidates = [(event, window) for event, window in candidates if window['available']
            and not window['crossed_reset'] and not window['startup'] and not window['base_push'] and not window['other_force_onset']]
        if candidates:
            candidates.sort(key=lambda item: item[1]['error_improvement_m'])
            event, window = candidates[len(candidates) // 2]
            step, env_id = event['step'], event['env_id']
            selection = slice(max(0, step - 10), step + 50)
            relative_tcp = data['post_tcp'][selection, env_id] - data['pre_tcp'][step, env_id]
            relative_target = data['post_target'][selection, env_id] - data['pre_tcp'][step, env_id]
            axis = int(np.argmax(np.ptp(relative_target, axis=0)))
            result['representatives'].append({'stream': stream, 'env_id': env_id, 'step': step, 'axis': axis,
                'time_s': ((np.arange(1000)[selection] - step + 1) * summary['policy_dt_s']).tolist(),
                'external_world_n': data['external_world'][selection, env_id].tolist(),
                'command_yaw_n': data['command_yaw'][selection, env_id].tolist(),
                'tcp_axis_m': relative_tcp[:, axis].tolist(), 'target_axis_m': relative_target[:, axis].tolist(),
                'error_m': np.linalg.norm(data['post_target'][selection, env_id] - data['post_tcp'][selection, env_id], axis=-1).tolist(),
                'joint2_saturation': saturation[selection, 0 if summary['mode'] == 'mean' else 1, env_id, joint2].tolist()})
    return result


def plot_results(reports: list[dict[str, Any]]) -> None:
    """Plot conditional saturation and representative original force responses.

    Args:
        reports: Analyzed completed runs, with JSON-compatible arrays and units.

    Note:
        Writes PNG files beside this script. Representative onsets are selected by
        median error improvement, independently per run, and are not paired trials.
    """
    labels = [f"{row['iteration']}\n{row['mode']}轨迹" for row in reports]
    positions = np.arange(len(reports))
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for variant, label in enumerate(('同状态均值请求', '同状态当前噪声请求', '同状态9000噪声请求')):
        values = [row['saturation_by_variant_action_percent'][variant][row['joint2_index']] for row in reports]
        axes[0].bar(positions + (variant - 1) * .25, values, width=.25, label=label)
    bottom = np.zeros(len(reports))
    for category, label in ((1, '仅采样请求饱和'), (2, '仅均值请求饱和'), (3, '两者均饱和')):
        values = np.asarray([row['overlap_by_action_percent'][category][row['joint2_index']] for row in reports])
        axes[1].bar(positions, values, bottom=bottom, label=label)
        bottom += values
    for axis in axes:
        axis.set(xticks=positions, xticklabels=labels, ylabel='全部物理子步占比（%）')
        axis.legend(fontsize=9)
        axis.grid(axis='y', alpha=.2)
    axes[0].set_title('arm_joint2：相同实际状态，不同动作请求')
    axes[1].set_title('arm_joint2：均值／采样限幅重叠分类')
    figure.tight_layout()
    figure.savefig(ROOT / '同状态力矩分解.png', dpi=150)
    plt.close(figure)
    for stream, stream_label in (('external', '实际外力'), ('command', '命令力')):
        figure, axes = plt.subplots(3, len(reports), figsize=(4.5 * len(reports), 10), squeeze=False)
        for column, report in enumerate(reports):
            representative = next((row for row in report['representatives'] if row['stream'] == stream), None)
            axes[0, column].set_title(f"{report['iteration']} {report['mode']}：{stream_label}起始")
            if representative is None:
                axes[1, column].text(.1, .5, '无符合条件的完整窗口')
                continue
            times = representative['time_s']
            for component, color in enumerate(('tab:red', 'tab:green', 'tab:blue')):
                axes[0, column].plot(times, np.asarray(representative['external_world_n'])[:, component], color=color, label=f'外力world {"XYZ"[component]}')
                axes[0, column].plot(times, np.asarray(representative['command_yaw_n'])[:, component], color=color, linestyle='--', label=f'命令yaw {"XYZ"[component]}')
            axes[0, column].set_ylabel('力（N）；坐标系见图例')
            axes[0, column].legend(fontsize=7, ncol=2)
            axes[1, column].plot(times, np.asarray(representative['tcp_axis_m']) * 100, label='TCP')
            axes[1, column].plot(times, np.asarray(representative['target_axis_m']) * 100, label='补偿目标')
            axes[1, column].set_ylabel(f'世界{"XYZ"[representative["axis"]]}相对位置（cm）')
            axes[1, column].legend(fontsize=8)
            axes[2, column].plot(times, np.asarray(representative['error_m']) * 100, label='TCP L2误差（cm）')
            axes[2, column].plot(times, np.asarray(representative['joint2_saturation']) * 100, label='joint2限幅（%）', alpha=.7)
            axes[2, column].set(xlabel='距受力起始时间（秒）', title=f"env={representative['env_id']}，step={representative['step']}")
            axes[2, column].legend(fontsize=8)
        for axis in axes.flat:
            axis.axvline(0., color='gray', linestyle=':')
            axis.grid(alpha=.2)
        figure.suptitle('各轨迹独立选取1秒误差变化中位事件；非同力流配对，不代表因果')
        figure.tight_layout()
        figure.savefig(ROOT / f'{stream_label}响应示例.png', dpi=140)
        plt.close(figure)


def main() -> None:
    """Analyze all completed diagnostic runs and write summaries and figures.

    Raises:
        AssertionError: Any independent numerical or archive check fails.
        OSError: A required completed-run input cannot be read.
    """
    folders = sorted((path for path in ROOT.glob('check_*') if (path / 'summary.json').exists()),
        key=lambda path: (int(path.name.split('_')[1]), path.name.split('_')[2]))
    reports = [analyze_run(folder) for folder in folders]
    (ROOT / 'analysis.json').write_text(json.dumps({'reports': reports,
        '说明': '同状态反事实只评估即时力矩；受力窗口是原任务中的观察，跨reset窗口单列，幸存窗口存在选择偏差。'},
        ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    if reports:
        plot_results(reports)
    print(json.dumps({'completed_runs': len(reports)}, ensure_ascii=False))


if __name__ == '__main__':
    main()

"""汇总固定输入命令敏感性，分别展示三个动作组与Encoder路径。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    """Pool real-input response statistics with their valid sample counts.

    Raises:
        AssertionError: The input data identity, expected record count or finite
            sensitivity statistics fail verification.
        OSError: Source response data or the installed CJK font cannot be read.

    Note:
        Writes JSON/PNG only. RMS is pooled by squared magnitudes; mean cosines
        are pooled by sample count. Neither statistic proves control correctness.
    """
    root = Path(__file__).resolve().parent
    source = json.loads((root / 'policy_response.json').read_text())
    assert len(source['datasets']) == 7 and len(source['records']) == 252
    aggregate: list[dict[str, Any]] = []
    rms_names = ('force_derivative_group_rms_per_n', 'equivalent_goal_derivative_group_rms_per_n',
        'positive_delta_total_group_rms', 'positive_delta_direct_group_rms', 'positive_delta_encoder_group_rms')
    for subset in ('all_valid', 'full_history_valid'):
        for delta in (1., 10.):
            for iteration in (9000, 10000, 12000):
                for axis in 'XYZ':
                    rows = [row for row in source['records'] if row['iteration'] == iteration
                        and row['axis'] == axis and row['delta_n'] == delta and row['subset'] == subset]
                    weights = np.asarray([row['samples'] for row in rows])
                    assert len(rows) == 7 and (weights > 0).all()
                    record: dict[str, Any] = {'subset': subset, 'delta_n': delta, 'iteration': iteration,
                        'axis': axis, 'samples': int(weights.sum())}
                    for name in rms_names:
                        record[name] = np.sqrt(np.average(np.square([row[name] for row in rows]), weights=weights, axis=0)).tolist()
                    record['force_goal_cosine_by_group'] = np.average([row['force_goal_cosine_by_group'] for row in rows], weights=weights, axis=0).tolist()
                    record['force_over_goal_group_rms_ratio'] = (np.asarray(record[rms_names[0]]) / np.asarray(record[rms_names[1]])).tolist()
                    record['per_probe_force_group_rms'] = {row['probe']: row[rms_names[0]] for row in rows}
                    assert all(np.isfinite(record[name]).all() for name in rms_names)
                    aggregate.append(record)
    result = {'aggregate': aggregate, 'source_samples': sum(row['samples'] for row in source['datasets']),
        '说明': '七组真实输入交叉三模型；不同probe是相关观测，统计不作独立置信区间或训练因果判断。'}
    (root / 'response_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
    font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    figure, axes = plt.subplots(2, 3, figsize=(14, 8))
    for column, group in enumerate(('转向', '驱动', '机械臂')):
        for axis in 'XYZ':
            rows = [row for row in aggregate if row['subset'] == 'full_history_valid' and row['delta_n'] == 1. and row['axis'] == axis]
            axes[0, column].plot([row['iteration'] for row in rows], [row['force_derivative_group_rms_per_n'][column] for row in rows], marker='o', label=f'命令{axis}')
            axes[1, column].plot([row['iteration'] for row in rows], [row['force_over_goal_group_rms_ratio'][column] for row in rows], marker='o', label=f'命令{axis}')
        axes[0, column].set(title=f'{group}：当前命令±1N响应', ylabel='动作RMS变化／N')
        axes[1, column].set(title=f'{group}：力命令／等效几何目标响应', ylabel='RMS比值（不要求为1）', xlabel='累计训练迭代')
    for axis in axes.flat:
        axis.grid(alpha=.2)
        axis.legend()
    figure.suptitle('固定真实输入，排除零填充历史；仅当前帧反事实，不是动力学验收')
    figure.tight_layout()
    figure.savefig(root / '命令敏感性.png', dpi=150)
    plt.close(figure)
    print(json.dumps({'aggregate_records': len(aggregate), 'source_samples': result['source_samples']}, ensure_ascii=False))


if __name__ == '__main__':
    main()

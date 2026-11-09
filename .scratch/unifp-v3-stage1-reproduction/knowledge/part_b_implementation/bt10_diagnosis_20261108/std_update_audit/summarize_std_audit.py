"""汇总标准差更新来源、配对零熵检查并生成中文图表。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np


def main() -> None:
    """Aggregate local artifacts without loading or updating any policy.

    Raises:
        OSError: Diagnostic JSON or plot font cannot be read or outputs cannot save.
        AssertionError: A decomposition, trial count or saved value is inconsistent.

    Note:
        Writes summary JSON and PNG next to this script. Trial ranges describe
        shuffling of one fixed rollout, not independent training-seed uncertainty.
    """
    root = Path(__file__).resolve().parent
    data = json.loads((root / 'std_updates.json').read_text())
    rows = []
    for result in data['results']:
        initial_std = np.array(result['point']['std'])
        original = np.array([trial['final_std'] for trial in result['trials']]) - initial_std
        zero_entropy = np.array([trial['zero_entropy_copy']['final_std'] for trial in result['trials']]) - initial_std
        contributions = {}
        for source in ('restored_momentum', 'rollout_surrogate', 'rollout_entropy'):
            contributions[source] = np.array([
                np.array([event['unrolled_conditional_delta'][source] for event in trial['events']]).sum(0)
                for trial in result['trials']])
        residual = float(np.abs(sum(contributions.values()) - original).max())
        assert residual < 2e-6
        assert original.shape == zero_entropy.shape == (3, 14)
        rows.append({'iteration': result['iteration'],
            'original_mean_delta_by_shuffle': original.mean(1).tolist(),
            'zero_entropy_mean_delta_by_shuffle': zero_entropy.mean(1).tolist(),
            'original_joint2_delta_by_shuffle': original[:, 9].tolist(),
            'zero_entropy_joint2_delta_by_shuffle': zero_entropy[:, 9].tolist(),
            'original_mean_delta': float(original.mean()),
            'zero_entropy_mean_delta': float(zero_entropy.mean()),
            'original_joint2_delta': float(original[:, 9].mean()),
            'zero_entropy_joint2_delta': float(zero_entropy[:, 9].mean()),
            'conditional_mean_delta': {key: float(value.mean()) for key, value in contributions.items()},
            'conditional_joint2_delta': {key: float(value[:, 9].mean()) for key, value in contributions.items()},
            'cumulative_reconstruction_max_error': residual,
            'event_reconstruction_max_error': max(event['unrolled_reconstruction_max_error']
                for trial in result['trials'] for event in trial['events']),
            'clip_factor_range': [min(event['clip_factor'] for trial in result['trials'] for event in trial['events']),
                                  max(event['clip_factor'] for trial in result['trials'] for event in trial['events'])],
            'restored_main_lr': result['restored_lr'],
            'std_adam_step_before': result['std_adam_step_before']})
    (root / 'summary.json').write_text(json.dumps({'rows': rows,
        '说明': '更新量是20次mini-batch后的参数变化；三种shuffle来自同一rollout，误差条仅为范围，不是统计置信区间。'},
        ensure_ascii=False, indent=2, allow_nan=False) + '\n')

    font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
    plt.rcParams['axes.unicode_minus'] = False
    figure, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    positions = np.arange(3)
    for axis, metric, title in zip(axes[:2], ('mean', 'joint2'), ('14维std平均更新量', 'arm_joint2 std更新量')):
        for offset, prefix, label, color in ((-.18, 'original', '原熵系数0.01', '#4279af'),
                                              (.18, 'zero_entropy', '内存副本熵系数0', '#d08640')):
            samples = np.array([row[f'{prefix}_{metric}_delta_by_shuffle'] for row in rows]) * 1e5
            means = samples.mean(1)
            ranges = np.stack((means - samples.min(1), samples.max(1) - means))
            axis.bar(positions + offset, means, .34, yerr=ranges, capsize=4, label=label, color=color)
        axis.set_title(title, fontproperties=font)
        axis.set_ylabel('std变化 × 10^-5', fontproperties=font)
        axis.legend(prop=font, fontsize=8)
    source_labels = (('restored_momentum', '存档继承动量', '#7c82ae'),
                     ('rollout_surrogate', '本轮策略梯度及传播', '#72a56f'),
                     ('rollout_entropy', '本轮熵梯度及传播', '#d08640'))
    for index, (key, label, color) in enumerate(source_labels):
        axes[2].bar(positions + (index - 1) * .24,
                    [row['conditional_mean_delta'][key] * 1e5 for row in rows], .23, label=label, color=color)
    axes[2].set_title('原更新的条件分解：14维平均', fontproperties=font)
    axes[2].set_ylabel('std变化 × 10^-5', fontproperties=font)
    axes[2].legend(prop=font, fontsize=8)
    for axis in axes:
        axis.axhline(0, color='black', linewidth=.7)
        axis.set_xticks(positions, [str(row['iteration']) for row in rows])
        axis.set_xlabel('存档累计训练迭代', fontproperties=font)
        axis.grid(axis='y', alpha=.2)
    figure.suptitle('真实固定rollout；每存档三种shuffle；内存更新，不是闭环训练效果', fontproperties=font)
    figure.savefig(root / '标准差更新来源.png', dpi=160)
    plt.close(figure)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

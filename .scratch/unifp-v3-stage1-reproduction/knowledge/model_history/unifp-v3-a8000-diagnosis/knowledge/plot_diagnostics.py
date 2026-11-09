"""Plot the retained four-motion, six-condition diagnostic summaries."""

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


root = Path(__file__).resolve().parent
matplotlib.rcParams.update({'font.family': 'Noto Sans CJK SC', 'axes.unicode_minus': False})
conditions = [
    ('wheels_gain1', '原资产／原增益', '#777777'),
    ('wheels_gain4', '原资产／增益×4', '#999999'),
    ('wheels_gain16', '原资产／增益×16', '#D8953E'),
    ('wheels_no_self', '原资产／关闭自碰撞', '#B0B0B0'),
    ('wheels_round', '光滑代理／关闭自碰撞', '#419B78'),
    ('wheels_round_gain4', '光滑代理／关闭自碰撞／增益×4', '#3979AE'),
]
reports = {name: json.loads((root / (name + '.json')).read_text()) for name, _, _ in conditions}
figure, axes = plt.subplots(2, 2, figsize=(15, 8.5), constrained_layout=True)
motions = [('forward', 0, 0.3, '前进速度', 'm/s'),
           ('backward', 0, -0.3, '后退速度', 'm/s'),
           ('lateral', 1, 0.2, '侧移速度', 'm/s'),
           ('yaw', 2, 0.3, '原地偏航角速度', 'rad/s')]
for axis, (case, component, target, title, unit) in zip(axes.flat, motions):
    values = [reports[name]['cases'][case]['body_velocity_mean'][component] for name, _, _ in conditions]
    positions = np.arange(len(conditions))
    axis.barh(positions, values, color=[color for _, _, color in conditions], height=0.6)
    axis.set_yticks(positions, [label for _, label, _ in conditions], fontsize=9)
    axis.invert_yaxis()
    axis.axvline(0, color='#777777', linewidth=0.7)
    axis.axvline(target, color='#333333', linestyle='--', linewidth=1)
    axis.set_xlim((-0.38, 0.04) if target < 0 else (-0.025, 0.38))
    for position, value in zip(positions, values):
        # Tiny negative values in positive-command panels need labels right of zero.
        label_x = max(value, 0.0) + 0.005 if target > 0 else value - 0.005
        axis.text(label_x, position, f'{value:.4f}',
                  va='center', ha='left' if target > 0 else 'right', fontsize=9)
    axis.set_title(f'{title}（虚线目标 {target:g} {unit}）', fontsize=12)
    axis.set_xlabel(unit)
    axis.grid(axis='x', alpha=0.18)
    axis.set_axisbelow(True)
    axis.spines[['top', 'right']].set_visible(False)
figure.suptitle('V3 带载轮系诊断：4环境，解析轮命令，8秒试验的后4秒均值\n光滑球代理仅用于隔离原因；原地旋转仍未解决', fontsize=14)
figure.savefig(root / 'wheel_diagnostics.png', dpi=180)
plt.close(figure)

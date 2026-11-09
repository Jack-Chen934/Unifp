"""根据归档指标重绘12000轮检查曲线，不执行仿真或训练。"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


REPORT_ROOT = Path(__file__).resolve().parent
BASELINE_PATH = REPORT_ROOT.parents[1] / "bt07/a_zero_regression_comparison.json"
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
font_manager.fontManager.addfont(FONT_PATH)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT_PATH).get_name()
plt.rcParams["axes.unicode_minus"] = False

iteration_rows = [json.loads(line) for line in (REPORT_ROOT / "iterations.jsonl").read_text().splitlines()]
figure, axes = plt.subplots(2, 2, figsize=(12, 8))
metric_specs = [
    ("episode/reward_rate", "训练奖励率", 1.0),
    ("episode/errors/tcp_compensated_l2_m/mean", "补偿末端误差（cm）", 100.0),
    ("auxiliary/force_ee_loss", "末端力辅助损失", 1.0),
    ("collection_s", "采样耗时（秒／轮）", 1.0),
]
for axis, (metric_name, label, scale) in zip(axes.flat, metric_specs):
    # Each displayed point averages 100 complete training iterations, excluding absent metrics.
    iteration_centers, metric_means = [], []
    for offset in range(0, len(iteration_rows), 100):
        window_rows = iteration_rows[offset:offset + 100]
        metric_values = [row[metric_name] * scale for row in window_rows if metric_name in row]
        if metric_values:
            iteration_centers.append(np.mean([row["completed_iterations"] for row in window_rows]))
            metric_means.append(np.mean(metric_values))
    axis.plot(iteration_centers, metric_means)
    axis.set(xlabel="累计训练迭代", ylabel=label)
    axis.grid(alpha=0.3)
figure.suptitle("B阶段8001—12000轮：每100轮均值；训练样本分布与零力评估不同")
figure.tight_layout()
figure.savefig(REPORT_ROOT / "training_trends.png", dpi=160)
plt.close(figure)

baseline = json.loads(BASELINE_PATH.read_text())["scenarios"]
scenario_names = ["S0", "S1", "S2"]
baseline_cm = [baseline[name]["summary"]["metrics"]["tcp_geometric_l2_m"]["mean"] * 100
               for name in scenario_names]
current_cm = [json.loads((REPORT_ROOT / f"{name}_summary.json").read_text())
              ["metrics"]["tcp_geometric_l2_m"]["mean"] * 100 for name in scenario_names]
scenario_positions = np.arange(3)
figure, axis = plt.subplots(figsize=(8, 4.5))
axis.bar(scenario_positions - 0.18, baseline_cm, width=0.36, label="A8000")
axis.bar(scenario_positions + 0.18, current_cm, width=0.36, label="B累计12000")
axis.set(xticks=scenario_positions, xticklabels=scenario_names, ylabel="末端L2均值（cm）",
         title="同协议零力回归：每组96个episode，均无物理失败")
axis.legend()
figure.tight_layout()
figure.savefig(REPORT_ROOT / "zero_force_comparison.png", dpi=160)
plt.close(figure)

"""从只读诊断输出汇总动作、噪声与失败，不运行仿真。"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


ROOT = Path(__file__).resolve().parent
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
font_manager.fontManager.addfont(font_path)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams["axes.unicode_minus"] = False
checkpoint_statistics = json.loads((ROOT / "checkpoint_statistics.json").read_text())
conditions = []
for directory in sorted(ROOT.glob("probe_*")):
    if not directory.is_dir() or not (directory / "summary.json").exists():
        continue
    summary = json.loads((directory / "summary.json").read_text())
    rows = [json.loads(line) for line in (directory / "steps.jsonl").read_text().splitlines()]
    assert len(rows) == summary["steps"] == 1000
    summary["tcp_l2_all_transitions_cm"] = float(np.mean([row["compensated_l2_before_reset_m"] for row in rows]) * 100)
    for name in ("action_mean_abs", "action_abs", "action_change_square_mean", "mean_change_square_mean", "torque_limit_fraction", "joint_velocity_rms_rad_s"):
        values = np.mean([row[name] for row in rows], axis=0)
        summary[name + "_by_action"] = values.tolist()
        summary[name + "_group_sum"] = [float(values[start:end].sum()) for start, end in ((0, 4), (4, 8), (8, 14))]
    summary["action_clip_fraction"] = float(np.mean([row["action_clip_fraction"] for row in rows]))
    conditions.append(summary)
output = {"checkpoint_statistics": checkpoint_statistics, "conditions": conditions,
          "说明": "诊断各128环境×20秒，无更新。失败比例按完成episode，固定时间内多次失败可重复计数，不是每环境至少一次失败率。"}
(ROOT / "probe_comparison.json").write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

figure, axes = plt.subplots(1, 3, figsize=(13, 4))
for group, label in enumerate(("转向", "驱动", "机械臂")):
    axes[0].plot([row["iteration"] for row in checkpoint_statistics["checkpoints"]],
                 [row["std_group_mean"][group] for row in checkpoint_statistics["checkpoints"]], marker="o", label=label)
axes[0].set(title="探索标准差持续增大", xlabel="累计迭代", ylabel="组内平均std（无量纲动作）")
for mode, label in (("mean", "均值动作"), ("sample", "采样动作")):
    chosen = sorted([row for row in conditions if row["mode"] == mode], key=lambda row: row["checkpoint_iteration"])
    axes[1].plot([row["checkpoint_iteration"] for row in chosen], [row["tcp_l2_all_transitions_cm"] for row in chosen], marker="o", label=label)
    axes[2].plot([row["checkpoint_iteration"] for row in chosen], [row["failure_fraction_of_ended"] * 100 for row in chosen], marker="o", label=label)
for row in conditions:
    if row["mode"] == "sample_std9000":
        axes[1].scatter([12000], [row["tcp_l2_all_transitions_cm"]], marker="*", s=160, label="12000均值＋9000噪声")
        axes[2].scatter([12000], [row["failure_fraction_of_ended"] * 100], marker="*", s=160, label="12000均值＋9000噪声")
axes[1].set(title="同训练分布的新episode诊断", xlabel="累计迭代", ylabel="补偿TCP L2均值（cm）")
axes[2].set(title="物理失败（非正式验收配额）", xlabel="累计迭代", ylabel="占已结束episode比例（%）")
for axis in axes:
    axis.grid(alpha=.25)
    axis.legend(fontsize=8)
figure.tight_layout()
figure.savefig(ROOT / "policy_diagnosis.png", dpi=170)
plt.close(figure)
for row in conditions:
    print(row["checkpoint_iteration"], row["mode"], round(row["tcp_l2_all_transitions_cm"], 3), row["ended_episodes"],
          row["failure_predicates_overlapping"], "action_change", row["action_change_square_mean_group_sum"])

# Compare only the frozen A/B protocol pairs; official robot data use another scene.
reference_force = json.loads((ROOT / "A8000_R2_summary.json").read_text())
current_force = json.loads((ROOT / "R2_summary.json").read_text())
figure, axes = plt.subplots(1, 3, figsize=(13, 4))
for index, (report, label) in enumerate(((reference_force, "A8000"), (current_force, "B12000"))):
    axes[0].bar(index, report["tracking"]["nonzero_task"]["tcp_compensated_l2_m"]["mean"] * 100, label=label)
    axes[1].bar(np.arange(3) + (index - .5) * .35,
                report["force_estimation"]["nonzero"]["axis_mae_n"], width=.35, label=label)
    contact = json.loads((ROOT / ("A8000_contact_summary.json" if index == 0 else "contact_summary.json")).read_text())
    zero = json.loads((ROOT / ("A8000_zero_control_summary.json" if index == 0 else "zero_control_summary.json")).read_text())
    axes[2].bar(np.arange(3) + (index - .5) * .35,
                [zero["amplitudes"]["zero"]["all_measurable_normal_error_mean_n"]]
                + [contact["amplitudes"][key]["all_measurable_normal_error_mean_n"] for key in ("half", "full")],
                width=.35, label=label)
axes[0].set(title="R2受力段跟踪退化", ylabel="补偿TCP L2（cm）", xticks=[0, 1], xticklabels=["A8000", "B12000"])
axes[1].set(title="R2非零外力估计改善", ylabel="估计绝对误差（N）", xticks=[0, 1, 2], xticklabels=["世界X", "世界Y", "世界Z"])
axes[2].set(title="固定墙同轨迹：零命令／按压力", ylabel="稳态法向绝对误差（N）", xticks=[0, 1, 2], xticklabels=["0 N对照", "10 N", "20 N"])
for axis in axes:
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=.25)
figure.tight_layout()
figure.savefig(ROOT / "evaluation_diagnosis.png", dpi=170)
plt.close(figure)

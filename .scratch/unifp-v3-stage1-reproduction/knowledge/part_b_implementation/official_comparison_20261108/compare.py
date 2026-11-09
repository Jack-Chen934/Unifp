"""从冻结日志重算B4000对比并绘图；不启动仿真或训练。"""

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


REPORT_ROOT = Path(__file__).resolve().parent
PART_B_ROOT = REPORT_ROOT.parent
PROJECT_ROOT = REPORT_ROOT.parents[4]
OFFICIAL_ROOT = Path(
    "/home/sorry/python_ws/interactive/research/literature/projects/unifiedforce/reproduction"
)
OFFICIAL_DATA = OFFICIAL_ROOT / "outputs/stage2_20261029/followup4000/analysis4000"
V3_DATA = PART_B_ROOT / "bt08/check_12000_20261108"


def load_json(path: Path) -> Any:
    """Read a frozen JSON artifact without changing its contents.

    Args:
        path: Absolute source artifact path.

    Returns:
        Decoded artifact, preserving its original units and keys.

    Raises:
        OSError: The source cannot be read.
        ValueError: The source is not valid JSON.
    """
    return json.loads(path.read_text())


def summarize_window(start: int, end: int) -> dict[str, Any]:
    """Align inclusive B iteration windows and retain each metric's denominator.

    Args:
        start: First B-only committed iteration, with 1 meaning cumulative 8001.
        end: Last B-only committed iteration, at most 4000.

    Returns:
        Official all-transition means and V3 completed-episode statistics.
        The V3 step weighting excludes episodes not yet completed at the boundary.
    """
    official_window = [row for row in official_rows if start <= row["completed_iterations"] - 8000 <= end]
    v3_window = [row for row in v3_rows if start <= row["completed_iterations"] - 8000 <= end]
    episode_blocks = [row for row in episode_summary["blocks100"] if start <= row["start"] <= end]
    # Each source block is exactly 100 committed iterations; no partial blocks are mixed.
    assert (start - 1) % 100 == 0 and end % 100 == 0
    episode_count = sum(row["episodes"] for row in episode_blocks)
    episode_steps = sum(row["steps"] for row in episode_blocks)
    official_resets = sum(row["resets"] for row in official_window)
    raw_names = episode_blocks[0]["raw_step_sums"]
    weighted_raw = {
        name: sum(row["raw_step_sums"][name] for row in episode_blocks) / episode_steps
        for name in raw_names
    }
    return {
        "b_iteration_range": [start, end],
        "official": {
            "return_mean": float(np.mean([row["mean_return"] for row in official_window])),
            "compensated_l2_cm_all_transitions": float(np.mean([row["mean_compensated_l2_m"] for row in official_window]) * 100),
            "force_axis_mae_n": np.mean([row["force_estimator_axis_mae_N"] for row in official_window], axis=0).tolist(),
            "non_timeout_resets": sum(row["non_timeout_resets"] for row in official_window),
            "resets": official_resets,
            "failure_fraction": sum(row["non_timeout_resets"] for row in official_window) / official_resets,
        },
        "v3": {
            "reward_rate_iteration_mean": float(np.mean([row["episode/reward_rate"] for row in v3_window if "episode/reward_rate" in row])),
            "compensated_l2_cm_iteration_mean": float(np.mean([row["episode/errors/tcp_compensated_l2_m/mean"] for row in v3_window if "episode/reward_rate" in row]) * 100),
            "compensated_l2_cm_completed_step_weighted": sum(row["error_step_sum_m"] for row in episode_blocks) / episode_steps * 100,
            "reward_rate_completed_step_weighted": sum(row["reward_return_sum"] for row in episode_blocks) / sum(row["duration_sum_s"] for row in episode_blocks),
            "physical_failures": sum(row["physical_failures"] for row in episode_blocks),
            "episodes": episode_count,
            "failure_fraction": sum(row["physical_failures"] for row in episode_blocks) / episode_count,
            "weighted_raw_rewards": weighted_raw,
            "weighted_reward_contributions_per_s": {name: value * reward_weights[name] for name, value in weighted_raw.items()},
            "force_ee_auxiliary_loss": float(np.mean([row["auxiliary/force_ee_loss"] for row in v3_window])),
        },
    }


official_rows = [json.loads(line) for line in (OFFICIAL_DATA / "physical_metrics.jsonl").read_text().splitlines()]
v3_rows = [json.loads(line) for line in (V3_DATA / "iterations.jsonl").read_text().splitlines()]
episode_summary = load_json(REPORT_ROOT / "v3_episode_blocks100.json")
formal_config = load_json(PART_B_ROOT / "bt08/final_evidence_r06/formal_config_snapshot.json")
reward_weights = formal_config["environment"]["rewards"]["scales"]
for rows in (official_rows, v3_rows):
    assert [row["completed_iterations"] for row in rows] == list(range(8001, 12001))
assert all(row["samples"] == 4096 * 24 for row in official_rows)
assert sum(row["episodes"] for row in episode_summary["blocks100"]) == episode_summary["line_count"]
assert episode_summary["committed_clock_range"] == [8003, 12000]

blocks100 = [summarize_window(start, start + 99) for start in range(1, 4001, 100)]
windows = [summarize_window(start, end) for start, end in [(1, 100), (201, 400), (3801, 4000), (1, 1000), (1001, 2000), (2001, 3000), (3001, 4000), (801, 1000)]]
for block in blocks100:
    weighted_metrics = block["v3"]
    assert abs(sum(weighted_metrics["weighted_reward_contributions_per_s"].values()) - weighted_metrics["reward_rate_completed_step_weighted"]) < 1e-5
official_evaluations = load_json(OFFICIAL_DATA / "summary.json")["evaluations"]
official_a = load_json(OFFICIAL_ROOT / "outputs/stage1_20261029/data/eval_8000.json")["all_steps"]
v3_evaluations = load_json(V3_DATA / "check_summary.json")["zero_force_comparison"]

# Recompute the official evaluation L2 directly from its archived per-transition errors.
npz_path = OFFICIAL_ROOT / "outputs/stage2_20261029/official_comparison/data/no_force_model12000.npz"
with np.load(npz_path) as raw_evaluation:
    verified_l2_m = float(np.linalg.norm(raw_evaluation["compensated_error_xyz_m"], axis=-1).mean())
assert abs(verified_l2_m - official_evaluations["no_force"]["mean_compensated_l2_m"]) < 1e-7

zero_force_rows = [{
    "name": "官方代码复现", "a_cm": official_a["mean_l2_m"] * 100,
    "b_cm": verified_l2_m * 100,
}]
zero_force_rows.extend({"name": f"V3 {name}", "a_cm": row["a8000_tcp_l2_cm"], "b_cm": row["b12000_tcp_l2_cm"]} for name, row in v3_evaluations.items())
for row in zero_force_rows:
    row["increase_percent"] = (row["b_cm"] / row["a_cm"] - 1) * 100

source_paths = [
    OFFICIAL_DATA / "physical_metrics.jsonl", OFFICIAL_DATA / "summary.json",
    OFFICIAL_ROOT / "train_stage2.py", OFFICIAL_ROOT / "evaluate_stage2.py",
    OFFICIAL_ROOT / "outputs/stage2_20261029/run_manifest.json",
    OFFICIAL_ROOT / "outputs/stage2_20261029/launch_record.json",
    OFFICIAL_ROOT / "outputs/stage1_20261029/source/legged_robot_b2z1_pos_force.py",
    OFFICIAL_ROOT / "outputs/stage1_20261029/data/eval_8000.json", npz_path,
    V3_DATA / "iterations.jsonl", V3_DATA / "check_summary.json",
    REPORT_ROOT / "v3_episode_blocks100.json",
    PART_B_ROOT / "bt08/final_evidence_r06/formal_config_snapshot.json",
    PROJECT_ROOT / "UniFP_v3/legged_gym/envs/v3/legged_robot_v3.py",
    PROJECT_ROOT / "UniFP_v3/legged_gym/b2_gym_learn/ppo_cse_pf/v3_episode_log.py",
    PROJECT_ROOT / "UniFP_v3/legged_gym/b2_gym_learn/ppo_cse_pf/v3_on_policy_runner.py",
]
summary = {
    "comparison_date": "2026-11-08", "cumulative_iterations": [8001, 12000],
    "transitions_per_experiment": 4000 * 4096 * 24,
    "windows": windows, "blocks100": blocks100, "zero_force": zero_force_rows,
    "official_b12000_evaluation": official_evaluations,
    "v3_episode_source": {key: value for key, value in episode_summary.items() if key != "blocks100"},
    "limitations": [
        "官方档案位于本机，原启动记录指向远程gpuf3；不是论文作者原始实验数据。",
        "奖励组成和分母不同，归一化仅表示各自相对早期趋势。",
        "官方误差覆盖全部transition；V3加权误差覆盖窗口内结束episode的完整历史，跨窗口且右端删失。",
        "失败率分母均为已结束episode，但官方和V3终止判据不同。",
        "独立评估地形、目标、push和配额不同；不能据绝对值做严格模型排名。",
        "V3没有有效B12000受力评估归档，辅助MSE不可当作牛顿MAE。",
    ],
    "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
}
(REPORT_ROOT / "comparison_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
font_manager.fontManager.addfont(font_path)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams["axes.unicode_minus"] = False
centers = np.arange(50.5, 4001, 100)
figure, axes = plt.subplots(2, 2, figsize=(13, 8))
official_reward = np.array([block["official"]["return_mean"] for block in blocks100])
v3_reward = np.array([block["v3"]["reward_rate_iteration_mean"] for block in blocks100])
axes[0, 0].plot(centers, official_reward / official_reward[0] * 100, label="官方：episode回报")
axes[0, 0].plot(centers, v3_reward / v3_reward[0] * 100, label="V3：episode奖励率")
axes[0, 0].set(ylabel="各自前100轮均值＝100", title="奖励相对趋势（组成不同）")
axes[0, 0].axvline(1000, color="gray", linestyle=":", alpha=.6)
axes[0, 0].annotate("官方进程重启\n回报缓存清空", xy=(1050, 104), xytext=(1400, 102), fontsize=8, arrowprops={"arrowstyle": "->", "color": "gray"})
axes[0, 1].plot(centers, [block["official"]["compensated_l2_cm_all_transitions"] for block in blocks100], label="官方：所有采样步")
axes[0, 1].plot(centers, [block["v3"]["compensated_l2_cm_completed_step_weighted"] for block in blocks100], label="V3：结束episode按步加权")
axes[0, 1].plot(centers, [block["v3"]["compensated_l2_cm_iteration_mean"] for block in blocks100], "--", alpha=.5, label="V3：原训练日志均值")
axes[0, 1].set(ylabel="补偿TCP L2均值（cm）", title="物理误差趋势（采样口径不同）")
for key, label in [("official", "官方：非timeout"), ("v3", "V3：物理失败")]:
    axes[1, 0].plot(centers, [block[key]["failure_fraction"] * 100 for block in blocks100], label=label)
axes[1, 0].set(ylabel="占已结束episode比例（%）", title="失败趋势（终止判据不同）")
for axis_index, axis_name in enumerate("XYZ"):
    axes[1, 1].plot(centers, [block["official"]["force_axis_mae_n"][axis_index] for block in blocks100], label=f"官方 {axis_name}")
axes[1, 1].set(ylabel="力估计MAE（N）", title="官方力估计；V3暂无相同口径")
for axis in axes.flat:
    axis.set_xlabel("B阶段新增训练迭代（两者每轮4096×24步）")
    axis.grid(alpha=.25)
    axis.legend(fontsize=8)
figure.suptitle("官方代码复现与V3：同预算B4000，每100轮聚合，均从各自A8000接续")
figure.tight_layout()
figure.savefig(REPORT_ROOT / "training_comparison.png", dpi=170)
plt.close(figure)

figure, axis = plt.subplots(figsize=(10, 5))
positions = np.arange(len(zero_force_rows))
axis.bar(positions - .18, [row["a_cm"] for row in zero_force_rows], .36, label="各自A8000")
bars = axis.bar(positions + .18, [row["b_cm"] for row in zero_force_rows], .36, label="各自B累计12000")
for bar, row in zip(bars, zero_force_rows):
    axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + .15, f"+{row['increase_percent']:.0f}%", ha="center")
axis.set(xticks=positions, xticklabels=[row["name"] for row in zero_force_rows], ylabel="零力TCP L2均值（cm）", ylim=(0, 11), title="相对各自A基线的精度变化；官方与V3评估协议不同")
axis.legend()
axis.grid(axis="y", alpha=.2)
figure.tight_layout()
figure.savefig(REPORT_ROOT / "zero_force_retention.png", dpi=170)
plt.close(figure)
print(json.dumps({"windows": windows[:3], "zero_force": zero_force_rows}, ensure_ascii=False, indent=2))

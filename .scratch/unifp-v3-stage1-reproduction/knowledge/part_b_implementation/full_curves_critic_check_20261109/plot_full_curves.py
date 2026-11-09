"""绘制正式A+B全程曲线，并核验恢复边界和奖励分解。"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

ROOT = Path(__file__).resolve().parent


def decorate(axes: np.ndarray, last_iteration: int) -> None:
    """Mark stage and resume boundaries on all shared training axes.

    Args:
        axes: Flattenable array of Matplotlib axes.
        last_iteration: Last committed iteration represented in this snapshot.

    Note:
        Changes axis styling in place; iteration 8000 ends stage A.
    """
    for axis in axes.flat:
        axis.axvline(8000, color="#bc4749", linestyle="--", linewidth=1.2)
        for boundary in (767, 12000):
            axis.axvline(boundary, color="grey", linestyle=":", alpha=0.65)
        axis.axvspan(8000, last_iteration, color="#e9c46a", alpha=0.08)
        axis.set_xlim(1, last_iteration)
        axis.grid(alpha=0.22)
        axis.set_xlabel("累计训练迭代")


def iteration_curve(axis: Any, rows: list[dict[str, Any]], key: str,
                    label: str, color: str, logarithmic: bool = False) -> None:
    """Draw original iterations plus non-overlapping 100-iteration means.

    Args:
        axis: Destination Matplotlib axis.
        rows: Consecutive committed iteration records.
        key: Scalar training-log field, excluding episode-weighted summaries.
        label: Chinese legend text.
        color: Matplotlib color specification.
        logarithmic: Whether to use a logarithmic vertical axis.
    """
    iterations = np.array([row["completed_iterations"] for row in rows])
    values = np.array([row[key] for row in rows])
    axis.plot(iterations, values, color=color, alpha=0.14, linewidth=0.45)
    blocks = [slice(start, min(start + 100, len(rows))) for start in range(0, len(rows), 100)]
    axis.plot([iterations[block].mean() for block in blocks],
              [values[block].mean() for block in blocks], color=color, label=label)
    if logarithmic:
        axis.set_yscale("log")


def main() -> None:
    """Validate source continuity and write figures, CSV and verification JSON.

    Raises:
        AssertionError: Counters, stage identities, finiteness or rewards disagree.

    Note:
        Reads archived snapshots only; formal training is never modified.
    """
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font_manager.fontManager.addfont(font_path)
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=font_path).get_name(),
                         "axes.unicode_minus": False, "font.size": 10})
    snapshot_path = ROOT / "training_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text())
    runs = snapshot["runs"]
    rows = [row for run in runs for row in run["iterations"]]
    last_iteration = rows[-1]["completed_iterations"]
    assert [row["completed_iterations"] for row in rows] == list(range(1, last_iteration + 1))
    for row in rows:
        assert row["training_policy_steps"] == 24 * row["completed_iterations"]
        assert row["training_transitions"] == 4096 * row["training_policy_steps"]
        assert all(np.isfinite(value) for value in row.values() if isinstance(value, (int, float)))
    for first, second in ((0, 1), (2, 3)):
        for manifest in ("asset_manifest", "source_manifest"):
            assert runs[first]["metadata"]["identity"][manifest] == runs[second]["metadata"]["identity"][manifest]
    assert all(run["metadata"]["identity"]["asset_manifest"] == runs[0]["metadata"]["identity"]["asset_manifest"] for run in runs)
    assert not runs[-1]["failure_json_exists"]
    assert sum(run["episode_selected_count"] for run in runs) == sum(
        window["episodes"] for run in runs for window in run["episode_windows_100"])

    # Merge the split 701–800 window before forming any failure ratio or weighted mean.
    merged: dict[int, dict[str, Any]] = {}
    reward_reconstruction_errors = []
    for run in runs:
        weights = run["metadata"]["configured_reward_weights"]
        for window in run["episode_windows_100"]:
            bucket = (window["first"] - 1) // 100
            target = merged.setdefault(bucket, {"first": window["first"], "last": window["last"],
                **{key: 0.0 for key in ("episodes", "failures", "steps", "duration_s", "reward_sum")},
                "errors_step_sum": {}, "reward_raw_step_sum": {}})
            target["first"] = min(target["first"], window["first"])
            target["last"] = max(target["last"], window["last"])
            for key in ("episodes", "failures", "steps", "duration_s", "reward_sum"):
                target[key] += window[key]
            for group in ("errors_step_sum", "reward_raw_step_sum"):
                for key, value in window[group].items():
                    target[group][key] = target[group].get(key, 0.0) + value
            raw_rate = sum(weights[key] * value / window["steps"]
                           for key, value in window["reward_raw_step_sum"].items())
            reward_reconstruction_errors.append(abs(raw_rate - window["reward_sum"] / window["duration_s"]))
    assert max(reward_reconstruction_errors) < 1e-4
    windows = [merged[key] for key in sorted(merged)]
    weights = runs[-1]["metadata"]["configured_reward_weights"]
    assert all(run["metadata"]["configured_reward_weights"] == weights for run in runs)
    table = []
    for window in windows:
        table.append({"first": window["first"], "last": window["last"],
            "episodes": int(window["episodes"]), "failures": int(window["failures"]),
            "failure_pct": 100 * window["failures"] / window["episodes"],
            "mean_duration_s": window["duration_s"] / window["episodes"],
            "reward_rate": window["reward_sum"] / window["duration_s"],
            **{key: value / window["steps"] for key, value in window["errors_step_sum"].items()},
            **{"contribution/" + key: weights[key] * value / window["steps"]
               for key, value in window["reward_raw_step_sum"].items()}})
    with (ROOT / "完整100轮窗口.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    abscissa = np.array([(row["first"] + row["last"]) / 2 for row in table])

    # Episode curves use step/time weights, avoiding means of short failed episodes.
    figure, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
    series = [("reward_rate", "奖励率（奖励／仿真秒）", 1),
              ("tcp_compensated_l2_m", "补偿TCP L2误差（cm）", 100),
              ("base_planar_speed_m_s", "底盘平面速度误差（m/s）", 1),
              ("base_yaw_rate_rad_s", "底盘偏航角速度误差（rad/s）", 1),
              ("failure_pct", "物理失败／完成episode（%）", 1),
              ("mean_duration_s", "完成episode平均时长（s）", 1)]
    for axis, (key, title, multiplier) in zip(axes.flat, series):
        axis.plot(abscissa, [row[key] * multiplier for row in table], color="#277da8", label="补偿TCP L2" if key == "tcp_compensated_l2_m" else "100轮窗口")
        axis.set_title(title)
    axes[0, 1].plot(abscissa, [100 * row["tcp_geometric_l2_m"] for row in table],
                    color="#e76f51", alpha=0.8, label="几何TCP L2")
    axes[0, 1].legend()
    decorate(axes, last_iteration)
    figure.suptitle(f"V6 r2 seed1正式训练全程：1—{last_iteration}轮\n红虚线：A→B；灰点线：恢复；末窗口不足100轮；训练含高斯采样")
    figures = [("完整训练曲线", figure)]

    figure, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
    for axis, key, label, logarithmic in [
        (axes[0, 0], "value_loss", "Critic value loss", True),
        (axes[0, 1], "surrogate_loss", "PPO surrogate loss", False),
        (axes[1, 0], "auxiliary_loss", "辅助总损失", True),
        (axes[2, 0], "main_learning_rate", "主优化器学习率", True)]:
        iteration_curve(axis, rows, key, label, "#277da8", logarithmic)
        axis.set_title(label)
    for key, label, color in [("base_velocity", "底盘速度", "#277da8"),
                              ("tcp_position", "TCP位置", "#f4a261"),
                              ("force_ee", "末端力", "#bc4749"),
                              ("force_base", "底盘力", "#6a994e")]:
        iteration_curve(axes[1, 1], rows, f"auxiliary/{key}_loss", label, color, True)
    axes[1, 1].set_title("辅助分支损失（归一化监督单位，不能直接当N）")
    axes[1, 1].legend()
    iteration_curve(axes[2, 1], rows, "collection_s", "采样", "#277da8")
    iteration_curve(axes[2, 1], rows, "update_s", "更新", "#e76f51")
    axes[2, 1].set_title("每轮耗时（实时时间秒）")
    axes[2, 1].legend()
    decorate(axes, last_iteration)
    figure.suptitle("优化过程全程：淡线为逐轮值，实线为100轮算术均值\n损失与任务效果必须分别判断；阶段A／B数据分布不同")
    figures.append(("完整优化曲线", figure))

    figure, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    groups = [("跟踪与存活奖励", ["tracking_ee_force_world", "tracking_lin_vel", "tracking_ang_vel", "alive"]),
              ("动作变化惩罚", ["action_rate_steering", "action_rate_drive", "action_rate_arm"]),
              ("力矩／速度／加速度／限幅惩罚", ["torques_", "dof_vel_", "dof_acc_", "torque_limits_"]),
              ("关节位置限位与其他物理惩罚", ["dof_pos_limits", "collision", "lin_vel_z", "ang_vel_xy", "base_height"])]
    for axis, (title, prefixes) in zip(axes.flat, groups):
        for prefix in prefixes:
            keys = [key for key in weights if key.startswith(prefix)]
            axis.plot(abscissa, [sum(row["contribution/" + key] for key in keys) for row in table], label=prefix)
        axis.set_title(title + "（奖励／秒）")
        axis.legend(fontsize=8)
    decorate(axes, last_iteration)
    figure.suptitle("实际奖励分解：原始项 × 已配置权重，按episode步数加权\ntracking_ee_force_world是力补偿位置目标跟踪，不等同于接触力成功")
    figures.append(("完整奖励分解", figure))
    with PdfPages(ROOT / "完整曲线.pdf") as pdf:
        for name, figure in figures:
            figure.savefig(ROOT / f"{name}.png", dpi=160)
            pdf.savefig(figure)
            plt.close(figure)
    verification = {"snapshot_sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
        "first_iteration": 1, "last_iteration": last_iteration, "iterations": len(rows),
        "unique_contiguous": True, "counter_formula_matches": True, "all_scalars_finite": True,
        "same_stage_source_and_asset_match": True, "all_runs_same_asset": True,
        "historical_failure_records": {run["name"]: run["failure_json_exists"] for run in runs},
        "reward_weights_unchanged": True, "max_reward_reconstruction_rate_error": max(reward_reconstruction_errors),
        "completed_episodes": sum(run["episode_selected_count"] for run in runs),
        "window_701_800": table[7], "snapshot_finished_at": snapshot["finished_at"]}
    (ROOT / "curve_verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in verification.items() if key != "window_701_800"}, ensure_ascii=False))


if __name__ == "__main__":
    main()

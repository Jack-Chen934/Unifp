"""分析三个冻结检查点的TD、GAE与五秒后续回报，不更新网络。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent


def statistics(values: np.ndarray) -> dict[str, float | int]:
    """Summarize finite diagnostic scalar samples without population inference.

    Args:
        values: Any-shaped finite sample array; flattened before reduction.

    Returns:
        Count, mean, standard deviation, absolute mean and selected quantiles.
    """
    values = values.astype(np.float64).ravel()
    if not len(values):
        return {"count": 0}
    return {"count": len(values), "mean": float(values.mean()), "std": float(values.std()),
            "mae": float(np.abs(values).mean()),
            **{f"p{int(quantile * 100)}": float(np.quantile(values, quantile))
               for quantile in (0.05, 0.5, 0.95)}}


def compute_gae(data: Any, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct training semantics in independent 24-step rollout windows.

    Args:
        data: NPZ mapping with scalar [T,N] values, rewards and end masks.
        gamma: Discount per 20 ms policy step.
        lam: GAE decay per policy step.

    Returns:
        Raw and rollout-normalized advantages [floor(T/24)*24,N].

    Note:
        Pure timeouts bootstrap once; true terminations do not bootstrap.
    """
    count = len(data["reward"]) // 24 * 24
    advantage = np.zeros_like(data["reward"][:count], dtype=np.float64)
    normalized = np.zeros_like(advantage)
    done = data["terminated"] | data["timeout"]
    compensated = data["reward"] + gamma * data["next_value"] * (data["timeout"] & ~data["terminated"])
    for start in range(0, count, 24):
        running = np.zeros(data["reward"].shape[1])
        for step in reversed(range(start, start + 24)):
            continuation = ~done[step]
            delta = compensated[step] + gamma * continuation * data["next_value"][step] - data["value"][step]
            running = delta + gamma * lam * continuation * running
            advantage[step] = running
        block = advantage[start:start + 24]
        normalized[start:start + 24] = (block - block.mean()) / (block.std(ddof=1) + 1e-8)
    return advantage, normalized


def future_returns(data: Any, gamma: float) -> dict[str, np.ndarray]:
    """Observe five-second returns with explicit failure and timeout boundaries.

    Args:
        data: Frozen rollout arrays with scalar [2000,64] value and reward.
        gamma: Discount per policy step.

    Returns:
        Aligned vectors for initial value, observed rewards, bootstrap tail,
        end kind (0 horizon, 1 failure, 2 timeout), and environment index.

    Note:
        Starts every 25 steps; overlapping samples are correlated observations.
        The bootstrap component is reported separately from observed rewards.
    """
    output: dict[str, list[np.ndarray]] = {name: [] for name in ("value", "observed", "bootstrap", "end_kind", "env")}
    for start in range(32, len(data["reward"]) - 250 + 1, 25):
        observed = np.zeros(64)
        bootstrap = np.zeros(64)
        active = np.ones(64, dtype=bool)
        kind = np.zeros(64, dtype=int)
        for offset in range(250):
            step = start + offset
            observed += active * gamma ** offset * data["reward"][step]
            failure = active & data["terminated"][step]
            timeout = active & data["timeout"][step] & ~failure
            bootstrap[timeout] = gamma ** (offset + 1) * data["next_value"][step, timeout]
            kind[failure], kind[timeout] = 1, 2
            active &= ~(failure | timeout)
        bootstrap[active] = gamma ** 250 * data["next_value"][start + 249, active]
        for name, values in (("value", data["value"][start]), ("observed", observed),
                             ("bootstrap", bootstrap), ("end_kind", kind), ("env", np.arange(64))):
            output[name].append(values)
    return {name: np.concatenate(values) for name, values in output.items()}


def main() -> None:
    """Write calibrated-return diagnostics, strata and a checked overview plot.

    Raises:
        AssertionError: Rewards, trajectory pairing, finiteness or source freezes fail.

    Note:
        Reads three independent frozen rollouts; no optimizer exists in this path.
    """
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font_manager.fontManager.addfont(font_path)
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=font_path).get_name(),
                         "axes.unicode_minus": False, "font.size": 10})
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    results = {}
    for column, iteration in enumerate((9000, 12000, 16000)):
        directory = ROOT / f"check_{iteration}"
        with np.load(directory / "trajectory.npz") as archive:
            data = {key: archive[key] for key in archive.files}
        summary = json.loads((directory / "summary.json").read_text())
        assert summary["parameters_unchanged"] and summary["policy_updates"] == 0
        assert all(np.isfinite(values).all() for values in data.values())
        gamma, lam = summary["gamma"], summary["lam"]
        weights = np.array([summary["reward_weights"][key] for key in summary["reward_names"]])
        reconstructed_reward = np.sum(data["raw_rewards"] * weights, axis=-1) * summary["policy_dt_s"]
        reward_error = float(np.max(np.abs(reconstructed_reward - data["reward"])))
        assert reward_error < 1e-5
        done = data["terminated"] | data["timeout"]
        pairing_error = float(np.max(np.abs(data["next_value"][:-1][~done[:-1]] - data["value"][1:][~done[:-1]])))
        assert pairing_error < 1e-5
        delta = data["reward"] + gamma * ~data["terminated"] * data["next_value"] - data["value"]
        advantage, normalized = compute_gae(data, gamma, lam)

        # Mark only the same episode's final second before an observed failure.
        preceding_failure = np.zeros_like(done)
        for step, env_id in np.argwhere(data["terminated"]):
            first = max(0, step - 49, step - int(data["episode_step"][step, env_id]))
            preceding_failure[first:step + 1, env_id] = True
        external_force_norm = np.linalg.norm(data["external_force"], axis=-1)
        strata = {"all": np.ones_like(done), "before_failure_1s": preceding_failure,
                  "physical_failure_step": data["terminated"],
                  "zero_external_force_le1N": external_force_norm <= 1,
                  "nonzero_external_force_gt1N": external_force_norm > 1,
                  "any_joint_saturated": data["saturation"].max(axis=-1) > 0}
        stratum_metrics = {}
        for name, mask in strata.items():
            usable = mask[:len(advantage)]
            stratum_metrics[name] = {"td": statistics(delta[mask]),
                "raw_gae": statistics(advantage[usable]),
                "normalized_gae": statistics(normalized[usable]),
                "normalized_gae_negative_fraction": float((normalized[usable] < 0).mean()) if usable.any() else None,
                "tcp_compensated_cm": statistics(100 * data["errors"][..., 5][mask])}
        future = future_returns(data, gamma)
        target = future["observed"] + future["bootstrap"]
        residual = future["value"] - target
        continuing = future["end_kind"] == 0
        return_metrics = {"value": statistics(future["value"]), "observed": statistics(future["observed"]),
            "bootstrap": statistics(future["bootstrap"]), "value_minus_return": statistics(residual),
            "rmse": float(np.sqrt(np.mean(residual ** 2))),
            "correlation": float(np.corrcoef(future["value"], target)[0, 1]),
            "bootstrap_absolute_share": float(np.abs(future["bootstrap"]).sum() / (np.abs(future["observed"]).sum() + np.abs(future["bootstrap"]).sum())),
            "end_counts": {str(kind): int((future["end_kind"] == kind).sum()) for kind in (0, 1, 2)},
            "continuing_value_minus_return": statistics(residual[continuing]),
            "continuing_bootstrap_absolute_share": float(np.abs(future["bootstrap"][continuing]).sum() / (np.abs(future["observed"][continuing]).sum() + np.abs(future["bootstrap"][continuing]).sum()))}
        bins = np.unique(np.quantile(future["value"], np.linspace(0, 1, 11)))
        bin_rows = []
        for index in range(len(bins) - 1):
            mask = (future["value"] >= bins[index]) & (future["value"] <= bins[index + 1] if index == len(bins) - 2 else future["value"] < bins[index + 1])
            bin_rows.append({"value": float(future["value"][mask].mean()), "return": float(target[mask].mean()), "count": int(mask.sum())})
        return_metrics["calibration_bins"] = bin_rows
        results[str(iteration)] = {"summary": summary, "reward_reconstruction_max_error": reward_error,
            "successor_pairing_max_error": pairing_error, "strata": stratum_metrics,
            "five_second_returns": return_metrics,
            "saturation_by_joint": data["saturation"].mean(axis=(0, 1)).tolist(),
            "action_change_squared_by_joint": np.mean(np.diff(data["action"], axis=0) ** 2, axis=(0, 1)).tolist(),
            "sampled_tcp_compensated_cm": float(100 * data["errors"][..., 5].mean()),
            "fresh_gae_residual_mse": float(np.mean(advantage ** 2))}
        np.savez_compressed(directory / "derived_advantages.npz", raw_gae=advantage, normalized_gae=normalized)

        axis = axes[0, column]
        axis.hexbin(future["value"], target, gridsize=35, mincnt=1, cmap="Blues", bins="log")
        limits = [min(future["value"].min(), target.min()), max(future["value"].max(), target.max())]
        axis.plot(limits, limits, "--", color="#bc4749", label="估值＝后续回报")
        axis.plot([row["value"] for row in bin_rows], [row["return"] for row in bin_rows], "o-", color="#f4a261", label="估值分位箱均值")
        axis.set_xlabel("行动前Critic V")
        axis.set_ylabel("随后5秒折扣奖励＋边界bootstrap")
        axis.set_title(f"{iteration}轮：偏差 {residual.mean():+.2f}；RMSE {return_metrics['rmse']:.2f}")
        axis.legend(fontsize=8)
        axis = axes[1, column]
        labels = ["所有转移", "失败前1秒", "失败当步"]
        selected_strata = [stratum_metrics[key] for key in ("all", "before_failure_1s", "physical_failure_step")]
        axis.bar(np.arange(3) - 0.18, [row["td"]["mean"] for row in selected_strata], 0.36, label="平均TD残差")
        axis.bar(np.arange(3) + 0.18, [row["raw_gae"]["mean"] for row in selected_strata], 0.36, label="平均原始GAE")
        axis.axhline(0, color="grey", linewidth=0.8)
        axis.set_xticks(np.arange(3), labels)
        axis.set_title(f"失败 {summary['physical_failures']}；超时 {summary['timeouts']}（64环境×40秒）")
        axis.legend(fontsize=8)
    figure.suptitle("Critic只读检查：冻结高斯策略，零更新；三检查点相同评估seed\n后续回报仍含边界自举，单条随机轨迹不是期望价值真值；重叠窗口不独立")
    figure.savefig(ROOT / "Critic只读检查.png", dpi=160)
    figure.savefig(ROOT / "Critic只读检查.pdf")
    plt.close(figure)
    (ROOT / "critic_metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    for iteration, result in results.items():
        print(iteration, json.dumps({"td": result["strata"]["all"]["td"],
            "future_bias": result["five_second_returns"]["value_minus_return"],
            "before_failure": result["strata"]["before_failure_1s"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

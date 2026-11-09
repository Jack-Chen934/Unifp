"""只读比较三个B检查点的高斯尺度、优化状态与同输入动作。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import torch


def main() -> None:
    """Inspect CPU checkpoint tensors and optionally evaluate shared observations.

    Raises:
        OSError: A checkpoint, source module, or output cannot be accessed.
        ValueError: A shared observation tensor has the wrong feature width.

    Note:
        Creates no simulator, learner, optimizer, or CUDA context. All checkpoint
        tensors are loaded on CPU; the optional inputs are genuine probe histories.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observations", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    module_path = args.source / "legged_gym/b2_gym_learn/ppo_cse_pf/v3_actor_critic.py"
    module_spec = importlib.util.spec_from_file_location("checkpoint_policy", module_path)
    policy_module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(policy_module)
    shared_observations = None
    if args.observations:
        shared_observations = torch.load(args.observations, map_location="cpu", weights_only=False)
        if shared_observations.ndim != 2 or shared_observations.shape[1] != 1856:
            raise ValueError("Expected real actor histories [samples,1856]")
    checkpoints: list[dict[str, Any]] = []
    shared_means = {}
    for iteration in (9000, 10000, 12000):
        checkpoint_path = args.checkpoints / f"model_{iteration:06d}.pt"
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = payload["model_state_dict"]
        std = state["std"]
        entry = {
            "iteration": iteration, "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            "std_by_action": std.tolist(),
            "std_group_mean": [float(std[start:end].mean()) for start, end in ((0, 4), (4, 8), (8, 14))],
            "independent_fixed_mean_noise_action_difference_square_sum": [float(2 * std[start:end].square().sum()) for start, end in ((0, 4), (4, 8), (8, 14))],
            "gaussian_entropy_sum": float(torch.log(std * (2 * torch.pi * torch.e) ** .5).sum()),
            "learning_rate_state": payload["learning_rate_state"],
            "optimizer_actual_lr": [group["lr"] for group in payload["main_optimizer_state_dict"]["param_groups"]],
            "entropy_coef": payload["config_snapshot"]["training"]["algorithm"]["entropy_coef"],
            "parameters_finite": all(bool(torch.isfinite(value).all()) for value in state.values()),
        }
        if shared_observations is not None:
            model = policy_module.V3ActorCritic(1856, 294, 12, 58, 14, **payload["config_snapshot"]["training"]["policy"])
            model.load_state_dict(state, strict=True)
            model.eval()
            with torch.inference_mode():
                means = model.act_inference({"obs": shared_observations})
            shared_means[iteration] = means
            entry["common_observation_action_mean_abs"] = means.abs().mean(dim=0).tolist()
            entry["common_observation_action_mean_p95_abs"] = torch.quantile(means.abs(), .95, dim=0).tolist()
            entry["common_observation_action_mean_clip_fraction"] = float((means.abs() >= 100).float().mean())
        checkpoints.append(entry)
    output = {"checkpoints": checkpoints,
              "std_ratio_12000_over_9000": [last / first for last, first in zip(checkpoints[-1]["std_by_action"], checkpoints[0]["std_by_action"])],
              "说明": "std为策略无量纲动作尺度。2Σstd²仅为固定均值独立噪声参考，不是闭环动作变化的严格分解；不把参数尺度增长直接等同于唯一根因。"}
    if shared_observations is not None:
        output["common_observations"] = {"path": str(args.observations), "shape": list(shared_observations.shape)}
        output["common_observation_mean_rms_change_vs_9000"] = {
            str(iteration): (means - shared_means[9000]).square().mean(dim=0).sqrt().tolist()
            for iteration, means in shared_means.items()
        }
    args.output.write_text(json.dumps(output, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    print(json.dumps(output, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

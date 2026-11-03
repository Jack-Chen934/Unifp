"""Verify one update at the actual seed-1 formal configuration; never run A8000."""

import json
from pathlib import Path
import sys

import isaacgym  # noqa: F401
import torch
from legged_gym.scripts.v3_args import get_v3_args
import legged_gym.envs  # noqa: F401
from legged_gym.envs.v3.legged_robot_v3 import V3PosForce
from legged_gym.utils.helpers import class_to_dict, parse_sim_params, set_seed
from legged_gym.utils.v3_config import resolve_v3_configs
from legged_gym.b2_gym_learn.ppo_cse_pf.v3_on_policy_runner import V3OnPolicyRunner
from legged_gym.b2_gym_learn.ppo_cse_pf.v3_run_identity import build_run_identity


def main() -> None:
    """Construct the approved configuration and execute exactly one capacity update.

    Raises:
        AssertionError: Arguments do not match the approved seed-1 baseline.
        OSError: Capacity evidence cannot be written.

    Note:
        Uses the preceding same-snapshot/device smoke run's measured runtime.
        This checkpoint is capacity evidence and must never seed formal A training.
    """
    args = get_v3_args("正式配置容量验证：仅运行一轮，不启动A8000")
    assert args.run_purpose == "formal" and args.seed == 1 and args.num_envs == 4096
    assert args.max_iterations == 8000 and args.sim_device == args.rl_device == "cuda:0"
    source_root = Path.cwd()
    validation_root = source_root.parent
    reference = torch.load(validation_root / "runs/smoke32/checkpoints/model_000001.pt",
                           map_location="cpu", weights_only=False)
    env_cfg, train_cfg = resolve_v3_configs(args, purpose="formal")
    snapshot = {"environment": class_to_dict(env_cfg), "training": class_to_dict(train_cfg)}
    params = parse_sim_params(args, {"sim": snapshot["environment"]["sim"]})
    identity = build_run_identity(source_root, snapshot, reference["identity"]["runtime"])
    for name, value in (("formal_config_snapshot", snapshot), ("formal_runtime", identity["runtime"]),
                        ("formal_identity", identity)):
        with (validation_root / f"{name}.json").open("x", encoding="utf-8") as target:
            json.dump(value, target, ensure_ascii=False, allow_nan=False, indent=2)
    set_seed(1)
    torch.cuda.set_device(torch.device(args.rl_device))
    torch.cuda.reset_peak_memory_stats()
    environment = V3PosForce(env_cfg, params, args.physics_engine, args.sim_device, True,
                             creation_properties=None)
    try:
        runner = V3OnPolicyRunner(environment, snapshot["training"], Path(args.output_dir),
                                 identity, snapshot, args.rl_device)
        runner.learn(1)
        result = {"completed_iterations": runner.completed_iterations,
                  "training_policy_steps": runner.training_policy_steps,
                  "training_transitions": runner.training_transitions,
                  "last_saved_iteration": runner.last_saved_iteration,
                  "torch_peak_memory_bytes": torch.cuda.max_memory_allocated(),
                  "new_episode_after_resume": False, "lineage": runner.lineage,
                  "validation_only": True, "正式训练必须重新从头开始": True}
        (runner.log_dir / "completion.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        environment.gym.destroy_sim(environment.sim)


if __name__ == "__main__":
    main()

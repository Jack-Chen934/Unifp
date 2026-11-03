"""Run the confirmed V3 A learner with explicit identity, output and resume source."""

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import isaacgym  # noqa: F401 -- Native Gym must be loaded before Torch.
from isaacgym import gymapi
import numpy as np
import torch
from legged_gym.scripts.v3_args import get_v3_args
import legged_gym.envs  # noqa: F401 -- Register before importing registry utils.
from legged_gym.envs.v3.legged_robot_v3 import V3PosForce
from legged_gym.utils.helpers import class_to_dict, parse_sim_params, set_seed
from legged_gym.utils.v3_config import resolve_v3_configs
from legged_gym.b2_gym_learn.ppo_cse_pf.v3_on_policy_runner import V3OnPolicyRunner
from legged_gym.b2_gym_learn.ppo_cse_pf.v3_actor_critic import V3ActorCritic
from legged_gym.b2_gym_learn.ppo_cse_pf.v3_ppo import V3PPO
from legged_gym.b2_gym_learn.ppo_cse_pf.v3_run_identity import build_run_identity, file_sha256
from legged_gym.b2_gym_learn.ppo_cse_pf.v3_preflight import assess_a_start_gate, validate_stage_artifacts


def main() -> None:
    """Build a same-device Gym/learner run or verify an already completed A8000.

    Raises:
        ValueError: Task, device, budget or checkpoint identity is incompatible.
        OSError: Run outputs or the explicit checkpoint cannot be accessed.

    Note:
        Creates a fresh output directory. Formal runs use remaining cumulative
        A budget; development/preflight max_iterations is an additional request.
        Checks formal A11 evidence and the frozen evaluation manifest before Gym.
        Never selects a latest file, changes existing environments or starts B.
    """
    args = get_v3_args("V3 part A training and complete-boundary resume")
    env_cfg, train_cfg = resolve_v3_configs(args, purpose=args.run_purpose)
    if args.reproduction_part != "A" or args.run_purpose == "evaluation":
        raise ValueError("This training entry supports part A only; evaluation is independent")
    if torch.device(args.sim_device) != torch.device(args.rl_device):
        raise ValueError("V3 simulation tensors and learning must share one explicit device")
    source_root = Path(__file__).resolve().parents[2]
    snapshot = {"environment": class_to_dict(env_cfg), "training": class_to_dict(train_cfg)}
    params = parse_sim_params(args, {"sim": snapshot["environment"]["sim"]})
    device = torch.device(args.rl_device)
    if (device.type == "cuda") != params.use_gpu_pipeline:
        raise ValueError("Training requires simulation tensor pipeline and learner on the same device")
    runtime = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
               "cuda_runtime": torch.version.cuda, "isaac_gym": "1.0.preview4",
               "device": str(device), "sim_device": args.sim_device,
               "physics_engine": str(args.physics_engine), "use_gpu_pipeline": params.use_gpu_pipeline,
               "physx_use_gpu": params.physx.use_gpu,
               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
               "gpu_driver": subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).splitlines()[0] if device.type == "cuda" else None,
               "torch_num_threads": torch.get_num_threads(), "platform": platform.platform(),
               "python_executable": sys.executable, "gym_import_path": isaacgym.__file__}
    if args.physics_engine != gymapi.SIM_PHYSX:
        raise ValueError("Confirmed V3 runs use PhysX")
    identity = build_run_identity(source_root, snapshot, runtime)
    resume_payload = None
    if args.resume:
        resume_payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        if resume_payload.get("identity") != identity:
            raise ValueError("Resume learning/source/asset/schema/runtime identity mismatch")
    if train_cfg.runner.max_iterations == 0:
        # Completion-only verification precedes all Gym construction.
        model = V3ActorCritic(1856, 294, 12, 58, 14, **snapshot["training"]["policy"])
        validator = V3PPO(model, device=str(device), **snapshot["training"]["algorithm"])
        V3OnPolicyRunner.validate_checkpoint(resume_payload, validator, identity, env_cfg.env.num_envs)
        if (resume_payload is None or resume_payload.get("format_version") != 1
                or resume_payload.get("checkpoint_kind") != "training_boundary"
                or resume_payload.get("progress") != {"completed_iterations": 8000, "training_policy_steps": 192000,
                                                       "training_transitions": env_cfg.env.num_envs * 192000, "rollout_step": 0}):
            raise ValueError("Zero-budget verification requires a complete A8000 training checkpoint")
        print(json.dumps({"已完成A8000": True, "未创建Gym或训练采样": True,
                          "checkpoint": args.resume, "sha256": file_sha256(Path(args.resume))}, ensure_ascii=False))
        return
    if args.run_purpose == "formal":
        if not args.start_evidence or not args.manifest:
            raise ValueError("A正式训练需要显式--start_evidence及已确认的--manifest")
        evidence = json.loads(Path(args.start_evidence).read_text(encoding="utf-8"))
        evaluation_manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        artifact_blockers = validate_stage_artifacts(evidence)
        if artifact_blockers:
            raise ValueError("A长训阶段报告未验证：" + "；".join(artifact_blockers))
        gate_report = assess_a_start_gate(evidence, identity, evaluation_manifest,
                                          formal_num_envs=env_cfg.env.num_envs)
        if not gate_report["ready"]:
            raise ValueError("A长训启动门控未通过：" + "；".join(gate_report["blockers"]))
    set_seed(train_cfg.seed)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    creation_properties = None
    if resume_payload is not None:
        creation_properties = {key: resume_payload["environment_training_state"][key]
                               for key in ("mass_com_delta", "friction")}
    environment = V3PosForce(env_cfg, params, args.physics_engine, args.sim_device, args.headless,
                             creation_properties=creation_properties)
    try:
        runner = V3OnPolicyRunner(environment, snapshot["training"], Path(args.output_dir), identity, snapshot, str(device))
        if args.resume:
            runner.load(Path(args.resume))
        requested = 8000 - runner.completed_iterations if args.run_purpose == "formal" else train_cfg.runner.max_iterations
        runner.learn(requested)
        peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        report = {"completed_iterations": runner.completed_iterations,
                  "training_policy_steps": runner.training_policy_steps,
                  "training_transitions": runner.training_transitions,
                  "last_saved_iteration": runner.last_saved_iteration,
                  "torch_peak_memory_bytes": peak_memory,
                  "new_episode_after_resume": bool(args.resume), "lineage": runner.lineage}
        (runner.log_dir / "completion.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(report, ensure_ascii=False), flush=True)
    finally:
        environment.gym.destroy_sim(environment.sim)


if __name__ == "__main__":
    main()

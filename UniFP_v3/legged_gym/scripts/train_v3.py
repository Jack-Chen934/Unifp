"""Run V3 with explicit stage identity and complete-state migration or resume."""

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
from legged_gym.b2_gym_learn.ppo_cse_pf.v3_run_identity import build_run_identity, file_sha256, validate_a_to_b_transition
from legged_gym.b2_gym_learn.ppo_cse_pf.v3_preflight import assess_a_start_gate, assess_b_start_gate, validate_stage_artifacts


def main() -> None:
    """Validate a same-device run and its explicit parent before Gym construction.

    Raises:
        ValueError: Task, device, budget or checkpoint identity is incompatible.
        OSError: Run outputs or the explicit checkpoint cannot be accessed.

    Note:
        Creates a fresh output directory. Formal runs use remaining cumulative
        stage budget; development/preflight max_iterations is an additional request.
        Checks formal A11/B11 evidence and frozen manifests before Gym.
        A-to-B requires an explicit switch and parent digest. B's gate checks the
        original A ancestor even when resuming an independent B checkpoint.
        Never selects a latest file or changes existing environments.
    """
    args = get_v3_args("V3 complete-boundary resume and explicit A-to-B migration")
    env_cfg, train_cfg = resolve_v3_configs(args, purpose=args.run_purpose)
    if args.run_purpose == "evaluation":
        raise ValueError("Evaluation uses the independent evaluation entry")
    if args.transition_from_a and (args.reproduction_part != "B" or not args.resume or not args.parent_sha256):
        raise ValueError("A-to-B requires part B, an explicit --resume and --parent_sha256")
    if args.parent_sha256 and not args.transition_from_a:
        raise ValueError("--parent_sha256 requires explicit --transition_from_a")
    if args.reproduction_part == "B" and not args.resume:
        raise ValueError("Part B requires an explicit complete parent checkpoint")
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
        # Identify the exact approved parent before deserializing or creating actors.
        if args.transition_from_a and file_sha256(Path(args.resume)) != args.parent_sha256:
            raise ValueError("Parent checkpoint SHA256 mismatch")
        resume_payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        if args.transition_from_a:
            validate_a_to_b_transition(resume_payload["identity"], identity, resume_payload["config_snapshot"], snapshot)
            model = V3ActorCritic(1856, 294, 12, 58, 14, **snapshot["training"]["policy"])
            validator = V3PPO(model, device=str(device), **snapshot["training"]["algorithm"])
            V3OnPolicyRunner.validate_checkpoint(resume_payload, validator, resume_payload["identity"], env_cfg.env.num_envs)
            if resume_payload["progress"]["completed_iterations"] != 8000:
                raise ValueError("A-to-B migration requires the completed A8000 boundary")
            del validator, model
        elif resume_payload.get("identity") != identity:
            raise ValueError("Resume learning/source/asset/schema/runtime identity mismatch")
    stage_end = 8000 if args.reproduction_part == "A" else 40000
    if train_cfg.runner.max_iterations == 0:
        if args.transition_from_a:
            raise ValueError("A-to-B migration requires a positive B update budget")
        # Completion-only verification precedes all Gym construction.
        model = V3ActorCritic(1856, 294, 12, 58, 14, **snapshot["training"]["policy"])
        validator = V3PPO(model, device=str(device), **snapshot["training"]["algorithm"])
        V3OnPolicyRunner.validate_checkpoint(resume_payload, validator, identity, env_cfg.env.num_envs)
        if (resume_payload is None or resume_payload.get("format_version") != 1
                or resume_payload.get("checkpoint_kind") != "training_boundary"
                or resume_payload.get("progress") != {"completed_iterations": stage_end, "training_policy_steps": stage_end * 24,
                                                       "training_transitions": env_cfg.env.num_envs * stage_end * 24, "rollout_step": 0}):
            raise ValueError("Zero-budget verification requires a complete stage-end training checkpoint")
        print(json.dumps({"reproduction_part": args.reproduction_part, "completed_iterations": stage_end,
                          "未创建Gym或训练采样": True,
                          "checkpoint": args.resume, "sha256": file_sha256(Path(args.resume))}, ensure_ascii=False))
        return
    if args.run_purpose == "formal":
        if not args.start_evidence or not args.manifest:
            raise ValueError("正式训练需要显式--start_evidence及已确认的--manifest")
        evidence = json.loads(Path(args.start_evidence).read_text(encoding="utf-8"))
        evaluation_manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        if args.reproduction_part == "B":
            ancestor = resume_payload if args.transition_from_a else torch.load(
                resume_payload["lineage"]["parent_checkpoint_path"], map_location="cpu", weights_only=False)
            ancestor_path = Path(args.resume if args.transition_from_a else resume_payload["lineage"]["parent_checkpoint_path"])
            # Same-stage B resumes can have B parents; follow the recorded lineage
            # until the original A checkpoint, keeping the supplied digest explicit.
            while ancestor["identity"]["reproduction_part"] == "B":
                ancestor_path = Path(ancestor["lineage"]["parent_checkpoint_path"])
                ancestor = torch.load(ancestor_path, map_location="cpu", weights_only=False)
            parent_metadata = {"sha256": file_sha256(ancestor_path),
                "reproduction_part": ancestor["identity"]["reproduction_part"],
                "completed_iterations": ancestor["progress"]["completed_iterations"],
                "num_envs": ancestor["environment_training_state"]["num_envs"],
                "seed": ancestor["identity"]["seed"],
                "asset_manifest": ancestor["identity"]["asset_manifest"],
                "schema_sha256": ancestor["identity"]["schema_sha256"]}
            gate_report = assess_b_start_gate(evidence, identity, parent_metadata, evaluation_manifest,
                json.loads((source_root / "configs/v3_b_contact_v1.json").read_text(encoding="utf-8")),
                json.loads((source_root / "configs/v3_a_initial_v1.json").read_text(encoding="utf-8")),
                formal_num_envs=env_cfg.env.num_envs)
        else:
            gate_report = assess_a_start_gate(evidence, identity, evaluation_manifest,
                                              formal_num_envs=env_cfg.env.num_envs)
            gate_report["blockers"].extend(validate_stage_artifacts(evidence))
            gate_report["ready"] = not gate_report["blockers"]
        if not gate_report["ready"]:
            raise ValueError("长训启动门控未通过：" + "；".join(gate_report["blockers"]))
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
            runner.load(Path(args.resume), transition_from_a=args.transition_from_a)
        requested = stage_end - runner.completed_iterations if args.run_purpose == "formal" else train_cfg.runner.max_iterations
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

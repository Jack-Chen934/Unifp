"""Evaluate one exact A8000 checkpoint and seal reproducible evidence."""

import json
import copy
import os
import platform
import tempfile
from pathlib import Path
from argparse import Namespace
from typing import Any, Dict, Iterable, Tuple

import isaacgym  # noqa: F401 -- Native Gym must be imported before Torch.
from isaacgym import gymapi
import torch
from legged_gym.scripts.v3_args import get_v3_args
import legged_gym.envs  # noqa: F401 -- Register before resolving V3 configs.
from legged_gym.b2_gym_learn.ppo_cse_pf.v3_actor_critic import V3ActorCritic
from legged_gym.b2_gym_learn.ppo_cse_pf.v3_evaluation import (
    EVALUATION_SCENARIOS,
    EvaluationVideo,
    configure_evaluation_scenario,
    seal_evaluation_archive,
    summarize_scenario_assessments,
    summarize_evaluation,
    validate_a8000_checkpoint,
    validate_evaluation_manifest,
    validate_policy_evaluation_checkpoint,
    summarize_force_evaluation,
)
from legged_gym.b2_gym_learn.ppo_cse_pf.v3_run_identity import build_run_identity, file_sha256
from legged_gym.envs.v3.legged_robot_v3 import V3PosForce
from legged_gym.envs.v3.v3_force_evaluation import V3ForceEvaluation
from legged_gym.utils.helpers import class_to_dict, parse_sim_params, set_seed
from legged_gym.utils.v3_config import resolve_v3_configs


def main() -> None:
    """Run explicit A/B read-only inference without a learner or optimizer.

    Raises:
        ValueError: Manifest, checkpoint, source identity, or output path is invalid.
        OSError: Evidence cannot be written or the simulator cannot be created.

    Note:
        Isaac Gym execution requires the remote V3 environment. Each scenario
        receives a new simulator and independent archive; no training is launched.
    """
    args = get_v3_args("V3 A8000 policy evaluation and evidence archive")
    if not args.checkpoint or not args.manifest:
        raise ValueError("Evaluation requires explicit --checkpoint and --manifest")
    checkpoint_path = Path(args.checkpoint).resolve()
    manifest_path = Path(args.manifest).resolve()
    archive_path = Path(args.output_dir).resolve()
    if archive_path.exists():
        raise FileExistsError(f"Evaluation archive destination already exists: {archive_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    force_protocol = args.protocol == "force_tracking"
    if args.video and not force_protocol:
        raise ValueError("Zero-force video uses the existing replay_v3.py entry")
    if force_protocol:
        if (manifest.get("protocol") != "force_tracking" or len(manifest["scenarios"]) != 14
                or manifest["events"] != {"start_s": [1., 9.], "rise_s": 2., "hold_s": 1., "fall_s": 2.}
                or manifest["action_mode"] != "actor_mean"):
            raise ValueError("Force evaluation requires the confirmed B10 manifest")
        if not manifest["range_frozen"] and not args.smoke:
            raise ValueError("Force range has not passed B-T07; use --smoke for development only")
    else:
        validate_evaluation_manifest(manifest)
    if args.run_purpose != "evaluation" or (args.protocol == "a8000" and args.reproduction_part != "A"):
        raise ValueError("Evaluation requires its explicit source stage and evaluation purpose")
    if args.seed != manifest["seed"] or (not args.smoke and args.num_envs and args.num_envs != manifest["num_envs"]):
        raise ValueError("CLI seed and environment count must match the frozen manifest")
    if args.episodes_per_env is not None and not args.smoke:
        raise ValueError("Episode quota overrides require explicit --smoke")
    if args.smoke:
        manifest = copy.deepcopy(manifest)
        manifest["num_envs"] = args.num_envs or manifest["num_envs"]
        manifest["episodes_per_env"] = args.episodes_per_env or manifest["episodes_per_env"]
        if manifest["num_envs"] < 1 or manifest["episodes_per_env"] < 1:
            raise ValueError("Smoke quotas must be positive")
    scenarios = tuple(manifest["scenarios"]) if force_protocol else EVALUATION_SCENARIOS
    if args.scenario:
        if args.scenario not in scenarios:
            raise ValueError("Unknown evaluation scenario")
        scenarios = (args.scenario,)

    source_root = Path(__file__).resolve().parents[2]
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_identity = payload.get("identity", {})
    if not checkpoint_identity.get("runtime"):
        raise ValueError("Checkpoint is missing its training runtime identity")
    source_snapshot = copy.deepcopy(payload["config_snapshot"])
    asset_name = Path(source_snapshot["environment"]["asset"]["file"]).name
    asset_relative = next(name for name in checkpoint_identity["asset_manifest"] if name.endswith("/" + asset_name))
    source_snapshot["environment"]["asset"]["file"] = str(source_root / asset_relative)
    active_identity = build_run_identity(source_root, source_snapshot, checkpoint_identity["runtime"])
    if args.reproduction_part != checkpoint_identity["reproduction_part"]:
        raise ValueError("CLI source stage differs from the checkpoint")
    if args.protocol == "a8000":
        validate_a8000_checkpoint(payload, active_identity)
        source_report = {"allowed_source_changes": []}
    else:
        source_report = validate_policy_evaluation_checkpoint(payload, active_identity)
    archive_path.mkdir(parents=True, exist_ok=False)
    reports: Dict[str, Dict[str, Any]] = {}
    checkpoint_sha256 = file_sha256(checkpoint_path)
    for scenario_name in scenarios:
        with tempfile.TemporaryDirectory(prefix=f"v3_a_{scenario_name}_") as temporary_name:
            temporary_path = Path(temporary_name)
            summary, snapshot = evaluate_scenario(
                args, payload, manifest, scenario_name=scenario_name, output_dir=temporary_path,
            )
            summary["checkpoint_sha256"] = checkpoint_sha256
            summary.update(protocol=args.protocol, smoke=args.smoke, source_compatibility=source_report,
                           source_completed_iterations=payload["progress"]["completed_iterations"],
                           source_reproduction_part=checkpoint_identity["reproduction_part"],
                           effective_quota={"num_envs": manifest["num_envs"], "episodes_per_env": manifest["episodes_per_env"]})
            summary["a8000_comparison"] = {"available": False, "原因": "未提供同协议A8000参考归档"}
            if args.reference_archive:
                reference_dir = Path(args.reference_archive) / scenario_name
                reference = json.loads((reference_dir / "summary.json").read_text(encoding="utf-8"))
                if (reference.get("source_completed_iterations") != 8000
                        or reference.get("source_reproduction_part") != "A"
                        or reference.get("protocol") != args.protocol
                        or reference.get("effective_quota") != summary["effective_quota"]
                        or file_sha256(reference_dir / "evaluation_manifest.json") != file_sha256(manifest_path)):
                    raise ValueError("A8000 reference must use the same protocol, manifest and quota")
                current_metrics = summary["tracking"]["nonzero_task"] if force_protocol else summary["metrics"]
                reference_metrics = reference["tracking"]["nonzero_task"] if force_protocol else reference["metrics"]
                summary["a8000_comparison"] = {"available": True,
                    "reference_checkpoint_sha256": reference["checkpoint_sha256"],
                    "metric_mean_deltas": {name: value["mean"] - reference_metrics[name]["mean"]
                        for name, value in current_metrics.items()
                        if name in reference_metrics and value.get("mean") is not None and reference_metrics[name].get("mean") is not None},
                    "physical_failure_rate_delta": (summary["episodes"]["physical_failure_rate"]
                        - reference["episodes"]["physical_failure_rate"])}
            if args.smoke:
                summary["assessment"] = "smoke_only"
            seal_evaluation_archive(
                archive_path / scenario_name, checkpoint_path, manifest_path,
                temporary_path / "policy_trace.jsonl", temporary_path / "episodes.jsonl",
                summary, active_identity, checkpoint_path.parent.parent / "iterations.jsonl", snapshot,
            )
            reports[scenario_name] = summary
    overall = ({"assessment": "pass" if len(reports) == 14 and all(report["assessment"] == "pass" for report in reports.values())
                else "smoke_only" if args.smoke else "ungraded_or_incomplete",
                "scenarios": {name: report["assessment"] for name, report in reports.items()}}
               if force_protocol else summarize_scenario_assessments(reports))
    overall.update({"checkpoint_sha256": checkpoint_sha256,
                    "evaluation_manifest_sha256": file_sha256(manifest_path),
                    "threshold_version": manifest["threshold_version"],
                    "scenario_archive_manifest_sha256": {
                        name: file_sha256(archive_path / name / "archive_manifest.json")
                        for name in scenarios},
                    "policy_updates": 0})
    summary_path = archive_path / "summary.json"
    summary_path.write_text(json.dumps(overall, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                            encoding="utf-8")
    summary_path.chmod(0o444)
    archive_path.chmod(0o555)
    print(json.dumps(overall, ensure_ascii=False, allow_nan=False), flush=True)


def evaluate_scenario(
    args: Namespace, payload: Dict[str, Any], manifest: Dict[str, Any], *,
    scenario_name: str, output_dir: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Collect one scenario using an isolated simulator and Actor mean actions.

    Args:
        args: Parsed V3 evaluation arguments with matching task seed and devices.
        payload: Real training checkpoint supplying policy weights and progress.
            The formal caller must first validate the exact A8000 identity.
        manifest: Scenario settings, episode quotas, seeds, and per-group thresholds.
        scenario_name: S0, S1, or S2 protocol to configure before actor creation.
        output_dir: Existing writable directory receiving raw JSONL evidence.

    Returns:
        Scenario metrics and the effective environment/training configuration.

    Raises:
        ValueError: Devices, physics engine, or collected metrics are incompatible.
        OSError: Evidence writing or simulator creation fails.

    Note:
        Creates and destroys one simulator, resets seeds, and writes trace files.
        No optimizer is created and checkpoint training counters are never advanced.
    """
    args.num_envs = manifest["num_envs"]
    # Resolve fresh configs before each simulator so scenario overrides cannot leak.
    env_cfg, train_cfg = resolve_v3_configs(args, purpose="evaluation")
    force_protocol = args.protocol == "force_tracking"
    if force_protocol:
        scenario = manifest["scenarios"][scenario_name]
        env_cfg.run.reproduction_part = "B"
        env_cfg.force_events.max_force_n = manifest["max_force_n"]
        env_cfg.commands.zero_vel_cmd_prob = 0.
        configure_evaluation_scenario(env_cfg, "S2" if scenario["push"] else "S1" if scenario["moving"] else "S0")
    else:
        # A's exact zero-force RNG and scene protocol also apply to B weights.
        env_cfg.run.reproduction_part = "A"
        env_cfg.commands.zero_vel_cmd_prob = 0.
        configure_evaluation_scenario(env_cfg, scenario_name)
    snapshot = {"environment": class_to_dict(env_cfg), "training": class_to_dict(train_cfg)}
    params = parse_sim_params(args, {"sim": snapshot["environment"]["sim"]})
    if args.physics_engine != gymapi.SIM_PHYSX:
        raise ValueError("A evaluation requires the confirmed PhysX simulator")
    device = torch.device(args.rl_device)
    if (device.type == "cuda") != params.use_gpu_pipeline:
        raise ValueError("Evaluation simulator and policy must use the same tensor device")
    set_seed(manifest["seed"])
    if device.type == "cuda":
        torch.cuda.set_device(device)
    environment = (V3ForceEvaluation(env_cfg, params, args.physics_engine, args.sim_device, True,
                                    scenario=scenario, episodes_per_env=manifest["episodes_per_env"], video=args.video)
                   if force_protocol else V3PosForce(env_cfg, params, args.physics_engine, args.sim_device, True))
    try:
        if force_protocol and args.video:
            environment.video_recorder = EvaluationVideo(environment, output_dir / "representative_env0_episode0.mp4")
        environment.set_evaluation_push_seed(manifest["push_seed"])
        environment.policy_trace_enabled = True
        model = V3ActorCritic(
            environment.num_obs, environment.num_privileged_obs,
            environment.cfg.env.num_pred_obs, environment.cfg.env.num_single_obs,
            environment.num_actions, **payload["config_snapshot"]["training"]["policy"],
        ).to(device)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.eval()
        observations = environment.get_observations()
        episode_counts = [0] * environment.num_envs
        episode_rows = []
        trace_path = output_dir / "policy_trace.jsonl"
        episodes_path = output_dir / "episodes.jsonl"
        trace_count = 0
        previous_force_phase = ["zero"] * environment.num_envs
        previous_force_hold_tail = [False] * environment.num_envs
        with trace_path.open("x", encoding="utf-8") as trace_file, episodes_path.open(
            "x", encoding="utf-8"
        ) as episode_file:
            max_policy_steps = manifest["episodes_per_env"] * environment.max_episode_length
            for evaluation_step in range(max_policy_steps):
                environment.set_evaluation_policy_step(payload["progress"]["training_policy_steps"] + evaluation_step)
                # Pair the estimator with its input labels before reset can replace them.
                labels_before_action = observations["obs_pred"].detach().cpu().tolist()
                yaw_before_action = environment.rpy[:, 2].detach().cpu().tolist()
                policy_info: Dict[str, torch.Tensor] = {}
                with torch.inference_mode():
                    actions = model.act_inference(observations, policy_info)
                observations, _, _, infos = environment.step(actions)
                estimates = policy_info["state_estimates"].detach().cpu().tolist()
                # Retain failures and stop collecting each environment at its own quota.
                for row in infos["policy_trace"]:
                    env_id = row["env_id"]
                    if episode_counts[env_id] >= manifest["episodes_per_env"]:
                        continue
                    row.update({
                        "completed_iterations": payload["progress"]["completed_iterations"],
                        "training_transitions": payload["progress"]["training_transitions"],
                        "evaluation_seed": manifest["seed"],
                        "scenario_name": scenario_name,
                        "push_seed": manifest["push_seed"],
                        "evaluation_step": evaluation_step,
                        "episode_index": episode_counts[env_id],
                        "scenario_id": f"{scenario_name}_seed{manifest['seed']}_env{env_id}_episode{episode_counts[env_id]}",
                        "state_estimates_scaled": estimates[env_id],
                        "state_estimate_labels_scaled": labels_before_action[env_id],
                        "prediction_yaw_rad": yaw_before_action[env_id],
                    })
                    if force_protocol:
                        row["estimator_force_phase"] = previous_force_phase[env_id]
                        row["estimator_force_hold_tail"] = previous_force_hold_tail[env_id]
                        previous_force_phase[env_id] = row["force_phase"]
                        previous_force_hold_tail[env_id] = row["force_hold_tail"]
                    trace_file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    trace_count += 1
                for record in infos["episode_records"]:
                    env_id = record["env_id"]
                    if episode_counts[env_id] >= manifest["episodes_per_env"]:
                        continue
                    record.update({"evaluation_seed": manifest["seed"],
                                   "scenario_name": scenario_name,
                                   "completed_iterations": payload["progress"]["completed_iterations"],
                                   "training_policy_steps": payload["progress"]["training_policy_steps"],
                                   "training_transitions": payload["progress"]["training_transitions"],
                                   "episode_index": episode_counts[env_id]})
                    episode_file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    episode_rows.append(record)
                    episode_counts[env_id] += 1
                    previous_force_phase[env_id] = "zero"
                    previous_force_hold_tail[env_id] = False
                if all(count >= manifest["episodes_per_env"] for count in episode_counts):
                    break
            trace_file.flush()
            os.fsync(trace_file.fileno())
            episode_file.flush()
            os.fsync(episode_file.fileno())
        if trace_count == 0:
            raise ValueError("Evaluation produced no policy-boundary trace")
        summary = (summarize_force_evaluation(_read_trace_rows(trace_path), episode_rows, manifest, scenario=scenario)
                   if force_protocol else summarize_evaluation(_read_trace_rows(trace_path), episode_rows, manifest, scenario_name=scenario_name))
        if not force_protocol:
            summary["checkpoint_iteration"] = payload["progress"]["completed_iterations"]
        summary.update({"evaluation_device": str(device),
                        "evaluation_environment_count": environment.num_envs,
                        "evaluation_seed": manifest["seed"], "push_seed": manifest["push_seed"],
                        "policy_updates": 0})
        summary["evaluation_runtime"] = {"python": platform.python_version(), "torch": torch.__version__,
                                          "device": str(device), "gym_import_path": isaacgym.__file__,
                                          "sim_device": args.sim_device, "physx_use_gpu": params.physx.use_gpu,
                                          "use_gpu_pipeline": params.use_gpu_pipeline}
        return summary, snapshot
    finally:
        try:
            if force_protocol and environment.video_recorder is not None:
                environment.video_recorder.close()
        finally:
            environment.gym.destroy_sim(environment.sim)


def _read_trace_rows(path: Path) -> Iterable[Dict[str, Any]]:
    """Yield policy-boundary trace rows without retaining the full run in memory.

    Args:
        path: Temporary UTF-8 JSONL trace generated by this evaluation.

    Yields:
        Parsed trace row dictionaries in file order.

    Raises:
        OSError: The trace file cannot be opened or read.
        json.JSONDecodeError: A trace line contains invalid JSON.
    """
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


if __name__ == "__main__":
    main()

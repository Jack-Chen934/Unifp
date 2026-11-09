"""Read-only fixed-wall contact evaluation using the unchanged V3 policy."""

import copy
import json
import os
import platform
import tempfile
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Tuple

import isaacgym  # noqa: F401 -- Native Gym must precede Torch.
from isaacgym import gymapi
import torch
import legged_gym.envs  # noqa: F401 -- Register before config resolution.
from legged_gym.b2_gym_learn.ppo_cse_pf.v3_actor_critic import V3ActorCritic
from legged_gym.b2_gym_learn.ppo_cse_pf.v3_evaluation import (
    EvaluationVideo, seal_evaluation_archive, summarize_contact_evaluation,
    validate_policy_evaluation_checkpoint,
)
from legged_gym.b2_gym_learn.ppo_cse_pf.v3_run_identity import build_run_identity, file_sha256
from legged_gym.envs.v3.v3_contact_evaluation import V3ContactEvaluation
from legged_gym.scripts.evaluate_v3 import _read_trace_rows
from legged_gym.scripts.v3_args import get_v3_args
from legged_gym.utils.helpers import class_to_dict, parse_sim_params, set_seed
from legged_gym.utils.v3_config import resolve_v3_configs


def evaluate_contact_scenario(args: Namespace, payload: Dict[str, Any], manifest: Dict[str, Any], *,
                              zero_control: bool, output_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Collect scripted contact episodes without a learner or optimizer.

    Args:
        args: Explicit source-stage, evaluation seed and simulator/tensor devices.
        payload: Complete A8000 or B checkpoint, already provenance-validated.
        manifest: Contact scene geometry, SI force range, quotas and thresholds.
        zero_control: Use the same world trajectory with a zero force command.
        output_dir: Existing new writable directory for traces and optional videos.

    Returns:
        Contact statistics/runtime/readback metadata and effective config snapshot.

    Raises:
        ValueError: Tensor device/pipeline or physics engine is incompatible.
        OSError: Trace/video evidence cannot be written.

    Note:
        Creates/destroys a simulator. Only load_state_dict is used; training RNG,
        optimizer, storage, counters and saved creation randomization are not restored.
    """
    args.num_envs = manifest["num_envs"]
    cfg, train_cfg = resolve_v3_configs(args, purpose="evaluation")
    cfg.run.reproduction_part = "B"
    for name in ("randomize_friction", "randomize_base_mass", "randomize_base_com",
                 "randomize_tool_mass", "randomize_execution_scale", "push_robots"):
        setattr(cfg.domain_rand, name, False)
    cfg.commands.ranges.lin_vel_x = [0., 0.]
    cfg.commands.ranges.lin_vel_y = [0., 0.]
    cfg.commands.ranges.ang_vel_yaw = [0., 0.]
    cfg.commands.zero_vel_cmd_prob = 0.
    snapshot = {"environment": class_to_dict(cfg), "training": class_to_dict(train_cfg)}
    params = parse_sim_params(args, {"sim": snapshot["environment"]["sim"]})
    device = torch.device(args.rl_device)
    if args.physics_engine != gymapi.SIM_PHYSX or (device.type == "cuda") != params.use_gpu_pipeline:
        raise ValueError("Contact evaluation requires PhysX and matching tensor pipeline/device")
    set_seed(manifest["seed"])
    if device.type == "cuda":
        torch.cuda.set_device(device)
    environment = V3ContactEvaluation(cfg, params, args.physics_engine, args.sim_device, True,
        manifest=manifest, zero_control=zero_control, video=args.video)
    try:
        quota = manifest["zero_control_episodes_per_env"] if zero_control else manifest["episodes_per_env"]
        if args.video:
            for episode_index in range(min(quota, 1 if zero_control else 2)):
                environment.video_recorders[episode_index] = EvaluationVideo(
                    environment, output_dir / f"representative_env0_episode{episode_index}.mp4")
        environment.policy_trace_enabled = True
        model = V3ActorCritic(environment.num_obs, environment.num_privileged_obs,
            environment.cfg.env.num_pred_obs, environment.cfg.env.num_single_obs,
            environment.num_actions, **payload["config_snapshot"]["training"]["policy"]).to(device)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.eval()
        observations = environment.get_observations()
        episode_counts = [0] * environment.num_envs
        episode_rows = []
        trace_path, episodes_path = output_dir / "policy_trace.jsonl", output_dir / "episodes.jsonl"
        with trace_path.open("x", encoding="utf-8") as trace_file, episodes_path.open("x", encoding="utf-8") as episode_file:
            for evaluation_step in range(quota * environment.max_episode_length):
                # Copy the input boundary before step can reset or replace measurement buffers.
                previous_reaction = environment.last_contact_force_world.clone().cpu().tolist()
                previous_valid = environment.measurement_valid.clone().cpu().tolist()
                previous_labels = observations["obs_pred"].clone().cpu().tolist()
                previous_yaw = environment.rpy[:, 2].clone()
                command = environment.commands[:, 9:12].clone()
                command_normal = previous_yaw.cos() * command[:, 0] - previous_yaw.sin() * command[:, 1]
                policy_info: Dict[str, torch.Tensor] = {}
                with torch.inference_mode():
                    actions = model.act_inference(observations, policy_info)
                predicted_yaw = policy_info["state_estimates"][:, 6:9] / .01
                predicted_world = torch.stack((previous_yaw.cos() * predicted_yaw[:, 0] - previous_yaw.sin() * predicted_yaw[:, 1],
                    previous_yaw.sin() * predicted_yaw[:, 0] + previous_yaw.cos() * predicted_yaw[:, 1], predicted_yaw[:, 2]), dim=1)
                observations, _, _, info = environment.step(actions)
                for row in info["policy_trace"]:
                    env_id = row["env_id"]
                    if episode_counts[env_id] >= quota:
                        continue
                    row.update(episode_index=episode_counts[env_id], evaluation_step=evaluation_step,
                        scenario_id=f"contact_{env_id}_{episode_counts[env_id]}",
                        state_estimates_scaled=policy_info["state_estimates"][env_id].cpu().tolist(),
                        state_estimate_labels_scaled=previous_labels[env_id],
                        predicted_contact_force_world_n=predicted_world[env_id].cpu().tolist(),
                        estimator_contact_truth_world_n=previous_reaction[env_id],
                        estimator_contact_valid=previous_valid[env_id],
                        contact_command_normal_n=float(command_normal[env_id]), prediction_yaw_rad=float(previous_yaw[env_id]))
                    trace_file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                for record in info["episode_records"]:
                    env_id = record["env_id"]
                    if episode_counts[env_id] >= quota:
                        continue
                    record.update(episode_index=episode_counts[env_id])
                    episode_file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    episode_rows.append(record)
                    episode_counts[env_id] += 1
                if all(count >= quota for count in episode_counts):
                    break
            for evidence_file in (trace_file, episode_file):
                evidence_file.flush()
                os.fsync(evidence_file.fileno())
        summary = summarize_contact_evaluation(_read_trace_rows(trace_path), episode_rows, manifest, zero_control=zero_control)
        # Native scene readbacks remain separate from the saved training identity.
        summary["scene_readback"] = {
            "robot_actor_indices": environment.robot_actor_indices.cpu().tolist(),
            "wall_body_indices": environment.wall_body_indices.cpu().tolist(),
            "robot_body_names": environment.gym.get_actor_rigid_body_names(environment.envs[0], environment.actor_handles[0]),
            "wall_root_states": environment.raw_root_states[[environment.gym.get_actor_index(env_handle, wall_handle, gymapi.DOMAIN_SIM)
                for env_handle, wall_handle in zip(environment.envs, environment.wall_handles)]].cpu().tolist(),
            "mass_com_delta": environment.mass_com_delta.cpu().tolist(), "friction": environment.friction.cpu().tolist(),
            "execution_scale_delta": environment.execution_scale_delta.cpu().tolist(),
            "classification": "八点双侧力残差与保守AABB几何排除；待判不算成功，不等于完整原生接触对清单"}
        summary["scene_readback"]["wall_shape_properties"] = [
            [{"friction": shape.friction, "restitution": shape.restitution,
              "contact_offset_m": shape.contact_offset, "rest_offset_m": shape.rest_offset}
             for shape in environment.gym.get_actor_rigid_shape_properties(env_handle, wall_handle)]
            for env_handle, wall_handle in zip(environment.envs, environment.wall_handles)]
        summary["effective_quota"] = {"num_envs": environment.num_envs, "episodes_per_env": quota}
        summary["evaluation_runtime"] = {"python": platform.python_version(), "torch": torch.__version__,
            "device": str(device), "gym_import_path": isaacgym.__file__, "sim_device": args.sim_device,
            "physx_use_gpu": params.physx.use_gpu, "use_gpu_pipeline": params.use_gpu_pipeline,
            "physics_dt_s": params.dt, "decimation": cfg.control.decimation, "policy_dt_s": environment.dt}
        return summary, snapshot
    finally:
        try:
            for recorder in environment.video_recorders.values():
                recorder.close()
        finally:
            environment.gym.destroy_sim(environment.sim)


def main() -> None:
    """Validate a complete source checkpoint and seal contact plus zero controls.

    Raises:
        ValueError: Source identity, geometry, seed, protocol or range is invalid.
        FileExistsError: Output already exists.
        OSError: Evidence cannot be read, recorded or sealed.

    Note:
        Loads training files read-only and writes a new immutable evaluation archive.
        Unfrozen or reduced development runs require --smoke and never receive pass.
    """
    args = get_v3_args("V3 fixed-wall read-only contact evaluation")
    if not args.checkpoint or not args.manifest or args.run_purpose != "evaluation" or args.protocol != "fixed_wall_contact":
        raise ValueError("Explicit checkpoint, manifest, evaluation purpose and fixed_wall_contact protocol are required")
    source_root = Path(__file__).resolve().parents[2]
    manifest_path, checkpoint_path, archive_path = Path(args.manifest).resolve(), Path(args.checkpoint).resolve(), Path(args.output_dir).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest["protocol"] != "fixed_wall_contact" or args.seed != manifest["seed"]
            or manifest["contact_peak_n"] != min(20., manifest["max_force_n"])):
        raise ValueError("Contact manifest differs from the confirmed protocol")
    for relative, expected in manifest["scene"]["source_sha256"].items():
        if file_sha256(source_root / relative) != expected:
            raise ValueError("Contact geometry source hash mismatch: " + relative)
    if not args.smoke and (not manifest["range_frozen"] or manifest["num_envs"] != 32
                          or manifest["episodes_per_env"] != 3 or manifest["zero_control_episodes_per_env"] != 1
                          or args.episodes_per_env is not None
                          or args.num_envs not in (None, manifest["num_envs"])):
        raise ValueError("Formal contact evaluation requires frozen range and full quota")
    if args.smoke:
        manifest = copy.deepcopy(manifest)
        manifest["num_envs"] = args.num_envs or manifest["num_envs"]
        manifest["episodes_per_env"] = args.episodes_per_env or manifest["episodes_per_env"]
    if archive_path.exists():
        raise FileExistsError(archive_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    identity = payload["identity"]
    if args.reproduction_part != identity["reproduction_part"]:
        raise ValueError("Source stage differs from checkpoint")
    source_snapshot = copy.deepcopy(payload["config_snapshot"])
    asset_name = Path(source_snapshot["environment"]["asset"]["file"]).name
    asset_relative = next(name for name in identity["asset_manifest"] if name.endswith("/" + asset_name))
    source_snapshot["environment"]["asset"]["file"] = str(source_root / asset_relative)
    active_identity = build_run_identity(source_root, source_snapshot, identity["runtime"])
    compatibility = validate_policy_evaluation_checkpoint(payload, active_identity)
    archive_path.mkdir(parents=True)
    reports = {}
    for zero_control in (False, True):
        name = "zero_control" if zero_control else "contact"
        with tempfile.TemporaryDirectory(prefix="v3_contact_" + name) as temporary:
            temporary_path = Path(temporary)
            summary, snapshot = evaluate_contact_scenario(args, payload, manifest, zero_control=zero_control, output_dir=temporary_path)
            summary.update(checkpoint_sha256=file_sha256(checkpoint_path), source_compatibility=compatibility,
                source_completed_iterations=payload["progress"]["completed_iterations"],
                source_reproduction_part=identity["reproduction_part"], smoke=args.smoke)
            if args.smoke:
                summary["assessment"] = "smoke_only"
            seal_evaluation_archive(archive_path / name, checkpoint_path, manifest_path,
                temporary_path / "policy_trace.jsonl", temporary_path / "episodes.jsonl", summary,
                active_identity, checkpoint_path.parent.parent / "iterations.jsonl", snapshot)
            reports[name] = summary["assessment"]
    overall = {"scenarios": reports, "policy_updates": 0, "checkpoint_sha256": file_sha256(checkpoint_path),
               "evaluation_manifest_sha256": file_sha256(manifest_path), "smoke": args.smoke}
    (archive_path / "summary.json").write_text(json.dumps(overall, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (archive_path / "summary.json").chmod(0o444)
    archive_path.chmod(0o555)
    print(json.dumps(overall, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

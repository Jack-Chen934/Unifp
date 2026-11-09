"""在原B训练分布中只读诊断动作与终止，结果不作为正式验收。"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
from typing import Any

import isaacgym  # noqa: F401
from isaacgym import gymapi, gymutil
import numpy as np
import torch
import legged_gym
import legged_gym.envs  # noqa: F401
from legged_gym.b2_gym_learn.ppo_cse_pf.v3_actor_critic import V3ActorCritic
from legged_gym.b2_gym_learn.ppo_cse_pf.v3_run_identity import build_run_identity, file_sha256
from legged_gym.envs.v3.legged_robot_v3 import V3PosForce
from legged_gym.utils.helpers import class_to_dict, set_seed
from legged_gym.utils.v3_config import resolve_v3_configs


class DiagnosticEnvironment(V3PosForce):
    """Observe pre-reset failure tensors without modifying termination or physics."""

    def _accumulate_episode_metrics(self, done_ids: torch.Tensor, terminated: torch.Tensor) -> list[dict[str, Any]]:
        """Capture terminal evidence, then delegate the original episode statistics.

        Args:
            done_ids: Environment indices [K] completing an episode this step.
            terminated: Physical-failure mask [N] before local reset.

        Returns:
            Original episode records, including the terminal transition.

        Note:
            Captures all failures as JSON. Body contacts are net forces, not pair
            contacts; an opposite vector only nominates a candidate counterpart.
        """
        self.last_pre_reset_error_mean_m = float(self._physical_tracking_errors()[:, 5].mean())
        for env_id in terminated.nonzero(as_tuple=False).flatten().tolist():
            roll, pitch = self.rpy[env_id, :2].tolist()
            contact_vectors = self.contact_forces[env_id].cpu().tolist()
            base_vector = np.asarray(contact_vectors[self.base_body_index])
            contact_norm = float(np.linalg.norm(base_vector))
            self.failure_events.append({
                "policy_step": self.inspection_step, "env_id": env_id,
                "episode_step": int(self.episode_length_buf[env_id]),
                "predicates": {"roll": abs(roll) > .8, "pitch": abs(pitch) > 1., "base_contact": contact_norm > 1.},
                "roll_rad": roll, "pitch_rad": pitch,
                "base_contact_vector_n": base_vector.tolist(), "base_contact_norm_n": contact_norm,
                "root_state": self.root_states[env_id].cpu().tolist(),
                "environment_origin_world_m": self.env_origins[env_id].cpu().tolist(),
                "dof_position_rad": self.dof_pos[env_id].cpu().tolist(),
                "body_pose_world_m_xyzw": {name: self.rigid_state[env_id, index, :7].cpu().tolist() for index, name in enumerate(self.body_names)},
                "body_contact_vectors_n": dict(zip(self.body_names, contact_vectors)),
                "external_force_world_n": self.applied_ee_force_world[env_id].cpu().tolist(),
                "command_force_yaw_n": self.commands[env_id, 9:12].cpu().tolist(),
                "commands_body": self.commands[env_id, :3].cpu().tolist(),
                "action": self.actions[env_id].cpu().tolist(),
            })
        records = super()._accumulate_episode_metrics(done_ids, terminated)
        self.ended_records.extend(records)
        return records


def main() -> None:
    """Run a bounded B probe and save per-step action/torque/error aggregates.

    Raises:
        ValueError: The frozen source, checkpoint, or diagnostic action mode differs.
        OSError: Input or independent output files cannot be accessed.

    Note:
        Uses new episodes with saved creation properties for the first N actors.
        No optimizer is created. A separate noise generator prevents action sampling
        from directly consuming the environment RNG. Reset divergence still changes
        later environment samples. The original training remains untouched.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("mean", "sample", "sample_std9000"), required=True)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--seconds", type=float, default=20.)
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    source_root = Path(legged_gym.__file__).resolve().parents[1]
    source_snapshot = copy.deepcopy(payload["config_snapshot"])
    asset_basename = Path(source_snapshot["environment"]["asset"]["file"]).name
    relative_asset = next(name for name in payload["identity"]["asset_manifest"] if name.endswith("/" + asset_basename))
    source_snapshot["environment"]["asset"]["file"] = str(source_root / relative_asset)
    assert build_run_identity(source_root, source_snapshot, payload["identity"]["runtime"]) == payload["identity"]
    assert payload["identity"]["reproduction_part"] == "B"
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    set_seed(1001)
    cfg_args = argparse.Namespace(reproduction_part="B", run_purpose="debug", seed=1001,
                                  num_envs=args.num_envs, max_iterations=1)
    cfg, _ = resolve_v3_configs(cfg_args, purpose="debug")
    # Debug identity and actor count differ; every behavior-affecting config is fixed.
    actual_config = class_to_dict(cfg)
    expected_config = copy.deepcopy(source_snapshot["environment"])
    for config in (actual_config, expected_config):
        config.pop("seed")
        config.pop("run")
        config["env"].pop("num_envs")
    assert actual_config == expected_config
    params = gymapi.SimParams()
    gymutil.parse_sim_config(class_to_dict(cfg.sim), params)
    params.use_gpu_pipeline = True
    params.physx.use_gpu = True
    params.physx.num_threads = 4
    properties = {key: payload["environment_training_state"][key][:args.num_envs] for key in ("mass_com_delta", "friction")}
    environment = DiagnosticEnvironment(cfg, params, gymapi.SIM_PHYSX, "cuda:0", True, creation_properties=properties)
    try:
        environment.failure_events, environment.ended_records = [], []
        environment.body_names = environment.gym.get_actor_rigid_body_names(environment.envs[0], environment.actor_handles[0])
        environment.set_evaluation_push_seed(2001)
        progress = payload["progress"]
        environment.set_training_progress(progress["completed_iterations"], progress["training_policy_steps"])
        model = V3ActorCritic(1856, 294, 12, 58, 14, **payload["config_snapshot"]["training"]["policy"]).to("cuda:0")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.eval()
        sampling_std = model.std.detach()
        if args.mode == "sample_std9000":
            earlier = torch.load(args.checkpoint.parent / "model_009000.pt", map_location="cpu", weights_only=False)
            sampling_std = earlier["model_state_dict"]["std"].to("cuda:0")
        noise_generator = torch.Generator(device="cuda:0").manual_seed(3001)
        observations = environment.get_observations()
        step_rows, shared_inputs = [], []
        previous_mean = None
        steps = round(args.seconds / environment.dt)
        ordered_dofs = environment.action_dof_indices
        for step in range(steps):
            environment.inspection_step = step
            environment.set_evaluation_policy_step(progress["training_policy_steps"] + step)
            if step % 100 == 0:
                shared_inputs.append(observations["obs"][:16].detach().cpu().clone())
            with torch.inference_mode():
                model.update_distribution(observations["obs"])
                action_mean = model.action_mean.clone()
                noise = torch.randn(action_mean.shape, device="cuda:0", generator=noise_generator)
                actions = action_mean if args.mode == "mean" else action_mean + noise * sampling_std
                action_difference = (actions.clamp(-100, 100) - environment.last_actions).square()
                mean_difference = torch.zeros_like(action_mean) if previous_mean is None else (action_mean - previous_mean).square()
                observations, _, _, infos = environment.step(actions)
            previous_mean = action_mean
            # Ignore local resets when measuring consecutive mean changes next step.
            previous_mean = previous_mean.clone()
            previous_mean[infos["terminated"] | infos["time_outs"]] = 0.
            torques = environment.torques[:, ordered_dofs]
            velocity = environment.dof_vel[:, ordered_dofs]
            errors = environment._physical_tracking_errors()
            row = {"step": step, "compensated_l2_before_reset_m": environment.last_pre_reset_error_mean_m,
                   "compensated_l2_after_reset_m": float(errors[:, 5].mean()),
                   "action_mean_abs": action_mean.abs().mean(dim=0).cpu().tolist(),
                   "action_abs": actions.abs().mean(dim=0).cpu().tolist(),
                   "action_change_square_mean": action_difference.mean(dim=0).cpu().tolist(),
                   "mean_change_square_mean": mean_difference.mean(dim=0).cpu().tolist(),
                   "torque_limit_fraction": (torques.abs() >= environment.torque_limits[ordered_dofs] * .999).float().mean(dim=0).cpu().tolist(),
                   "joint_velocity_rms_rad_s": velocity.square().mean(dim=0).sqrt().cpu().tolist(),
                   "action_clip_fraction": float((actions.abs() >= 100).float().mean())}
            step_rows.append(row)
            if (step + 1) % 250 == 0:
                print(json.dumps({"step": step + 1, "failures": len(environment.failure_events)}), flush=True)
        ended_counts = Counter(record["end_type"] for record in environment.ended_records)
        predicate_counts = Counter(key for event in environment.failure_events for key, active in event["predicates"].items() if active)
        completed_steps = sum(record["steps"] for record in environment.ended_records)
        summary = {"checkpoint_iteration": progress["completed_iterations"], "checkpoint_sha256": file_sha256(args.checkpoint),
                   "mode": args.mode, "num_envs": args.num_envs, "steps": steps, "seconds": steps * environment.dt,
                   "ended_episodes": dict(ended_counts), "failure_predicates_overlapping": dict(predicate_counts),
                   "failure_fraction_of_ended": ended_counts["physical_failure"] / max(1, len(environment.ended_records)),
                   "completed_episode_step_weighted_tcp_l2_m": sum(record["errors"]["tcp_compensated_l2_m"]["mean"] * record["steps"] for record in environment.ended_records) / max(1, completed_steps),
                   "std_by_action": sampling_std.cpu().tolist(), "dof_names": environment.dof_names,
                   "body_names": environment.body_names, "policy_updates": 0,
                   "说明": "新episode诊断，非训练历史复演，非正式验收；完成episode误差含终止步，step误差为reset后诊断。"}
        for filename, value in (("summary.json", summary), ("failures.json", environment.failure_events), ("episodes.json", environment.ended_records)):
            (args.output / filename).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
        (args.output / "steps.jsonl").write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in step_rows))
        torch.save(torch.cat(shared_inputs), args.output / "shared_observations.pt")
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    finally:
        environment.gym.destroy_sim(environment.sim)


if __name__ == "__main__":
    main()

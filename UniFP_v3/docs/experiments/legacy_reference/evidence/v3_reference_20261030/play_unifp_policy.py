"""Play a saved UniFP student policy in Isaac Lab.

This is an inference/visualization entry point.  It loads only the actor
weights from a checkpoint, feeds the robot-specific actor history to the
released UniFP ActorCritic, and sends deterministic mean actions to the
Isaac Lab vector environment.  Omit --headless to open the Isaac Sim GUI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description="Play a trained UniFP actor in Isaac Lab")
parser.add_argument("--robot", choices=("b2z1", "ranger"), default="b2z1")
parser.add_argument("--ranger_explicit_pd", action="store_true", help="Replay historical Ranger explicit-PD dynamics")
parser.add_argument(
    "--checkpoint",
    type=Path,
    default=ROOT / "logs/ppo-baseline-official-timing-64-300/model_smoke.pt",
    help="Checkpoint containing model_state_dict",
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument(
    "--force_schedule", type=Path,
    help="Evaluation-only JSON tape of world external forces and yaw-frame force commands",
)
parser.add_argument(
    "--output",
    type=Path,
    default=ROOT / "logs/policy-play.json",
    help="Optional JSON summary path",
)
parser.add_argument(
    "--trace",
    type=Path,
    default=None,
    help="Optional per-policy-step trajectory JSON path",
)
parser.add_argument("--domain_randomization", action="store_true")
parser.add_argument("--force_curriculum", action="store_true")
parser.add_argument("--curriculum_start_step", type=int, default=192001)
parser.add_argument("--rough_terrain", action="store_true")
parser.add_argument("--push_robots", action="store_true")
parser.add_argument("--push_interval_steps", type=int, default=400)
parser.add_argument("--max_push_vel_xy", type=float, default=0.8)
parser.add_argument(
    "--observation_noise",
    action="store_true",
    help="Add the training-time actor observation noise",
)
parser.add_argument(
    "--no_command_sampling",
    action="store_true",
    help="Keep reset velocity commands at zero",
)
parser.add_argument(
    "--fixed_goal",
    action="store_true",
    help="Disable randomized end-effector goal trajectories",
)
parser.add_argument(
    "--visualize_goals",
    action="store_true",
    help="Show target and actual end-effector markers in the Isaac Sim GUI",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.num_envs < 1 or args.steps < 1:
    parser.error("--num_envs and --steps must be positive")
if args.curriculum_start_step < 0:
    parser.error("--curriculum_start_step must be nonnegative")
if args.push_interval_steps < 1:
    parser.error("--push_interval_steps must be positive")
if not args.checkpoint.is_file():
    parser.error(f"Checkpoint does not exist: {args.checkpoint}")
force_schedule = None
if args.force_schedule is not None:
    if not args.force_curriculum:
        parser.error("--force_schedule requires --force_curriculum")
    force_schedule = json.loads(args.force_schedule.read_text())
    if (force_schedule.get("policy_dt") != 0.02
            or force_schedule.get("external_frame") != "world"
            or force_schedule.get("command_frame") != "base_yaw"):
        parser.error("Force tape must use 0.02 s steps, world external forces and base_yaw commands")
    if len(force_schedule["samples"]) < args.steps:
        parser.error("Force tape is shorter than --steps")

app = AppLauncher(args).app

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402
from isaaclab.utils.math import quat_apply_yaw, yaw_quat  # noqa: E402

from unifp_vec_env import UniFPVecEnv  # noqa: E402

sys.path.insert(0, str(ROOT / "UniFP"))
from legged_gym.b2_gym_learn.ppo_cse_pf.actor_critic import ActorCritic  # noqa: E402


def main():
    if args.robot == "ranger":
        from ranger_unifp_vec_env import RangerUniFPVecEnv
        env_class = RangerUniFPVecEnv
    else:
        env_class = UniFPVecEnv
    env = env_class(
        num_envs=args.num_envs,
        device=args.device,
        domain_randomization=args.domain_randomization,
        force_curriculum=args.force_curriculum,
        observation_noise=args.observation_noise,
        command_sampling=not args.no_command_sampling,
        random_goals=not args.fixed_goal,
        curriculum_start_step=args.curriculum_start_step,
        rough_terrain=args.rough_terrain,
        push_robots=args.push_robots,
        push_interval_steps=args.push_interval_steps,
        max_push_vel_xy=args.max_push_vel_xy,
        **({"ranger_explicit_pd": args.ranger_explicit_pd} if args.robot == "ranger" else {}),
    )

    try:
        if force_schedule is not None:
            force_tape = {
                key: torch.tensor([row[key] for row in force_schedule["samples"][:args.steps]],
                                  dtype=torch.float32, device=env.device)
                for key in ("external_world_n", "command_yaw_n")
            }
            for value in force_tape.values():
                if value.shape != (args.steps, args.num_envs, 3) or not torch.isfinite(value).all():
                    raise ValueError("Force tape must contain finite [steps, num_envs, 3] arrays")

            def replay_force(global_policy_step, episode_steps, *, substep_index=0):
                # A global tape continues across episode resets and consumes no RNG.
                # Hold each force constant over the four physics substeps.
                env.curriculum.command.value[:] = force_tape["command_yaw_n"][global_policy_step]
                env.curriculum.external.value[:] = force_tape["external_world_n"][global_policy_step]
                return env.curriculum.command.value, env.curriculum.external.value

            env.curriculum.update = replay_force

        goal_marker = actual_marker = None
        if args.visualize_goals:
            goal_marker = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/World/Visuals/UniFP/goal",
                    markers={
                        "goal": sim_utils.SphereCfg(
                            radius=0.055,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.1, 1.0, 0.1),
                            ),
                        ),
                    },
                )
            )
            actual_marker = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/World/Visuals/UniFP/actual",
                    markers={
                        "actual": sim_utils.SphereCfg(
                            radius=0.04,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.1, 0.3, 1.0),
                            ),
                        ),
                    },
                )
            )

        model = ActorCritic(
            env.num_obs,
            env.num_privileged_obs,
            env.num_pred_obs,
            env.num_single_obs,
            env.num_actions,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            init_noise_std=1.0,
        ).to(env.device)

        checkpoint = torch.load(
            args.checkpoint,
            map_location=env.device,
            weights_only=True,
        )
        if "model_state_dict" not in checkpoint:
            raise KeyError("Checkpoint does not contain model_state_dict")
        if checkpoint.get("config", {}).get("robot", "b2z1") != args.robot:
            raise ValueError("Checkpoint belongs to a different robot")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        observations = env.get_observations()
        reward_sum = 0.0
        done_count = 0
        min_base_height = float("inf")
        max_action_abs = 0.0
        max_latent_target_abs = 0.0
        trace = []
        begin = time.perf_counter()

        with torch.inference_mode():
            for step in range(args.steps):
                diagnostics = None
                if args.trace is not None:
                    # Estimate and target refer to the SAME pre-action history.
                    # obs_pred is a supervision label, never an actor input.
                    prediction = model.get_student_latent(observations["obs"])
                    truth_yaw = observations["obs_pred"][:, 6:9] * 100.0
                    force_world = quat_apply_yaw(env.robot.data.root_quat_w, truth_yaw)
                    command_yaw = env.actor.commands[:, 9:12].clone()
                    command_world = quat_apply_yaw(env.robot.data.root_quat_w, command_yaw)
                    base_pos = env.robot.data.root_pos_w - env.scene.env_origins
                    ee_pos = env.robot.data.body_pos_w[:, env.gripper_id]
                    force_target = env.goal.goal_world + (force_world + command_world) / 200.0
                    indices = env.controller.action_indices
                    fields = {
                        "force_estimate_yaw_n": prediction[:, 6:9] * 100.0,
                        "force_truth_yaw_n": truth_yaw,
                        "force_truth_world_n": force_world,
                        "force_command_yaw_n": command_yaw,
                        "force_command_world_n": command_world,
                        "force_target_world_m": force_target,
                        "ee_position_world_m": ee_pos,
                        "goal_position_world_m": env.goal.goal_world,
                        "base_position_m": base_pos,
                        "base_yaw_quaternion_wxyz": yaw_quat(env.robot.data.root_quat_w),
                        "base_velocity_body_m_s": env.robot.data.root_lin_vel_b,
                        "joint_position_rad": env.robot.data.joint_pos[:, indices],
                        "joint_velocity_rad_s": env.robot.data.joint_vel[:, indices],
                        "joint_applied_effort_nm": env.robot.data.applied_torque[:, indices],
                        "episode_steps": env.episode_length_buf,
                    }
                    diagnostics = {key: value.detach().cpu().tolist() for key, value in fields.items()}
                    diagnostics["policy_time_s"] = step * 0.02
                    if force_schedule is not None:
                        diagnostics["next_external_world_n"] = force_tape["external_world_n"][step].cpu().tolist()
                        diagnostics["next_command_yaw_n"] = force_tape["command_yaw_n"][step].cpu().tolist()
                # act_inference uses the encoder and deterministic actor mean.
                actions = model.act_inference({"obs": observations["obs"]})
                if actions.shape != (args.num_envs, env.num_actions):
                    raise RuntimeError(
                        f"Policy returned {tuple(actions.shape)}, expected "
                        f"({args.num_envs}, {env.num_actions})"
                    )
                if not torch.isfinite(actions).all():
                    raise RuntimeError("Policy returned nonfinite actions")

                max_action_abs = max(
                    max_action_abs,
                    float(actions.abs().max().item()),
                )
                observations, rewards, dones, _ = env.step(actions)
                if goal_marker is not None:
                    goal_marker.visualize(translations=env.goal.goal_world)
                    actual_marker.visualize(
                        translations=env.robot.data.body_pos_w[:, env.gripper_id]
                    )
                reward_sum += float(rewards.sum().item())
                done_count += int(dones.sum().item())
                min_base_height = min(
                    min_base_height,
                    float(
                        (
                            env.robot.data.root_pos_w[:, 2]
                            - env.scene.env_origins[:, 2]
                        ).min().item()
                    ),
                )

                # Decoder predictions are diagnostics; observations["obs_pred"]
                # is the supervision label. Neither is fed directly into actor.
                latent_target = model.get_student_latent(
                    observations["obs"]
                )
                max_latent_target_abs = max(
                    max_latent_target_abs,
                    float(latent_target.abs().max().item()),
                )

                if args.trace is not None:
                    root_pos = (
                        env.robot.data.root_pos_w - env.scene.env_origins
                    )
                    ee_pos = env.robot.data.body_pos_w[:, env.gripper_id]
                    trace.append(
                        {
                            "step": step + 1,
                            "policy_time_s": (step + 1) * 0.02,
                            "base_position_m": root_pos.detach().cpu().tolist(),
                            "base_height_m": root_pos[:, 2].detach().cpu().tolist(),
                            "ee_position_world_m": ee_pos.detach().cpu().tolist(),
                            "goal_position_world_m": env.goal.goal_world.detach().cpu().tolist(),
                            "current_goal_sphere": env.goal.current.detach().cpu().tolist(),
                            "action": actions.detach().cpu().tolist(),
                            "action_abs_max": actions.abs().amax(dim=-1).detach().cpu().tolist(),
                            "reward": rewards.detach().cpu().tolist(),
                            "done": dones.detach().cpu().tolist(),
                            "pre_action": diagnostics,
                        }
                    )

                if (step + 1) % max(1, min(100, args.steps)) == 0:
                    print(
                        "UNIFP_PLAY_PROGRESS",
                        json.dumps(
                            {
                                "step": step + 1,
                                "steps": args.steps,
                                "done_count": done_count,
                                "min_base_height_m": min_base_height,
                                "max_action_abs": max_action_abs,
                            }
                        ),
                        flush=True,
                    )

        elapsed = time.perf_counter() - begin
        report = {
            "robot": args.robot,
            "asset_version": getattr(env, "asset_version", None),
            "asset_path": getattr(env, "asset_path", None),
            "low_level_control": "implicit_pd" if getattr(env.controller, "implicit_pd", False) else "explicit_pd",
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_iterations": int(checkpoint.get("iterations", -1)),
            "num_envs": args.num_envs,
            "policy_steps": args.steps,
            "mean_reward": reward_sum / (args.num_envs * args.steps),
            "done_count": done_count,
            "min_base_height_m": min_base_height,
            "max_action_abs": max_action_abs,
            "max_decoder_prediction_abs": max_latent_target_abs,
            "elapsed_s": elapsed,
            "environment_policy_steps_per_s": (
                args.num_envs * args.steps / elapsed
            ),
            "domain_randomization": args.domain_randomization,
            "force_curriculum": args.force_curriculum,
            "force_schedule": str(args.force_schedule.resolve()) if args.force_schedule else None,
            "force_schedule_labels": force_schedule.get("environment_labels") if force_schedule else None,
            "rough_terrain": args.rough_terrain,
            "push_robots": args.push_robots,
            "note": (
                "This checkpoint comes from a short PPO integration run and "
                "is not a converged UniFP policy."
            ),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
        if args.trace is not None:
            args.trace.parent.mkdir(parents=True, exist_ok=True)
            args.trace.write_text(
                json.dumps({
                    "report": report,
                    "time_contract": "pre_action: observation/estimate/label before action; other fields: after transition, possibly reset",
                    "force_contract": "estimate and truth are external force in base_yaw, N; next_* is force applied over the following transition; force_target=(external_world+command_world)/200+goal",
                    "samples": trace,
                }, indent=2),
                encoding="utf-8",
            )
        print("UNIFP_PLAY_PASS", json.dumps(report), flush=True)
    finally:
        env.close()
        app.close(skip_cleanup=True)


try:
    main()
except BaseException as exc:
    print("UNIFP_PLAY_FAIL:", type(exc).__name__, str(exc), flush=True)
    raise

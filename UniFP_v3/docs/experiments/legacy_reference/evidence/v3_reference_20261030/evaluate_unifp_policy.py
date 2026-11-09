"""Run a fixed-seed UniFP policy evaluation and save quantitative metrics.

The evaluator uses the same student actor and Isaac Lab environment as
``play_unifp_policy.py``.  By default it evaluates the randomized spherical
end-effector trajectory with velocity-command sampling and force curriculum
disabled, so position tracking can be measured without mixing in those two
signals.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

from unifp_eval_protocol import load_schedule, make_schedule, validate_trace

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description="Evaluate a UniFP student policy")
parser.add_argument(
    "--checkpoint",
    type=Path,
    default=ROOT / "logs/ppo-baseline-official-timing-64-300/model_smoke.pt",
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=6000)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument(
    "--output",
    type=Path,
    default=ROOT / "logs/eval-v2-iter124-6000.json",
)
parser.add_argument(
    "--trace",
    type=Path,
    default=ROOT / "logs/eval-v2-iter124-6000.trace.json",
)
parser.add_argument(
    "--csv",
    type=Path,
    default=ROOT / "logs/eval-v2-iter124-6000.csv",
)
parser.add_argument("--domain_randomization", action="store_true")
parser.add_argument("--force_curriculum", action="store_true")
parser.add_argument("--curriculum_start_step", type=int, default=192001)
parser.add_argument("--rough_terrain", action="store_true")
parser.add_argument("--push_robots", action="store_true")
parser.add_argument("--push_interval_steps", type=int, default=400)
parser.add_argument("--max_push_vel_xy", type=float, default=0.8)
parser.add_argument("--observation_noise", action="store_true")
parser.add_argument(
    "--sample_commands",
    action="store_true",
    help="Also sample the random velocity commands used during training",
)
parser.add_argument(
    "--fixed_goal",
    action="store_true",
    help="Disable randomized end-effector goal trajectories",
)
parser.add_argument('--goal_schedule', type=Path, help='Shared JSON command tape; created if absent')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.num_envs < 1 or args.steps < 1:
    parser.error("--num_envs and --steps must be positive")
if args.seed < 0:
    parser.error("--seed must be nonnegative")
if args.curriculum_start_step < 0:
    parser.error("--curriculum_start_step must be nonnegative")
if not args.checkpoint.is_file():
    parser.error(f"Checkpoint does not exist: {args.checkpoint}")

if args.fixed_goal and args.goal_schedule:
    parser.error('--fixed_goal and --goal_schedule are mutually exclusive')
schedule = None
if args.goal_schedule:
    if args.goal_schedule.exists():
        schedule = load_schedule(args.goal_schedule, args.steps)
    else:
        schedule = make_schedule(args.steps, args.seed)
        args.goal_schedule.parent.mkdir(parents=True, exist_ok=True)
        args.goal_schedule.write_text(json.dumps(schedule, indent=2))
app = AppLauncher(args).app

import torch  # noqa: E402

from unifp_vec_env import UniFPVecEnv  # noqa: E402

sys.path.insert(0, str(ROOT / "UniFP"))
from legged_gym.b2_gym_learn.ppo_cse_pf.actor_critic import ActorCritic  # noqa: E402


def _percentiles(values):
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "p50": float(torch.quantile(tensor, 0.50).item()),
        "p95": float(torch.quantile(tensor, 0.95).item()),
        "p99": float(torch.quantile(tensor, 0.99).item()),
        "max": float(tensor.max().item()),
    }


def main():
    env = UniFPVecEnv(
        num_envs=args.num_envs,
        device=args.device,
        domain_randomization=args.domain_randomization,
        force_curriculum=args.force_curriculum,
        observation_noise=args.observation_noise,
        command_sampling=args.sample_commands,
        random_goals=not args.fixed_goal,
        curriculum_start_step=args.curriculum_start_step,
        seed=args.seed,
        rough_terrain=args.rough_terrain,
        push_robots=args.push_robots,
        push_interval_steps=args.push_interval_steps,
        max_push_vel_xy=args.max_push_vel_xy,
        capture_diagnostics=True,
    )

    try:
        if schedule is not None:
            env.goal.install_schedule(schedule)
            env.reset()
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
        checkpoint = torch.load(args.checkpoint, map_location=env.device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        observations = env.get_observations()
        position_errors = []
        position_errors_xyz = []
        roll_values = []
        pitch_values = []
        base_heights = []
        reward_values = []
        action_values = []
        adaptation_rmse_values = []
        adaptation_mae_values = []
        done_count = 0
        trace = []
        episodes = []
        episode_return = torch.zeros(args.num_envs, device=env.device)
        episode_terms = {}
        begin = time.perf_counter()

        with torch.inference_mode():
            for step in range(args.steps):
                # Snapshot owns every tensor; step() may mutate goals and reset bodies.
                state = env.state_snapshot()
                ee_position = state['ee_position_world_m']
                goal_position = state['goal_position_world_m']
                position_error_xyz_tensor = state['position_error_xyz_m']
                position_error_tensor = state['position_error_m']
                roll, pitch = state['roll_rad'], state['pitch_rad']
                base_height = state['base_height_m']
                predicted_obs = model.get_student_latent(observations["obs"])
                adaptation_error = predicted_obs - observations["obs_pred"]
                actions = model.act_inference({"obs": observations["obs"]})

                position_errors.extend(position_error_tensor.cpu().tolist())
                position_errors_xyz.extend(position_error_xyz_tensor.cpu().tolist())
                roll_values.extend(roll.abs().cpu().tolist())
                pitch_values.extend(pitch.abs().cpu().tolist())
                base_heights.extend(base_height.cpu().tolist())
                action_values.extend(actions.abs().amax(dim=-1).cpu().tolist())
                adaptation_rmse_values.extend(
                    torch.sqrt((adaptation_error.square().mean(dim=-1))).cpu().tolist()
                )
                adaptation_mae_values.extend(
                    adaptation_error.abs().mean(dim=-1).cpu().tolist()
                )

                next_observations, rewards, dones, extras = env.step(actions)
                done_count += int(dones.sum().item())
                reward_values.extend(rewards.cpu().tolist())

                pre_reset = {k: v.cpu().tolist() for k,v in extras['pre_reset_state'].items()}
                reason_masks = {k: v.cpu().tolist() for k,v in extras['termination_reasons'].items()}
                reward_terms = {k: v.cpu().tolist() for k,v in extras['reward_terms'].items()}
                row = {k: v.cpu().tolist() for k,v in state.items()}
                row.update({
                    'step': step+1, 'policy_time_s': step*0.02,
                    'action_abs_max': actions.abs().amax(dim=-1).cpu().tolist(),
                    'reward': rewards.cpu().tolist(), 'done': dones.cpu().tolist(),
                    'adaptation_rmse': adaptation_error.square().mean(dim=-1).sqrt().cpu().tolist(),
                    'adaptation_mae': adaptation_error.abs().mean(dim=-1).cpu().tolist(),
                    'transition': {'policy_time_s': (step+1)*0.02,
                                   'pre_reset_state': pre_reset,
                                   'termination_reasons': reason_masks,
                                   'reward_terms': reward_terms},
                })
                trace.append(row)
                episode_return += rewards
                for key,value in extras['reward_terms'].items():
                    if key not in episode_terms:
                        episode_terms[key] = torch.zeros_like(rewards)
                    episode_terms[key] += value
                for env_id in dones.nonzero(as_tuple=False).flatten().tolist():
                    episodes.append({
                        'env_id': env_id, 'end_step': step+1,
                        'length_steps': pre_reset['episode_steps'][env_id],
                        'return': float(episode_return[env_id].item()),
                        'reasons': [k for k,v in reason_masks.items() if v[env_id]],
                        'terminal_state': {k:v[env_id] for k,v in pre_reset.items()},
                        'last_action': actions[env_id].cpu().tolist(),
                        'last_reward_terms': {k:v[env_id] for k,v in reward_terms.items()},
                        'episode_reward_terms': {k:float(v[env_id].item()) for k,v in episode_terms.items()},
                    })
                episode_return[dones] = 0
                for value in episode_terms.values():
                    value[dones] = 0
                observations = next_observations

                if (step + 1) % 1000 == 0:
                    print(
                        "UNIFP_EVAL_PROGRESS",
                        json.dumps(
                            {
                                "step": step + 1,
                                "steps": args.steps,
                                "done_count": done_count,
                                "mean_position_error_m": sum(position_errors) / len(position_errors),
                            }
                        ),
                        flush=True,
                    )

        elapsed = time.perf_counter() - begin
        xyz_tensor = torch.tensor(position_errors_xyz, dtype=torch.float64)
        report = {
            "schema_version": 2,
            "state_timing": "top-level state at t before action; transition at t+1 before reset",
            "goal_mode": "global_command_tape" if schedule else ("initial_segment_only" if args.fixed_goal else "episode_random"),
            "goal_schedule_sha256": schedule['sha256'] if schedule else None,
            "goal_schedule": str(args.goal_schedule.resolve()) if schedule else None,
            "goal_segments_presented": len({r['segment'] for r in schedule['samples'][:args.steps]}) if schedule else None,
            "reward_summary": env.rewards.summary(),
            "completed_episode_mean_steps": sum(e['length_steps'] for e in episodes)/max(len(episodes),1),
            "completed_episode_count": len(episodes),
            "incomplete_episodes": [dict(env_id=i, length_steps=int(env.episode_length_buf[i].item()),
                                         episode_return=float(episode_return[i].item())) for i in range(args.num_envs)],
            "terminal_max_abs_roll_rad": max((abs(e['terminal_state']['roll_rad']) for e in episodes), default=None),
            "terminal_max_abs_pitch_rad": max((abs(e['terminal_state']['pitch_rad']) for e in episodes), default=None),
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_iterations": int(checkpoint.get("iterations", -1)),
            "seed": args.seed,
            "num_envs": args.num_envs,
            "policy_steps": args.steps,
            "simulated_seconds": args.steps * 0.02,
            "mean_position_error_m": float(sum(position_errors) / len(position_errors)),
            "position_error_percentiles_m": _percentiles(position_errors),
            "position_error_rmse_xyz_m": torch.sqrt(xyz_tensor.square().mean(dim=0)).tolist(),
            "position_error_mean_abs_xyz_m": xyz_tensor.abs().mean(dim=0).tolist(),
            "mean_abs_roll_rad": float(sum(roll_values) / len(roll_values)),
            "max_abs_roll_rad": float(max(roll_values)),
            "mean_abs_pitch_rad": float(sum(pitch_values) / len(pitch_values)),
            "max_abs_pitch_rad": float(max(pitch_values)),
            "min_base_height_m": float(min(base_heights)),
            "mean_reward": float(sum(reward_values) / len(reward_values)),
            "done_count": done_count,
            "mean_max_action_abs": float(sum(action_values) / len(action_values)),
            "max_action_abs": float(max(action_values)),
            "mean_adaptation_rmse": float(sum(adaptation_rmse_values) / len(adaptation_rmse_values)),
            "p95_adaptation_rmse": _percentiles(adaptation_rmse_values)["p95"],
            "mean_adaptation_mae": float(sum(adaptation_mae_values) / len(adaptation_mae_values)),
            "goal_resample_count": env.goal.resample_count,
            "goal_rejected_count": env.goal.rejected_count,
            "elapsed_s": elapsed,
            "environment_policy_steps_per_s": args.num_envs * args.steps / elapsed,
            "domain_randomization": args.domain_randomization,
            "force_curriculum": args.force_curriculum,
            "rough_terrain": args.rough_terrain,
            "push_robots": args.push_robots,
            "command_sampling": args.sample_commands,
            "random_goals": not args.fixed_goal,
            "note": (
                "Fixed-seed local evaluation. This is a development checkpoint and "
                "not a converged UniFP policy or a real-robot measurement."
            ),
        }
        payload = {'schema_version': 2, 'report': report, 'samples': trace, 'episodes': episodes}
        validate_trace(payload)
        if schedule is not None:
            for index,row in enumerate(trace):
                expected = torch.tensor(schedule['samples'][index]['sphere'])
                if not torch.allclose(torch.tensor(row['goal_sphere']), expected.expand(args.num_envs,-1), atol=1e-6):
                    raise AssertionError('Command tape changed with reset count')
                expected_next = torch.tensor(schedule['samples'][index+1]['sphere'])
                if not torch.allclose(torch.tensor(row['transition']['pre_reset_state']['goal_sphere']),
                                      expected_next.expand(args.num_envs,-1), atol=1e-6):
                    raise AssertionError('Transition goal is at the wrong tape position')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        args.output.with_suffix('.episodes.json').write_text(
            json.dumps({'report': report, 'episodes': episodes}, indent=2), encoding='utf-8'
        )
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        args.trace.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "step", "policy_time_s", "env_id", "base_height_m", "roll_abs_rad",
                    "pitch_abs_rad", "position_error_m", "reward", "done",
                    "action_abs_max", "adaptation_rmse", "adaptation_mae",
                ]
            )
            for row in trace:
                for env_id in range(args.num_envs):
                    writer.writerow(
                        [
                            row["step"], row["policy_time_s"], env_id,
                            row["base_height_m"][env_id], row["roll_abs_rad"][env_id],
                            row["pitch_abs_rad"][env_id], row["position_error_m"][env_id],
                            row["reward"][env_id], row["done"][env_id],
                            row["action_abs_max"][env_id], row["adaptation_rmse"][env_id],
                            row["adaptation_mae"][env_id],
                        ]
                    )
        print("UNIFP_EVAL_PASS", json.dumps(report), flush=True)
    finally:
        env.close()
        app.close(skip_cleanup=True)


try:
    main()
except BaseException as exc:
    print("UNIFP_EVAL_FAIL:", type(exc).__name__, str(exc), flush=True)
    raise

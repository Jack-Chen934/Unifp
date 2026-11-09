"""Exercise the callable UniFP PPO environment contract in Isaac Lab."""

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--robot", choices=("b2z1", "ranger"), default="b2z1")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=48)
parser.add_argument("--output", type=Path, default=Path("logs/vec-env-smoke.json"))
parser.add_argument("--domain_randomization", action="store_true")
parser.add_argument("--force_curriculum", action="store_true")
parser.add_argument("--curriculum_start_step", type=int, default=192001)
parser.add_argument("--rough_terrain", action="store_true")
parser.add_argument("--push_robots", action="store_true")
parser.add_argument("--push_interval_steps", type=int, default=400)
parser.add_argument("--max_push_vel_xy", type=float, default=0.8)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402
from unifp_vec_env import UniFPVecEnv  # noqa: E402


def main():
    if args.robot == "ranger":
        from ranger_unifp_vec_env import RangerUniFPVecEnv
        env_class = RangerUniFPVecEnv
    else:
        env_class = UniFPVecEnv
    env = env_class(
        num_envs=args.num_envs, device=args.device, domain_randomization=args.domain_randomization,
        force_curriculum=args.force_curriculum, curriculum_start_step=args.curriculum_start_step,
        observation_noise=False, command_sampling=True, random_goals=True,
        rough_terrain=args.rough_terrain, push_robots=args.push_robots,
        push_interval_steps=args.push_interval_steps, max_push_vel_xy=args.max_push_vel_xy,
    )
    try:
        obs = env.get_observations()
        assert {key: tuple(value.shape) for key, value in obs.items()} == {
            "obs": (args.num_envs, env.num_obs), "privileged_obs": (args.num_envs, env.num_privileged_obs),
            "obs_pred": (args.num_envs, 12),
        }
        width = env.num_single_obs
        critic_width = env.num_privileged_obs // 3
        motor_slice = env.critic_slices["motor_strength_delta"]
        assert torch.count_nonzero(obs["obs"].view(args.num_envs, 32, width)[:, :-1]) == 0
        # A kept reference must not change when the next environment step
        # shifts histories; PPO stores those transition observations.
        initial = obs["obs"]
        initial_snapshot = initial.clone()
        motor_before = env.domain.motor_strength.clone()
        reward_sum = 0.0
        done_sum = 0
        max_reward_tieout = 0.0
        reset_checked = False
        for step in range(args.steps):
            action = torch.zeros(args.num_envs, env.num_actions, device=env.device)
            action[:, getattr(env.controller, "base_action_count", 12):] = 0.1 * torch.sin(torch.tensor(step * 0.05, device=env.device))
            obs, reward, done, extras = env.step(action)
            assert torch.isfinite(reward).all()
            assert all(torch.isfinite(t).all() for t in obs.values())
            critic_frame = obs["privileged_obs"].view(args.num_envs, 3, critic_width)[:, -1]
            assert torch.allclose(critic_frame[:, :12], obs["obs_pred"])
            assert torch.allclose(
                critic_frame[:, motor_slice],
                env.domain.motor_strength - 1.0, atol=1e-5,
            )
            if step == 0:
                assert torch.equal(initial, initial_snapshot), "Previous PPO observation mutated in place"
            if step == 15:
                assert torch.equal(env.get_observations()["obs"], obs["obs"])
                env.reset(torch.tensor([0], device=env.device))
                reset_checked = True
                assert torch.count_nonzero(env.get_observations()["obs"].view(args.num_envs, 32, width)[0, :-1]) == 0
                assert torch.isfinite(env.get_observations()["obs"]).all()
            tieout = (reward - torch.stack(list(extras["reward_terms"].values())).sum(dim=0)).abs().max().item()
            max_reward_tieout = max(max_reward_tieout, tieout)
            reward_sum += reward.sum().item()
            done_sum += int(done.sum().item())
            assert extras["time_outs"].shape == (args.num_envs,)
        assert reset_checked
        # Force truth changes privileged targets, never the deployable actor
        # frame when physical state/actions/commands are held fixed.
        zero = torch.zeros(args.num_envs, 3, device=env.device)
        actor_before = env.actor.observe(env.robot, action, add_noise=False)[0].clone()
        target_before = env.privileged.observe(
            env.robot, env.controller, env.scene["feet"], env.actor.gait_phase, action,
            env.actor.commands, env.goal, zero, zero, push_history=False)[0]
        target_after = env.privileged.observe(
            env.robot, env.controller, env.scene["feet"], env.actor.gait_phase, action,
            env.actor.commands, env.goal, zero + 10., zero, push_history=False)[0]
        assert not torch.equal(target_before[:, 6:9], target_after[:, 6:9])
        assert torch.equal(actor_before, env.actor.observe(env.robot, action, add_noise=False)[0])
        torque = env.controller.compute(action, env.robot.data.joint_pos, env.robot.data.joint_vel,
                                        motor_strength=env.domain.motor_strength)
        assert (torque.abs() <= env.controller.effort_limits + 1e-5).all()
        # Official motor randomization scales actions, leaving zero-action
        # restoring/damping torque unchanged for the same physical state.
        zero_action = torch.zeros_like(action)
        nominal = env.controller.compute(zero_action, env.robot.data.joint_pos, env.robot.data.joint_vel,
                                         motor_strength=torch.ones_like(action))
        randomized = env.controller.compute(zero_action, env.robot.data.joint_pos, env.robot.data.joint_vel,
                                            motor_strength=env.domain.motor_strength)
        assert torch.equal(nominal, randomized)
        report = {
            "robot": args.robot, "force_truth_isolation_checked": True, "torque_limits_checked": True,
            "num_envs": args.num_envs, "policy_steps": args.steps,
            "observation_shapes": {key: list(value.shape) for key, value in obs.items()},
            "mean_reward": reward_sum / (args.num_envs * args.steps),
            "terminations": done_sum, "max_reward_tieout": max_reward_tieout,
            "partial_reset_checked": reset_checked,
            "motor_resampled_for_env0": bool(not torch.equal(motor_before[0], env.domain.motor_strength[0])),
            "domain_randomization": env.domain.summary(),
            "force_curriculum": env.curriculum.summary(),
            "disturbances": env.disturbance_summary(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("UNIFP_VEC_ENV_PASS", json.dumps({k: v for k, v in report.items() if k not in
                                                   ("domain_randomization", "force_curriculum")}), flush=True)
    finally:
        env.close()
        app.close()


if __name__ == "__main__":
    main()

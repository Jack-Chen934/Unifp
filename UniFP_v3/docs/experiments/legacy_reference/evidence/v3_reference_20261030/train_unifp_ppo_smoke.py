"""Local training using UniFP's released PPO and policy modules.

Starts Isaac Sim through AppLauncher and imports released code unchanged.
The official experiment's full scale needs many more iterations and terrain.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--robot", choices=("b2z1", "ranger"), default="b2z1")
parser.add_argument("--ranger_explicit_pd", action="store_true", help="Use historical Ranger explicit-PD dynamics")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--save_interval", type=int, default=50)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--iterations", type=int, default=1)
parser.add_argument("--steps_per_env", type=int, default=24)
parser.add_argument("--output_dir", type=Path, default=Path("logs/ppo-smoke"))
parser.add_argument("--domain_randomization", action="store_true")
parser.add_argument("--force_curriculum", action="store_true")
parser.add_argument("--curriculum_start_step", type=int, default=192001)
parser.add_argument("--resume_checkpoint", type=Path, default=None)
parser.add_argument("--rough_terrain", action="store_true")
parser.add_argument("--push_robots", action="store_true")
parser.add_argument("--push_interval_steps", type=int, default=400)
parser.add_argument("--max_push_vel_xy", type=float, default=0.8)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if min(args.num_envs, args.iterations, args.steps_per_env) < 1:
    parser.error("num_envs, iterations, and steps_per_env must be positive")
if args.seed < 0 or args.save_interval < 1:
    parser.error("seed must be nonnegative and save_interval positive")
if args.output_dir.joinpath("model_smoke.pt").exists():
    parser.error("Output already contains a model; use a fresh output_dir to preserve it")
if args.curriculum_start_step < 0:
    parser.error("curriculum_start_step must be nonnegative")
app = AppLauncher(args).app

import torch  # noqa: E402
from unifp_vec_env import UniFPVecEnv  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent / "UniFP"))
from legged_gym.b2_gym_learn.ppo_cse_pf.actor_critic import ActorCritic  # noqa: E402
from legged_gym.b2_gym_learn.ppo_cse_pf.ppo import PPO  # noqa: E402


def main():
    torch.manual_seed(args.seed)
    if args.robot == "ranger":
        from ranger_unifp_vec_env import RangerUniFPVecEnv
        env_class = RangerUniFPVecEnv
    else:
        env_class = UniFPVecEnv
    env = env_class(
        num_envs=args.num_envs, device=args.device, domain_randomization=args.domain_randomization,
        force_curriculum=args.force_curriculum, observation_noise=True, command_sampling=True, random_goals=True,
        curriculum_start_step=args.curriculum_start_step, seed=args.seed,
        rough_terrain=args.rough_terrain, push_robots=args.push_robots,
        push_interval_steps=args.push_interval_steps, max_push_vel_xy=args.max_push_vel_xy,
        **({"ranger_explicit_pd": args.ranger_explicit_pd} if args.robot == "ranger" else {}),
    )
    try:
        model = ActorCritic(
            env.num_obs, env.num_privileged_obs, env.num_pred_obs, env.num_single_obs,
            env.num_actions, actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128],
            init_noise_std=1.0,
        ).to(env.device)
        algorithm = PPO(
            model, device=env.device, entropy_coef=0.01, num_learning_epochs=5,
            num_mini_batches=4, learning_rate=1e-3, schedule="adaptive",
            gamma=0.99, lam=0.95, desired_kl=0.01,
        )
        algorithm.init_storage(
            env.num_envs, args.steps_per_env, [env.num_obs], [env.num_privileged_obs],
            [env.num_pred_obs], [env.num_actions],
        )
        previous_iterations = 0
        if args.resume_checkpoint is not None:
            saved = torch.load(args.resume_checkpoint, map_location=env.device, weights_only=True)
            model.load_state_dict(saved["model_state_dict"])
            algorithm.optimizer.load_state_dict(saved["optimizer_state_dict"])
            algorithm.adaptation_module_optimizer.load_state_dict(saved["adaptation_optimizer_state_dict"])
            previous_iterations = int(saved["iterations"])
            if int(saved["num_envs"]) != env.num_envs:
                raise ValueError("Resume checkpoint num_envs differs from current run")
            # The official force trigger uses a global policy-step counter.
            if saved.get('steps_per_env', args.steps_per_env) != args.steps_per_env:
                raise ValueError('Resume rollout length differs from checkpoint')
            old_config = saved.get('config', {})
            if old_config.get('robot', 'b2z1') != args.robot:
                raise ValueError('Resume checkpoint belongs to a different robot')
            if args.robot == "ranger" and old_config.get('ranger_explicit_pd', True) != args.ranger_explicit_pd:
                print("CONTROL_ADAPTATION: restoring learned weights/optimizers/counters with changed Ranger PD solver; dynamics differ from the saved run", flush=True)
            if args.robot == "ranger" and old_config.get('ranger_asset_version', 'v2') != env.asset_version:
                print(f"ASSET_ADAPTATION: {old_config.get('ranger_asset_version', 'v2')} -> {env.asset_version}; geometry/dynamics differ from the saved run", flush=True)
            for key in ('seed', 'domain_randomization', 'force_curriculum', 'curriculum_start_step',
                        'rough_terrain', 'push_robots', 'push_interval_steps', 'max_push_vel_xy'):
                if key in old_config and old_config[key] != getattr(args,key):
                    raise ValueError(f'Resume configuration differs: {key}')
            env.global_policy_steps = int(saved.get('global_policy_steps', previous_iterations * args.steps_per_env))
            algorithm.learning_rate = float(saved.get('learning_rate', algorithm.optimizer.param_groups[0]['lr']))
            for group in algorithm.optimizer.param_groups:
                group['lr'] = algorithm.learning_rate
        observations = env.get_observations()
        initial_learning_rate = algorithm.learning_rate
        initial_global_policy_steps = env.global_policy_steps
        begin = time.perf_counter()
        history = []
        learning_modules = {
            "encoder": model.adaptation_encoder_module, "decoder": model.adaptation_decoder_module,
            "actor": model.actor_body, "critic": model.critic_body,
        }
        initial_weights = {name: module[0].weight.detach().clone() for name, module in learning_modules.items()}
        args.output_dir.mkdir(parents=True, exist_ok=True)
        config = {k: str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
        config['low_level_control'] = 'implicit_pd' if getattr(env.controller, 'implicit_pd', False) else 'explicit_pd'
        if args.robot == "ranger":
            config.update(ranger_asset_version=env.asset_version, asset_path=env.asset_path)
        (args.output_dir/'config.json').write_text(json.dumps(config, indent=2))

        def save_checkpoint(iterations):
            data = {
                'schema_version': 2, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': algorithm.optimizer.state_dict(),
                'adaptation_optimizer_state_dict': algorithm.adaptation_module_optimizer.state_dict(),
                'iterations': iterations, 'num_envs': args.num_envs,
                'steps_per_env': args.steps_per_env, 'global_policy_steps': env.global_policy_steps,
                'total_environment_transitions': env.global_policy_steps*args.num_envs,
                'learning_rate': algorithm.learning_rate, 'config': config,
                'resume_semantics': ('fresh random model/optimizers; counters start at zero' if args.resume_checkpoint is None
                                     else 'model/optimizer/counters restored; simulation and RNG restart from seed'),
            }
            path = args.output_dir/f'model_iter_{iterations:06d}.pt'
            torch.save(data, path.with_suffix('.tmp'))
            path.with_suffix('.tmp').replace(path)
            # Keep the conventional final/latest filename for existing playback tools.
            torch.save(data, args.output_dir/'model_smoke.tmp')
            (args.output_dir/'model_smoke.tmp').replace(args.output_dir/'model_smoke.pt')
            (args.output_dir/'progress.json').write_text(json.dumps({
                'iterations': iterations, 'global_policy_steps': env.global_policy_steps,
                'learning_rate': algorithm.learning_rate, 'history': history,
                'force_curriculum_summary': env.curriculum.summary(),
            },indent=2))
            return args.output_dir/'model_smoke.pt'

        for iteration in range(args.iterations):
            total_reward = 0.0
            total_dones = 0
            with torch.inference_mode():
                for _ in range(args.steps_per_env):
                    actions = algorithm.act(
                        observations["obs"], observations["privileged_obs"], observations["obs_pred"],
                    )
                    observations, rewards, dones, infos = env.step(actions)
                    algorithm.process_env_step(rewards, dones, infos)
                    total_reward += rewards.sum().item()
                    total_dones += int(dones.sum().item())
                algorithm.compute_returns(observations["privileged_obs"])
            value_loss, surrogate_loss, adaptation_loss, adaptation_breakdown = algorithm.update()
            metrics = {
                "iteration": previous_iterations + iteration + 1,
                "global_policy_steps": env.global_policy_steps,
                "learning_rate": algorithm.learning_rate,
                "mean_reward": total_reward / (args.num_envs * args.steps_per_env),
                "dones": total_dones,
                "value_loss": value_loss,
                "surrogate_loss": surrogate_loss,
                "adaptation_loss": adaptation_loss,
                "adaptation_breakdown": adaptation_breakdown,
            }
            if any(not torch.isfinite(torch.tensor(v)) for v in (value_loss, surrogate_loss, adaptation_loss)):
                raise RuntimeError(f"Nonfinite PPO loss in iteration {iteration + 1}")
            history.append(metrics)
            print("UNIFP_PPO_ITER", json.dumps(metrics), flush=True)
            if iteration == 0 or (iteration+1) % args.save_interval == 0 or iteration+1 == args.iterations:
                checkpoint = save_checkpoint(previous_iterations+iteration+1)
        module_weights_changed = {name: not torch.equal(initial_weights[name], module[0].weight)
                                  for name, module in learning_modules.items()}
        weights_changed = all(module_weights_changed.values())
        if not weights_changed:
            raise RuntimeError(f"Official learner failed to update modules: {module_weights_changed}")
        saved = torch.load(checkpoint, map_location=env.device, weights_only=True)
        if saved["iterations"] != previous_iterations + args.iterations:
            raise RuntimeError("Checkpoint did not roundtrip")
        report = {
            "seed": args.seed, "config": config,
            "global_policy_steps": env.global_policy_steps,
            "total_environment_transitions": env.global_policy_steps*args.num_envs,
            "learning_rate": algorithm.learning_rate,
            "reward_summary": env.rewards.summary(),
            "initial_learning_rate": initial_learning_rate,
            "initial_global_policy_steps": initial_global_policy_steps,
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(env.device) if str(env.device).startswith('cuda') else 0,
            "num_envs": args.num_envs, "steps_per_env": args.steps_per_env,
            "iterations": previous_iterations + args.iterations,
            "iterations_this_run": args.iterations, "elapsed_s": time.perf_counter() - begin,
            "history": history, "weights_changed": weights_changed,
            "module_weights_changed": module_weights_changed,
            "checkpoint": str(checkpoint.resolve()),
            "domain_randomization": args.domain_randomization,
            "force_curriculum": args.force_curriculum,
            "curriculum_start_step": args.curriculum_start_step,
            "force_curriculum_summary": env.curriculum.summary(),
            "disturbance_summary": env.disturbance_summary(),
            "limitations": (
                "Local training budget does not establish convergence or paper-level reproduction. "
                "Rough terrain is a reduced Isaac Lab rough-flat approximation when enabled."
            ),
        }
        (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("UNIFP_PPO_PASS", json.dumps({k: v for k, v in report.items() if k != "history"}), flush=True)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)  # SimulationApp.close can otherwise mask a failed run.
    finally:
        env.close()
        app.close()


if __name__ == "__main__":
    main()

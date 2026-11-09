"""只读采集冻结策略的价值、真实奖励与reset前控制证据。"""

from __future__ import annotations

import argparse
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


class ObservedEnvironment(V3PosForce):
    """Read original per-substep torques and pre-reset policy metrics."""

    def _compute_v3_torques(self, actions: torch.Tensor) -> torch.Tensor:
        """Delegate control and count saturation at each physical substep.

        Args:
            actions: Original dimensionless policy-order requests [N,14].

        Returns:
            Unchanged native-DOF torques [N,14], in N m.

        Note:
            Only diagnostic counters change; control and random draws are untouched.
        """
        torques = super()._compute_v3_torques(actions)
        if getattr(self, "measurement_ready", False):
            self.saturated_counts += (
                torques[:, self.action_dof_indices].abs()
                >= 0.999 * self.torque_limits[self.action_dof_indices]
            )
        return torques

    def _accumulate_episode_metrics(
        self, done_ids: torch.Tensor, terminated: torch.Tensor
    ) -> list[dict[str, Any]]:
        """Save metrics before reset and preserve original episode accounting.

        Args:
            done_ids: Episode-ending actor indices [K].
            terminated: Physical-failure mask [N], before reset.

        Returns:
            Original completed episode records, including terminal transitions.

        Note:
            Stages a detached CPU row for this interval without modifying the task.
        """
        self.measured_errors = self._physical_tracking_errors().detach().cpu().numpy()
        self.measured_command_force = self.commands[:, 9:12].detach().cpu().numpy()
        self.measured_external_force = self.applied_ee_force_world.detach().cpu().numpy()
        self.measured_raw_rewards = self.reward_raw.detach().cpu().numpy()
        records = super()._accumulate_episode_metrics(done_ids, terminated)
        self.completed_records.extend(records)
        return records


def main() -> None:
    """Collect bounded fresh trajectories with no model or optimizer updates.

    Raises:
        AssertionError: Source identity, task settings, terminal masks or metrics differ.
        OSError: A checkpoint or a new diagnostic output cannot be accessed.

    Note:
        Creates 64 actors, executes 2000 policy steps, and destroys its simulator.
        Formal training and all checkpoint files remain untouched.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    set_seed(1001)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    snapshot = copy.deepcopy(payload["config_snapshot"])
    source = Path(legged_gym.__file__).resolve().parents[1]
    asset_basename = Path(snapshot["environment"]["asset"]["file"]).name
    relative_asset = next(
        name for name in payload["identity"]["asset_manifest"]
        if name.endswith("/" + asset_basename)
    )
    snapshot["environment"]["asset"]["file"] = str(source / relative_asset)
    assert build_run_identity(source, snapshot, payload["identity"]["runtime"]) == payload["identity"]
    cfg, _ = resolve_v3_configs(
        argparse.Namespace(reproduction_part="B", run_purpose="debug", seed=1001,
                           num_envs=64, max_iterations=1), purpose="debug"
    )
    actual, expected = class_to_dict(cfg), copy.deepcopy(snapshot["environment"])
    for config in (actual, expected):
        config.pop("seed")
        config.pop("run")
        config["env"].pop("num_envs")
    assert actual == expected
    parameters = gymapi.SimParams()
    gymutil.parse_sim_config(class_to_dict(cfg.sim), parameters)
    parameters.use_gpu_pipeline = parameters.physx.use_gpu = True
    parameters.physx.num_threads = 4
    properties = {
        key: payload["environment_training_state"][key][:64]
        for key in ("mass_com_delta", "friction")
    }
    environment = ObservedEnvironment(
        cfg, parameters, gymapi.SIM_PHYSX, "cuda:0", True, creation_properties=properties
    )
    try:
        environment.completed_records = []
        environment.saturated_counts = torch.zeros(64, 14, device="cuda:0", dtype=torch.long)
        environment.measurement_ready = True
        environment.set_evaluation_push_seed(2001)
        progress = payload["progress"]
        environment.set_training_progress(progress["completed_iterations"], progress["training_policy_steps"])
        model = V3ActorCritic(1856, 294, 12, 58, 14, **snapshot["training"]["policy"]).cuda()
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.eval()
        initial_parameters = {name: value.detach().clone() for name, value in model.state_dict().items()}
        noise_generator = torch.Generator(device="cuda:0").manual_seed(3001)
        observations = environment.get_observations()
        rows: dict[str, list[np.ndarray]] = {
            name: [] for name in (
                "value", "next_value", "reward", "terminated", "timeout", "errors",
                "saturation", "action_mean", "action", "command_force", "external_force",
                "raw_rewards", "episode_step", "push"
            )
        }
        for step in range(2000):
            environment.set_evaluation_policy_step(progress["training_policy_steps"] + step)
            environment.saturated_counts.zero_()
            with torch.inference_mode():
                value = model.evaluate(observations["privileged_obs"]).squeeze(-1)
                model.update_distribution(observations["obs"])
                mean = model.action_mean.clone()
                action = mean + torch.randn(mean.shape, device="cuda:0", generator=noise_generator) * model.std
                episode_step = environment.episode_length_buf.clone()
                observations, reward, done, infos = environment.step(action)
                next_value = model.evaluate(observations["privileged_obs"]).squeeze(-1)
                ending_ids = infos["terminal_env_ids"]
                if ending_ids.numel():
                    next_value[ending_ids] = model.evaluate(infos["terminal_privileged_obs"]).squeeze(-1)
                assert torch.equal(done.bool(), infos["terminated"] | infos["time_outs"])
                step_data = {
                    "value": value, "next_value": next_value, "reward": reward,
                    "terminated": infos["terminated"], "timeout": infos["time_outs"],
                    "errors": environment.measured_errors,
                    "saturation": environment.saturated_counts.float() / cfg.control.decimation,
                    "action_mean": mean, "action": action,
                    "command_force": environment.measured_command_force,
                    "external_force": environment.measured_external_force,
                    "raw_rewards": environment.measured_raw_rewards,
                    "episode_step": episode_step,
                    "push": np.full(64, infos["base_push_applied"], dtype=bool),
                }
                for name, item in step_data.items():
                    rows[name].append(item.detach().cpu().numpy() if isinstance(item, torch.Tensor) else item)
            if (step + 1) % 500 == 0:
                print(json.dumps({"iteration": progress["completed_iterations"], "steps": step + 1}), flush=True)
        arrays = {name: np.stack(values) for name, values in rows.items()}
        assert all(np.isfinite(value).all() for value in arrays.values())
        assert all(torch.equal(initial_parameters[name], value) for name, value in model.state_dict().items())
        np.savez_compressed(args.output / "trajectory.npz", **arrays)
        (args.output / "episodes.json").write_text(json.dumps(environment.completed_records, ensure_ascii=False) + "\n")
        summary = {
            "checkpoint_iteration": progress["completed_iterations"],
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "source_identity_matches": True, "parameters_unchanged": True,
            "policy_updates": 0, "num_envs": 64, "policy_steps": 2000,
            "transitions": 128000, "gamma": snapshot["training"]["algorithm"]["gamma"],
            "lam": snapshot["training"]["algorithm"]["lam"],
            "policy_dt_s": environment.dt, "physical_substeps_per_policy_step": cfg.control.decimation,
            "reward_names": list(environment.reward_names),
            "reward_weights": environment.configured_reward_weights,
            "std": model.std.detach().cpu().tolist(),
            "physical_failures": int(arrays["terminated"].sum()),
            "timeouts": int(arrays["timeout"].sum()),
            "script_sha256": file_sha256(Path(__file__)),
            "说明": "新episode采样，非训练历史复演；高斯采样独立噪声；真正终止前后继价值已单独保存；零策略更新。",
        }
        (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    finally:
        environment.gym.destroy_sim(environment.sim)


if __name__ == "__main__":
    main()

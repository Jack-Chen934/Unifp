"""Observe the real B training entry during approved full-scale resume checks."""

import json
import os
import resource
import subprocess
import time
from pathlib import Path
from typing import Any, Dict

import isaacgym  # noqa: F401
import numpy as np
import torch
from legged_gym.scripts import train_v3
from legged_gym.b2_gym_learn.ppo_cse_pf.v3_on_policy_runner import V3OnPolicyRunner


def assert_tensor_tree_equal(actual: Any, expected: Any) -> None:
    """Compare real restored nested state without consuming RNG or changing it.

    Args:
        actual: Live optimizer/model/RNG state, potentially on CUDA.
        expected: CPU checkpoint state with the same nested structure.

    Raises:
        AssertionError: A tensor value, mapping key, sequence or scalar differs.
    """
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual.detach().cpu(), expected.detach().cpu(), rtol=0., atol=0.)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_tensor_tree_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for live, saved in zip(actual, expected):
            assert_tensor_tree_equal(live, saved)
    else:
        assert actual == expected


class ObservedRunner(V3OnPolicyRunner):
    """Add run-owned evidence around the unchanged production load/learn path."""

    def load(self, path: Path, *, transition_from_a: bool = False) -> None:
        """Verify complete restoration and the single reset's pre-consumption RNG.

        Args:
            path: Explicit full A8000 or B8010 checkpoint.
            transition_from_a: Use the production A-to-B validation when true.

        Raises:
            AssertionError: Learning state, creation samples, clocks or RNG differ.

        Note:
            Wraps only this diagnostic instance's reset while calling real load.
            Physics properties are never rewritten after actor creation.
        """
        payload = torch.load(path, map_location="cpu", weights_only=False)
        reset = self.env.reset
        reset_calls = []

        def checked_reset() -> Any:
            """Check restored RNG before the real reset consumes random samples.

            Returns:
                The unchanged production reset result.

            Raises:
                AssertionError: Any restored random generator state differs.

            Note:
                Records the call and executes the original reset exactly once.
            """
            captured = self._capture_rng()
            assert_tensor_tree_equal(captured, payload["rng_state"])
            reset_calls.append(True)
            return reset()

        self.env.reset = checked_reset
        try:
            super().load(path, transition_from_a=transition_from_a)
        finally:
            self.env.reset = reset
        assert len(reset_calls) == 1
        assert_tensor_tree_equal(self.alg.actor_critic.state_dict(), payload["model_state_dict"])
        assert_tensor_tree_equal(self.alg.optimizer.state_dict(), payload["main_optimizer_state_dict"])
        assert_tensor_tree_equal(self.alg.adaptation_module_optimizer.state_dict(), payload["auxiliary_optimizer_state_dict"])
        assert self.alg.learning_rate == payload["learning_rate_state"]["main"]
        state = self.env.get_training_state()
        for key in ("mass_com_delta", "friction"):
            assert_tensor_tree_equal(state[key], payload["environment_training_state"][key])
        assert self.env.last_push_policy_step == payload["environment_training_state"]["last_push_policy_step"]
        assert not self.env.applied_ee_force_world.any() and not self.env.pending_ee_force_world.any()
        assert not self.env.episode_length_buf.any()
        self.restore_report = {"model_and_both_adam_equal": True, "rng_before_reset_equal": True,
            "creation_samples_equal": True, "fresh_reset_calls": len(reset_calls), "initial_forces_zero": True,
            "parent_progress": payload["progress"], "parent_learning_rates": payload["learning_rate_state"],
            "parent_adam_steps": {prefix: sorted({int(value["step"]) for value in payload[prefix + "_optimizer_state_dict"]["state"].values()})
                                  for prefix in ("main", "auxiliary")}}

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Collect actual rollout force/label telemetry while production PPO learns.

        Args:
            num_learning_iterations: Approved additional complete updates, 10 or 1.
            init_at_random_ep_len: Forwarded unchanged; production requires false.

        Raises:
            AssertionError: Force labels never reach rollout or Adam fails to advance.

        Note:
            Writes diagnostics in the new validation run only. Does not alter
            control, observations, rewards, optimizer settings or physical states.
        """
        assert self.env.num_envs == 4096 and self.identity["reproduction_part"] == "B"
        action = self.alg.act
        step = self.env.step
        telemetry: Dict[str, Any] = {"policy_steps": 0, "nonzero_label_rows": 0,
            "external_force_peak_n": 0., "command_force_peak_n": 0., "physical_failures": 0,
            "first_step_physical_failures": None}

        def observed_act(obs: torch.Tensor, privileged: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            """Observe actual auxiliary targets passed into the production rollout.

            Args:
                obs: Actor history [4096,1856].
                privileged: Critic history [4096,294].
                labels: Previous-boundary auxiliary targets [4096,12].

            Returns:
                Original sampled policy actions [4096,14].

            Note:
                Counts labels without changing values or RNG consumption.
            """
            telemetry["nonzero_label_rows"] += int((labels[:, 6:9] != 0.).any(dim=1).sum())
            return action(obs, privileged, labels)

        def observed_step(actions: torch.Tensor) -> Any:
            """Read submitted interval force and failure counts around real step.

            Args:
                actions: Original Gaussian policy actions [4096,14].

            Returns:
                Unmodified production observations, rewards, dones and infos.

            Note:
                Samples telemetry only; normal failures still reset locally.
            """
            telemetry["external_force_peak_n"] = max(telemetry["external_force_peak_n"], float(self.env.pending_ee_force_world.abs().max()))
            telemetry["command_force_peak_n"] = max(telemetry["command_force_peak_n"], float(self.env.commands[:, 9:12].abs().max()))
            result = step(actions)
            failures = int(result[3]["terminated"].sum())
            if telemetry["policy_steps"] == 0:
                telemetry["first_step_physical_failures"] = failures
            telemetry["physical_failures"] += failures
            telemetry["policy_steps"] += 1
            return result

        self.alg.act, self.env.step = observed_act, observed_step
        started = time.perf_counter()
        try:
            super().learn(num_learning_iterations, init_at_random_ep_len)
        finally:
            self.alg.act, self.env.step = action, step
        assert telemetry["policy_steps"] == 24 * num_learning_iterations
        assert telemetry["nonzero_label_rows"] > 0 and telemetry["external_force_peak_n"] > 0.
        assert telemetry["first_step_physical_failures"] < 4096
        adam_steps = {name: sorted({int(value["step"]) for value in optimizer.state_dict()["state"].values()})
                      for name, optimizer in (("main", self.alg.optimizer), ("auxiliary", self.alg.adaptation_module_optimizer))}
        for name in adam_steps:
            assert min(adam_steps[name]) > min(self.restore_report["parent_adam_steps"][name])
        report = {"passed": True, "num_envs": 4096, "iterations": num_learning_iterations,
            "completed_iterations": self.completed_iterations, "training_policy_steps": self.training_policy_steps,
            "training_transitions": self.training_transitions, "rollout_step": self.alg.storage.step,
            "telemetry": telemetry, "restoration": self.restore_report, "adam_steps": adam_steps,
            "wall_time_s": time.perf_counter() - started,
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
            "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device),
            "cpu_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "gpu_process_memory": subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], text=True),
            "pid": os.getpid(), "identity": self.identity}
        (self.log_dir / "native_training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    train_v3.V3OnPolicyRunner = ObservedRunner
    train_v3.main()

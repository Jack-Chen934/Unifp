"""Run approved B learning with read-only observations at saved boundaries."""

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import isaacgym  # noqa: F401
from legged_gym.scripts import train_v3
from legged_gym.b2_gym_learn.ppo_cse_pf.v3_on_policy_runner import V3OnPolicyRunner


def observation_commands(checkpoint: Path, destination: Path, iteration: int,
                         source_root: Path, force_references: Dict[str, str]) -> List[Tuple[str, List[str]]]:
    """Build frozen intermediate/final commands without executing or sampling.

    Args:
        checkpoint: Atomically saved complete B training checkpoint.
        destination: New evaluation root for this cumulative training boundary.
        iteration: Cumulative completed iteration, including the A8000 parent.
        source_root: Frozen UniFP_v3 checkout containing the three manifests.
        force_references: Per-scenario A8000 archive roots from verified start evidence.

    Returns:
        Named CLI commands: S0/S1/S2, selected force conditions, contact and 0N.
        Non-observation boundaries return an empty list.

    Raises:
        ValueError: An A reference differs in evaluation fields beyond its schedule.
        OSError: A required frozen manifest cannot be read.

    Note:
        The contact evaluator runs its contact and zero-control quotas together.
        Force comparisons use the original sealed manifest so scheduling changes
        do not invalidate the evaluator's exact archive hash requirement.
    """
    force_path = source_root / "configs/v3_b_initial_v1.json"
    force_manifest = json.loads(force_path.read_text())
    schedule = force_manifest["observation_schedule"]
    final = iteration == schedule["final_cumulative_iteration"]
    if not final and iteration not in schedule["intermediate_cumulative_iterations"]:
        return []
    common = ["--reproduction_part", "B", "--run_purpose", "evaluation",
              "--checkpoint", str(checkpoint), "--num_envs", "32", "--sim_device", "cuda:0",
              "--rl_device", "cuda:0", "--pipeline", "gpu", "--num_threads", "4", "--headless"]
    zero_path = source_root / schedule["zero_force_manifest"]
    zero_manifest = json.loads(zero_path.read_text())
    commands = [("zero_force", [sys.executable, "-m", "legged_gym.scripts.evaluate_v3", *common,
        "--seed", str(zero_manifest["seed"]), "--protocol", "zero_force_regression",
        "--manifest", str(zero_path), "--output_dir", str(destination / "zero_force")])]
    scenarios = list(force_manifest["scenarios"]) if final else schedule["intermediate_force_scenarios"]
    for scenario in scenarios:
        reference_manifest_path = Path(force_references[scenario]) / scenario / "evaluation_manifest.json"
        reference_manifest = json.loads(reference_manifest_path.read_text())
        # Scheduling chooses when to evaluate; every field defining the evaluation must match.
        if ({name: value for name, value in force_manifest.items() if name != "observation_schedule"}
                != {name: value for name, value in reference_manifest.items() if name != "observation_schedule"}):
            raise ValueError("A8000 reference evaluation protocol differs beyond observation_schedule")
        commands.append((scenario, [sys.executable, "-m", "legged_gym.scripts.evaluate_v3", *common,
            "--seed", str(force_manifest["seed"]), "--protocol", "force_tracking",
            "--manifest", str(reference_manifest_path), "--scenario", scenario,
            "--output_dir", str(destination / scenario), "--video",
            "--reference_archive", force_references[scenario]]))
    contact_path = source_root / schedule["contact_manifest"]
    contact_manifest = json.loads(contact_path.read_text())
    commands.append(("contact", [sys.executable, "-m", "legged_gym.scripts.evaluate_v3_contact", *common,
        "--seed", str(contact_manifest["seed"]), "--protocol", "fixed_wall_contact",
        "--manifest", str(contact_path), "--output_dir", str(destination / "contact"), "--video"]))
    return commands


class ObservedBRunner(V3OnPolicyRunner):
    """Observe saved complete boundaries while retaining the original PPO loop."""

    force_references: Dict[str, str] = {}

    def _append_json(self, filename: str, record: Dict[str, Any]) -> None:
        """Persist original logs and inspect finite metrics every 1000 B updates.

        Args:
            filename: Original run-owned JSONL basename.
            record: Unmodified learning/checkpoint metrics from the real runner.

        Note:
            Ordinary physical failures remain episode statistics and never stop
            training. The original writer rejects non-finite JSON metrics.
        """
        super()._append_json(filename, record)
        if filename == "iterations.jsonl" and (self.completed_iterations - 8000) % 1000 == 0:
            super()._append_json("observation_health.jsonl", {
                "completed_iterations": self.completed_iterations, "finite_metrics": True,
                "training_policy_steps": self.training_policy_steps,
                "training_transitions": self.training_transitions,
                "collection_s": record["collection_s"], "update_s": record["update_s"],
                "episode_metrics": {key: value for key, value in record.items() if key.startswith("episode/")}})

    def save(self, path: Path) -> None:
        """Save normally, then pause learning for independent frozen evaluations.

        Args:
            path: New complete-boundary checkpoint selected by the original runner.

        Raises:
            subprocess.CalledProcessError: An evaluation fails to execute.
            AssertionError: An evaluation reports parameter updates.

        Note:
            Uses the same GPU sequentially while training is paused. Child
            processes cannot consume the parent's RNG or mutate its model.
            Performance assessments are recorded without gating further learning.
        """
        super().save(path)
        destination = self.log_dir / "observations" / f"iteration_{self.completed_iterations:06d}"
        source_root = Path(train_v3.__file__).resolve().parents[2]
        commands = observation_commands(path, destination, self.completed_iterations, source_root, self.force_references)
        if not commands:
            return
        destination.mkdir(parents=True, exist_ok=False)
        for name, command in commands:
            started = time.perf_counter()
            with (destination / f"{name}.console.log").open("x") as console:
                subprocess.run(command, cwd=source_root, stdout=console, stderr=subprocess.STDOUT, check=True)
            summary_path = destination / name / "summary.json"
            summary = json.loads(summary_path.read_text())
            assert summary["policy_updates"] == 0
            super()._append_json("observation_events.jsonl", {
                "completed_iterations": self.completed_iterations, "scenario": name,
                "summary_path": str(summary_path), "policy_updates": 0,
                "wall_time_s": time.perf_counter() - started,
                "assessment": summary.get("assessment", summary.get("scenarios"))})


if __name__ == "__main__":
    evidence_path = Path(sys.argv[sys.argv.index("--start_evidence") + 1])
    ObservedBRunner.force_references = json.loads(evidence_path.read_text())["stages"]["evaluation_baseline"]["force_reference_archives"]
    train_v3.V3OnPolicyRunner = ObservedBRunner
    train_v3.main()

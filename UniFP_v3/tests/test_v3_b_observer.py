"""Verify approved B observation boundaries and subprocess isolation."""

import json
import copy
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from tools.train_v3_b_observed import ObservedBRunner, observation_commands
from legged_gym.b2_gym_learn.ppo_cse_pf.v3_on_policy_runner import V3OnPolicyRunner


def test_b_observation_commands_preserve_schedule_and_protocols(tmp_path: Path) -> None:
    """Use full frozen quotas at the early, later and final boundaries.

    Args:
        tmp_path: New synthetic checkpoint and evaluation root; no Gym is created.

    Raises:
        AssertionError: Schedule, seed, source stage or evaluated conditions differ.
    """
    source_root = Path(__file__).resolve().parents[1]
    checkpoint = tmp_path / "model.pt"
    references = {name: str(tmp_path / "A8000" / name)
                  for name in json.loads((source_root / "configs/v3_b_initial_v1.json").read_text())["scenarios"]}
    reference_manifest = json.loads((source_root / "configs/v3_b_initial_v1.json").read_text())
    reference_manifest["observation_schedule"]["intermediate_cumulative_iterations"] = [16000, 24000, 32000]
    for name, reference in references.items():
        archive = Path(reference) / name
        archive.mkdir(parents=True)
        (archive / "evaluation_manifest.json").write_text(json.dumps(reference_manifest))
    for iteration in (8000, 9000, 15800, 16200, 39800):
        assert observation_commands(checkpoint, tmp_path, iteration, source_root, references) == []
    for iteration in (12000, 16000, 24000, 32000, 40000):
        commands = observation_commands(checkpoint, tmp_path, iteration, source_root, references)
        names = [name for name, _ in commands]
        assert names[0] == "zero_force" and names[-1] == "contact"
        assert len(names) == (16 if iteration == 40000 else 3)
        assert names[1:-1] == (list(json.loads((source_root / "configs/v3_b_initial_v1.json").read_text())["scenarios"])
                               if iteration == 40000 else ["R2"])
        for name, command in commands:
            assert command[command.index("--reproduction_part") + 1] == "B"
            assert command[command.index("--num_envs") + 1] == "32"
            assert "--smoke" not in command and "--max_iterations" not in command
            assert command[command.index("--checkpoint") + 1] == str(checkpoint)
            if name in references:
                assert command[command.index("--reference_archive") + 1] == references[name]


def test_b_save_observes_only_after_original_checkpoint(tmp_path: Path, monkeypatch: Any) -> None:
    """Exercise the actual save hook while replacing only filesystem/subprocess I/O.

    Args:
        tmp_path: Run-owned observation output root.
        monkeypatch: Pytest fixture replacing external save, launch and JSON append.

    Raises:
        AssertionError: Evaluation precedes save, changes updates or loses an event.
    """
    runner = object.__new__(ObservedBRunner)
    runner.log_dir, runner.completed_iterations = tmp_path, 16000
    runner.force_references = {"R2": str(tmp_path / "A8000/R2")}
    reference_archive = Path(runner.force_references["R2"]) / "R2"
    reference_archive.mkdir(parents=True)
    reference_archive.joinpath("evaluation_manifest.json").write_text(
        (Path(__file__).resolve().parents[1] / "configs/v3_b_initial_v1.json").read_text())
    events: List[Dict[str, Any]] = []
    saved: List[Path] = []

    def save_boundary(instance: Any, path: Path) -> None:
        """Record original checkpoint publication without training state.

        Args:
            instance: Live diagnostic runner, unused by the storage replacement.
            path: Complete-boundary destination selected by the real hook.
        """
        saved.append(path)

    def evaluate(command: List[str], **kwargs: Any) -> None:
        """Publish a read-only summary in place of launching native simulation.

        Args:
            command: Real frozen evaluation CLI built by the production wrapper.
            **kwargs: Standard subprocess arguments retained by the caller.

        Raises:
            AssertionError: The parent checkpoint was not saved first.
        """
        assert saved == [tmp_path / "model_016000.pt"]
        output = Path(command[command.index("--output_dir") + 1])
        output.mkdir()
        (output / "summary.json").write_text(json.dumps({"policy_updates": 0, "assessment": "fail"}))

    def append(instance: Any, filename: str, record: Dict[str, Any]) -> None:
        """Collect observation events while omitting external durable writes.

        Args:
            instance: Live runner whose checkpoint has already been saved.
            filename: Real event basename selected by the observer.
            record: Read-only evaluation result, including performance failures.
        """
        events.append(record)

    monkeypatch.setattr(V3OnPolicyRunner, "save", save_boundary)
    monkeypatch.setattr(V3OnPolicyRunner, "_append_json", append)
    monkeypatch.setattr("tools.train_v3_b_observed.subprocess.run", evaluate)
    runner.save(tmp_path / "model_016000.pt")
    assert [event["scenario"] for event in events] == ["zero_force", "R2", "contact"]
    assert all(event["assessment"] == "fail" and event["policy_updates"] == 0 for event in events)


def test_observer_reference_reaches_real_evaluator_guard(tmp_path: Path) -> None:
    """Exercise observer CLI and evaluator reference checks with real manifests.

    Args:
        tmp_path: Independent A archive and B evaluation output, without native Gym.

    Raises:
        AssertionError: Observation-only changes block comparison or wrong
            thresholds pass the observer's protocol check.
    """
    from legged_gym.scripts import evaluate_v3

    source_root = Path(__file__).resolve().parents[1]
    active_manifest = json.loads((source_root / "configs/v3_b_initial_v1.json").read_text())
    archived_manifest = copy.deepcopy(active_manifest)
    archived_manifest["observation_schedule"]["intermediate_cumulative_iterations"] = [16000, 24000, 32000]
    reference_root = tmp_path / "A8000/R2"
    reference_archive = reference_root / "R2"
    reference_archive.mkdir(parents=True)
    reference_manifest_path = reference_archive / "evaluation_manifest.json"
    reference_manifest_path.write_text(json.dumps(archived_manifest))
    metrics = {"tcp_compensated_l2_m": {"mean": .1}}
    reference = {"source_completed_iterations": 8000, "source_reproduction_part": "A",
                 "protocol": "force_tracking", "effective_quota": {"num_envs": 32, "episodes_per_env": 3},
                 "tracking": {"nonzero_task": metrics}, "episodes": {"physical_failure_rate": 0.},
                 "checkpoint_sha256": "original_a8000"}
    (reference_archive / "summary.json").write_text(json.dumps(reference))
    checkpoint = tmp_path / "model_012000.pt"
    checkpoint.write_bytes(b"fixture; torch loader replaced before deserialization")
    commands = observation_commands(checkpoint, tmp_path / "evaluation", 12000,
                                    source_root, {"R2": str(reference_root)})
    command = next(command for name, command in commands if name == "R2")
    arguments = Namespace(checkpoint=str(checkpoint), manifest=command[command.index("--manifest") + 1],
        output_dir=str(tmp_path / "evaluation"), protocol="force_tracking", video=True,
        run_purpose="evaluation", reproduction_part="B", seed=1001, num_envs=32,
        episodes_per_env=None, smoke=False, scenario="R2", reference_archive=str(reference_root))
    identity = {"runtime": {"device": "cpu"}, "reproduction_part": "B",
                "asset_manifest": {"resources/robots/fixture/urdf/robot.urdf": "fixture"}}
    payload = {"identity": identity, "progress": {"completed_iterations": 12000},
               "config_snapshot": {"environment": {"asset": {"file": "/fixture/robot.urdf"}}}}
    summary = {"tracking": {"nonzero_task": metrics}, "episodes": {"physical_failure_rate": 0.}}
    with patch.object(evaluate_v3, "get_v3_args", return_value=arguments), \
            patch.object(evaluate_v3.torch, "load", return_value=payload), \
            patch.object(evaluate_v3, "build_run_identity", return_value=identity), \
            patch.object(evaluate_v3, "validate_policy_evaluation_checkpoint", return_value={}), \
            patch.object(evaluate_v3, "evaluate_scenario", return_value=(summary, {})), \
            patch.object(evaluate_v3, "seal_evaluation_archive", side_effect=RuntimeError("reached archive after real reference guard")):
        with pytest.raises(RuntimeError, match="reached archive after real reference guard"):
            evaluate_v3.main()
    # Threshold changes must still block; only the observation schedule is irrelevant.
    archived_manifest["thresholds"]["force_nonzero_axis_mae_n"] += 1.
    reference_manifest_path.write_text(json.dumps(archived_manifest))
    with pytest.raises(ValueError, match="protocol"):
        observation_commands(checkpoint, tmp_path / "different_threshold", 12000,
                             source_root, {"R2": str(reference_root)})

"""Check A-start gate decisions from compact evidence fixtures."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict

from legged_gym.b2_gym_learn.ppo_cse_pf.v3_evaluation import REQUIRED_A_THRESHOLD_METRICS
from legged_gym.b2_gym_learn.ppo_cse_pf.v3_preflight import (
    assess_a_start_gate,
    validate_stage_artifacts,
)
from legged_gym.b2_gym_learn.ppo_cse_pf.v3_run_identity import build_run_identity


def make_manifest(thresholds: Any = "complete") -> Dict[str, Any]:
    """Build a minimal frozen manifest with either complete or absent limits.

    Args:
        thresholds: ``"complete"`` selects the full required metric set; None
            represents the ungraded pre-threshold state.

    Returns:
        Valid scenario manifest accepted by the production validator.
    """
    limits = ({name: {"max": 1.0} for name in REQUIRED_A_THRESHOLD_METRICS}
              if thresholds == "complete" else thresholds)
    return {"schema_version": 2, "reproduction_part": "A", "checkpoint_iteration": 8000,
            "num_envs": 1, "seed": 1, "push_seed": 2, "episodes_per_env": 1,
            "action_mode": "actor_mean",
            "scenario_protocol": "v3_a_three_scenarios_v1",
            "thresholds": ({scenario: {name: spec for name, spec in limits.items()
                                        if scenario == "S2" or not name.startswith("push_recovery_")}
                            for scenario in ("S0", "S1", "S2")} if limits is not None else None),
            "threshold_version": "a11-v1" if limits is not None else None}


def make_evidence(identity: Dict[str, Any]) -> Dict[str, Any]:
    """Build reports matching the approved A11 test and capacity contract.

    Args:
        identity: Compact source and runtime identity reused by active checkout.

    Returns:
        Schema-1 evidence object with all required stages marked complete.
    """
    formal_digest = "formal-config"
    return {"schema_version": 1, "identity": identity,
            "formal_config_sha256": formal_digest,
            "stages": {
                "unit_tests": {"passed": True, "identity": copy.deepcopy(identity)},
                "gym_smoke": {"passed": True, "num_envs": 32,
                              "zero_ee_command_force": True, "zero_ee_external_force": True,
                              "finite_core": True,
                              "reset_isolated": True, "identity": copy.deepcopy(identity)},
                "training_resume": {"passed": True, "num_envs": 512,
                                     "iterations": 10, "resumed_iterations": 1,
                                     "completed_iterations": 11, "counters_consistent": True,
                                     "checkpoint_identity_matches": True, "identity": copy.deepcopy(identity)},
                "capacity": {"passed": True, "num_envs": 4096,
                             "formal_config_sha256": formal_digest, "rollout_update": True,
                             "identity": copy.deepcopy(identity)},
            }}


def test_a_start_gate_accepts_complete_matching_evidence() -> None:
    """Allow the start only when all reports and thresholds match the checkout.

    Raises:
        AssertionError: Complete compatible test evidence is incorrectly blocked.
    """
    identity = {"official_commit": "base", "source_manifest": {"runner.py": "abc"},
                "asset_manifest": {"robot.urdf": "def"}, "schema_sha256": "schema",
                "runtime": {"device": "cuda:0"}, "learning_config_sha256": "formal-config"}

    report = assess_a_start_gate(make_evidence(identity), identity, make_manifest(), formal_num_envs=4096)

    assert report["ready"]
    assert report["blockers"] == []


def test_a_start_gate_blocks_stale_identity_missing_capacity_and_unapproved_limits() -> None:
    """Block stale runs and incomplete A10/A11 prerequisites with useful reasons.

    Raises:
        AssertionError: Missing prerequisites do not produce actionable blockers.
    """
    active_identity = {"official_commit": "base", "source_manifest": {"runner.py": "new"},
                       "asset_manifest": {"robot.urdf": "def"}, "schema_sha256": "schema",
                       "runtime": {"device": "cuda:0"}, "learning_config_sha256": "formal-config"}
    evidence = make_evidence({**active_identity, "source_manifest": {"runner.py": "old"}})
    del evidence["stages"]["capacity"]

    report = assess_a_start_gate(evidence, active_identity, make_manifest(None), formal_num_envs=4096)

    assert not report["ready"]
    assert any("source_manifest" in blocker for blocker in report["blockers"])
    assert any("capacity" in blocker for blocker in report["blockers"])
    assert any("阈值尚未确认" in blocker for blocker in report["blockers"])


def test_matching_summary_cannot_hide_a_stale_individual_gym_report() -> None:
    """Reject an old Gym report even when the top-level summary was refreshed.

    Raises:
        AssertionError: An old source report is accepted as current Gym evidence.
    """
    identity = {"official_commit": "base", "source_manifest": {"runner.py": "new"},
                "asset_manifest": {"robot.urdf": "def"}, "schema_sha256": "schema",
                "runtime": {"device": "cuda:0"}, "learning_config_sha256": "formal-config"}
    evidence = make_evidence(identity)
    for stage in evidence["stages"].values():
        stage["identity"] = copy.deepcopy(identity)
    evidence["stages"]["gym_smoke"]["identity"]["source_manifest"] = {"runner.py": "old"}

    report = assess_a_start_gate(evidence, identity, make_manifest(), formal_num_envs=4096)

    assert not report["ready"]
    assert any("gym_smoke" in blocker and "source_manifest" in blocker
               for blocker in report["blockers"])


def test_capacity_must_match_the_explicit_formal_environment_count() -> None:
    """Check the selected formal scale rather than assuming 4096 environments.

    Raises:
        AssertionError: Matching capacity is rejected or a smaller run is accepted.
    """
    identity = {"official_commit": "base", "source_manifest": {"runner.py": "new"},
                "asset_manifest": {"robot.urdf": "def"}, "schema_sha256": "schema",
                "runtime": {"device": "cuda:0"}, "learning_config_sha256": "formal-config"}
    evidence = make_evidence(identity)
    evidence["stages"]["capacity"]["num_envs"] = 512

    matching = assess_a_start_gate(evidence, identity, make_manifest(), formal_num_envs=512)
    wrong_scale = assess_a_start_gate(evidence, identity, make_manifest(), formal_num_envs=4096)

    assert matching["ready"]
    assert not wrong_scale["ready"]
    assert any("capacity" in blocker and "num_envs" in blocker
               for blocker in wrong_scale["blockers"])


def test_nonzero_injected_end_effector_force_blocks_a_training() -> None:
    """Reject A evidence when commanded force is zero but injected force is not.

    Raises:
        AssertionError: A nonzero external end-effector force is allowed in A.
    """
    identity = {"official_commit": "base", "source_manifest": {"runner.py": "new"},
                "asset_manifest": {"robot.urdf": "def"}, "schema_sha256": "schema",
                "runtime": {"device": "cuda:0"}, "learning_config_sha256": "formal-config"}
    evidence = make_evidence(identity)
    evidence["stages"]["gym_smoke"]["zero_ee_external_force"] = False

    report = assess_a_start_gate(evidence, identity, make_manifest(), formal_num_envs=4096)

    assert not report["ready"]
    assert any("zero_ee_external_force" in blocker for blocker in report["blockers"])


def test_stage_artifacts_are_required_hash_checked_and_bound_to_stage_fields(tmp_path: Path) -> None:
    """Require each formal stage to reference a matching hash-pinned report.

    Args:
        tmp_path: Temporary directory used for synthetic stage-report files.

    Raises:
        AssertionError: Missing, modified, or mismatched reports are accepted.
    """
    evidence = {"stages": {name: {"passed": True, "identity": {"source": "test"}}
                           for name in ("unit_tests", "gym_smoke", "training_resume", "capacity")}}

    assert len(validate_stage_artifacts(evidence)) == 4

    for name, stage in evidence["stages"].items():
        report_path = tmp_path / f"{name}.json"
        report_bytes = json.dumps({"stage": name, "result": stage}).encode("utf-8")
        report_path.write_bytes(report_bytes)
        stage["artifact"] = {"path": str(report_path),
                             "sha256": hashlib.sha256(report_bytes).hexdigest()}

    assert validate_stage_artifacts(evidence) == []
    evidence["stages"]["gym_smoke"]["passed"] = False
    assert any("内容与证据字段不一致" in blocker
               for blocker in validate_stage_artifacts(evidence))
    evidence["stages"]["gym_smoke"]["passed"] = True
    Path(evidence["stages"]["gym_smoke"]["artifact"]["path"]).write_text("{}", encoding="utf-8")
    assert any("SHA-256不匹配" in blocker
               for blocker in validate_stage_artifacts(evidence))


def test_cli_reports_missing_prerequisites_without_gym_or_overwriting_evidence(tmp_path: Path) -> None:
    """Exercise the read-only CLI against the actual checkout with incomplete inputs.

    Args:
        tmp_path: Temporary directory holding only synthetic gate inputs and output.

    Raises:
        AssertionError: The CLI passes missing prerequisites or replaces an old report.
    """
    source_root = Path(__file__).resolve().parents[1]
    snapshot = {
        "environment": {
            "asset": {"file": str(source_root / "resources/robots/v3_a1_desktop/urdf/a1arm_rgm3_desktop_fixed_gripper.urdf")},
            "env": {"num_envs": 4096},
            "run": {"reproduction_part": "A", "run_purpose": "formal"},
        },
        "training": {"seed": 0, "runner": {"max_iterations": 8000}},
    }
    # This is test input, not a claim about the real remote runtime or results.
    runtime = {"device": "synthetic-test-runtime"}
    identity = build_run_identity(source_root, snapshot, runtime)
    evidence = make_evidence(identity)
    evidence["formal_config_sha256"] = identity["learning_config_sha256"]
    evidence["stages"] = {}
    inputs = {"evidence": evidence, "config_snapshot": snapshot,
              "runtime": runtime, "manifest": make_manifest(None)}
    command = [sys.executable, "-m", "legged_gym.scripts.preflight_v3"]
    for name, value in inputs.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        command.extend(["--" + name, str(path)])
    output_path = tmp_path / "gate.json"
    command.extend(["--output", str(output_path)])

    result = subprocess.run(command, cwd=source_root, text=True, capture_output=True, check=False)
    original_report = output_path.read_bytes()
    report = json.loads(original_report)
    repeated = subprocess.run(command, cwd=source_root, text=True, capture_output=True, check=False)

    assert result.returncode == 1, result.stderr
    assert not report["ready"]
    assert any("阈值尚未确认" in blocker for blocker in report["blockers"])
    assert any("capacity" in blocker for blocker in report["blockers"])
    assert repeated.returncode != 0
    assert output_path.read_bytes() == original_report


def test_start_gate_checks_all_three_scenario_threshold_tables() -> None:
    """Block formal training when one scenario's frozen thresholds are absent.

    Raises:
        AssertionError: A partial three-scenario manifest passes the A start gate.
    """
    identity = {"official_commit": "base", "source_manifest": {"runner.py": "abc"},
                "asset_manifest": {"robot.urdf": "def"}, "schema_sha256": "schema",
                "runtime": {"device": "cuda:0"}, "learning_config_sha256": "formal-config"}
    manifest = make_manifest()
    del manifest["thresholds"]["S2"]
    report = assess_a_start_gate(make_evidence(identity), identity, manifest, formal_num_envs=4096)
    assert not report["ready"]
    assert any("manifest" in blocker and "S2" in blocker for blocker in report["blockers"])

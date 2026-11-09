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


def test_b_gate_accepts_hashed_engineering_evidence_without_demanding_b_performance(tmp_path: Path) -> None:
    """Accept full-scale engineering results even when A cannot perform B tasks.

    Args:
        tmp_path: New machine-report files with explicit synthetic identities.

    Raises:
        AssertionError: Complete B evidence is rejected, or edited reports,
            incomplete capacity, wrong parents or unfrozen force ranges pass.
    """
    from legged_gym.b2_gym_learn.ppo_cse_pf.v3_preflight import assess_b_start_gate

    source_root = Path(__file__).resolve().parents[1]
    force_manifest = json.loads((source_root / "configs/v3_b_initial_v1.json").read_text())
    contact_manifest = json.loads((source_root / "configs/v3_b_contact_v1.json").read_text())
    zero_manifest = json.loads((source_root / "configs/v3_a_initial_v1.json").read_text())
    identity = {"official_commit": "base", "source_manifest": {"v3.py": "bytes"},
                "asset_manifest": {"v6.urdf": "v6"}, "schema_sha256": "schema",
                "runtime": {"device": "cuda:0"}, "learning_config_sha256": "formal-config",
                "reproduction_part": "B", "seed": 1}
    parent = {"sha256": "a" * 64, "reproduction_part": "A", "completed_iterations": 8000,
              "num_envs": 4096, "seed": 1, "asset_manifest": identity["asset_manifest"],
              "schema_sha256": "schema"}
    stages = {
        "unit_tests": {"passed": True},
        "force_smoke": {"passed": True, "two_runtimes": True, "com_force_verified": True,
                        "reset_isolated": True},
        "contact_smoke": {"passed": True, "two_runtimes": True, "measurement_verified": True,
                          "reset_isolated": True},
        "zero_force_regression": {"passed": True, "scenarios": ["S0", "S1", "S2"],
                                  "num_envs": 32, "episodes_per_env": 3, "policy_updates": 0},
        "training_resume": {"passed": True, "num_envs": 4096, "iterations": 10,
                            "resumed_iterations": 1, "completed_iterations": 8011,
                            "counters_consistent": True, "full_state_restored": True,
                            "nonzero_force_learned": True},
        "capacity": {"passed": True, "num_envs": 4096, "rollout_update": True,
                     "stress_verified": True, "physics_warnings_absent": True},
        "evaluation_baseline": {"passed": True, "force_scenarios": 14,
                                "force_episodes": 1344, "contact_episodes": 96,
                                "zero_control_episodes": 32, "policy_updates": 0,
                                "cost_recorded": True},
    }
    for name, stage in stages.items():
        stage["identity"] = copy.deepcopy(identity)
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"stage": name, "result": stage}))
        stage["artifact"] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    evidence = {"schema_version": 1, "identity": identity, "formal_config_sha256": "formal-config",
                "parent_sha256": parent["sha256"], "stages": stages,
                "manifest_sha256": {name: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
                    for name, value in (("force", force_manifest), ("contact", contact_manifest), ("zero_force", zero_manifest))}}
    ready = assess_b_start_gate(evidence, identity, parent, force_manifest, contact_manifest, zero_manifest)
    assert ready["ready"], ready["blockers"]
    # Ordinary failures and poor A force tracking are deliberately not gate fields.
    broken = copy.deepcopy(evidence)
    del broken["stages"]["capacity"]
    assert not assess_b_start_gate(broken, identity, parent, force_manifest, contact_manifest, zero_manifest)["ready"]
    wrong_parent = {**parent, "sha256": "b" * 64}
    assert not assess_b_start_gate(evidence, identity, wrong_parent, force_manifest, contact_manifest, zero_manifest)["ready"]
    unfrozen = {**force_manifest, "range_frozen": False}
    assert not assess_b_start_gate(evidence, identity, parent, unfrozen, contact_manifest, zero_manifest)["ready"]
    Path(stages["capacity"]["artifact"]["path"]).write_text("{}")
    assert not assess_b_start_gate(evidence, identity, parent, force_manifest, contact_manifest, zero_manifest)["ready"]


def test_b_cli_reports_missing_real_evidence_before_simulation(tmp_path: Path) -> None:
    """Expose B's missing full-scale stages through the actual read-only CLI.

    Args:
        tmp_path: Synthetic formal configuration and declared parent metadata.

    Raises:
        AssertionError: The B CLI crashes instead of reporting missing evidence.
    """
    source_root = Path(__file__).resolve().parents[1]
    snapshot = {"environment": {"asset": {"file": str(source_root / "resources/robots/v3_a1_desktop/urdf/a1arm_rgm3_desktop_fixed_gripper.urdf")},
                               "run": {"reproduction_part": "B", "run_purpose": "formal"},
                               "env": {"num_envs": 4096}},
                "training": {"seed": 1, "runner": {"max_iterations": 32000}}}
    runtime = {"device": "synthetic"}
    identity = build_run_identity(source_root, snapshot, runtime)
    inputs = {"config_snapshot": snapshot, "runtime": runtime,
              "parent_metadata": {"sha256": "a" * 64},
              "evidence": {"schema_version": 1, "identity": identity, "stages": {}}}
    command = [sys.executable, "-m", "legged_gym.scripts.preflight_v3"]
    for name, value in inputs.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value))
        command.extend(["--" + name, str(path)])
    for flag, filename in (("manifest", "v3_b_initial_v1.json"), ("contact_manifest", "v3_b_contact_v1.json"),
                           ("zero_force_manifest", "v3_a_initial_v1.json")):
        command.extend(["--" + flag, str(source_root / "configs" / filename)])
    output = tmp_path / "gate.json"
    command.extend(["--output", str(output)])
    result = subprocess.run(command, cwd=source_root, capture_output=True, text=True, check=False)
    assert result.returncode == 1, result.stderr
    report = json.loads(output.read_text())
    assert not report["ready"]
    assert any("training_resume" in blocker for blocker in report["blockers"])

"""Check A-evaluation evidence calculations without loading Isaac Gym."""

import json
import os
from pathlib import Path
from typing import Any, Dict
from types import SimpleNamespace

import pytest

from legged_gym.b2_gym_learn.ppo_cse_pf.v3_evaluation import (
    REQUIRED_A_THRESHOLD_METRICS,
    seal_evaluation_archive,
    summarize_evaluation,
    validate_a8000_checkpoint,
    validate_evaluation_manifest,
)
from legged_gym.b2_gym_learn.ppo_cse_pf.v3_run_identity import file_sha256


def make_manifest(thresholds: Any = None, threshold_version: Any = None) -> Dict[str, Any]:
    """Build one compact frozen evaluation manifest for metric tests.

    Args:
        thresholds: Optional named maximum thresholds.
        threshold_version: Version label for approved thresholds or None.

    Returns:
        Minimal valid manifest for one environment and two episodes.
    """
    return {"schema_version": 2, "reproduction_part": "A", "checkpoint_iteration": 8000,
            "num_envs": 1, "seed": 9281, "push_seed": 1312, "episodes_per_env": 2,
            "action_mode": "actor_mean",
            "scenario_protocol": "v3_a_three_scenarios_v1",
            "thresholds": ({scenario: {name: spec for name, spec in thresholds.items()
                                        if scenario == "S2" or not name.startswith("push_recovery_")}
                            for scenario in ("S0", "S1", "S2")} if thresholds is not None else None),
            "threshold_version": threshold_version}


def make_trace_row(step: int, tcp_error: float, estimate: float = 0.0) -> Dict[str, Any]:
    """Create one physical transition and scaled estimator label pair.

    Args:
        step: Evaluation policy-step index.
        tcp_error: Known TCP L2 error in meters.
        estimate: Uniform error in the first six scaled estimate dimensions.

    Returns:
        JSON-compatible trace row consumed by the real evaluator.
    """
    return {"scenario_id": "seed9281_env0_episode0", "evaluation_step": step,
            "base_push_applied": False,
            "physical_errors": {"tcp_geometric_l1_m": tcp_error,
                                "tcp_geometric_l2_m": tcp_error,
                                "base_planar_speed_m_s": 0.2,
                                "base_yaw_rate_rad_s": 0.1},
            "state_estimate_labels_scaled": [0.0] * 12,
            "state_estimates_scaled": [estimate] * 12}


def test_a8000_validation_rejects_short_or_stale_checkpoints() -> None:
    """Require exact progress and source identity before policy evaluation.

    Returns:
        None.

    Raises:
        AssertionError: A non-A8000 checkpoint is accepted for evaluation.
    """
    identity = {"reproduction_part": "A"}
    payload = {"format_version": 1, "checkpoint_kind": "training_boundary",
               "model_state_dict": {}, "main_optimizer_state_dict": {"state": {1: {}}, "param_groups": []},
               "auxiliary_optimizer_state_dict": {"state": {1: {}}, "param_groups": []},
               "main_parameter_names": [], "auxiliary_parameter_names": [],
               "learning_rate_state": {"main": 0.001, "auxiliary": 0.001},
               "scheduler_states": {},
               "rng_state": {"python": [], "numpy": [], "torch": [], "cuda": {}},
               "identity": identity,
               "config_snapshot": {}, "lineage": {},
               "progress": {"completed_iterations": 8000, "training_policy_steps": 192000,
                            "training_transitions": 384000, "rollout_step": 0},
               "environment_training_state": {"reproduction_part": "A", "num_envs": 2, "completed_iterations": 8000,
                                               "training_policy_steps": 192000,
                                               "training_transitions": 384000}}
    validate_a8000_checkpoint(payload, identity)
    payload["progress"]["completed_iterations"] = 7999
    with pytest.raises(ValueError, match="A8000"):
        validate_a8000_checkpoint(payload, identity)


def test_metrics_recompute_distribution_estimation_and_failure_denominator() -> None:
    """Preserve both successful and failed episode outcomes in the denominator.

    Returns:
        None.

    Raises:
        AssertionError: Metrics or the episode denominator differ from known values.
    """
    thresholds = {name: {"max": 1.0} for name in REQUIRED_A_THRESHOLD_METRICS}
    thresholds["physical_failure_rate"]["max"] = 0.5
    manifest = make_manifest(thresholds, "a11-v1")
    validate_evaluation_manifest(manifest)
    episodes = [{"env_id": 0, "end_type": "timeout", "errors": {}, "reward_return": 2.0, "reward_rate": 1.0},
                {"env_id": 0, "end_type": "physical_failure", "errors": {}, "reward_return": 1.0, "reward_rate": 0.5}]
    trace_rows = [make_trace_row(step, 0.1, 0.1) for step in range(50)]
    trace_rows[0]["base_push_applied"] = True
    result = summarize_evaluation(
        trace_rows, episodes, manifest, scenario_name="S2"
    )
    assert result["metrics"]["tcp_geometric_l2_m"] == {
        "mean": pytest.approx(0.1), "p95": pytest.approx(0.1), "sample_count": 50}
    assert result["episodes"] == {"completed": 2, "expected": 2, "completion_rate": 1.0,
                                  "physical_failures": 1, "timeouts": 1,
                                  "failure_denominator": 2, "physical_failure_rate": 0.5}
    assert result["metrics"]["state_estimation/base_vx_m_s_mae"]["mean"] == pytest.approx(0.05)
    assert result["metrics"]["episode_reward_return"]["mean"] == 1.5
    assert result["metrics"]["episode_reward_rate"]["mean"] == 0.75
    assert result["assessment"] == "pass"


def test_missing_thresholds_leave_report_ungraded() -> None:
    """Never emit a pass result before the versioned A11 thresholds exist.

    Returns:
        None.

    Raises:
        AssertionError: An unthresholded report is labeled as passing.
    """
    result = summarize_evaluation(
        [make_trace_row(0, 0.1)], [{"env_id": 0, "end_type": "timeout", "errors": {},
                                  "reward_return": 1.0, "reward_rate": 1.0}], make_manifest(), scenario_name="S0"
    )
    assert result["assessment"] == "ungraded_or_incomplete"
    assert result["threshold_results"] is None


def test_incomplete_a_threshold_table_is_rejected() -> None:
    """Require every A metric category before a manifest can claim a threshold version.

    Returns:
        None.

    Raises:
        AssertionError: A partial A threshold table is accepted.
    """
    with pytest.raises(ValueError, match="omit required A metrics"):
        validate_evaluation_manifest(make_manifest({"tcp_geometric_l2_m": {"max": 0.1}}, "a11-v1"))


def test_archive_copies_checkpoint_hashes_artifacts_and_removes_write_bits(tmp_path: Path) -> None:
    """Seal exact inputs and verify all archived artifacts against their hashes.

    Args:
        tmp_path: Pytest-managed directory for generated evidence files.

    Returns:
        None.

    Raises:
        AssertionError: Archive bytes, hashes, plots, or write permissions are wrong.
    """
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"a8000-checkpoint")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(make_manifest()), encoding="utf-8")
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(json.dumps(make_trace_row(0, 0.1)) + "\n", encoding="utf-8")
    episodes_path = tmp_path / "episodes.jsonl"
    episodes_path.write_text(json.dumps({"end_type": "timeout", "reward_return": 1.0,
                                        "reward_rate": 1.0, "errors": {
        "tcp_geometric_l2_m": {"mean": 0.1}, "base_planar_speed_m_s": {"mean": 0.2}}}) + "\n",
        encoding="utf-8")
    iterations_path = tmp_path / "iterations.jsonl"
    iterations_path.write_text(json.dumps({"completed_iterations": 8000,
                                          "training_transitions": 100,
                                          "episode/reward_rate": 1.0,
                                          "episode/errors/tcp_geometric_l2_m/mean": 0.2}) + "\n",
                                encoding="utf-8")
    archive = tmp_path / "sealed"
    inventory = seal_evaluation_archive(
        archive, checkpoint, manifest_path, trace_path, episodes_path,
        {"assessment": "ungraded_or_incomplete"}, {"schema_sha256": "schema"},
        iterations_path, {"evaluation": True},
    )
    assert inventory["checkpoint_sha256"] == file_sha256(archive / "checkpoint_a8000.pt")
    assert inventory["artifacts"]["policy_trace.jsonl"] == file_sha256(archive / "policy_trace.jsonl")
    assert (archive / "plots" / "evaluation_episode_errors.png").exists()
    assert (archive / "plots" / "state_estimator_error_distribution.png").exists()
    assert not (archive / "summary.json").stat().st_mode & 0o222
    assert not archive.stat().st_mode & 0o222
    for path in archive.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)
    archive.chmod(0o755)



def test_frozen_three_scenario_manifest_requires_each_applicable_threshold() -> None:
    """Accept the approved table and reject a missing scenario or push limit.

    Raises:
        AssertionError: Incomplete evaluation requirements are accepted.
    """
    path = Path(__file__).resolve().parents[1] / "configs/v3_a_initial_v1.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_evaluation_manifest(manifest)
    del manifest["thresholds"]["S2"]["push_recovery_tcp_l2_m"]
    with pytest.raises(ValueError, match="S2.*required A metrics"):
        validate_evaluation_manifest(manifest)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    del manifest["thresholds"]["S0"]
    with pytest.raises(ValueError, match="S0.*S1.*S2"):
        validate_evaluation_manifest(manifest)


@pytest.mark.parametrize("scenario_name,failures,expected", [
    ("S0", 4, "pass"), ("S0", 5, "ungraded_or_incomplete"),
    ("S1", 4, "pass"), ("S1", 5, "ungraded_or_incomplete"),
    ("S2", 9, "pass"), ("S2", 10, "ungraded_or_incomplete"),
])
def test_scenario_specific_failure_limits_and_push_windows(
    scenario_name: str, failures: int, expected: str,
) -> None:
    """Judge each scenario using all 96 episodes and its own applicable limits.

    Args:
        scenario_name: Approved S0, S1, or S2 protocol.
        failures: Number of physical failures among the 96 completed episodes.
        expected: Expected assessment from the independently approved table.

    Raises:
        AssertionError: Failures are filtered or an inapplicable push gate is used.
    """
    path = Path(__file__).resolve().parents[1] / "configs/v3_a_initial_v1.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    episodes = [{"env_id": index // 3, "end_type": "physical_failure" if index < failures else "timeout",
                 "reward_return": 1.0, "reward_rate": 1.0} for index in range(96)]
    rows = [make_trace_row(step, 0.04) for step in range(50)]
    for row in rows:
        row["physical_errors"]["base_planar_speed_m_s"] = 0.03
    rows[0]["base_push_applied"] = scenario_name == "S2"
    result = summarize_evaluation(rows, episodes, manifest, scenario_name=scenario_name)
    assert result["assessment"] == expected
    assert result["episodes"]["failure_denominator"] == 96
    if scenario_name == "S2":
        truncated = summarize_evaluation(rows[:49], episodes, manifest, scenario_name=scenario_name)
        assert truncated["assessment"] == "ungraded_or_incomplete"
        assert truncated["push_recovery_windows"][0]["window_policy_steps"] == 49


def test_scenario_configuration_preserves_free_base_and_unrelated_settings() -> None:
    """Change only base command ranges and the push switch on isolated configs.

    Raises:
        AssertionError: A scenario changes the task or leaves S0 moving.
    """
    from legged_gym.b2_gym_learn.ppo_cse_pf.v3_evaluation import configure_evaluation_scenario

    for scenario_name in ("S0", "S1", "S2"):
        config = SimpleNamespace(
            commands=SimpleNamespace(ranges=SimpleNamespace(
                lin_vel_x=[-0.6, 0.6], lin_vel_y=[-0.4, 0.4], ang_vel_yaw=[-0.6, 0.6])),
            domain_rand=SimpleNamespace(push_robots=True, push_interval_s=8.0),
            asset=SimpleNamespace(fix_base_link=False),
        )
        configure_evaluation_scenario(config, scenario_name)
        assert config.domain_rand.push_robots == (scenario_name == "S2")
        assert config.commands.ranges.lin_vel_x == ([0.0, 0.0] if scenario_name == "S0" else [-0.6, 0.6])
        assert config.commands.ranges.lin_vel_y == ([0.0, 0.0] if scenario_name == "S0" else [-0.4, 0.4])
        assert config.commands.ranges.ang_vel_yaw == ([0.0, 0.0] if scenario_name == "S0" else [-0.6, 0.6])
        assert config.asset.fix_base_link is False
        assert config.domain_rand.push_interval_s == 8.0


def test_overall_assessment_requires_all_three_scenarios() -> None:
    """Reject a missing or failed group even when the other reports pass.

    Raises:
        AssertionError: A partial or mixed assessment is reported as passing.
    """
    from legged_gym.b2_gym_learn.ppo_cse_pf.v3_evaluation import summarize_scenario_assessments

    reports = {name: {"assessment": "pass"} for name in ("S0", "S1", "S2")}
    assert summarize_scenario_assessments(reports)["assessment"] == "pass"
    reports["S1"]["assessment"] = "ungraded_or_incomplete"
    assert summarize_scenario_assessments(reports)["assessment"] == "ungraded_or_incomplete"
    del reports["S1"]
    assert summarize_scenario_assessments(reports)["assessment"] == "ungraded_or_incomplete"


def test_completed_episode_total_cannot_replace_per_environment_quota() -> None:
    """Require three episodes from every environment instead of only a total of 96.

    Raises:
        AssertionError: Repeating one environment's episodes gives complete evidence.
    """
    path = Path(__file__).resolve().parents[1] / "configs/v3_a_initial_v1.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    episodes = [{"env_id": 0, "end_type": "timeout", "reward_return": 1.0,
                 "reward_rate": 1.0} for _ in range(96)]
    result = summarize_evaluation([make_trace_row(0, 0.04)], episodes, manifest, scenario_name="S0")
    assert result["evidence_complete"] is False

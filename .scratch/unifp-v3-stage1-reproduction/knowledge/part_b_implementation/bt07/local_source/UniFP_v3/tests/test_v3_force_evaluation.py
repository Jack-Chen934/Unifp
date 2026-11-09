"""Check scripted force evaluation through the agreed event and statistics seams."""

import math
import copy
from pathlib import Path
from unittest.mock import patch

import isaacgym  # noqa: F401
import torch
import pytest

from legged_gym.envs.v3.v3_force_evaluation import V3ForceEvaluation
from test_v3_force_transitions import make_step_environment


def test_random_trace_separates_disabled_command_and_active_external_phases() -> None:
    """Keep independent random streams and omit fictitious fixed-axis groups.

    Raises:
        AssertionError: A disabled event gains a phase, command and external
            phases are conflated, or random XYZ vectors acquire fixed-axis IDs.
    """
    from legged_gym.envs.v3.legged_robot_v3 import V3PosForce

    environment = make_step_environment()
    environment.__class__ = V3ForceEvaluation
    environment.force_scenario = {"mode": "R"}
    environment.video_recorder = None
    environment.evaluation_direction_indices = torch.zeros(2, dtype=torch.long)
    for stream in ("command", "external"):
        state = environment.force_state[stream]
        state["start_tick"].zero_()
        state["ramp_ticks"].fill_(50.)
        state["enabled"].fill_(stream == "external")
    # At 1.6 s the external stream is in the last half of its 1 s plateau.
    parent_rows = [{"env_id": 0, "policy_input_time_s": 1.6}]
    with patch.object(V3PosForce, "_build_policy_trace", return_value=parent_rows):
        row = environment._build_policy_trace(False, torch.zeros(2, dtype=torch.bool),
                                              torch.zeros(2, dtype=torch.bool))[0]
    assert row["command_force_phase"] == "zero"
    assert row["external_force_phase"] == "hold"
    assert row["external_force_hold_tail"]
    assert row["force_phase"] == "hold"
    assert row["direction_index"] is None


@pytest.mark.parametrize("mode", ["M", "E", "C"])
def test_scripted_force_balances_directions_and_rotates_only_published_commands(mode: str) -> None:
    """Use fixed world +X and yaw 90 degrees at a full 30 N plateau.

    Args:
        mode: Matched forces, external-only forces or command-only forces.

    Raises:
        AssertionError: A scripted command rotates the external force, skips an
            endpoint, or assigns directions based on the policy's performance.
    """
    environment = make_step_environment()
    environment.__class__ = V3ForceEvaluation
    environment.force_scenario = {"mode": mode, "amplitude_fraction": .5, "moving": False, "push": False}
    environment.evaluation_episodes_per_env = 3
    environment._initialize_force_events()
    environment._reset_force_events(torch.arange(2))
    assert environment.evaluation_direction_indices.tolist() == [0, 3]
    environment.episode_length_buf.fill_(150)  # t=3 s, first plateau starts.
    environment.rpy[:, 2] = math.pi / 2
    environment._advance_force_events(torch.arange(2))
    expected_external = torch.zeros(2, 3) if mode == "C" else torch.tensor([[30., 0., 0.], [0., -30., 0.]])
    expected_command = {"M": [[0., 30., 0.], [30., 0., 0.]], "E": [[0., 0., 0.], [0., 0., 0.]],
                        "C": [[0., -30., 0.], [-30., 0., 0.]]}[mode]
    torch.testing.assert_close(environment.pending_ee_force_world, expected_external)
    torch.testing.assert_close(environment.commands[:, 9:12], torch.tensor(expected_command), atol=1e-5, rtol=1e-5)
    environment.episode_length_buf.fill_(300)  # t=6 s, exactly zero.
    environment._advance_force_events(torch.arange(2))
    assert torch.count_nonzero(environment.pending_ee_force_world) == 0
    assert torch.count_nonzero(environment.commands[:, 9:12]) == 0


def test_force_summary_keeps_nonzero_denominator_and_zero_predictor_reference() -> None:
    """Keep ten zero samples from diluting two 20 N estimation errors.

    Raises:
        AssertionError: Zero samples dilute force errors, undefined directions
            become zero degrees, or physical failures disappear from the quota.
    """
    from legged_gym.b2_gym_learn.ppo_cse_pf.v3_evaluation import summarize_force_evaluation

    manifest = {"num_envs": 1, "episodes_per_env": 2, "range_frozen": False,
                "thresholds": {"static": {"M_tcp_l2_m": .08, "other_tcp_l2_m": .10,
                                            "base_planar_speed_m_s": .10, "base_yaw_rate_rad_s": .15,
                                            "physical_failure_rate": .05},
                               "force_nonzero_axis_mae_n": 10., "force_nonzero_zero_reference_ratio": .5,
                               "force_zero_axis_mae_n": 3., "state_mae": [.1, .1, .05, .05, .1, .15]},
                "threshold_version": "v3-b-initial-v1"}
    rows = []
    for step in range(12):
        force_n = 20. if step >= 10 else 0.
        rows.append({"env_id": 0, "episode_index": step // 6, "evaluation_step": step,
                     "scenario_id": f"fixture_{step // 6}", "policy_input_time_s": step * .02,
                     "force_phase": "hold", "estimator_force_phase": "hold",
                     "force_hold_tail": step == 10, "estimator_force_hold_tail": step == 11,
                     "direction_index": 0, "base_push_applied": False,
                     "end_effector_force_command_yaw_n": [0., 0., 0.],
                     "external_end_effector_force_yaw_n": [force_n, 0., 0.],
                     "state_estimates_scaled": [0.] * 9 + [.02, -.01, .03],
                     "state_estimate_labels_scaled": [0.] * 6 + [.01 * force_n, 0., 0.] + [0.] * 3,
                     "physical_errors": {"tcp_geometric_l1_m": .05, "tcp_geometric_l2_m": .05,
                                         "tcp_compensated_l1_m": .06, "tcp_compensated_l2_m": .06,
                                         "base_planar_speed_m_s": .02, "base_yaw_rate_rad_s": .03},
                     "net_compensation_force_world_n": [force_n, 0., 0.], "torque_saturation_fraction": 0.,
                     "tcp_world_m": [0., 0., 0.], "compensated_target_world_m": [.06, 0., 0.],
                     "target_world_m": [.05, 0., 0.]})
    episodes = [{"env_id": 0, "end_type": "timeout", "errors": {}},
                {"env_id": 0, "end_type": "physical_failure", "errors": {}}]
    result = summarize_force_evaluation(rows, episodes, manifest, scenario={"mode": "E", "moving": False, "push": False})
    assert result["force_estimation"]["nonzero"]["sample_count"] == 2
    assert result["force_estimation"]["nonzero"]["axis_mae_n"] == pytest.approx([20., 0., 0.])
    assert result["force_estimation"]["nonzero"]["zero_reference_l2_n"] == pytest.approx(20.)
    assert result["force_estimation"]["nonzero"]["l2_mean_n"] == pytest.approx(20.)
    assert result["force_estimation"]["zero"]["sample_count"] == 10
    assert result["phase_metrics"]["hold_tail"]["task_samples"] == 1
    assert result["phase_metrics"]["hold_tail"]["estimator_samples"] == 1
    assert result["phase_metrics"]["hold_tail"]["force_axis_mae_n"] == pytest.approx([20., 0., 0.])
    assert result["direction"]["undefined_prediction_count"] == 2
    assert result["direction"]["mean_degrees"] is None
    assert result["episodes"]["failure_denominator"] == 2
    assert result["episodes"]["physical_failure_rate"] == .5
    assert result["assessment"] != "pass"
    assert result["base_force_zero_input"]["sample_count"] == 12
    assert result["base_force_zero_input"]["axis_bias_n"] == pytest.approx([2., -1., 3.])
    assert result["base_force_zero_input"]["axis_mae_n"] == pytest.approx([2., 1., 3.])
    for row in rows:
        row.update(command_force_phase="zero", command_force_hold_tail=False,
                   external_force_phase="hold", external_force_hold_tail=row["force_hold_tail"],
                   direction_index=None)
    random_result = summarize_force_evaluation(rows, episodes, manifest,
        scenario={"mode": "R", "moving": False, "push": False})
    assert random_result["directions"] == {}
    streams = random_result["stream_phase_metrics"]
    assert streams["command"]["zero"]["task_samples"] == 12
    assert streams["external"]["hold"]["task_samples"] == 12
    assert streams["external"]["hold_tail"]["estimator_samples"] == 1


def test_delay_distinguishes_one_tick_lag_early_crossing_and_truncation() -> None:
    """Check a hand-worked 20 ms delay while the second event is unobserved.

    Raises:
        AssertionError: Missing/early crossings become a numeric zero delay.
    """
    from legged_gym.b2_gym_learn.ppo_cse_pf.v3_evaluation import _force_delay_diagnostics

    rows = [{"time_s": time_s, "truth_n": [truth, 0., 0.], "prediction_n": [prediction, 0., 0.],
             "yaw_rad": 0., "direction_index": 0}
            for time_s, truth, prediction in ((0., 0., 0.), (1., 0., 0.), (2., 10., 0.), (2.02, 11., 10.), (6., 0., 0.))]
    manifest, scenario = {"max_force_n": 20.}, {"amplitude_fraction": 1.}
    result = _force_delay_diagnostics({"episode0": rows}, manifest, scenario)
    assert result[0]["status"] == "valid" and result[0]["delay_s"] == pytest.approx(.02)
    assert result[1]["status"] == "truncated" and result[1]["delay_s"] is None
    rows[1]["prediction_n"][0] = 12.
    result = _force_delay_diagnostics({"episode0": rows}, manifest, scenario)
    assert result[0]["status"] == "early" and result[0]["delay_s"] is None


def test_explicit_evaluation_preserves_a8000_requirement_and_rejects_asset_changes(tmp_path: Path) -> None:
    """Allow recorded B evaluator source changes while keeping A progress strict.

    Args:
        tmp_path: Independent real-network fixture and tiny source/asset identities.

    Raises:
        AssertionError: A short checkpoint or altered asset passes inference validation.
    """
    from legged_gym.b2_gym_learn.ppo_cse_pf.v3_evaluation import validate_policy_evaluation_checkpoint
    from test_v3_runner import make_runner
    from test_v3_run_identity import make_transition_identity_fixture

    parent_identity, _, config, _ = make_transition_identity_fixture(tmp_path / "source")
    runner = make_runner(tmp_path / "parent", identity=parent_identity, config_snapshot=config)
    runner.learn(1)
    payload = torch.load(runner.log_dir / "checkpoints/model_000001.pt", map_location="cpu", weights_only=False)
    with pytest.raises(ValueError, match="A8000"):
        validate_policy_evaluation_checkpoint(payload, parent_identity)
    for state in (payload["progress"], payload["environment_training_state"]):
        state.update(completed_iterations=8000, training_policy_steps=192000, training_transitions=192000)
    active = copy.deepcopy(parent_identity)
    active["source_manifest"]["legged_gym/scripts/evaluate_v3.py"] = "new_evaluation_bytes"
    report = validate_policy_evaluation_checkpoint(payload, active)
    assert report["allowed_source_changes"] == ["legged_gym/scripts/evaluate_v3.py"]
    active["asset_manifest"]["changed"] = "invalid_asset"
    with pytest.raises(ValueError, match="asset/schema"):
        validate_policy_evaluation_checkpoint(payload, active)

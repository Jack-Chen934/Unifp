"""Read-only A-stage evaluation metrics and immutable evidence packaging."""

import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from .v3_run_identity import A_TO_B_SOURCE_PATHS, file_sha256, validate_force_snapshot


PHYSICAL_TRACE_METRICS = (
    "tcp_geometric_l1_m", "tcp_geometric_l2_m",
    "base_planar_speed_m_s", "base_yaw_rate_rad_s",
)
EVALUATION_SCENARIOS = ("S0", "S1", "S2")
PUSH_RECOVERY_WINDOW_POLICY_STEPS = 50
PUSH_THRESHOLD_METRICS = {"push_recovery_tcp_l2_m", "push_recovery_base_speed_error_m_s"}
REQUIRED_A_THRESHOLD_METRICS = set(PHYSICAL_TRACE_METRICS) | {
    "physical_failure_rate", "push_recovery_tcp_l2_m", "push_recovery_base_speed_error_m_s",
    *(f"state_estimation/{name}_mae" for name in (
        "base_vx_m_s", "base_vy_m_s", "base_vz_m_s", "tcp_radius_m",
        "tcp_elevation_rad", "tcp_azimuth_rad",
    )),
}


class EvaluationVideo:
    """Record existing Gym camera frames before reset, without a second control loop."""

    def __init__(self, environment: Any, path: Path) -> None:
        """Create a fixed env0 camera and raw RGB ffmpeg encoder at 25 fps.

        Args:
            environment: Existing simulator with camera graphics already enabled.
            path: New MP4 output path; ffmpeg refuses overwriting another video.

        Raises:
            RuntimeError: Gym cannot allocate the camera.
            OSError: ffmpeg is unavailable or cannot start.
        """
        from isaacgym import gymapi

        self.environment = environment
        properties = gymapi.CameraProperties()
        properties.width, properties.height = 640, 480
        self.camera = environment.gym.create_camera_sensor(environment.envs[0], properties)
        if self.camera < 0:
            raise RuntimeError("Evaluation camera allocation failed")
        self.encoder = subprocess.Popen([
            "ffmpeg", "-v", "error", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", "640x480", "-r", "25", "-i", "-", "-an", "-c:v", "libx264",
            "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p", str(path),
        ], stdin=subprocess.PIPE)

    def capture(self) -> None:
        """Encode one refreshed physical frame before episode reset replaces the pose.

        Raises:
            OSError: The encoder pipe fails.

        Note:
            Refreshes graphics only; never advances physics, commands or counters.
        """
        from isaacgym import gymapi

        environment = self.environment
        position = environment.root_states[0, :3].detach().cpu().tolist()
        environment.gym.fetch_results(environment.sim, True)
        environment.gym.set_camera_location(self.camera, environment.envs[0],
            gymapi.Vec3(position[0] + 2.3, position[1] - 2.3, position[2] + 1.6),
            gymapi.Vec3(position[0], position[1], position[2] + .3))
        environment.gym.step_graphics(environment.sim)
        environment.gym.render_all_camera_sensors(environment.sim)
        pixels = environment.gym.get_camera_image(environment.sim, environment.envs[0], self.camera, gymapi.IMAGE_COLOR)
        rgb = np.asarray(pixels).reshape(480, 640, 4)[:, :, :3].copy()
        self.encoder.stdin.write(rgb.tobytes())

    def close(self) -> None:
        """Finish MP4 publication and report encoder failure.

        Raises:
            OSError: ffmpeg exits unsuccessfully.
        """
        self.encoder.stdin.close()
        if self.encoder.wait(timeout=30) != 0:
            raise OSError("Evaluation video encoding failed")


def summarize_contact_evaluation(
    trace_rows: Iterable[Dict[str, Any]], episode_rows: List[Dict[str, Any]],
    manifest: Dict[str, Any], *, zero_control: bool,
) -> Dict[str, Any]:
    """Assess fixed-wall episodes with all failures retained in planned quotas.

    Args:
        trace_rows: Executed intervals with eight-point mean wall force in world N,
            contact validity/status and the action-input episode time in seconds.
        episode_rows: Completed episodes including physical failures, keyed by
            logical env_id and episode_index; never filtered by measurement quality.
        manifest: Geometry, force amplitudes and the frozen B10 threshold table.
        zero_control: Assess the independent zero-command geometric control.

    Returns:
        Per-episode windows, amplitude-wise planned denominators and missing-data
        diagnostics. An unfrozen range or incomplete quota never receives pass.

    Raises:
        ValueError: No transition was collected.
    """
    trajectories: Dict[tuple, List[Dict[str, Any]]] = {}
    decoder_errors, decoder_invalid = [], 0
    for row in trace_rows:
        trajectories.setdefault((row["env_id"], row["episode_index"]), []).append(row)
        if row.get("estimator_contact_valid", False):
            decoder_errors.append(np.asarray(row["predicted_contact_force_world_n"])
                                  - row["estimator_contact_truth_world_n"])
        else:
            decoder_invalid += 1
    if not trajectories:
        raise ValueError("Contact evaluation produced no trace")
    endings = {(row["env_id"], row["episode_index"]): row["end_type"] for row in episode_rows}
    thresholds = manifest["thresholds"]
    episodes = []
    # Time windows refer to executed action intervals, not reset successor clocks.
    for key, rows in trajectories.items():
        times = np.asarray([row["policy_input_time_s"] for row in rows])
        valid = np.asarray([row["contact_measurement_valid"] for row in rows], dtype=bool)
        clean = np.asarray([row["contact_status"] == "clean_expected_contact" for row in rows])
        ambiguous = np.asarray([row["contact_status"] == "ambiguous_contact" for row in rows])
        forces = np.asarray([row["wall_force_mean_world_n"] for row in rows], dtype=np.float64)
        # SimParams.dt is float32; recover nominal microsecond boundaries without
        # shifting the simulator clock or losing the 19.00 s policy interval.
        window_times = np.round(times, decimals=6)
        steady = (window_times >= 9.) & (window_times < 11.)
        release = (window_times >= 19.) & (window_times < 20.)
        residual = (window_times >= 13.) & (window_times < 15.)
        steady_valid, release_valid = steady & valid, release & valid
        peak_n = rows[0]["contact_peak_n"]
        normal_error = float(np.abs(forces[steady_valid, 0] - peak_n).mean()) if steady_valid.any() else None
        tangential = float(np.linalg.norm(forces[steady_valid, 1:], axis=1).mean()) if steady_valid.any() else None
        release_force = float(np.abs(forces[release_valid, 0]).mean()) if release_valid.any() else None
        clean_fraction = float((steady & valid & clean & (forces[:, 0] > thresholds["contact_normal_min_n"])).sum() / max(steady.sum(), 1))
        limit_n = max(thresholds["force_error_absolute_n"], thresholds["force_error_peak_fraction"] * peak_n)
        complete_windows = int(steady_valid.sum()) == 100 and int(release_valid.sum()) == 50
        success = bool(endings.get(key) == "timeout" and complete_windows and valid.all() and not ambiguous.any()
                       and clean_fraction >= thresholds["clean_contact_fraction_min"]
                       and normal_error is not None and normal_error <= limit_n
                       and tangential is not None and tangential <= limit_n
                       and release_force is not None and release_force <= thresholds["release_abs_force_max_n"])
        episodes.append({"env_id": key[0], "episode_index": key[1], "peak_n": peak_n,
                         "end_type": endings.get(key, "incomplete"), "success": success if not zero_control else None,
                         "trace_samples": len(rows), "missing_measurement_samples": int((~valid).sum()),
                         "ambiguous_samples": int(ambiguous.sum()), "steady_samples": int(steady.sum()),
                         "steady_valid_samples": int(steady_valid.sum()), "clean_contact_fraction": clean_fraction,
                         "normal_error_mean_n": normal_error, "tangential_mean_n": tangential,
                         "release_valid_samples": int(release_valid.sum()), "release_abs_normal_mean_n": release_force,
                         "zero_command_residual_normal_mean_n": float(np.abs(forces[residual & valid, 0]).mean())
                             if (residual & valid).any() else None})
    quota = manifest["zero_control_episodes_per_env"] if zero_control else manifest["episodes_per_env"]
    amplitudes = {}
    for name, peak_n in (("zero", 0.),) if zero_control else (("half", manifest["contact_peak_n"] / 2.), ("full", manifest["contact_peak_n"])):
        group = [episode for episode in episodes if episode["peak_n"] == peak_n]
        expected = sum(zero_control or (env_id * quota + episode_index) % 2 == (name == "full")
                       for env_id in range(manifest["num_envs"]) for episode_index in range(quota))
        failures = sum(episode["end_type"] == "physical_failure" for episode in group)
        successes = sum(episode["success"] is True for episode in group)
        rate = successes / expected if expected else None
        failure_rate = failures / expected if expected else None
        amplitudes[name] = {"peak_n": peak_n, "expected_episodes": expected, "observed_episodes": len(group),
                            "successes": successes, "success_rate": rate if not zero_control else None,
                            "physical_failures": failures, "physical_failure_rate": failure_rate,
                            "measurable_steady_episodes": sum(episode["normal_error_mean_n"] is not None for episode in group),
                            "all_measurable_normal_error_mean_n": float(np.mean([episode["normal_error_mean_n"] for episode in group
                                if episode["normal_error_mean_n"] is not None]))
                                if any(episode["normal_error_mean_n"] is not None for episode in group) else None,
                            "passed": bool(not zero_control and expected and len(group) == expected
                                and rate >= thresholds["success_rate_min"]
                                and failure_rate <= thresholds["physical_failure_rate_max"])}
    complete = all(sum(episode["env_id"] == env_id and episode["end_type"] != "incomplete" for episode in episodes) == quota
                   for env_id in range(manifest["num_envs"]))
    return {"protocol": "fixed_wall_contact", "zero_control": zero_control, "episodes": episodes,
            "amplitudes": amplitudes, "evidence_complete": complete, "policy_updates": 0,
            "contact_decoder_generalization": {
                "valid_samples": len(decoder_errors), "invalid_boundary_samples": decoder_invalid,
                "axis_mae_world_n": np.abs(decoder_errors).mean(axis=0).tolist() if decoder_errors else None,
                "l2_mean_n": float(np.linalg.norm(decoder_errors, axis=1).mean()) if decoder_errors else None,
                "说明": "动作前估力对齐上一个有效物理子步反力，仅作泛化诊断，不是人工注入力监督"},
            "threshold_version": manifest["threshold_version"],
            "assessment": "pass" if not zero_control and complete and manifest["range_frozen"]
                and all(group["passed"] for group in amplitudes.values()) else "control_only" if zero_control else "ungraded_or_incomplete"}


def summarize_force_evaluation(
    trace_rows: Iterable[Dict[str, Any]], episode_rows: List[Dict[str, Any]],
    manifest: Dict[str, Any], *, scenario: Dict[str, Any],
) -> Dict[str, Any]:
    """Summarize one B force condition without diluting nonzero samples with zeros.

    Args:
        trace_rows: Executed transitions with action-before estimator/label pairs.
        episode_rows: All completed episodes, including early physical failures.
        manifest: B10 quota, candidate thresholds and force-range freezing status.
        scenario: One fixed M/E/C or random R condition with moving/push flags.

    Returns:
        Separate task, force-estimation, phase, direction and failure denominators,
        with base-force zero-input bias in newtons. Missing measurements remain
        None and never count as a threshold pass.

    Raises:
        ValueError: No trace was collected or required numeric fields are absent.
    """
    # Keep reduced physical samples only, rather than entire observation histories.
    physical_names = PHYSICAL_TRACE_METRICS + ("tcp_compensated_l1_m", "tcp_compensated_l2_m")
    physical, predicted, truth, state_errors, task_active, phases, label_phases = [], [], [], [], [], [], []
    directions, saturation, residuals, hold_tail, label_hold_tail = [], [], [], [], []
    signed_geometric, signed_compensated = [], []
    base_force_zero_errors = []  # Signed yaw-frame XYZ errors in N, only for exact-zero labels.
    stream_phases = {stream: [] for stream in ("command", "external")}
    stream_hold_tail = {stream: [] for stream in ("command", "external")}
    push_windows: Dict[str, List[List[float]]] = {}
    completed_push_windows: List[List[List[float]]] = []
    delay_samples: Dict[str, List[Dict[str, Any]]] = {}
    for row in trace_rows:
        physical.append([row["physical_errors"][name] for name in physical_names])
        estimate = np.asarray(row["state_estimates_scaled"], dtype=np.float64)
        label = np.asarray(row["state_estimate_labels_scaled"], dtype=np.float64)
        predicted.append(estimate[6:9] / .01)
        truth.append(label[6:9] / .01)
        state_errors.append((estimate[:6] - label[:6]) / np.asarray((2., 2., 2., .5, 1., 1.3)))
        if np.all(label[9:12] == 0.):
            base_force_zero_errors.append((estimate[9:12] - label[9:12]) / .01)
        task_active.append(any(value != 0. for value in row["end_effector_force_command_yaw_n"]
                               + row["external_end_effector_force_yaw_n"]))
        phases.append(row["force_phase"])
        hold_tail.append(row.get("force_hold_tail", False))
        label_hold_tail.append(row.get("estimator_force_hold_tail", False))
        label_phases.append(row["estimator_force_phase"])
        if scenario["mode"] == "R":
            for stream in ("command", "external"):
                stream_phases[stream].append(row[f"{stream}_force_phase"])
                stream_hold_tail[stream].append(row[f"{stream}_force_hold_tail"])
        directions.append(row["direction_index"])
        saturation.append(row["torque_saturation_fraction"])
        residuals.append(row["net_compensation_force_world_n"])
        signed_geometric.append(np.asarray(row["tcp_world_m"]) - row["target_world_m"])
        signed_compensated.append(np.asarray(row["tcp_world_m"]) - row["compensated_target_world_m"])
        trajectory = row["scenario_id"]
        if row["base_push_applied"]:
            push_windows[trajectory] = []
        if trajectory in push_windows:
            push_windows[trajectory].append([row["physical_errors"]["tcp_compensated_l2_m"],
                                             row["physical_errors"]["base_planar_speed_m_s"]])
            if len(push_windows[trajectory]) == PUSH_RECOVERY_WINDOW_POLICY_STEPS:
                completed_push_windows.append(push_windows.pop(trajectory))
        if scenario["mode"] in {"M", "E"}:
            delay_samples.setdefault(trajectory, []).append({
                "time_s": row["policy_input_time_s"], "truth_n": truth[-1].tolist(),
                "prediction_n": predicted[-1].tolist(), "yaw_rad": row.get("prediction_yaw_rad", 0.),
                "direction_index": row["direction_index"],
            })
    if not physical:
        raise ValueError("Force evaluation produced no trace")
    physical_array = np.asarray(physical)
    task_mask = np.asarray(task_active, dtype=bool)
    predicted_array, truth_array = np.asarray(predicted), np.asarray(truth)
    force_nonzero = np.any(truth_array != 0., axis=1)
    force_errors = predicted_array - truth_array
    force_metrics = {}
    for name, mask in (("nonzero", force_nonzero), ("zero", ~force_nonzero)):
        force_metrics[name] = {
            "sample_count": int(mask.sum()),
            "axis_mae_n": np.abs(force_errors[mask]).mean(axis=0).tolist() if mask.any() else None,
            "l2_mean_n": float(np.linalg.norm(force_errors[mask], axis=1).mean()) if mask.any() else None,
            "zero_reference_l2_n": float(np.linalg.norm(truth_array[mask], axis=1).mean()) if mask.any() else None,
        }
    tracking = {}
    for name, mask in (("all", np.ones(len(physical), dtype=bool)), ("nonzero_task", task_mask), ("zero_task", ~task_mask)):
        tracking[name] = {metric: {"mean": float(physical_array[mask, index].mean()) if mask.any() else None,
                                   "p95": float(np.quantile(physical_array[mask, index], .95)) if mask.any() else None,
                                   "sample_count": int(mask.sum())}
                          for index, metric in enumerate(physical_names)}
    phase_metrics = {}
    for phase in ("zero", "rise", "hold", "fall", "hold_tail"):
        mask = np.asarray(hold_tail) if phase == "hold_tail" else np.asarray(phases) == phase
        label_mask = np.asarray(label_hold_tail) if phase == "hold_tail" else np.asarray(label_phases) == phase
        phase_metrics[phase] = {"task_samples": int(mask.sum()), "estimator_samples": int(label_mask.sum()),
                               "tcp_compensated_l2_mean_m": float(physical_array[mask, 5].mean()) if mask.any() else None,
                               "force_axis_mae_n": np.abs(force_errors[label_mask]).mean(axis=0).tolist() if label_mask.any() else None}
    # Random command/external events overlap without sharing phase boundaries.
    stream_phase_metrics = {}
    if scenario["mode"] == "R":
        for stream in ("command", "external"):
            stream_phase_metrics[stream] = {}
            for phase in ("zero", "rise", "hold", "fall", "hold_tail"):
                mask = (np.asarray(stream_hold_tail[stream], dtype=bool) if phase == "hold_tail"
                        else np.asarray(stream_phases[stream]) == phase)
                stream_phase_metrics[stream][phase] = {
                    "task_samples": int(mask.sum()),
                    "tcp_compensated_l2_mean_m": float(physical_array[mask, 5].mean()) if mask.any() else None,
                }
                if stream == "external":
                    stream_phase_metrics[stream][phase].update(
                        estimator_samples=phase_metrics[phase]["estimator_samples"],
                        force_axis_mae_n=phase_metrics[phase]["force_axis_mae_n"])
    # Undefined predictions remain in the direction denominator rather than 0 degrees.
    direction_mask = (np.asarray(label_phases) == "hold") & (np.linalg.norm(truth_array, axis=1) >= 5.)
    predicted_norm = np.linalg.norm(predicted_array, axis=1)
    defined = direction_mask & (predicted_norm >= 1.)
    cosine = np.sum(predicted_array[defined] * truth_array[defined], axis=1) / (
        predicted_norm[defined] * np.linalg.norm(truth_array[defined], axis=1))
    angles = np.degrees(np.arccos(np.clip(cosine, -1., 1.)))
    expected_episodes = manifest["num_envs"] * manifest["episodes_per_env"]
    episode_counts = [sum(row["env_id"] == env_id for row in episode_rows) for env_id in range(manifest["num_envs"])]
    physical_failures = sum(row["end_type"] == "physical_failure" for row in episode_rows)
    failure_rate = physical_failures / len(episode_rows) if episode_rows else None
    thresholds = manifest["thresholds"]
    table = thresholds["R2" if scenario["push"] else "moving" if scenario["moving"] else "static"]
    task_metric = "tcp_geometric_l2_m" if scenario["mode"] == "M" else "tcp_compensated_l2_m"
    tcp_limit = table["M_tcp_l2_m" if scenario["mode"] == "M" else "other_tcp_l2_m"]
    checks = {}
    for metric, limit in ((task_metric, tcp_limit), ("base_planar_speed_m_s", table["base_planar_speed_m_s"]),
                          ("base_yaw_rate_rad_s", table["base_yaw_rate_rad_s"])):
        value = tracking["nonzero_task"][metric]["mean"]
        checks[metric] = value is not None and value <= limit
    checks["physical_failure_rate"] = failure_rate is not None and failure_rate <= table["physical_failure_rate"]
    nonzero, zero = force_metrics["nonzero"], force_metrics["zero"]
    if scenario["mode"] != "C":
        checks["force_nonzero_axis_mae_n"] = nonzero["axis_mae_n"] is not None and max(nonzero["axis_mae_n"]) <= thresholds["force_nonzero_axis_mae_n"]
        checks["force_nonzero_zero_reference_ratio"] = nonzero["l2_mean_n"] is not None and nonzero["l2_mean_n"] <= thresholds["force_nonzero_zero_reference_ratio"] * nonzero["zero_reference_l2_n"]
    checks["force_zero_axis_mae_n"] = zero["axis_mae_n"] is not None and max(zero["axis_mae_n"]) <= thresholds["force_zero_axis_mae_n"]
    state_mae = np.abs(state_errors).mean(axis=0)
    checks["base_velocity_tcp_estimation"] = bool(np.all(state_mae <= thresholds["state_mae"]))
    push_metrics = None
    if completed_push_windows:
        push_metrics = np.asarray(completed_push_windows).mean(axis=(0, 1)).tolist()
    if scenario["push"]:
        checks["push_recovery"] = bool(push_metrics is not None
            and push_metrics[0] <= thresholds["push_compensated_tcp_l2_m"]
            and push_metrics[1] <= thresholds["push_base_speed_m_s"])
    complete = all(count == manifest["episodes_per_env"] for count in episode_counts)
    direction_reports = {}
    for direction in (() if scenario["mode"] == "R" else range(6)):
        mask = (np.asarray(directions) == direction) & task_mask
        direction_reports[str(direction)] = {"nonzero_task_samples": int(mask.sum()),
            "tcp_l2_mean_m": float(physical_array[mask, physical_names.index(task_metric)].mean()) if mask.any() else None}
    episode_metrics = {name: float(np.mean([row["errors"][name]["mean"] for row in episode_rows if name in row["errors"]]))
                       for name in physical_names if any(name in row["errors"] for row in episode_rows)}
    return {
        "tracking": tracking, "phase_metrics": phase_metrics, "force_estimation": force_metrics,
        "stream_phase_metrics": stream_phase_metrics,
        "base_force_zero_input": {
            "sample_count": len(base_force_zero_errors),
            "axis_bias_n": np.mean(base_force_zero_errors, axis=0).tolist() if base_force_zero_errors else None,
            "axis_mae_n": np.abs(base_force_zero_errors).mean(axis=0).tolist() if base_force_zero_errors else None,
        },
        "state_estimation_mae": state_mae.tolist(), "episode_equal_weight_errors": episode_metrics,
        "signed_tcp_error_m": {"geometric_xyz_mean": np.mean(signed_geometric, axis=0).tolist(),
                               "compensated_xyz_mean": np.mean(signed_compensated, axis=0).tolist(),
                               "geometric_xyz_mae": np.abs(signed_geometric).mean(axis=0).tolist(),
                               "compensated_xyz_mae": np.abs(signed_compensated).mean(axis=0).tolist()},
        "torque_saturation_fraction": float(np.mean(saturation)),
        "net_compensation_residual_mean_n": float(np.linalg.norm(residuals, axis=1).mean()),
        "direction": {"eligible_samples": int(direction_mask.sum()),
                      "undefined_prediction_count": int((direction_mask & ~defined).sum()),
                      "mean_degrees": float(angles.mean()) if angles.size else None},
        "directions": direction_reports,
        "delay": _force_delay_diagnostics(delay_samples, manifest, scenario),
        "push_recovery": {"complete_windows": len(completed_push_windows),
                          "incomplete_windows": len(push_windows), "mean_tcp_and_speed": push_metrics},
        "episodes": {"completed": len(episode_rows), "expected": expected_episodes,
                     "physical_failures": physical_failures, "failure_denominator": len(episode_rows),
                     "physical_failure_rate": failure_rate},
        "evidence_complete": complete, "threshold_results": checks,
        "threshold_version": manifest["threshold_version"], "policy_updates": 0,
        "assessment": "pass" if complete and manifest["range_frozen"] and all(checks.values()) else "ungraded_or_incomplete",
    }


def _force_delay_diagnostics(trajectories: Dict[str, List[Dict[str, Any]]],
                             manifest: Dict[str, Any], scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Compare first half-peak crossings without shifting the recorded prediction.

    Args:
        trajectories: Reduced time/yaw/force samples grouped by complete episode ID.
        manifest: Frozen force range and event times; development fixtures may omit them.
        scenario: Fixed M/E amplitude fraction or a random condition without delay grading.

    Returns:
        Per-event crossing delay (s) or an explicit early/not-crossed/truncated status.
    """
    if not trajectories or "max_force_n" not in manifest:
        return []
    half_peak_n = manifest["max_force_n"] * scenario["amplitude_fraction"] / 2.
    axes = np.asarray(((1., 0., 0.), (-1., 0., 0.), (0., 1., 0.), (0., -1., 0.), (0., 0., 1.), (0., 0., -1.)))
    reports = []
    for episode_id, rows in trajectories.items():
        time_s = np.asarray([row["time_s"] for row in rows])
        yaw = np.asarray([row["yaw_rad"] for row in rows])
        projections = []
        axis = axes[rows[0]["direction_index"]]
        for field in ("truth_n", "prediction_n"):
            force_yaw = np.asarray([row[field] for row in rows])
            world = np.stack((np.cos(yaw) * force_yaw[:, 0] - np.sin(yaw) * force_yaw[:, 1],
                              np.sin(yaw) * force_yaw[:, 0] + np.cos(yaw) * force_yaw[:, 1], force_yaw[:, 2]), axis=1)
            projections.append(world @ axis)
        for start_s in (1., 9.):
            window = (time_s >= start_s) & (time_s <= start_s + 5.)
            true_cross = np.flatnonzero(window & (projections[0] >= half_peak_n))
            predicted_cross = np.flatnonzero(window & (projections[1] >= half_peak_n))
            baseline = (time_s >= start_s - .5) & (time_s <= start_s)
            status, delay_s = "valid", None
            if np.any(projections[1][baseline] >= half_peak_n):
                status = "early"
            elif not true_cross.size or time_s[-1] < start_s + 5. - .021:
                status = "truncated"
            elif not predicted_cross.size:
                status = "not_crossed"
            else:
                delay_s = float(time_s[predicted_cross[0]] - time_s[true_cross[0]])
            reports.append({"episode_id": episode_id, "start_s": start_s, "status": status, "delay_s": delay_s})
    return reports


def validate_evaluation_manifest(manifest: Dict[str, Any]) -> None:
    """Validate the small frozen scenario contract required by A evaluation.

    Args:
        manifest: JSON object declaring checkpoint boundary, environment count,
            independent task and push seeds, completed episode count, and thresholds.

    Raises:
        ValueError: A required field, scenario, action mode, or threshold is invalid.
    """
    required = {"schema_version", "reproduction_part", "checkpoint_iteration", "num_envs",
                "seed", "push_seed", "episodes_per_env", "action_mode", "scenario_protocol", "thresholds",
                "threshold_version"}
    if set(manifest) != required:
        raise ValueError(f"Evaluation manifest fields must be exactly {sorted(required)}")
    if (manifest["schema_version"] != 2 or manifest["reproduction_part"] != "A"
            or manifest["checkpoint_iteration"] != 8000):
        raise ValueError("A evaluation requires schema 2 and the exact A8000 boundary")
    if any(type(manifest[key]) is not int or manifest[key] < minimum for key, minimum in
           (("num_envs", 1), ("seed", 0), ("push_seed", 0), ("episodes_per_env", 1))):
        raise ValueError("Evaluation environment count, seed, or episode count is invalid")
    if manifest["action_mode"] != "actor_mean":
        raise ValueError("A evaluation uses deterministic Actor means")
    if manifest["scenario_protocol"] != "v3_a_three_scenarios_v1":
        raise ValueError("Evaluation scenario protocol must match the confirmed A protocol")
    thresholds = manifest["thresholds"]
    if thresholds is not None:
        if not isinstance(thresholds, dict) or set(thresholds) != set(EVALUATION_SCENARIOS):
            raise ValueError("Approved thresholds must contain exactly S0, S1, S2")
        for scenario_name, scenario_thresholds in thresholds.items():
            required_metrics = REQUIRED_A_THRESHOLD_METRICS - (
                PUSH_THRESHOLD_METRICS if scenario_name != "S2" else set()
            )
            if not isinstance(scenario_thresholds, dict):
                raise ValueError(f"{scenario_name} thresholds must be a metric mapping")
            missing_metrics = required_metrics - set(scenario_thresholds)
            if missing_metrics:
                raise ValueError(f"{scenario_name} thresholds omit required A metrics: {sorted(missing_metrics)}")
            if set(scenario_thresholds) != required_metrics:
                raise ValueError(f"{scenario_name} thresholds contain inapplicable metrics")
            for name, specification in scenario_thresholds.items():
                if (not isinstance(specification, dict) or set(specification) != {"max"}
                        or type(specification["max"]) not in (int, float)
                        or not math.isfinite(specification["max"]) or specification["max"] < 0):
                    raise ValueError(f"Threshold for {scenario_name}/{name} must contain one numeric max")
        if not isinstance(manifest["threshold_version"], str) or not manifest["threshold_version"]:
            raise ValueError("Approved thresholds require a version identifier")
    elif manifest["threshold_version"] is not None:
        raise ValueError("An unapproved threshold set must have a null version")



def configure_evaluation_scenario(environment_config: Any, scenario_name: str) -> None:
    """Apply one evaluation protocol to a freshly resolved environment config.

    Args:
        environment_config: Independent V3 config, mutated before Gym creation.
        scenario_name: S0 sets zero base velocity commands without pushes; S1
            retains moving commands without pushes; S2 retains both baseline settings.

    Raises:
        ValueError: The scenario name is not S0, S1, or S2.

    Note:
        The base remains free. Target sampling, noise, physical randomization,
        command resampling intervals, and push magnitudes remain unchanged.
    """
    if scenario_name not in EVALUATION_SCENARIOS:
        raise ValueError(f"Unknown evaluation scenario: {scenario_name}")
    environment_config.domain_rand.push_robots = scenario_name == "S2"
    if scenario_name == "S0":
        environment_config.commands.ranges.lin_vel_x = [0.0, 0.0]
        environment_config.commands.ranges.lin_vel_y = [0.0, 0.0]
        environment_config.commands.ranges.ang_vel_yaw = [0.0, 0.0]

def validate_a8000_checkpoint(payload: Dict[str, Any], active_identity: Dict[str, Any]) -> None:
    """Require an exact A8000 training boundary and matching active provenance.

    Args:
        payload: CPU-loaded V3 training-boundary checkpoint dictionary.
        active_identity: Recomputed source, asset, schema, and config identity.

    Raises:
        ValueError: Checkpoint format, A-stage progress, or identity is incompatible.
    """
    required = {"format_version", "checkpoint_kind", "model_state_dict", "main_optimizer_state_dict",
                "auxiliary_optimizer_state_dict", "main_parameter_names", "auxiliary_parameter_names",
                "learning_rate_state", "scheduler_states", "progress", "rng_state",
                "environment_training_state", "identity", "config_snapshot", "lineage"}
    if not required.issubset(payload):
        raise ValueError("Evaluation requires a complete V3 training-boundary checkpoint")
    progress = payload["progress"]
    environment_state = payload["environment_training_state"]
    training_envs = environment_state.get("num_envs")
    optimizer_states = (payload["main_optimizer_state_dict"],
                        payload["auxiliary_optimizer_state_dict"])
    if (payload["format_version"] != 1 or payload["checkpoint_kind"] != "training_boundary"
            or payload["identity"] != active_identity or active_identity["reproduction_part"] != "A"
            or environment_state.get("reproduction_part") != "A"
            or payload["scheduler_states"] != {}
            or any(not optimizer.get("state") for optimizer in optimizer_states)
            or set(payload["learning_rate_state"]) != {"main", "auxiliary"}
            or set(payload["rng_state"]) != {"python", "numpy", "torch", "cuda"}
            or type(training_envs) is not int or training_envs < 1
            or progress != {"completed_iterations": 8000, "training_policy_steps": 192000,
                            "training_transitions": training_envs * 192000, "rollout_step": 0}
            or environment_state.get("completed_iterations") != 8000
            or environment_state.get("training_policy_steps") != 192000
            or environment_state.get("training_transitions") != training_envs * 192000):
        raise ValueError("Checkpoint must be the exact, identity-matched A8000 boundary")


def validate_policy_evaluation_checkpoint(payload: Dict[str, Any], active_identity: Dict[str, Any]) -> Dict[str, Any]:
    """Validate explicit A/B evaluation provenance without constructing optimizers.

    Args:
        payload: Complete training boundary, retaining its original source identity.
        active_identity: Current source fingerprint with the saved learning config,
            relocated asset path and source runtime used only for compatibility.

    Returns:
        Recorded allowed source differences, distinct from evaluation scene identity.

    Raises:
        ValueError: Stage, complete progress, asset/schema/config or source is incompatible.
    """
    original = payload["identity"]
    changed = sorted(name for name in original["source_manifest"].keys() | active_identity["source_manifest"].keys()
                     if original["source_manifest"].get(name) != active_identity["source_manifest"].get(name))
    allowed = A_TO_B_SOURCE_PATHS if original["reproduction_part"] == "A" else {
        "legged_gym/scripts/evaluate_v3.py", "legged_gym/scripts/evaluate_v3_contact.py",
        "legged_gym/scripts/v3_args.py", "legged_gym/envs/v3/v3_force_evaluation.py",
        "legged_gym/envs/v3/v3_contact_evaluation.py", "legged_gym/b2_gym_learn/ppo_cse_pf/v3_evaluation.py",
    }
    if set(changed) - allowed:
        raise ValueError("Evaluation source has unapproved changes: " + ", ".join(sorted(set(changed) - allowed)))
    if any(original.get(key) != active_identity.get(key) for key in original if key != "source_manifest"):
        raise ValueError("Evaluation source asset/schema/learning identity mismatch")
    if original["reproduction_part"] == "A":
        # The old exact-A8000 validator stays strict; only explicit new protocols
        # accept the documented source changes before invoking its format checks.
        validate_a8000_checkpoint(payload, original)
    else:
        progress, state = payload["progress"], payload["environment_training_state"]
        iterations = progress["completed_iterations"]
        steps = iterations * 24
        if (payload["format_version"] != 1 or payload["checkpoint_kind"] != "training_boundary"
                or original["reproduction_part"] != "B" or not 8000 <= iterations <= 40000
                or progress != {"completed_iterations": iterations, "training_policy_steps": steps,
                                "training_transitions": state["num_envs"] * steps, "rollout_step": 0}
                or state["reproduction_part"] != "B"
                or any(state.get(key) != progress[key] for key in ("completed_iterations", "training_policy_steps", "training_transitions"))
                or any(not payload[key].get("state") for key in ("main_optimizer_state_dict", "auxiliary_optimizer_state_dict"))):
            raise ValueError("Evaluation requires a complete B training boundary")
        validate_force_snapshot(state.get("force_state"), state["num_envs"])
    return {"allowed_source_changes": changed, "source_reproduction_part": original["reproduction_part"]}


def summarize_evaluation(
    trace_rows: Iterable[Dict[str, Any]], episode_rows: List[Dict[str, Any]],
    manifest: Dict[str, Any], *, scenario_name: str,
) -> Dict[str, Any]:
    """Aggregate trace-level errors while preserving every completed episode.

    Args:
        trace_rows: Policy-boundary records from evaluation, before reset mutation.
        episode_rows: Completed episode summaries emitted by the V3 environment.
        manifest: Validated schema-2 manifest containing all three threshold tables.
        scenario_name: S0, S1, or S2 identifying the applicable threshold table.

    Returns:
        JSON-compatible metrics, denominators, completeness and threshold status.

    Raises:
        ValueError: No trace data exists or an expected metric is absent.
    """
    if scenario_name not in EVALUATION_SCENARIOS:
        raise ValueError(f"Unknown evaluation scenario: {scenario_name}")
    metric_values: Dict[str, List[float]] = {name: [] for name in PHYSICAL_TRACE_METRICS}
    estimation_error: List[np.ndarray] = []
    trace_count = 0
    target_phase_elapsed_count = 0
    last_trace_by_scenario: Dict[str, Dict[str, Any]] = {}
    active_push_windows: Dict[str, Dict[str, Any]] = {}
    push_recovery = []
    for row in trace_rows:
        trace_count += 1
        target_phase_elapsed_count += bool(row.get("target_segment_complete", False))
        errors = row["physical_errors"]
        for name in PHYSICAL_TRACE_METRICS:
            metric_values[name].append(float(errors[name]))
        predicted = row.get("state_estimate_labels_scaled")
        if predicted is not None:
            actual = row["state_estimate_labels_scaled"]
            predicted_state = row["state_estimates_scaled"]
            # Decode the base velocity and TCP spherical labels to SI/radian units.
            scale = np.asarray((2.0, 2.0, 2.0, 0.5, 1.0, 1.3), dtype=np.float64)
            estimation_error.append((np.asarray(predicted_state[:6]) - np.asarray(actual[:6])) / scale)
        scenario_id = row["scenario_id"]
        if row["base_push_applied"]:
            previous = last_trace_by_scenario.get(scenario_id)
            active_push_windows[scenario_id] = {
                "evaluation_step": row["evaluation_step"],
                "pre_push_tcp_l2_m": previous["physical_errors"]["tcp_geometric_l2_m"] if previous else None,
                "post_push_tcp_l2_m": [],
                "post_push_base_speed_error_m_s": [],
            }
        push_window = active_push_windows.get(scenario_id)
        if push_window is not None and row["evaluation_step"] >= push_window["evaluation_step"]:
            push_window["post_push_tcp_l2_m"].append(errors["tcp_geometric_l2_m"])
            push_window["post_push_base_speed_error_m_s"].append(errors["base_planar_speed_m_s"])
            if len(push_window["post_push_tcp_l2_m"]) == PUSH_RECOVERY_WINDOW_POLICY_STEPS:
                push_recovery.append({
                    "scenario_id": scenario_id,
                    "evaluation_step": push_window["evaluation_step"],
                    "pre_push_tcp_l2_m": push_window["pre_push_tcp_l2_m"],
                    "post_push_tcp_l2_mean_m": float(np.mean(push_window["post_push_tcp_l2_m"])),
                    "post_push_base_speed_error_mean_m_s": float(
                        np.mean(push_window["post_push_base_speed_error_m_s"])
                    ),
                    "window_policy_steps": PUSH_RECOVERY_WINDOW_POLICY_STEPS,
                })
                del active_push_windows[scenario_id]
        last_trace_by_scenario[scenario_id] = row
    if not trace_count:
        raise ValueError("Evaluation produced no policy-boundary trace")
    for scenario_id, push_window in active_push_windows.items():
        push_recovery.append({
            "scenario_id": scenario_id,
            "evaluation_step": push_window["evaluation_step"],
            "pre_push_tcp_l2_m": push_window["pre_push_tcp_l2_m"],
            "post_push_tcp_l2_mean_m": float(np.mean(push_window["post_push_tcp_l2_m"])),
            "post_push_base_speed_error_mean_m_s": float(
                np.mean(push_window["post_push_base_speed_error_m_s"])
            ),
            "window_policy_steps": len(push_window["post_push_tcp_l2_m"]),
        })
    metrics = {}
    for name, values in metric_values.items():
        samples = np.asarray(values, dtype=np.float64)
        metrics[name] = {"mean": float(samples.mean()), "p95": float(np.quantile(samples, 0.95)),
                         "sample_count": int(samples.size)}
    for metric_name, episode_field in (("episode_reward_return", "reward_return"),
                                       ("episode_reward_rate", "reward_rate")):
        samples = np.asarray([row[episode_field] for row in episode_rows], dtype=np.float64)
        if samples.size:
            metrics[metric_name] = {"mean": float(samples.mean()),
                                    "p95": float(np.quantile(samples, 0.95)),
                                    "sample_count": int(samples.size)}
    if estimation_error:
        error_array = np.stack(estimation_error)
        mae_by_dimension = np.abs(error_array).mean(axis=0)
        rmse_by_dimension = np.sqrt(np.square(error_array).mean(axis=0))
        dimensions = ["base_vx_m_s", "base_vy_m_s", "base_vz_m_s",
                      "tcp_radius_m", "tcp_elevation_rad", "tcp_azimuth_rad"]
        metrics["state_estimation_base_velocity_tcp_mae_rmse"] = {
            "mae_by_dimension": mae_by_dimension.tolist(),
            "rmse_by_dimension": rmse_by_dimension.tolist(),
            "dimensions": dimensions,
        }
        for index, dimension in enumerate(dimensions):
            metrics[f"state_estimation/{dimension}_mae"] = {"mean": float(mae_by_dimension[index])}
            metrics[f"state_estimation/{dimension}_rmse"] = {"mean": float(rmse_by_dimension[index])}
    full_push_windows = [window for window in push_recovery
                         if window["window_policy_steps"] == PUSH_RECOVERY_WINDOW_POLICY_STEPS]
    if full_push_windows:
        metrics["push_recovery_tcp_l2_m"] = {
            "mean": float(np.mean([window["post_push_tcp_l2_mean_m"] for window in full_push_windows])),
            "window_count": len(full_push_windows),
        }
        metrics["push_recovery_base_speed_error_m_s"] = {
            "mean": float(np.mean([
                window["post_push_base_speed_error_mean_m_s"] for window in full_push_windows
            ])),
            "window_count": len(full_push_windows),
        }
    expected_episodes = manifest["num_envs"] * manifest["episodes_per_env"]
    physical_failures = sum(row["end_type"] == "physical_failure" for row in episode_rows)
    timeouts = sum(row["end_type"] == "timeout" for row in episode_rows)
    failure_rate = physical_failures / len(episode_rows) if episode_rows else None
    metrics["physical_failure_rate"] = {"mean": failure_rate}
    metrics["target_movement_phase_elapsed_rate"] = {
        "mean": target_phase_elapsed_count / trace_count,
        "sample_count": trace_count,
    }
    thresholds = manifest["thresholds"][scenario_name] if manifest["thresholds"] is not None else None
    threshold_results: Optional[Dict[str, bool]] = None
    if thresholds is not None:
        threshold_results = {
            name: metrics[name]["mean"] is not None
            and metrics[name]["mean"] <= specification["max"]
            for name, specification in thresholds.items()
            if name in metrics and "mean" in metrics[name]
        }
    episode_counts = {env_id: 0 for env_id in range(manifest["num_envs"])}
    for row in episode_rows:
        episode_counts[row["env_id"]] += 1
    complete = all(count == manifest["episodes_per_env"] for count in episode_counts.values())
    passed = bool(complete and threshold_results and len(threshold_results) == len(thresholds)
                  and all(threshold_results.values()))
    return {
        "scenario_name": scenario_name,
        "checkpoint_iteration": manifest["checkpoint_iteration"],
        "trace_policy_steps": trace_count,
        "episodes": {"completed": len(episode_rows), "expected": expected_episodes,
                     "completion_rate": len(episode_rows) / expected_episodes,
                     "physical_failures": physical_failures, "timeouts": timeouts,
                     "failure_denominator": len(episode_rows),
                     "physical_failure_rate": failure_rate},
        "metrics": metrics,
        "threshold_version": manifest["threshold_version"],
        "threshold_results": threshold_results,
        "push_recovery_windows": push_recovery,
        "paper_comparison_scope": {
            "fig_3b": "V3逐维TCP球坐标与底盘速度估计误差分布；单位见维度标签",
            "fig_3b_limit": "资产、目标分布和环境协议不同；不重建论文未公开的原始数据点",
            "fig_3c_fig_3d": "受力仿真与实机结果不属于本A阶段评估",
        },
        "evidence_complete": complete,
        "assessment": "pass" if passed else "ungraded_or_incomplete",
    }



def summarize_scenario_assessments(reports: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Combine independent scenario decisions without pooling their metrics.

    Args:
        reports: Scenario-keyed summaries from the same checkpoint and manifest.

    Returns:
        Overall decision and the separate scenario decisions; any missing or
        non-passing scenario prevents an overall pass.
    """
    assessments = {name: reports[name]["assessment"] for name in EVALUATION_SCENARIOS if name in reports}
    passed = len(assessments) == len(EVALUATION_SCENARIOS) and all(
        assessment == "pass" for assessment in assessments.values()
    )
    return {"assessment": "pass" if passed else "ungraded_or_incomplete", "scenarios": assessments}

def make_evaluation_plots(
    archive_dir: Path, training_iterations_path: Path, episode_rows: List[Dict[str, Any]],
    trace_path: Path,
) -> None:
    """Render raw training trends and the final evaluation episode distribution.

    Args:
        archive_dir: Writable archive directory containing the ``plots`` folder.
        training_iterations_path: Training run's iterations.jsonl file.
        episode_rows: Completed evaluation episode records, including failures.
        trace_path: Evaluation policy-boundary trace with scaled estimator outputs.

    Raises:
        OSError: A required training log or output plot cannot be read or written.
        ValueError: A training log row is malformed.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with training_iterations_path.open(encoding="utf-8") as source:
        iterations = [json.loads(line) for line in source if line.strip()]
    if not iterations:
        raise ValueError("Training iterations log is empty")
    iteration = [row["completed_iterations"] for row in iterations]
    transitions = [row["training_transitions"] for row in iterations]
    plot_specs = (("episode/reward_rate", "Episode reward rate", "reward_rate"),
                  ("episode/errors/tcp_geometric_l2_m/mean", "TCP L2 error (m)", "tcp_l2_m"))
    for field, title, filename in plot_specs:
        values = [row.get(field, np.nan) for row in iterations]
        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.plot(iteration, values, linewidth=1.0)
        axis.set(xlabel="Completed training iterations", ylabel=title, title=title)
        figure.tight_layout()
        figure.savefig(archive_dir / "plots" / f"{filename}_by_iteration.png", dpi=150)
        axis.set_xlabel("Cumulative environment transitions")
        axis.set_xlim(transitions[0], max(transitions[-1], transitions[0] + 1))
        axis.lines[0].set_xdata(transitions)
        figure.tight_layout()
        figure.savefig(archive_dir / "plots" / f"{filename}_by_transitions.png", dpi=150)
        plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    if episode_rows:
        axes[0].boxplot([row["errors"]["tcp_geometric_l2_m"]["mean"] for row in episode_rows])
        axes[1].boxplot([row["errors"]["base_planar_speed_m_s"]["mean"] for row in episode_rows])
    else:
        axes[0].text(0.5, 0.5, "No completed episodes", ha="center", va="center")
        axes[1].text(0.5, 0.5, "No completed episodes", ha="center", va="center")
    axes[0].set_title("TCP L2 error (m)")
    axes[1].set_title("Base planar speed error (m/s)")
    axes[0].set_xticks(())
    axes[1].set_xticks(())
    figure.tight_layout()
    figure.savefig(archive_dir / "plots" / "evaluation_episode_errors.png", dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    if episode_rows:
        axes[0].boxplot([row["reward_return"] for row in episode_rows])
        axes[1].boxplot([row["reward_rate"] for row in episode_rows])
    else:
        axes[0].text(0.5, 0.5, "No completed episodes", ha="center", va="center")
        axes[1].text(0.5, 0.5, "No completed episodes", ha="center", va="center")
    axes[0].set_title("Episode reward return")
    axes[1].set_title("Episode reward rate (per second)")
    axes[0].set_xticks(())
    axes[1].set_xticks(())
    figure.tight_layout()
    figure.savefig(archive_dir / "plots" / "evaluation_episode_rewards.png", dpi=150)
    plt.close(figure)

    dimensions = ("Base vx (m/s)", "Base vy (m/s)", "Base vz (m/s)",
                  "TCP radius (m)", "TCP elevation (rad)", "TCP azimuth (rad)")
    scales = np.asarray((2.0, 2.0, 2.0, 0.5, 1.0, 1.3), dtype=np.float64)
    estimation_errors = [[] for _ in dimensions]
    with trace_path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            predicted = np.asarray(row["state_estimates_scaled"][:6], dtype=np.float64)
            actual = np.asarray(row["state_estimate_labels_scaled"][:6], dtype=np.float64)
            absolute_error = np.abs(predicted - actual) / scales
            for index, value in enumerate(absolute_error):
                estimation_errors[index].append(float(value))
    figure, axes = plt.subplots(2, 3, figsize=(12, 7))
    for index, axis in enumerate(axes.flat):
        if estimation_errors[index]:
            axis.boxplot(estimation_errors[index])
        else:
            axis.text(0.5, 0.5, "No trace", ha="center", va="center")
        axis.set_title(dimensions[index])
        axis.set_xticks(())
        axis.set_ylabel("Absolute error")
    figure.suptitle("V3 A8000 state-estimation error distribution")
    figure.tight_layout()
    figure.savefig(archive_dir / "plots" / "state_estimator_error_distribution.png", dpi=150)
    plt.close(figure)


def seal_evaluation_archive(
    archive_dir: Path, checkpoint_path: Path, manifest_path: Path, trace_path: Path,
    episodes_path: Path, summary: Dict[str, Any], identity: Dict[str, Any],
    training_iterations_path: Path, evaluation_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Copy evaluation evidence and the exact model into a read-only archive.

    Args:
        archive_dir: New destination directory; existing destinations are rejected.
        checkpoint_path: Exact complete-boundary A8000 checkpoint used for inference.
        manifest_path: Frozen scenario/threshold JSON file.
        trace_path: Temporary complete policy-boundary JSONL file.
        episodes_path: Temporary completed-episode JSONL file.
        summary: Recomputed metrics and failure denominator.
        identity: Validated source, asset, schema and learning-configuration identity.
        training_iterations_path: Source training iterations.jsonl for trend plots.
        evaluation_config: Effective evaluation environment and learner settings.

    Returns:
        Archive inventory with SHA256 values and checkpoint identity.

    Raises:
        FileExistsError: The destination already exists.
        OSError: Copying, writing, hashing, or permission changes fail.
    """
    archive_dir.mkdir(parents=True, exist_ok=False)
    (archive_dir / "plots").mkdir()
    checkpoint_filename = "checkpoint_a8000.pt" if summary.get("protocol", "a8000") == "a8000" else "checkpoint.pt"
    shutil.copyfile(checkpoint_path, archive_dir / checkpoint_filename)
    shutil.copyfile(manifest_path, archive_dir / "evaluation_manifest.json")
    shutil.copyfile(trace_path, archive_dir / "policy_trace.jsonl")
    shutil.copyfile(episodes_path, archive_dir / "episodes.jsonl")
    for video_path in trace_path.parent.glob("representative_*.mp4"):
        shutil.copyfile(video_path, archive_dir / video_path.name)
    shutil.copyfile(training_iterations_path, archive_dir / "training_iterations.jsonl")
    (archive_dir / "effective_evaluation_config.json").write_text(
        json.dumps(evaluation_config, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (archive_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8"
    )
    make_evaluation_plots(
        archive_dir, archive_dir / "training_iterations.jsonl", _read_json_lines(episodes_path),
        archive_dir / "policy_trace.jsonl",
    )
    if summary.get("protocol") == "force_tracking":
        make_force_evaluation_plots(archive_dir, archive_dir / "policy_trace.jsonl")
    if summary.get("protocol") == "fixed_wall_contact":
        make_contact_evaluation_plots(archive_dir, archive_dir / "policy_trace.jsonl")
    inventory = {path.relative_to(archive_dir).as_posix(): file_sha256(path)
                 for path in sorted(archive_dir.rglob("*")) if path.is_file()}
    if inventory[checkpoint_filename] != file_sha256(checkpoint_path):
        raise OSError("Archived checkpoint SHA256 differs from the evaluated checkpoint")
    archive_manifest = {
        "checkpoint_sha256": inventory[checkpoint_filename],
        "evaluation_manifest_sha256": inventory["evaluation_manifest.json"],
        "artifacts": inventory,
        "identity": identity,
        "git_commit": _git_commit(Path(__file__).resolve().parents[3]),
        "git_status": _git_status(Path(__file__).resolve().parents[3]),
        "working_diff_sha256": _git_diff_sha256(Path(__file__).resolve().parents[3]),
    }
    (archive_dir / "archive_manifest.json").write_text(
        json.dumps(archive_manifest, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8"
    )
    for path in archive_dir.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    archive_dir.chmod(0o555)
    return archive_manifest


def make_contact_evaluation_plots(archive_dir: Path, trace_path: Path) -> None:
    """Plot fixed env0 representatives and per-episode contact error diagnostics.

    Args:
        archive_dir: Writable candidate archive with an existing plots directory.
        trace_path: Complete contact JSONL including failed/missing intervals.

    Raises:
        OSError: Trace or image output cannot be read/written.

    Note:
        Selects episode0/1 by identifier, never by quality. All forces are in world
        coordinates, with positive pressure and negative robot reaction distinguished.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    for episode_index in (0, 1):
        rows = [row for row in _read_trace_stream(trace_path) if row["env_id"] == 0 and row["episode_index"] == episode_index]
        if not rows:
            continue
        time_s = [row["policy_input_time_s"] for row in rows]
        figure, axes = plt.subplots(3, 1, figsize=(9, 9))
        forces = np.asarray([row["wall_force_mean_world_n"] for row in rows])
        commands = np.asarray([row["contact_command_normal_n"] for row in rows])
        estimates = np.asarray([row["predicted_contact_force_world_n"] for row in rows])
        previous_truth = np.asarray([row["estimator_contact_truth_world_n"]
            if row["estimator_contact_valid"] else [np.nan] * 3 for row in rows])
        axes[0].plot(time_s, commands, label="按压力命令")
        axes[0].plot(time_s, forces[:, 0], label="墙受力区间均值")
        axes[0].plot(time_s, -estimates[:, 0], label="动作前估力（上一边界）")
        axes[0].plot(time_s, -previous_truth[:, 0], linestyle=":", label="上一边界机器人反力投影")
        axes[0].set(ylabel="法向力（N）")
        axes[0].legend()
        axes[1].plot(time_s, forces[:, 1], label="世界系 Y")
        axes[1].plot(time_s, forces[:, 2], label="世界系 Z")
        axes[1].set(ylabel="切向力（N）")
        axes[1].legend()
        tcp = np.asarray([row["tcp_world_m"] for row in rows])
        target = np.asarray([row["target_world_m"] for row in rows])
        for axis in range(3):
            axes[2].plot(time_s, tcp[:, axis] - target[:, axis], label="XYZ"[axis])
        axes[2].set(xlabel="时间（s）", ylabel="TCP 世界系误差（m）")
        axes[2].legend()
        figure.tight_layout()
        figure.savefig(archive_dir / "plots" / f"contact_env0_episode{episode_index}.png", dpi=150)
        plt.close(figure)


def make_force_evaluation_plots(archive_dir: Path, trace_path: Path) -> None:
    """Plot fixed representative forces and all-sample TCP error distributions.

    Args:
        archive_dir: Writable archive with an existing plots directory.
        trace_path: Sealed-candidate JSONL trace; retains every failed trajectory.

    Raises:
        OSError: Trace or image output cannot be accessed.

    Note:
        Selects env0/episode0 by fixed ID, never by tracking quality.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    representative, geometric_errors, compensated_errors = [], [], []
    for row in _read_trace_stream(trace_path):
        geometric_errors.append(np.asarray(row["tcp_world_m"]) - row["target_world_m"])
        compensated_errors.append(np.asarray(row["tcp_world_m"]) - row["compensated_target_world_m"])
        if row["env_id"] == 0 and row["episode_index"] == 0:
            representative.append(row)
    figure, axes = plt.subplots(2, 3, figsize=(12, 7))
    time_s = [row["policy_input_time_s"] for row in representative]
    if representative:
        predicted = np.asarray([row["state_estimates_scaled"][6:9] for row in representative]) / .01
        truth = np.asarray([row["state_estimate_labels_scaled"][6:9] for row in representative]) / .01
        for index in range(3):
            axes[0, index].plot(time_s, truth[:, index], label="F true")
            axes[0, index].plot(time_s, predicted[:, index], label="F estimated")
            axes[0, index].set(xlabel="t (s)", ylabel=f"F{'XYZ'[index]} yaw (N)")
            axes[0, index].legend()
        axes[1, 0].plot(time_s, [row["physical_errors"]["tcp_geometric_l2_m"] for row in representative], label="geometric")
        axes[1, 0].plot(time_s, [row["physical_errors"]["tcp_compensated_l2_m"] for row in representative], label="compensated")
        axes[1, 0].set(xlabel="t (s)", ylabel="TCP L2 (m)")
        axes[1, 0].legend()
    for values, axis, label in ((geometric_errors, axes[1, 1], "geometric"), (compensated_errors, axes[1, 2], "compensated")):
        errors = np.asarray(values)
        for index in range(3):
            samples = np.sort(np.abs(errors[:, index]))
            axis.plot(samples, np.arange(1, samples.size + 1) / samples.size, label="XYZ"[index])
        axis.set(xlabel=f"TCP {label} abs error (m)", ylabel="ECDF")
        axis.legend()
    figure.tight_layout()
    figure.savefig(archive_dir / "plots/force_tracking.png", dpi=150)
    plt.close(figure)


def _read_trace_stream(path: Path) -> Iterable[Dict[str, Any]]:
    """Stream physical trace rows without retaining full JSON dictionaries.

    Args:
        path: Existing UTF-8 JSONL policy trace.

    Yields:
        One decoded physical transition per nonempty line.

    Raises:
        OSError: The source is unreadable.
        json.JSONDecodeError: A nonempty line is invalid JSON.
    """
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def _read_json_lines(path: Path) -> List[Dict[str, Any]]:
    """Read JSON records from one newline-delimited evidence file.

    Args:
        path: Existing UTF-8 JSONL file.

    Returns:
        Parsed JSON object for each non-empty line.

    Raises:
        OSError: The file cannot be read.
        json.JSONDecodeError: A line contains invalid JSON.
    """
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _git_commit(repository: Path) -> str:
    """Read the development checkout commit for archive provenance.

    Args:
        repository: Git checkout root.

    Returns:
        Full HEAD commit identifier.

    Raises:
        subprocess.CalledProcessError: The path is not a valid Git checkout.
    """
    import subprocess
    return subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()


def _git_status(repository: Path) -> str:
    """Read tracked and untracked checkout paths for source-diff provenance.

    Args:
        repository: Git checkout root.

    Returns:
        Porcelain status output.

    Raises:
        subprocess.CalledProcessError: Git status cannot be read.
    """
    import subprocess
    return subprocess.check_output(["git", "-C", str(repository), "status", "--short"], text=True)


def _git_diff_sha256(repository: Path) -> str:
    """Hash tracked working-tree changes without copying source files.

    Args:
        repository: Git checkout root.

    Returns:
        SHA256 digest of the binary Git diff.

    Raises:
        subprocess.CalledProcessError: Git diff cannot be produced.
    """
    import hashlib
    import subprocess
    diff = subprocess.check_output(["git", "-C", str(repository), "diff", "--binary"])
    return hashlib.sha256(diff).hexdigest()

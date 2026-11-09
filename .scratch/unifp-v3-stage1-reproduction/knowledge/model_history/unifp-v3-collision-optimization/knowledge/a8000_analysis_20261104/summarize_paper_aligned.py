"""Summarize sealed traces in explicit physical coordinates without rescoring."""

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


def summarize_scenario(trace_path: Path) -> Dict[str, Any]:
    """Measure per-axis tracking, Cartesian estimation, and actuator targets.

    Args:
        trace_path: Sealed policy-boundary JSONL; position units are meters.

    Returns:
        JSON-ready statistics over all rows, including physical failure samples.
        Motion-target diagnostics alone exclude each episode's first second.

    Raises:
        OSError: The source trace cannot be read.
        KeyError: A required field is absent from the frozen trace schema.

    Note:
        Does not change archives, thresholds, simulation state, or policies.
        Cartesian estimator errors use paired pre-action labels and predictions
        in the task's yaw-aligned frame, not a guessed paper reference frame.
    """
    tracking_errors = []
    estimation_errors = []
    movement_phases = []
    position_targets = []
    drive_targets = []
    first_step_errors = []
    for line in trace_path.open(encoding="utf-8"):
        row = json.loads(line)
        error = np.asarray(row["tcp_world_m"]) - np.asarray(row["target_world_m"])
        tracking_errors.append(error)
        movement_phases.append(not row["target_segment_complete"])
        if row["episode_step"] == 1:
            first_step_errors.append(float(np.linalg.norm(error)))
        # Undo the exact supervision scaling before spherical-to-Cartesian conversion.
        spherical_states = np.asarray([
            row["state_estimate_labels_scaled"][3:6],
            row["state_estimates_scaled"][3:6],
        ]) / np.asarray([0.5, 1.0, 1.3])
        radius, elevation, azimuth = spherical_states.T
        cartesian_states = np.column_stack((
            radius * np.cos(elevation) * np.cos(azimuth),
            radius * np.cos(elevation) * np.sin(azimuth),
            radius * np.sin(elevation),
        ))
        estimation_errors.append(cartesian_states[1] - cartesian_states[0])
        if row["time_s"] >= 1.0:
            scaled_actions = np.asarray(row["action"]) * (
                1.0 + np.asarray(row["domain_randomization"]["execution_scale_delta"])
            )
            position_targets.append(np.clip(0.25 * scaled_actions[:4], -1.5708, 1.5708))
            drive_targets.append(np.clip(5.0 * scaled_actions[4:8], -20.0, 20.0))
    tracking = np.abs(np.asarray(tracking_errors))
    estimation = np.abs(np.asarray(estimation_errors))
    moving = np.asarray(movement_phases, dtype=bool)
    tracking_l2 = np.linalg.norm(tracking, axis=1)
    estimation_l2 = np.linalg.norm(estimation, axis=1)
    steering = np.asarray(position_targets)
    drive = np.asarray(drive_targets)
    return {
        "sample_count": len(tracking),
        "tracking_world_xyz_mae_m": tracking.mean(axis=0).tolist(),
        "tracking_world_xyz_p95_m": np.quantile(tracking, 0.95, axis=0).tolist(),
        "tracking_world_xyz_fraction_below_10cm": (tracking < 0.1).mean(axis=0).tolist(),
        "tracking_l2_fraction_below_10cm": float((tracking_l2 < 0.1).mean()),
        "moving_target_tcp_l2_mean_m": float(tracking_l2[moving].mean()),
        "holding_target_tcp_l2_mean_m": float(tracking_l2[~moving].mean()),
        "estimation_yaw_xyz_mae_m": estimation.mean(axis=0).tolist(),
        "estimation_yaw_xyz_p95_m": np.quantile(estimation, 0.95, axis=0).tolist(),
        "estimation_yaw_xyz_fraction_below_5cm": (estimation < 0.05).mean(axis=0).tolist(),
        "estimation_l2_mean_m": float(estimation_l2.mean()),
        "first_step_tcp_l2_mean_m": float(np.mean(first_step_errors)),
        "first_step_tcp_l2_min_max_m": [float(min(first_step_errors)), float(max(first_step_errors))],
        "steering_target_mean_abs_rad": np.abs(steering).mean(axis=0).tolist(),
        "steering_target_fraction_abs_above_60deg": (np.abs(steering) > np.pi / 3).mean(axis=0).tolist(),
        "drive_target_mean_abs_rad_s": np.abs(drive).mean(axis=0).tolist(),
        "drive_target_fraction_clipped": (np.abs(drive) >= 20.0).mean(axis=0).tolist(),
    }


if __name__ == "__main__":
    archive_root = Path("/mnt/data_8THDD/myl_unifp_v3_stage1/evaluations/a8000_seed1_v3_collision_v1_20261104_r02")
    summaries = {
        name: summarize_scenario(archive_root / name / "policy_trace.jsonl")
        for name in ("S0", "S1", "S2")
    }
    print(json.dumps(summaries, ensure_ascii=False, indent=2, allow_nan=False))

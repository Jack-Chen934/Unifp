"""Read-only A-stage evaluation metrics and immutable evidence packaging."""

import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from .v3_run_identity import file_sha256


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
    shutil.copyfile(checkpoint_path, archive_dir / "checkpoint_a8000.pt")
    shutil.copyfile(manifest_path, archive_dir / "evaluation_manifest.json")
    shutil.copyfile(trace_path, archive_dir / "policy_trace.jsonl")
    shutil.copyfile(episodes_path, archive_dir / "episodes.jsonl")
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
    inventory = {path.relative_to(archive_dir).as_posix(): file_sha256(path)
                 for path in sorted(archive_dir.rglob("*")) if path.is_file()}
    if inventory["checkpoint_a8000.pt"] != file_sha256(checkpoint_path):
        raise OSError("Archived checkpoint SHA256 differs from the evaluated checkpoint")
    archive_manifest = {
        "checkpoint_sha256": inventory["checkpoint_a8000.pt"],
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

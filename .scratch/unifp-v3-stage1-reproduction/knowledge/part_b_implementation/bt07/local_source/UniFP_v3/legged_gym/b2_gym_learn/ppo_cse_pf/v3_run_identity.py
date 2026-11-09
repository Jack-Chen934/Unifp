"""Fingerprint actual learning sources, loaded assets and the V3 semantic schema."""

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import torch


def validate_force_snapshot(snapshot: Dict[str, Any], num_envs: int) -> None:
    """Validate B diagnostic event tensors at checkpoint boundaries only.

    Args:
        snapshot: Two event streams plus applied/pending world newtons and local ticks.
        num_envs: Required robot count, shared with saved creation properties.

    Raises:
        ValueError: Required keys, finite tensor shapes or clock dtypes are invalid.
    """
    fields = {"command", "external", "pending_ee_force_world", "applied_ee_force_world", "episode_tick"}
    if not isinstance(snapshot, dict) or set(snapshot) != fields:
        raise ValueError("Incomplete B force snapshot")
    tensors = [(snapshot["pending_ee_force_world"], (num_envs, 3), torch.float32),
               (snapshot["applied_ee_force_world"], (num_envs, 3), torch.float32),
               (snapshot["episode_tick"], (num_envs,), torch.long)]
    for stream in ("command", "external"):
        state = snapshot[stream]
        if set(state) != {"peak", "enabled", "start_tick", "next_start_tick", "ramp_ticks"}:
            raise ValueError("Incomplete B force event state")
        for name, shape, dtype in (("peak", (num_envs, 3), torch.float32),
                                   ("enabled", (num_envs,), torch.bool),
                                   ("start_tick", (num_envs,), torch.long),
                                   ("next_start_tick", (num_envs,), torch.long),
                                   ("ramp_ticks", (num_envs,), torch.float32)):
            tensors.append((state[name], shape, dtype))
    if any(not isinstance(value, torch.Tensor) or value.shape != shape
           or value.dtype != dtype or not torch.isfinite(value).all()
           for value, shape, dtype in tensors):
        raise ValueError("Invalid B force snapshot tensors")


# Only files implementing the approved B stage may differ from the frozen A source.
A_TO_B_SOURCE_PATHS = {
    "legged_gym/envs/v3/legged_robot_v3.py",
    "legged_gym/envs/v3/v3_pos_force_config.py",
    "legged_gym/envs/v3/v3_force_evaluation.py",
    "legged_gym/envs/v3/v3_contact_evaluation.py",
    "legged_gym/scripts/train_v3.py",
    "legged_gym/scripts/v3_args.py",
    "legged_gym/scripts/evaluate_v3.py",
    "legged_gym/scripts/evaluate_v3_contact.py",
    "legged_gym/scripts/preflight_v3.py",
    "legged_gym/utils/v3_config.py",
    "legged_gym/b2_gym_learn/ppo_cse_pf/v3_run_identity.py",
    "legged_gym/b2_gym_learn/ppo_cse_pf/v3_on_policy_runner.py",
    "legged_gym/b2_gym_learn/ppo_cse_pf/v3_evaluation.py",
    "legged_gym/b2_gym_learn/ppo_cse_pf/v3_preflight.py",
    "legged_gym/b2_gym_learn/ppo_cse_pf/v3_episode_log.py",
}


def _compatible_learning_config(config_snapshot: Dict[str, Any], asset_relative_path: str) -> Dict[str, Any]:
    """Normalize only the output fields already excluded from same-stage identity.

    Args:
        config_snapshot: Full effective environment and training configuration.
        asset_relative_path: Loaded URDF path relative to its own source checkout.

    Returns:
        Independent config with portable asset path and unchanged learning settings.

    Raises:
        KeyError: Required environment, asset or training runner settings are absent.
    """
    compatible_config = copy.deepcopy(config_snapshot)
    runner_config = compatible_config["training"]["runner"]
    for field in ("max_iterations", "save_interval", "run_name", "tensorboard", "resume", "load_run", "checkpoint", "resume_path"):
        runner_config.pop(field, None)
    compatible_config["environment"]["asset"]["file"] = asset_relative_path
    # Newly declared but inactive A fields do not change the frozen A semantics.
    if compatible_config["environment"]["run"]["reproduction_part"] == "A":
        compatible_config["environment"].pop("force_events", None)
    commands = compatible_config["environment"].get("commands", {})
    if commands.get("zero_vel_cmd_prob") == 0.0:
        commands.pop("zero_vel_cmd_prob")
    return compatible_config


def file_sha256(path: Path) -> str:
    """Hash exact file bytes without loading a whole checkpoint into memory.

    Args:
        path: Existing source, asset or checkpoint file.

    Returns:
        Lowercase SHA256 digest of its contents.

    Raises:
        OSError: The file cannot be read.
    """
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_run_identity(source_root: Path, config_snapshot: Dict[str, Any], runtime: Dict[str, Any]) -> Dict[str, Any]:
    """Describe strict same-part continuation using actual content and effective config.

    Args:
        source_root: V3 checkout containing legged_gym and the active asset.
        config_snapshot: Full pre-construction effective environment/training config.
        runtime: Explicit interpreter/dependency, simulator, pipeline and CUDA identity.

    Returns:
        JSON-compatible identity including source/asset manifests and semantic schema.

    Raises:
        OSError: A required source or asset file cannot be read.
        ValueError: Effective config contains non-finite JSON values.
    """
    environment = config_snapshot["environment"]
    # Output location, display/save frequency and the requested extra budget do not
    # alter the learning process. Everything else remains in strict compatibility.
    asset_path = Path(environment["asset"]["file"])
    compatible_config = _compatible_learning_config(config_snapshot, asset_path.relative_to(source_root).as_posix())
    asset_root = asset_path.parent.parent
    source_files = sorted((source_root / "legged_gym").rglob("*.py")) + [source_root / "SOURCE_BASELINE.json"]
    asset_files = sorted(path for path in asset_root.rglob("*") if path.is_file())
    schema = {
        "version": 1, "actor_frame": 58, "actor_history": 32, "critic_frame": 98,
        "critic_history": 3, "action_dim": 14, "prediction_dim": 12,
        "actor_fields": ["roll_pitch_rad:0:2", "body_angular_velocity_rad_s_x0.25:2:5",
                         "steering_arm_position_residual_rad:5:15", "joint_velocity_rad_s_x0.05:15:29",
                         "previous_executed_action:29:43", "scaled_command:43:58"],
        "critic_fields": ["auxiliary_labels", "base_mass_kg_com_local_m_tool_mass_kg_delta",
                          "friction", "execution_scale_minus_one", "wheel_world_z_contact_gt_5N",
                          "body_gravity", "body_angular_velocity_x0.25", "position_residual",
                          "joint_velocity_x0.05", "previous_action", "scaled_command", "compensated_spherical_target"],
        "joint_order": [f"{corner}_steering_joint" for corner in ("fl", "fr", "bl", "br")]
                       + [f"{corner}_wheel_drive_joint" for corner in ("fl", "fr", "bl", "br")]
                       + [f"arm_joint{index}" for index in range(1, 7)],
        "action_semantics": "raw diagonal Gaussian; steering/arm default-angle residual; drive rad/s target; execution scale before hard clipping",
        "command_scale": [2.0, 2.0, 0.25, 0.5, 1.0, 1.3, 0.5, 0.5, 0.5, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
        "auxiliary_groups": ["body_linear_velocity_m_s_x2", "yaw_spherical_tcp_m_rad_x(0.5,1,1.3)",
                             "injected_ee_force_yaw_N_x0.01", "injected_base_force_yaw_N_x0.01"],
        "auxiliary_mse_coefficients": [0.04, 0.04, 1.0, 1.0],
        "networks": "1856-512-256-128-64; actor122-512-256-128-14; decoder64-128-64-12; critic294-512-256-128-1; ELU; direct_std",
        "timeout_bootstrap": "true_terminal_successor_v_once_done_masked_gae",
    }
    return {"task": "v3_pos_force", "seed": config_snapshot["training"]["seed"],
            "reproduction_part": environment["run"]["reproduction_part"],
            "run_purpose": environment["run"]["run_purpose"], "runtime": runtime,
            "official_commit": json.loads((source_root / "SOURCE_BASELINE.json").read_text())["commit"],
            "source_manifest": {path.relative_to(source_root).as_posix(): file_sha256(path) for path in source_files},
            "asset_manifest": {path.relative_to(source_root).as_posix(): file_sha256(path) for path in asset_files},
            "schema": schema, "schema_sha256": hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest(),
            "learning_config_sha256": hashlib.sha256(json.dumps(compatible_config, sort_keys=True, allow_nan=False).encode()).hexdigest()}


def validate_a_to_b_transition(
    parent_identity: Dict[str, Any], target_identity: Dict[str, Any],
    parent_config: Dict[str, Any], target_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Check a deliberate stage migration without weakening same-stage resume.

    Args:
        parent_identity: Original A source, schema, asset and runtime fingerprints.
        target_identity: Actual B fingerprints computed before Gym construction.
        parent_config: Complete effective config stored in the A checkpoint.
        target_config: Actual resolved B config, with the same environment count.

    Returns:
        Explicit changed source paths, config values and portable runtime locations.

    Raises:
        ValueError: A fingerprint, stage, physics, schema or learner setting differs
            outside the approved A-to-B changes.
        KeyError: Required identity or configuration fields are missing.

    Note:
        Reads dictionaries only; it does not modify either identity or config.
    """
    # Verify that both full configs describe the bytes fingerprinted at each end.
    normalized_configs = []
    for identity, configuration in ((parent_identity, parent_config), (target_identity, target_config)):
        asset_file = Path(configuration["environment"]["asset"]["file"]).as_posix()
        asset_paths = [name for name in identity["asset_manifest"] if asset_file.endswith("/" + name)]
        if len(asset_paths) != 1:
            raise ValueError("Configuration asset path is missing from its fingerprint")
        compatible = _compatible_learning_config(configuration, asset_paths[0])
        digest = hashlib.sha256(json.dumps(compatible, sort_keys=True, allow_nan=False).encode()).hexdigest()
        if digest != identity["learning_config_sha256"]:
            raise ValueError("Configuration fingerprint mismatch")
        normalized_configs.append(compatible)
    if parent_identity["reproduction_part"] != "A" or target_identity["reproduction_part"] != "B":
        raise ValueError("Stage migration requires an explicit A parent and B target")
    for field in ("task", "seed", "official_commit", "asset_manifest", "schema", "schema_sha256"):
        if parent_identity[field] != target_identity[field]:
            raise ValueError(f"A-to-B incompatible {field}")

    # Permit the B09 event and command parameters explicitly, keeping all other
    # physics/learner settings strict and preserving the full changes in lineage.
    parent_normalized, target_normalized = normalized_configs
    config_changes = {}
    parent_events = parent_normalized["environment"].pop("force_events", None)
    target_events = target_normalized["environment"].pop("force_events", None)
    if target_events is not None:
        expected_events = {"enabled_probability": 0.8, "interval_s": [3.5, 9.0],
                           "ramp_s": [1.0, 3.0], "hold_s": 1.0}
        if ({key: value for key, value in target_events.items() if key != "max_force_n"} != expected_events
                or not 0.0 < target_events.get("max_force_n", 0.0) <= 60.0):
            raise ValueError("Unapproved B force event configuration")
    if parent_events != target_events:
        config_changes["environment.force_events"] = {"parent": parent_events, "target": target_events}
    before_probability = parent_normalized["environment"].get("commands", {}).pop("zero_vel_cmd_prob", 0.0)
    after_probability = target_normalized["environment"].get("commands", {}).pop("zero_vel_cmd_prob", 0.0)
    if before_probability != after_probability:
        if before_probability != 0.0 or after_probability != 0.3:
            raise ValueError("Unapproved B zero-command configuration")
        config_changes["environment.commands.zero_vel_cmd_prob"] = {"parent": before_probability, "target": after_probability}
    for section, group, field in (("environment", "run", "reproduction_part"),
                                  ("environment", "run", "run_purpose"),
                                  ("training", "runner", "experiment_name")):
        before = parent_normalized[section][group].pop(field, None)
        after = target_normalized[section][group].pop(field, None)
        if before != after:
            config_changes[f"{section}.{group}.{field}"] = {"parent": before, "target": after}
    if parent_normalized != target_normalized:
        raise ValueError("A-to-B learning or physics configuration mismatch")
    source_changes = sorted(name for name in parent_identity["source_manifest"].keys() | target_identity["source_manifest"].keys()
                            if parent_identity["source_manifest"].get(name) != target_identity["source_manifest"].get(name))
    if set(source_changes) - A_TO_B_SOURCE_PATHS:
        raise ValueError("A-to-B unapproved source changes: " + ", ".join(sorted(set(source_changes) - A_TO_B_SOURCE_PATHS)))
    runtime_changes = {name: {"parent": parent_identity["runtime"].get(name), "target": target_identity["runtime"].get(name)}
                       for name in parent_identity["runtime"].keys() | target_identity["runtime"].keys()
                       if parent_identity["runtime"].get(name) != target_identity["runtime"].get(name)}
    if set(runtime_changes) - {"python_executable", "gym_import_path", "cuda_visible_devices"}:
        raise ValueError("A-to-B incompatible runtime dependencies or simulation device")
    return {"source_changes": source_changes, "config_changes": config_changes, "runtime_changes": runtime_changes}

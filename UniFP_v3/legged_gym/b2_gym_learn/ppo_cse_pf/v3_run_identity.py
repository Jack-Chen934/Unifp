"""Fingerprint actual learning sources, loaded assets and the V3 semantic schema."""

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict


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
    compatible_config = copy.deepcopy(config_snapshot)
    runner_config = compatible_config["training"]["runner"]
    for field in ("max_iterations", "save_interval", "run_name", "tensorboard", "resume", "load_run", "checkpoint", "resume_path"):
        runner_config.pop(field, None)
    asset_path = Path(environment["asset"]["file"])
    compatible_config["environment"]["asset"]["file"] = asset_path.relative_to(source_root).as_posix()
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

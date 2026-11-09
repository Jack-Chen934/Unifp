"""Read-only evidence checks for the V3 A long-training start gate."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from .v3_evaluation import validate_evaluation_manifest


IDENTITY_FIELDS = ("official_commit", "source_manifest", "asset_manifest", "schema_sha256", "runtime")
REQUIRED_STAGES = ("unit_tests", "gym_smoke", "training_resume", "capacity")


def validate_stage_artifacts(evidence: Dict[str, Any]) -> List[str]:
    """Verify that each declared stage matches a hash-pinned JSON report.

    Args:
        evidence: A-start evidence containing stage records and absolute report paths.

    Returns:
        Blocker messages for absent, unreadable, stale, or mismatched stage reports.

    Note:
        Reads report files but does not execute tests or certify who produced them.
    """
    blockers: List[str] = []
    stages = evidence.get("stages")
    if not isinstance(stages, dict):
        return ["缺少阶段报告：stages"]

    for name in REQUIRED_STAGES:
        stage = stages.get(name)
        artifact = stage.get("artifact") if isinstance(stage, dict) else None
        if not isinstance(artifact, dict):
            blockers.append(f"阶段缺少机器报告引用：{name}")
            continue
        report_path = artifact.get("path")
        expected_sha256 = artifact.get("sha256")
        if not isinstance(report_path, str) or not Path(report_path).is_absolute():
            blockers.append(f"阶段报告路径必须为绝对路径：{name}")
            continue
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            blockers.append(f"阶段报告缺少有效SHA-256：{name}")
            continue
        try:
            report_bytes = Path(report_path).read_bytes()
            report = json.loads(report_bytes.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            blockers.append(f"阶段报告无法读取：{name}")
            continue
        if hashlib.sha256(report_bytes).hexdigest() != expected_sha256:
            blockers.append(f"阶段报告SHA-256不匹配：{name}")
            continue
        stage_record = {key: value for key, value in stage.items() if key != "artifact"}
        if report != {"stage": name, "result": stage_record}:
            blockers.append(f"阶段报告内容与证据字段不一致：{name}")
    return blockers


def assess_a_start_gate(
    evidence: Dict[str, Any],
    active_identity: Dict[str, Any],
    evaluation_manifest: Dict[str, Any],
    *,
    formal_num_envs: int,
) -> Dict[str, Any]:
    """Check whether required A-stage test and provenance evidence is complete.

    Args:
        evidence: JSON-compatible ticket-07 evidence object. It contains the
            source identity, formal configuration digest, and named test reports.
        active_identity: Identity recomputed from the checkout and runtime that
            will be used for formal A training.
        evaluation_manifest: Frozen A evaluation scenarios and approved limits.
        formal_num_envs: Explicit positive environment count from the formal
            configuration, not inferred from a smaller development run.

    Returns:
        A report with a boolean ``ready`` field and actionable blocker strings.

    Raises:
        ValueError: An input object has an unsupported schema or malformed shape.
    """
    if type(formal_num_envs) is not int or formal_num_envs < 1:
        raise ValueError("Formal environment count must be a positive integer")
    if evidence.get("schema_version") != 1:
        raise ValueError("A-start evidence must use schema_version 1")
    report_identity = evidence.get("identity")
    if not isinstance(report_identity, dict):
        raise ValueError("A-start evidence requires an identity object")
    if not isinstance(evidence.get("formal_config_sha256"), str):
        raise ValueError("A-start evidence requires formal_config_sha256")

    blockers: List[str] = []
    for field in IDENTITY_FIELDS:
        if report_identity.get(field) != active_identity.get(field):
            blockers.append(f"源码／资产／schema／运行身份不匹配：{field}")
    if evidence["formal_config_sha256"] != active_identity.get("learning_config_sha256"):
        blockers.append("正式学习配置摘要与当前A配置不匹配")

    stages = evidence.get("stages")
    if not isinstance(stages, dict):
        raise ValueError("A-start evidence requires a stages object")
    # Each report owns its provenance; refreshing only the summary cannot make
    # an older physical test apply to a changed source or asset.
    for name in ("unit_tests", "gym_smoke", "training_resume", "capacity"):
        stage = stages.get(name, {})
        stage_identity = stage.get("identity", {}) if isinstance(stage, dict) else {}
        fields = IDENTITY_FIELDS if name != "unit_tests" else IDENTITY_FIELDS[:-1]
        for field in fields:
            if not active_identity.get(field) or stage_identity.get(field) != active_identity[field]:
                blockers.append(f"阶段报告身份不匹配：{name}（{field}）")
    _check_stage(stages, "unit_tests", blockers, passed=True)
    _check_stage(stages, "gym_smoke", blockers, passed=True, num_envs=32,
                 zero_ee_command_force=True, zero_ee_external_force=True,
                 finite_core=True, reset_isolated=True)
    _check_stage(stages, "training_resume", blockers, passed=True, num_envs=512,
                 iterations=10, resumed_iterations=1, completed_iterations=11,
                 counters_consistent=True, checkpoint_identity_matches=True)
    _check_stage(stages, "capacity", blockers, passed=True, num_envs=formal_num_envs,
                 formal_config_sha256=evidence["formal_config_sha256"], rollout_update=True)

    try:
        validate_evaluation_manifest(evaluation_manifest)
    except ValueError as error:
        blockers.append(f"A10评估manifest未满足启动要求：{error}")
    if evaluation_manifest.get("thresholds") is None:
        blockers.append("A10绝对阈值尚未确认")

    return {"ready": not blockers, "blockers": blockers,
            "formal_num_envs": formal_num_envs,
            "formal_config_sha256": evidence["formal_config_sha256"],
            "identity_matches": not any("身份不匹配" in blocker for blocker in blockers)}


def _check_stage(
    stages: Dict[str, Any],
    name: str,
    blockers: List[str],
    **required_values: Any,
) -> None:
    """Append a blocker when one named test report is absent or inconsistent.

    Args:
        stages: Named reports supplied by the test and capacity workflows.
        name: Required report key.
        blockers: Mutable list receiving the user-facing blocker description.
        **required_values: Exact report fields required for this A11 stage.

    Returns:
        None.

    Note:
        Appends to blockers without changing the supplied stage evidence.
    """
    stage = stages.get(name)
    if not isinstance(stage, dict):
        blockers.append(f"缺少阶段报告：{name}")
        return
    mismatches = [field for field, expected in required_values.items() if stage.get(field) != expected]
    if mismatches:
        blockers.append(f"阶段报告未通过或配置不符：{name}（{', '.join(mismatches)}）")

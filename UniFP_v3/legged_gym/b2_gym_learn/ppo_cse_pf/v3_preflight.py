"""Read-only evidence checks for the V3 A/B long-training start gates."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from .v3_evaluation import validate_evaluation_manifest


IDENTITY_FIELDS = ("official_commit", "source_manifest", "asset_manifest", "schema_sha256", "runtime")
REQUIRED_STAGES = ("unit_tests", "gym_smoke", "training_resume", "capacity")
B_REQUIRED_STAGES = ("unit_tests", "force_smoke", "contact_smoke", "zero_force_regression",
                     "training_resume", "capacity", "evaluation_baseline")


def validate_stage_artifacts(evidence: Dict[str, Any], *,
                             required_stages: tuple = REQUIRED_STAGES) -> List[str]:
    """Verify that each declared stage matches a hash-pinned JSON report.

    Args:
        evidence: A-start evidence containing stage records and absolute report paths.
        required_stages: Explicit stage names; defaults to the unchanged A contract.

    Returns:
        Blocker messages for absent, unreadable, stale, or mismatched stage reports.

    Note:
        Reads report files but does not execute tests or certify who produced them.
    """
    blockers: List[str] = []
    stages = evidence.get("stages")
    if not isinstance(stages, dict):
        return ["缺少阶段报告：stages"]

    for name in required_stages:
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


def assess_b_start_gate(
    evidence: Dict[str, Any], active_identity: Dict[str, Any], parent_metadata: Dict[str, Any],
    force_manifest: Dict[str, Any], contact_manifest: Dict[str, Any],
    zero_force_manifest: Dict[str, Any], *, formal_num_envs: int = 4096,
) -> Dict[str, Any]:
    """Require complete B engineering evidence before any formal Gym creation.

    Args:
        evidence: Schema-1 reports, exact parent digest and canonical manifest hashes.
        active_identity: Recomputed formal B source, V6 asset, schema and runtime.
        parent_metadata: Actual A8000 digest, stage, seed, count and asset/schema IDs.
        force_manifest: Frozen 14-condition protocol with force range in N.
        contact_manifest: Frozen wall protocol with geometry and SI force thresholds.
        zero_force_manifest: Unchanged A three-scenario protocol and thresholds.
        formal_num_envs: Confirmed complete-parent count, fixed at 4096 for B.

    Returns:
        Readiness, actionable blockers and diagnostics; A's B tracking performance
        and ordinary physical failures are not engineering blockers.

    Raises:
        ValueError: The evidence container has an unsupported schema.

    Note:
        Reads hash-pinned report files once at startup; never creates simulation,
        rewrites reports or changes training/evaluation thresholds.
    """
    if evidence.get("schema_version") != 1:
        raise ValueError("B-start evidence must use schema_version 1")
    blockers = validate_stage_artifacts(evidence, required_stages=B_REQUIRED_STAGES)
    if formal_num_envs != 4096 or active_identity.get("reproduction_part") != "B":
        blockers.append("B正式配置须为4096环境和显式B阶段")
    for field in IDENTITY_FIELDS:
        if not active_identity.get(field) or evidence.get("identity", {}).get(field) != active_identity[field]:
            blockers.append(f"B源码／资产／schema／运行身份不匹配：{field}")
    if evidence.get("formal_config_sha256") != active_identity.get("learning_config_sha256"):
        blockers.append("B正式学习配置摘要不匹配")
    if (evidence.get("parent_sha256") != parent_metadata.get("sha256")
            or parent_metadata.get("reproduction_part") != "A"
            or parent_metadata.get("completed_iterations") != 8000
            or parent_metadata.get("num_envs") != formal_num_envs
            or parent_metadata.get("seed") != active_identity.get("seed")
            or parent_metadata.get("asset_manifest") != active_identity.get("asset_manifest")
            or parent_metadata.get("schema_sha256") != active_identity.get("schema_sha256")):
        blockers.append("B正式父SHA／A8000／seed／4096／V6资产或schema不匹配")
    # Hashes bind the complete protocol and thresholds, not merely its version label.
    for name, manifest in (("force", force_manifest), ("contact", contact_manifest), ("zero_force", zero_force_manifest)):
        digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        if evidence.get("manifest_sha256", {}).get(name) != digest:
            blockers.append(f"B评估协议摘要不匹配：{name}")
    if (not force_manifest.get("range_frozen") or not contact_manifest.get("range_frozen")
            or force_manifest.get("max_force_n") != 60.0 or contact_manifest.get("max_force_n") != 60.0
            or contact_manifest.get("contact_peak_n") != 20.0
            or len(force_manifest.get("scenarios", {})) != 14
            or force_manifest.get("thresholds") is None or contact_manifest.get("thresholds") is None):
        blockers.append("B力范围／接触幅值或完整评估协议尚未冻结")
    try:
        validate_evaluation_manifest(zero_force_manifest)
    except ValueError as error:
        blockers.append(f"A零力回归协议不完整：{error}")
    stages = evidence.get("stages", {})
    for name in B_REQUIRED_STAGES:
        fields = IDENTITY_FIELDS[:-1] if name == "unit_tests" else IDENTITY_FIELDS
        for field in fields:
            if stages.get(name, {}).get("identity", {}).get(field) != active_identity.get(field):
                blockers.append(f"B阶段报告身份不匹配：{name}（{field}）")
    _check_stage(stages, "unit_tests", blockers, passed=True)
    _check_stage(stages, "force_smoke", blockers, passed=True, two_runtimes=True,
                 com_force_verified=True, reset_isolated=True)
    _check_stage(stages, "contact_smoke", blockers, passed=True, two_runtimes=True,
                 measurement_verified=True, reset_isolated=True)
    _check_stage(stages, "zero_force_regression", blockers, passed=True,
                 scenarios=["S0", "S1", "S2"], num_envs=32, episodes_per_env=3, policy_updates=0)
    _check_stage(stages, "training_resume", blockers, passed=True, num_envs=4096,
                 iterations=10, resumed_iterations=1, completed_iterations=8011,
                 counters_consistent=True, full_state_restored=True, nonzero_force_learned=True)
    _check_stage(stages, "capacity", blockers, passed=True, num_envs=4096,
                 rollout_update=True, stress_verified=True, physics_warnings_absent=True)
    _check_stage(stages, "evaluation_baseline", blockers, passed=True, force_scenarios=14,
                 force_episodes=1344, contact_episodes=96, zero_control_episodes=32,
                 policy_updates=0, cost_recorded=True)
    return {"ready": not blockers, "blockers": blockers, "formal_num_envs": formal_num_envs,
            "diagnostics": {"max_force_n": force_manifest.get("max_force_n"),
                            "ordinary_physical_failures_are_blockers": False,
                            "b_performance_required_of_a_parent": False}}


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

"""从已完成的实际报告汇总B工程准入，不生成虚构的测试通过状态。"""

import copy
import hashlib
import json
import math
import re
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict

import isaacgym  # noqa: F401
import torch
import legged_gym.envs  # noqa: F401
from legged_gym.b2_gym_learn.ppo_cse_pf.v3_run_identity import build_run_identity
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.v3_config import resolve_v3_configs


ROOT = Path("/mnt/data_8THDD/myl_unifp_v3_stage1/validation/b_stage_20261107_r01")
SOURCE = ROOT / "source_r06/UniFP_v3"


def digest(path: Path) -> str:
    """Hash an existing evidence file without changing it.

    Args:
        path: Original artifact whose full bytes are bound to the report.

    Returns:
        Hexadecimal SHA256 content digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    """Read one original machine report without altering runtime or identity.

    Args:
        path: UTF-8 JSON report in the data-volume validation tree.

    Returns:
        The exact report object decoded from disk.
    """
    return json.loads(path.read_text())


def reference(path: Path) -> Dict[str, str]:
    """Pin a source artifact by absolute location and byte digest.

    Args:
        path: Existing original machine report or immutable trace/archive index.

    Returns:
        Absolute path and SHA256, preserving the original report bytes.
    """
    return {"path": str(path), "sha256": digest(path)}


def verify_archive(directory: Path) -> Dict[str, Any]:
    """Check every sealed archive byte and return its unmodified summary.

    Args:
        directory: Existing scenario archive produced by the real evaluator.

    Returns:
        Verified original scenario summary, including retained failures.

    Raises:
        AssertionError: A quota, update count or archived artifact is inconsistent.
    """
    archive = read_json(directory / "archive_manifest.json")
    artifacts = archive["artifacts"]
    for name, item in artifacts.items():
        expected = item["sha256"] if isinstance(item, dict) else item
        assert digest(directory / name) == expected, directory / name
    summary = read_json(directory / "summary.json")
    assert summary["policy_updates"] == 0 and summary["evidence_complete"]
    if summary.get("protocol") == "fixed_wall_contact":
        assert all(episode["end_type"] != "incomplete" for episode in summary["episodes"])
    else:
        assert summary["episodes"]["completed"] == summary["episodes"]["expected"]
    return summary


def main() -> None:
    """Aggregate verified engineering stages and prepare the actual formal identity.

    Raises:
        AssertionError: Any claimed engineering result lacks its original evidence.
        FileNotFoundError: A required run has not completed and published its report.

    Note:
        Writes only new evidence/configuration reports. No simulation or learning
        is started, and no historical report is rewritten to a newer source hash.
    """
    output = ROOT / "bt08_evidence_r06"
    train_dir, resume_dir = ROOT / "bt08_train10_allrng_r04", ROOT / "bt08_resume1_allrng_r04"
    train = read_json(train_dir / "native_training_report.json")
    resume = read_json(resume_dir / "native_training_report.json")
    peak_dir = ROOT / "bt08_peak4096_xz_r04"
    peak = read_json(peak_dir / "report.json")
    bridge_path = ROOT / "bt08_source_bridge_r06.json"
    assert read_json(bridge_path)["passed"]
    args = Namespace(reproduction_part="B", run_purpose="formal", seed=1, num_envs=4096,
                     max_iterations=None)
    cfg, training = resolve_v3_configs(args, purpose="formal")
    snapshot = {"environment": class_to_dict(cfg), "training": class_to_dict(training)}
    runtime = train["identity"]["runtime"]
    identity = build_run_identity(SOURCE, snapshot, runtime)
    bridge = read_json(bridge_path)["versions"]["source_r04"]
    assert train["identity"]["source_manifest"] == {**bridge["original_manifest"], "SOURCE_BASELINE.json": identity["source_manifest"]["SOURCE_BASELINE.json"]}
    assert [item["path"] for item in bridge["changed_paths"]] == ["legged_gym/scripts/evaluate_v3_contact.py"]
    unit = read_json(ROOT / "bt08_unit_report_r06.json")
    assert unit["passed"] and unit["source_manifest"] == identity["source_manifest"]
    parent_path = Path("/mnt/data_8THDD/myl_unifp_v3_stage1/runs/a8000_seed1_v3_v6r2_resume0767_20261105/checkpoints/model_008000.pt")
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    parent_metadata = {"sha256": digest(parent_path), "reproduction_part": parent["identity"]["reproduction_part"],
        "completed_iterations": parent["progress"]["completed_iterations"],
        "num_envs": parent["environment_training_state"]["num_envs"], "seed": parent["identity"]["seed"],
        "asset_manifest": parent["identity"]["asset_manifest"], "schema_sha256": parent["identity"]["schema_sha256"]}
    assert train["completed_iterations"] == 8010 and resume["completed_iterations"] == 8011
    assert train["iterations"] == 10 and resume["iterations"] == 1
    for report in (train, resume):
        assert report["passed"] and report["rollout_step"] == 0
        assert report["restoration"]["model_and_both_adam_equal"]
        assert report["restoration"]["rng_before_reset_equal"]
        assert report["restoration"]["fresh_reset_calls"] == 1
        assert report["restoration"]["creation_samples_equal"]
        assert report["telemetry"]["nonzero_label_rows"] > 0
    assert resume["adam_steps"] == {"main": [160220], "auxiliary": [160220]}
    assert peak["passed"] and peak["num_envs"] == 4096 and peak["policy_steps"] == 500
    assert peak["policy_updates"] == 0 and peak["parent_creation_samples"]
    bad_physics = re.compile(r"(foundLost.*(exceed|overflow)|max_gpu_contact_pairs.*(exceed|overflow)|out of memory|illegal memory|Failed to allocate|missing.*interaction)", re.I)
    logs = [ROOT / (name + ".log") for name in (train_dir.name, resume_dir.name, peak_dir.name)]
    for path in logs:
        assert not bad_physics.search(path.read_text()), path
    for directory in (train_dir, resume_dir):
        for line in (directory / "iterations.jsonl").read_text().splitlines():
            record = json.loads(line)
            assert all(math.isfinite(value) for value in record.values() if isinstance(value, (int, float)))
        checkpoint = directory / "checkpoints" / ("model_008010.pt" if directory == train_dir else "model_008011.pt")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert all(bool(torch.isfinite(value).all()) for value in payload["model_state_dict"].values())
    source_evidence = ROOT / "bt07_original_reports"
    smoke_refs = {name: reference(source_evidence / name) for name in (
        "local_force_interface_r02_report.json", "remote_force_interface_r02_report.json",
        "local_contact_interface_r02_report.json", "remote_contact_interface_r01_report.json")}
    for name in smoke_refs:
        original = read_json(source_evidence / name)
        assert original["local_reset_isolated"]
        if "force" in name:
            assert original["num_envs"] == 4 and original["policy_steps"] == 1750
            assert original["free_body"]["relative_error"] < 1e-4
            assert original["free_body"]["angular_increment_rad_s"] == 0.
        else:
            assert original["wall_pose_preserved"] and min(original["clean_contact_consecutive_intervals"]) >= 2
    zero_report = read_json(source_evidence / "a_zero_regression_comparison.json")
    assert zero_report["status"] == "passed"
    for scenario, item in zero_report["scenarios"].items():
        summary = verify_archive(Path(item["archive"]))
        assert summary["episodes"]["completed"] == 96 and summary["assessment"] == "pass"
    force_manifest = read_json(SOURCE / "configs/v3_b_initial_v1.json")
    baseline, costs, references = {}, [], {}
    for index, name in enumerate(force_manifest["scenarios"]):
        queue = "first" if index < 7 else "second"
        queue_root = ROOT / f"bt08_baseline_{queue}_r04"
        archive = queue_root / name / name
        summary = verify_archive(archive)
        assert summary["episodes"]["completed"] == 96
        assert len(list(archive.glob("*.mp4"))) == 1
        assert (archive / "plots/force_tracking.png").is_file()
        assert not bad_physics.search((queue_root / (name + ".log")).read_text())
        baseline[name] = summary
        references[name] = str(archive.parent)
        cost = next(json.loads(line) for line in (queue_root / "costs.jsonl").read_text().splitlines()
                    if json.loads(line)["scenario"] == name)
        assert cost["returncode"] == 0 and cost["peak_process_gpu_memory_mib"] > 0
        costs.append({**cost, "scenario_output_bytes": sum(p.stat().st_size for p in archive.rglob("*") if p.is_file())})
    contact_root = ROOT / "bt08_contact_r06/contact"
    for name, expected in (("contact", 96), ("zero_control", 32)):
        summary = verify_archive(contact_root / name)
        assert len(summary["episodes"]) == expected
        assert len(list((contact_root / name).glob("*.mp4"))) == (2 if name == "contact" else 1)
        assert (contact_root / name / "plots/contact_env0_episode0.png").is_file()
        assert not bad_physics.search((contact_root / (name + ".console.log")).read_text())
        baseline[name] = summary
    costs.extend(json.loads(line) for line in (contact_root.parent / "costs.jsonl").read_text().splitlines())
    contact_cost = read_json(ROOT / "bt08_contact_tree_cost_r06.json")
    assert contact_cost["passed"] and contact_cost["peak_tree_gpu_memory_mib"] > 0
    residency_path = ROOT / "bt08_residency_r05/observation_residency_report.json"
    residency = read_json(residency_path)
    assert residency["passed"] and residency["parent_and_children_share_gpu"]
    assert all(item["parent_state_unchanged"] for item in residency["scenarios"])
    common = {"identity": identity, "source_applicability": reference(bridge_path),
              "说明": "identity表示本阶段证据验证适用的最终目标；原始运行身份以所引用原报告为准，未改写旧运行。"}
    stages = {
        "unit_tests": {"passed": True, "report": reference(ROOT / "bt08_unit_report_r06.json")},
        "force_smoke": {"passed": True, "two_runtimes": True, "com_force_verified": True,
            "reset_isolated": True, "original_reports": {k:v for k,v in smoke_refs.items() if "force" in k}},
        "contact_smoke": {"passed": True, "two_runtimes": True, "measurement_verified": True,
            "reset_isolated": True, "original_reports": {k:v for k,v in smoke_refs.items() if "contact" in k}},
        "zero_force_regression": {"passed": True, "scenarios": ["S0", "S1", "S2"], "num_envs": 32,
            "episodes_per_env": 3, "policy_updates": 0, "original_report": reference(source_evidence / "a_zero_regression_comparison.json")},
        "training_resume": {"passed": True, "num_envs": 4096, "iterations": 10, "resumed_iterations": 1,
            "completed_iterations": 8011, "counters_consistent": True, "full_state_restored": True,
            "nonzero_force_learned": True, "reports": [reference(train_dir / "native_training_report.json"), reference(resume_dir / "native_training_report.json")]},
        "capacity": {"passed": True, "num_envs": 4096, "rollout_update": True, "stress_verified": True,
            "physics_warnings_absent": True, "peak_report": reference(peak_dir / "report.json"),
            "log_reports": [reference(path) for path in logs], "observation_residency": reference(residency_path)},
        "evaluation_baseline": {"passed": True, "force_scenarios": 14, "force_episodes": 1344,
            "contact_episodes": 96, "zero_control_episodes": 32, "policy_updates": 0, "cost_recorded": True,
            "force_reference_archives": references, "contact_reference_archive": str(contact_root),
            "costs": costs, "contact_tree_cost": contact_cost},
    }
    output.mkdir(exist_ok=False)
    for name, stage in stages.items():
        stage.update(copy.deepcopy(common))
        path = output / (name + ".json")
        path.write_text(json.dumps({"stage": name, "result": stage}, ensure_ascii=False, indent=2) + "\n")
        stage["artifact"] = reference(path)
    evidence = {"schema_version": 1, "identity": identity,
        "formal_config_sha256": identity["learning_config_sha256"], "parent_sha256": parent_metadata["sha256"],
        "manifest_sha256": {name: hashlib.sha256(json.dumps(read_json(SOURCE / "configs" / filename), sort_keys=True).encode()).hexdigest()
            for name, filename in (("force", "v3_b_initial_v1.json"), ("contact", "v3_b_contact_v1.json"), ("zero_force", "v3_a_initial_v1.json"))},
        "stages": stages}
    for filename, report in (("start_evidence.json", evidence), ("formal_config_snapshot.json", snapshot),
                             ("target_identity.json", identity), ("runtime.json", runtime),
                             ("parent_metadata.json", parent_metadata), ("baseline_summaries.json", baseline)):
        (output / filename).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"completed": True, "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

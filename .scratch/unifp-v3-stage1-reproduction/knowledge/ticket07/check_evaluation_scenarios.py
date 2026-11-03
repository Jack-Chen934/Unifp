"""Exercise real short-checkpoint scenario collection without claiming A8000."""

from collections import Counter
import json
from pathlib import Path
from typing import Any, Dict

import isaacgym  # noqa: F401
import torch
from legged_gym.scripts.evaluate_v3 import evaluate_scenario
from legged_gym.scripts.v3_args import get_v3_args
from legged_gym.b2_gym_learn.ppo_cse_pf.v3_evaluation import (
    EVALUATION_SCENARIOS, summarize_scenario_assessments, validate_a8000_checkpoint,
    validate_evaluation_manifest,
)
from legged_gym.b2_gym_learn.ppo_cse_pf.v3_run_identity import build_run_identity, file_sha256


def main() -> None:
    """Check sequential simulators, raw evidence, quotas, and exact-boundary rejection.

    Raises:
        AssertionError: Scenario configuration or captured evidence violates A10.
        OSError: Raw evidence cannot be written.

    Note:
        Uses the real one-update model. Development reports keep its actual progress
        and have no approved assessment; the formal CLI's A8000 guard stays intact.
    """
    args = get_v3_args("三组评估开发检查，非A8000效果验收")
    project_root = Path.cwd()
    checkpoint = project_root.parent / "runs/smoke32/checkpoints/model_000001.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    manifest = json.loads((project_root / "configs/v3_a_initial_v1.json").read_text())
    validate_evaluation_manifest(manifest)
    active_identity = build_run_identity(project_root, payload["config_snapshot"], payload["identity"]["runtime"])
    assert active_identity == payload["identity"]
    try:
        validate_a8000_checkpoint(payload, active_identity)
    except ValueError:
        short_checkpoint_rejected = True
    else:
        short_checkpoint_rejected = False
    assert short_checkpoint_rejected
    # This is an explicitly ungraded development manifest, not the frozen formal file.
    manifest["checkpoint_iteration"] = payload["progress"]["completed_iterations"]
    manifest["thresholds"] = None
    manifest["threshold_version"] = None
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=False)
    reports: Dict[str, Dict[str, Any]] = {}
    for scenario_name in EVALUATION_SCENARIOS:
        scenario_dir = output_root / scenario_name
        scenario_dir.mkdir()
        summary, snapshot = evaluate_scenario(
            args, payload, manifest, scenario_name=scenario_name, output_dir=scenario_dir)
        assert summary["checkpoint_iteration"] == 1 and summary["policy_updates"] == 0
        assert summary["evidence_complete"] and summary["episodes"]["completed"] == 96
        episodes = [json.loads(line) for line in (scenario_dir / "episodes.jsonl").read_text().splitlines()]
        assert Counter(row["env_id"] for row in episodes) == Counter({index: 3 for index in range(32)})
        trace_count = 0
        pushes = 0
        nonzero_command_rows = 0
        with (scenario_dir / "policy_trace.jsonl").open(encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                trace_count += 1
                pushes += row["base_push_applied"]
                nonzero_command_rows += any(value != 0 for value in row["commands_body"])
                assert row["scenario_name"] == scenario_name and row["completed_iterations"] == 1
                for key in ("end_effector_force_command_yaw_n", "external_end_effector_force_yaw_n"):
                    assert all(value == 0 for value in row[key])
        assert nonzero_command_rows == 0 if scenario_name == "S0" else nonzero_command_rows > 0
        if scenario_name != "S2":
            assert pushes == 0
        assert snapshot["environment"]["domain_rand"]["push_robots"] == (scenario_name == "S2")
        summary.update({"用途": "真实1轮模型的开发场景验证，非A8000验收", "trace_rows": trace_count,
                        "push_rows": pushes, "nonzero_base_command_rows": nonzero_command_rows,
                        "checkpoint_sha256": file_sha256(checkpoint)})
        for name, value in (("summary", summary), ("effective_config", snapshot)):
            (scenario_dir / f"{name}.json").write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        reports[scenario_name] = summary
    result = {"passed": True, "开发检查通过不等于效果通过": True,
              "short_checkpoint_rejected_by_a8000_guard": short_checkpoint_rejected,
              "checkpoint_sha256": file_sha256(checkpoint),
              "overall": summarize_scenario_assessments(reports), "scenarios": reports}
    (output_root / "verified_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

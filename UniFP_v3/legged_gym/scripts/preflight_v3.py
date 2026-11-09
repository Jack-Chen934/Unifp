"""Evaluate saved A-start evidence without starting simulation or training."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from legged_gym.b2_gym_learn.ppo_cse_pf.v3_preflight import assess_a_start_gate, assess_b_start_gate, validate_stage_artifacts
from legged_gym.b2_gym_learn.ppo_cse_pf.v3_run_identity import build_run_identity


def main() -> int:
    """Read evidence inputs, write the gate report, and return its readiness.

    Returns:
        Zero when all required evidence is present; one when training is blocked.

    Raises:
        FileExistsError: The report destination already exists.
        OSError: An input or output file cannot be accessed.
        ValueError: An input document is invalid.

    Note:
        Creates only the requested report and parent directory. Runtime inputs
        are declared records; formal training checks its actual runtime again.
    """
    parser = argparse.ArgumentParser(description="只读检查V3 A阶段长训启动证据")
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--config_snapshot", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--contact_manifest", type=Path)
    parser.add_argument("--zero_force_manifest", type=Path)
    parser.add_argument("--parent_metadata", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"启动门控报告已存在：{args.output}")

    evidence: Dict[str, Any] = json.loads(args.evidence.read_text(encoding="utf-8"))
    snapshot: Dict[str, Any] = json.loads(args.config_snapshot.read_text(encoding="utf-8"))
    runtime: Dict[str, Any] = json.loads(args.runtime.read_text(encoding="utf-8"))
    part = snapshot["environment"]["run"]["reproduction_part"]
    if part not in {"A", "B"} or snapshot["environment"]["run"]["run_purpose"] != "formal":
        raise ValueError("启动门控需要显式的A／B阶段formal配置")
    # Recompute content from this checkout instead of trusting a saved identity.
    source_root = Path(__file__).resolve().parents[2]
    identity = build_run_identity(source_root, snapshot, runtime)
    manifest: Dict[str, Any] = json.loads(args.manifest.read_text(encoding="utf-8"))
    if part == "B":
        if not args.contact_manifest or not args.zero_force_manifest or not args.parent_metadata:
            raise ValueError("B启动门控需要接触／零力manifest和实际父元数据")
        report = assess_b_start_gate(evidence, identity,
            json.loads(args.parent_metadata.read_text(encoding="utf-8")), manifest,
            json.loads(args.contact_manifest.read_text(encoding="utf-8")),
            json.loads(args.zero_force_manifest.read_text(encoding="utf-8")),
            formal_num_envs=snapshot["environment"]["env"]["num_envs"])
    else:
        report = assess_a_start_gate(evidence, identity, manifest,
                                     formal_num_envs=snapshot["environment"]["env"]["num_envs"])
        report["blockers"].extend(validate_stage_artifacts(evidence))
    report["ready"] = not report["blockers"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        output.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

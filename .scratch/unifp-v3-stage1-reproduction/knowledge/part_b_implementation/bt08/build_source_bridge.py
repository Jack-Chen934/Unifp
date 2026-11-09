"""核验旧原生证据与最终源码的适用关系，不改写任何旧报告。"""

import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Dict


def definitions(path: Path) -> Dict[str, str]:
    """Index top-level functions and individual class methods by AST content.

    Args:
        path: Existing Python source in an immutable validation checkout.

    Returns:
        Qualified definition names mapped to location-independent syntax trees.
    """
    result = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result[node.name] = ast.dump(node, include_attributes=False)
        elif isinstance(node, ast.ClassDef):
            for method in node.body:
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    result[node.name + "." + method.name] = ast.dump(method, include_attributes=False)
    return result


def main() -> None:
    """Bind unchanged physical/evaluation paths to original and final hashes.

    Raises:
        AssertionError: Any source change exceeds the independently reviewed set.

    Note:
        Writes a new applicability report only. Original runtime, source hashes
        and results remain preserved and are never relabeled as fresh execution.
    """
    root = Path("/mnt/data_8THDD/myl_unifp_v3_stage1/validation/b_stage_20261107_r01")
    final = root / "source_r06/UniFP_v3"
    changes_allowed = {
        "legged_gym/b2_gym_learn/ppo_cse_pf/v3_evaluation.py": {"summarize_force_evaluation"},
        "legged_gym/envs/v3/v3_force_evaluation.py": {"V3ForceEvaluation._build_policy_trace"},
        "legged_gym/b2_gym_learn/ppo_cse_pf/v3_preflight.py": {"validate_stage_artifacts", "assess_b_start_gate"},
        "legged_gym/scripts/preflight_v3.py": {"main"},
        "legged_gym/scripts/train_v3.py": {"main"},
        "legged_gym/scripts/evaluate_v3_contact.py": {"main", "evaluate_contact_groups"},
    }
    report: Dict[str, Any] = {"passed": True, "final_source": str(final), "versions": {},
        "说明": "旧G1/G3及A零力路径不执行受力汇总／随机力trace或启动门控。r04→r06只改变接触评估入口的进程组织及封存目录移动权限；真实训练／恢复／压力／14组受力调用链字节相同，接触默认入口以r06全配额重测。不同原生设备身份原样保留，报告声明对目标配置的适用性，不声称旧运行发生在GPU1。"}
    for version in ("source", "source_r02", "source_r03", "source_r04", "source_r05"):
        previous = root / version / "UniFP_v3"
        before_manifest, after_manifest, changes = {}, {}, []
        for path in sorted((final / "legged_gym").rglob("*.py")):
            relative = str(path.relative_to(final))
            before = previous / relative
            assert before.is_file()
            before_manifest[relative] = hashlib.sha256(before.read_bytes()).hexdigest()
            after_manifest[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            if before_manifest[relative] != after_manifest[relative]:
                assert relative in changes_allowed, relative
                old_definitions, new_definitions = definitions(before), definitions(path)
                names = sorted(name for name in old_definitions.keys() | new_definitions.keys()
                               if old_definitions.get(name) != new_definitions.get(name))
                assert set(names) <= changes_allowed[relative], (relative, names)
                changes.append({"path": relative, "changed_definitions": names,
                    "before_sha256": before_manifest[relative], "after_sha256": after_manifest[relative]})
        # Whole-file equality covers shared physics, layout, force application and control.
        for relative in ("legged_gym/envs/v3/legged_robot_v3.py",
                         "legged_gym/envs/v3/v3_contact_evaluation.py",
                         "legged_gym/envs/v3/v3_pos_force_config.py",
                         "legged_gym/b2_gym_learn/ppo_cse_pf/v3_on_policy_runner.py",
                         "legged_gym/b2_gym_learn/ppo_cse_pf/v3_ppo.py"):
            assert before_manifest[relative] == after_manifest[relative]
        report["versions"][version] = {"original_manifest": before_manifest,
            "final_manifest": after_manifest, "changed_paths": changes}
    (root / "bt08_source_bridge_r06.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": True, "versions": list(report["versions"])}))


if __name__ == "__main__":
    main()

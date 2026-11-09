"""在独立GPU进程补齐冻结12000评估，不修改训练与有效零力结果。"""

import datetime
import json
import os
from pathlib import Path
import subprocess
import sys


SOURCE_ROOT = Path("/mnt/data_8THDD/myl_unifp_v3_stage1/validation/b_stage_20261107_r01/source_resume12000_20261108/UniFP_v3")
VALIDATION_ROOT = Path("/mnt/data_8THDD/myl_unifp_v3_stage1/validation/b_diagnostics_20261108_r01")
OUTPUT_ROOT = Path("/mnt/data_8THDD/myl_unifp_v3_stage1/evaluations/b12000_supplement_20261108_r01")
CHECKPOINT = Path("/mnt/data_8THDD/myl_unifp_v3_stage1/runs/b32000_seed1_v3_v6r2_from_a8000_20261108/checkpoints/model_012000.pt")
EVIDENCE = SOURCE_ROOT.parents[1] / "resume_evidence12000_20261108/start_evidence.json"
evidence = json.loads(EVIDENCE.read_text())
reference_root = Path(evidence["stages"]["evaluation_baseline"]["force_reference_archives"]["R2"])
reference_manifest = reference_root / "R2/evaluation_manifest.json"
original_manifest = json.loads(reference_manifest.read_text())
current_manifest = json.loads((SOURCE_ROOT / "configs/v3_b_initial_v1.json").read_text())
assert {key: value for key, value in original_manifest.items() if key != "observation_schedule"} == {
    key: value for key, value in current_manifest.items() if key != "observation_schedule"
}
OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
common = ["--reproduction_part", "B", "--run_purpose", "evaluation", "--checkpoint", str(CHECKPOINT),
          "--num_envs", "32", "--sim_device", "cuda:0", "--rl_device", "cuda:0", "--pipeline", "gpu",
          "--num_threads", "4", "--headless"]
commands = [
    ("R2", [sys.executable, "-m", "legged_gym.scripts.evaluate_v3", *common,
            "--seed", str(original_manifest["seed"]), "--protocol", "force_tracking", "--manifest", str(reference_manifest),
            "--scenario", "R2", "--output_dir", str(OUTPUT_ROOT / "R2"), "--video", "--reference_archive", str(reference_root)]),
    ("contact", [sys.executable, "-m", "legged_gym.scripts.evaluate_v3_contact", *common,
                 "--seed", "1001", "--protocol", "fixed_wall_contact", "--manifest", str(SOURCE_ROOT / "configs/v3_b_contact_v1.json"),
                 "--output_dir", str(OUTPUT_ROOT / "contact"), "--video"]),
]
assert json.loads((SOURCE_ROOT / "configs/v3_b_contact_v1.json").read_text())["seed"] == 1001
status = {"pid": os.getpid(), "gpu": os.environ["CUDA_VISIBLE_DEVICES"], "source": str(SOURCE_ROOT),
          "checkpoint": str(CHECKPOINT), "output": str(OUTPUT_ROOT), "commands": commands, "results": [],
          "started_at": datetime.datetime.now().astimezone().isoformat(), "state": "running"}
(VALIDATION_ROOT / "assessment_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
for name, command in commands:
    started_at = datetime.datetime.now().astimezone().isoformat()
    with (VALIDATION_ROOT / f"{name}.console.log").open("x") as console:
        result = subprocess.run(command, cwd=SOURCE_ROOT, stdout=console, stderr=subprocess.STDOUT)
    status["results"].append({"name": name, "returncode": result.returncode, "started_at": started_at,
                              "ended_at": datetime.datetime.now().astimezone().isoformat()})
    (VALIDATION_ROOT / "assessment_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
status["state"] = "completed" if all(row["returncode"] == 0 for row in status["results"]) else "error"
(VALIDATION_ROOT / "assessment_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")

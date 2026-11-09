"""在空闲容量的独立GPU上逐个运行七个无更新诊断场景。"""

import datetime
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = Path("/mnt/data_8THDD/myl_unifp_v3_stage1/validation/b_stage_20261107_r01/source_resume12000_20261108/UniFP_v3")
CHECKPOINT_ROOT = Path("/mnt/data_8THDD/myl_unifp_v3_stage1/runs/b32000_seed1_v3_v6r2_from_a8000_20261108/checkpoints")
status = {"state": "running", "pid": __import__("os").getpid(), "results": []}
(ROOT / "probe_status.json").write_text(json.dumps(status, indent=2) + "\n")
conditions = [(iteration, mode) for iteration in (9000, 10000, 12000) for mode in ("mean", "sample")]
conditions.append((12000, "sample_std9000"))
for iteration, mode in conditions:
    name = f"probe_{iteration}_{mode}"
    command = [sys.executable, str(ROOT / "probe_policy.py"), "--checkpoint", str(CHECKPOINT_ROOT / f"model_{iteration:06d}.pt"),
               "--output", str(ROOT / name), "--mode", mode, "--num-envs", "128", "--seconds", "20"]
    started_at = datetime.datetime.now().astimezone().isoformat()
    with (ROOT / f"{name}.console.log").open("x") as console:
        result = subprocess.run(command, cwd=SOURCE_ROOT, stdout=console, stderr=subprocess.STDOUT)
    status["results"].append({"name": name, "returncode": result.returncode, "started_at": started_at,
                              "ended_at": datetime.datetime.now().astimezone().isoformat()})
    (ROOT / "probe_status.json").write_text(json.dumps(status, indent=2) + "\n")
    if result.returncode != 0:
        break
status["state"] = "completed" if len(status["results"]) == len(conditions) and all(row["returncode"] == 0 for row in status["results"]) else "error"
(ROOT / "probe_status.json").write_text(json.dumps(status, indent=2) + "\n")

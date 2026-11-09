"""串行完成最终观察器的10＋1轮、峰值压力及接触基准。"""

import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    """Execute approved native checks in order on the explicitly selected GPU.

    Raises:
        subprocess.CalledProcessError: Any engineering stage exits unsuccessfully.

    Note:
        Writes new run directories only. Does not launch formal B training.
    """
    root = Path("/mnt/data_8THDD/myl_unifp_v3_stage1/validation/b_stage_20261107_r01")
    source = root / "source_r04/UniFP_v3"
    parent = Path("/mnt/data_8THDD/myl_unifp_v3_stage1/runs/a8000_seed1_v3_v6r2_resume0767_20261105/checkpoints/model_008000.pt")
    common = ["--reproduction_part", "B", "--run_purpose", "preflight", "--seed", "1",
              "--num_envs", "4096", "--sim_device", "cuda:0", "--rl_device", "cuda:0",
              "--pipeline", "gpu", "--num_threads", "4", "--headless"]
    jobs = [
        ("bt08_train10_allrng_r04", [sys.executable, "tools/check_v3_b_training.py", *common,
            "--max_iterations", "10", "--transition_from_a", "--resume", str(parent),
            "--parent_sha256", "323a6688fe42559db1d0c545b778b456551ce0ac472f8169fec5ebce2d849dbc"]),
        ("bt08_resume1_allrng_r04", [sys.executable, "tools/check_v3_b_training.py", *common,
            "--max_iterations", "1", "--resume", str(root / "bt08_train10_allrng_r04/checkpoints/model_008010.pt")]),
        ("bt08_peak4096_r04b", [sys.executable, "tools/check_v3_b_capacity.py", *common,
            "--max_iterations", "10", "--checkpoint", str(parent)]),
    ]
    for name, command in jobs:
        print("开始 " + name, flush=True)
        with (root / (name + ".log")).open("x") as log:
            subprocess.run([*command, "--output_dir", str(root / name)], cwd=source,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        print("完成 " + name, flush=True)
    command = [sys.executable, str(root / "staging/run_baselines.py"), "--queue", "contact",
               "--source", str(source), "--output", str(root / "bt08_contact_r04"),
               "--checkpoint", str(parent)]
    with (root / "bt08_contact_r04.log").open("x") as log:
        subprocess.run(command, cwd=source, stdout=log, stderr=subprocess.STDOUT, check=True)
    print("完成接触与0N基准；未启动正式B训练", flush=True)


if __name__ == "__main__":
    main()

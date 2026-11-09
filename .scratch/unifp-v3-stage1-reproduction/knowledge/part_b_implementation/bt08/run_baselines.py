"""串行调用正式评估入口，记录全配额A父基准的真实成本。"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main() -> None:
    """Run one approved baseline queue without retries or quota changes.

    Raises:
        RuntimeError: An evaluator fails or evidence cannot be collected.
        FileExistsError: The queue output already exists.

    Note:
        Starts read-only evaluation subprocesses and records cost on the data
        volume. CUDA_VISIBLE_DEVICES selects an explicitly checked free GPU.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--queue", choices=("first", "second", "contact"), required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.source / "configs/v3_b_initial_v1.json").read_text())
    all_scenarios = list(manifest["scenarios"])
    scenarios = all_scenarios[:7] if args.queue == "first" else all_scenarios[7:] if args.queue == "second" else ["contact"]
    for scenario in scenarios:
        contact = scenario == "contact"
        command = [sys.executable, "-m", "legged_gym.scripts.evaluate_v3_contact" if contact else "legged_gym.scripts.evaluate_v3",
                   "--reproduction_part", "A", "--run_purpose", "evaluation", "--seed", "1001",
                   "--num_envs", "32", "--checkpoint", str(args.checkpoint),
                   "--protocol", "fixed_wall_contact" if contact else "force_tracking",
                   "--manifest", "configs/v3_b_contact_v1.json" if contact else "configs/v3_b_initial_v1.json",
                   "--output_dir", str(args.output / scenario), "--sim_device", "cuda:0", "--rl_device", "cuda:0",
                   "--pipeline", "gpu", "--num_threads", "4", "--headless", "--video"]
        if not contact:
            command.extend(["--scenario", scenario])
        started = time.perf_counter()
        peak_memory_mib = 0
        with (args.output / f"{scenario}.log").open("x") as log:
            process = subprocess.Popen(command, cwd=args.source, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                memory_rows = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], text=True)
                for row in memory_rows.splitlines():
                    fields = row.split(",")
                    if fields[0].strip() == str(process.pid) and fields[1].strip().isdigit():
                        peak_memory_mib = max(peak_memory_mib, int(fields[1]))
                time.sleep(5.)
        cost = {"scenario": scenario, "returncode": process.returncode,
                "wall_time_s": time.perf_counter() - started, "peak_process_gpu_memory_mib": peak_memory_mib,
                "output_bytes": sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file()),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
        with (args.output / "costs.jsonl").open("a") as output:
            output.write(json.dumps(cost) + "\n")
        print(json.dumps(cost), flush=True)
        if process.returncode:
            raise RuntimeError(f"评估失败，保留日志并停止本队列：{scenario}")


if __name__ == "__main__":
    main()

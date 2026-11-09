"""只读汇总已结束的B4000训练日志；不导入Gym，不修改远程运行。"""

import hashlib
import json
from pathlib import Path


EPISODE_PATH = Path(
    "/mnt/data_8THDD/myl_unifp_v3_stage1/runs/"
    "b32000_seed1_v3_v6r2_from_a8000_20261108/episodes.jsonl"
)
blocks = {}
source_digest = hashlib.sha256()
clock_min, clock_max, line_count = None, None, 0
for source_line in EPISODE_PATH.open("rb"):
    source_digest.update(source_line)
    episode = json.loads(source_line)
    line_count += 1
    # Episode records use the pre-update clock; iteration rows use the committed clock.
    committed_iteration = episode["completed_iterations"] + 1
    relative_iteration = committed_iteration - 8000
    clock_min = committed_iteration if clock_min is None else min(clock_min, committed_iteration)
    clock_max = committed_iteration if clock_max is None else max(clock_max, committed_iteration)
    if not 1 <= relative_iteration <= 4000:
        continue
    block_index = (relative_iteration - 1) // 100
    block = blocks.setdefault(block_index, {
        "start": block_index * 100 + 1, "end": (block_index + 1) * 100,
        "episodes": 0, "physical_failures": 0, "steps": 0,
        "error_step_sum_m": 0.0, "error_episode_sum_m": 0.0,
        "reward_return_sum": 0.0, "duration_sum_s": 0.0, "raw_step_sums": {},
    })
    steps = episode["steps"]
    error_mean_m = episode["errors"]["tcp_compensated_l2_m"]["mean"]
    block["episodes"] += 1
    block["physical_failures"] += episode["end_type"] == "physical_failure"
    block["steps"] += steps
    block["error_step_sum_m"] += steps * error_mean_m
    block["error_episode_sum_m"] += error_mean_m
    block["reward_return_sum"] += episode["reward_return"]
    block["duration_sum_s"] += episode["duration_s"]
    for metric_name, metric_value in episode["raw_reward_means"].items():
        block["raw_step_sums"][metric_name] = (
            block["raw_step_sums"].get(metric_name, 0.0) + steps * metric_value
        )

print(json.dumps({
    "source": str(EPISODE_PATH), "sha256": source_digest.hexdigest(),
    "line_count": line_count, "committed_clock_range": [clock_min, clock_max],
    "clock_correction": "episode.completed_iterations + 1 = iteration.completed_iterations",
    "blocks100": [blocks[key] for key in sorted(blocks)],
}, ensure_ascii=False, allow_nan=False, indent=2))

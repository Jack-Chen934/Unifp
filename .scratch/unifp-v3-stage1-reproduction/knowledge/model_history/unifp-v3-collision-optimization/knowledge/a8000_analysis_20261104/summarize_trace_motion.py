"""Summarize completed evaluation traces without changing sealed evidence."""

import json
from pathlib import Path

import numpy as np

# Use the frozen remote evaluation; write results to stdout outside its archive.
archive_root = Path('/mnt/data_8THDD/myl_unifp_v3_stage1/evaluations/a8000_seed1_v3_collision_v1_20261104_r02')
scenario_summaries = {}
for scenario_name in ('S0', 'S1', 'S2'):
    scenario_root = archive_root / scenario_name
    summary_path = scenario_root / 'summary.json'
    if not summary_path.exists():
        continue
    trace_path = scenario_root / 'policy_trace.jsonl'
    if not trace_path.exists():
        trace_path = next(scenario_root.rglob('*trace*.jsonl'))
    # Accumulate vector moments rather than retaining hundreds of MB of rows.
    sample_count = 0
    command_norm_sum = 0.0
    actual_norm_sum = 0.0
    command_square_sum = np.zeros(3, dtype=np.float64)
    command_actual_product_sum = np.zeros(3, dtype=np.float64)
    command_sum = np.zeros(3, dtype=np.float64)
    actual_sum = np.zeros(3, dtype=np.float64)
    actual_square_sum = np.zeros(3, dtype=np.float64)
    for line in trace_path.open(encoding='utf-8'):
        row = json.loads(line)
        # Exclude the first second only for this supplemental motion diagnostic.
        if row['time_s'] < 1.0:
            continue
        command = np.asarray(row['commands_body'], dtype=np.float64)
        actual = np.asarray((*row['base_linear_velocity_body_m_s'][:2], row['base_angular_velocity_body_rad_s'][2]), dtype=np.float64)
        sample_count += 1
        command_norm_sum += float(np.linalg.norm(command[:2]))
        actual_norm_sum += float(np.linalg.norm(actual[:2]))
        command_square_sum += command ** 2
        actual_square_sum += actual ** 2
        command_actual_product_sum += command * actual
        command_sum += command
        actual_sum += actual
    projection_gains = [float(cross / square) if square > 0.0 else None for cross, square in zip(command_actual_product_sum, command_square_sum)]
    correlations = []
    for index in range(3):
        covariance = command_actual_product_sum[index] - command_sum[index] * actual_sum[index] / sample_count
        command_variance = command_square_sum[index] - command_sum[index] ** 2 / sample_count
        actual_variance = actual_square_sum[index] - actual_sum[index] ** 2 / sample_count
        denominator = np.sqrt(max(0.0, command_variance * actual_variance))
        correlations.append(float(covariance / denominator) if denominator > 0.0 else None)
    scenario_summaries[scenario_name] = {
        'sample_count_after_first_second': sample_count,
        'mean_command_planar_speed_m_s': command_norm_sum / sample_count,
        'mean_actual_planar_speed_m_s': actual_norm_sum / sample_count,
        'command_to_actual_projection_gains_vx_vy_yaw': projection_gains,
        'command_to_actual_correlations_vx_vy_yaw': correlations,
    }
print(json.dumps(scenario_summaries, ensure_ascii=False, indent=2, allow_nan=False))

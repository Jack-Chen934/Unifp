"""Measure isolated candidate timing through the existing V3 runtime probe."""

import argparse
import json
from pathlib import Path
import sys

# Native Gym must be loaded before the runtime probe imports torch.
from isaacgym import gymapi


def main() -> None:
    """Apply diagnostic-only timing/gain settings and measure real runtime steps.

    Raises:
        RuntimeError: Native creation or indexed reset isolation fails.
        OSError: The report cannot be saved.

    Note:
        Changes config classes in this disposable process only. No optimizer,
        formal source file, or training checkpoint is modified.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--asset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--physics-step-factor', type=int, choices=(1, 2), default=1)
    parser.add_argument('--gain-scale', type=float, default=1.0)
    parser.add_argument('--num-envs', type=int, default=32)
    parser.add_argument('--steps', type=int, default=100)
    arguments = parser.parse_args()
    sys.path.insert(0, str(arguments.source_root))
    sys.path.insert(0, str(arguments.source_root / 'tools'))
    from legged_gym.envs.v3.v3_pos_force_config import V3PosForceCfg
    from probe_training_collision_runtime import run_probe

    # Preserve policy time while measuring the proposed inner-loop refinement.
    V3PosForceCfg.sim.dt /= arguments.physics_step_factor
    V3PosForceCfg.control.decimation *= arguments.physics_step_factor
    for joint_name in V3PosForceCfg.control.damping:
        if joint_name.endswith('_wheel_drive_joint'):
            V3PosForceCfg.control.damping[joint_name] *= arguments.gain_scale
    arguments.nominal = True
    arguments.physics_only = False
    report = run_probe(arguments)
    report['diagnostic_control'] = {
        'physics_dt_s': V3PosForceCfg.sim.dt,
        'decimation': V3PosForceCfg.control.decimation,
        'drive_gain_Nm_s_rad': 4.5477 * arguments.gain_scale,
        'limitations_zh': '仅独立进程短测；不包含PPO，不是最终身份验收。',
    }
    arguments.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in (
        'shape_count', 'step_mean_s', 'physical_failure_events',
        'movement_physical_failure_events', 'arm_physical_failure_events',
        'torch_peak_memory_bytes', 'diagnostic_control')}, ensure_ascii=False))


if __name__ == '__main__':
    main()

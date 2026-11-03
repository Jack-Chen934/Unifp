"""通过真实训练入口确认旧资产检查点在创建仿真前被拒绝。"""

import json
from pathlib import Path
import subprocess
import sys


def main() -> None:
    """Try a pinned old checkpoint and record its expected strict rejection.

    Returns:
        None.

    Raises:
        AssertionError: The old checkpoint is accepted or the output is created.
        OSError: The child command or evidence write fails.

    Note:
        Runs the real training entry, which must reject identity before Gym.
        Inherits the final validation environment and preserves all old outputs.
    """
    root = Path(__file__).resolve().parent
    source = root / 'source/UniFP_v3'
    output = root / 'rejected_old_resume'
    checkpoint = root.parent / 'ticket07_20261102_r03/runs/capacity4096/checkpoints/model_000001.pt'
    command = [sys.executable, '-m', 'legged_gym.scripts.train_v3',
               '--reproduction_part', 'A', '--run_purpose', 'development',
               '--seed', '1', '--num_envs', '4096', '--max_iterations', '1',
               '--resume', str(checkpoint), '--output_dir', str(output),
               '--sim_device', 'cuda:0', '--rl_device', 'cuda:0', '--headless']
    result = subprocess.run(command, cwd=source, text=True, capture_output=True, check=False)
    log = result.stdout + result.stderr
    rejected = (result.returncode != 0 and
                'Resume learning/source/asset/schema/runtime identity mismatch' in log and
                not output.exists() and 'Physics Engine:' not in log)
    (root / 'old_resume_rejection.console.log').write_text(log, encoding='utf-8')
    report = {'passed': rejected, 'returncode': result.returncode,
              'checkpoint': str(checkpoint), 'output_created': output.exists(),
              'command': command}
    (root / 'old_resume_rejection.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    assert rejected, log


if __name__ == '__main__':
    main()

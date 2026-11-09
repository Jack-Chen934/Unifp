"""修正临时目录所在磁盘后，从归档错误处继续已授权对照。"""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from run_experiment import ORIGIN, evaluate, execute, summarize, train_command, write_status


def main() -> None:
    """Preserve failed publication evidence and resume the approved remaining stages.

    Raises:
        AssertionError: Prior stages, output device or original failure differ.
        OSError: Old evidence cannot be preserved or new status cannot be written.
        RuntimeError: A remaining child exits with an execution error.

    Note:
        Changes only TMPDIR and orchestration state. Reuses completed parent S0/S1/S2
        and R2 archives, reruns parent contact/zero-control in a fresh destination,
        then runs the original two 200-update trials and all terminal evaluations.
        Never changes training launchers, hyperparameters or the formal process.
    """
    root = Path(__file__).resolve().parent
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '3'
    assert Path(tempfile.gettempdir()).stat().st_dev == root.stat().st_dev
    previous = json.loads((root / 'status.json').read_text())
    assert previous['state'] == 'error'
    assert previous['active_stage'] == 'evaluate_parent_9000_contact'
    assert all(row.get('returncode') == 0 for row in previous['stages'][:-1])
    assert previous['stages'][-1]['returncode'] == 1
    assert not (root / 'control_200').exists() and not (root / 'no_entropy_200').exists()
    (root / 'status_before_archive_resume.json').write_text(json.dumps(previous, ensure_ascii=False, indent=2) + '\n')
    failed = root / 'evaluations/parent_9000/contact'
    preserved = root / 'evaluations/parent_9000/contact_failed_publish_r01'
    failed.rename(preserved)
    state: dict[str, Any] = {**previous, 'pid': os.getpid(), 'state': 'running',
        'resumed_at': datetime.now().astimezone().isoformat(), 'active_stage': None,
        'previous_error_preserved': 'status_before_archive_resume.json',
        'publication_fix': {'TMPDIR': tempfile.gettempdir(), 'same_filesystem_verified': True}}
    state.pop('error', None)
    state.pop('error_type', None)
    state.pop('ended_at', None)
    write_status(root, state)
    try:
        execute('evaluate_parent_9000_contact_r02', previous['stages'][-1]['command'], root, state)
        summary = json.loads((root / 'evaluations/parent_9000/contact/summary.json').read_text())
        assert summary['policy_updates'] == 0
        for group, coefficient in (('control', .01), ('no_entropy', 0.)):
            execute(f'train_{group}_200', train_command(root, f'{group}_200', coefficient, 200, ORIGIN), root, state)
            evaluate(root / f'{group}_200/checkpoints/model_009200.pt', f'{group}_9200', root, state)
        summarize(root)
        state.update(state='completed', active_stage=None, ended_at=datetime.now().astimezone().isoformat())
    except Exception as error:
        state.update(state='error', error_type=type(error).__name__, error=str(error),
                     ended_at=datetime.now().astimezone().isoformat())
        raise
    finally:
        write_status(root, state)


if __name__ == '__main__':
    main()

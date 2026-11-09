"""在GPU3串行执行四条独立诊断轨迹，保留完成或失败状态。"""

import datetime
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
SOURCE = Path('/mnt/data_8THDD/myl_unifp_v3_stage1/validation/b_stage_20261107_r01/source_resume12000_20261108/UniFP_v3')
CHECKPOINTS = Path('/mnt/data_8THDD/myl_unifp_v3_stage1/runs/b32000_seed1_v3_v6r2_from_a8000_20261108/checkpoints')
status = {'state': 'running', 'pid': os.getpid(), 'gpu': os.environ['CUDA_VISIBLE_DEVICES'], 'results': []}
(ROOT / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
for iteration, mode in ((9000, 'mean'), (9000, 'sample'), (12000, 'mean'), (12000, 'sample')):
    name = f'check_{iteration}_{mode}'
    started = datetime.datetime.now().astimezone().isoformat()
    with (ROOT / f'{name}.console.log').open('x') as console:
        result = subprocess.run([sys.executable, str(ROOT / 'collect_execution.py'), '--checkpoint',
            str(CHECKPOINTS / f'model_{iteration:06d}.pt'), '--mode', mode, '--output', str(ROOT / name)],
            cwd=SOURCE, stdout=console, stderr=subprocess.STDOUT)
    status['results'].append({'name': name, 'returncode': result.returncode,
        'started_at': started, 'ended_at': datetime.datetime.now().astimezone().isoformat()})
    (ROOT / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
    if result.returncode:
        break
status['state'] = 'completed' if len(status['results']) == 4 and all(row['returncode'] == 0 for row in status['results']) else 'error'
(ROOT / 'status.json').write_text(json.dumps(status, indent=2) + '\n')

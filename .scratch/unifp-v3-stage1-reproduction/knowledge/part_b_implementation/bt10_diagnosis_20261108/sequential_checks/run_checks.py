"""在独立GPU依次执行三个检查点的诊断，不修改正式训练。"""

import datetime
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
SOURCE = Path('/mnt/data_8THDD/myl_unifp_v3_stage1/validation/b_stage_20261107_r01/source_resume12000_20261108/UniFP_v3')
CHECKPOINTS = Path('/mnt/data_8THDD/myl_unifp_v3_stage1/runs/b32000_seed1_v3_v6r2_from_a8000_20261108/checkpoints')
status = {'state':'running','pid':os.getpid(),'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'results':[]}
(ROOT/'status.json').write_text(json.dumps(status,indent=2)+'\n')
for iteration in (9000,10000,12000):
    started = datetime.datetime.now().astimezone().isoformat()
    with (ROOT/f'check_{iteration}.console.log').open('x') as console:
        result = subprocess.run([sys.executable,str(ROOT/'collect_checks.py'),'--checkpoint',
            str(CHECKPOINTS/f'model_{iteration:06d}.pt'),'--output',str(ROOT/f'check_{iteration}')],
            cwd=SOURCE,stdout=console,stderr=subprocess.STDOUT)
    status['results'].append({'iteration':iteration,'returncode':result.returncode,
        'started_at':started,'ended_at':datetime.datetime.now().astimezone().isoformat()})
    (ROOT/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    if result.returncode:
        break
status['state'] = 'completed' if len(status['results'])==3 and all(row['returncode']==0 for row in status['results']) else 'error'
(ROOT/'status.json').write_text(json.dumps(status,indent=2)+'\n')

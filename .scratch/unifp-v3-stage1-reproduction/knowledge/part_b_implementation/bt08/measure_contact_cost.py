"""记录接触评估父进程及其子进程的合计GPU显存。"""

import json
from pathlib import Path
import subprocess
import time
from typing import Dict, Set


def process_parents() -> Dict[int, int]:
    """Read process ancestry without inspecting unrelated process data.

    Returns:
        Operating-system process IDs mapped to their parent IDs.
    """
    rows = subprocess.check_output(['ps', '-eo', 'pid=,ppid='], text=True)
    return {int(fields[0]): int(fields[1]) for row in rows.splitlines() if len(fields := row.split()) == 2}


def main() -> None:
    """Sample the active contact queue's full process tree until it exits.

    Raises:
        AssertionError: The explicitly named project queue is not currently active.

    Note:
        Reads process/GPU status only; never changes process state or resources.
    """
    root = Path('/mnt/data_8THDD/myl_unifp_v3_stage1/validation/b_stage_20261107_r01')
    rows = subprocess.check_output(['ps', '-u', 'myl', '-o', 'pid=,args='], text=True)
    candidates = [int(row.split()[0]) for row in rows.splitlines()
                  if 'run_baselines.py --queue contact' in row and 'bt08_contact_r06' in row]
    assert len(candidates) == 1, candidates
    queue_pid = candidates[0]
    peak_mib, samples = 0, 0
    while queue_pid in (parents := process_parents()):
        descendants: Set[int] = {queue_pid}
        while (expanded := descendants | {pid for pid, parent in parents.items() if parent in descendants}) != descendants:
            descendants = expanded
        memory_rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                               '--format=csv,noheader,nounits'], text=True)
        memory_mib = sum(int(fields[1]) for row in memory_rows.splitlines()
                         if len(fields := row.split(',')) == 2 and fields[0].strip().isdigit()
                         and int(fields[0]) in descendants and fields[1].strip().isdigit())
        peak_mib = max(peak_mib, memory_mib)
        samples += 1
        time.sleep(5.)
    report = {'passed': peak_mib > 0, 'queue_pid': queue_pid, 'peak_tree_gpu_memory_mib': peak_mib,
              'samples': samples, '说明': '自采样开始的实测峰值，包含封装父进程及其全部子进程；不含其他实验／训练驻留。队列启动前段未采样，不宣称连续全程峰值。'}
    (root / 'bt08_contact_tree_cost_r06.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()

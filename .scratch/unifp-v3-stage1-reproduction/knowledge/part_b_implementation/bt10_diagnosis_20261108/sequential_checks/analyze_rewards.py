"""按完整episode时长加权核对训练奖励贡献，不改变奖励。"""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
blocks = json.loads((ROOT.parents[1]/'official_comparison_20261108/v3_episode_blocks100.json').read_text())
metadata = json.loads((ROOT/'episode_metadata.json').read_text())
weights = metadata['configured_reward_weights']
windows = []
for end in (9000,10000,12000):
    selected = [row for row in blocks['blocks100'] if end-8000-199 <= row['start'] and row['end'] <= end-8000]
    assert len(selected)==2
    steps = sum(row['steps'] for row in selected)
    contributions = {name: weights[name]*sum(row['raw_step_sums'][name] for row in selected)/steps for name in weights}
    return_rate = sum(row['reward_return_sum'] for row in selected)/sum(row['duration_sum_s'] for row in selected)
    error = abs(sum(contributions.values())-return_rate)
    assert error < 1e-4
    windows.append({'start':end-199,'end':end,'episodes':sum(row['episodes'] for row in selected),
        'steps':steps,'reward_rate':return_rate,'reward_accounting_abs_error':error,
        'tcp_l2_cm':100*sum(row['error_step_sum_m'] for row in selected)/steps,
        'failure_fraction':sum(row['physical_failures'] for row in selected)/sum(row['episodes'] for row in selected),
        'contribution_per_second':contributions})
output = {'windows':windows,'weights':weights,'episode_source':blocks['source'],
    'episode_sha256':blocks['sha256'], 'line_count':blocks['line_count'],
    '说明':'每窗口由结束在该窗口的完整episode按步数加权，episode可能跨窗口；不等同策略梯度贡献。熵项独立于环境奖励。'}
(ROOT/'reward_check.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
for row in windows:
    print(row['end'], row['reward_rate'], row['tcp_l2_cm'], row['contribution_per_second'])

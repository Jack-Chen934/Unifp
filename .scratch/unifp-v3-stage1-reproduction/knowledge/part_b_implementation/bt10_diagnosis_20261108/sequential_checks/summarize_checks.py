"""汇总顺序诊断数据并生成中文图表，不运行仿真。"""

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


ROOT=Path(__file__).resolve().parent
font_path='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family']=font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams['axes.unicode_minus']=False
latent=json.loads((ROOT/'latent_crosscheck.json').read_text())
rewards=json.loads((ROOT/'reward_check.json').read_text())
summaries=[json.loads(path.read_text()) for path in sorted(ROOT.glob('check_*/summary.json'))]
updates=[{'iteration':int(path.parent.name.split('_')[1]),**json.loads(path.read_text())}
         for path in sorted(ROOT.glob('check_*/update_observation.json'))]
summaries.sort(key=lambda row:row['iteration'])
updates.sort(key=lambda row:row['iteration'])
update_summary=[]
for report in updates:
    row={'iteration':report['iteration']}
    for kind in ('main','auxiliary'):
        events=[event for event in report['events'] if event['kind']==kind]
        row[kind]={'steps':len(events),
            'mean_action_rms_change':float(np.mean([event['mean_action_rms_change'] for event in events])),
            'max_action_rms_change':float(np.max([event['mean_action_rms_change'] for event in events])),
            'mean_gaussian_kl':float(np.mean([event['gaussian_kl_mean'] for event in events])),
            'mean_held_force_mae_change_n':float(np.mean([np.mean(event['force_mae_after_n'])-np.mean(event['force_mae_before_n']) for event in events]))}
        if kind=='auxiliary':
            cosines=np.array([event['encoder_gradient_cosine_with_preceding_main'] for event in events])
            row[kind]['adjacent_gradient_cosine_mean']=float(cosines.mean())
            row[kind]['adjacent_negative_count']=int((cosines<0).sum())
    update_summary.append(row)
(ROOT/'update_summary.json').write_text(json.dumps(update_summary,ensure_ascii=False,indent=2)+'\n')

figure,axes=plt.subplots(1,3,figsize=(14,4))
reference=next(row for row in latent['datasets'] if row['probe']=='probe_9000_mean')
for component,key in [('总变化','total_action_rms'),('新Actor下换Encoder','encoder_change_with_new_actor_rms'),('旧Encoder下换Actor','actor_change_with_old_encoder_rms')]:
    axes[0].plot([row['iteration'] for row in reference['comparisons']],
        [np.sqrt(np.mean(np.square(row[key]))) for row in reference['comparisons']],marker='o',label=component)
axes[0].set(title='固定9000真实观测：模块变化',xlabel='累计迭代',ylabel='组RMS的均方根（动作单位）')
for kind,label in [('main','PPO主更新'),('auxiliary','辅助更新')]:
    axes[1].plot([row['iteration'] for row in update_summary],[row[kind]['mean_action_rms_change'] for row in update_summary],marker='o',label=label)
axes[1].set(title='每次Adam更新引起的动作变化',xlabel='累计迭代',ylabel='固定观测动作RMS变化')
for row in update_summary:
    axes[2].bar(str(row['iteration']),row['auxiliary']['adjacent_gradient_cosine_mean'])
axes[2].set(title='相邻主／辅助Encoder梯度',xlabel='累计迭代',ylabel='梯度cos均值（相邻参数点）')
for axis in axes:
    axis.grid(alpha=.2)
for axis in axes[:2]:
    axis.legend(fontsize=8)
figure.tight_layout();figure.savefig(ROOT/'latent_update_checks.png',dpi=160);plt.close(figure)

figure,axes=plt.subplots(1,2,figsize=(13,4))
names=['tracking_ee_force_world','tracking_lin_vel','tracking_ang_vel','alive','action_rate_steering','action_rate_drive','action_rate_arm','dof_pos_limits']
labels=['末端跟踪','底盘线速','底盘偏航','存活','转向动作变化','驱动动作变化','臂动作变化','关节位置边界']
for index,row in enumerate(rewards['windows']):
    axes[0].bar(np.arange(len(names))+(index-1)*.25,[row['contribution_per_second'][name] for name in names],width=.25,label=str(row['end']))
axes[0].set(xticks=np.arange(len(names)),xticklabels=labels,ylabel='每秒奖励贡献',title='训练完整episode按步加权')
axes[0].tick_params(axis='x',labelrotation=25)
axes[1].plot([row['end'] for row in rewards['windows']],[row['reward_rate'] for row in rewards['windows']],marker='o',label='总环境奖励率')
axes[1].set(xlabel='窗口结束迭代',ylabel='奖励／秒',title='总奖励与控制效果同步退化')
for axis in axes:
    axis.grid(axis='y',alpha=.2);axis.legend(fontsize=8)
figure.tight_layout();figure.savefig(ROOT/'reward_checks.png',dpi=160);plt.close(figure)

figure,axes=plt.subplots(1,2,figsize=(13,4))
for group,label in enumerate(('转向','驱动','机械臂')):
    start,end=((0,4),(4,8),(8,14))[group]
    axes[0].plot([row['iteration'] for row in summaries],[np.mean(row['saturation_fraction_by_action'][start:end])*100 for row in summaries],marker='o',label=label)
    axes[1].plot([row['iteration'] for row in summaries],[max(row['longest_saturation_s_by_action'][start:end]) for row in summaries],marker='o',label=label)
axes[0].set(xlabel='累计迭代',ylabel='全部物理子步限幅占比（%）',title='同B分布独立采样，128×20秒')
axes[1].set(xlabel='累计迭代',ylabel='最长连续限幅（秒）',title='单环境单关节最长持续时间')
for axis in axes:
    axis.legend();axis.grid(alpha=.2)
figure.tight_layout();figure.savefig(ROOT/'saturation_checks.png',dpi=160);plt.close(figure)
print(json.dumps({'update_summary':update_summary},ensure_ascii=False),flush=True)

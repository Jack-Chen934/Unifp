"""Authorized, fixed-budget Ranger force continuation. Frozen native algorithms."""
import argparse
import copy
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from collections import deque

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
BASE = AUDIT / 'gym_ranger_fresh512_fixedcom_20261028_v1'
OLD = AUDIT / 'gym_ranger_force_stage_20261028_v1'
ROOT = BASE.parents[3]
REPORT = ROOT / 'PROJECT_REPORT.md'
PLAN = json.loads((HERE / 'plan.json').read_text())
SOURCE = Path(PLAN['source_checkpoint'])
PYTHON = '/home/sorry/.cache/unifp/isaacgym_preview4/torch231/bin/python'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.%s.tmp' % os.getpid())
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def read(path):
    return json.loads(Path(path).read_text())


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def protect():
    for path, expected in PLAN['protected_sha256'].items():
        assert sha(path) == expected, path
    assert sha(SOURCE) == PLAN['source_checkpoint_sha256']
    if (HERE / 'dispatch.json').exists():
        d = read(HERE / 'dispatch.json')
        assert sha(__file__) == d['runner_sha256']
        assert sha(HERE / 'plan.json') == d['plan_sha256']


def evaluator(iteration):
    e = module('frozen_force_eval', OLD / 'evaluate_stage_verified.py')
    e.OUT = HERE / 'evaluation' / ('i%06d' % iteration)
    e.jobs = lambda: [(iteration, seed, mode, force) for seed in PLAN['evaluation_seeds']
                     for mode in ('mean', 'sample') for force in (False, True)]
    e.checkpoint = lambda i: HERE / 'train' / ('model_iter_%06d.pt' % i)
    return e


def update_report(state, iteration, error=None):
    # One authoritative report, three scoped copies; historical blocks retained.
    start, end = '<!-- ranger-force-continue16000:start -->', '<!-- ranger-force-continue16000:end -->'
    text = [start, '**Ranger受力续训新授权：13000→16000，新增3000轮（累计受力8000轮）。**',
            '用户明确授权继续至16000；原13000分段完整验证后追加3000轮，替代旧13000总终点，到16000完整验证后硬停。',
            '权威目录`AUDIT/%s`。当前状态`%s`，最近完成训练轮次%d。' % (HERE.name, state, iteration),
            '源13000检查点SHA `%s`；512环境×24步、8线程、固定全身标称COM／critic COM偏移0，原PPO／辅助联合更新与每轴±60N课程保持。' % PLAN['source_checkpoint_sha256'],
            '模型、两Adam、LR、计数及Torch／NumPy／Python RNG恢复须看实值核验；PhysX／task／history／episode统计及力脉冲现场重建，不能称完全连续或单独归因暂停。',
            '每500轮两验证seed266391／266392×mean／sample×zero／force，共六边界48条32环境×1000步独立验证，训练父进程保留现场等待。',
            '逐轮日志补齐奖励分项与控制误差；重建后的初段episode return包含不完整episode，另记本进程完整episode，不冒充旧8000→9000缺失return。']
    if (HERE / 'restoration_verification.json').exists():
        text.append('恢复实值核验已通过；首13001同批原生更新副本一致性以`preflight.json`为准。')
    summaries = [read(p) for p in sorted(HERE.glob('boundary_summary_*.json'))]
    if summaries:
        text += ['', '|轮次|seed|模式|力条件|几何末端cm|偏移目标cm|XY|yaw|估力RMSE／恒零N/轴|持续零力误报N|非超时次数|',
                 '|---|---|---|---|---:|---:|---:|---:|---|---:|---:|']
        for s in summaries:
            for r in s['results']:
                m = r['metrics']
                z = m['sustained_zero_force_estimate_norm_n']
                text.append('|%d|%d|%s|%s|%.3f|%.3f|%.5f|%.5f|%.3f／%.3f|%s|%d|' % (
                    s['iteration'], r['seed'], r['action_mode'], '受力' if r['force_curriculum'] else '全零',
                    m['ee_error_mean_cm'], m['ee_force_target_error_mean_cm'], m['xy_rmse_m_s'], m['yaw_rmse_rad_s'],
                    m['force_estimate_rmse_per_axis_n'], m['zero_predictor_rmse_per_axis_n'],
                    '%.3f' % z if z is not None else '无样本', m['non_timeout_terminations']))
        text += ['', '全部result／trace／initial-state／checkpoint实值SHA与float64评分复算、现场隔离、100／300／500训练尾窗见各boundary_summary与evaluation_isolation。mean／sample分开解释，不以总reward或中途最好点代替终点保持与零力偏置检查。']
    if error:
        text.append('**新异常：%s；证据保留，未自动从检查点重跑。**' % error)
    text += ['正式Ranger保持冻结。官方C.4确述受力初降后恢复；官方4096环境与本支512环境／本体任务不同，不以官方曲线保证本支自愈。', end]
    block = '\n'.join(text) + '\n'
    content = REPORT.read_text()
    if start in content:
        assert content.count(start) == 3
        import re
        content = re.sub(re.escape(start) + r'.*?' + re.escape(end) + '\n?', lambda _: block, content, flags=re.S)
    else:
        pos = content.find('\n') + 1
        content = content[:pos] + '\n' + block + '\n' + content[pos:]
        for heading in ('### 8.98 ', '### 8.99 '):
            pos = content.index('\n', content.index(heading)) + 1
            content = content[:pos] + '\n' + block + '\n' + content[pos:]
    tmp = REPORT.with_suffix('.continuation.tmp')
    tmp.write_text(content)
    os.replace(tmp, REPORT)


def snapshot(base, env, model, alg):
    state = base.isolation_state(env, model, alg)
    state['force_curriculum'] = copy.deepcopy({k: (vars(v) if k in ('command', 'external') else v)
                                            for k, v in vars(env.curriculum).items()})
    state['root_and_joints'] = copy.deepcopy({k: getattr(env.robot.data, k) for k in
           ('root_pos_w', 'root_quat_w', 'root_vel_w', 'joint_pos', 'joint_vel')})
    return state


def windows(rows):
    fields = ('mean_reward_per_step', 'mean_episode_reward', 'mean_complete_episode_reward',
              'ee_error_mean_cm', 'ee_force_target_error_mean_cm', 'xy_rmse_m_s', 'yaw_rmse_rad_s',
              'force_estimate_rmse_per_axis_n', 'zero_predictor_rmse_per_axis_n')
    result = {}
    for size in (100, 300, 500):
        selected = rows[-size:]
        values = {k: [r[k] for r in selected if r.get(k) is not None] for k in fields}
        result[str(size)] = {k: sum(v) / len(v) if v else None for k, v in values.items()}
        result[str(size)].update(observed_iterations=len(selected),
            non_timeout_terminations=sum(r['non_timeout_terminations'] for r in selected),
            reward_terms={k: sum(r['reward_terms'][k] for r in selected) / len(selected)
                          for k in selected[0]['reward_terms']})
    return result


def boundary(base, env, model, alg, iteration, path, rows):
    import numpy as np
    protect()
    e = evaluator(iteration)
    e.OUT.mkdir(parents=True, exist_ok=False)
    before = snapshot(base, env, model, alg)
    digest = sha(path)
    results = []
    for index, job in enumerate(e.jobs()):
        status = dict(state='evaluating', iteration=iteration, pid=os.getpid(),
                      current_job=list(job), completed_evaluations=index, updated_utc=now())
        write(HERE / 'status.json', status)
        write(HERE / 'train/status.json', status)
        with (e.OUT / ('job_%02d.log' % index)).open('w') as log:
            child = subprocess.Popen([PYTHON, __file__, 'evaluate', '--iteration', str(iteration),
                                      '--index', str(index)], stdout=log, stderr=subprocess.STDOUT)
            status['evaluation_pid'] = child.pid
            write(HERE / 'status.json', status)
            rc = child.wait()
        assert rc == 0, ('Validation failed; no automatic retry', job, rc)
        out = e.job_dir(job)
        r = read(out / 'result.json')
        assert r['state'] == 'complete' and r['checkpoint_sha256'] == digest
        assert sha(out / 'trace.npz') == r['trace_sha256']
        assert sha(out / 'initial_state.pt') == r['initial_state_sha256']
        with np.load(out / 'trace.npz') as z:
            trace = {k: z[k] if z[k].dtype == np.bool_ else z[k].astype(np.float64) for k in z.files}
        metrics = e.score(trace, np)
        diff = []
        for k, v in metrics.items():
            assert (v is None) == (r['metrics'][k] is None), k
            if v is not None:
                diff.append(abs(v - r['metrics'][k]))
                assert np.isclose(v, r['metrics'][k], rtol=1e-10, atol=1e-10), (k, v, r['metrics'][k])
        proof = dict(passed=True, iteration=iteration, checkpoint_sha256=digest,
                     result_sha256=sha(out / 'result.json'), trace_sha256=r['trace_sha256'],
                     initial_state_sha256=r['initial_state_sha256'], metrics=metrics,
                     maximum_difference=max(diff))
        write(out / 'independent_verification.json', proof)
        r.update(directory=str(out), result_sha256=proof['result_sha256'],
                 independent_float64_recomputation_max_difference=max(diff))
        results.append(r)
    after = snapshot(base, env, model, alg)
    assert base.equal(before, after), 'Evaluation changed training state; do not resume'
    write(HERE / ('evaluation_isolation_%06d.json' % iteration), dict(passed=True,
          iteration=iteration, checkpoint_sha256=digest, training_worker_pid=os.getpid(),
          model_two_adam_lr_rng_history_observations_physics_force_curriculum_unchanged=True,
          simulation_steps_during_evaluation=0))
    write(HERE / ('boundary_summary_%06d.json' % iteration), dict(iteration=iteration,
          checkpoint=str(path), checkpoint_sha256=digest, results=results,
          independent_trace_recomputation_passed=True, training_windows=windows(rows), created_utc=now()))
    protect()
    update_report('running' if iteration < PLAN['final_iteration'] else 'final_validation_complete', iteration)
    plot()


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [json.loads(l) for l in (HERE / 'train/timings.jsonl').read_text().splitlines()]
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    for ax, field, label in zip(axes, ('mean_reward_per_step', 'ee_force_target_error_mean_cm', 'ee_error_mean_cm'),
                                ('Reward / policy step', 'Force-offset target error (cm)', 'Geometric goal error (cm)')):
        ax.plot([r['iteration'] for r in rows], [r[field] for r in rows], alpha=.25, linewidth=.5)
        means = [sum(r[field] for r in rows[max(0,i-99):i+1])/len(rows[max(0,i-99):i+1]) for i in range(len(rows))]
        ax.plot([r['iteration'] for r in rows], means, label='Training trailing 100 iterations')
        ax.set_ylabel(label); ax.grid(alpha=.25); ax.legend()
    axes[-1].set_xlabel('Total training iteration; resumed from 13000; physics/history rebuilt')
    fig.tight_layout(); fig.savefig(HERE / 'training_curves.png', dpi=160); plt.close(fig)
    summaries = [read(p) for p in sorted(HERE.glob('boundary_summary_*.json'))]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for ax, field in zip(axes.flat, ('ee_force_target_error_mean_cm','force_rmse_improvement_over_zero_n',
                                    'sustained_zero_force_estimate_norm_n','xy_rmse_m_s')):
        for seed in PLAN['evaluation_seeds']:
            for mode in ('mean','sample'):
                for force in (False, True):
                    rs = [next(r for r in s['results'] if r['seed']==seed and r['action_mode']==mode
                               and r['force_curriculum']==force) for s in summaries]
                    ax.plot([s['iteration'] for s in summaries], [r['metrics'][field] for r in rs],
                            marker='o', linestyle='-' if force else '--', label='%d %s %s'%(seed,mode,'force' if force else 'zero'))
        ax.set_title(field); ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=6); fig.tight_layout(); fig.savefig(HERE/'validation_curves.png',dpi=160); plt.close(fig)


def train():
    protect()
    base = module('ranger_continue_base', BASE / 'run.py')
    env, ns = base.native_backend(512, PLAN['training_seed'])
    import numpy as np
    import torch
    torch.set_num_threads(8)
    guarded = env.robot.set_external_force_and_torque
    env.robot.set_external_force_and_torque = next(c.cell_contents for c in guarded.__closure__
        if callable(c.cell_contents) and getattr(c.cell_contents, '__name__', '') == 'set_external_force_and_torque')
    model, alg = base.learner(env)
    source = torch.load(SOURCE, map_location='cpu', weights_only=False)
    assert source['iterations'] == 13000 and source['global_policy_steps'] == 312000
    assert source['num_envs'] == 512 and source['steps_per_env'] == 24
    assert source['force_stage'] and source['com_randomization'] is False
    model.load_state_dict(source['model_state_dict'], strict=True)
    alg.optimizer.load_state_dict(source['optimizer_state_dict'])
    alg.adaptation_module_optimizer.load_state_dict(source['adaptation_optimizer_state_dict'])
    alg.learning_rate = float(source['learning_rate'])
    env.global_policy_steps = 312000
    env.curriculum.start_step = 192001
    env.force_curriculum_enabled = True
    base.restore_rng(source['rng'])
    restore = dict(passed=True, checked_utc=now(), source_checkpoint_sha256=sha(SOURCE),
        model_exact=base.equal(model.state_dict(), source['model_state_dict']),
        ppo_adam_exact=base.equal(alg.optimizer.state_dict(), source['optimizer_state_dict']),
        auxiliary_adam_exact=base.equal(alg.adaptation_module_optimizer.state_dict(), source['adaptation_optimizer_state_dict']),
        rng_exact=base.equal(base.rng(), source['rng']), learning_rate_exact=alg.learning_rate==source['learning_rate'],
        ppo_param_group_lr_exact=all(g['lr']==alg.learning_rate for g in alg.optimizer.param_groups),
        restored_adam_steps=base.optimizer_steps(alg), restored_global_policy_steps=env.global_policy_steps,
        force_start_global_policy_step=env.curriculum.start_step,
        physx_task_history_episode_statistics_force_pulses_rebuilt=True,
        rng_restored_before_new_episode_phase_draw=True)
    for k in ('model_exact','ppo_adam_exact','auxiliary_adam_exact','rng_exact','learning_rate_exact','ppo_param_group_lr_exact'):
        assert restore[k], k
    assert restore['restored_adam_steps'] == {'ppo':[260000],'auxiliary':[260000]}
    write(HERE/'restoration_verification.json',restore)
    env.episode_length_buf[:] = torch.randint_like(env.episode_length_buf,high=env.shared_contract['max_episode_length'])
    x = env.get_observations()
    nominal_data = torch.load(AUDIT/'gym_ranger_20261024/lab_transport_defaults/shape_properties.pt',map_location='cpu',weights_only=False)
    meta = read(AUDIT/'gym_ranger_20261024/lab_transport_defaults/result.json')
    indices = [meta['body_names'].index(n) for n in env.robot.body_names]
    nominal = nominal_data['coms'][:,indices,:3].expand(env.num_envs,-1,-1)
    com_delta = float((env.robot.root_physx_view.get_coms()[:,:,:3]-nominal).abs().max())
    assert com_delta < 2e-6 and not env.domain.mass_deltas[:,1:4].count_nonzero()
    init = dict(passed=True,restoration=restore,com_nominal_max_abs_m=com_delta,critic_com_offset_max_m=0.,
                other_domain_randomization=env.domain.summary(),force_max_per_axis_n=60,
                actual_force_max_before_first_transition_n=float(env.curriculum.external.value.abs().max()))
    write(HERE/'initialization_verification.json',init)
    out = HERE/'train'; out.mkdir(exist_ok=False)
    base.tensor_save(out/'initial_state.pt', snapshot(base, env, model, alg))
    init['initial_state_sha256'] = sha(out/'initial_state.pt')
    write(HERE/'initialization_verification.json', init)
    ticks = Path('/proc/%d/stat'%os.getpid()).read_text().split()[21]
    write(HERE/'worker_launch.json',dict(pid=os.getpid(),process_start_ticks=ticks,started_utc=now()))
    pre = {}
    compute = env.rewards.compute
    def capture(robot,feet,contacts,controller,actions,torques,commands,goal,applied,base_force,frame,slices):
        cfg=env.shared_state.pulse.cfg.commands
        moving=((commands[:,0].abs()>cfg.lin_vel_x_clip)|(commands[:,1].abs()>cfg.lin_vel_y_clip)|
                (commands[:,2].abs()>cfg.ang_vel_yaw_clip))
        ee=robot.data.body_pos_w[:,env.gripper_id]-goal.goal_world
        cmd_world=ns['quat_apply'](ns['yaw_quat'](robot.data.root_quat_w),commands[:,9:12])
        offset=(applied+cmd_world)/env.shared_state.pulse.gripper_force_kps
        vel=robot.data.root_lin_vel_b[:,:2]-commands[:,:2]*moving[:,None]
        pre.update(ee=float(ee.norm(dim=-1).mean())*100,
                   offset=float((ee-offset).norm(dim=-1).mean())*100,
                   xy=float(vel.square().mean()),
                   yaw=float((robot.data.root_ang_vel_b[:,2]-commands[:,2]).square().mean()))
        return compute(robot,feet,contacts,controller,actions,torques,commands,goal,applied,base_force,frame,slices)
    env.rewards.compute=capture
    ep_rewards=torch.zeros(512); has_reset=torch.zeros(512,dtype=torch.bool)
    returns=deque(maxlen=100); full_returns=deque(maxlen=100)
    rows=[]; started=time.monotonic()
    update_report('running_initialization_verified',13000)
    for iteration in range(13001,PLAN['final_iteration']+1):
        sums=dict(reward=0.,ee=0.,offset=0.,xy=0.,yaw=0.,force_squared=0.,zero_squared=0.)
        terms={}; non_timeout=0; actual_peak=command_peak=0.
        with torch.inference_mode():
            for t in range(24):
                pred=model.get_student_latent(x['obs'])[:,6:9]*100
                truth=x['obs_pred'][:,6:9]*100
                sums['force_squared']+=float((pred-truth).square().mean())
                sums['zero_squared']+=float(truth.square().mean())
                action=alg.act(x['obs'],x['privileged_obs'],x['obs_pred'])
                x,reward,done,info=env.step(action)
                expected=reward+alg.gamma*alg.transition.values.squeeze(1)*info['time_outs']
                alg.process_env_step(reward,done,info)
                assert torch.allclose(alg.storage.rewards[t,:,0],expected,atol=1e-6,rtol=1e-6)
                sums['reward']+=float(reward.mean())
                for k in ('ee','offset','xy','yaw'): sums[k]+=pre[k]
                non_timeout+=int((done & ~info['time_outs'].bool()).sum())
                for k,v in info['reward_terms'].items(): terms[k]=terms.get(k,0.)+float(v.mean())/24
                ep_rewards+=reward; ids=done.nonzero().flatten()
                returns.extend(ep_rewards[ids].tolist()); full_returns.extend(ep_rewards[ids[has_reset[ids]]].tolist())
                ep_rewards[ids]=0; has_reset[ids]=True
                actual_peak=max(actual_peak,float(env.curriculum.external.value.abs().max()))
                command_peak=max(command_peak,float(env.curriculum.command.value.abs().max()))
                assert actual_peak<=60.0001 and command_peak<=60.0001
            alg.compute_returns(x['privileged_obs'])
        if iteration==13001:
            duplicate=copy.deepcopy(alg); saved_rng=base.rng()
            duplicate_losses=duplicate.update(); duplicate_rng=base.rng(); base.restore_rng(saved_rng)
        losses=alg.update()
        if iteration==13001:
            checks=dict(model=base.equal(duplicate.actor_critic.state_dict(),model.state_dict()),
                        ppo_adam=base.equal(duplicate.optimizer.state_dict(),alg.optimizer.state_dict()),
                        auxiliary_adam=base.equal(duplicate.adaptation_module_optimizer.state_dict(),alg.adaptation_module_optimizer.state_dict()),
                        losses=base.equal(duplicate_losses,losses),rng=base.equal(duplicate_rng,base.rng()),
                        lr=duplicate.learning_rate==alg.learning_rate)
            assert all(checks.values()),checks
            del duplicate
            write(HERE/'preflight.json',dict(passed=True,same_batch_native_duplicate_update_checks=checks,
                iteration=iteration,global_policy_steps=env.global_policy_steps,adam_steps=base.optimizer_steps(alg),
                no_extra_environment_steps=True,restoration_verified=True))
        counts=base.optimizer_steps(alg)
        assert counts=={'ppo':[iteration*20],'auxiliary':[iteration*20]},counts
        assert env.global_policy_steps==iteration*24
        assert all(torch.isfinite(v).all() for v in model.state_dict().values())
        row=dict(iteration=iteration,global_policy_steps=env.global_policy_steps,adam_steps=counts,
                 mean_reward_per_step=sums['reward']/24,
                 mean_episode_reward=sum(returns)/len(returns) if returns else None,
                 mean_complete_episode_reward=sum(full_returns)/len(full_returns) if full_returns else None,
                 ee_error_mean_cm=sums['ee']/24,ee_force_target_error_mean_cm=sums['offset']/24,
                 xy_rmse_m_s=(sums['xy']/24)**.5,yaw_rmse_rad_s=(sums['yaw']/24)**.5,
                 force_estimate_rmse_per_axis_n=(sums['force_squared']/24)**.5,
                 zero_predictor_rmse_per_axis_n=(sums['zero_squared']/24)**.5,
                 force_metrics_include_initial_history=True,non_timeout_terminations=non_timeout,
                 reward_terms=terms,value_loss=float(losses[0]),surrogate_loss=float(losses[1]),
                 official_logged_auxiliary_loss=float(losses[2]),learning_rate=alg.learning_rate,
                 action_std=float(model.std.mean()),actual_force_peak_n=actual_peak,
                 command_force_peak_n=command_peak,elapsed_s=time.monotonic()-started)
        rows.append(row)
        with (out/'timings.jsonl').open('a') as f: f.write(json.dumps(row,allow_nan=False)+'\n')
        status=dict(state='running',iteration=iteration,pid=os.getpid(),process_start_ticks=ticks,latest=row,updated_utc=now())
        write(out/'status.json',status); write(HERE/'status.json',status)
        if iteration==13001 or iteration%25==0: print('FORCE_CONTINUE_PROGRESS',json.dumps(row),flush=True)
        if iteration in PLAN['evaluation_iterations']:
            path=out/('model_iter_%06d.pt'%iteration)
            ck=dict(model_state_dict=model.state_dict(),optimizer_state_dict=alg.optimizer.state_dict(),
                    adaptation_optimizer_state_dict=alg.adaptation_module_optimizer.state_dict(),iterations=iteration,
                    global_policy_steps=env.global_policy_steps,num_envs=512,steps_per_env=24,
                    learning_rate=alg.learning_rate,rng=base.rng(),source_checkpoint_sha256=sha(SOURCE),
                    force_stage=True,com_randomization=False)
            base.tensor_save(path,ck)
            boundary(base,env,model,alg,iteration,path,rows)
    protect()
    summaries=[read(HERE/('boundary_summary_%06d.json'%i)) for i in PLAN['evaluation_iterations']]
    assert len(summaries)==6 and sum(len(s['results']) for s in summaries)==48
    result=dict(state='complete',iteration=16000,global_policy_steps=env.global_policy_steps,
                adam_steps=base.optimizer_steps(alg),training_rows=len(rows),new_force_iterations=3000,
                cumulative_force_iterations=8000,boundaries_verified=6,independent_evaluations_verified=48,
                final_checkpoint_sha256=sha(out/'model_iter_016000.pt'),
                protected_sha256_passed=True,automatic_extension=False,completed_utc=now(),
                best_validation_selection_not_unseen_test=True,
                best_force_offset_validation_iteration=min(summaries,key=lambda s:sum(r['metrics']['ee_force_target_error_mean_cm']
                    for r in s['results'] if r['force_curriculum']))['iteration'])
    write(HERE/'completion.json',result); write(out/'completion.json',result); write(HERE/'status.json',result)
    update_report('complete',16000)
    sys.stdout.flush(); os._exit(0)


def supervise():
    lock=(HERE/'launch.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (HERE/'launch.json').exists(),'Already launched; do not restart'
    protect()
    with (HERE/'worker.log').open('w') as log:
        child=subprocess.Popen([PYTHON,__file__,'train'],stdout=log,stderr=subprocess.STDOUT)
        write(HERE/'launch.json',dict(supervisor_pid=os.getpid(),
            supervisor_start_ticks=Path('/proc/%d/stat'%os.getpid()).read_text().split()[21],
            worker_pid=child.pid,worker_start_ticks=Path('/proc/%d/stat'%child.pid).read_text().split()[21],
            source_checkpoint_sha256=sha(SOURCE),started_utc=now(),runner_sha256=sha(__file__),
            additional_iterations=3000,final_iteration=16000,automatic_restart=False))
        rc=child.wait()
    if rc!=0 or not (HERE/'completion.json').exists():
        last=read(HERE/'status.json') if (HERE/'status.json').exists() else {}
        failure=dict(state='failed',worker_exit_code=rc,last_status=last,failed_utc=now(),automatic_retry=False)
        write(HERE/'failure.json',failure); write(HERE/'status.json',failure)
        update_report('failed',last.get('iteration',13000),str(failure))
        raise RuntimeError('Continuation stopped; evidence retained, no automatic retry')
    write(HERE/'supervisor_completion.json',dict(state='complete',worker_exit_code=rc,completed_utc=now()))


def main():
    p=argparse.ArgumentParser(); p.add_argument('command',choices=('supervise','train','evaluate','check'))
    p.add_argument('--iteration',type=int); p.add_argument('--index',type=int); args=p.parse_args()
    if args.command=='evaluate':
        protect(); evaluator(args.iteration).worker(args.index)
    elif args.command=='train': train()
    elif args.command=='supervise': supervise()
    else:
        protect()
        assert PLAN['final_iteration']==16000 and PLAN['initial_iteration']==13000
        assert len(evaluator(13500).jobs())==8
        assert REPORT.exists()
        print('STATIC_CHECK_PASSED',len(PLAN['protected_sha256']),str(REPORT))


if __name__=='__main__':
    try: main()
    except BaseException as exc:
        if len(sys.argv)>1 and sys.argv[1]=='train':
            write(HERE/'worker_exception.json',dict(error=repr(exc),traceback=traceback.format_exc(),failed_utc=now()))
        raise

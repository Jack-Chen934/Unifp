"""Finite, isolated Ranger baseline. Native PPO is imported without modification.

Training remains alive through serial external validations. No checkpoint resume,
automatic retries, policy promotion, or force stage is implemented.
"""
import argparse
import ast
import contextlib
import copy
import datetime
import fcntl
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time
import traceback
from collections import deque

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
ROOT = AUDIT.parents[2]
PYTHON = '/home/sorry/.cache/unifp/isaacgym_preview4/torch231/bin/python'
REPORT = ROOT / 'PROJECT_REPORT.md'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def read(path):
    return json.loads(Path(path).read_text())


def protect():
    for path, digest in read(HERE / 'plan.json')['protected_sha256'].items():
        assert sha(path) == digest, path


def native_backend(n, seed):
    # The frozen backend transports existing Ranger task functions to Gym.
    # Three exact envelope edits only, all before the first reset.
    sys.path[:0] = [str(ROOT), str(ROOT / 'UniFP')]
    spec = importlib.util.spec_from_file_location('ranger_frozen_audit', HERE / 'audit_snapshot.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT, module.OUT = ROOT, AUDIT
    tree = ast.parse((HERE / 'audit_snapshot.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'gym_ranger_backend')
    source = ast.get_source_segment((HERE / 'audit_snapshot.py').read_text(), node)
    edits = {'torch.set_num_threads(2)': 'torch.set_num_threads(8)',
             'self.global_policy_steps=240000': 'self.global_policy_steps=0',
             'self.force_curriculum_enabled=True': 'self.force_curriculum_enabled=False'}
    for before, after in edits.items():
        assert source.count(before) == 1, before
        source = source.replace(before, after)
    exec(compile(source, str(HERE / 'backend_envelope.py'), 'exec'), module.__dict__)
    env, ns = module.gym_ranger_backend(n, seed, max_force_n=60., fixed_com=True,
                                       init_before_prepare=True, official_shared=True)
    assert env.global_policy_steps == 0 and not env.force_curriculum_enabled
    env.shared_contract['remaining_differences'][-1] = 'Morphology-specific reset/observations; CPU learner, 512 training environments'
    env.curriculum.start_step = 10**12  # Disabled independently of the global counter.
    original_force = env.robot.set_external_force_and_torque

    def zero_force_guard(**kw):
        assert kw['forces'].count_nonzero() == kw['torques'].count_nonzero() == 0
        return original_force(**kw)

    env.robot.set_external_force_and_torque = zero_force_guard
    return env, ns


def learner(env):
    from legged_gym.b2_gym_learn.ppo_cse_pf.actor_critic import ActorCritic
    from legged_gym.b2_gym_learn.ppo_cse_pf.ppo import PPO
    model = ActorCritic(1856, 294, 12, 58, 14, actor_hidden_dims=[512, 256, 128],
                        critic_hidden_dims=[512, 256, 128], init_noise_std=1.)
    alg = PPO(model, device='cpu', entropy_coef=.01, num_learning_epochs=5,
              num_mini_batches=4, learning_rate=1e-3, schedule='adaptive', gamma=.99,
              lam=.95, desired_kl=.01)
    alg.init_storage(env.num_envs, 24, [1856], [294], [12], [14])
    assert not alg.optimizer.state and not alg.adaptation_module_optimizer.state
    return model, alg


def rng():
    import torch
    import numpy as np
    return dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate())


def restore_rng(state):
    import torch
    import numpy as np
    torch.set_rng_state(state['torch'])
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])


def equal(a, b):
    import torch
    import numpy as np
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) == type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def tensor_save(path, state):
    import torch
    assert not path.exists(), f'Refuse checkpoint overwrite: {path}'
    tmp = path.with_suffix('.tmp')
    torch.save(state, tmp)
    os.replace(tmp, path)


def initialization(env, ns, model, alg, out, randomized_episode):
    import torch
    assert torch.get_num_threads() == 8
    assert env.global_policy_steps == 0
    assert env.domain.mass_deltas[:, 1:4].count_nonzero() == 0
    view = env.robot.root_physx_view
    physical_com = view.get_coms()
    lab = torch.load(AUDIT / 'gym_ranger_20261024/lab_transport_defaults/shape_properties.pt', weights_only=False)
    meta = read(AUDIT / 'gym_ranger_20261024/lab_transport_defaults/result.json')
    indices = [meta['body_names'].index(name) for name in env.robot.body_names]
    nominal = lab['coms'][:, indices, :3].expand(env.num_envs, -1, -1)
    delta = float((physical_com[:, :, :3] - nominal).abs().max())
    assert delta < 2e-6, delta
    assert not alg.optimizer.state and not alg.adaptation_module_optimizer.state
    if randomized_episode:
        # Preserve the reference shared by termination, force and rewards.
        env.episode_length_buf[:] = torch.randint_like(env.episode_length_buf, high=env.shared_contract['max_episode_length'])
    data = dict(model_state_dict=copy.deepcopy(model.state_dict()),
                optimizer_state_dict=copy.deepcopy(alg.optimizer.state_dict()),
                adaptation_optimizer_state_dict=copy.deepcopy(alg.adaptation_module_optimizer.state_dict()),
                observations=copy.deepcopy(env.get_observations()), physics=env.state_snapshot(),
                masses=view.get_masses(), coms=physical_com, inertias=view.get_inertias(),
                materials=view.get_material_properties(), motor_strength=env.domain.motor_strength.clone(),
                episode_length_buf=env.episode_length_buf.clone(), rng=rng())
    tensor_save(out / 'initial_state.pt', data)
    result = dict(passed=True, fresh_model=True, two_adam_state_counts=[0, 0],
                  global_policy_steps=0, num_envs=env.num_envs, threads=8,
                  nominal_com_all_bodies_max_difference_m=delta, critic_com_delta_max_m=0.,
                  fixed_com=True, zero_actual_and_command_ee_force=True,
                  randomized_episode_lengths=randomized_episode,
                  episode_length_alias_preserved=env.episode_length_buf is env.rewards.episode_steps,
                  transport=env.transport, domain=env.domain.summary(),
                  goals=env.shared_state.goal.goal_ee_ranges, reward_scales=env.rewards.scales,
                  initial_state_sha256=sha(out / 'initial_state.pt'),
                  remaining_limits=read(HERE / 'plan.json')['limitations'])
    assert result['episode_length_alias_preserved']
    write(out / 'initialization_verification.json', result)
    return result


def instrument(env, ns):
    import torch
    pre = {}
    original = env.rewards.compute

    def capture(robot, feet, contacts, controller, actions, torques, commands,
                goal, force, base_force, frame, slices):
        assert force.count_nonzero() == base_force.count_nonzero() == commands[:, 9:15].count_nonzero() == 0
        target = commands[:, :2].clone()
        cfg = env.shared_state.pulse.cfg.commands
        moving = ((target[:, 0].abs() > cfg.lin_vel_x_clip) |
                  (target[:, 1].abs() > cfg.lin_vel_y_clip) |
                  (commands[:, 2].abs() > cfg.ang_vel_yaw_clip))
        target *= moving[:, None]
        pre.update(ee=robot.data.body_pos_w[:, env.gripper_id].clone(), goal=goal.goal_world.clone(),
                   velocity=robot.data.root_lin_vel_b.clone(), angular_velocity=robot.data.root_ang_vel_b.clone(),
                   velocity_command=commands[:, :3].clone(), velocity_target=target,
                   force=force.clone(), force_command=commands[:, 9:12].clone())
        return original(robot, feet, contacts, controller, actions, torques, commands,
                        goal, force, base_force, frame, slices)

    env.rewards.compute = capture
    checks = dict(history_shift_max_abs=0., fresh_timeout_steps=0, reset_history_count=0)

    def step(actions):
        # No RNG draws or extra observe() calls in measurement.
        previous = env.actor.history.clone()
        x, reward, done, info = env.step(actions)
        assert equal(info['time_outs'], env.rewards.last_termination_masks['timeout'])
        active = ~done
        if active.any():
            diff = float((env.actor.history[active, :-1] - previous[active, 1:]).abs().max())
            checks['history_shift_max_abs'] = max(checks['history_shift_max_abs'], diff)
        if done.any():
            assert env.actor.history[done, :-1].count_nonzero() == 0
            checks['reset_history_count'] += int(done.sum())
        checks['fresh_timeout_steps'] += 1
        assert checks['history_shift_max_abs'] == 0
        assert x['obs_pred'][:, 6:12].count_nonzero() == 0
        assert env.actor.commands[:, 9:15].count_nonzero() == 0
        assert all(torch.isfinite(v).all() for v in x.values()) and torch.isfinite(reward).all()
        return x, reward, done, info

    return step, pre, checks


def checkpoint(env, model, alg, iteration, out):
    state = dict(model_state_dict=model.state_dict(), optimizer_state_dict=alg.optimizer.state_dict(),
                 adaptation_optimizer_state_dict=alg.adaptation_module_optimizer.state_dict(),
                 iterations=iteration, iter=iteration, global_policy_steps=env.global_policy_steps,
                 num_envs=512, steps_per_env=24, seed=read(HERE / 'plan.json')['seed'],
                 learning_rate=alg.learning_rate, rng=rng(), config=read(HERE / 'plan.json'),
                 simulation_history_not_checkpointed=True, candidate_only=True)
    path = out / f'model_iter_{iteration:06d}.pt'
    tensor_save(path, state)
    return path


def optimizer_steps(alg):
    return {name: sorted({int(s['step']) for s in opt.state.values()})
            for name, opt in [('ppo', alg.optimizer), ('auxiliary', alg.adaptation_module_optimizer)]}


def metrics(trace):
    import torch
    ee = (trace['ee'].double() - trace['goal'].double()).norm(dim=-1)
    xy = trace['velocity'].double()[:, :, :2] - trace['velocity_target'].double()
    yaw = trace['angular_velocity'].double()[:, :, 2] - trace['velocity_command'].double()[:, :, 2]
    sustained = trace['full_history'].bool()
    force_est = trace['predictions'].double()[:, :, 6:9] * 100.
    assert trace['force'].count_nonzero() == trace['force_command'].count_nonzero() == trace['target'][:, :, 6:12].count_nonzero() == 0
    return dict(ee_error_mean_cm=float(ee.mean() * 100), ee_error_p95_cm=float(torch.quantile(ee.flatten(), .95) * 100),
                xy_rmse_m_s=float(xy.square().mean().sqrt()), yaw_rmse_rad_s=float(yaw.square().mean().sqrt()),
                non_timeout_terminations=int((trace['done'].bool() & ~trace['timeout'].bool()).sum()),
                timeouts=int((trace['done'].bool() & trace['timeout'].bool()).sum()),
                mean_reward_per_step=float(trace['reward'].double().mean()),
                sustained_zero_force_estimate_norm_n=float(force_est[sustained].norm(dim=-1).mean()) if sustained.any() else None,
                sustained_zero_samples=int(sustained.sum()), actual_force_max_n=0., command_force_max_n=0.,
                velocity_prediction_rmse_m_s=float(((trace['predictions'].double()[:, :, :3] - trace['target'].double()[:, :, :3]) / 2.).square().mean().sqrt()),
                ee_state_prediction_mse_normalized=float((trace['predictions'].double()[:, :, 3:6] - trace['target'].double()[:, :, 3:6]).square().mean()))


def evaluate(args):
    # Import Gym before torch via the backend, as required by Preview 4.
    plan = read(HERE / 'plan.json')
    out = Path(args.output)
    out.mkdir(exist_ok=False)
    env, ns = native_backend(32, args.seed)
    import torch
    model, alg = learner(env)
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model_state_dict'])
    model.eval().requires_grad_(False)
    initial = copy.deepcopy(model.state_dict())
    initialization(env, ns, model, alg, out, False)
    step, pre, checks = instrument(env, ns)
    x = env.get_observations()
    history_count = torch.ones(32, dtype=torch.long)
    generator = torch.Generator().manual_seed(args.seed + 190000)
    trace = {}

    def add(key, value):
        trace.setdefault(key, []).append(value.detach().clone())

    with torch.inference_mode():
        for t in range(plan['evaluation_steps']):
            add('target', x['obs_pred'])
            add('predictions', model.get_student_latent(x['obs']))
            add('full_history', history_count >= 32)
            action = model.act_inference(x)
            if args.action_mode == 'sample':
                action = action + model.std * torch.randn(action.shape, generator=generator)
            add('actions', action)
            x, reward, done, info = step(action)
            for key, value in pre.items():
                add(key, value)
            add('done', done)
            add('timeout', info['time_outs'])
            add('reward', reward)
            history_count += 1
            history_count[done] = 1
    trace = {key: torch.stack(value) for key, value in trace.items()}
    assert all(torch.isfinite(v).all() for v in trace.values())
    assert equal(initial, model.state_dict()) and optimizer_steps(alg) == {'ppo': [], 'auxiliary': []}
    tensor_save(out / 'trace.pt', trace)
    write(out / 'result.json', dict(state='complete', checkpoint=str(args.checkpoint),
          checkpoint_sha256=sha(args.checkpoint), iteration=ck['iterations'], seed=args.seed,
          action_mode=args.action_mode, metrics=metrics(trace), checks=checks,
          model_unchanged=True, optimizer_updates=0, trace_sha256=sha(out / 'trace.pt')))
    protect()
    # Avoid known teardown segfault after all complete evidence is durable.
    sys.stdout.flush()
    os._exit(0)


def isolation_state(env, model, alg):
    return copy.deepcopy(dict(model=model.state_dict(), optimizer=alg.optimizer.state_dict(),
          auxiliary=alg.adaptation_module_optimizer.state_dict(), lr=alg.learning_rate, rng=rng(),
          actor_history=env.actor.history, critic_history=env.privileged.history,
          state=env.state_snapshot(), observations=env.get_observations(),
          episode_lengths=env.episode_length_buf, global_steps=env.global_policy_steps,
          common_steps=env._shared_common_step_counter, goal=env.shared_state.goal.goal_timer,
          goal_end=env.shared_state.goal.ee_goal_sphere))


def boundary(env, model, alg, iteration, path):
    import torch
    plan = read(HERE / 'plan.json')
    before = isolation_state(env, model, alg)
    write(HERE / 'status.json', dict(state='evaluating', iteration=iteration, pid=os.getpid(), updated_utc=now()))
    results = []
    for seed in plan['evaluation_seeds']:
        for mode in ('mean', 'sample'):
            out = HERE / f'eval_{iteration:06d}_s{seed}_{mode}'
            cmd = [PYTHON, str(HERE / 'run.py'), 'evaluate', '--checkpoint', str(path),
                   '--seed', str(seed), '--action-mode', mode, '--output', str(out)]
            with (HERE / 'evaluation.log').open('a') as log:
                completed = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
            assert completed.returncode == 0, (out, completed.returncode)
            result = read(out / 'result.json')
            assert result['checkpoint_sha256'] == sha(path)
            trace = torch.load(out / 'trace.pt', map_location='cpu', weights_only=False)
            recomputed = metrics(trace)
            assert equal(result['metrics'], recomputed)
            results.append(dict(directory=str(out), result_sha256=sha(out / 'result.json'),
                                trace_sha256=sha(out / 'trace.pt'), seed=seed, action_mode=mode,
                                metrics=recomputed))
    after = isolation_state(env, model, alg)
    assert equal(before, after), 'Evaluation isolation mismatch; do not resume learner'
    write(HERE / f'evaluation_isolation_{iteration:06d}.json', dict(passed=True, iteration=iteration,
          checkpoint_sha256=sha(path), model_two_adam_lr_rng_history_physics_observations_unchanged=True,
          training_worker_preserved_pid=os.getpid(), simulation_steps_during_evaluation=0))
    rows = [json.loads(line) for line in (HERE / 'train' / 'timings.jsonl').read_text().splitlines()] if (HERE / 'train' / 'timings.jsonl').exists() else []
    fields = ('mean_reward_per_step', 'mean_episode_reward', 'ee_error_mean_cm', 'xy_rmse_m_s', 'yaw_rmse_rad_s')
    windows = {}
    for size in (100, 300, 500):
        selected = rows[-size:]
        windows[str(size)] = {key: sum(r[key] for r in selected if r.get(key) is not None) /
                             max(1, sum(r.get(key) is not None for r in selected)) for key in fields}
        windows[str(size)]['observed_iterations'] = len(selected)
    summary = dict(iteration=iteration, checkpoint=str(path), checkpoint_sha256=sha(path),
                   results=results, independent_trace_recomputation_passed=True,
                   training_windows=windows, created_utc=now())
    write(HERE / f'boundary_summary_{iteration:06d}.json', summary)
    protect()
    if rows:
        plot()
    update_report('running' if iteration < 8000 else 'complete', iteration)


def update_report(state, iteration, error=None):
    summaries = [read(path) for path in sorted(HERE.glob('boundary_summary_*.json'))]
    block = ['<!-- ranger-fresh512-fixedcom:start -->',
             f'**Ranger新基础控制对比（北京时间{datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}；{state}，已完成{iteration}轮）。** '
             '用户批准Isaac Gym、单种子266281从随机参数与两空Adam连续0→8000；512环境×24步、8线程，全程末端命令力／实际力0。'
             '按用户最新要求固定资产标称质心，保留质量、摩擦、电机幅度、观测噪声和8秒底盘速度扰动。'
             f'权威目录`AUDIT/{HERE.name}`；初始化、预检、计划与进程见该目录JSON，不重启旧队列。',
             '', '原生PPO与辅助任务联合更新未改；学习率1e-3自适应KL、辅助1e-5，gamma0.99、lambda0.95、5epoch×4minibatch。'
             'Ranger v3_original、14动作：转向300／20、轮速阻尼20／幅度5rad/s、机械臂80／4及原力矩限制。'
             'A1目标球半径0.35～0.75m、仰角0.15～1.4、方位±1.2；使用已有工作空间，不为缩小误差改范围。'
             '平地、禁自碰、32／1求解迭代、隐式PD、CPU学习／GPU PhysX是明确平台／形态差异；工作空间并非全域IK可达性证明。'
             '轮式底盘不使用步态奖励，保留17项适用原生奖励公式／权重。原始总奖励不直接与B2数值比较。',
             '', '训练现场在验证期间保留，不从检查点恢复。每500轮保存完整模型／两Adam／LR／计数／三类RNG；'
             '0轮及每500轮用266291／266292×均值／采样四条件、32环境×1000步独立串行验证并float64复算。'
             '训练global从0开始，无原B2 runner的初始零动作一步；episode长度原位随机，保持奖励／终止计数别名。'
             '8000完整边界固定结束，不进入受力，不按回升延长，不替换正式Ranger策略。', '',
             '|轮数|末端cm（四条件均值）|XY RMSE m/s|yaw RMSE rad/s|非超时终止合计|持续零力误报N|',
             '|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        ms = [r['metrics'] for r in s['results']]
        avg = lambda key: sum(m[key] for m in ms) / len(ms)
        block.append(f"|{s['iteration']}|{avg('ee_error_mean_cm'):.2f}|{avg('xy_rmse_m_s'):.5f}|{avg('yaw_rmse_rad_s'):.5f}|{sum(m['non_timeout_terminations'] for m in ms)}|{avg('sustained_zero_force_estimate_norm_n'):.4f}|")
    if error:
        block += ['', '本支新异常：' + str(error) + '。证据保留，未自动恢复重跑。']
    if (HERE / 'completion.json').exists():
        completion = read(HERE / 'completion.json')
        block += ['', f"完整固定预算已结束，全部{completion['boundaries_verified']}个边界、{completion['independent_evaluations_verified']}条验证复算完成；"
                  f"按四条件平均末端误差选出的最佳验证点为{completion['best_validation_iteration']}轮。"
                  '这是验证选点，不是未见测试；终点和100／300／500轮尾窗见boundary_summary_008000.json。']
    if (HERE / 'setup_failure_1.json').exists():
        block += ['', '正式学习前两次预检分别因动态库／Ninja路径缺失未能导入Gym，仿真／模型／更新均未开始；'
                  '已保留setup_failure_1／2.json，补齐既有运行时环境后重新预检，未恢复或重跑正式训练。']
    block += ['', '这是本体与任务适配后的比较实验，不能单独分离128→512或质心影响；零力误报低不证明非零估力成功。',
              '<!-- ranger-fresh512-fixedcom:end -->', '']
    text = '\n'.join(block)
    with (HERE / 'report.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        report = REPORT.read_text()
        import re
        pattern = r'<!-- ranger-fresh512-fixedcom:start -->.*?<!-- ranger-fresh512-fixedcom:end -->\n?'
        report = re.sub(pattern, '', report, flags=re.S)
        for heading in ('### 8.98 ', '### 8.99 '):
            index = report.index('\n', report.index(heading)) + 1
            report = report[:index] + '\n' + text + report[index:]
        index = report.index('\n') + 1
        report = report[:index] + '\n' + text + report[index:]
        tmp = REPORT.with_suffix('.md.tmp')
        tmp.write_text(report)
        os.replace(tmp, REPORT)


def train(preflight=False):
    plan = read(HERE / 'plan.json')
    out = HERE / ('preflight' if preflight else 'train')
    out.mkdir(exist_ok=False)
    env, ns = native_backend(512, plan['preflight_seed'] if preflight else plan['seed'])
    import torch
    model, alg = learner(env)
    init = initialization(env, ns, model, alg, out, True)
    step, pre, checks = instrument(env, ns)
    if not preflight:
        write(HERE / 'initialization_verification.json', init)
    x = env.get_observations()
    episode_rewards = torch.zeros(512)
    episode_lengths = torch.zeros(512, dtype=torch.long)
    rewards_buffer, lengths_buffer = deque(maxlen=100), deque(maxlen=100)
    start = time.monotonic()
    initial_weights = copy.deepcopy(model.state_dict())
    if not preflight:
        path = checkpoint(env, model, alg, 0, out)
        boundary(env, model, alg, 0, path)
    limit = 2 if preflight else 8000
    for iteration in range(1, limit + 1):
        t0 = time.monotonic()
        sums = dict(reward=0., ee=0., xy=0., yaw=0., done=0, timeout=0)
        terms = {}
        with torch.inference_mode():
            for t in range(24):
                action = alg.act(x['obs'], x['privileged_obs'], x['obs_pred'])
                x, reward, done, info = step(action)
                expected = reward + alg.gamma * alg.transition.values.squeeze(1) * info['time_outs']
                alg.process_env_step(reward, done, info)
                assert torch.allclose(alg.storage.rewards[t, :, 0], expected, atol=1e-6, rtol=1e-6)
                sums['reward'] += float(reward.mean())
                sums['ee'] += float((pre['ee'] - pre['goal']).norm(dim=-1).mean()) * 100
                sums['xy'] += float((pre['velocity'][:, :2] - pre['velocity_target']).square().mean())
                sums['yaw'] += float((pre['angular_velocity'][:, 2] - pre['velocity_command'][:, 2]).square().mean())
                sums['done'] += int(done.sum())
                sums['timeout'] += int(info['time_outs'].sum())
                for key, value in info['reward_terms'].items():
                    terms[key] = terms.get(key, 0.) + float(value.mean()) / 24
                episode_rewards += reward
                episode_lengths += 1
                ids = done.nonzero().flatten()
                rewards_buffer.extend(episode_rewards[ids].tolist())
                lengths_buffer.extend(episode_lengths[ids].tolist())
                episode_rewards[ids] = 0
                episode_lengths[ids] = 0
            alg.compute_returns(x['privileged_obs'])
        t1 = time.monotonic()
        # Preflight only: execute a duplicate native update on the identical
        # captured batch/RNG. It must match the real native update exactly.
        if preflight and iteration == 1:
            duplicate = copy.deepcopy(alg)
            saved_rng = rng()
            duplicate_losses = duplicate.update()
            duplicate_rng = rng()
            restore_rng(saved_rng)
        losses = alg.update()
        if preflight and iteration == 1:
            assert equal(duplicate.actor_critic.state_dict(), model.state_dict())
            assert equal(duplicate.optimizer.state_dict(), alg.optimizer.state_dict())
            assert equal(duplicate.adaptation_module_optimizer.state_dict(), alg.adaptation_module_optimizer.state_dict())
            assert equal(duplicate_losses, losses) and equal(duplicate_rng, rng())
            del duplicate
        steps = optimizer_steps(alg)
        assert steps == {'ppo': [iteration * 20], 'auxiliary': [iteration * 20]}, steps
        assert env.global_policy_steps == iteration * 24
        assert all(torch.isfinite(v).all() for v in model.state_dict().values())
        row = dict(iteration=iteration, global_policy_steps=env.global_policy_steps,
                   samples=iteration * 512 * 24, mean_reward_per_step=sums['reward'] / 24,
                   mean_episode_reward=sum(rewards_buffer) / len(rewards_buffer) if rewards_buffer else None,
                   mean_episode_length=sum(lengths_buffer) / len(lengths_buffer) if lengths_buffer else None,
                   ee_error_mean_cm=sums['ee'] / 24, xy_rmse_m_s=(sums['xy'] / 24)**.5,
                   yaw_rmse_rad_s=(sums['yaw'] / 24)**.5, done_count=sums['done'], timeout_count=sums['timeout'],
                   non_timeout_terminations=sums['done'] - sums['timeout'], reward_terms=terms,
                   value_loss=float(losses[0]), surrogate_loss=float(losses[1]),
                   official_logged_auxiliary_loss=float(losses[2]), learning_rate=alg.learning_rate,
                   action_std=float(model.std.mean()), collection_s=t1-t0, learning_s=time.monotonic()-t1,
                   elapsed_s=time.monotonic()-start, adam_steps=steps, force_command_max_n=0., actual_force_max_n=0.)
        with (out / 'timings.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')
        status = dict(state='running', pid=os.getpid(), iteration=iteration, latest=row, updated_utc=now())
        write(out / 'status.json', status)
        if not preflight:
            write(HERE / 'status.json', status)
        if iteration == 1 or iteration % 25 == 0:
            print('RANGER_FRESH_PROGRESS', json.dumps(row), flush=True)
        if not preflight and iteration % 500 == 0:
            path = checkpoint(env, model, alg, iteration, out)
            boundary(env, model, alg, iteration, path)
    changes = {prefix: float(sum((v - initial_weights[k]).double().square().sum()
               for k, v in model.state_dict().items() if k.startswith(prefix)).sqrt())
               for prefix in ('actor_body', 'critic_body', 'adaptation_encoder_module', 'adaptation_decoder_module')}
    assert all(v > 0 for v in changes.values())
    result = dict(state='complete', iteration=limit, pid=os.getpid(), checks=checks,
                  module_change_l2=changes, initialization_verified=True, adam_steps=optimizer_steps(alg),
                  elapsed_s=time.monotonic()-start, native_update_duplicate_equal=True if preflight else None,
                  actual_and_command_force_always_zero=True, completed_utc=now())
    protect()
    write(out / 'completion.json', result)
    if preflight:
        write(HERE / 'preflight.json', result)
    else:
        summaries = [read(HERE / f'boundary_summary_{i:06d}.json') for i in range(0, 8001, 500)]
        best = min(summaries, key=lambda s: sum(r['metrics']['ee_error_mean_cm'] for r in s['results']))
        result.update(boundaries_verified=len(summaries), independent_evaluations_verified=4*len(summaries),
                      best_validation_iteration=best['iteration'], validation_selection_not_unseen_test=True,
                      final_checkpoint_sha256=sha(out / 'model_iter_008000.pt'))
        plot()
        write(HERE / 'completion.json', result)
        write(HERE / 'status.json', result)
        update_report('complete', 8000)
    sys.stdout.flush()
    os._exit(0)


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [json.loads(line) for line in (HERE / 'train/timings.jsonl').read_text().splitlines()]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for ax, field, label in zip(axes, ['mean_episode_reward', 'ee_error_mean_cm', 'xy_rmse_m_s'],
                                ['Episode reward (100 completed episodes)', 'EE error (cm)', 'XY velocity RMSE (m/s)']):
        ax.plot([r['iteration'] for r in rows], [r[field] for r in rows], alpha=.4, linewidth=.7)
        ax.set_ylabel(label)
        ax.grid(alpha=.25)
    summaries = [read(p) for p in sorted(HERE.glob('boundary_summary_*.json'))]
    axes[1].plot([s['iteration'] for s in summaries],
                 [sum(r['metrics']['ee_error_mean_cm'] for r in s['results']) / 4 for s in summaries],
                 'o-', label='Independent validation: four conditions')
    axes[1].legend()
    axes[-1].set_xlabel('Completed PPO iterations; fresh continuous training')
    fig.tight_layout()
    fig.savefig(HERE / 'training_trend.png', dpi=160)
    fig.savefig(HERE / 'training_trend.pdf')
    plt.close(fig)


def supervise():
    with (HERE / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (HERE / 'launch.json').exists(), 'Launch already consumed; no retries'
        protect()
        launch = dict(supervisor_pid=os.getpid(), started_utc=now(), command=sys.argv,
                      process_start_ticks=Path('/proc/self/stat').read_text().split()[21])
        write(HERE / 'launch.json', launch)
        runtime_env = os.environ.copy()
        runtime_env.update(LD_LIBRARY_PATH='/home/sorry/.cache/unifp/isaacgym_preview4/py38/lib',
                           PATH='/home/sorry/.cache/unifp/isaacgym_preview4/py38/bin:' + runtime_env['PATH'],
                           OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', MAX_JOBS='2',
                           MPLBACKEND='Agg', WANDB_MODE='offline', RANGER_ASSET_VERSION='v3_original')
        for role in ('preflight', 'train'):
            if role == 'preflight' and (HERE / 'preflight.json').exists():
                assert read(HERE / 'preflight.json')['state'] == 'complete'
                continue
            if role == 'train':
                assert read(HERE / 'preflight.json')['state'] == 'complete'
            with (HERE / f'{role}.log').open('x') as log:
                child = subprocess.Popen([PYTHON, str(HERE / 'run.py'), role],
                         stdout=log, stderr=subprocess.STDOUT, env=runtime_env)
                launch[role + '_pid'] = child.pid
                launch[role + '_start_ticks'] = Path(f'/proc/{child.pid}/stat').read_text().split()[21]
                write(HERE / 'launch.json', launch)
                write(HERE / 'status.json', dict(state=role, pid=child.pid, supervisor_pid=os.getpid(), updated_utc=now()))
                rc = child.wait()
            assert rc == 0, (role, rc)
        assert read(HERE / 'completion.json')['iteration'] == 8000
        protect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('role', choices=['supervise', 'preflight', 'train', 'evaluate', 'plot'])
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--action-mode', choices=['mean', 'sample'])
    args = parser.parse_args()
    try:
        protect()
        if args.role == 'supervise': supervise()
        elif args.role == 'preflight': train(True)
        elif args.role == 'train': train(False)
        elif args.role == 'evaluate': evaluate(args)
        elif args.role == 'plot': plot()
    except BaseException as error:
        failure = dict(state='failed', role=args.role, pid=os.getpid(), error=repr(error),
                       traceback=traceback.format_exc(), failed_utc=now(), automatic_restart=False)
        path = Path(args.output) / 'failure.json' if args.output else HERE / f'failure_{args.role}.json'
        write(path, failure)
        if args.role != 'evaluate':
            write(HERE / 'status.json', failure)
            iteration = read(HERE / 'train/status.json').get('iteration', 0) if (HERE / 'train/status.json').exists() else 0
            update_report('failed', iteration, repr(error))
        raise


if __name__ == '__main__':
    main()

"""Frozen force-stage validation. Each trajectory owns a fresh Gym process.

The original evaluator is retained. No training source or checkpoint is edited.
"""
import argparse
import copy
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'gym_ranger_fresh512_fixedcom_20261028_v1'
OUT = HERE / 'evaluation_verified'
PYTHON = '/home/sorry/.cache/unifp/isaacgym_preview4/torch231/bin/python'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def base_module():
    spec = importlib.util.spec_from_file_location('frozen_ranger_validation', BASE / 'run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def jobs():
    return [(iteration, seed, mode, force) for iteration in (8000, 9000)
            for seed in (266391, 266392) for mode in ('mean', 'sample')
            for force in (False, True)]


def checkpoint(iteration):
    return (BASE if iteration == 8000 else HERE) / f'train/model_iter_{iteration:06d}.pt'


def job_dir(job):
    iteration, seed, mode, force = job
    return OUT / f'i{iteration:05d}_s{seed}_{mode}_{"force" if force else "zero"}'


def score(t, library, torch_mode=False):
    """Called with Torch in workers and NumPy float64 in the supervisor."""
    if torch_mode:
        norm = lambda x: x.square().sum(-1).sqrt()
        rmse = lambda x: float(x.square().mean().sqrt())
        mean = lambda x: float(x.mean())
        count = lambda x: int(x.sum())
    else:
        norm = lambda x: library.sqrt(library.square(x).sum(-1))
        rmse = lambda x: float(library.sqrt(library.square(x).mean()))
        mean = lambda x: float(x.mean())
        count = lambda x: int(x.sum())
    truth = t['target'][..., 6:9] * 100.
    prediction = t['predictions'][..., 6:9] * 100.
    full = t['full_history']
    nonzero = full & (norm(truth) > 0.)
    zero = full & t['sustained_zero']
    error = prediction - truth
    force_rmse = rmse(error[full]) if count(full) else None
    baseline = rmse(truth[full]) if count(full) else None
    result = {
        'ee_error_mean_cm': mean(norm(t['ee_error'])) * 100.,
        'ee_force_target_error_mean_cm': mean(norm(t['ee_force_target_error'])) * 100.,
        'xy_rmse_m_s': rmse(t['velocity_error'][..., :2]),
        'yaw_rmse_rad_s': rmse(t['velocity_error'][..., 2]),
        'force_estimate_rmse_per_axis_n': force_rmse,
        'zero_predictor_rmse_per_axis_n': baseline,
        'force_rmse_improvement_over_zero_n': baseline - force_rmse if baseline is not None else None,
        'nonzero_force_vector_error_rmse_n': rmse(norm(error[nonzero])) if count(nonzero) else None,
        'nonzero_zero_predictor_vector_rmse_n': rmse(norm(truth[nonzero])) if count(nonzero) else None,
        'sustained_zero_force_estimate_norm_n': mean(norm(prediction[zero])) if count(zero) else None,
        'non_timeout_terminations': count(t['done'] & ~t['timeout']),
        'timeouts': count(t['done'] & t['timeout']),
        'full_history_samples': count(full),
        'nonzero_force_samples': count(nonzero),
        'sustained_zero_samples': count(zero),
        'mean_reward_per_step': mean(t['reward']),
        'velocity_prediction_rmse_m_s': rmse((t['predictions'][..., :3] - t['target'][..., :3]) / 2.),
        'ee_state_prediction_mse_normalized': rmse(t['predictions'][..., 3:6] - t['target'][..., 3:6]) ** 2,
        'actual_force_max_n': float(abs(t['applied_force']).max()),
        'command_force_max_n': float(abs(t['force_command']).max()),
    }
    return result


def worker(index):
    job = jobs()[index]
    iteration, seed, mode, force = job
    out = job_dir(job)
    out.mkdir(exist_ok=False)
    base = base_module()
    base.protect()
    # Gym must be imported by the backend before importing Torch.
    env, ns = base.native_backend(32, seed)
    import torch
    import numpy as np
    guarded = env.robot.set_external_force_and_torque
    original = next(c.cell_contents for c in guarded.__closure__
                    if callable(c.cell_contents) and getattr(c.cell_contents, '__name__', '') == 'set_external_force_and_torque')
    env.robot.set_external_force_and_torque = original
    env.force_curriculum_enabled = force
    env.curriculum.start_step = 192001
    env.global_policy_steps = 192001 if force else 0
    model, alg = base.learner(env)
    ck = torch.load(checkpoint(iteration), map_location='cpu', weights_only=False)
    assert ck['iterations'] == iteration
    model.load_state_dict(ck['model_state_dict'], strict=True)
    model.eval().requires_grad_(False)
    initial_weights = copy.deepcopy(model.state_dict())
    torch.save(dict(physics=env.state_snapshot(), masses=env.robot.root_physx_view.get_masses(),
                    coms=env.robot.root_physx_view.get_coms(),
                    materials=env.robot.root_physx_view.get_material_properties(),
                    motor_strength=env.domain.motor_strength.clone(),
                    observations=env.get_observations(), rng=base.rng()), out / 'initial_state.pt')
    pre = {}
    compute = env.rewards.compute

    def capture(robot, feet, contacts, controller, actions, torques, commands,
                goal, applied, base_force, frame, slices):
        cfg = env.shared_state.pulse.cfg.commands
        moving = ((commands[:, 0].abs() > cfg.lin_vel_x_clip) |
                  (commands[:, 1].abs() > cfg.lin_vel_y_clip) |
                  (commands[:, 2].abs() > cfg.ang_vel_yaw_clip))
        xy_target = commands[:, :2] * moving[:, None]
        ee_error = robot.data.body_pos_w[:, env.gripper_id] - goal.goal_world
        cmd_world = ns['quat_apply'](ns['yaw_quat'](robot.data.root_quat_w), commands[:, 9:12])
        offset = (applied + cmd_world) / env.shared_state.pulse.gripper_force_kps
        pre.update(ee_error=ee_error.clone(), ee_force_target_error=(ee_error-offset).clone(),
                   velocity_error=torch.cat((robot.data.root_lin_vel_b[:, :2]-xy_target,
                                             robot.data.root_ang_vel_b[:, 2:3]-commands[:, 2:3]), -1).clone(),
                   applied_force=applied.clone(), force_command=commands[:, 9:12].clone())
        return compute(robot, feet, contacts, controller, actions, torques, commands,
                       goal, applied, base_force, frame, slices)

    env.rewards.compute = capture
    x = env.get_observations()
    history = torch.ones(32, dtype=torch.long)
    zero_streak = torch.zeros(32, dtype=torch.long)
    generator = torch.Generator().manual_seed(seed + 190000)
    trace = {}

    def add(key, value):
        trace.setdefault(key, []).append(value.detach().clone())

    with torch.inference_mode():
        for _ in range(1000):
            truth = x['obs_pred'][:, 6:9]
            zero_streak = torch.where(truth.abs().sum(-1) == 0, zero_streak + 1, torch.zeros_like(zero_streak))
            add('target', x['obs_pred'])
            add('predictions', model.get_student_latent(x['obs']))
            add('full_history', history >= 32)
            add('sustained_zero', zero_streak >= 32)
            action = model.act_inference(x)
            if mode == 'sample':
                action = action + model.std * torch.randn(action.shape, generator=generator)
            add('actions', action)
            x, reward, done, info = env.step(action)
            for key, value in pre.items():
                add(key, value)
            add('done', done)
            add('timeout', info['time_outs'])
            add('reward', reward)
            history += 1
            history[done] = 1
            zero_streak[done] = 0
    trace = {key: torch.stack(values) for key, values in trace.items()}
    assert all(torch.isfinite(value).all() for value in trace.values())
    assert base.equal(initial_weights, model.state_dict())
    assert base.optimizer_steps(alg) == {'ppo': [], 'auxiliary': []}
    metrics = score({k: v if v.dtype == torch.bool else v.double() for k, v in trace.items()}, torch, True)
    assert metrics['actual_force_max_n'] <= 60.0001 and metrics['command_force_max_n'] <= 60.0001
    if not force:
        assert metrics['actual_force_max_n'] == metrics['command_force_max_n'] == 0
        assert not trace['target'][..., 6:12].count_nonzero()
    else:
        assert metrics['actual_force_max_n'] > 0 and metrics['command_force_max_n'] > 0
    np.savez_compressed(out / 'trace.npz', **{k: v.numpy() for k, v in trace.items()})
    base.protect()
    write(out / 'result.json', dict(state='complete', iteration=iteration, seed=seed, action_mode=mode,
          force_curriculum=force, metrics=metrics, checkpoint_sha256=sha(checkpoint(iteration)),
          trace_sha256=sha(out / 'trace.npz'), initial_state_sha256=sha(out / 'initial_state.pt'),
          model_unchanged=True, optimizer_updates=0, completed_utc=now()))
    # Evidence is durable; use the existing validated worker exit convention.
    sys.stdout.flush()
    os._exit(0)


def supervise():
    OUT.mkdir(exist_ok=False)
    completion = json.loads((HERE / 'completion.json').read_text())
    expected = {8000: completion['source_checkpoint_sha256'], 9000: completion['final_checkpoint_sha256']}
    for iteration, digest in expected.items():
        assert sha(checkpoint(iteration)) == digest
    write(HERE / 'evaluation_launch.json', dict(pid=os.getpid(), started_utc=now(),
          evaluator_sha256=sha(__file__), original_evaluator_sha256=sha(HERE / 'evaluate_stage.py'),
          jobs=jobs(), checkpoint_sha256=expected, optimizer_updates=0,
          note='Corrected isolated evaluator; original retained. Exact-zero grouping; no small-force gate.'))
    results = []
    import numpy as np
    for index, job in enumerate(jobs()):
        with (OUT / f'job_{index:02d}.log').open('w') as log:
            child = subprocess.Popen([PYTHON, __file__, '--worker', str(index)], stdout=log, stderr=subprocess.STDOUT)
            write(HERE / 'evaluation_status.json', dict(state='running', completed=len(results), total=16,
                  current_job=job, worker_pid=child.pid, supervisor_pid=os.getpid(), updated_utc=now()))
            rc = child.wait()
        if rc != 0:
            raise RuntimeError(f'Validation job {index} exited {rc}; evidence retained, no automatic retry')
        out = job_dir(job)
        result = json.loads((out / 'result.json').read_text())
        assert result['trace_sha256'] == sha(out / 'trace.npz')
        assert result['checkpoint_sha256'] == expected[job[0]]
        with np.load(out / 'trace.npz') as z:
            trace = {k: z[k] if z[k].dtype == np.bool_ else z[k].astype(np.float64) for k in z.files}
        metrics = score(trace, np)
        differences = []
        for key, value in metrics.items():
            saved = result['metrics'][key]
            assert (value is None) == (saved is None), key
            if value is not None:
                differences.append(abs(value-saved))
                assert np.isclose(value, saved, rtol=1e-10, atol=1e-10), (key, value, saved)
        result['independent_float64_recomputation_max_difference'] = max(differences)
        result['directory'] = str(out)
        results.append(result)
        write(out / 'independent_verification.json', dict(passed=True, metrics=metrics,
              maximum_difference=max(differences), result_sha256=sha(out / 'result.json'),
              trace_sha256=sha(out / 'trace.npz')))
        print('VALIDATION_COMPLETED', index+1, '/16', job, flush=True)
    base_module().protect()
    for iteration, digest in expected.items():
        assert sha(checkpoint(iteration)) == digest
    write(HERE / 'evaluation_summary.json', dict(state='complete', results=results,
          checkpoint_sha256=expected, all_16_independent_recomputations_passed=True,
          protected_source_sha_passed=True, independent_training_seeds=1,
          note='Same validation seeds do not guarantee bitwise identical physical trajectories.', completed_utc=now()))
    write(HERE / 'evaluation_status.json', dict(state='complete', completed=16, total=16, completed_utc=now()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int)
    args = parser.parse_args()
    try:
        worker(args.worker) if args.worker is not None else supervise()
    except BaseException as exc:
        destination = (job_dir(jobs()[args.worker]) / 'failure.json') if args.worker is not None else (HERE / 'evaluation_failure.json')
        write(destination, dict(error=repr(exc), traceback=traceback.format_exc(), failed_utc=now(), automatic_retry=False))
        if args.worker is None:
            write(HERE / 'evaluation_status.json', dict(state='failed', error=repr(exc), automatic_retry=False))
        raise

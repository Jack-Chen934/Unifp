"""顺序调度恢复验收、两组200轮训练和原冻结评估。"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


SOURCE = Path('/mnt/data_8THDD/myl_unifp_v3_stage1/validation/b_stage_20261107_r01/source_resume12000_20261108/UniFP_v3')
ORIGIN = Path('/mnt/data_8THDD/myl_unifp_v3_stage1/runs/b32000_seed1_v3_v6r2_from_a8000_20261108/checkpoints/model_009000.pt')


def write_status(root: Path, state: dict[str, Any]) -> None:
    """Publish the orchestrator's last committed phase atomically.

    Args:
        root: Existing experiment evidence directory.
        state: JSON-compatible status including child commands and return codes.

    Raises:
        OSError: Status cannot be persisted.
    """
    temporary = root / '.status.tmp'
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(root / 'status.json')


def execute(name: str, command: list[str], root: Path, state: dict[str, Any]) -> None:
    """Run one isolated stage and retain its exact command and console output.

    Args:
        name: Unique smoke, training or evaluation stage identifier.
        command: Explicit argv executed without a shell in the frozen source root.
        root: Independent experiment directory owning all stage logs.
        state: Mutable orchestrator status, published before and after execution.

    Raises:
        RuntimeError: The child exits with an execution/interface error.
        FileExistsError: A stage log already exists; old evidence is not overwritten.

    Note:
        Ordinary failed episodes or unmet performance thresholds still return zero
        from the existing evaluators and do not prevent the next planned stage.
    """
    row = {'name': name, 'command': command, 'started_at': datetime.now().astimezone().isoformat()}
    state['stages'].append(row)
    state['active_stage'] = name
    write_status(root, state)
    with (root / f'{name}.console.log').open('x') as console:
        result = subprocess.run(command, cwd=SOURCE, stdout=console, stderr=subprocess.STDOUT)
    row.update(returncode=result.returncode, ended_at=datetime.now().astimezone().isoformat())
    write_status(root, state)
    if result.returncode:
        raise RuntimeError(f'Stage {name} exited with code {result.returncode}')


def train_command(root: Path, group: str, coefficient: float, iterations: int, checkpoint: Path) -> list[str]:
    """Build the fixed-schema trial command for one approved budget.

    Args:
        root: Evidence root containing the diagnostic launcher.
        group: Unique new run directory basename.
        coefficient: Control 0.01 or experimental zero entropy coefficient.
        iterations: Additional complete updates, at most 200.
        checkpoint: Original B9000 or this experiment's validated smoke checkpoint.

    Returns:
        Exact argv for a fresh output run using physical GPU3 via inherited env.
    """
    return [sys.executable, str(root / 'run_entropy_trial.py'), '--origin', str(ORIGIN),
            '--checkpoint', str(checkpoint), '--output', str(root / group),
            '--iterations', str(iterations), '--entropy-coef', str(coefficient)]


def evaluate(checkpoint: Path, name: str, root: Path, state: dict[str, Any]) -> None:
    """Run all approved frozen zero-force, R2 and contact/zero-control protocols.

    Args:
        checkpoint: Exact complete parent or 9200 checkpoint, never modified.
        name: Evaluation object label used in independent output paths.
        root: Evidence root receiving stage logs and immutable archives.
        state: Mutable published orchestration state.

    Raises:
        AssertionError: The A-reference manifest differs beyond observation schedule.
        RuntimeError: An evaluator has an execution error or changes policy parameters.
    """
    evidence = json.loads((SOURCE.parents[1] / 'resume_evidence12000_20261108/start_evidence.json').read_text())
    reference = Path(evidence['stages']['evaluation_baseline']['force_reference_archives']['R2'])
    force_manifest = reference / 'R2/evaluation_manifest.json'
    sealed, current = json.loads(force_manifest.read_text()), json.loads((SOURCE / 'configs/v3_b_initial_v1.json').read_text())
    assert {key: value for key, value in sealed.items() if key != 'observation_schedule'} == {
        key: value for key, value in current.items() if key != 'observation_schedule'}
    common = ['--reproduction_part', 'B', '--run_purpose', 'evaluation', '--checkpoint', str(checkpoint),
        '--num_envs', '32', '--sim_device', 'cuda:0', '--rl_device', 'cuda:0', '--pipeline', 'gpu',
        '--num_threads', '4', '--headless', '--seed', '1001']
    destination = root / 'evaluations' / name
    commands = [
        ('zero_force', [sys.executable, '-m', 'legged_gym.scripts.evaluate_v3', *common,
            '--protocol', 'zero_force_regression', '--manifest', str(SOURCE / 'configs/v3_a_initial_v1.json'),
            '--output_dir', str(destination / 'zero_force')]),
        ('R2', [sys.executable, '-m', 'legged_gym.scripts.evaluate_v3', *common,
            '--protocol', 'force_tracking', '--manifest', str(force_manifest), '--scenario', 'R2',
            '--reference_archive', str(reference), '--output_dir', str(destination / 'R2'), '--video']),
        ('contact', [sys.executable, '-m', 'legged_gym.scripts.evaluate_v3_contact', *common,
            '--protocol', 'fixed_wall_contact', '--manifest', str(SOURCE / 'configs/v3_b_contact_v1.json'),
            '--output_dir', str(destination / 'contact'), '--video'])]
    for protocol, command in commands:
        execute(f'evaluate_{name}_{protocol}', command, root, state)
        summary = json.loads((destination / protocol / 'summary.json').read_text())
        assert summary['policy_updates'] == 0


def summarize(root: Path) -> None:
    """Write a compact Chinese comparison while preserving full frozen summaries.

    Args:
        root: Completed experiment directory containing all runs and evaluations.

    Raises:
        AssertionError: A trial budget or evaluation update count is inconsistent.
        OSError: A required completed artifact cannot be read.

    Note:
        Reports observed differences without declaring causal certainty or B success.
        Does not alter thresholds, checkpoints, source files or archived traces.
    """
    result: dict[str, Any] = {'groups': {}, 'evaluations': {}, 'formal_training_modified': False}
    for name in ('control_200', 'no_entropy_200'):
        directory = root / name
        completion = json.loads((directory / 'completion.json').read_text())
        assert completion['completed_iterations'] == 9200 and completion['additional_iterations'] == 200
        logs = [json.loads(line) for line in (directory / 'iterations.jsonl').read_text().splitlines()]
        episodes = [json.loads(line) for line in (directory / 'episodes.jsonl').read_text().splitlines()]
        total_steps = sum(row['steps'] for row in episodes)
        result['groups'][name] = {'completion': completion, 'std_initial': json.loads((directory / 'startup.json').read_text())['std'],
            'std_final': logs[-1]['diagnostic/std_by_action'], 'iterations': len(logs),
            'completed_episodes': len(episodes),
            'failure_fraction_of_completed': sum(row['end_type'] == 'physical_failure' for row in episodes) / len(episodes),
            'completed_episode_step_weighted_tcp_m': sum(row['steps'] * row['errors']['tcp_compensated_l2_m']['mean'] for row in episodes) / total_steps,
            'last_50_joint2_saturation': sum(row['diagnostic/torque_saturation_fraction'][9] for row in logs[-50:]) / 50,
            'mean_iteration_s': sum(row['collection_s'] + row['update_s'] for row in logs) / len(logs)}
    for name in ('parent_9000', 'control_9200', 'no_entropy_9200'):
        result['evaluations'][name] = {}
        for protocol in ('zero_force', 'R2', 'contact'):
            directory = root / 'evaluations' / name / protocol
            result['evaluations'][name][protocol] = {str(path.relative_to(directory)): json.loads(path.read_text())
                for path in directory.rglob('summary.json')}
    (root / 'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    lines = ['# B探索熵200轮独立对照结果', '',
        '两组从同一B9000完整存档开始，4096环境，各200轮；唯一主动学习参数差异为entropy_coef。正式训练未修改。', '',
        '| 组别 | 训练完成episode补偿TCP均值 | 完成episode物理失败率 | 后50轮joint2限幅 |',
        '|---|---:|---:|---:|']
    for name, row in result['groups'].items():
        label = '原熵0.01' if name == 'control_200' else '零熵0'
        lines.append(f"| {label} | {row['completed_episode_step_weighted_tcp_m']*100:.3f} cm | {row['failure_fraction_of_completed']*100:.3f}% | {row['last_50_joint2_saturation']*100:.3f}% |")
    lines += ['', '以上为训练采样轨迹统计，episode可能跨越窗口；不能替代均值策略冻结评估。',
        '父与两终点的S0／S1／S2、R2、接触／0N完整结果见comparison.json和evaluations各冻结summary。',
        '单seed200轮仅用于判断机制趋势，不能据此宣布全部B效果复现；新预算不自动延长。', '']
    (root / '实验结果.md').write_text('\n'.join(lines))


def main() -> None:
    """Run approved stages after the separately launched first control smoke passes.

    Raises:
        AssertionError: Smoke completion, initial-state pairing or a quota is invalid.
        RuntimeError: Any child has an execution error; evidence remains for diagnosis.

    Note:
        Runs sequentially on GPU3 and survives SSH disconnects when detached by the
        launcher. Publishes error status on failure and never restarts formal runs.
    """
    root = Path(__file__).resolve().parent
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '3'
    assert json.loads((root / 'smoke_control/completion.json').read_text())['completed_iterations'] == 9010
    state: dict[str, Any] = {'pid': os.getpid(), 'gpu': 3, 'state': 'running', 'stages': [],
        'started_at': datetime.now().astimezone().isoformat(), 'source': str(SOURCE), 'origin': str(ORIGIN),
        'scripts_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in root.glob('*.py')}}
    write_status(root, state)
    try:
        execute('smoke_no_entropy_10', train_command(root, 'smoke_no_entropy', 0., 10, ORIGIN), root, state)
        first = json.loads((root / 'smoke_control/startup.json').read_text())
        second = json.loads((root / 'smoke_no_entropy/startup.json').read_text())
        assert first['initial_state_digests'] == second['initial_state_digests']
        (root / 'paired_startup_verification.json').write_text(json.dumps({
            'identical_observations_properties_execution_root_q_rng': True,
            'initial_state_digests': first['initial_state_digests'], 'restored_learning_state_equal_both': True}, indent=2) + '\n')
        for group, coefficient in (('control', .01), ('no_entropy', 0.)):
            execute(f'smoke_{group}_resume_1', train_command(root, f'smoke_{group}_resume', coefficient, 1,
                root / f'smoke_{group}/checkpoints/model_009010.pt'), root, state)
            assert json.loads((root / f'smoke_{group}_resume/completion.json').read_text())['completed_iterations'] == 9011
        evaluate(ORIGIN, 'parent_9000', root, state)
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

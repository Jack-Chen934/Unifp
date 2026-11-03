"""Summarize actual ticket-07 logs and checkpoints without running simulation."""

from collections import Counter
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, Tuple

import torch

VALIDATION_ROOT = Path('/mnt/data_8THDD/myl_unifp_v3_stage1/validation/ticket07_20261102_r02')
PROJECT_ROOT = VALIDATION_ROOT / 'UniFP_v3'
sys.path.insert(0, str(PROJECT_ROOT))
from legged_gym.b2_gym_learn.ppo_cse_pf.v3_run_identity import build_run_identity, file_sha256


def read_runtime_result(filename: str) -> Dict[str, Any]:
    """Read the final JSON result from an already completed Gym check.

    Args:
        filename: Console-log basename inside the validation evidence directory.

    Returns:
        The check's final one-line JSON result.

    Raises:
        OSError: The log cannot be read.
        ValueError: No final JSON result exists.
    """
    for line in reversed((VALIDATION_ROOT / 'evidence' / filename).read_text().splitlines()):
        if line.startswith('{'):
            return json.loads(line)
    raise ValueError(f'检查未输出JSON结果：{filename}')


def inspect_run(name: str, num_envs: int, iterations: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Check run counters, finite values, optimizer progress and source identity.

    Args:
        name: Run directory basename, containing metadata and checkpoint outputs.
        num_envs: Expected number of parallel environments.
        iterations: Expected cumulative completed PPO updates.

    Returns:
        A JSON-compatible result and the CPU-loaded checkpoint for restore checks.

    Raises:
        OSError: A required run output cannot be read.
        ValueError: A required output is malformed.

    Note:
        Loads checkpoints onto CPU and does not construct Gym or an optimizer.
    """
    run_root = VALIDATION_ROOT / 'runs' / name
    completion = json.loads((run_root / 'completion.json').read_text())
    metadata = json.loads((run_root / 'episode_metadata.json').read_text())
    iteration_rows = [json.loads(line) for line in (run_root / 'iterations.jsonl').read_text().splitlines()]
    episodes_path = run_root / 'episodes.jsonl'
    episodes = [json.loads(line) for line in episodes_path.read_text().splitlines()] if episodes_path.exists() else []
    checkpoint_path = run_root / 'checkpoints' / f'model_{iterations:06d}.pt'
    payload = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    actual_identity = build_run_identity(PROJECT_ROOT, payload['config_snapshot'], payload['identity']['runtime'])
    expected_progress = {'completed_iterations': iterations, 'training_policy_steps': iterations * 24,
                         'training_transitions': iterations * 24 * num_envs, 'rollout_step': 0}
    optimizer_steps = {
        prefix: sorted({float(state['step']) for state in payload[prefix + '_optimizer_state_dict']['state'].values()})
        for prefix in ('main', 'auxiliary')
    }
    algorithm_config = payload['config_snapshot']['training']['algorithm']
    expected_adam_steps = iterations * algorithm_config['num_learning_epochs'] * algorithm_config['num_mini_batches']
    console = (VALIDATION_ROOT / 'evidence' / f'{name}.console.log').read_text()
    checks = {
        'approved_contact_pair_capacity': payload['config_snapshot']['environment']['sim']['physx']['max_gpu_contact_pairs'] == 2**25,
        'official_buffer_multiplier_preserved': payload['config_snapshot']['environment']['sim']['physx']['default_buffer_size_multiplier'] == 5,
        'checkpoint_progress': payload['progress'] == expected_progress,
        'completion_counters': all(completion[key] == expected_progress[key] for key in
                                   ('completed_iterations', 'training_policy_steps', 'training_transitions')),
        'checkpoint_identity': actual_identity == payload['identity'] == metadata['identity'],
        'main_adam_steps': optimizer_steps['main'] == [float(expected_adam_steps)],
        'auxiliary_adam_steps': optimizer_steps['auxiliary'] == [float(expected_adam_steps)],
        'finite_iteration_metrics': all(math.isfinite(value) for row in iteration_rows for value in row.values()
                                        if isinstance(value, (float, int))),
        'finite_model': all(bool(torch.isfinite(value).all()) for value in payload['model_state_dict'].values()),
        'expected_environment_count': payload['environment_training_state']['num_envs'] == num_envs,
        'no_missing_physics_interactions': 'simulation will miss interactions' not in console,
        'iteration_sequence': [row['completed_iterations'] for row in iteration_rows] ==
                              list(range(iterations - len(iteration_rows) + 1, iterations + 1)),
        'iteration_counters': all(row['training_policy_steps'] == row['completed_iterations'] * 24 and
                                  row['training_transitions'] == row['training_policy_steps'] * num_envs
                                  for row in iteration_rows),
    }
    durations = [row['duration_s'] for row in episodes]
    tcp_errors = [row['errors']['tcp_geometric_l2_m']['mean'] for row in episodes]
    result = {
        'passed': all(checks.values()), 'checks': checks, 'completion': completion,
        'effective_physx_config': payload['config_snapshot']['environment']['sim']['physx'],
        'identity': payload['identity'], 'checkpoint_sha256': file_sha256(checkpoint_path),
        'checkpoint_path': str(checkpoint_path), 'optimizer_steps': optimizer_steps,
        'raw_output_sha256': {path.name: file_sha256(path) for path in
                              (run_root / 'completion.json', run_root / 'episode_metadata.json',
                               run_root / 'iterations.jsonl', run_root / 'checkpoints.jsonl')},
        'collection_s': sum(row['collection_s'] for row in iteration_rows),
        'update_s': sum(row['update_s'] for row in iteration_rows),
        'finished_episodes': len(episodes), 'end_types': dict(Counter(row['end_type'] for row in episodes)),
        'distinct_failed_envs': len({row['env_id'] for row in episodes if row['end_type'] == 'physical_failure'}),
        'mean_finished_episode_duration_s': sum(durations) / len(durations) if durations else None,
        'mean_finished_episode_tcp_l2_m': sum(tcp_errors) / len(tcp_errors) if tcp_errors else None,
        'first_policy_step_physical_failures': sum(row['end_type'] == 'physical_failure' and
                                                  row['training_policy_steps'] == (iterations - 1) * 24
                                                  for row in episodes) if name == 'resume512' else None,
    }
    return result, payload


if '--capacity-probes' in sys.argv:
    probe_results = {}
    for probe_name in ('capacity4096_buffer32', 'capacity4096_pairs25'):
        probe_results[probe_name], _ = inspect_run(probe_name, 4096, 1)
    probe_summary_path = VALIDATION_ROOT / 'evidence' / 'capacity_probes_summary.json'
    with probe_summary_path.open('x', encoding='utf-8') as probe_file:
        json.dump(probe_results, probe_file, ensure_ascii=False, allow_nan=False, indent=2)
        probe_file.write('\n')
    print(json.dumps({name: {key: value for key, value in result.items() if key != 'identity'}
                      for name, result in probe_results.items()}, ensure_ascii=False, indent=2))
    raise SystemExit(0)


nominal_result = read_runtime_result('nominal32.console.log')
randomization_result = read_runtime_result('randomization8.console.log')
run_results: Dict[str, Any] = {}
checkpoints: Dict[str, Any] = {}
for run_name, environment_count, iteration_count in (
    ('smoke32', 32, 1), ('train512', 512, 10), ('resume512', 512, 11), ('capacity4096', 4096, 1)
):
    run_results[run_name], checkpoints[run_name] = inspect_run(run_name, environment_count, iteration_count)

parent = checkpoints['train512']
resumed = checkpoints['resume512']
restore_checks = {
    'same_identity': parent['identity'] == resumed['identity'],
    'parent_checkpoint_sha256': resumed['lineage']['parent_checkpoint_sha256'] == run_results['train512']['checkpoint_sha256'],
    'creation_mass_com_preserved': torch.equal(parent['environment_training_state']['mass_com_delta'],
                                             resumed['environment_training_state']['mass_com_delta']),
    'creation_friction_preserved': torch.equal(parent['environment_training_state']['friction'],
                                             resumed['environment_training_state']['friction']),
    'first_resume_step_no_physical_failure': run_results['resume512']['first_policy_step_physical_failures'] == 0,
}
summary = {
    'schema_version': 1, 'nominal32': nominal_result, 'randomization8': randomization_result,
    'runs': run_results, 'restore_checks': restore_checks,
    'short_checks_passed': all(result['passed'] for result in run_results.values()) and all(restore_checks.values()),
    'formal_training_ready': False,
    'remaining': ['A10绝对阈值待确认', '正式训练配置与环境数尚未冻结', 'A8000尚未启动，未评价复现效果'],
}
unit_log = (VALIDATION_ROOT / 'evidence' / 'unit_cpu.console.log').read_text()
reward_log = (VALIDATION_ROOT / 'evidence' / 'reward_unit.console.log').read_text()
stage_records = {
    'unit_tests': {
        'passed': '31 passed' in unit_log and 'Ran 5 tests' in reward_log and '\nOK\n' in reward_log,
        'identity': checkpoints['smoke32']['identity'], 'local_cpu_tests': 31, 'remote_reward_tests': 5,
    },
    'gym_smoke': {
        'passed': run_results['smoke32']['passed'] and nominal_result['非超时终止'] == 0,
        'identity': checkpoints['smoke32']['identity'], 'num_envs': 32,
        'zero_ee_command_force': nominal_result['零末端力命令'],
        'zero_ee_external_force': randomization_result['push单次触发且末端力标签为零'],
        'finite_core': run_results['smoke32']['passed'],
        'reset_isolated': nominal_result['局部reset不影响其他环境'],
    },
    'training_resume': {
        'passed': run_results['train512']['passed'] and run_results['resume512']['passed'] and all(restore_checks.values()),
        'identity': parent['identity'], 'num_envs': 512, 'iterations': 10,
        'resumed_iterations': resumed['progress']['completed_iterations'] - parent['progress']['completed_iterations'],
        'completed_iterations': resumed['progress']['completed_iterations'],
        'counters_consistent': run_results['train512']['checks']['iteration_counters'] and
                               run_results['resume512']['checks']['iteration_counters'],
        'checkpoint_identity_matches': restore_checks['same_identity'],
    },
    'capacity': {
        'passed': run_results['capacity4096']['passed'], 'identity': checkpoints['capacity4096']['identity'],
        'num_envs': 4096, 'rollout_update': run_results['capacity4096']['passed'],
        'formal_config_sha256': checkpoints['capacity4096']['identity']['learning_config_sha256'],
        'config_status': 'candidate_preflight',
    },
}
stage_logs = {
    'unit_tests': ('unit_cpu.console.log', 'reward_unit.console.log'),
    'gym_smoke': ('nominal32.console.log', 'randomization8.console.log', 'smoke32.console.log'),
    'training_resume': ('train512.console.log', 'resume512.console.log'),
    'capacity': ('capacity4096.console.log',),
}
for stage_name, stage_result in stage_records.items():
    stage_result['raw_artifacts'] = [
        {'path': str(VALIDATION_ROOT / 'evidence' / filename),
         'sha256': file_sha256(VALIDATION_ROOT / 'evidence' / filename)}
        for filename in stage_logs[stage_name]
    ]
    stage_path = VALIDATION_ROOT / 'evidence' / f'{stage_name}.report.json'
    with stage_path.open('x', encoding='utf-8') as stage_file:
        json.dump({'stage': stage_name, 'result': stage_result}, stage_file,
                  ensure_ascii=False, allow_nan=False, indent=2)
        stage_file.write('\n')
    stage_result['artifact'] = {'path': str(stage_path), 'sha256': file_sha256(stage_path)}
summary['stages'] = stage_records
summary['short_checks_passed'] = summary['short_checks_passed'] and all(
    stage['passed'] for stage in stage_records.values()
)
summary_path = VALIDATION_ROOT / 'evidence' / 'verified_summary.json'
with summary_path.open('x', encoding='utf-8') as output_file:
    json.dump(summary, output_file, ensure_ascii=False, allow_nan=False, indent=2)
    output_file.write('\n')
print(json.dumps({
    'short_checks_passed': summary['short_checks_passed'], 'restore_checks': restore_checks,
    'runs': {name: {key: value for key, value in result.items() if key not in ('identity',)}
             for name, result in run_results.items()},
}, ensure_ascii=False, allow_nan=False, indent=2))

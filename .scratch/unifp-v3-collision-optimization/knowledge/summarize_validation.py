"""核验独立碰撞资产短测的真实计数、身份与正式启动证据。"""

from collections import Counter
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, Tuple

import torch
import numpy as np

VALIDATION_ROOT = Path('/mnt/data_8THDD/myl_unifp_v3_stage1/validation/collision_optimization_20261103_r02')
PROJECT_ROOT = VALIDATION_ROOT / 'source/UniFP_v3'
sys.path.insert(0, str(PROJECT_ROOT))
from legged_gym.b2_gym_learn.ppo_cse_pf.v3_run_identity import build_run_identity, file_sha256
from legged_gym.b2_gym_learn.ppo_cse_pf.v3_preflight import assess_a_start_gate, validate_stage_artifacts


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
    run_root = VALIDATION_ROOT / name
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
    console = (VALIDATION_ROOT / f'{name}.console.log').read_text()
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


def read_runtime_result(filename: str) -> Dict[str, Any]:
    """Read the final JSON emitted by an already completed runtime check.

    Args:
        filename: Console-log basename relative to the validation directory.

    Returns:
        Final JSON object containing the original check's assertions.

    Raises:
        ValueError: The log has no JSON result.
        OSError: The log is unreadable.
    """
    for line in reversed((VALIDATION_ROOT / filename).read_text().splitlines()):
        if line.startswith('{'):
            return json.loads(line)
    raise ValueError(f'检查未输出JSON结果：{filename}')


def main() -> None:
    """Verify real short runs and bind each gate stage to immutable evidence.

    Returns:
        None.

    Raises:
        AssertionError: An expected run, restore or identity check fails.
        OSError: A required artifact cannot be read or a report already exists.

    Note:
        Reads checkpoints onto CPU only, never creates Gym or starts training.
        Writes new stage reports and a gate report; preserves all raw outputs.
    """
    results: Dict[str, Any] = {}
    checkpoints: Dict[str, Any] = {}
    for name, num_envs, iterations in (('run512', 512, 10), ('resume512', 512, 11),
                                      ('run4096', 4096, 100), ('smoke32', 32, 1),
                                      ('capacity4096', 4096, 1)):
        results[name], checkpoints[name] = inspect_run(name, num_envs, iterations)
    parent, resumed = checkpoints['run512'], checkpoints['resume512']
    restore_checks = {
        'same_identity': parent['identity'] == resumed['identity'],
        'parent_checkpoint_sha256': resumed['lineage']['parent_checkpoint_sha256'] == results['run512']['checkpoint_sha256'],
        'creation_mass_com_preserved': torch.equal(parent['environment_training_state']['mass_com_delta'], resumed['environment_training_state']['mass_com_delta']),
        'creation_friction_preserved': torch.equal(parent['environment_training_state']['friction'], resumed['environment_training_state']['friction']),
        'first_resume_step_no_physical_failure': results['resume512']['first_policy_step_physical_failures'] == 0,
    }
    rejection = json.loads((VALIDATION_ROOT / 'old_resume_rejection.json').read_text())
    old_payload = torch.load(rejection['checkpoint'], map_location='cpu', weights_only=False)
    cross_asset_resume_rejected = rejection['passed'] and old_payload['identity']['asset_manifest'] != parent['identity']['asset_manifest']

    cpu_tests = json.loads((VALIDATION_ROOT / 'local_cpu_tests.json').read_text())
    nominal = read_runtime_result('nominal32.console.log')
    randomization = read_runtime_result('randomization8.console.log')
    identity = checkpoints['capacity4096']['identity']
    stages = {
        'unit_tests': {
            'passed': cpu_tests['exit_code'] == 0 and cpu_tests['passed_tests'] == 48 and
                      cpu_tests['source_manifest'] == identity['source_manifest'] and
                      'Ran 5 tests' in (VALIDATION_ROOT / 'reward_unit.console.log').read_text() and
                      '\nOK\n' in (VALIDATION_ROOT / 'reward_unit.console.log').read_text(),
            'identity': identity, 'local_cpu_tests': 48, 'remote_reward_tests': 5,
        },
        'gym_smoke': {
            'passed': results['smoke32']['passed'] and nominal['非超时终止'] == 0,
            'identity': checkpoints['smoke32']['identity'], 'num_envs': 32,
            'zero_ee_command_force': nominal['零末端力命令'],
            'zero_ee_external_force': randomization['push单次触发且末端力标签为零'],
            'finite_core': results['smoke32']['passed'],
            'reset_isolated': nominal['局部reset不影响其他环境'],
        },
        'training_resume': {
            'passed': results['run512']['passed'] and results['resume512']['passed'] and all(restore_checks.values()),
            'identity': parent['identity'], 'num_envs': 512, 'iterations': 10,
            'resumed_iterations': 1, 'completed_iterations': 11,
            'counters_consistent': results['run512']['checks']['iteration_counters'] and results['resume512']['checks']['iteration_counters'],
            'checkpoint_identity_matches': restore_checks['same_identity'],
        },
        'capacity': {
            'passed': results['capacity4096']['passed'], 'identity': identity,
            'num_envs': 4096, 'rollout_update': results['capacity4096']['passed'],
            'formal_config_sha256': identity['learning_config_sha256'],
            'config_status': 'formal_seed1_capacity_validation_only_one_update',
        },
    }
    stage_logs = {
        'unit_tests': ['local_cpu_tests.json', 'reward_unit.console.log'],
        'gym_smoke': ['nominal32.console.log', 'randomization8.console.log', 'smoke32.console.log'],
        'training_resume': ['run512.console.log', 'resume512.console.log', 'old_resume_rejection.json', 'old_resume_rejection.console.log'],
        'capacity': ['capacity4096.console.log'],
    }
    for name, stage in stages.items():
        stage['raw_artifacts'] = [{'path': str(VALIDATION_ROOT / filename), 'sha256': file_sha256(VALIDATION_ROOT / filename)} for filename in stage_logs[name]]
        report_path = VALIDATION_ROOT / f'{name}.report.json'
        with report_path.open('x', encoding='utf-8') as output:
            json.dump({'stage': name, 'result': stage}, output, ensure_ascii=False, allow_nan=False, indent=2)
        stage['artifact'] = {'path': str(report_path), 'sha256': file_sha256(report_path)}
    evidence = {'schema_version': 1, 'identity': identity,
                'formal_config_sha256': identity['learning_config_sha256'], 'stages': stages}
    manifest = json.loads((PROJECT_ROOT / 'configs/v3_a_initial_v1.json').read_text())
    gate = assess_a_start_gate(evidence, identity, manifest, formal_num_envs=4096)
    gate['blockers'].extend(validate_stage_artifacts(evidence))
    gate['ready'] = not gate['blockers']

    timing = json.loads((VALIDATION_ROOT / 'timing4096.json').read_text())
    intervals = np.array(timing['iteration_intervals_s'])
    rows = [json.loads(line) for line in (VALIDATION_ROOT / 'run4096/iterations.jsonl').read_text().splitlines()]
    performance = {
        'creation_s': timing['creation_s'], 'learn_wall_s': timing['learn_wall_s'],
        'iterations': 100, 'mean_wall_per_iteration_s': timing['learn_wall_s'] / 100,
        'interval_median_s': float(np.median(intervals)), 'interval_p95_s': float(np.percentile(intervals, 95)),
        'warm_after_10_interval_mean_s': float(intervals[10:].mean()),
        'warm_after_10_collection_mean_s': float(np.mean([row['collection_s'] for row in rows[10:]])),
        'warm_after_10_update_mean_s': float(np.mean([row['update_s'] for row in rows[10:]])),
        'checkpoint_writes': timing['checkpoint_writes'],
        'estimated_a8000_hours': timing['learn_wall_s'] / 100 * 8000 / 3600,
        'torch_peak_memory_bytes': results['run4096']['completion']['torch_peak_memory_bytes'],
        'target_passed': timing['completed'] and timing['learn_wall_s'] / 100 <= 4.5,
        '说明': '100轮开发短测含正常日志及最终保存；区间含上一轮尾部IO，最后尾部IO完整计入learn_wall_s。长训策略、GPU共享负载及额外定期保存可改变耗时。',
    }
    baseline_step = json.loads((VALIDATION_ROOT / 'collision_baseline_randomized512.json').read_text())
    candidate_step = json.loads((VALIDATION_ROOT / 'collision_candidate_randomized512.json').read_text())
    baseline_physics = json.loads((VALIDATION_ROOT / 'collision_baseline_physics512.json').read_text())
    candidate_physics = json.loads((VALIDATION_ROOT / 'collision_candidate_physics512.json').read_text())
    performance['same_gpu_512'] = {
        'baseline_full_step_s': baseline_step['step_mean_s'],
        'candidate_full_step_s': candidate_step['step_mean_s'],
        'full_step_speedup': baseline_step['step_mean_s'] / candidate_step['step_mean_s'],
        'baseline_physics_step_s': baseline_physics['physics_policy_step_mean_s'],
        'candidate_physics_step_s': candidate_physics['physics_policy_step_mean_s'],
        'physics_step_speedup': baseline_physics['physics_policy_step_mean_s'] / candidate_physics['physics_policy_step_mean_s'],
        '说明': '相同最终源码、GPU1、seed1、2**25容量，独立进程依次100步。物理段与完整step轨迹可能分离，不相减计算精确占比。',
    }
    summary = {'schema_version': 1, 'runs': results, 'restore_checks': restore_checks,
               'cross_asset_resume_rejected': cross_asset_resume_rejected,
               'performance': performance, 'gate': gate,
               'short_checks_passed': all(result['passed'] for result in results.values()) and all(restore_checks.values()) and cross_asset_resume_rejected,
               'new_formal_training_started': False, 'a8000_effect_assessed': False}
    for name, value in (('start_evidence', evidence), ('start_gate', gate), ('verified_summary', summary)):
        with (VALIDATION_ROOT / f'{name}.json').open('x', encoding='utf-8') as output:
            json.dump(value, output, ensure_ascii=False, allow_nan=False, indent=2)
            output.write('\n')
    print(json.dumps({'short_checks_passed': summary['short_checks_passed'], 'restore_checks': restore_checks,
                      'gate': gate, 'performance': performance}, ensure_ascii=False, indent=2))
    assert summary['short_checks_passed'] and gate['ready'] and performance['target_passed']


if __name__ == '__main__':
    main()



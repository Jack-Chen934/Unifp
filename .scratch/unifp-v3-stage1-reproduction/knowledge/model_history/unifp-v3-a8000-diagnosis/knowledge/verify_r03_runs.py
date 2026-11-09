"""Verify native short-run state, identity and resume continuity on CPU."""

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, Tuple

import torch


def inspect_run(validation_root: Path, source_root: Path, name: str,
                num_envs: int, iterations: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Inspect real checkpoints and logs without constructing a simulator.

    Args:
        validation_root: Directory containing completed isolated validation runs.
        source_root: Fixed V3 source tree used by every inspected run.
        name: Run directory basename.
        num_envs: Expected parallel environment count.
        iterations: Expected cumulative count of complete training updates.

    Returns:
        A check report and the loaded CPU checkpoint used for resume comparisons.

    Raises:
        OSError: A checkpoint or required log cannot be read.
        KeyError: A checkpoint lacks a required training-boundary field.
    """
    from legged_gym.b2_gym_learn.ppo_cse_pf.v3_run_identity import build_run_identity

    run_root = validation_root / name
    checkpoint_path = run_root / 'checkpoints' / f'model_{iterations:06d}.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    completion = json.loads((run_root / 'completion.json').read_text())
    metadata = json.loads((run_root / 'episode_metadata.json').read_text())
    rows = [json.loads(line) for line in (run_root / 'iterations.jsonl').read_text().splitlines()]
    config = checkpoint['config_snapshot']
    control = config['environment']['control']
    simulation = config['environment']['sim']
    expected_progress = {'completed_iterations': iterations, 'training_policy_steps': iterations * 24,
                         'training_transitions': iterations * 24 * num_envs, 'rollout_step': 0}
    algorithm = config['training']['algorithm']
    expected_adam_steps = iterations * algorithm['num_learning_epochs'] * algorithm['num_mini_batches']
    optimizer_steps = {
        prefix: sorted({float(state['step']) for state in
                        checkpoint[prefix + '_optimizer_state_dict']['state'].values()})
        for prefix in ('main', 'auxiliary')}
    console = (validation_root / f'{name}.console.log').read_text()
    checks = {
        'progress': checkpoint['progress'] == expected_progress,
        'completion': all(completion[key] == expected_progress[key] for key in
                          ('completed_iterations', 'training_policy_steps', 'training_transitions')),
        'identity': build_run_identity(source_root, config, checkpoint['identity']['runtime'])
                    == checkpoint['identity'] == metadata['identity'],
        'finite_model': all(bool(torch.isfinite(value).all())
                            for value in checkpoint['model_state_dict'].values()),
        'finite_metrics': all(math.isfinite(value) for row in rows for value in row.values()
                              if isinstance(value, (float, int))),
        'main_adam': optimizer_steps['main'] == [float(expected_adam_steps)],
        'auxiliary_adam': optimizer_steps['auxiliary'] == [float(expected_adam_steps)],
        'num_envs': checkpoint['environment_training_state']['num_envs'] == num_envs,
        'iteration_sequence': [row['completed_iterations'] for row in rows]
                              == list(range(iterations - len(rows) + 1, iterations + 1)),
        'iteration_counters': all(row['training_policy_steps'] == row['completed_iterations'] * 24
                                  and row['training_transitions'] == row['training_policy_steps'] * num_envs
                                  for row in rows),
        'physics_timing': simulation['dt'] == 0.0025 and control['decimation'] == 8,
        'contact_capacity': simulation['physx']['max_gpu_contact_pairs'] == 2**25,
        'no_missing_interactions': 'simulation will miss interactions' not in console,
    }
    episodes_path = run_root / 'episodes.jsonl'
    episodes = ([json.loads(line) for line in episodes_path.read_text().splitlines()]
                if episodes_path.exists() else [])
    first_resume_failures = sum(row['end_type'] == 'physical_failure'
                               and row['training_policy_steps'] == (iterations - 1) * 24
                               for row in episodes) if name == 'resume512' else None
    return {'passed': all(checks.values()), 'checks': checks, 'completion': completion,
            'optimizer_steps': optimizer_steps, 'first_resume_physical_failures': first_resume_failures,
            'physical_failures': sum(row['end_type'] == 'physical_failure' for row in episodes),
            'finished_episodes': len(episodes),
            'mean_collection_s': sum(row['collection_s'] for row in rows) / len(rows),
            'mean_update_s': sum(row['update_s'] for row in rows) / len(rows)}, checkpoint


def main() -> None:
    """Validate completed short runs and write one auditable evidence summary.

    Raises:
        OSError: Required validation artifacts cannot be read or written.
        SystemExit: Real counts, physical checks, restore state, or identity fails.

    Note:
        Reads checkpoint tensors on CPU only. Does not run learning, change the
        fixed source, or turn development evidence into a formal startup approval.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--validation-root', type=Path, required=True)
    arguments = parser.parse_args()
    validation_root = arguments.validation_root
    source_root = validation_root / 'source/UniFP_v3'
    sys.path.insert(0, str(source_root))
    results = {}
    checkpoints = {}
    for name, num_envs, iterations in (('smoke32', 32, 1), ('run512', 512, 10),
                                      ('resume512', 512, 11), ('run4096', 4096, 20)):
        results[name], checkpoints[name] = inspect_run(validation_root, source_root, name,
                                                       num_envs, iterations)
    parent = checkpoints['run512']
    resumed = checkpoints['resume512']
    from legged_gym.b2_gym_learn.ppo_cse_pf.v3_run_identity import file_sha256

    restore = {
        'identity': parent['identity'] == resumed['identity'],
        'lineage_hash': resumed['lineage']['parent_checkpoint_sha256'] == file_sha256(
            validation_root / 'run512/checkpoints/model_000010.pt'),
        'creation_mass_com': torch.equal(parent['environment_training_state']['mass_com_delta'],
                                         resumed['environment_training_state']['mass_com_delta']),
        'creation_friction': torch.equal(parent['environment_training_state']['friction'],
                                         resumed['environment_training_state']['friction']),
        'first_resume_step': results['resume512']['first_resume_physical_failures'] == 0,
    }
    physical = {name: json.loads((validation_root / f'{name}.json').read_text())
                for name in ('final_motion_gate', 'final_reset_tcp', 'final_nominal32', 'final_randomized32')}
    physics_checks = {
        'motion': physical['final_motion_gate']['passed'],
        'reset_tcp': physical['final_reset_tcp']['passed'],
        'nominal_zero_failures': all(physical['final_nominal32'][key] == 0 for key in
                                     ('physical_failure_events', 'movement_physical_failure_events',
                                      'arm_physical_failure_events')),
        'nominal_reset': all(physical['final_nominal32']['local_reset_other_rows_unchanged'].values()),
        'randomized_zero_failures': physical['final_randomized32']['physical_failure_events'] == 0,
        'randomized_reset': all(physical['final_randomized32']['local_reset_other_rows_unchanged'].values()),
    }
    timing = json.loads((validation_root / 'timing4096.json').read_text())
    report = {'runs': results, 'restore_checks': restore, 'physics_checks': physics_checks,
              'timing4096': timing, 'formal_training_started': False}
    report['passed'] = (all(run['passed'] for run in results.values())
                        and all(restore.values()) and all(physics_checks.values()))
    (validation_root / 'verified_summary.json').write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()

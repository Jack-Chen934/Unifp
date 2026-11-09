"""Bind completed r03 validation outputs to the existing A-start gate."""

import argparse
import json
from pathlib import Path
import sys

import torch

from verify_r03_runs import inspect_run


def main() -> None:
    """Verify native reports and write hash-pinned stage evidence without training.

    Raises:
        AssertionError: Native tests, capacity, provenance, or the start gate fails.
        OSError: A required artifact cannot be read or a report already exists.

    Note:
        Loads checkpoints on CPU only. Creates stage reports and start evidence;
        does not launch training or approve the separate observation schedule.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--validation-root', type=Path, required=True)
    validation_root = parser.parse_args().validation_root
    source_root = validation_root / 'source/UniFP_v3'
    sys.path.insert(0, str(source_root))
    from legged_gym.b2_gym_learn.ppo_cse_pf.v3_preflight import assess_a_start_gate, validate_stage_artifacts
    from legged_gym.b2_gym_learn.ppo_cse_pf.v3_run_identity import file_sha256

    # Capacity uses the full formal configuration but executes exactly one update.
    summary = json.loads((validation_root / 'verified_summary.json').read_text())
    assert summary['passed'], 'Short-run regression failed'
    capacity, checkpoint = inspect_run(validation_root, source_root, 'capacity4096', 4096, 1)
    assert capacity['passed'], capacity['checks']
    identity = checkpoint['identity']
    assert identity == json.loads((validation_root / 'formal_identity.json').read_text())
    local_tests = json.loads((validation_root / 'local_unit_evidence.json').read_text())
    assert local_tests['passed'] and local_tests['test_count'] == 46
    assert local_tests['source_manifest'] == identity['source_manifest']
    for log in local_tests['logs'].values():
        assert file_sha256(validation_root / log['basename']) == log['sha256']
    for name, count in (('control_unit.log', 4), ('rewards_unit.log', 5)):
        content = (validation_root / name).read_text()
        assert f'Ran {count} tests' in content and '\nOK\n' in content

    # A's actual step path rejects nonzero force channels before accepting a step.
    # The completed smoke run therefore supplies force and finite-state evidence.
    smoke_checkpoint = torch.load(
        validation_root / 'smoke32/checkpoints/model_000001.pt',
        map_location='cpu', weights_only=False)
    resume_checkpoint = torch.load(
        validation_root / 'run512/checkpoints/model_000010.pt',
        map_location='cpu', weights_only=False)
    stages = {
        'unit_tests': {'passed': True, 'identity': identity,
                       'local_tests': 46, 'remote_config_tests': 4, 'remote_reward_tests': 5},
        'gym_smoke': {'passed': summary['runs']['smoke32']['passed'],
                      'identity': smoke_checkpoint['identity'], 'num_envs': 32,
                      'zero_ee_command_force': True, 'zero_ee_external_force': True,
                      'finite_core': summary['runs']['smoke32']['checks']['finite_model'],
                      'reset_isolated': all(summary['physics_checks'].values()),
                      'force_evidence': 'V3PosForce.step rejects nonzero A force channels'},
        'training_resume': {'passed': all(summary['restore_checks'].values()),
                            'identity': resume_checkpoint['identity'], 'num_envs': 512,
                            'iterations': 10, 'resumed_iterations': 1, 'completed_iterations': 11,
                            'counters_consistent': all(summary['runs'][name]['checks']['iteration_counters']
                                                       for name in ('run512', 'resume512')),
                            'checkpoint_identity_matches': summary['restore_checks']['identity']},
        'capacity': {'passed': capacity['passed'], 'identity': identity, 'num_envs': 4096,
                     'rollout_update': capacity['passed'],
                     'formal_config_sha256': identity['learning_config_sha256']},
    }
    raw_files = {
        'unit_tests': ['local_unit_evidence.json', 'control_unit.log', 'rewards_unit.log'],
        'gym_smoke': ['verified_summary.json', 'final_reset_tcp.json', 'final_motion_gate.json',
                      'final_nominal32.json', 'final_randomized32.json', 'smoke32.console.log'],
        'training_resume': ['verified_summary.json', 'run512.console.log', 'resume512.console.log'],
        'capacity': ['capacity4096.console.log', 'capacity4096/completion.json',
                     'capacity4096/iterations.jsonl', 'formal_identity.json'],
    }
    # Each stage owns its immutable report, rather than inheriting a summary flag.
    for name, stage in stages.items():
        stage['raw_artifacts'] = [{'path': str(validation_root / filename),
                                   'sha256': file_sha256(validation_root / filename)}
                                  for filename in raw_files[name]]
        report_path = validation_root / f'{name}.report.json'
        with report_path.open('x', encoding='utf-8') as output:
            json.dump({'stage': name, 'result': stage}, output, ensure_ascii=False, indent=2)
        stage['artifact'] = {'path': str(report_path), 'sha256': file_sha256(report_path)}
    evidence = {'schema_version': 1, 'identity': identity,
                'formal_config_sha256': identity['learning_config_sha256'], 'stages': stages}
    manifest = json.loads((source_root / 'configs/v3_a_initial_v1.json').read_text())
    gate = assess_a_start_gate(evidence, identity, manifest, formal_num_envs=4096)
    gate['blockers'].extend(validate_stage_artifacts(evidence))
    gate['ready'] = not gate['blockers']
    for name, value in (('start_evidence', evidence), ('start_gate', gate)):
        with (validation_root / f'{name}.json').open('x', encoding='utf-8') as output:
            json.dump(value, output, ensure_ascii=False, allow_nan=False, indent=2)
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    assert gate['ready'], gate['blockers']


if __name__ == '__main__':
    main()

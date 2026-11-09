"""Observe a complete intermediate checkpoint with the frozen A scenarios."""

import json
from pathlib import Path
import shutil
import subprocess

import isaacgym  # noqa: F401
import torch
import legged_gym
from legged_gym.scripts.evaluate_v3 import evaluate_scenario
from legged_gym.scripts.v3_args import get_v3_args
from legged_gym.b2_gym_learn.ppo_cse_pf.v3_evaluation import (
    EVALUATION_SCENARIOS, validate_evaluation_manifest,
)
from legged_gym.b2_gym_learn.ppo_cse_pf.v3_run_identity import build_run_identity, file_sha256


def main() -> None:
    """Evaluate actual intermediate progress without relabeling it as A8000.

    Raises:
        AssertionError: Checkpoint identity, progress or model state is invalid.
        OSError: Required evidence cannot be read, created or sealed.

    Note:
        Uses separate simulators, no optimizer, and Actor means. Keeps the exact
        frozen manifest and all failed samples; thresholds are reference values
        only until the official A8000 evaluation is completed.
    """
    args = get_v3_args('中途冻结场景观察，不替代A8000验收')
    checkpoint_path = Path(args.checkpoint).resolve()
    manifest_path = Path(args.manifest).resolve()
    output = Path(args.output_dir).resolve()
    source_root = Path(legged_gym.__file__).resolve().parents[1]
    manifest = json.loads(manifest_path.read_text())
    validate_evaluation_manifest(manifest)
    assert args.run_purpose == 'evaluation' and args.reproduction_part == 'A'
    assert args.seed == manifest['seed']
    payload = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    identity = build_run_identity(source_root, payload['config_snapshot'], payload['identity']['runtime'])
    assert identity == payload['identity']
    progress = payload['progress']
    iteration = progress['completed_iterations']
    assert payload['checkpoint_kind'] == 'training_boundary' and payload['format_version'] == 1
    assert 0 < iteration < 8000 and progress == {
        'completed_iterations': iteration, 'training_policy_steps': iteration * 24,
        'training_transitions': iteration * 24 * payload['environment_training_state']['num_envs'],
        'rollout_step': 0}
    assert all(bool(torch.isfinite(value).all()) for value in payload['model_state_dict'].values())
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(manifest_path, output / 'evaluation_manifest.json')
    reports = {}
    for name in EVALUATION_SCENARIOS:
        scenario_root = output / name
        scenario_root.mkdir()
        summary, snapshot = evaluate_scenario(args, payload, manifest,
                                               scenario_name=name, output_dir=scenario_root)
        summary.update(checkpoint_iteration=iteration, observation_only=True)
        # Add actual motion magnitudes; tracking error alone can hide immobility.
        sample_count, command_speed_sum, actual_speed_sum = 0, 0.0, 0.0
        with (scenario_root / 'policy_trace.jsonl').open() as trace:
            for line in trace:
                row = json.loads(line)
                command = row['commands_body']
                velocity = row['base_linear_velocity_body_m_s']
                command_speed_sum += (command[0]**2 + command[1]**2)**0.5
                actual_speed_sum += (velocity[0]**2 + velocity[1]**2)**0.5
                sample_count += 1
        summary['observed_planar_motion'] = {
            'command_mean_m_s': command_speed_sum / sample_count,
            'actual_mean_m_s': actual_speed_sum / sample_count}
        for filename, record in (('summary.json', summary), ('config_snapshot.json', snapshot)):
            (scenario_root / filename).write_text(
                json.dumps(record, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
        reports[name] = summary
    result = {
        'checkpoint_iteration': iteration, 'observation_only': True, 'a8000_accepted': False,
        'policy_updates': 0, 'scenarios': reports,
        'physical_failures': sum(report['episodes']['physical_failures'] for report in reports.values()),
        'checkpoint_sha256': file_sha256(checkpoint_path),
        'frozen_manifest_sha256': file_sha256(manifest_path),
        'git_snapshot_commit': subprocess.check_output(
            ['git', '-C', str(source_root), 'rev-parse', 'HEAD'], text=True).strip()}
    (output / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    inventory = {path.relative_to(output).as_posix(): file_sha256(path)
                 for path in sorted(output.rglob('*')) if path.is_file()}
    (output / 'archive_manifest.json').write_text(json.dumps(inventory, indent=2) + '\n')
    for path in output.rglob('*'):
        path.chmod(0o555 if path.is_dir() else 0o444)
    output.chmod(0o555)
    print(json.dumps({'iteration': iteration, 'physical_failures': result['physical_failures']}, ensure_ascii=False))


if __name__ == '__main__':
    main()

"""Run the unchanged learner with approved boundary observations."""

import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List

from legged_gym.scripts import train_v3
from legged_gym.b2_gym_learn.ppo_cse_pf import v3_on_policy_runner as runner_module


def main() -> None:
    """Attach boundary observations and delegate to the original training entry.

    Raises:
        subprocess.CalledProcessError: A separate observation process fails.
        Exception: The original learner reports invalid state or another failure.

    Note:
        Preserves simulation state, RNG and PPO updates. Evaluation blocks the
        training process at a saved boundary and runs in a separate interpreter.
        Physical failures remain recorded episodes and do not stop the run.
    """
    original_flush = runner_module.flush_episode_records
    original_append = runner_module.V3OnPolicyRunner._append_json
    original_save = runner_module.V3OnPolicyRunner.save
    physical_failures_in_run = 0

    def flush_records(directory: Path, records: List[Dict[str, Any]]) -> Dict[str, float]:
        """Count failures in episode batches already persisted by the real learner.

        Args:
            directory: Original run-owned output directory.
            records: Original completed episodes, including physical failures.

        Returns:
            Unchanged original episode metrics.
        """
        nonlocal physical_failures_in_run
        metrics = original_flush(directory, records)
        physical_failures_in_run += sum(record['end_type'] == 'physical_failure' for record in records)
        return metrics

    def append_record(runner: Any, filename: str, record: Dict[str, Any]) -> None:
        """Log normally, then inspect health at complete learning boundaries.

        Args:
            runner: Original live learner with an empty completed rollout.
            filename: Original JSONL log basename.
            record: Unchanged finite learning or checkpoint record.

        Note:
            Counts failures since this run started; parent-run episodes retain
            their original logs and are linked by the learner's resume lineage.
        """
        original_append(runner, filename, record)
        if filename != 'iterations.jsonl':
            return
        if runner.completed_iterations % 1000 == 0:
            original_append(runner, 'observation_health.jsonl', {
                'completed_iterations': runner.completed_iterations,
                'physical_failures': physical_failures_in_run,
                'physical_failure_count_scope': 'current_run',
                'finite_metrics': True, 'training_policy_steps': runner.training_policy_steps,
                'collection_s': record['collection_s'], 'update_s': record['update_s'],
                'latest_episode_metrics': {key: value for key, value in record.items()
                                           if key.startswith('episode/')}})

    def save_and_observe(runner: Any, checkpoint: Path) -> None:
        """Publish normally and run independent frozen scenarios every 2000 updates.

        Args:
            runner: Original learner, paused at its saved complete boundary.
            checkpoint: Atomically published original checkpoint path.

        Raises:
            subprocess.CalledProcessError: Evaluation fails to complete.
        """
        original_save(runner, checkpoint)
        if runner.completed_iterations % 2000 != 0:
            return
        observation_root = runner.log_dir / 'observations' / f'iteration_{runner.completed_iterations:06d}'
        observation_root.parent.mkdir(exist_ok=True)
        source_root = Path(train_v3.__file__).resolve().parents[2]
        manifest = json.loads((source_root / 'configs/v3_a_initial_v1.json').read_text())
        entry = (['-m', 'legged_gym.scripts.evaluate_v3'] if runner.completed_iterations == 8000
                 else [str(Path(__file__).with_name('evaluate_observation.py'))])
        command = [sys.executable, *entry, '--reproduction_part', 'A', '--run_purpose', 'evaluation',
                   '--seed', str(manifest['seed']), '--checkpoint', str(checkpoint),
                   '--manifest', str(source_root / 'configs/v3_a_initial_v1.json'),
                   '--output_dir', str(observation_root), '--sim_device', 'cuda:0',
                   '--rl_device', 'cuda:0', '--headless']
        with observation_root.with_suffix('.console.log').open('x') as console:
            subprocess.run(command, stdout=console, stderr=subprocess.STDOUT, check=True)
        summary = json.loads((observation_root / 'summary.json').read_text())
        if runner.completed_iterations < 8000:
            physical_failures = summary['physical_failures']
        else:
            physical_failures = sum(json.loads((observation_root / name / 'summary.json').read_text())[
                'episodes']['physical_failures'] for name in ('S0', 'S1', 'S2'))
        original_append(runner, 'observation_events.jsonl', {
            'completed_iterations': runner.completed_iterations, 'physical_failures': physical_failures,
            'summary_path': str(observation_root / 'summary.json'), 'policy_updates': 0})

    runner_module.flush_episode_records = flush_records
    runner_module.V3OnPolicyRunner._append_json = append_record
    runner_module.V3OnPolicyRunner.save = save_and_observe
    train_v3.main()


if __name__ == '__main__':
    main()

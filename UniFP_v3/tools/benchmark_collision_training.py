"""Measure the released training entry without modifying its learning code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any


def main() -> None:
    """Wrap creation, iteration logging and saves in one development run.

    Returns:
        None.

    Raises:
        Exception: Any original training exception is propagated unchanged.

    Note:
        Remaining arguments are passed to train_v3. Writes timing evidence only
        to --timing-output; the original entry owns normal logs/checkpoints.
        Iteration intervals include the preceding iteration's final logging and
        save. Total learn wall time includes all final logging and checkpoint IO.
    """
    parser = argparse.ArgumentParser(description='测量原训练入口的完整耗时')
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--timing-output', type=Path, required=True)
    arguments, training_arguments = parser.parse_known_args()
    sys.path.insert(0, str(arguments.source_root))
    from legged_gym.scripts import train_v3

    # Store timestamps outside the learner: optimization, clocks, RNG and
    # checkpoint serialization remain the original implementation.
    report: dict[str, Any] = {'iteration_intervals_s': [], 'checkpoint_writes': [],
                              'completed': False}
    original_environment = train_v3.V3PosForce
    runner_class = train_v3.V3OnPolicyRunner
    original_learn = runner_class.learn
    original_append = runner_class._append_json
    original_save = runner_class.save
    last_iteration_time = 0.0

    def create_environment(*args: Any, **kwargs: Any) -> Any:
        """Time the original simulator constructor with its unchanged inputs.

        Args:
            *args: Original environment positional constructor arguments.
            **kwargs: Original environment keyword constructor arguments.

        Returns:
            Original fully initialized V3 environment.
        """
        started = time.perf_counter()
        environment = original_environment(*args, **kwargs)
        report['creation_s'] = time.perf_counter() - started
        return environment

    def timed_learn(runner: Any, *args: Any, **kwargs: Any) -> None:
        """Measure a complete learn call including its checkpoint writes.

        Args:
            runner: Original V3 runner instance.
            *args: Original learn positional arguments, including iteration count.
            **kwargs: Original learn keyword arguments.

        Returns:
            None.
        """
        nonlocal last_iteration_time
        started = time.perf_counter()
        last_iteration_time = started
        original_learn(runner, *args, **kwargs)
        train_v3.torch.cuda.synchronize()
        report['learn_wall_s'] = time.perf_counter() - started
        report['completed_iterations'] = runner.completed_iterations

    def timed_append(runner: Any, filename: str, record: dict) -> None:
        """Capture iteration completion boundaries after the normal JSON write.

        Args:
            runner: Original V3 runner instance.
            filename: Existing JSONL artifact basename.
            record: Unchanged finite JSON-compatible log record.

        Returns:
            None.
        """
        nonlocal last_iteration_time
        original_append(runner, filename, record)
        if filename == 'iterations.jsonl':
            now = time.perf_counter()
            report['iteration_intervals_s'].append(now - last_iteration_time)
            last_iteration_time = now

    def timed_save(runner: Any, *args: Any, **kwargs: Any) -> Any:
        """Measure original checkpoint serialization and disk writes.

        Args:
            runner: Original V3 runner instance.
            *args: Unchanged checkpoint save positional arguments.
            **kwargs: Unchanged checkpoint save keyword arguments.

        Returns:
            Original save result.
        """
        started = time.perf_counter()
        result = original_save(runner, *args, **kwargs)
        report['checkpoint_writes'].append({'iteration': runner.completed_iterations,
                                            'seconds': time.perf_counter() - started})
        return result

    train_v3.V3PosForce = create_environment
    runner_class.learn = timed_learn
    runner_class._append_json = timed_append
    runner_class.save = timed_save
    sys.argv = [sys.argv[0], *training_arguments]
    try:
        train_v3.main()
        report['completed'] = True
    finally:
        arguments.timing_output.parent.mkdir(parents=True, exist_ok=True)
        arguments.timing_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()

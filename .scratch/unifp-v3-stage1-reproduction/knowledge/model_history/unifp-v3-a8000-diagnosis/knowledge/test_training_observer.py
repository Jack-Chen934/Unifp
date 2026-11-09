"""Check observation scheduling and failure handling without loading Gym."""

import importlib.util
from pathlib import Path
import subprocess
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


class ObserverTests(unittest.TestCase):
    """Verify control flow while leaving the real learner out of the process."""

    def test_boundaries_and_failures(self) -> None:
        """Require scheduled observations and continuation after episode failures.

        Raises:
            AssertionError: Cadence, failure statistics or error propagation differ.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = []

            class Runner:
                """Supply only the original interfaces used by the wrapper."""

                def _append_json(self, filename: str, record: dict) -> None:
                    """Record log calls.

                    Args:
                        filename: Original output basename.
                        record: Original metrics dictionary.
                    """
                    calls.append(('append', filename, record))

                def save(self, checkpoint: Path) -> None:
                    """Record the original save.

                    Args:
                        checkpoint: Complete-boundary output path.
                    """
                    calls.append(('save', checkpoint))

            def flush_records(directory: Path, records: list) -> dict:
                """Return the original summary.

                Args:
                    directory: Run-owned directory, unused by this stub.
                    records: Completed episodes supplied by the learner.

                Returns:
                    Unchanged episode count.
                """
                return {'episode_count': len(records)}

            learner = ModuleType('legged_gym.scripts.train_v3')
            learner.__file__ = str(root / 'source/legged_gym/scripts/train_v3.py')
            runner_module = ModuleType('legged_gym.b2_gym_learn.ppo_cse_pf.v3_on_policy_runner')
            runner_module.V3OnPolicyRunner = Runner
            runner_module.flush_episode_records = flush_records

            def original_main() -> None:
                """Exercise installed callbacks in their real boundary order.

                Raises:
                    AssertionError: Cadence, logging or physical-failure handling differs.
                """
                runner = Runner()
                runner.log_dir = root / 'run'
                runner.log_dir.mkdir()
                runner.completed_iterations, runner.training_policy_steps = 999, 23976
                record = {'collection_s': 1.0, 'update_s': 0.1, 'episode/count': 0}
                runner._append_json('iterations.jsonl', record)
                self.assertEqual(len(calls), 1)
                runner.completed_iterations, runner.training_policy_steps = 1000, 24000
                runner._append_json('iterations.jsonl', record)
                self.assertEqual(calls[-1][1], 'observation_health.jsonl')
                runner.save(root / 'normal.pt')
                self.assertEqual(calls[-1], ('save', root / 'normal.pt'))
                runner.completed_iterations, runner.training_policy_steps = 2000, 48000
                manifest = root / 'source/configs/v3_a_initial_v1.json'
                manifest.parent.mkdir(parents=True)
                manifest.write_text('{"seed":1001}')
                output = runner.log_dir / 'observations/iteration_002000'
                output.mkdir(parents=True)
                (output / 'summary.json').write_text('{"physical_failures":0}')
                with patch.object(module.subprocess, 'run') as execute:
                    runner.save(root / 'milestone.pt')
                    self.assertEqual(execute.call_count, 1)
                    self.assertIn(str(root / 'milestone.pt'), execute.call_args.args[0])
                    self.assertEqual(execute.call_args.args[0][execute.call_args.args[0].index('--seed') + 1], '1001')
                self.assertEqual(calls[-1][1], 'observation_events.jsonl')
                summary = runner_module.flush_episode_records(root, [{'end_type': 'physical_failure'}])
                self.assertEqual(summary, {'episode_count': 1})
                runner.completed_iterations = 2001
                runner._append_json('iterations.jsonl', record)
                self.assertEqual(calls[-1], ('append', 'iterations.jsonl', record))
                runner.completed_iterations = 3000
                runner._append_json('iterations.jsonl', record)
                self.assertEqual(calls[-1][2]['physical_failures'], 1)
                self.assertEqual(calls[-1][2]['physical_failure_count_scope'], 'current_run')
                runner.completed_iterations = 4000
                runner_module.flush_episode_records(root, [{'end_type': 'timeout'},
                                                           {'end_type': 'physical_failure'}])
                output = runner.log_dir / 'observations/iteration_004000'
                output.mkdir()
                (output / 'summary.json').write_text('{"physical_failures":2}')
                with patch.object(module.subprocess, 'run'):
                    runner.save(root / 'with_failures.pt')
                self.assertEqual(calls[-1][1], 'observation_events.jsonl')
                self.assertEqual(calls[-1][2]['physical_failures'], 2)
                runner.completed_iterations = 5000
                runner._append_json('iterations.jsonl', record)
                self.assertEqual(calls[-1][2]['physical_failures'], 2)
                runner.completed_iterations = 6000
                with patch.object(module.subprocess, 'run',
                                  side_effect=subprocess.CalledProcessError(1, ['evaluation'])):
                    with self.assertRaises(subprocess.CalledProcessError):
                        runner.save(root / 'broken_evaluation.pt')
                runner.completed_iterations = 8000
                output = runner.log_dir / 'observations/iteration_008000'
                output.mkdir()
                (output / 'summary.json').write_text('{"accepted":false}')
                for scenario_name in ('S0', 'S1', 'S2'):
                    (output / scenario_name).mkdir()
                    (output / scenario_name / 'summary.json').write_text(
                        '{"episodes":{"physical_failures":1}}')
                with patch.object(module.subprocess, 'run') as execute:
                    runner.save(root / 'final.pt')
                    self.assertIn('legged_gym.scripts.evaluate_v3', execute.call_args.args[0])
                self.assertEqual(calls[-1][2]['physical_failures'], 3)
                self.assertEqual((output / 'summary.json').read_text(), '{"accepted":false}')

            learner.main = original_main
            stubs = {
                'legged_gym': ModuleType('legged_gym'),
                'legged_gym.scripts': SimpleNamespace(train_v3=learner),
                'legged_gym.b2_gym_learn': ModuleType('legged_gym.b2_gym_learn'),
                'legged_gym.b2_gym_learn.ppo_cse_pf': SimpleNamespace(v3_on_policy_runner=runner_module),
            }
            spec = importlib.util.spec_from_file_location('observed', Path(__file__).with_name('train_observed.py'))
            module = importlib.util.module_from_spec(spec)
            with patch.dict('sys.modules', stubs):
                spec.loader.exec_module(module)
                module.main()


if __name__ == '__main__':
    unittest.main()

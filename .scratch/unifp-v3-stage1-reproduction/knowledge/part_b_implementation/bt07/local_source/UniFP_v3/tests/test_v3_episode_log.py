"""Check bounded completed-episode logging and equal-episode aggregation."""

import json
from pathlib import Path
import tempfile
import unittest

from legged_gym.b2_gym_learn.ppo_cse_pf.v3_episode_log import flush_episode_records, write_episode_metadata


class EpisodeLogTests(unittest.TestCase):
    """Validate the agreed completed-episode disk and aggregation seam."""

    def test_flush_uses_equal_episode_weights_and_skips_empty_batches(self) -> None:
        """Write two unequal-length episodes once and return their equal-weight mean.

        Returns:
            None.
        """
        records = [
            {"steps": 1, "reward_rate": 2.0, "reward_return": 0.04,
             "raw_reward_means": {"alive": 1.0}, "errors": {"tcp_l2_m": {"mean": 0.1, "p95": 0.1}},
             "timeout_bootstrap_bonus": 0.0},
            {"steps": 9, "reward_rate": 4.0, "reward_return": 0.72,
             "raw_reward_means": {"alive": 1.0}, "errors": {"tcp_l2_m": {"mean": 0.3, "p95": 0.4}},
             "timeout_bootstrap_bonus": 0.2},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            summary = flush_episode_records(directory, records)
            self.assertEqual(summary["episode_count"], 2)
            self.assertEqual(summary["reward_rate"], 3.0)
            self.assertAlmostEqual(summary["errors/tcp_l2_m/mean"], 0.2)
            before = (directory / "episodes.jsonl").read_bytes()
            self.assertEqual(len(before.splitlines()), 2)
            self.assertEqual(json.loads(before.splitlines()[1])["steps"], 9)
            self.assertEqual(flush_episode_records(directory, []), {"episode_count": 0})
            self.assertEqual((directory / "episodes.jsonl").read_bytes(), before)

    def test_metadata_is_written_once_and_unresolved_timeout_is_not_fabricated(self) -> None:
        """Preserve run metadata and reject a missing bootstrap value before append.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_episode_metadata(directory, {"policy_dt_s": 0.02, "configured_weights": {"alive": 1.5}})
            with self.assertRaises(FileExistsError):
                write_episode_metadata(directory, {"policy_dt_s": 0.005})
            with self.assertRaises(ValueError):
                flush_episode_records(directory, [{"timeout_bootstrap_bonus": None}])
            self.assertFalse((directory / "episodes.jsonl").exists())


if __name__ == "__main__":
    unittest.main()

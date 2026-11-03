"""Write bounded episode batches and summarize completed episodes equally."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List


def flush_episode_records(
    log_directory: Path, records: List[Dict[str, Any]]
) -> Dict[str, float]:
    """Append one iteration's completed episodes and return equal-episode means.

    Args:
        log_directory: Run-owned output directory; episodes.jsonl is appended here.
        records: Completed episodes with raw means, physical errors and a PPO-populated
            timeout bonus. Empty batches produce only an episode count of zero.

    Returns:
        Flat metric names and equal-episode means for TensorBoard; no stale values.

    Raises:
        ValueError: A record is non-finite or its PPO timeout bonus is unresolved.
        OSError: Creating, appending or syncing the output file fails.

    Note:
        Serializes before appending, flushes once per batch, and retains no records.
        The caller submits each batch once; recovery uses a separate run directory.
    """
    if not records:
        return {"episode_count": 0}
    if any(record["timeout_bootstrap_bonus"] is None for record in records):
        raise ValueError("PPO must resolve timeout bonuses before normal episode logging")
    serialized = "".join(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n" for record in records)
    log_directory.mkdir(parents=True, exist_ok=True)
    with (log_directory / "episodes.jsonl").open("a", encoding="utf-8") as output:
        output.write(serialized)
        output.flush()
        os.fsync(output.fileno())
    # All episodes count equally; their lengths never become aggregation weights.
    summary = {"episode_count": len(records)}
    for name in ("reward_rate", "reward_return", "steps", "timeout_bootstrap_bonus"):
        summary[name] = sum(record[name] for record in records) / len(records)
    for name in records[0]["raw_reward_means"]:
        summary["raw/" + name] = sum(record["raw_reward_means"][name] for record in records) / len(records)
    for name in records[0]["errors"]:
        for statistic in ("mean", "p95"):
            summary[f"errors/{name}/{statistic}"] = sum(
                record["errors"][name][statistic] for record in records
            ) / len(records)
    return summary


def write_episode_metadata(log_directory: Path, metadata: Dict[str, Any]) -> None:
    """Save immutable reward weights, period and run identity before episode logging.

    Args:
        log_directory: New run output directory; created if absent.
        metadata: JSON-compatible source/asset/config identity, raw weights,
            effective weights, policy period and metric units supplied by Runner.

    Raises:
        ValueError: A floating-point metadata value is non-finite.
        FileExistsError: Metadata already exists; run identities are never overwritten.
        OSError: Creating or syncing metadata fails.

    Note:
        Creates episode_metadata.json exclusively; does not mutate the input dictionary.
    """
    serialized = json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    log_directory.mkdir(parents=True, exist_ok=True)
    with (log_directory / "episode_metadata.json").open("x", encoding="utf-8") as output:
        output.write(serialized)
        output.flush()
        os.fsync(output.fileno())

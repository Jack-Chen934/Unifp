"""Check isolated contact/zero-control CLI execution and archive layout."""

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import pytest
from legged_gym.scripts import evaluate_v3_contact as entry


def test_contact_groups_use_fresh_processes_and_preserve_original_arguments(tmp_path: Path, monkeypatch: Any) -> None:
    """Retain complete quotas and video flags while isolating native lifetimes.

    Args:
        tmp_path: Writable parent archive owned by this test.
        monkeypatch: Fixture replacing only subprocess execution.

    Raises:
        AssertionError: Either scene is omitted, shares a process or loses flags.
    """
    arguments = ["--output_dir", str(tmp_path), "--num_envs", "32", "--video",
                 "--seed", "1001", "--protocol", "fixed_wall_contact"]
    calls: List[List[str]] = []

    def evaluate(command: List[str], **kwargs: Any) -> None:
        """Publish a sealed child archive without starting the native SDK.

        Args:
            command: Real single-scenario CLI including unchanged video/quota flags.
            **kwargs: Standard subprocess check options.
        """
        calls.append(command)
        output = Path(command[command.index("--output_dir") + 1])
        name = command[command.index("--scenario") + 1]
        (output / name).mkdir(parents=True)
        (output / name / "summary.json").write_text(json.dumps({"policy_updates": 0, "assessment": name}))
        (output / name / "summary.json").chmod(0o444)
        (output / name).chmod(0o555)
        output.chmod(0o555)

    monkeypatch.setattr(entry.subprocess, "run", evaluate)
    reports = entry.evaluate_contact_groups(arguments, tmp_path)
    assert reports == {"contact": "contact", "zero_control": "zero_control"}
    assert len(calls) == 2
    for name, command in zip(reports, calls):
        assert command[1:3] == ["-m", "legged_gym.scripts.evaluate_v3_contact"]
        assert command[command.index("--scenario") + 1] == name
        assert command[command.index("--num_envs") + 1] == "32"
        assert "--video" in command and "--smoke" not in command
        assert (tmp_path / name / "summary.json").exists()
        assert (tmp_path / name).stat().st_mode & 0o222 == 0
        assert (tmp_path / name / "summary.json").stat().st_mode & 0o222 == 0


def test_contact_groups_propagate_native_failure_without_retry(tmp_path: Path, monkeypatch: Any) -> None:
    """Stop once a native child fails, retaining its logs and completed archives.

    Args:
        tmp_path: New parent archive with no prior successful child.
        monkeypatch: Fixture injecting a real subprocess failure at the SDK seam.

    Raises:
        AssertionError: A failing scene is hidden or automatically retried.
    """
    calls: List[List[str]] = []

    def fail(command: List[str], **kwargs: Any) -> None:
        """Raise the native process failure without publishing a fake summary.

        Args:
            command: Requested scene command.
            **kwargs: Original subprocess launch options.

        Raises:
            subprocess.CalledProcessError: Injected segmentation fault exit status.
        """
        calls.append(command)
        raise subprocess.CalledProcessError(-11, command)

    monkeypatch.setattr(entry.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        entry.evaluate_contact_groups(["--output_dir", str(tmp_path)], tmp_path)
    assert len(calls) == 1

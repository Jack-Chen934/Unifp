"""Verify identity at source, asset and effective-learning configuration boundaries."""

import copy
import json
from pathlib import Path

from legged_gym.b2_gym_learn.ppo_cse_pf.v3_run_identity import build_run_identity


def test_identity_allows_extra_budget_but_detects_learning_source_and_asset_changes(tmp_path: Path) -> None:
    """Keep output budgets orthogonal to the content actually used for learning.

    Args:
        tmp_path: Pytest-managed directory for a compact independent source/asset tree.

    Raises:
        AssertionError: Same-part compatibility ignores actual learning differences.
    """
    (tmp_path / "legged_gym").mkdir()
    source = tmp_path / "legged_gym/task.py"
    source.write_text("CONTROL_DT = 0.02\n")
    (tmp_path / "SOURCE_BASELINE.json").write_text(json.dumps({"commit": "fixed-official"}))
    asset_directory = tmp_path / "resources/robot/urdf"
    asset_directory.mkdir(parents=True)
    asset = asset_directory / "robot.urdf"
    asset.write_text('<robot name="v3"/>')
    mesh = asset_directory.parent / "wheel.stl"
    mesh.write_bytes(b"wheel geometry")
    configuration = {"environment": {"asset": {"file": str(asset)}, "run": {"reproduction_part": "A", "run_purpose": "preflight"}},
                     "training": {"seed": 0, "algorithm": {"gamma": 0.99},
                                  "runner": {"max_iterations": 10, "save_interval": 200}}}
    baseline = build_run_identity(tmp_path, configuration, {"device": "cpu"})
    changed = copy.deepcopy(configuration)
    changed["training"]["runner"]["max_iterations"] = 1
    changed["training"]["runner"]["save_interval"] = 1
    assert build_run_identity(tmp_path, changed, {"device": "cpu"}) == baseline
    changed["training"]["algorithm"]["gamma"] = 0.95
    assert build_run_identity(tmp_path, changed, {"device": "cpu"}) != baseline
    source.write_text("CONTROL_DT = 0.01\n")
    assert build_run_identity(tmp_path, configuration, {"device": "cpu"})["source_manifest"] != baseline["source_manifest"]
    mesh.write_bytes(b"different wheel")
    assert build_run_identity(tmp_path, configuration, {"device": "cpu"})["asset_manifest"] != baseline["asset_manifest"]

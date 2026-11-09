"""Verify identity at source, asset and effective-learning configuration boundaries."""

import copy
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest

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


def make_transition_identity_fixture(directory: Path) -> Tuple[Dict[str, Any], ...]:
    """Build real fingerprints for an A/B pair with unchanged physics and learner.

    Args:
        directory: Temporary source root containing a tiny frozen robot asset.

    Returns:
        Parent identity, child identity, parent config and child config, in order.

    Note:
        Creates only fixture files; it never opens an actual training checkpoint.
    """
    source = directory / "legged_gym/scripts/train_v3.py"
    source.parent.mkdir(parents=True)
    source.write_text("# A entry fixture\n")
    (directory / "SOURCE_BASELINE.json").write_text(json.dumps({"commit": "fixed-official"}))
    asset = directory / "resources/robot/urdf/robot.urdf"
    asset.parent.mkdir(parents=True)
    asset.write_text('<robot name="v3"/>')
    parent_config = {
        "environment": {"asset": {"file": str(asset)}, "env": {"num_envs": 1},
                        "run": {"reproduction_part": "A", "run_purpose": "formal"},
                        "control": {"decimation": 8}},
        "training": {"seed": 0, "algorithm": {"gamma": 0.99}, "policy": {},
                     "runner": {"num_steps_per_env": 24, "max_iterations": 8000,
                                "save_interval": 200, "tensorboard": False}},
    }
    parent = build_run_identity(directory, parent_config, {"device": "cpu"})
    child_config = copy.deepcopy(parent_config)
    child_config["environment"]["run"] = {"reproduction_part": "B", "run_purpose": "development"}
    child_config["training"]["runner"]["max_iterations"] = 10
    source.write_text("# B entry fixture\n")
    child = build_run_identity(directory, child_config, {"device": "cpu"})
    return parent, child, parent_config, child_config


def test_a_to_b_checks_config_and_records_only_approved_stage_differences(tmp_path: Path) -> None:
    """Permit explicit stage changes while rejecting changed physics or algorithm.

    Args:
        tmp_path: Temporary source and asset fixture directory.

    Raises:
        AssertionError: Migration accepts unrelated changes or omits its provenance.
    """
    from legged_gym.b2_gym_learn.ppo_cse_pf.v3_run_identity import validate_a_to_b_transition

    parent, child, parent_config, child_config = make_transition_identity_fixture(tmp_path)
    report = validate_a_to_b_transition(parent, child, parent_config, child_config)
    assert report["source_changes"] == ["legged_gym/scripts/train_v3.py"]
    assert set(report["config_changes"]) == {
        "environment.run.reproduction_part", "environment.run.run_purpose"}
    for section, field, value in (("algorithm", "gamma", 0.95), ("runner", "num_steps_per_env", 12)):
        changed_config = copy.deepcopy(child_config)
        changed_config["training"][section][field] = value
        changed_identity = build_run_identity(tmp_path, changed_config, {"device": "cpu"})
        with pytest.raises(ValueError, match="configuration"):
            validate_a_to_b_transition(parent, changed_identity, parent_config, changed_config)
    for field in ("asset_manifest", "schema", "source_manifest"):
        incompatible = copy.deepcopy(child)
        incompatible[field]["unapproved"] = "changed"
        with pytest.raises(ValueError):
            validate_a_to_b_transition(parent, incompatible, parent_config, child_config)
    with pytest.raises(ValueError, match="fingerprint"):
        validate_a_to_b_transition(parent, child, parent_config, parent_config)
    changed_runtime = copy.deepcopy(child)
    changed_runtime["runtime"]["python_executable"] = "/another/snapshot/python"
    assert validate_a_to_b_transition(parent, changed_runtime, parent_config, child_config)["runtime_changes"]
    changed_runtime["runtime"]["torch"] = "different dependency"
    with pytest.raises(ValueError, match="runtime"):
        validate_a_to_b_transition(parent, changed_runtime, parent_config, child_config)

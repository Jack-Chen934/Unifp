"""Check physics advancement and torque capture without native Gym imports."""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch


PROBE_PATH = Path(__file__).with_name("probe_wheels.py")


def load_advance() -> Any:
    """Load the real advancement function without importing native Gym.

    Returns:
        The diagnostic function compiled from its actual source definition.

    Raises:
        StopIteration: The shared CPU/GPU advancement function is absent.
    """
    tree = ast.parse(PROBE_PATH.read_text())
    definition = next(node for node in tree.body
                      if isinstance(node, ast.FunctionDef) and node.name == "advance_interval")
    namespace = {"Any": Any, "torch": torch,
                 "gymtorch": SimpleNamespace(unwrap_tensor=lambda value: value)}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(PROBE_PATH), "exec"), namespace)
    return namespace["advance_interval"]


@pytest.mark.parametrize("cpu_pipeline", [False, True])
def test_interval_advances_both_pipelines_and_captures_peak(cpu_pipeline: bool) -> None:
    """Require four native steps and their actual torque peak in either pipeline.

    Args:
        cpu_pipeline: Whether native results must be fetched before CPU readback.

    Raises:
        AssertionError: Simulation is skipped, fetch timing is wrong, or a
            recomputed feedback torque replaces the actual substep peak.
    """
    gym = Mock()
    environment = SimpleNamespace(
        cfg=SimpleNamespace(control=SimpleNamespace(decimation=4)),
        gym=gym, sim=object(), _update_base_state=Mock(),
        _compute_v3_torques=Mock(side_effect=[torch.full((4, 14), value)
                                             for value in (1.0, -7.0, 3.0, -2.0)]))
    peak = load_advance()(environment, torch.zeros(4, 14), cpu_pipeline)
    assert gym.simulate.call_count == 4
    assert gym.set_dof_actuation_force_tensor.call_count == 4
    assert gym.fetch_results.call_count == (4 if cpu_pipeline else 0)
    assert gym.refresh_dof_state_tensor.call_count == 4
    assert torch.equal(peak, torch.full((4, 14), 7.0))
    environment._update_base_state.assert_called_once()


def test_settle_and_motion_call_the_shared_advancement_unconditionally() -> None:
    """Catch the original call-site bug that skipped all GPU simulation steps.

    Raises:
        AssertionError: Either diagnostic loop conditionally skips advancement.
    """
    tree = ast.parse(PROBE_PATH.read_text())
    main = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "main")
    loops = [node for node in ast.walk(main) if isinstance(node, ast.For)
             and isinstance(node.target, ast.Name)
             and node.target.id in ("settle_step", "step")]
    assert len(loops) == 2
    for loop in loops:
        assignments = [node for node in loop.body if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == "applied_torque_peak"
                               for target in node.targets)]
        assert len(assignments) == 1
        assert isinstance(assignments[0].value, ast.Call)
        assert assignments[0].value.func.id == "advance_interval"

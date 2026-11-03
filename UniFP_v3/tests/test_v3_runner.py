"""Test complete learning boundaries using an explicit simulator-boundary fixture."""

import copy
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Tuple

import numpy as np
import pytest
import torch

from legged_gym.b2_gym_learn.ppo_cse_pf.v3_on_policy_runner import V3OnPolicyRunner


class TensorEnvironment:
    """Replace only the external simulator, keeping the real model/PPO/Adam stack."""

    num_envs = 1
    num_obs = 1856
    num_privileged_obs = 294
    # BaseTask exposes aggregate widths; V3 frame/label widths remain in cfg.env.
    cfg = SimpleNamespace(env=SimpleNamespace(num_pred_obs=12, num_single_obs=58))
    num_actions = 14
    dt = 0.02
    configured_reward_weights = {"test_reward": 1.0}
    reward_scales = {"test_reward": 0.02}

    def __init__(self) -> None:
        """Initialize deterministic fixture state and observable physical call counts."""
        self.progress = (0, 0)
        self.step_calls = 0
        self.reset_calls = 0
        self.pending_episode_records = []
        self.fail_on_step = None

    def set_training_progress(self, completed_iterations: int, training_policy_steps: int) -> None:
        """Publish a training clock without advancing the fixture's simulator.

        Args:
            completed_iterations: Fully committed PPO iteration count.
            training_policy_steps: Action-boundary clock in per-environment steps.
        """
        self.progress = (completed_iterations, training_policy_steps)

    def get_observations(self) -> Dict[str, torch.Tensor]:
        """Return independent published CPU tensors with the actual V3 schema.

        Returns:
            Actor/Critic histories and environment supervision labels.
        """
        return {"obs": torch.full((1, 1856), 0.01),
                "privileged_obs": torch.full((1, 294), 0.02),
                "obs_pred": torch.full((1, 12), 0.03)}

    def reset(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Start a new episode and visibly consume its random initialization.

        Returns:
            A fresh observation package and empty reset information.

        Note:
            Consumes one value from each CPU RNG to verify restore/reset ordering.
        """
        self.reset_calls += 1
        random.random()
        np.random.random()
        torch.rand(1)
        return self.get_observations(), {}

    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Simulate one deterministic transition or a deliberately injected I/O failure.

        Args:
            actions: Real policy samples [1,14]; no action-dependent dynamics here.

        Returns:
            Valid V3 observations, environment reward, done and terminal info.

        Raises:
            RuntimeError: The configured simulator failure is reached.
        """
        self.step_calls += 1
        if self.step_calls == self.fail_on_step:
            raise RuntimeError("injected simulator failure")
        infos = {"terminated": torch.zeros(1, dtype=torch.bool),
                 "time_outs": torch.zeros(1, dtype=torch.bool),
                 "terminal_env_ids": torch.empty(0, dtype=torch.long),
                 "terminal_privileged_obs": torch.empty(0, 294), "episode_records": []}
        return self.get_observations(), torch.tensor([0.1]), torch.zeros(1, dtype=torch.long), infos

    def get_training_state(self) -> Dict[str, Any]:
        """Return fixture persistence at the same public environment boundary.

        Returns:
            Detached progress plus an independently copied creation-time property.
        """
        iterations, steps = self.progress
        return {"reproduction_part": "A", "num_envs": 1, "completed_iterations": iterations,
                "training_policy_steps": steps, "training_transitions": steps,
                "friction": torch.tensor([[0.75]])}

    def load_training_state(self, state: Dict[str, Any]) -> None:
        """Restore persistence without advancing or resetting the fixture.

        Args:
            state: Same-schema creation-time properties and published progress.
        """
        assert float(state["friction"][0, 0]) == 0.75
        self.progress = state["completed_iterations"], state["training_policy_steps"]


def make_runner(directory: Path) -> V3OnPolicyRunner:
    """Build a real CPU learner with file logging and TensorBoard explicitly disabled.

    Args:
        directory: New independent run directory under pytest's temporary root.

    Returns:
        Runner using the real fixed-schema V3 model and baseline optimizer counts.
    """
    torch.set_num_threads(2)
    configuration = {"runner": {"num_steps_per_env": 24, "save_interval": 200, "tensorboard": False},
                     "policy": {}, "algorithm": {}}
    return V3OnPolicyRunner(TensorEnvironment(), configuration, directory,
                           identity={"task": "v3_pos_force", "seed": 0, "reproduction_part": "A"},
                           config_snapshot=configuration, device="cpu")


def test_checkpoint_round_trip_restores_both_adams_rng_and_next_update(tmp_path: Path) -> None:
    """Restore in independent objects and compare the next real main/auxiliary update.

    Args:
        tmp_path: Pytest-managed temporary filesystem directory.

    Raises:
        AssertionError: Learning/RNG state differs or saving consumes randomness.
    """
    first = make_runner(tmp_path / "first")
    first.learn(1)
    checkpoint = tmp_path / "first/checkpoints/model_000001.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    torch_rng = torch.get_rng_state().clone()
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    first.save(tmp_path / "first/checkpoints/verified_copy.pt")
    torch.testing.assert_close(torch.get_rng_state(), torch_rng)
    assert random.getstate() == python_rng
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_rng[1])
    assert saved["progress"] == {"completed_iterations": 1, "training_policy_steps": 24,
                                  "training_transitions": 24, "rollout_step": 0}
    assert saved["main_optimizer_state_dict"]["state"]
    assert saved["auxiliary_optimizer_state_dict"]["state"]
    restored = make_runner(tmp_path / "restored")
    restored.load(checkpoint)
    assert restored.env.reset_calls == 1
    assert restored.completed_iterations == 1 and restored.training_policy_steps == 24
    expected_rng = torch.rand(3)
    expected_python = random.random()
    expected_numpy = np.random.random()
    reference = make_runner(tmp_path / "reference")
    reference.load(checkpoint)
    torch.testing.assert_close(torch.rand(3), expected_rng)
    assert random.random() == expected_python and np.random.random() == expected_numpy
    reference.learn(1)
    expected_model = copy.deepcopy(reference.alg.actor_critic.state_dict())
    restored.load(checkpoint)
    torch.rand(3)
    random.random()
    np.random.random()
    restored.learn(1)
    for name, parameter in restored.alg.actor_critic.state_dict().items():
        torch.testing.assert_close(parameter, expected_model[name], rtol=0.0, atol=0.0)
    assert restored.lineage["parent_checkpoint_sha256"]
    with pytest.raises(FileExistsError):
        restored.save(tmp_path / "restored/checkpoints/model_000002.pt")
    broken = copy.deepcopy(saved)
    del broken["auxiliary_optimizer_state_dict"]
    broken_path = tmp_path / "broken.pt"
    torch.save(broken, broken_path)
    with pytest.raises(ValueError):
        make_runner(tmp_path / "rejected").load(broken_path)
    missing_momentum = copy.deepcopy(saved)
    missing_momentum["auxiliary_optimizer_state_dict"]["state"].popitem()
    missing_momentum_path = tmp_path / "missing_momentum.pt"
    torch.save(missing_momentum, missing_momentum_path)
    with pytest.raises(ValueError, match="Incomplete auxiliary Adam state"):
        make_runner(tmp_path / "rejected_momentum").load(missing_momentum_path)


def test_7999_to_8000_stops_without_an_extra_transition(tmp_path: Path) -> None:
    """Exercise the budget boundary without pretending to have trained 7999 real rounds.

    Args:
        tmp_path: Pytest-managed temporary filesystem directory.

    Raises:
        AssertionError: A new action/reset is requested after the completed A boundary.
    """
    runner = make_runner(tmp_path / "stop")
    runner.completed_iterations = 7999
    runner.training_policy_steps = 191976
    runner.training_transitions = 191976
    runner.env.set_training_progress(7999, 191976)
    runner.learn(1)
    assert runner.completed_iterations == 8000
    assert runner.env.step_calls == 24
    assert runner.training_policy_steps == 192000 and runner.training_transitions == 192000
    assert len(list((tmp_path / "stop/checkpoints").glob("*.pt"))) == 1
    resets = runner.env.reset_calls
    runner.learn(0)
    assert runner.env.step_calls == 24 and runner.env.reset_calls == resets
    with pytest.raises(ValueError):
        runner.learn(1)


def test_partial_collection_is_not_committed_or_saved(tmp_path: Path) -> None:
    """Preserve the original exception and reject a checkpoint after a half rollout.

    Args:
        tmp_path: Pytest-managed temporary filesystem directory.

    Raises:
        AssertionError: Failed transitions are counted as complete learning iterations.
    """
    runner = make_runner(tmp_path / "failure")
    runner.env.fail_on_step = 3
    with pytest.raises(RuntimeError, match="injected simulator failure"):
        runner.learn(1)
    assert runner.completed_iterations == 0 and runner.training_policy_steps == 0
    assert not list((tmp_path / "failure/checkpoints").glob("*.pt"))
    assert (tmp_path / "failure/failure.json").is_file()
    with pytest.raises(RuntimeError):
        runner.save(tmp_path / "failure/checkpoints/partial.pt")


def test_failed_checkpoint_publication_preserves_prior_file_and_completed_memory_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inject a filesystem failure after a real update and preserve the last checkpoint.

    Args:
        tmp_path: Pytest-managed temporary filesystem directory.
        monkeypatch: Pytest fixture replacing only the operating-system publication seam.

    Raises:
        AssertionError: A failed publication overwrites a prior file or claims success.
    """
    runner = make_runner(tmp_path / "publication")
    runner.learn(1)
    previous = tmp_path / "publication/checkpoints/model_000001.pt"
    original_bytes = previous.read_bytes()

    def reject_publication(source: object, destination: object) -> None:
        """Emulate an OS publication error while leaving previously written files alone.

        Args:
            source: Run-owned temporary checkpoint path.
            destination: New final checkpoint path.

        Raises:
            OSError: Always, representing unavailable filesystem publication.
        """
        raise OSError("injected checkpoint publication failure")

    monkeypatch.setattr("os.replace", reject_publication)
    with pytest.raises(OSError, match="publication failure"):
        runner.learn(1)
    assert runner.completed_iterations == 2 and runner.last_saved_iteration == 1
    assert previous.read_bytes() == original_bytes
    assert not (tmp_path / "publication/checkpoints/model_000002.pt").exists()
    assert not list((tmp_path / "publication/checkpoints").glob("*.tmp"))

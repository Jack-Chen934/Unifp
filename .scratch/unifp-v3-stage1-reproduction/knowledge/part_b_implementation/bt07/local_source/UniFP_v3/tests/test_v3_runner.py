"""Test complete learning boundaries using an explicit simulator-boundary fixture."""

import copy
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytest
import torch

from legged_gym.b2_gym_learn.ppo_cse_pf.v3_on_policy_runner import V3OnPolicyRunner
from legged_gym.b2_gym_learn.ppo_cse_pf.v3_run_identity import file_sha256
from test_v3_run_identity import make_transition_identity_fixture


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
        self.reproduction_part = "A"
        self.loaded_state: Dict[str, Any] = {}
        self.reset_samples: Tuple[float, float, float] = (0.0, 0.0, 0.0)

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
        self.reset_samples = (random.random(), np.random.random(), float(torch.rand(1)))
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
        state = {"reproduction_part": self.reproduction_part, "num_envs": 1, "completed_iterations": iterations,
                "training_policy_steps": steps, "training_transitions": steps,
                "friction": torch.tensor([[0.75]]), "mass_com_delta": torch.tensor([[1.0, 0.01, 0.0, 0.0, -0.1]]),
                "last_push_policy_step": 191600}
        if self.reproduction_part == "B":
            # Fixed simulator-boundary data, not an alternative event implementation.
            event = {"peak": torch.tensor([[20., 0., 0.]]), "enabled": torch.ones(1, dtype=torch.bool),
                     "start_tick": torch.zeros(1, dtype=torch.long),
                     "next_start_tick": torch.tensor([200]), "ramp_ticks": torch.tensor([50.])}
            state["force_state"] = {"command": copy.deepcopy(event), "external": copy.deepcopy(event),
                                    "pending_ee_force_world": torch.tensor([[10., 0., 0.]]),
                                    "applied_ee_force_world": torch.tensor([[20., 0., 0.]]),
                                    "episode_tick": torch.tensor([25])}
        return state

    def load_training_state(self, state: Dict[str, Any], *, transition_from_a: bool = False) -> None:
        """Restore persistence without advancing or resetting the fixture.

        Args:
            state: Same-schema creation-time properties and published progress.
            transition_from_a: Whether the Runner explicitly migrated an A checkpoint.
        """
        assert float(state["friction"][0, 0]) == 0.75
        assert state["reproduction_part"] == self.reproduction_part
        self.loaded_state = copy.deepcopy(state)
        self.progress = state["completed_iterations"], state["training_policy_steps"]


def make_runner(directory: Path, *, identity: Optional[Dict[str, Any]] = None,
                config_snapshot: Optional[Dict[str, Any]] = None) -> V3OnPolicyRunner:
    """Build a real CPU learner with file logging and TensorBoard explicitly disabled.

    Args:
        directory: New independent run directory under pytest's temporary root.
        identity: Explicit fixture fingerprint; None uses the original minimal A identity.
        config_snapshot: Full migration fixture config; None keeps the compact A config.

    Returns:
        Runner using the real fixed-schema V3 model and baseline optimizer counts.
    """
    torch.set_num_threads(2)
    configuration = {"runner": {"num_steps_per_env": 24, "save_interval": 200, "tensorboard": False},
                     "policy": {}, "algorithm": {}}
    environment = TensorEnvironment()
    if identity is None:
        identity = {"task": "v3_pos_force", "seed": 0, "reproduction_part": "A"}
    environment.reproduction_part = identity["reproduction_part"]
    return V3OnPolicyRunner(environment, configuration, directory, identity=identity,
                           config_snapshot=configuration if config_snapshot is None else config_snapshot, device="cpu")


def test_a8000_migration_restores_learning_state_before_one_fresh_b_episode(tmp_path: Path) -> None:
    """Carry real Adam state into B without changing the A parent or adding steps.

    Args:
        tmp_path: Independent temporary source and small tensor-environment runs.

    Raises:
        AssertionError: State is lost, RNG resets too early, or A/B resume is conflated.

    Note:
        Only the tensor fixture clock is moved to 8000 after one real update. This
        is a boundary test, not an actual A8000 training or native capacity result.
    """
    parent_identity, target_identity, parent_config, target_config = make_transition_identity_fixture(tmp_path / "source")
    parent = make_runner(tmp_path / "parent", identity=parent_identity, config_snapshot=parent_config)
    parent.learn(1)
    parent.alg.learning_rate = 0.000321
    parent.alg.optimizer.param_groups[0]["lr"] = 0.000321
    parent.completed_iterations, parent.training_policy_steps, parent.training_transitions = 8000, 192000, 192000
    parent.env.set_training_progress(8000, 192000)
    checkpoint = parent.log_dir / "checkpoints/fixture_a8000.pt"
    parent.save(checkpoint)
    parent_digest = file_sha256(checkpoint)
    expected_reset_samples = (random.random(), np.random.random(), float(torch.rand(1)))
    child = make_runner(tmp_path / "child", identity=target_identity, config_snapshot=target_config)
    child.load(checkpoint, transition_from_a=True)
    assert child.completed_iterations == 8000 and child.training_policy_steps == 192000
    assert child.training_transitions == 192000 and child.env.step_calls == 0
    assert child.env.reset_calls == 1 and child.episodes_ready
    assert child.env.reset_samples == expected_reset_samples
    assert child.alg.learning_rate == 0.000321 and child.alg.optimizer.param_groups[0]["lr"] == 0.000321
    assert child.alg.adaptation_module_optimizer.param_groups[0]["lr"] == 1e-5
    for name, parameter in child.alg.actor_critic.state_dict().items():
        torch.testing.assert_close(parameter, parent.alg.actor_critic.state_dict()[name], rtol=0.0, atol=0.0)
    for actual_optimizer, expected_optimizer in ((child.alg.optimizer, parent.alg.optimizer),
                                                (child.alg.adaptation_module_optimizer, parent.alg.adaptation_module_optimizer)):
        actual_state, expected_state = actual_optimizer.state_dict(), expected_optimizer.state_dict()
        assert actual_state["param_groups"] == expected_state["param_groups"]
        for parameter_id, slots in expected_state["state"].items():
            for name, value in slots.items():
                torch.testing.assert_close(actual_state["state"][parameter_id][name], value, rtol=0.0, atol=0.0)
    assert child.env.loaded_state["reproduction_part"] == "B"
    assert child.env.loaded_state["last_push_policy_step"] == 191600
    torch.testing.assert_close(child.env.loaded_state["mass_com_delta"], parent.env.get_training_state()["mass_com_delta"])
    assert file_sha256(checkpoint) == parent_digest
    assert child.lineage["parent_checkpoint_sha256"] == parent_digest
    assert child.lineage["restore_mode"] == "A_to_B"
    assert child.lineage["stage_transition"]["source_changes"] == ["legged_gym/scripts/train_v3.py"]
    with pytest.raises(ValueError, match="identity"):
        make_runner(tmp_path / "strict", identity=target_identity, config_snapshot=target_config).load(checkpoint)
    completed_a = make_runner(tmp_path / "completed_a", identity=parent_identity, config_snapshot=parent_config)
    completed_a.load(checkpoint)
    completed_a.learn(0)
    assert completed_a.env.reset_calls == 0 and completed_a.env.step_calls == 0


def test_migration_entry_rejects_wrong_parent_digest_before_loading_or_creating_gym(tmp_path: Path) -> None:
    """Reject an incorrectly named parent at the real CLI before unpickling it.

    Args:
        tmp_path: Temporary invalid checkpoint and unused output directory.

    Raises:
        AssertionError: CLI misses the explicit migration mode or opens invalid data.
    """
    checkpoint = tmp_path / "not_a_checkpoint.pt"
    checkpoint.write_bytes(b"not a pickle; its hash must be checked first")
    output_directory = tmp_path / "must_not_be_created"
    result = subprocess.run(
        [sys.executable, "-m", "legged_gym.scripts.train_v3", "--reproduction_part", "B",
         "--run_purpose", "development", "--seed", "0", "--num_envs", "1", "--max_iterations", "1",
         "--sim_device", "cpu", "--rl_device", "cpu", "--headless", "--resume", str(checkpoint),
         "--transition_from_a", "--parent_sha256", "0" * 64, "--output_dir", str(output_directory)],
        capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "Parent checkpoint SHA256 mismatch" in result.stderr, result.stderr
    assert not output_directory.exists()


def test_b_migrates_updates_and_resumes_both_adams_before_one_new_episode(tmp_path: Path) -> None:
    """Run real A-to-B-to-B learning while keeping event snapshots diagnostic only.

    Args:
        tmp_path: Independent small tensor-environment checkpoints, never formal A data.

    Raises:
        AssertionError: B rejects its budget, loses Adam slots, resets twice, or
            accepts a B checkpoint missing its required force snapshot.
    """
    parent_identity, identity, parent_config, config = make_transition_identity_fixture(tmp_path / "source")
    parent = make_runner(tmp_path / "parent", identity=parent_identity, config_snapshot=parent_config)
    parent.learn(1)
    parent.completed_iterations = 8000
    parent.training_policy_steps = parent.training_transitions = 192000
    parent.env.set_training_progress(8000, 192000)
    checkpoint = parent.log_dir / "checkpoints/fixture_a8000.pt"
    parent.save(checkpoint)
    child = make_runner(tmp_path / "child", identity=identity, config_snapshot=config)
    child.load(checkpoint, transition_from_a=True)
    child.learn(1)
    assert child.completed_iterations == 8001 and child.env.step_calls == 24
    assert child.env.reset_calls == 1
    saved_path = child.log_dir / "checkpoints/model_008001.pt"
    saved = torch.load(saved_path, map_location="cpu", weights_only=False)
    assert "force_state" in saved["environment_training_state"]
    restored = make_runner(tmp_path / "restored_b", identity=identity, config_snapshot=config)
    restored.load(saved_path)
    assert restored.env.reset_calls == 1 and restored.env.step_calls == 0
    assert restored.env.loaded_state["last_push_policy_step"] == 191600
    assert restored.lineage["parent_checkpoint_sha256"] == file_sha256(saved_path)
    restored.learn(1)
    assert restored.completed_iterations == 8002 and restored.env.reset_calls == 1
    for optimizer in (restored.alg.optimizer, restored.alg.adaptation_module_optimizer):
        assert all(int(slots["step"]) == 60 for slots in optimizer.state.values())
    broken = copy.deepcopy(saved)
    del broken["environment_training_state"]["force_state"]
    broken_path = tmp_path / "incomplete_b.pt"
    torch.save(broken, broken_path)
    rejected = make_runner(tmp_path / "rejected_b", identity=identity, config_snapshot=config)
    with pytest.raises(ValueError, match="force"):
        rejected.load(broken_path)
    assert rejected.env.reset_calls == 0


def test_migration_rejects_wrong_boundary_missing_adam_and_environment_count(tmp_path: Path) -> None:
    """Reject unusable parents before touching learner parameters or episode state.

    Args:
        tmp_path: Source fixture plus deliberately incomplete checkpoint variants.

    Raises:
        AssertionError: A rejected parent changes the new learner or starts an episode.
    """
    parent_identity, target_identity, parent_config, target_config = make_transition_identity_fixture(tmp_path / "source")
    parent = make_runner(tmp_path / "parent", identity=parent_identity, config_snapshot=parent_config)
    parent.learn(1)
    saved = torch.load(parent.log_dir / "checkpoints/model_000001.pt", map_location="cpu", weights_only=False)
    child = make_runner(tmp_path / "child", identity=target_identity, config_snapshot=target_config)
    initial_parameters = copy.deepcopy(child.alg.actor_critic.state_dict())
    invalid_cases = [("wrong_boundary", saved, "A8000")]
    complete_fixture = copy.deepcopy(saved)
    complete_fixture["progress"].update(completed_iterations=8000, training_policy_steps=192000, training_transitions=192000)
    complete_fixture["environment_training_state"].update(completed_iterations=8000, training_policy_steps=192000, training_transitions=192000)
    missing_adam = copy.deepcopy(complete_fixture)
    missing_adam["auxiliary_optimizer_state_dict"]["state"].popitem()
    invalid_cases.append(("missing_adam", missing_adam, "Incomplete auxiliary Adam"))
    wrong_count = copy.deepcopy(complete_fixture)
    wrong_count["environment_training_state"]["num_envs"] = 2
    invalid_cases.append(("wrong_count", wrong_count, "environment count"))
    for name, payload, message in invalid_cases:
        checkpoint = tmp_path / (name + ".pt")
        torch.save(payload, checkpoint)
        with pytest.raises(ValueError, match=message):
            child.load(checkpoint, transition_from_a=True)
        assert child.env.reset_calls == 0 and child.env.loaded_state == {}
        for parameter_name, value in child.alg.actor_critic.state_dict().items():
            torch.testing.assert_close(value, initial_parameters[parameter_name], rtol=0.0, atol=0.0)


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


@pytest.mark.parametrize("part,start,end", [("A", 7999, 8000), ("B", 39999, 40000)])
def test_stage_end_stops_without_an_extra_transition(tmp_path: Path, part: str, start: int, end: int) -> None:
    """Exercise the stage budget without pretending to have trained preceding rounds.

    Args:
        tmp_path: Pytest-managed temporary filesystem directory.
        part: A or B stage whose cumulative endpoint is under test.
        start: Synthetic fixture counter one complete update before the endpoint.
        end: Approved cumulative endpoint, 8000 or 40000.

    Raises:
        AssertionError: A new action/reset is requested after the completed A boundary.
    """
    runner = make_runner(tmp_path / "stop", identity={"reproduction_part": part})
    runner.completed_iterations = start
    runner.training_policy_steps = start * 24
    runner.training_transitions = start * 24
    runner.env.set_training_progress(start, start * 24)
    runner.learn(1)
    assert runner.completed_iterations == end
    assert runner.env.step_calls == 24
    assert runner.training_policy_steps == end * 24 and runner.training_transitions == end * 24
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

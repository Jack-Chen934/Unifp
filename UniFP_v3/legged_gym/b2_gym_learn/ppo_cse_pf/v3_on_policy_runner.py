"""V3 complete-update clocks, offline logs and strict atomic training checkpoints."""

import json
import math
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from .v3_actor_critic import V3ActorCritic
from .v3_episode_log import flush_episode_records, write_episode_metadata
from .v3_ppo import V3PPO
from .v3_run_identity import file_sha256, validate_a_to_b_transition, validate_force_snapshot


class V3OnPolicyRunner:
    """Keep learning progress independent of logging, resets and optimizer substeps."""

    def __init__(self, env: Any, train_cfg: Dict[str, Any], log_dir: Path,
                 identity: Dict[str, Any], config_snapshot: Dict[str, Any], device: str = "cpu") -> None:
        """Construct a learner without starting or resetting an episode.

        Args:
            env: V3 environment implementing the confirmed A04/A09 public interfaces.
            train_cfg: Effective policy, algorithm and runner dictionaries.
            log_dir: New run-owned directory; existing directories are rejected.
            identity: Source/asset/schema/effective-learning/runtime compatibility data.
            config_snapshot: Full effective configuration for provenance.
            device: Explicit shared simulation/learning tensor device.

        Raises:
            ValueError: The rollout, task part or save interval is invalid.
            FileExistsError: Another run already owns the output directory.
        """
        self.env, self.cfg = env, train_cfg["runner"]
        self.identity, self.config_snapshot = identity, config_snapshot
        self.device = torch.device(device)
        if self.cfg["num_steps_per_env"] != 24 or identity["reproduction_part"] not in {"A", "B"}:
            raise ValueError("This runner requires an explicit A/B stage with T=24")
        if type(self.cfg["save_interval"]) is not int or self.cfg["save_interval"] < 1:
            raise ValueError("save_interval must be a positive integer")
        model = V3ActorCritic(env.num_obs, env.num_privileged_obs, env.cfg.env.num_pred_obs,
                             env.cfg.env.num_single_obs, env.num_actions, **train_cfg["policy"])
        self.alg = V3PPO(model, device=str(self.device), **train_cfg["algorithm"])
        self.alg.init_storage(env.num_envs, 24, [env.num_obs], [env.num_privileged_obs], [env.cfg.env.num_pred_obs], [env.num_actions])
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=False)
        (self.log_dir / "checkpoints").mkdir()
        self.completed_iterations = 0
        self.training_policy_steps = 0
        self.training_transitions = 0
        self.last_saved_iteration: Optional[int] = None
        self.lineage = {"run_id": self.log_dir.name, "reproduction_part": identity["reproduction_part"],
                        "parent_checkpoint_path": None, "parent_checkpoint_sha256": None}
        self.at_boundary = True
        self.episodes_ready = False
        self.metadata_written = False
        self.writer: Any = None
        self.save_diagnostic: Dict[str, Any] = {}

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Collect complete additional updates within the A8000 or B40000 boundary.

        Args:
            num_learning_iterations: Additional complete updates, not a new total.
            init_at_random_ep_len: Must remain false; fictitious episode ages are unapproved.

        Raises:
            ValueError: The budget exceeds its stage end or requests random initial ages.
            RuntimeError: A prior partial update makes this runner unusable.
            OSError: Logging or verified checkpoint publication fails.

        Note:
            Commits progress only after all main/auxiliary updates and storage clear.
            On failure, preserves diagnostics and re-raises the original exception.
        """
        stage_end = 8000 if self.identity["reproduction_part"] == "A" else 40000
        stage_start = 0 if self.identity["reproduction_part"] == "A" else 8000
        if (type(num_learning_iterations) is not int or num_learning_iterations < 0
                or not stage_start <= self.completed_iterations <= stage_end
                or self.completed_iterations + num_learning_iterations > stage_end or init_at_random_ep_len):
            raise ValueError("Invalid additional stage budget or unapproved initial episode ages")
        self._require_boundary()
        if num_learning_iterations == 0:
            if self.completed_iterations != stage_end or self.last_saved_iteration != stage_end:
                raise ValueError("Zero-budget completion requires a verified stage-end checkpoint")
            return
        target = self.completed_iterations + num_learning_iterations
        records = []
        try:
            if not self.episodes_ready:
                self.env.set_training_progress(self.completed_iterations, self.training_policy_steps)
                self.env.reset()
                self.episodes_ready = True
            if not self.metadata_written:
                write_episode_metadata(self.log_dir, {
                    "identity": self.identity, "config_snapshot": self.config_snapshot, "lineage": self.lineage,
                    "configured_reward_weights": self.env.configured_reward_weights,
                    "effective_reward_weights": self.env.reward_scales, "policy_dt_s": self.env.dt,
                    "units": {"tcp": "m", "base_planar_speed": "m/s", "base_yaw_rate": "rad/s"},
                    "episode_clock": "Runner action-before clock; iteration logs use committed completion",
                    "resume_episode_policy": "all new episodes after persistent-state and RNG restore",
                })
                self.metadata_written = True
            if self.cfg.get("tensorboard", True) and self.writer is None:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"), flush_secs=10)
            self.alg.actor_critic.train()
            observations = self.env.get_observations()
            while self.completed_iterations < target:
                started = time.perf_counter()
                self.at_boundary = False
                records = []
                for rollout_step in range(24):
                    self.env.set_training_progress(self.completed_iterations, self.training_policy_steps + rollout_step)
                    actions = self.alg.act(observations["obs"], observations["privileged_obs"], observations["obs_pred"])
                    observations, rewards, dones, infos = self.env.step(actions)
                    self.alg.process_env_step(rewards, dones, infos)
                    records.extend(infos["episode_records"])
                    self.env.pending_episode_records = []
                self.alg.compute_returns(observations["privileged_obs"])
                collected = time.perf_counter()
                losses = self.alg.update()
                updated = time.perf_counter()
                # The same boundary owns parameters, both Adam histories and all clocks.
                self.completed_iterations += 1
                self.training_policy_steps += 24
                self.training_transitions += 24 * self.env.num_envs
                self.env.set_training_progress(self.completed_iterations, self.training_policy_steps)
                self.at_boundary = True
                episode_summary = flush_episode_records(self.log_dir, records)
                metrics = {"completed_iterations": self.completed_iterations,
                           "training_policy_steps": self.training_policy_steps,
                           "training_transitions": self.training_transitions,
                           "simulated_time_per_environment_s": self.training_policy_steps * self.env.dt,
                           "collection_s": collected - started, "update_s": updated - collected,
                           "value_loss": losses[0], "surrogate_loss": losses[1], "auxiliary_loss": losses[2],
                           "main_learning_rate": self.alg.learning_rate,
                           **{"auxiliary/" + key: value for key, value in losses[3].items()},
                           **{"episode/" + key: value for key, value in episode_summary.items()}}
                self._append_json("iterations.jsonl", metrics)
                if self.writer is not None:
                    for name, value in metrics.items():
                        self.writer.add_scalar(name, value, self.completed_iterations)
                    self.writer.flush()
                print(f"V3 {self.identity['reproduction_part']}: I={self.completed_iterations}, S={self.training_policy_steps}, transitions={self.training_transitions}, value={losses[0]:.5g}, aux={losses[2]:.5g}", flush=True)
                if self.completed_iterations % self.cfg["save_interval"] == 0 or self.completed_iterations == target:
                    self.save(self.log_dir / "checkpoints" / f"model_{self.completed_iterations:06d}.pt")
        except Exception as error:
            diagnostic = {"error_type": type(error).__name__, "error": str(error),
                          "errno": getattr(error, "errno", None), "checkpoint_save": self.save_diagnostic,
                          "completed_iterations": self.completed_iterations,
                          "last_saved_iteration": self.last_saved_iteration,
                          "rollout_step": self.alg.storage.step, "update_in_progress": self.alg.update_in_progress,
                          "episode_records": records + self.env.pending_episode_records}
            try:
                (self.log_dir / "failure.json").write_text(json.dumps(diagnostic, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
            except (OSError, ValueError) as diagnostic_error:
                print(f"诊断文件写入失败：{diagnostic_error}", flush=True)
            raise
        finally:
            if self.writer is not None:
                self.writer.close()
                self.writer = None

    def _append_json(self, filename: str, record: Dict[str, Any]) -> None:
        """Append and synchronize one run-owned structured record.

        Args:
            filename: JSONL basename inside this new run directory.
            record: Finite JSON-compatible metrics or checkpoint identity.

        Raises:
            OSError: Writing, flushing or syncing the record fails.
            ValueError: A numeric record contains NaN or infinity.
        """
        serialized = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
        with (self.log_dir / filename).open("a", encoding="utf-8") as output:
            output.write(serialized)
            output.flush()
            os.fsync(output.fileno())

    def _require_boundary(self) -> None:
        """Reject incomplete or contradictory learning states before save/continuation.

        Raises:
            RuntimeError: Collection/update is partial or progress is inconsistent.
        """
        if (not self.at_boundary or self.alg.storage.step != 0 or self.alg.update_in_progress
                or self.alg.transition.actions is not None or self.training_policy_steps != 24 * self.completed_iterations
                or self.training_transitions != self.env.num_envs * self.training_policy_steps):
            raise RuntimeError("A complete empty-rollout learning boundary is required")

    def _capture_rng(self) -> Dict[str, Any]:
        """Capture global CPU randomness and only the explicitly used CUDA stream.

        Returns:
            Python, NumPy, Torch CPU and named used-device CUDA RNG states.
        """
        cuda_states = {}
        if self.device.type == "cuda":
            cuda_states[str(self.device)] = torch.cuda.get_rng_state(self.device).cpu().clone()
        return {"python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state().clone(), "cuda": cuda_states}

    def _restore_rng(self, state: Dict[str, Any]) -> None:
        """Restore the saved streams after object construction and before a new reset.

        Args:
            state: Exact CPU and used-CUDA state dictionary from the checkpoint.

        Raises:
            ValueError: The recorded CUDA device set differs from the active device.
        """
        expected_devices = {str(self.device)} if self.device.type == "cuda" else set()
        if set(state["cuda"]) != expected_devices:
            raise ValueError("Checkpoint CUDA RNG device set mismatch")
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"].cpu())
        for device, rng in state["cuda"].items():
            torch.cuda.set_rng_state(rng.cpu(), torch.device(device))

    @staticmethod
    def validate_checkpoint(payload: Dict[str, Any], algorithm: V3PPO, identity: Dict[str, Any], num_envs: int) -> None:
        """Validate the training format and parameter/state associations before mutation.

        Args:
            payload: CPU-loaded complete-boundary dictionary.
            algorithm: Fresh same-schema model and optimizers used only for validation.
            identity: Expected source/asset/schema/runtime compatibility record.
            num_envs: Required number of parallel training environments.

        Raises:
            ValueError: Required state, identity, tensor structure or progress is invalid.
        """
        required = {"format_version", "checkpoint_kind", "model_state_dict", "main_optimizer_state_dict",
                    "auxiliary_optimizer_state_dict", "main_parameter_names", "auxiliary_parameter_names",
                    "learning_rate_state", "scheduler_states", "progress", "rng_state",
                    "environment_training_state", "identity", "config_snapshot", "lineage"}
        if not required.issubset(payload) or payload["format_version"] != 1 or payload["checkpoint_kind"] != "training_boundary":
            raise ValueError("Incomplete or unknown training checkpoint format")
        if payload["identity"] != identity or payload["scheduler_states"] != {}:
            raise ValueError("Checkpoint learning/source/asset/schema/runtime identity mismatch")
        progress = payload["progress"]
        if set(progress) != {"completed_iterations", "training_policy_steps", "training_transitions", "rollout_step"} or any(type(value) is not int for value in progress.values()):
            raise ValueError("Invalid checkpoint progress fields")
        iterations = progress["completed_iterations"]
        steps = progress["training_policy_steps"]
        stage_start, stage_end = (0, 8000) if identity["reproduction_part"] == "A" else (8000, 40000)
        if not stage_start <= iterations <= stage_end or steps != 24 * iterations or progress["training_transitions"] != num_envs * steps or progress["rollout_step"] != 0:
            raise ValueError("Checkpoint progress is not a complete stage boundary")
        environment_state = payload["environment_training_state"]
        if environment_state.get("num_envs") != num_envs or environment_state.get("reproduction_part") != identity["reproduction_part"]:
            raise ValueError("Checkpoint environment count or stage identity mismatch")
        if any(environment_state.get(name) != progress[name] for name in ("completed_iterations", "training_policy_steps", "training_transitions")):
            raise ValueError("Environment and learning progress disagree")
        if identity["reproduction_part"] == "B":
            validate_force_snapshot(environment_state.get("force_state"), num_envs)
        current_model = algorithm.actor_critic.state_dict()
        saved_model = payload["model_state_dict"]
        if set(saved_model) != set(current_model):
            raise ValueError("Checkpoint model parameter names differ")
        for name, tensor in saved_model.items():
            reference = current_model[name]
            if not isinstance(tensor, torch.Tensor) or tensor.shape != reference.shape or tensor.dtype != reference.dtype or not torch.isfinite(tensor).all():
                raise ValueError(f"Invalid model tensor: {name}")
        if not (saved_model["std"] > 0.0).all():
            raise ValueError("Checkpoint action standard deviation must be positive")
        rates = payload["learning_rate_state"]
        for prefix, parameters, names in (("main", algorithm.main_parameters, algorithm.main_parameter_names),
                                          ("auxiliary", algorithm.auxiliary_parameters, algorithm.auxiliary_parameter_names)):
            if payload[prefix + "_parameter_names"] != [names]:
                raise ValueError(f"Checkpoint {prefix} parameter ordering differs")
            optimizer_state = payload[prefix + "_optimizer_state_dict"]
            groups = optimizer_state["param_groups"]
            if len(groups) != 1 or len(groups[0]["params"]) != len(parameters):
                raise ValueError(f"Invalid {prefix} optimizer groups")
            rate = rates[prefix]
            if not math.isfinite(rate) or rate <= 0.0 or groups[0]["lr"] != rate or (prefix == "auxiliary" and rate != 1e-5):
                raise ValueError(f"Invalid {prefix} learning rate state")
            parameter_ids = groups[0]["params"]
            if len(set(parameter_ids)) != len(parameter_ids) or not set(optimizer_state["state"]).issubset(parameter_ids):
                raise ValueError(f"Invalid {prefix} Adam parameter associations")
            for parameter_id, parameter in zip(parameter_ids, parameters):
                state = optimizer_state["state"].get(parameter_id, {})
                if (iterations > 0 or state) and not {"step", "exp_avg", "exp_avg_sq"}.issubset(state):
                    raise ValueError(f"Incomplete {prefix} Adam state")
                for name, tensor in state.items():
                    if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all():
                        raise ValueError(f"Invalid {prefix} Adam tensor: {name}")
                    if name == "step":
                        if tensor.numel() != 1 or float(tensor) < 0.0:
                            raise ValueError("Invalid Adam step")
                    elif tensor.shape != parameter.shape or tensor.dtype != parameter.dtype:
                        raise ValueError(f"Invalid {prefix} Adam tensor shape/dtype: {name}")
        rng = payload["rng_state"]
        if set(rng) != {"python", "numpy", "torch", "cuda"} or not isinstance(rng["torch"], torch.Tensor):
            raise ValueError("Incomplete checkpoint RNG state")
        if set(rng["cuda"]) != ({str(algorithm.device)} if algorithm.device.type == "cuda" else set()):
            raise ValueError("Checkpoint CUDA RNG device set mismatch")

    def save(self, path: Path) -> None:
        """Publish a validated full checkpoint atomically without consuming randomness.

        Args:
            path: New .pt target inside this run's checkpoints directory.

        Raises:
            RuntimeError: Learning is not at a complete empty-rollout boundary.
            FileExistsError: The requested target already exists.
            ValueError: Payload validation fails.
            OSError: Serialization, fsync, readback, hashing or publication fails.

        Note:
            Keeps previous published files; cleans only its own unpublished temp file.
            A published file is retained if subsequent directory sync/logging fails.
        """
        self._require_boundary()
        path = Path(path)
        if path.parent.resolve() != (self.log_dir / "checkpoints").resolve() or path.suffix != ".pt":
            raise ValueError("Checkpoint target must be a .pt in this run's checkpoints directory")
        if path.exists():
            raise FileExistsError(path)
        self.save_diagnostic = {"path": str(path), "operation": "capture_state"}
        payload = {"format_version": 1, "checkpoint_kind": "training_boundary",
                   "model_state_dict": self.alg.actor_critic.state_dict(),
                   "main_optimizer_state_dict": self.alg.optimizer.state_dict(),
                   "auxiliary_optimizer_state_dict": self.alg.adaptation_module_optimizer.state_dict(),
                   "main_parameter_names": [self.alg.main_parameter_names],
                   "auxiliary_parameter_names": [self.alg.auxiliary_parameter_names],
                   "learning_rate_state": {"main": self.alg.learning_rate, "auxiliary": self.alg.adaptation_module_optimizer.param_groups[0]["lr"]},
                   "scheduler_states": {}, "progress": {"completed_iterations": self.completed_iterations,
                       "training_policy_steps": self.training_policy_steps, "training_transitions": self.training_transitions, "rollout_step": 0},
                   "environment_training_state": self.env.get_training_state(), "rng_state": self._capture_rng(),
                   "identity": self.identity, "config_snapshot": self.config_snapshot, "lineage": self.lineage}
        self.save_diagnostic["operation"] = "create_temporary"
        descriptor, temporary_name = tempfile.mkstemp(prefix="." + path.stem + "_", suffix=".tmp", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            self.save_diagnostic["operation"] = "serialize_flush_fsync"
            with os.fdopen(descriptor, "wb") as output:
                torch.save(payload, output)
                output.flush()
                os.fsync(output.fileno())
            self.save_diagnostic["operation"] = "readback_validate"
            readback = torch.load(temporary, map_location="cpu", weights_only=False)
            self.validate_checkpoint(readback, self.alg, self.identity, self.env.num_envs)
            self.save_diagnostic["operation"] = "sha256"
            digest = file_sha256(temporary)
            if path.exists():
                raise FileExistsError(path)
            self.save_diagnostic["operation"] = "replace"
            os.replace(temporary, path)
            self.save_diagnostic["operation"] = "directory_fsync"
            directory_descriptor = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
            self.save_diagnostic["operation"] = "record_publication"
            self._append_json("checkpoints.jsonl", {"path": str(path), "sha256": digest, **payload["progress"]})
            self.last_saved_iteration = self.completed_iterations
            self.save_diagnostic["operation"] = "complete"
        finally:
            if temporary.exists():
                temporary.unlink()

    def load(self, path: Path, *, transition_from_a: bool = False) -> None:
        """Strictly restore learning state, then start new episodes from restored RNG.

        Args:
            path: Explicit complete checkpoint; no automatic latest-file selection.
            transition_from_a: Validate an A8000 parent against the approved B
                differences. False requires the original strict same-stage identity.

        Raises:
            RuntimeError: This runner already holds partial learning state.
            ValueError: Source identity, state or output lineage is incompatible.
            OSError: The explicit source cannot be read.

        Note:
            Restores both Adam histories and actual lr before RNG, then resets once.
            Does not resume PhysX caches or unfinished episode histories.
        """
        self._require_boundary()
        if self.metadata_written:
            raise RuntimeError("Resume requires a fresh output run before learning begins")
        path = Path(path).resolve()
        if self.log_dir.resolve() in path.parents:
            raise ValueError("Resume requires an independent output run directory")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        transition_report = None
        if transition_from_a:
            transition_report = validate_a_to_b_transition(
                payload["identity"], self.identity, payload["config_snapshot"], self.config_snapshot)
            self.validate_checkpoint(payload, self.alg, payload["identity"], self.env.num_envs)
            if payload["progress"]["completed_iterations"] != 8000:
                raise ValueError("A-to-B migration requires the completed A8000 boundary")
        else:
            self.validate_checkpoint(payload, self.alg, self.identity, self.env.num_envs)
        # Map only the approved stage field; creation samples and clocks stay intact.
        environment_state = dict(payload["environment_training_state"])
        if transition_from_a:
            environment_state["reproduction_part"] = "B"
        self.alg.actor_critic.load_state_dict(payload["model_state_dict"], strict=True)
        self.alg.optimizer.load_state_dict(payload["main_optimizer_state_dict"])
        self.alg.adaptation_module_optimizer.load_state_dict(payload["auxiliary_optimizer_state_dict"])
        self.alg.learning_rate = payload["learning_rate_state"]["main"]
        for name in ("completed_iterations", "training_policy_steps", "training_transitions"):
            setattr(self, name, payload["progress"][name])
        self.env.load_training_state(environment_state, transition_from_a=transition_from_a)
        self._restore_rng(payload["rng_state"])
        self.env.set_training_progress(self.completed_iterations, self.training_policy_steps)
        # A verified completed stage needs no new episode or sample.
        stage_end = 8000 if self.identity["reproduction_part"] == "A" else 40000
        if transition_from_a or self.completed_iterations < stage_end:
            self.env.reset()
            self.episodes_ready = True
        self.last_saved_iteration = self.completed_iterations
        self.lineage["parent_checkpoint_path"] = str(path)
        self.lineage["parent_checkpoint_sha256"] = file_sha256(path)
        self.lineage["restore_mode"] = "A_to_B" if transition_from_a else "same_stage"
        if transition_report is not None:
            self.lineage["stage_transition"] = transition_report
            self.lineage["parent_identity"] = payload["identity"]
            self.lineage["fresh_episodes_after_restore"] = True
        self._require_boundary()

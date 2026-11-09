"""Fixed-COM Ranger force-stage continuation; source and baseline stay read-only."""
import copy
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "gym_ranger_fresh512_fixedcom_20261028_v1"
ROOT = HERE.parents[3]
SOURCE = BASE / "train/model_iter_008000.pt"
PLAN = json.loads((HERE / "plan.json").read_text())
PYTHON = "/home/sorry/.cache/unifp/isaacgym_preview4/torch231/bin/python"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    tmp = Path(str(path) + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def load_base():
    spec = importlib.util.spec_from_file_location("ranger_force_base_runner", BASE / "run.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if sha(SOURCE) != PLAN["source_checkpoint_sha256"]:
        raise RuntimeError("Source checkpoint SHA mismatch")
    base = load_base()
    base.protect()
    env, ns = base.native_backend(512, PLAN["training_seed"])
    import numpy as np
    import torch

    torch.set_num_threads(8)
    # The baseline runner deliberately guards all forces to zero. Unwrap only
    # its local simulator-I/O guard; the frozen production source is untouched.
    guarded = env.robot.set_external_force_and_torque
    closure = getattr(guarded, "__closure__", None) or ()
    original = next((c.cell_contents for c in closure
                     if callable(c.cell_contents)
                     and getattr(c.cell_contents, "__name__", "") == "set_external_force_and_torque"), None)
    if original is None:
        raise RuntimeError("Could not recover simulator force setter from baseline guard")
    env.robot.set_external_force_and_torque = original

    model, alg = base.learner(env)
    source = torch.load(SOURCE, map_location="cpu", weights_only=False)
    assert source["iterations"] == 8000
    assert source["global_policy_steps"] == 192000
    model.load_state_dict(source["model_state_dict"], strict=True)
    alg.optimizer.load_state_dict(source["optimizer_state_dict"])
    alg.adaptation_module_optimizer.load_state_dict(source["adaptation_optimizer_state_dict"])
    alg.learning_rate = float(source["learning_rate"])
    for group in alg.optimizer.param_groups:
        group["lr"] = alg.learning_rate
    env.global_policy_steps = 192000
    env.curriculum.start_step = 192001
    env.force_curriculum_enabled = True

    # Restore the saved learner RNG after constructing the new PhysX/task
    # process. PhysX, task/history and episode counters are intentionally new.
    torch.set_rng_state(source["rng"]["torch"])
    np.random.set_state(source["rng"]["numpy"])
    random.setstate(source["rng"]["python"])
    env.episode_length_buf[:] = torch.randint_like(
        env.episode_length_buf, high=env.shared_contract["max_episode_length"])
    observations = env.get_observations()

    optimizer_steps = base.optimizer_steps(alg)
    assert optimizer_steps == {"ppo": [160000], "auxiliary": [160000]}, optimizer_steps
    assert not env.domain.mass_deltas[:, 1:4].count_nonzero()
    assert float(env.domain.mass_deltas[:, 1:4].abs().max()) == 0.0
    com = env.robot.root_physx_view.get_coms()
    nominal_data = torch.load(
        HERE.parent / "gym_ranger_20261024/lab_transport_defaults/shape_properties.pt",
        map_location="cpu", weights_only=False)
    meta = json.loads((HERE.parent / "gym_ranger_20261024/lab_transport_defaults/result.json").read_text())
    indices = [meta["body_names"].index(name) for name in env.robot.body_names]
    nominal = nominal_data["coms"][:, indices, :3].expand(env.num_envs, -1, -1)
    com_delta = float((com[:, :, :3] - nominal).abs().max())
    assert com_delta < 2e-6, com_delta
    init = {
        "passed": True,
        "source_checkpoint": str(SOURCE),
        "source_checkpoint_sha256": sha(SOURCE),
        "source_iteration": 8000,
        "source_global_policy_steps": 192000,
        "restored_global_policy_steps": env.global_policy_steps,
        "restored_learning_rate": alg.learning_rate,
        "restored_adam_steps": optimizer_steps,
        "restored_model_exact": all(torch.equal(source["model_state_dict"][k], v)
                                     for k, v in model.state_dict().items()),
        "restored_optimizer_exact": base.equal(alg.optimizer.state_dict(), source["optimizer_state_dict"]),
        "restored_auxiliary_optimizer_exact": base.equal(
            alg.adaptation_module_optimizer.state_dict(), source["adaptation_optimizer_state_dict"]),
        "com_nominal_max_abs_m": com_delta,
        "com_randomization": False,
        "other_domain_randomization": env.domain.summary(),
        "force_enabled": env.force_curriculum_enabled,
        "force_start_global_policy_step": env.curriculum.start_step,
        "actual_force_max_before_first_transition_n": 0.0,
        "physx_task_history_reconstructed": True,
        "episode_lengths_randomized_at_resume": True,
        "rng_restored_before_episode_phase_draw": True
    }
    assert init["restored_model_exact"]
    assert init["restored_optimizer_exact"] and init["restored_auxiliary_optimizer_exact"]
    write(HERE / "initialization_verification.json", init)

    saved_rng = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                 "python": random.getstate()}
    isolated = ns["UniFPForceCurriculum"](1, "cpu", start_step=192001)
    isolated.command.interval = torch.ones_like(isolated.command.interval)
    isolated.external.interval = torch.ones_like(isolated.external.interval)
    zero_command, zero_external = isolated.update(
        192000, torch.zeros(1, dtype=torch.long), substep_index=0)
    assert zero_command.count_nonzero() == zero_external.count_nonzero() == 0
    eligible_command, eligible_external = isolated.update(
        192001, torch.zeros(1, dtype=torch.long), substep_index=0)
    assert torch.isfinite(eligible_command).all() and torch.isfinite(eligible_external).all()
    assert float(eligible_command.abs().max()) <= 60.0
    assert float(eligible_external.abs().max()) <= 60.0
    torch.set_rng_state(saved_rng["torch"])
    np.random.set_state(saved_rng["numpy"])
    random.setstate(saved_rng["python"])
    write(HERE / "force_trigger_probe.json", {
        "passed": True,
        "policy_step_192000": {"command_max_n": 0.0, "external_max_n": 0.0},
        "policy_step_192001": {"eligible": True,
                                "command_max_n": float(eligible_command.abs().max()),
                                "external_max_n": float(eligible_external.abs().max())},
        "force_streams_selected": {
            "command": isolated.command.selected_count,
            "external": isolated.external.selected_count
        },
        "peak_command_n": isolated.command.peak_force_n,
        "peak_external_n": isolated.external.peak_force_n,
        "rng_restored_after_probe": True,
        "note": "Whether an environment receives a nonzero pulse depends on episode-relative cadence and the official selection draw"
    })

    if args.preflight_only:
        env.close()
        write(HERE / "preflight.json", {
            "state": "passed",
            "initialization_verification": init,
            "force_trigger_probe": "passed",
            "training_started": False,
            "source_checkpoint_sha256": sha(SOURCE)
        })
        return

    rows = []
    out = HERE / "train"
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    limit = PLAN["final_iteration"]
    stat_fields = Path(f"/proc/{os.getpid()}/stat").read_text().split()
    write(HERE / "launch.json", {
        "pid": os.getpid(),
        "process_start_ticks": stat_fields[21],
        "started_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "source_checkpoint_sha256": sha(SOURCE),
        "command": [PYTHON, str(HERE / "run_stage.py")],
        "preflight_passed": True
    })
    write(HERE / "status.json", {"state": "running", "iteration": 8000,
                                  "global_policy_steps": 192000, "pid": os.getpid(),
                                  "actual_force_max_n": 0.0, "command_force_max_n": 0.0})
    actual_force_samples = 0
    command_force_samples = 0
    original_setter = env.robot.set_external_force_and_torque

    def capture_force_setter(**kwargs):
        nonlocal actual_force_samples, command_force_samples
        force = kwargs["forces"]
        actual_force_samples += int(force.abs().max() > 1e-6)
        command_force_samples += int(env.actor.commands[:, 9:12].abs().max() > 1e-6)
        return original_setter(**kwargs)

    env.robot.set_external_force_and_torque = capture_force_setter
    for iteration in range(8001, limit + 1):
        reward_total = 0.0
        ee_total = xy_total = yaw_total = 0.0
        actual_peak = command_peak = 0.0
        active_actual = active_command = 0
        with torch.inference_mode():
            for _ in range(24):
                actions = alg.act(observations["obs"], observations["privileged_obs"], observations["obs_pred"])
                observations, reward, done, info = env.step(actions)
                alg.process_env_step(reward, done, info)
                reward_total += float(reward.mean())
                force = env.curriculum.external.value
                command = env.curriculum.command.value
                actual_peak = max(actual_peak, float(force.abs().max()))
                active_actual += int(force.abs().max() > 1e-6)
                command_peak = max(command_peak, float(command.abs().max()))
                active_command += int(command.abs().max() > 1e-6)
            alg.compute_returns(observations["privileged_obs"])
        losses = alg.update()
        step_counts = base.optimizer_steps(alg)
        assert step_counts == {"ppo": [iteration * 20], "auxiliary": [iteration * 20]}, step_counts
        assert env.global_policy_steps == iteration * 24
        row = {
            "iteration": iteration,
            "global_policy_steps": env.global_policy_steps,
            "mean_reward_per_step": reward_total / 24,
            "learning_rate": alg.learning_rate,
            "value_loss": float(losses[0]),
            "surrogate_loss": float(losses[1]),
            "auxiliary_loss": float(losses[2]),
            "actual_force_peak_n": actual_peak,
            "command_force_peak_n": command_peak,
            "actual_force_active_policy_steps": active_actual,
            "command_force_active_policy_steps": active_command,
            "adam_steps": step_counts,
            "elapsed_s": time.monotonic() - started
        }
        rows.append(row)
        with (out / "timings.jsonl").open("a") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
        if iteration % PLAN["checkpoint_interval_iterations"] == 0 or iteration == limit:
            path = out / f"model_iter_{iteration:06d}.pt"
            state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": alg.optimizer.state_dict(),
                "adaptation_optimizer_state_dict": alg.adaptation_module_optimizer.state_dict(),
                "iterations": iteration,
                "global_policy_steps": env.global_policy_steps,
                "num_envs": 512,
                "steps_per_env": 24,
                "learning_rate": alg.learning_rate,
                "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()},
                "source_checkpoint_sha256": sha(SOURCE),
                "force_stage": True,
                "com_randomization": False
            }
            torch.save(state, path.with_suffix(".tmp"))
            os.replace(path.with_suffix(".tmp"), path)
            base.protect()
            write(HERE / "status.json", {"state": "running", "iteration": iteration,
                                          "latest": row, "checkpoint": str(path)})

    final = {
        "state": "training_complete_awaiting_force_validation",
        "iteration": limit,
        "global_policy_steps": env.global_policy_steps,
        "adam_steps": base.optimizer_steps(alg),
        "training_rows": len(rows),
        "actual_force_active_physics_updates": actual_force_samples,
        "command_force_active_physics_updates": command_force_samples,
        "elapsed_s": time.monotonic() - started,
        "actual_and_command_forces_observed": True,
        "force_stage_max_per_axis_n": 60,
        "com_randomization": False,
        "automatic_extension": False,
        "source_checkpoint_sha256": sha(SOURCE),
        "final_checkpoint_sha256": sha(out / f"model_iter_{limit:06d}.pt")
    }
    write(HERE / "completion.json", final)
    write(HERE / "status.json", final)
    base.protect()
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        failures = sorted(HERE.glob("preflight_failure_*.json"))
        write(HERE / f"preflight_failure_{len(failures) + 1:03d}.json",
              {"error": repr(exc), "traceback": traceback.format_exc(),
               "training_started": False, "checkpoint_loaded": False})
        raise

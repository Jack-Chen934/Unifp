"""Verify the approved 4096-robot peak-load capacity without policy updates."""

import json
import resource
import time
from pathlib import Path
from typing import Any, Dict

import isaacgym  # noqa: F401
import torch
from legged_gym.b2_gym_learn.ppo_cse_pf.v3_actor_critic import V3ActorCritic
from legged_gym.envs.v3.legged_robot_v3 import V3PosForce
from legged_gym.scripts.v3_args import get_v3_args
from legged_gym.utils.helpers import class_to_dict, parse_sim_params, set_seed
from legged_gym.utils.v3_config import resolve_v3_configs


class PeakLoadEnvironment(V3PosForce):
    """Replace only the event source with simultaneous world-space cube peaks."""

    def _advance_force_events(self, env_ids: torch.Tensor) -> None:
        """Hold both world forces at (+60,0,+60) N across local resets.

        Args:
            env_ids: Logical robot rows [K] at the publication boundary.

        Note:
            External force is submitted at the existing COM slot. Command force
            is expressed in current yaw coordinates and never submitted to PhysX.
            X/Z reach the approved maximum; yaw rotation preserves the horizontal
            component bounds for every heading. Force norm is about 85 N.
        """
        peak_n = self.cfg.force_events.max_force_n
        self.pending_ee_force_world[env_ids] = torch.tensor((peak_n, 0., peak_n), device=self.device)
        yaw_rad = self.rpy[env_ids, 2]
        self.commands[env_ids, 9:12] = torch.stack((
            peak_n * yaw_rad.cos(),
            -peak_n * yaw_rad.sin(),
            torch.full_like(yaw_rad, peak_n),
        ), dim=1)


def main() -> None:
    """Run 500 policy intervals with parent creation samples and no optimizer.

    Raises:
        AssertionError: Load coverage, finite state or parent dimensions differ.
        FileExistsError: The explicit output directory already exists.

    Note:
        Creates an isolated simulator and diagnostic files, then destroys it.
        Ordinary physical terminations retain the production local reset path.
    """
    args = get_v3_args("V3 B full-scale zero-update peak-load check")
    assert args.run_purpose == "preflight" and args.reproduction_part == "B"
    assert args.num_envs == 4096 and args.checkpoint
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg, training = resolve_v3_configs(args, purpose="preflight")
    assert cfg.force_events.max_force_n == 60.
    snapshot = {"environment": class_to_dict(cfg), "training": class_to_dict(training)}
    params = parse_sim_params(args, {"sim": snapshot["environment"]["sim"]})
    set_seed(args.seed)
    torch.cuda.set_device(args.rl_device)
    properties = {key: payload["environment_training_state"][key]
                  for key in ("mass_com_delta", "friction")}
    environment = PeakLoadEnvironment(cfg, params, args.physics_engine, args.sim_device,
                                     True, creation_properties=properties)
    try:
        model = V3ActorCritic(1856, 294, 12, 58, 14,
                             **payload["config_snapshot"]["training"]["policy"]).to(environment.device)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.eval()
        observations = environment.get_observations()
        started = time.perf_counter()
        failures = 0
        with (output / "peak_trace.jsonl").open("x") as trace:
            for policy_step in range(500):
                expected_world_n = torch.tensor((60., 0., 60.), device=environment.device)
                assert bool((environment.pending_ee_force_world == expected_world_n).all())
                assert bool((environment.commands[:, 9:12].abs() <= 60.).all())
                with torch.inference_mode():
                    actions = model.act_inference(observations, {})
                observations, _, _, info = environment.step(actions)
                # Check all robots on-device; store aggregate evidence, not millions of rows.
                for tensor in (*observations.values(), environment.root_states, environment.dof_pos,
                               environment.dof_vel, environment.torques):
                    assert bool(torch.isfinite(tensor).all())
                failures += int(info["terminated"].sum())
                trace.write(json.dumps({"policy_step": policy_step, "robots_loaded": 4096,
                    "external_world_n": [60., 0., 60.], "command_world_n": [60., 0., 60.],
                    "physical_failures": int(info["terminated"].sum()), "finite_core": True}) + "\n")
        report: Dict[str, Any] = {"passed": True, "num_envs": 4096, "policy_steps": 500,
            "simulated_duration_s": 500 * environment.dt, "policy_updates": 0,
            "external_world_n": [60., 0., 60.], "command_world_n": [60., 0., 60.],
            "physical_failures": failures, "finite_core": True, "parent_creation_samples": True,
            "wall_time_s": time.perf_counter() - started,
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "cpu_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "effective_config": snapshot}
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({key: value for key, value in report.items() if key != "effective_config"}), flush=True)
    finally:
        environment.gym.destroy_sim(environment.sim)


if __name__ == "__main__":
    main()

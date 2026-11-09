"""Scripted force sources for read-only V3 evaluation; control stays in V3PosForce."""

from typing import Any, Dict, List

import torch

from .legged_robot_v3 import V3PosForce


class V3ForceEvaluation(V3PosForce):
    """Evaluate M/E/C world-axis events or the unchanged random B event source."""

    def __init__(self, cfg: Any, sim_params: Any, physics_engine: Any, sim_device: str,
                 headless: bool, *, scenario: Dict[str, Any], episodes_per_env: int, video: bool = False) -> None:
        """Construct one fixed scenario before the parent initializes and resets.

        Args:
            cfg: Fresh B evaluation config; task force range is in newtons.
            sim_params: Preview4 native simulator settings.
            physics_engine: PhysX engine identifier.
            sim_device: Explicit simulator tensor device shared with inference.
            headless: Whether to omit the viewer.
            scenario: Mode M/E/C/R, amplitude fraction, moving and push flags.
            episodes_per_env: Fixed quota used to balance the six world directions.
            video: Allocate graphics for the optional representative episode recording.

        Note:
            Creates evaluation-only state, with no optimizer or training restoration.
        """
        self.force_scenario = scenario
        self.evaluation_episodes_per_env = episodes_per_env
        self.video_enabled = video
        self.video_recorder = None
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def create_sim(self) -> None:
        """Enable optional camera graphics while delegating all physics creation.

        Note:
            Uses the same camera allocation pattern as the existing replay tool.
        """
        if self.video_enabled:
            self.graphics_device_id = self.sim_device_id
        super().create_sim()

    def _initialize_force_events(self) -> None:
        """Allocate normal force state and deterministic episode-direction counters.

        Note:
            Counters start at -1 so the parent's first reset selects episode zero.
        """
        super()._initialize_force_events()
        self.evaluation_episode_indices = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        self.evaluation_direction_indices = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    def _reset_force_events(self, env_ids: torch.Tensor) -> None:
        """Select the next fixed direction and restart only the requested robots.

        Args:
            env_ids: Long robot indices [K], with local episode ticks already zero.

        Note:
            Failures count as episodes; direction selection never examines rewards.
        """
        self.evaluation_episode_indices[env_ids] += 1
        self.evaluation_direction_indices[env_ids] = (
            env_ids * self.evaluation_episodes_per_env + self.evaluation_episode_indices[env_ids]
        ) % 6
        if self.force_scenario["mode"] == "R":
            super()._reset_force_events(env_ids)
        else:
            self.applied_ee_force_world[env_ids] = 0.
            self.rigid_body_forces[self.ee_force_body_indices[env_ids]] = 0.
            self._advance_force_events(env_ids)

    def _advance_force_events(self, env_ids: torch.Tensor) -> None:
        """Publish fixed world scripts in current yaw coordinates or delegate R events.

        Args:
            env_ids: Selected robot rows [K] at one 20 ms publication boundary.

        Note:
            Never changes applied labels or steps physics. Two five-second events
            start at 1 and 9 seconds, with 2 s rise, 1 s hold and 2 s fall.
        """
        mode = self.force_scenario["mode"]
        if mode == "R":
            super()._advance_force_events(env_ids)
            return
        time_s = self.episode_length_buf[env_ids] * self.dt
        envelope = torch.zeros_like(time_s)
        for start_s in (1., 9.):
            age_s = time_s - start_s
            envelope += torch.minimum(age_s / 2., (5. - age_s) / 2.).clamp(0., 1.)
        directions = torch.tensor(((1., 0., 0.), (-1., 0., 0.), (0., 1., 0.),
                                   (0., -1., 0.), (0., 0., 1.), (0., 0., -1.)), device=self.device)
        world_force = directions[self.evaluation_direction_indices[env_ids]] * envelope[:, None]
        world_force *= self.cfg.force_events.max_force_n * self.force_scenario["amplitude_fraction"]
        self.pending_ee_force_world[env_ids] = world_force if mode != "C" else 0.
        command_world = -world_force if mode == "M" else world_force if mode == "C" else torch.zeros_like(world_force)
        yaw = self.rpy[env_ids, 2]
        self.commands[env_ids, 9:12] = torch.stack((
            yaw.cos() * command_world[:, 0] + yaw.sin() * command_world[:, 1],
            -yaw.sin() * command_world[:, 0] + yaw.cos() * command_world[:, 1], command_world[:, 2],
        ), dim=1)

    def _build_policy_trace(self, base_push_applied: bool, terminated: torch.Tensor,
                            time_outs: torch.Tensor) -> List[Dict[str, Any]]:
        """Add independent executed event phases and matching-force yaw residual.

        Args:
            base_push_applied: Whether the independent velocity push occurred.
            terminated: Physical failure flags [N] before reset.
            time_outs: Pure timeout flags [N] before reset.

        Returns:
            Parent rows with world force, per-stream phase and saturation data.
            Random vector events have no fixed-axis direction index. The legacy
            phase selects external events, or command events in command-only mode.
        """
        rows = super()._build_policy_trace(base_push_applied, terminated, time_outs)
        applied_world = self.applied_ee_force_world.detach().cpu().tolist()
        residual_world = ((self._compensated_target_world() - self._geometric_target_world())
                          * self.gripper_force_kps).detach().cpu().tolist()
        saturated = (self.torques.abs() >= self.torque_limits * .999).float().mean(dim=1).cpu().tolist()
        for row in rows:
            env_id = row["env_id"]
            time_s = row["policy_input_time_s"]
            mode = self.force_scenario["mode"]
            # Each random source has its own clock and disabled cycles remain zero.
            for stream in ("command", "external"):
                if mode == "R":
                    state = self.force_state[stream]
                    age_s = time_s - float(state["start_tick"][env_id]) * self.dt
                    ramp_s = float(state["ramp_ticks"][env_id]) * self.dt
                    enabled = bool(state["enabled"][env_id])
                else:
                    age_s = time_s - (9. if time_s >= 9. else 1.)
                    ramp_s = 2.
                    enabled = mode != ("E" if stream == "command" else "C")
                phase = ("zero" if not enabled or age_s <= 0. or age_s >= 2 * ramp_s + 1.
                         else "rise" if age_s < ramp_s else "hold" if age_s <= ramp_s + 1. else "fall")
                row[f"{stream}_force_phase"] = phase
                row[f"{stream}_force_hold_tail"] = phase == "hold" and age_s >= ramp_s + .5
            primary_stream = "command" if mode == "C" else "external"
            row.update(external_end_effector_force_world_n=applied_world[env_id],
                       net_compensation_force_world_n=residual_world[env_id],
                       force_phase=row[f"{primary_stream}_force_phase"],
                       force_hold_tail=row[f"{primary_stream}_force_hold_tail"],
                       direction_index=None if mode == "R" else int(self.evaluation_direction_indices[env_id]),
                       torque_saturation_fraction=saturated[env_id])
        if self.video_recorder is not None and int(self.evaluation_episode_indices[0]) == 0:
            if int(self.episode_length_buf[0]) % 2 == 0 or terminated[0] or time_outs[0]:
                self.video_recorder.capture()
        return rows

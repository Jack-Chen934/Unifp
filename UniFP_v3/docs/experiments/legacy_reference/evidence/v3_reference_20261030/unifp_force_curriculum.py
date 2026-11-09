"""Gripper force curriculum from the released UniFP B2+Z1 training task.

The released step() calls _push_gripper after global_steps > 8000 * 24
inside each of four physics substeps, while _push_robot_base is commented out.
Two independent gripper streams use the same interval/duration distributions:
command in base-yaw coordinates, external disturbance in world coordinates.
"""

from __future__ import annotations

import torch


OFFICIAL_START_POLICY_STEP = 8000 * 24 + 1
POLICY_DT = 0.02


class _ForcePulse:
    def __init__(self, num_envs, device, *, interval_steps=None, max_force_n=60.0):
        self.num_envs = num_envs
        self.device = device
        self.interval_override = interval_steps
        self.max_force_n = float(max_force_n)
        self.interval = self._sample_interval(num_envs)
        self.target = torch.zeros(num_envs, 3, device=device)
        self.value = torch.zeros_like(self.target)
        self.duration = torch.ones(num_envs, device=device)
        self.start_step = torch.zeros(num_envs, dtype=torch.int64, device=device)
        self.active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.selected_count = 0
        self.peak_force_n = 0.0

    def _sample_interval(self, size):
        if self.interval_override is not None:
            return torch.full((size,), self.interval_override, device=self.device, dtype=torch.int64)
        # torch.randint upper bound is exclusive; source samples [175, 450).
        return torch.randint(175, 450, (size,), device=self.device)

    def reset(self, env_ids):
        self.interval[env_ids] = self._sample_interval(len(env_ids))
        self.target[env_ids] = 0.0
        self.value[env_ids] = 0.0
        self.duration[env_ids] = 1.0
        self.start_step[env_ids] = 0
        self.active[env_ids] = False

    def update(self, episode_steps):
        self.value.zero_()
        # The original uses episode_length_buf modulo the current sampled
        # interval. This also permits a pulse at episode step zero.
        begin = (episode_steps % self.interval == 0)
        begin_ids = begin.nonzero(as_tuple=False).flatten()
        if len(begin_ids):
            self.active[begin_ids] = torch.rand(len(begin_ids), device=self.device) < 0.8
            self.start_step[begin_ids] = episode_steps[begin_ids]
            self.target[begin_ids] = (torch.rand(len(begin_ids), 3, device=self.device) * 2.0 - 1.0) * self.max_force_n
            # Source duration is sampled from [1,3] s then clipped to at most
            # half the remaining interval after its 1 s settling period.
            sampled = 50.0 + torch.rand(len(begin_ids), device=self.device) * 100.0
            maximum = (self.interval[begin_ids].float() - 50.0) / 2.0
            self.duration[begin_ids] = torch.minimum(sampled, maximum).clamp(min=1.0)
            self.selected_count += int(self.active[begin_ids].sum().item())
        # Rise for duration, hold 1 s, descend for duration. At the endpoint
        # the official source clears the selection and resamples interval.
        elapsed = (episode_steps - self.start_step).float()
        rise = (elapsed / self.duration).clamp(0.0, 1.0)
        fall = ((elapsed - self.duration - 50.0) / self.duration).clamp(0.0, 1.0)
        factor = (rise - fall).clamp(0.0, 1.0)
        self.value[:] = self.target * factor[:, None] * self.active[:, None]
        finished = self.active & (elapsed >= 2.0 * self.duration + 50.0)
        finished_ids = finished.nonzero(as_tuple=False).flatten()
        if len(finished_ids):
            self.active[finished_ids] = False
            self.target[finished_ids] = 0.0
            self.value[finished_ids] = 0.0
            self.interval[finished_ids] = self._sample_interval(len(finished_ids))
        self.peak_force_n = max(self.peak_force_n, float(self.value.abs().max().item()))
        return self.value


class UniFPForceCurriculum:
    def __init__(self, num_envs, device, *, start_step=OFFICIAL_START_POLICY_STEP,
                 interval_steps=None, max_force_n=60.0):
        if max_force_n <= 0:
            raise ValueError("max_force_n must be positive")
        self.start_step = start_step
        self.command = _ForcePulse(num_envs, device, interval_steps=interval_steps, max_force_n=max_force_n)
        self.external = _ForcePulse(num_envs, device, interval_steps=interval_steps, max_force_n=max_force_n)
        self.policy_dt = POLICY_DT
        self.enabled_steps = 0
        self.enabled_physics_updates = 0

    def reset(self, env_ids):
        self.command.reset(env_ids)
        self.external.reset(env_ids)

    def update(self, global_policy_step, episode_steps, *, substep_index: int = 0):
        if global_policy_step < self.start_step:
            self.command.value.zero_()
            self.external.value.zero_()
            return self.command.value, self.external.value
        if substep_index == 0:
            self.enabled_steps += 1
        self.enabled_physics_updates += 1
        return self.command.update(episode_steps), self.external.update(episode_steps)

    def summary(self):
        return {
            "official_start_policy_step": OFFICIAL_START_POLICY_STEP,
            "configured_start_policy_step": self.start_step,
            "configured_max_force_per_axis_n": self.command.max_force_n,
            "interval_override_steps": self.command.interval_override,
            "enabled_policy_steps": self.enabled_steps,
            "enabled_physics_updates": self.enabled_physics_updates,
            "command_pulses_selected": self.command.selected_count,
            "external_pulses_selected": self.external.selected_count,
            "command_peak_abs_n": self.command.peak_force_n,
            "external_peak_abs_n": self.external.peak_force_n,
            "base_force_curriculum_enabled": False,
        }

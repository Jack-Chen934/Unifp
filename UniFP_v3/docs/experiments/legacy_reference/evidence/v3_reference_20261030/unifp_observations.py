"""UniFP B2+Z1 actor input contract, independent of the Isaac Gym runtime.

Mirrors the released compute_observations() actor branch: 15 scaled commands,
73 values per frame, and a zero-initialized 32-frame FIFO. This is an input
contract for porting the policy, not a trained policy or full task generator.
"""

from __future__ import annotations

import math

import torch

from isaaclab.utils.math import euler_xyz_from_quat

from unifp_stage1_control import ACTION_JOINT_NAMES

COMMAND_NAMES = (
    "lin_vel_x", "lin_vel_y", "ang_vel_yaw",
    "ee_radius", "ee_pitch", "ee_yaw",
    "ee_roll", "ee_orientation_pitch", "ee_orientation_yaw",
    "ee_force_x", "ee_force_y", "ee_force_z",
    "base_force_x", "base_force_y", "base_force_z",
)
COMMAND_SCALES = (2.0, 2.0, 0.25, 0.5, 1.0, 1.3, 0.5, 0.5, 0.5, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01)
ACTOR_FIELDS = (
    ("roll_pitch", 2), ("base_angular_velocity", 3),
    ("joint_position_error", 17), ("joint_velocity", 17),
    ("current_action", 17), ("gait_sin_cos", 2), ("commands", 15),
)
ACTOR_SLICES = {}
_offset = 0
for _name, _width in ACTOR_FIELDS:
    ACTOR_SLICES[_name] = slice(_offset, _offset + _width)
    _offset += _width
assert _offset == 73 and len(COMMAND_NAMES) == len(COMMAND_SCALES) == 15


class UniFPActorInputs:
    """Build ordered actor observations for the already-ported PD simulation."""

    def __init__(self, robot, controller, num_envs: int):
        self.num_envs = num_envs
        self.device = robot.data.joint_pos.device
        self.joint_ids = controller.action_indices
        expected = getattr(controller, "action_names", ACTION_JOINT_NAMES)
        assert tuple(robot.joint_names[i] for i in self.joint_ids.tolist()) == expected
        self.num_actions = len(self.joint_ids)
        self.position_ids = getattr(controller, "position_indices", self.joint_ids)
        self.with_gait = getattr(controller, "with_gait", True)
        self.frame_size = 5 + len(self.position_ids) + 2 * self.num_actions + 15 + 2 * self.with_gait
        self.default_joint_pos = controller.default_positions[self.joint_ids]
        self.commands = torch.zeros(num_envs, 15, device=self.device)
        self.command_scales = torch.tensor(COMMAND_SCALES, device=self.device)
        self.gait_phase = torch.zeros(num_envs, device=self.device)
        self.history = torch.zeros(num_envs, 32, self.frame_size, device=self.device)
        self.noise_scale = torch.zeros(self.frame_size, device=self.device)
        # Released code uses randn_like(), despite a uniform-noise comment.
        self.noise_scale[0:2] = 0.05
        self.noise_scale[2:5] = 0.2 * 0.25
        end_positions = 5 + len(self.position_ids)
        self.noise_scale[5:end_positions] = 0.01
        self.noise_scale[end_positions:end_positions + self.num_actions] = 1.5 * 0.05

    def reset(self, env_ids):
        self.history[env_ids] = 0.0
        self.gait_phase[env_ids] = 0.0
        self.commands[env_ids] = 0.0

    def update_gait(self):
        if not self.with_gait:
            return
        # Official cycle_time=0.64 s; standing commands pin phase to zero.
        self.gait_phase = torch.remainder(self.gait_phase + 0.02 / 0.64, 1.0)
        walking = (self.commands[:, 0].abs() > 0.1) | (self.commands[:, 1].abs() > 0.1) | (self.commands[:, 2].abs() > 0.2)
        self.gait_phase[~walking] = 0.0

    def observe(self, robot, actions: torch.Tensor, add_noise: bool = False):
        if actions.shape != (self.num_envs, self.num_actions):
            raise ValueError(f"Unexpected action shape {tuple(actions.shape)}")
        roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)
        phase = 2.0 * math.pi * self.gait_phase
        components = [
            torch.stack((roll, pitch), dim=-1),
            robot.data.root_ang_vel_b * 0.25,
            robot.data.joint_pos[:, self.position_ids] - robot.data.default_joint_pos[:, self.position_ids],
            robot.data.joint_vel[:, self.joint_ids] * 0.05,
            actions,
        ]
        if self.with_gait:
            components.append(torch.stack((torch.sin(phase), torch.cos(phase)), dim=-1))
        components.append(self.commands * self.command_scales)
        frame = torch.cat(components, dim=-1)
        assert frame.shape == (self.num_envs, self.frame_size)
        if add_noise:
            frame = frame + torch.randn_like(frame) * self.noise_scale
        self.history[:, :-1] = self.history[:, 1:].clone()
        self.history[:, -1] = frame
        flat = self.history.reshape(self.num_envs, -1)
        assert torch.isfinite(flat).all().item()
        return frame, flat

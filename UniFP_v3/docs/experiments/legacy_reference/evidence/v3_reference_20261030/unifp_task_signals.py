"""UniFP end-effector trajectory and privileged training signals.

The order and scales mirror the released B2+Z1 compute_observations(). The
22 mass fields are *randomization deltas*, not raw link masses. When the
randomization flag is enabled, they are read back from applied physical
properties; otherwise their true value is zero. Motor truth follows the
actual per-environment action scaling used by the PD controller.
"""

from __future__ import annotations

import math

import torch

from isaaclab.utils.math import quat_apply_inverse, quat_apply_yaw, quat_from_euler_xyz, yaw_quat


PRED_FIELDS = (("base_velocity_b", 3), ("ee_position_sphere", 3),
               ("gripper_disturbance_yaw", 3), ("base_disturbance_yaw", 3))
CRITIC_FIELDS = (
    ("supervision", 12), ("leg_reference_error", 12), ("mass_randomization_deltas", 22),
    ("friction_coefficient", 1), ("motor_strength_delta", 17), ("stance", 4),
    ("foot_contact", 4), ("projected_gravity", 3), ("base_angular_velocity", 3),
    ("joint_position_error", 17), ("joint_velocity", 17), ("action", 17),
    ("gait_sin_cos", 2), ("commands", 15), ("force_offset_ee_goal_sphere", 3),
)


def field_slices(fields):
    result, start = {}, 0
    for name, width in fields:
        result[name] = slice(start, start + width)
        start += width
    return result


PRED_SLICES = field_slices(PRED_FIELDS)
CRITIC_SLICES = field_slices(CRITIC_FIELDS)
assert PRED_SLICES["base_disturbance_yaw"].stop == 12
assert CRITIC_SLICES["force_offset_ee_goal_sphere"].stop == 149


def sphere_to_cart(sphere: torch.Tensor) -> torch.Tensor:
    radius, pitch, yaw = sphere.unbind(dim=-1)
    return torch.stack((radius * pitch.cos() * yaw.cos(),
                        radius * pitch.cos() * yaw.sin(), radius * pitch.sin()), dim=-1)


def cart_to_sphere(cart: torch.Tensor) -> torch.Tensor:
    x, y, z = cart.unbind(dim=-1)
    return torch.stack((torch.linalg.vector_norm(cart, dim=-1),
                        torch.atan2(z, torch.hypot(x, y)), torch.atan2(y, x)), dim=-1)


class UniFPEEGoal:
    """Released spherical interpolation, hold, rejection, and yaw-frame center."""

    def __init__(self, num_envs: int, device, *, random_goals: bool = True,
                 offset=(0.2, 0.0, 0.8), initial=(0.66, math.pi / 4, 0.0),
                 first_goal=(0.66, 0.0, 0.0),
                 goal_lower=(0.35, -2 * math.pi / 5, -3 * math.pi / 5),
                 goal_upper=(0.95, 2 * math.pi / 5, 3 * math.pi / 5)):
        self.num_envs = num_envs
        self.device = device
        self.random_goals = random_goals
        self.start = torch.zeros(num_envs, 3, device=device)
        self.goal = torch.zeros_like(self.start)
        self.current = torch.zeros_like(self.start)
        self.orientation_delta = torch.zeros_like(self.start)
        self.timer = torch.zeros(num_envs, device=device)
        self.traj_steps = torch.zeros_like(self.timer)
        self.total_steps = torch.zeros_like(self.timer)
        self.goal_world = torch.zeros_like(self.start)
        self.orientation_quat_w = torch.zeros(num_envs, 4, device=device)
        self.resample_count = 0
        self.rejected_count = 0
        self._offset = torch.tensor(offset, device=device)
        self._initial = torch.tensor(initial, device=device)
        self._first_goal = torch.tensor(first_goal, device=device)
        self._goal_lower = torch.tensor(goal_lower, device=device)
        self._goal_upper = torch.tensor(goal_upper, device=device)
        self._lower = torch.tensor((-0.7, -0.2, -0.8), device=device)
        self._upper = torch.tensor((0.25, 0.2, -0.15), device=device)
        self._collision_t = torch.linspace(0.0, 1.0, 10, device=device)
        self.schedule = None
        self.schedule_index = 0

    def install_schedule(self, schedule):
        """Evaluation-only command tape; robot resets preserve its global index."""
        self.schedule = torch.tensor(
            [row['sphere'] + row['orientation_delta'] for row in schedule['samples']],
            device=self.device,
        )
        self.schedule_index = 0

    def center_world(self, robot) -> torch.Tensor:
        # Official center uses world root XY, zero world Z, then yaw-rotated
        # (0.2, 0, 0.8): vertical base motion never shifts the target.
        root_xy_on_ground = torch.cat((robot.data.root_pos_w[:, :2],
                                       torch.zeros(self.num_envs, 1, device=self.device)), dim=-1)
        return root_xy_on_ground + quat_apply_yaw(robot.data.root_quat_w, self._offset.expand(self.num_envs, -1))

    def reset(self, env_ids):
        self.start[env_ids] = self._initial
        self.goal[env_ids] = self._first_goal
        self.current[env_ids] = self._initial
        self.orientation_delta[env_ids] = 0.0
        self.timer[env_ids] = 0.0
        self.traj_steps[env_ids] = (1.0 + 2.0 * torch.rand(len(env_ids), device=self.device)) / 0.02
        self.total_steps[env_ids] = self.traj_steps[env_ids] + (
            0.5 + 1.5 * torch.rand(len(env_ids), device=self.device)) / 0.02

    def _reject(self, env_ids):
        interp = torch.lerp(self.start[env_ids, None], self.goal[env_ids, None],
                            self._collision_t[None, :, None])
        points = sphere_to_cart(interp)
        in_box = ((points > self._lower) & (points < self._upper)).all(dim=-1).any(dim=-1)
        below = (points[..., 2] < -0.7).any(dim=-1)
        return in_box | below

    def _resample(self, env_ids):
        if not self.random_goals or len(env_ids) == 0:
            return
        self.start[env_ids] = self.goal[env_ids].clone()
        self.orientation_delta[env_ids] = torch.rand(len(env_ids), 3, device=self.device) - 0.5
        remaining = env_ids
        for _ in range(10):
            samples = torch.rand(len(remaining), 3, device=self.device)
            self.goal[remaining] = self._goal_lower + (self._goal_upper - self._goal_lower) * samples
            rejected = self._reject(remaining)
            self.rejected_count += int(rejected.sum().item())
            remaining = remaining[rejected]
            if len(remaining) == 0:
                break
        self.timer[env_ids] = 0.0
        self.resample_count += len(env_ids)

    def update(self, robot, commands: torch.Tensor, env_ids=None, *, advance: bool = True):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return
        if self.schedule is not None:
            sample = self.schedule[self.schedule_index]
            self.start[env_ids] = sample[:3]
            self.goal[env_ids] = sample[:3]
            self.orientation_delta[env_ids] = sample[3:]
            self.timer[env_ids] = 0
            self.traj_steps[env_ids] = 1
            advance = False
        progress = (self.timer[env_ids] / self.traj_steps[env_ids]).clamp(0.0, 1.0)
        self.current[env_ids] = torch.lerp(self.start[env_ids], self.goal[env_ids], progress[:, None])
        commands[env_ids, 3:6] = self.current[env_ids]
        self.goal_world[env_ids] = self.center_world(robot)[env_ids] + quat_apply_yaw(
            robot.data.root_quat_w[env_ids], sphere_to_cart(self.current[env_ids]))
        # Official roll/pitch/yaw target used by the end-effector orientation
        # reward; the released policy's command slots 6:9 remain zero.
        goal_vector_w = self.goal_world[env_ids] - self.center_world(robot)[env_ids]
        default_yaw = torch.atan2(goal_vector_w[:, 1], goal_vector_w[:, 0])
        self.orientation_quat_w[env_ids] = quat_from_euler_xyz(
            self.orientation_delta[env_ids, 0] + math.pi / 2,
            -self.current[env_ids, 1] + 0.38 + self.orientation_delta[env_ids, 1],
            self.orientation_delta[env_ids, 2] + default_yaw,
        )
        if advance:
            self.timer[env_ids] += 1.0
            overdue = self.timer[env_ids] > self.total_steps[env_ids]
            self._resample(env_ids[overdue])


class UniFPPrivilegedInputs:
    """12 supervision targets plus 149 x 3 privileged critic FIFO."""

    def __init__(self, robot, controller, scene, *, friction: torch.Tensor,
                 mass_deltas: torch.Tensor | None = None, motor_strength: torch.Tensor | None = None):
        self.num_envs = scene.num_envs
        self.device = scene.device
        self.leg_ids = controller.action_indices[:12]
        self.action_ids = controller.action_indices
        self.default = controller.default_positions
        self.ee_id = robot.body_names.index("ee_gripper_link")
        self.foot_ids = [scene["feet"].body_names.index(n) for n in
                         ("FL_foot", "FR_foot", "RL_foot", "RR_foot")]
        if friction.shape != (self.num_envs, 1):
            raise ValueError(f"Expected measured friction ({self.num_envs}, 1), got {friction.shape}")
        self.friction = friction
        self.mass_deltas = (torch.zeros(self.num_envs, 22, device=self.device)
                            if mass_deltas is None else mass_deltas)
        self.motor_strength = (torch.ones(self.num_envs, 17, device=self.device)
                               if motor_strength is None else motor_strength)
        if self.mass_deltas.shape != (self.num_envs, 22) or self.motor_strength.shape != (self.num_envs, 17):
            raise ValueError("Privileged randomization truth has unexpected dimensions")
        self.history = torch.zeros(self.num_envs, 3, 149, device=self.device)

    def reset(self, env_ids):
        self.history[env_ids] = 0.0

    def observe(self, robot, controller, feet, gait_phase, actions, commands, goal,
                gripper_force_w, base_force_w, *, push_history: bool = True):
        quat = robot.data.root_quat_w
        yaw = yaw_quat(quat)
        center = goal.center_world(robot)
        ee_cart_yaw = quat_apply_inverse(yaw, robot.data.body_pos_w[:, self.ee_id] - center)
        gripper_force_yaw = quat_apply_inverse(yaw, gripper_force_w)
        base_force_yaw = quat_apply_inverse(yaw, base_force_w)
        target = torch.cat((robot.data.root_lin_vel_b * 2.0,
                            cart_to_sphere(ee_cart_yaw) * torch.tensor((0.5, 1.0, 1.3), device=self.device),
                            gripper_force_yaw * 0.01, base_force_yaw * 0.01), dim=-1)
        assert target.shape == (self.num_envs, 12)

        sin_pos = torch.sin(2.0 * math.pi * gait_phase)
        left, right = sin_pos + 0.5, sin_pos - 0.5
        stance = torch.stack((left >= 0, right < 0, right < 0, left >= 0), dim=-1).float()
        left = torch.where(left > 0, torch.zeros_like(left), left)
        right = torch.where(right < 0, torch.zeros_like(right), right)
        scale = 0.17 / (1.0 - 0.5)
        reference = self.default[self.leg_ids].expand(self.num_envs, -1).clone()
        for index in (1, 10):
            reference[:, index] -= left * scale
        for index in (2, 11):
            reference[:, index] += left * scale * 2.0
        for index in (4, 7):
            reference[:, index] += right * scale
        for index in (5, 8):
            reference[:, index] -= right * scale * 2.0
        leg_error = robot.data.joint_pos[:, self.leg_ids] - reference

        mass_deltas = self.mass_deltas
        motor_delta = self.motor_strength - 1.0
        contact = (feet.data.net_forces_w[:, self.foot_ids, 2] > 5.0).float()
        offset_world = goal.goal_world + (gripper_force_w + quat_apply_yaw(quat, commands[:, 9:12])) / 200.0
        offset_sphere = cart_to_sphere(quat_apply_inverse(yaw, offset_world - center))
        offset_sphere *= torch.tensor((0.5, 1.0, 1.3), device=self.device)
        frame = torch.cat((target, leg_error, mass_deltas, self.friction, motor_delta,
                           stance, contact, robot.data.projected_gravity_b,
                           robot.data.root_ang_vel_b * 0.25,
                           robot.data.joint_pos[:, self.action_ids] - self.default[self.action_ids],
                           robot.data.joint_vel[:, self.action_ids] * 0.05, actions,
                           torch.stack((sin_pos, torch.cos(2.0 * math.pi * gait_phase)), dim=-1),
                           commands * torch.tensor((2, 2, 0.25, 0.5, 1, 1.3, 0.5, 0.5, 0.5,
                                                    0.01, 0.01, 0.01, 0.01, 0.01, 0.01), device=self.device),
                           offset_sphere), dim=-1)
        assert frame.shape == (self.num_envs, 149)
        if push_history:
            self.history[:, :-1] = self.history[:, 1:].clone()
            self.history[:, -1] = frame
        flat = self.history.reshape(self.num_envs, 447)
        assert torch.isfinite(flat).all().item()
        return target, frame, flat

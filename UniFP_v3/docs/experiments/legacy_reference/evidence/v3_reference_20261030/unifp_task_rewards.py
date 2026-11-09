"""Released UniFP B2+Z1 active rewards and episode termination.

Only nonzero scales from b2z1_pos_force_config.py are included. Each scale is
multiplied by the 0.02 s policy timestep, as _prepare_reward_function does.
Inputs remain physical quantities: commands are unscaled and forces are the
actual world-frame applied external forces, distinct from command forces.
"""

from __future__ import annotations

import math

import torch

from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse, quat_apply_yaw, yaw_quat

from unifp_stage1_control import ACTION_JOINT_NAMES


REWARD_SCALES = {
    "feet_contact_number": 2.0,
    "tracking_lin_vel_force_world": 2.0,
    "tracking_ang_vel": 1.0,
    "torques": -5e-6,
    "stand_still": 0.5,
    "ref_dof_leg": 1.0,
    "alive": 1.5,
    "lin_vel_z": -1.5,
    "feet_air_time": 1.0,
    "feet_height": 1.0,
    "ang_vel_xy": -0.02,
    "dof_acc": -2.5e-7,
    "dof_vel": -8e-4,
    "dof_acc_arm": -4.5e-7,
    "dof_vel_arm": -2e-4,
    "collision": -5.0,
    "action_rate": -0.02,
    "action_rate_arm": -0.045,
    "dof_pos_limits": -10.0,
    "torque_limits": -0.005,
    "hip_pos": -0.5,
    "feet_drag": -0.0008,
    "feet_contact_forces": -0.001,
    "base_height": -2.0,
    "feet_pos_xy": -0.5,
    "feet_height_high": -15.0,
    "tracking_ee_force_world": 2.0,
}
POLICY_DT = 0.02
MAX_EPISODE_STEPS = math.ceil(20.0 / POLICY_DT)


class UniFPTaskRewards:
    def __init__(self, robot, controller, scene):
        self.device = scene.device
        self.num_envs = scene.num_envs
        self.origins = scene.env_origins
        self.joint_ids = controller.action_indices
        self.default = controller.default_positions
        self.effort_limits = controller.effort_limits
        self.soft_pos_limits = robot.data.joint_pos_limits[0, self.joint_ids].clone()
        centers = self.soft_pos_limits.mean(dim=-1)
        halves = (self.soft_pos_limits[:, 1] - self.soft_pos_limits[:, 0]) * 0.4
        self.soft_pos_limits[:, 0] = centers - halves
        self.soft_pos_limits[:, 1] = centers + halves
        self.hip_ids = torch.tensor([robot.joint_names.index(f"{leg}_hip_joint")
                                     for leg in ("FR", "FL", "RR", "RL")], device=self.device)
        self.foot_names = tuple(f"{leg}_foot" for leg in ("FL", "FR", "RL", "RR"))
        self.foot_ids = [robot.body_names.index(name) for name in self.foot_names]
        self.thigh_ids = [robot.body_names.index(f"{leg}_thigh") for leg in ("FL", "FR", "RL", "RR")]
        self.penalized_names = tuple(name for name in robot.body_names
                                     if any(token in name for token in ("thigh", "calf", "base_link")))
        self.prev_vel = torch.zeros(self.num_envs, 17, device=self.device)
        self.prev_actions = torch.zeros(self.num_envs, 17, device=self.device)
        self.last_contacts = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
        self.feet_air_time = torch.zeros(self.num_envs, 4, device=self.device)
        self.episode_steps = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self.term_sums = {name: 0.0 for name in REWARD_SCALES}
        self.total_sum = 0.0
        self.step_count = 0
        # The released B2+Z1 configuration has an empty
        # ``terminate_after_contacts_on`` list.  Keep contact termination as
        # an explicit counter so a future asset/config change cannot silently
        # add a reset condition.
        self.termination_counts = {
            "termination_contact": 0,
            "roll": 0,
            "pitch": 0,
            "timeout": 0,
            "safety_height": 0,
        }

    def reset(self, env_ids):
        self.prev_vel[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.last_contacts[env_ids] = False
        self.feet_air_time[env_ids] = 0.0
        self.episode_steps[env_ids] = 0

    def termination(self, robot, *, safety_height: bool = False, contact_forces=None):
        """Return reset and timeout masks using the released task rules.

        UniFP's released ``terminate_after_contacts_on`` is empty, so contact
        forces do not terminate an episode.  ``safety_height`` is retained as
        an opt-in development guard for unstable asset conversions; it is
        disabled by the vector environment by default to preserve the
        original reset distribution.
        """
        roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)
        bad_roll = roll.abs() > 0.8
        bad_pitch = pitch.abs() > 1.0
        timeout = self.episode_steps > MAX_EPISODE_STEPS
        if contact_forces is None:
            contact_termination = torch.zeros_like(timeout)
        else:
            # This branch is intentionally opt-in: the released config has no
            # termination contact names.  It is useful for experiments that
            # explicitly provide a non-empty termination body set.
            contact_termination = torch.linalg.vector_norm(contact_forces, dim=-1).amax(dim=-1) > 1.0
        if safety_height:
            safety_height_mask = (robot.data.root_pos_w[:, 2] - self.origins[:, 2]) < 0.25
        else:
            safety_height_mask = torch.zeros_like(timeout)
        self.last_termination_masks = {
            "termination_contact": contact_termination, "roll": bad_roll,
            "pitch": bad_pitch, "timeout": timeout, "safety_height": safety_height_mask,
        }
        for key, value in self.last_termination_masks.items():
            self.termination_counts[key] += int(value.sum().item())
        return contact_termination | bad_roll | bad_pitch | timeout | safety_height_mask, timeout

    def compute(self, robot, feet, contacts_all, controller, actions, torques,
                commands, goal, gripper_force_w, base_force_w, critic_frame, critic_slices):
        if tuple(robot.joint_names[i] for i in self.joint_ids.tolist()) != ACTION_JOINT_NAMES:
            raise RuntimeError("Reward joint order differs from released B2+Z1 policy order")
        N = self.num_envs
        q = robot.data.joint_pos[:, self.joint_ids]
        dq = robot.data.joint_vel[:, self.joint_ids]
        tau = torques[:, self.joint_ids]
        base_v = robot.data.root_lin_vel_b
        base_w = robot.data.root_ang_vel_b
        feet_forces = feet.data.net_forces_w[:, [feet.body_names.index(n) for n in self.foot_names]]
        feet_pos = robot.data.body_pos_w[:, self.foot_ids]
        thighs_pos = robot.data.body_pos_w[:, self.thigh_ids]
        foot_vel = robot.data.body_lin_vel_w[:, self.foot_ids]
        contact = feet_forces[..., 2] > 5.0
        stance = critic_frame[:, critic_slices["stance"]].bool()
        walking = ((commands[:, 0].abs() > 0.1) | (commands[:, 1].abs() > 0.1)
                   | (commands[:, 2].abs() > 0.2))
        terms = {}
        terms["feet_contact_number"] = torch.where(contact == stance, 1.0, -0.3).mean(dim=-1)
        base_force_yaw = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), base_force_w)
        v_target = commands[:, :2] + (base_force_yaw[:, :2] + commands[:, 12:14]) / 200.0
        moving = ((v_target[:, 0].abs() > 0.1) | (v_target[:, 1].abs() > 0.1)
                  | (commands[:, 2].abs() > 0.2))
        v_target = v_target * moving[:, None]
        terms["tracking_lin_vel_force_world"] = torch.exp(-((v_target - base_v[:, :2]) ** 2).sum(dim=-1) / 0.25)
        terms["tracking_ang_vel"] = torch.exp(-((commands[:, 2] - base_w[:, 2]) ** 2) / 0.25)
        terms["torques"] = tau[:, :12].square().sum(dim=-1)
        terms["stand_still"] = torch.exp(-((q[:, :12] - self.default[self.joint_ids[:12]]).abs().sum(dim=-1)) * 0.05) * (~walking)
        leg_error = critic_frame[:, critic_slices["leg_reference_error"]]
        terms["ref_dof_leg"] = torch.exp(-leg_error.abs().sum(dim=-1) * 0.1)
        terms["alive"] = torch.ones(N, device=self.device)
        terms["lin_vel_z"] = base_v[:, 2].square()
        contact_filt = (feet_forces[..., 2] > 1.0) | self.last_contacts
        first_contact = (self.feet_air_time > 0.0) & contact_filt
        self.feet_air_time += POLICY_DT
        terms["feet_air_time"] = ((self.feet_air_time - 0.5) * first_contact).sum(dim=-1) * walking
        self.feet_air_time *= ~contact_filt
        self.last_contacts = feet_forces[..., 2] > 1.0
        terms["feet_height"] = torch.clamp(feet_pos[:, :2, 2].max(dim=-1).values - 0.10, max=0.0) * walking
        terms["ang_vel_xy"] = base_w[:, :2].square().sum(dim=-1)
        accel = (self.prev_vel - dq) / POLICY_DT
        terms["dof_acc"] = accel[:, :12].square().sum(dim=-1)
        terms["dof_vel"] = dq[:, :12].square().sum(dim=-1)
        terms["dof_acc_arm"] = accel[:, 12:17].square().sum(dim=-1)
        terms["dof_vel_arm"] = dq[:, 12:17].square().sum(dim=-1)
        penalized_ids = [contacts_all.body_names.index(name) for name in self.penalized_names]
        penalized_forces = contacts_all.data.net_forces_w[:, penalized_ids]
        terms["collision"] = (torch.linalg.vector_norm(penalized_forces, dim=-1) > 0.1).float().sum(dim=-1)
        action_delta = (self.prev_actions - actions).square()
        terms["action_rate"] = action_delta[:, :12].sum(dim=-1)
        terms["action_rate_arm"] = action_delta[:, 12:17].sum(dim=-1)
        low, high = self.soft_pos_limits.unbind(dim=-1)
        terms["dof_pos_limits"] = ((low - q).clamp(min=0.0) + (q - high).clamp(min=0.0)).sum(dim=-1)
        # The released _reward_torque_limits sums all 19 DOFs, including the
        # two held wrist/gripper joints. The other torque reward uses 12 legs.
        terms["torque_limits"] = (torques.abs() - 0.9 * self.effort_limits).clamp(min=0.0).sum(dim=-1)
        terms["hip_pos"] = (robot.data.joint_pos[:, self.hip_ids] - self.default[self.hip_ids]).square().sum(dim=-1)
        terms["feet_drag"] = (torch.linalg.vector_norm(feet_forces, dim=-1) * foot_vel.abs().sum(dim=-1)).sum(dim=-1)
        terms["feet_contact_forces"] = (torch.linalg.vector_norm(feet_forces, dim=-1) - 200.0).clamp(min=0.0).sum(dim=-1)
        base_height = robot.data.root_pos_w[:, 2] - self.origins[:, 2]
        terms["base_height"] = (base_height - 0.50).square()
        terms["feet_pos_xy"] = torch.linalg.vector_norm(feet_pos[:, :, :2] - thighs_pos[:, :, :2], dim=-1).mean(dim=-1)
        terms["feet_height_high"] = (feet_pos[:, :, 2].max(dim=-1).values - 0.20).clamp(min=0.0) * walking
        ee_id = robot.body_names.index("ee_gripper_link")
        ee_target = goal.goal_world + (gripper_force_w + quat_apply_yaw(robot.data.root_quat_w, commands[:, 9:12])) / 200.0
        ee_error_l1 = (robot.data.body_pos_w[:, ee_id] - ee_target).abs().sum(dim=-1)
        terms["tracking_ee_force_world"] = torch.exp(-2.0 * ee_error_l1)
        if set(terms) != set(REWARD_SCALES):
            raise RuntimeError(f"Reward coverage mismatch: {set(terms) ^ set(REWARD_SCALES)}")
        weighted = {name: terms[name] * scale * POLICY_DT for name, scale in REWARD_SCALES.items()}
        total = torch.stack(tuple(weighted.values())).sum(dim=0)
        if not torch.isfinite(total).all():
            raise RuntimeError("Nonfinite UniFP reward")
        self.prev_vel[:] = dq
        self.prev_actions[:] = actions
        self.step_count += N
        for name, value in weighted.items():
            self.term_sums[name] += float(value.sum().item())
        self.total_sum += float(total.sum().item())
        return total, weighted

    def summary(self):
        divisor = max(self.step_count, 1)
        return {
            "active_terms": len(self.term_sums),
            "mean_total_per_env_step": self.total_sum / divisor,
            "mean_weighted_per_env_step": {name: value / divisor for name, value in self.term_sums.items()},
            "termination_counts": self.termination_counts,
            "official_timeout_policy_steps": MAX_EPISODE_STEPS,
        }

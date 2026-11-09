"""Ranger+A1 morphology adapter for the unchanged released UniFP learner.

Reuses the Isaac Lab rollout/reset loop, history inputs, goal interpolation,
randomization, force curriculum and termination. Import after AppLauncher.
"""

from pathlib import Path
import os

import torch
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, quat_apply_yaw, yaw_quat

from ranger_force_experiment_config import ARM_NAMES, ARM_EFFORT_LIMITS
from unifp_domain_randomization import UniFPDomainRandomization
from unifp_force_curriculum import UniFPForceCurriculum
from unifp_observations import COMMAND_SCALES, UniFPActorInputs
from unifp_task_rewards import POLICY_DT, REWARD_SCALES, UniFPTaskRewards
from unifp_task_signals import UniFPEEGoal, cart_to_sphere, field_slices
from unifp_vec_env import UniFPSceneCfg, UniFPVecEnv


ASSET_VERSION = os.environ.get("RANGER_ASSET_VERSION", "v3_original")
ASSET_FILES = {"v2": "a1arm_rgm3_approx_sim_v2.usd", "v3": "v3/a1arm_rgm3_approx_sim_v3.usd",
               "v3_original": "v3_original/a1arm_rgm3_original_movable.usd"}
if ASSET_VERSION not in ASSET_FILES:
    raise ValueError(f"RANGER_ASSET_VERSION must be one of {tuple(ASSET_FILES)}")
USD_PATH = Path(__file__).resolve().parent / "robot_description/a1arm_description/assets" / ASSET_FILES[ASSET_VERSION]
ROOT_HEIGHT = .26 if ASSET_VERSION == "v2" else .32
CORNERS = ("fl", "fr", "bl", "br")
ACTION_NAMES = (tuple(f"{c}_steering_joint" for c in CORNERS)
                + tuple(f"{c}_wheel_drive_joint" for c in CORNERS) + tuple(ARM_NAMES))
ARM_HOME = (0., 1., -1.5, 0., 0., 0.)
CRITIC_FIELDS = (
    ("supervision", 12), ("mass_randomization_deltas", 5),
    ("friction_coefficient", 1), ("motor_strength_delta", 14),
    ("wheel_contact", 4), ("projected_gravity", 3), ("base_angular_velocity", 3),
    ("joint_position_error", 10), ("joint_velocity", 14), ("action", 14),
    ("commands", 15), ("force_offset_ee_goal_sphere", 3),
)
CRITIC_SLICES = field_slices(CRITIC_FIELDS)
CRITIC_SIZE = sum(width for _, width in CRITIC_FIELDS)


@configclass
class RangerSceneCfg(UniFPSceneCfg):
    light = AssetBaseCfg(prim_path="/World/RangerLight", spawn=sim_utils.DomeLightCfg(
        intensity=1800., color=(.85, .9, 1.)))
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=str(USD_PATH), activate_contact_sensors=True),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0., 0., ROOT_HEIGHT), joint_pos={
                **dict(zip(ARM_NAMES, ARM_HOME)), ".*_axis": 0.,
                ".*_steering_joint": 0., ".*_wheel_drive_joint": 0.,
            },
        ),
        actuators={
            "arm": ImplicitActuatorCfg(joint_names_expr=["arm_joint[1-6]"],
                                      effort_limit_sim=ARM_EFFORT_LIMITS, stiffness=80., damping=4.),
            "gripper": ImplicitActuatorCfg(joint_names_expr=["gripper.*"],
                                          effort_limit_sim=10., stiffness=20., damping=2.),
            "steering": ImplicitActuatorCfg(joint_names_expr=[".*_steering_joint"],
                                           effort_limit_sim=35., stiffness=300., damping=20.),
            "drive": ImplicitActuatorCfg(joint_names_expr=[".*_wheel_drive_joint"],
                                        effort_limit_sim=80., stiffness=0., damping=20.),
        },
    )
    feet = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/(fl|fr|bl|br)_Link", history_length=1)


class RangerJointPD:
    action_names = ACTION_NAMES
    base_action_count = 8
    with_gait = False

    def __init__(self, robot, *, implicit_pd=True):
        self.robot, self.implicit_pd = robot, implicit_pd
        device = robot.data.joint_pos.device
        self.action_indices = torch.tensor([robot.joint_names.index(n) for n in ACTION_NAMES], device=device)
        self.position_action_indices = torch.tensor([0, 1, 2, 3, 8, 9, 10, 11, 12, 13], device=device)
        self.position_indices = self.action_indices[self.position_action_indices]
        self.drive_ids = self.action_indices[4:8]
        self.default_positions = robot.data.default_joint_pos[0].clone()
        self.kp = torch.full_like(self.default_positions, 20.)
        self.kd = torch.full_like(self.default_positions, 2.)
        self.effort_limits = torch.full_like(self.default_positions, 10.)
        self.kp[self.action_indices[:4]], self.kd[self.action_indices[:4]] = 300., 20.
        self.kp[self.drive_ids], self.kd[self.drive_ids] = 0., 20.
        self.kp[self.action_indices[8:]], self.kd[self.action_indices[8:]] = 80., 4.
        self.effort_limits[self.action_indices[:4]] = 35.
        self.effort_limits[self.drive_ids] = 80.
        self.effort_limits[self.action_indices[8:]] = torch.tensor(
            [ARM_EFFORT_LIMITS[n] for n in ARM_NAMES], device=device)

    def compute(self, actions, q, dq, *, motor_strength):
        # Released UniFP randomizes action amplitude, not the feedback gains.
        actions = actions * motor_strength
        target = self.default_positions.expand_as(q).clone()
        target[:, self.position_indices] += .25 * actions[:, self.position_action_indices]
        velocity = torch.zeros_like(dq)
        # 5 rad/s: nominal 0.5 m/s with v3's 0.1 m contact radius
        # (historical v2 used 0.153 m).
        velocity[:, self.drive_ids] = 5. * actions[:, 4:8]
        if self.implicit_pd:
            # Solve the same PD law inside PhysX: explicit 5 ms feedback
            # oscillates on the low-inertia steering and A1 wrist joints.
            self.robot.set_joint_position_target(target)
            self.robot.set_joint_velocity_target(velocity)
        torque = self.kp * (target - q) + self.kd * (velocity - dq)
        return torch.clamp(torque, -self.effort_limits, self.effort_limits)


class RangerPrivilegedInputs:
    def __init__(self, robot, controller, scene, domain):
        self.num_envs, self.device = scene.num_envs, scene.device
        self.ee_id = robot.body_names.index("arm_seg6")
        self.wheel_ids = [scene["feet"].body_names.index(f"{c}_Link") for c in CORNERS]
        self.domain = domain
        self.scales = torch.tensor(COMMAND_SCALES, device=self.device)
        self.sphere_scales = self.scales[3:6]
        self.history = torch.zeros(self.num_envs, 3, CRITIC_SIZE, device=self.device)

    def reset(self, env_ids):
        self.history[env_ids] = 0.

    def observe(self, robot, controller, feet, gait_phase, actions, commands, goal,
                gripper_force_w, base_force_w, *, push_history=True):
        yaw = yaw_quat(robot.data.root_quat_w)
        center = goal.center_world(robot)
        ee = robot.data.body_pos_w[:, self.ee_id]
        target = torch.cat((robot.data.root_lin_vel_b * 2.,
                            cart_to_sphere(quat_apply_inverse(yaw, ee - center)) * self.sphere_scales,
                            quat_apply_inverse(yaw, gripper_force_w) * .01,
                            quat_apply_inverse(yaw, base_force_w) * .01), dim=-1)
        offset = goal.goal_world + (gripper_force_w + quat_apply_yaw(yaw, commands[:, 9:12])) / 200.
        frame = torch.cat((
            target, self.domain.mass_deltas[:, :5], self.domain.friction,
            self.domain.motor_strength - 1.,
            (feet.data.net_forces_w[:, self.wheel_ids, 2] > 5.).float(),
            robot.data.projected_gravity_b, robot.data.root_ang_vel_b * .25,
            robot.data.joint_pos[:, controller.position_indices] - controller.default_positions[controller.position_indices],
            robot.data.joint_vel[:, controller.action_indices] * .05, actions,
            commands * self.scales,
            cart_to_sphere(quat_apply_inverse(yaw, offset - center)) * self.sphere_scales,
        ), dim=-1)
        assert frame.shape == (self.num_envs, CRITIC_SIZE)
        if push_history:
            self.history[:, :-1] = self.history[:, 1:].clone()
            self.history[:, -1] = frame
        return target, frame, self.history.reshape(self.num_envs, -1)


class RangerRewards(UniFPTaskRewards):
    """Applicable released reward formulas/scales, with wheel/arm indexing."""

    def __init__(self, robot, controller, scene):
        self.device, self.num_envs, self.origins = scene.device, scene.num_envs, scene.env_origins
        self.joint_ids, self.default = controller.action_indices, controller.default_positions
        self.effort_limits = controller.effort_limits
        self.ee_id = robot.body_names.index("arm_seg6")
        self.position_ids = controller.position_action_indices
        self.soft_pos_limits = robot.data.joint_pos_limits[0, controller.position_indices].clone()
        center = self.soft_pos_limits.mean(dim=-1)
        half = (self.soft_pos_limits[:, 1] - self.soft_pos_limits[:, 0]) * .4
        self.soft_pos_limits[:, 0], self.soft_pos_limits[:, 1] = center - half, center + half
        # Original collision penalty covers legs/base, not the contacting arm.
        # Wheels have intentional ground contact; retain the base-body term.
        self.penalized_ids = [scene["robot_contacts"].body_names.index("base_link")]
        self.prev_vel = torch.zeros(self.num_envs, 14, device=self.device)
        self.prev_actions = torch.zeros_like(self.prev_vel)
        self.last_contacts = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
        self.feet_air_time = torch.zeros(self.num_envs, 4, device=self.device)
        self.episode_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        names = ("tracking_lin_vel_force_world", "tracking_ang_vel", "tracking_ee_force_world",
                 "torques", "alive", "lin_vel_z", "ang_vel_xy", "dof_acc", "dof_vel",
                 "dof_acc_arm", "dof_vel_arm", "collision", "action_rate", "action_rate_arm",
                 "dof_pos_limits", "torque_limits", "base_height")
        self.scales = {name: REWARD_SCALES[name] for name in names}
        self.term_sums = dict.fromkeys(names, 0.)
        self.total_sum, self.step_count = 0., 0
        self.termination_counts = dict.fromkeys(("termination_contact", "roll", "pitch", "timeout", "safety_height"), 0)

    def compute(self, robot, feet, contacts_all, controller, actions, torques,
                commands, goal, gripper_force_w, base_force_w, critic_frame, critic_slices):
        q, dq = robot.data.joint_pos[:, self.joint_ids], robot.data.joint_vel[:, self.joint_ids]
        v, w = robot.data.root_lin_vel_b, robot.data.root_ang_vel_b
        base_force = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), base_force_w)
        target = commands[:, :2] + (base_force[:, :2] + commands[:, 12:14]) / 200.
        moving = (target[:, 0].abs() > .1) | (target[:, 1].abs() > .1) | (commands[:, 2].abs() > .2)
        target = target * moving[:, None]
        accel, delta = (self.prev_vel - dq) / POLICY_DT, (self.prev_actions - actions).square()
        low, high = self.soft_pos_limits.unbind(dim=-1)
        pos = q[:, self.position_ids]
        ee_target = goal.goal_world + (gripper_force_w + quat_apply_yaw(robot.data.root_quat_w, commands[:, 9:12])) / 200.
        terms = {
            "tracking_lin_vel_force_world": torch.exp(-(target - v[:, :2]).square().sum(-1) / .25),
            "tracking_ang_vel": torch.exp(-(commands[:, 2] - w[:, 2]).square() / .25),
            "tracking_ee_force_world": torch.exp(-2. * (robot.data.body_pos_w[:, self.ee_id] - ee_target).abs().sum(-1)),
            "torques": torques[:, self.joint_ids[:8]].square().sum(-1),
            "alive": torch.ones(self.num_envs, device=self.device),
            "lin_vel_z": v[:, 2].square(), "ang_vel_xy": w[:, :2].square().sum(-1),
            "dof_acc": accel[:, :8].square().sum(-1), "dof_vel": dq[:, :8].square().sum(-1),
            "dof_acc_arm": accel[:, 8:].square().sum(-1), "dof_vel_arm": dq[:, 8:].square().sum(-1),
            "collision": (torch.linalg.vector_norm(contacts_all.data.net_forces_w[:, self.penalized_ids], dim=-1) > .1).float().sum(-1),
            "action_rate": delta[:, :8].sum(-1), "action_rate_arm": delta[:, 8:].sum(-1),
            "dof_pos_limits": ((low - pos).clamp(min=0.) + (pos - high).clamp(min=0.)).sum(-1),
            "torque_limits": (torques.abs() - .9 * self.effort_limits).clamp(min=0.).sum(-1),
            "base_height": (robot.data.root_pos_w[:, 2] - self.origins[:, 2] - ROOT_HEIGHT).square(),
        }
        weighted = {name: terms[name] * scale * POLICY_DT for name, scale in self.scales.items()}
        total = torch.stack(tuple(weighted.values())).sum(0)
        if not torch.isfinite(total).all():
            raise RuntimeError("Nonfinite Ranger reward")
        self.prev_vel[:], self.prev_actions[:] = dq, actions
        self.step_count += self.num_envs
        for name, value in weighted.items():
            self.term_sums[name] += float(value.sum().item())
        self.total_sum += float(total.sum().item())
        return total, weighted


class RangerUniFPVecEnv(UniFPVecEnv):
    asset_version, asset_path = ASSET_VERSION, str(USD_PATH)
    num_single_obs, num_obs = 58, 58 * 32
    num_privileged_obs, num_actions = CRITIC_SIZE * 3, 14
    critic_slices, scene_cfg_class = CRITIC_SLICES, RangerSceneCfg

    def __init__(self, *, ranger_explicit_pd=False, **kwargs):
        self.implicit_pd = not ranger_explicit_pd
        if ranger_explicit_pd:
            # Reproduce historical checkpoints under their original dynamics.
            def explicit_scene_cfg(**scene_kwargs):
                cfg = RangerSceneCfg(**scene_kwargs)
                for actuator in cfg.robot.actuators.values():
                    actuator.stiffness = actuator.damping = 0.
                return cfg
            self.scene_cfg_class = explicit_scene_cfg
        if kwargs.get("rough_terrain", False):
            raise ValueError("Ranger adapter currently supports flat ground only")
        if not USD_PATH.is_file():
            raise FileNotFoundError(USD_PATH)
        super().__init__(**kwargs)

    def _create_task(self, domain_randomization, random_goals, curriculum_start_step):
        self.controller = RangerJointPD(self.robot, implicit_pd=self.implicit_pd)
        self.actor = UniFPActorInputs(self.robot, self.controller, self.num_envs)
        self.domain = UniFPDomainRandomization(
            self.robot, self.controller, self.scene, enabled=domain_randomization,
            com_body="base_link", gripper_body="arm_seg6",
        )
        self.privileged = RangerPrivilegedInputs(self.robot, self.controller, self.scene, self.domain)
        self.rewards = RangerRewards(self.robot, self.controller, self.scene)
        # URDF arm mounting origin + nominal root height, preserving yaw-only
        # center semantics. Start from the validated reference arm pose.
        offset = (.174, 0., ROOT_HEIGHT + .09535)
        home_relative = torch.tensor((.34592088 - offset[0], .000003377, .818693792 + ROOT_HEIGHT - offset[2]))
        initial = tuple(cart_to_sphere(home_relative).tolist())
        self.goal = UniFPEEGoal(
            self.num_envs, self.device, random_goals=random_goals,
            offset=offset, initial=initial, first_goal=(.66, 1., 0.),
            goal_lower=(.35, .15, -1.2), goal_upper=(.75, 1.4, 1.2),
        )
        # Adapt magnitude to the smaller A1, retaining released pulse timing,
        # independent command/disturbance streams, and the delayed start.
        self.curriculum = UniFPForceCurriculum(
            self.num_envs, self.device, start_step=curriculum_start_step, max_force_n=15.,
        )

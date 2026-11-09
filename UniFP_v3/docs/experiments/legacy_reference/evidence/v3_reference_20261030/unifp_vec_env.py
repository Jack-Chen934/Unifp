"""Callable B2+Z1 vector environment for the released UniFP PPO interface.

Import this module only after Isaac Lab's AppLauncher has started Isaac Sim.
The environment owns one SimulationContext; call close() after use.
"""

from __future__ import annotations

import math
import random
import numpy as np

import torch
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg, HfRandomUniformTerrainCfg
from isaaclab.terrains.height_field.hf_terrains import random_uniform_terrain
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz

from unifp_domain_randomization import UniFPDomainRandomization
from unifp_force_curriculum import UniFPForceCurriculum
from unifp_observations import UniFPActorInputs
from unifp_stage1_control import (
    ALL_JOINT_NAMES, DECIMATION, DEFAULT_ANGLES, NamedJointPD, PHYSICS_DT, USD_PATH, urdf_joint_limits,
)
from unifp_task_rewards import MAX_EPISODE_STEPS, UniFPTaskRewards
from unifp_task_signals import CRITIC_SLICES, UniFPEEGoal, UniFPPrivilegedInputs


_LIMITS = urdf_joint_limits()


@configclass
class UniFPSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0)
        ),
    )
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(USD_PATH), activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.6), joint_pos=DEFAULT_ANGLES),
        actuators={"torque_control": ImplicitActuatorCfg(
            joint_names_expr=list(ALL_JOINT_NAMES), stiffness=0.0, damping=0.0,
            effort_limit_sim={name: _LIMITS[name][0] for name in ALL_JOINT_NAMES},
            velocity_limit_sim={name: _LIMITS[name][1] for name in ALL_JOINT_NAMES},
        )},
    )
    feet = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*_foot", update_period=0.0, history_length=1)
    robot_contacts = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", update_period=0.0, history_length=1)


def _unifp_rough_flat_terrain(difficulty, cfg):
    """Generate the released rough-flat distribution for one terrain tile.

    The Isaac Gym task samples an amplitude uniformly from zero to
    ``0.05 * difficulty`` and then calls ``random_uniform_terrain`` with
    symmetric heights.  Isaac Lab's terrain generator passes difficulty to
    the sub-terrain function, so this wrapper preserves that dependency while
    reusing Isaac Lab's native height-field implementation.
    """
    max_amplitude = max(0.0, float(cfg.noise_range[1])) * float(np.clip(difficulty, 0.0, 1.0))
    amplitude = float(np.random.uniform(0.0, max_amplitude))
    return random_uniform_terrain(0.0, cfg.replace(noise_range=(-amplitude, amplitude)))


@configclass
class UniFPRoughFlatTerrainCfg(HfRandomUniformTerrainCfg):
    function = _unifp_rough_flat_terrain


# The released task uses only the ``rough flat`` sub-terrain.  The default
# vector environment remains flat; rough terrain is opt-in because the full
# official 10x20, 25 m-border map is expensive on an 8 GB GPU.
UNIFP_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=2.0,
    num_rows=2,
    num_cols=2,
    curriculum=False,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=None,
    use_cache=False,
    sub_terrains={
        "rough_flat": UniFPRoughFlatTerrainCfg(
            proportion=1.0,
            noise_range=(-0.05, 0.05),
            noise_step=0.005,
            downsampled_scale=0.075,
            border_width=0.0,
        ),
    },
)


@configclass
class UniFPRoughSceneCfg(UniFPSceneCfg):
    """Small development rough terrain matching UniFP's rough-flat mode."""

    ground = None
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=UNIFP_ROUGH_TERRAINS_CFG,
        max_init_terrain_level=1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
        ),
    )


class UniFPVecEnv:
    """UniFP PPO's `reset/get_observations/step` contract on Isaac Lab.

    `step` returns ({obs, privileged_obs, obs_pred}, reward, done, extras).
    Observations correspond to the state after automatic per-environment reset;
    reward and done correspond to the pre-reset transition.
    """

    num_obs = 2336
    num_single_obs = 73
    num_privileged_obs = 447
    num_pred_obs = 12
    num_actions = 17
    critic_slices = CRITIC_SLICES
    scene_cfg_class = None
    max_episode_length = MAX_EPISODE_STEPS

    def __init__(self, *, num_envs: int, device: str = "cuda:0",
                 domain_randomization: bool = True, force_curriculum: bool = True,
                 observation_noise: bool = True, command_sampling: bool = True,
                 random_goals: bool = True, curriculum_start_step: int = 192001,
                 seed: int = 0, safety_height_termination: bool = False,
                 rough_terrain: bool = False, push_robots: bool = False,
                 push_interval_steps: int = 400, max_push_vel_xy: float = 0.8,
                 capture_diagnostics: bool = False):
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        if self.scene_cfg_class is None and not USD_PATH.is_file():
            raise FileNotFoundError(USD_PATH)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.capture_diagnostics = capture_diagnostics
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
            dt=PHYSICS_DT, device=device,
            physx=sim_utils.PhysxCfg(enable_external_forces_every_iteration=True),
        ))
        scene_cfg_cls = self.scene_cfg_class or (UniFPRoughSceneCfg if rough_terrain else UniFPSceneCfg)
        self.scene = InteractiveScene(scene_cfg_cls(num_envs=num_envs, env_spacing=3.0))
        self.sim.reset()
        self.robot = self.scene["robot"]
        self.num_envs = num_envs
        self.device = self.scene.device
        self._all_ids = torch.arange(num_envs, device=self.device)
        self._create_task(domain_randomization, random_goals, curriculum_start_step)
        self.force_curriculum_enabled = force_curriculum
        self.safety_height_termination = safety_height_termination
        self.rough_terrain = rough_terrain
        self.push_robots_enabled = push_robots
        if push_interval_steps < 1:
            raise ValueError("push_interval_steps must be positive")
        if max_push_vel_xy < 0:
            raise ValueError("max_push_vel_xy must be nonnegative")
        self.push_interval_steps = int(push_interval_steps)
        self.max_push_vel_xy = float(max_push_vel_xy)
        self.push_count = 0
        self.push_peak_velocity = 0.0
        self.observation_noise = observation_noise
        self.command_sampling = command_sampling
        self.zero_wrench = torch.zeros(num_envs, 1, 3, device=self.device)
        self.gripper_id = self.privileged.ee_id
        self.global_policy_steps = 0
        self.episode_length_buf = self.rewards.episode_steps
        self.obs_buf = torch.zeros(num_envs, self.num_obs, device=self.device)
        self.privileged_obs_buf = torch.zeros(num_envs, self.num_privileged_obs, device=self.device)
        self.obs_pred = torch.zeros(num_envs, self.num_pred_obs, device=self.device)
        self.rew_buf = torch.zeros(num_envs, device=self.device)
        self.reset_buf = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.extras: dict = {}
        self.reset()

    def _create_task(self, domain_randomization, random_goals, curriculum_start_step):
        self.controller = NamedJointPD.from_robot(self.robot)
        self.actor = UniFPActorInputs(self.robot, self.controller, self.num_envs)
        self.domain = UniFPDomainRandomization(self.robot, self.controller, self.scene, enabled=domain_randomization)
        self.privileged = UniFPPrivilegedInputs(
            self.robot, self.controller, self.scene, friction=self.domain.friction,
            mass_deltas=self.domain.mass_deltas, motor_strength=self.domain.motor_strength,
        )
        self.rewards = UniFPTaskRewards(self.robot, self.controller, self.scene)
        self.goal = UniFPEEGoal(self.num_envs, self.device, random_goals=random_goals)
        self.curriculum = UniFPForceCurriculum(self.num_envs, self.device, start_step=curriculum_start_step)

    def _sample_velocity_commands(self, env_ids: torch.Tensor):
        if not self.command_sampling or env_ids.numel() == 0:
            return
        size = len(env_ids)
        samples = torch.rand(size, 3, device=self.device)
        self.actor.commands[env_ids, 0] = samples[:, 0] * 1.2 - 0.6
        self.actor.commands[env_ids, 1] = samples[:, 1] * 0.8 - 0.4
        self.actor.commands[env_ids, 2] = samples[:, 2] * 1.2 - 0.6
        stationary = torch.rand(size, device=self.device) < 0.3
        self.actor.commands[env_ids[stationary], :3] = 0.0
        commands = self.actor.commands[env_ids, :3]
        small = ((commands[:, 0].abs() <= 0.1) & (commands[:, 1].abs() <= 0.1)
                 & (commands[:, 2].abs() <= 0.2))
        self.actor.commands[env_ids[small], :3] = 0.0

    def _reset_indices(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        root = self.robot.data.default_root_state[env_ids].clone()
        root[:, :3] += self.scene.env_origins[env_ids]
        joints = self.robot.data.default_joint_pos[env_ids].clone()
        if self.domain.enabled:
            # Isaac Gym resets leg joints in [0.5,1.5] default and perturbs
            # the five action-arm joints by +/-0.5 rad. The held joints stay fixed.
            split = getattr(self.controller, "base_action_count", 12)
            leg_ids = self.controller.action_indices[:split]
            arm_ids = self.controller.action_indices[split:]
            joints[:, leg_ids] *= 0.5 + torch.rand(len(env_ids), len(leg_ids), device=self.device)
            joints[:, arm_ids] += torch.rand(len(env_ids), len(arm_ids), device=self.device) - 0.5
            root[:, :2] += torch.rand(len(env_ids), 2, device=self.device) * 2.0 - 1.0
            yaw = (torch.rand(len(env_ids), device=self.device) * 2.0 - 1.0) * (math.pi / 2)
            root[:, 3:7] = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
            root[:, 7:13] = torch.rand(len(env_ids), 6, device=self.device) - 0.5
            self.domain.resample_motor(env_ids)
        self.robot.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(root[:, 7:13], env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joints, self.robot.data.default_joint_vel[env_ids], env_ids=env_ids)
        self.scene.reset(env_ids)
        self.actor.reset(env_ids)
        self.privileged.reset(env_ids)
        self.rewards.reset(env_ids)
        self.curriculum.reset(env_ids)
        self.goal.reset(env_ids)
        self._sample_velocity_commands(env_ids)
        self.goal.update(self.robot, self.actor.commands, env_ids, advance=False)

    def _observations(self, actions: torch.Tensor, gripper_force_w: torch.Tensor):
        _, self.obs_buf = self.actor.observe(self.robot, actions, add_noise=self.observation_noise)
        self.obs_pred, _, self.privileged_obs_buf = self.privileged.observe(
            self.robot, self.controller, self.scene["feet"], self.actor.gait_phase, actions,
            self.actor.commands, self.goal, gripper_force_w, torch.zeros_like(gripper_force_w),
        )
        # Own the returned tensors: the next actor/critic FIFO shift mutates
        # its backing storage, while PPO still holds the previous transition.
        self.obs_buf = self.obs_buf.clamp(-100.0, 100.0).clone()
        self.privileged_obs_buf = self.privileged_obs_buf.clamp(-100.0, 100.0).clone()
        self.obs_pred = self.obs_pred.clamp(-100.0, 100.0).clone()
        return self.get_observations()

    def get_observations(self):
        return {"obs": self.obs_buf, "privileged_obs": self.privileged_obs_buf, "obs_pred": self.obs_pred}

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def state_snapshot(self):
        """Owned state tensors at one instant, safe across simulation and reset."""
        ee = self.robot.data.body_pos_w[:, self.gripper_id].clone()
        goal = self.goal.goal_world.clone()
        error = ee - goal
        roll, pitch, _ = euler_xyz_from_quat(self.robot.data.root_quat_w)
        return {
            "ee_position_world_m": ee, "goal_position_world_m": goal,
            "position_error_xyz_m": error,
            "position_error_m": torch.linalg.vector_norm(error, dim=-1),
            "base_height_m": (self.robot.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]).clone(),
            "roll_rad": roll.clone(), "pitch_rad": pitch.clone(),
            "roll_abs_rad": roll.abs(), "pitch_abs_rad": pitch.abs(),
            "episode_steps": self.episode_length_buf.clone(),
            "goal_sphere": self.goal.current.clone(),
            "goal_orientation_delta": self.goal.orientation_delta.clone(),
            "joint_position_rad": self.robot.data.joint_pos[:, self.controller.action_indices].clone(),
            "joint_velocity_rad_s": self.robot.data.joint_vel[:, self.controller.action_indices].clone(),
        }

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._all_ids
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
        self._reset_indices(env_ids)
        zeros = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self.rew_buf.zero_()
        self.reset_buf.zero_()
        obs = self._observations(zeros, torch.zeros(self.num_envs, 3, device=self.device))
        return obs, self.privileged_obs_buf

    def step(self, actions: torch.Tensor):
        if actions.shape != (self.num_envs, self.num_actions):
            raise ValueError(f"Expected actions ({self.num_envs}, {self.num_actions}), got {tuple(actions.shape)}")
        actions = actions.to(self.device).clamp(-100.0, 100.0)
        if not torch.isfinite(actions).all():
            raise ValueError("Nonfinite policy action")
        self.actor.commands[:, 9:15] = 0.0
        force = torch.zeros_like(self.zero_wrench)
        for substep in range(DECIMATION):
            if self.force_curriculum_enabled:
                cmd, disturbance = self.curriculum.update(
                    self.global_policy_steps, self.episode_length_buf, substep_index=substep,
                )
                self.actor.commands[:, 9:12] = cmd
                force[:, 0] = disturbance
            torques = self.controller.compute(
                actions, self.robot.data.joint_pos, self.robot.data.joint_vel,
                motor_strength=self.domain.motor_strength,
            )
            # Implicit PD sends position/velocity targets in compute(); do not
            # add the diagnostic PD effort a second time as feedforward torque.
            implicit_pd = getattr(self.controller, "implicit_pd", False)
            self.robot.set_joint_effort_target(torch.zeros_like(torques) if implicit_pd else torques)
            self.robot.set_external_force_and_torque(
                forces=force, torques=self.zero_wrench, body_ids=[self.gripper_id], is_global=True,
            )
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(PHYSICS_DT)
        if implicit_pd:
            # Isaac Lab's limited PD effort estimate, also used by its
            # ImplicitActuator; this is not a measured constraint torque.
            torques = self.robot.data.applied_torque
        self.actor.update_gait()
        # The released post-physics callback resamples velocity commands at
        # episode steps 250, 500, ... before reward/observation computation.
        next_length = self.episode_length_buf + 1
        self._sample_velocity_commands((next_length % 250 == 0).nonzero(as_tuple=False).flatten())
        if self.goal.schedule is not None:
            self.goal.schedule_index = self.global_policy_steps + 1
        self.goal.update(self.robot, self.actor.commands)
        self.episode_length_buf += 1
        # Released UniFP calls _push_robots() every ceil(8 s / 0.02 s) policy
        # steps.  It emulates an impulse by setting the base x/y velocity,
        # with a 2.5x boost for standing commands in the B2+Z1 task.
        if self.push_robots_enabled and (self.global_policy_steps + 1) % self.push_interval_steps == 0:
            push_vel = torch.rand(self.num_envs, 2, device=self.device) * (2.0 * self.max_push_vel_xy) - self.max_push_vel_xy
            standing = self.actor.commands[:, :3].abs().sum(dim=-1) == 0.0
            push_vel = torch.where(standing[:, None], push_vel * 2.5, push_vel)
            root_vel = self.robot.data.root_vel_w.clone()
            root_vel[:, :2] = push_vel
            self.robot.write_root_velocity_to_sim(root_vel)
            self.push_count += self.num_envs
            self.push_peak_velocity = max(self.push_peak_velocity, float(push_vel.abs().max().item()))
        applied_force = force[:, 0].clone()
        reward_frame = self.privileged.observe(
            self.robot, self.controller, self.scene["feet"], self.actor.gait_phase, actions,
            self.actor.commands, self.goal, applied_force, torch.zeros_like(applied_force),
            push_history=False,
        )[1]
        self.rew_buf, components = self.rewards.compute(
            self.robot, self.scene["feet"], self.scene["robot_contacts"],
            self.controller, actions, torques, self.actor.commands, self.goal,
            applied_force, torch.zeros_like(applied_force), reward_frame, self.critic_slices,
        )
        # Official UniFP has no contact termination bodies and no height
        # guard.  The latter remains an explicit opt-in for debugging a bad
        # asset conversion, rather than silently changing training dynamics.
        done, timeouts = self.rewards.termination(
            self.robot, safety_height=self.safety_height_termination,
        )
        self.reset_buf = done.clone()
        self.extras = {"time_outs": timeouts.clone(), "reward_terms": components}
        if self.capture_diagnostics:
            # Capture the true resulting state before _reset_indices overwrites it.
            self.extras["pre_reset_state"] = self.state_snapshot()
            self.extras["termination_reasons"] = {
                key: value.clone() for key, value in self.rewards.last_termination_masks.items()
            }
        reset_ids = done.nonzero(as_tuple=False).flatten()
        if reset_ids.numel():
            self._reset_indices(reset_ids)
        observed_actions = actions.clone()
        observed_actions[reset_ids] = 0.0
        applied_force[reset_ids] = 0.0
        obs = self._observations(observed_actions, applied_force)
        self.global_policy_steps += 1
        return obs, self.rew_buf.clone(), self.reset_buf.clone(), self.extras

    def disturbance_summary(self):
        return {
            "rough_terrain": self.rough_terrain,
            "push_robots": self.push_robots_enabled,
            "push_interval_steps": self.push_interval_steps,
            "push_interval_s": self.push_interval_steps * DECIMATION * PHYSICS_DT,
            "max_push_vel_xy": self.max_push_vel_xy,
            "push_count": self.push_count,
            "push_peak_abs_velocity": self.push_peak_velocity,
        }

    def close(self):
        self.sim.clear_instance()

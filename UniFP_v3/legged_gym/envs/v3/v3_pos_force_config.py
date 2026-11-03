"""Configuration classes for the V3 zero-end-effector-force task."""

import math
import os

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class V3PosForceCfg(LeggedRobotCfg):
    """Define the V3 zero-force task without changing the official B2 task."""

    class env(LeggedRobotCfg.env):
        num_envs = 32
        num_actions = 14
        frame_stack = 32
        c_frame_stack = 3
        num_single_obs = 58
        num_pred_obs = 12
        num_observations = frame_stack * num_single_obs
        single_num_privileged_obs = 98
        num_privileged_obs = c_frame_stack * single_num_privileged_obs
        episode_length_s = 20.0

    class asset(LeggedRobotCfg.asset):
        file = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "resources/robots/v3_a1_desktop_train_v1/urdf/a1arm_rgm3_desktop_fixed_gripper.urdf",
        )
        name = "v3_a1"
        foot_name = "_Link"
        terminate_after_contacts_on = ["base_link"]
        collapse_fixed_joints = False
        replace_cylinder_with_capsule = False

    class init_state(LeggedRobotCfg.init_state):
        # Desktop wheel collision minimum is about -0.330 m in base coordinates.
        pos = [0.0, 0.0, 0.34]
        default_joint_angles = {
            "fl_steering_joint": 0.0,
            "fr_steering_joint": 0.0,
            "bl_steering_joint": 0.0,
            "br_steering_joint": 0.0,
            "fl_wheel_drive_joint": 0.0,
            "fr_wheel_drive_joint": 0.0,
            "bl_wheel_drive_joint": 0.0,
            "br_wheel_drive_joint": 0.0,
            "arm_joint1": 0.0,
            "arm_joint2": 1.0,
            "arm_joint3": -1.5,
            "arm_joint4": 0.0,
            "arm_joint5": 0.0,
            "arm_joint6": 0.0,
        }

    class control(LeggedRobotCfg.control):
        decimation = 4
        control_type = "P"
        action_scale = 0.25
        wheel_action_scale = 5.0
        wheel_velocity_limit = 20.0
        # A03 short-test candidates; these are not frozen long-training values.
        stiffness = {
            "fl_steering_joint": 24.9218,
            "fr_steering_joint": 24.9218,
            "bl_steering_joint": 24.9218,
            "br_steering_joint": 24.9218,
            "fl_wheel_drive_joint": 0.0,
            "fr_wheel_drive_joint": 0.0,
            "bl_wheel_drive_joint": 0.0,
            "br_wheel_drive_joint": 0.0,
            "arm_joint1": 27.8093,
            "arm_joint2": 300.072,
            "arm_joint3": 118.943,
            "arm_joint4": 1.40418,
            "arm_joint5": 12.9504,
            "arm_joint6": 1.06007,
        }
        damping = {
            "fl_steering_joint": 0.664582,
            "fr_steering_joint": 0.664582,
            "bl_steering_joint": 0.664582,
            "br_steering_joint": 0.664582,
            "fl_wheel_drive_joint": 0.227385,
            "fr_wheel_drive_joint": 0.227385,
            "bl_wheel_drive_joint": 0.227385,
            "br_wheel_drive_joint": 0.227385,
            "arm_joint1": 1.48316,
            "arm_joint2": 13.7176,
            "arm_joint3": 4.75773,
            "arm_joint4": 0.0249632,
            "arm_joint5": 0.414412,
            "arm_joint6": 0.0188456,
        }

    class commands(LeggedRobotCfg.commands):
        num_commands = 15
        heading_command = False
        resampling_time = 5.0
        class ranges:
            lin_vel_x = [-0.6, 0.6]
            lin_vel_y = [-0.4, 0.4]
            ang_vel_yaw = [-0.6, 0.6]

    class goal_ee:
        # A08 candidate domain for validation; workspace margins are not frozen.
        radius = [0.45, 0.70]
        elevation = [0.0, math.radians(60.0)]
        azimuth = [math.radians(-60.0), math.radians(60.0)]
        move_time = [1.0, 3.0]
        hold_time = [0.5, 2.0]
        num_path_check_samples = 10
        min_tcp_height = 0.05
        max_resamples = 10

    class rewards(LeggedRobotCfg.rewards):
        base_height_target = 0.32
        only_positive_rewards = False
        tracking_sigma = 0.25
        tracking_ee_sigma = 1.0
        soft_dof_pos_limit = 0.95
        soft_torque_limit = 0.90
        collision_force_threshold = 0.1

        # A08 first validation candidates; independent of the official B2 scales.
        class scales:
            tracking_ee_force_world = 2.0
            tracking_lin_vel = 2.0
            tracking_ang_vel = 1.0
            alive = 1.5
            torques_steering = -5e-6
            torques_drive = -5e-6
            torques_arm = -5e-6
            action_rate_steering = -0.02
            action_rate_drive = -0.02
            action_rate_arm = -0.045
            dof_vel_steering = -8e-4
            dof_vel_arm = -2e-4
            dof_acc_steering = -2.5e-7
            dof_acc_drive = -2.5e-7
            dof_acc_arm = -4.5e-7
            torque_limits_steering = -0.005
            torque_limits_drive = -0.005
            torque_limits_arm = -0.005
            dof_pos_limits = -10.0
            lin_vel_z = -1.5
            ang_vel_xy = -0.02
            base_height = -2.0
            collision = -5.0

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"
        curriculum = False
        measure_heights = False
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0

    class domain_rand(LeggedRobotCfg.domain_rand):
        # A09 validation candidates, applied independently of run_purpose.
        randomize_friction = True
        friction_range = [0.3, 2.0]
        randomize_base_mass = True
        base_mass_scale_range = [0.9, 1.1]
        randomize_base_com = True
        base_com_range = [-0.02, 0.02]
        randomize_tool_mass = True
        tool_mass_scale_range = [0.9, 1.1]
        randomize_execution_scale = True
        execution_scale_range = [0.85, 1.15]
        push_robots = True
        push_interval_s = 8.0
        max_push_vel_xy = 0.8

    class sim(LeggedRobotCfg.sim):
        dt = 0.005
        substeps = 1

        class physx(LeggedRobotCfg.sim.physx):
            # A11-Q6: avoid missing collision interactions at 4096 V3 environments.
            max_gpu_contact_pairs = 2**25

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0

    class run:
        reproduction_part = None
        run_purpose = None


class V3PosForceCfgPPO(LeggedRobotCfgPPO):
    """Define the V3 policy configuration using the official learner structure."""

    seed = 0
    runner_class_name = "V3OnPolicyRunner"

    class policy(LeggedRobotCfgPPO.policy):
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"
        init_noise_std = 1.0

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = "V3ActorCritic"
        algorithm_class_name = "V3PPO"
        num_steps_per_env = 24
        max_iterations = 8000
        experiment_name = "v3_pos_force_stage_a"

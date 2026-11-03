"""V3 environment for the zero-force reproduction stage."""

import math
import random

import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_apply, quat_rotate_inverse

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.v3_asset_collision import apply_desktop_assembly_filters


ACTION_JOINT_NAMES = (
    "fl_steering_joint", "fr_steering_joint", "bl_steering_joint", "br_steering_joint",
    "fl_wheel_drive_joint", "fr_wheel_drive_joint", "bl_wheel_drive_joint", "br_wheel_drive_joint",
    "arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4", "arm_joint5", "arm_joint6",
)
POSITION_JOINT_NAMES = ACTION_JOINT_NAMES[:4] + ACTION_JOINT_NAMES[8:]
WHEEL_JOINT_NAMES = ACTION_JOINT_NAMES[4:8]
POSITION_OBS_INDICES = (0, 1, 2, 3, 8, 9, 10, 11, 12, 13)
COMMAND_SCALE = (2.0, 2.0, 0.25, 0.5, 1.0, 1.3, 0.5, 0.5, 0.5,
                 0.01, 0.01, 0.01, 0.01, 0.01, 0.01)
TRAINING_POLICY_STEPS_PER_ITERATION = 24
ARM_MOUNT_X = 0.174
ARM_MOUNT_Z = 0.09535

REWARD_NAMES = (
    "tracking_ee_force_world", "tracking_lin_vel", "tracking_ang_vel", "alive",
    "torques_steering", "torques_drive", "torques_arm",
    "action_rate_steering", "action_rate_drive", "action_rate_arm",
    "dof_vel_steering", "dof_vel_arm",
    "dof_acc_steering", "dof_acc_drive", "dof_acc_arm",
    "torque_limits_steering", "torque_limits_drive", "torque_limits_arm",
    "dof_pos_limits", "lin_vel_z", "ang_vel_xy", "base_height", "collision",
)
COLLISION_BODY_NAMES = (
    "base_link", "arm_base", "arm_seg1", "arm_seg2", "arm_seg3", "arm_seg4",
    "arm_seg5", "arm_seg6", "gripper1", "gripper2", "aluminum_profiles",
    "wrist_camera_connector", "wrist_camera", "rear_top_camera_connector",
    "rear_top_camera", "right_camera_connector", "right_camera",
    "left_camera_connector", "left_camera",
)
ERROR_METRIC_NAMES = (
    "tcp_geometric_l1_m", "tcp_geometric_l2_m", "base_planar_speed_m_s", "base_yaw_rate_rad_s",
)


class V3PosForce(LeggedRobot):
    """Run the V3 robot with the released UniFP simulator and learner interface."""

    def _process_dof_props(self, props: Any, env_id: int) -> Any:
        """Set explicit effort mode while preserving the base DOF limit contract.

        Args:
            props: Gym DOF properties for one V3 asset instance.
            env_id: Environment index whose properties are initialized.

        Returns:
            DOF property array with imported servo stiffness removed.

        Note:
            V3 computes and submits explicit PD torques each physics step. The
            importer can assign near-maximum stiffness to continuous joints;
            keeping it alongside effort control prevents wheel response.
        """
        props = super()._process_dof_props(props, env_id)
        # The base callback stores reward soft bounds; control requires URDF hard bounds.
        if env_id == 0:
            self.dof_pos_limits[:, 0] = torch.as_tensor(props["lower"].copy(), device=self.device)
            self.dof_pos_limits[:, 1] = torch.as_tensor(props["upper"].copy(), device=self.device)
        props["driveMode"][:] = gymapi.DOF_MODE_EFFORT
        props["stiffness"][:] = 0.0
        return props

    def _create_envs(self) -> None:
        """Create V3 actors and install the desktop asset's local assembly rules.

        Returns:
            None.

        Raises:
            ValueError: Imported collision shapes differ from the asset manifest.

        Note:
            Applies filters before prepare_sim. Keeps nonadjacent self-collision
            enabled and leaves the official robot creation code unchanged.
        """
        super()._create_envs()
        manifest_path = Path(self.cfg.asset.file).parent.parent / 'asset_manifest.json'
        for env_handle, actor_handle in zip(self.envs, self.actor_handles):
            apply_desktop_assembly_filters(self.gym, env_handle, actor_handle, manifest_path)
        self._initialize_creation_randomization()

    def _process_rigid_body_props(self, props: Any, env_id: int) -> Any:
        """Keep the imported nominal bodies until the V3 creation pass.

        Args:
            props: Imported Gym body properties for one actor.
            env_id: Actor environment index; no body is randomized in this callback.

        Returns:
            Unchanged properties; V3 applies named mass/COM changes after creation.
        """
        return props

    def _initialize_creation_randomization(self) -> None:
        """Capture nominal body properties and apply A09 creation-lifetime samples.

        Note:
            Runs once before prepare_sim. Samples mass scales, local COM offsets,
            and friction independently of run purpose, then publishes actual readbacks.
        """
        randomization = self.cfg.domain_rand
        self.nominal_mass_properties = []
        requested_delta = torch.zeros(self.num_envs, 5)
        for env_id, (env_handle, actor_handle) in enumerate(zip(self.envs, self.actor_handles)):
            properties = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            names = self.gym.get_actor_rigid_body_names(env_handle, actor_handle)
            nominal_bodies = []
            for name in ("base_link", "arm_seg6"):
                body = properties[names.index(name)]
                nominal_bodies.append((
                    body.mass, (body.com.x, body.com.y, body.com.z),
                    tuple(tuple(getattr(getattr(body.inertia, row), column) for column in "xyz")
                          for row in "xyz"),
                ))
            self.nominal_mass_properties.append(nominal_bodies)
            if randomization.randomize_base_mass:
                lower, upper = randomization.base_mass_scale_range
                requested_delta[env_id, 0] = nominal_bodies[0][0] * (lower + torch.rand(()).item() * (upper - lower) - 1.0)
            if randomization.randomize_base_com:
                lower, upper = randomization.base_com_range
                requested_delta[env_id, 1:4] = lower + torch.rand(3) * (upper - lower)
            if randomization.randomize_tool_mass:
                lower, upper = randomization.tool_mass_scale_range
                requested_delta[env_id, 4] = nominal_bodies[1][0] * (lower + torch.rand(()).item() * (upper - lower) - 1.0)
        if randomization.randomize_friction:
            requested_friction = self.friction_coeffs.reshape(self.num_envs, 1).clone()
        else:
            requested_friction = torch.ones(self.num_envs, 1)
        if self.creation_properties is not None:
            requested_delta = self.creation_properties["mass_com_delta"]
            requested_friction = self.creation_properties["friction"]
        self._apply_creation_properties(requested_delta, requested_friction)

    def _apply_creation_properties(
        self, mass_com_delta: torch.Tensor, friction: torch.Tensor
    ) -> None:
        """Apply and read back named physical properties without inertia recomputation.

        Args:
            mass_com_delta: [N,5] base mass kg, local COM xyz m, tool mass kg changes.
            friction: [N,1] coefficient applied to every robot collision shape.

        Raises:
            RuntimeError: Gym readback does not match the requested mass, COM or friction.

        Note:
            Fixed-geometry mass scaling also scales the nominal inertia tensor.
            COM translation leaves the central inertia unchanged. No reset or RNG use.
        """
        requested_delta = mass_com_delta.detach().cpu()
        requested_friction = friction.detach().cpu()
        actual_delta = torch.zeros_like(requested_delta)
        actual_friction = torch.zeros_like(requested_friction)
        for env_id, (env_handle, actor_handle) in enumerate(zip(self.envs, self.actor_handles)):
            properties = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            names = self.gym.get_actor_rigid_body_names(env_handle, actor_handle)
            for slot, name in enumerate(("base_link", "arm_seg6")):
                body = properties[names.index(name)]
                nominal_mass, nominal_com, nominal_inertia = self.nominal_mass_properties[env_id][slot]
                mass_change = float(requested_delta[env_id, 0 if slot == 0 else 4])
                body.mass = nominal_mass + mass_change
                mass_scale = body.mass / nominal_mass
                for row_index, row in enumerate("xyz"):
                    for column_index, column in enumerate("xyz"):
                        setattr(getattr(body.inertia, row), column,
                                nominal_inertia[row_index][column_index] * mass_scale)
                com_delta = requested_delta[env_id, 1:4] if slot == 0 else torch.zeros(3)
                body.com = gymapi.Vec3(*(nominal_com[index] + float(com_delta[index]) for index in range(3)))
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, properties, recomputeInertia=False)
            readback = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            base = readback[names.index("base_link")]
            tool = readback[names.index("arm_seg6")]
            nominal_base, nominal_tool = self.nominal_mass_properties[env_id]
            for slot, body in enumerate((base, tool)):
                nominal_mass, _, nominal_inertia = self.nominal_mass_properties[env_id][slot]
                mass_scale = body.mass / nominal_mass
                for row_index, row in enumerate("xyz"):
                    for column_index, column in enumerate("xyz"):
                        actual_inertia = getattr(getattr(body.inertia, row), column)
                        expected_inertia = nominal_inertia[row_index][column_index] * mass_scale
                        if not math.isclose(actual_inertia, expected_inertia, abs_tol=1e-6, rel_tol=1e-5):
                            raise RuntimeError(f"V3 inertia readback mismatch in env {env_id}")
            actual_delta[env_id] = torch.tensor((
                base.mass - nominal_base[0], base.com.x - nominal_base[1][0],
                base.com.y - nominal_base[1][1], base.com.z - nominal_base[1][2],
                tool.mass - nominal_tool[0],
            ))
            shape_properties = self.gym.get_actor_rigid_shape_properties(env_handle, actor_handle)
            coefficient = float(requested_friction[env_id, 0])
            for shape in shape_properties:
                shape.friction = coefficient
            self.gym.set_actor_rigid_shape_properties(env_handle, actor_handle, shape_properties)
            shapes_readback = self.gym.get_actor_rigid_shape_properties(env_handle, actor_handle)
            if any(not math.isclose(shape.friction, coefficient, rel_tol=1e-5, abs_tol=1e-6)
                   for shape in shapes_readback):
                raise RuntimeError(f"V3 friction readback mismatch in env {env_id}")
            actual_friction[env_id, 0] = shapes_readback[0].friction
        if not torch.allclose(actual_delta, requested_delta, atol=1e-5, rtol=1e-5):
            raise RuntimeError("V3 mass/COM readback mismatch")
        self.mass_com_delta = actual_delta.to(self.device)
        self.friction = actual_friction.to(self.device)

    def _apply_scheduled_base_push(self) -> bool:
        """Apply at most one synchronized velocity push at a published training boundary.

        Returns:
            Whether this step injected independently sampled world XY base velocities.

        Note:
            Reads Runner's clock without advancing it. Explicit domain configuration
            controls pushes in every run purpose; EE force stays zero.
        """
        interval = int(round(self.cfg.domain_rand.push_interval_s / self.dt))
        evaluation_active = self.evaluation_policy_step is not None
        progress = self.evaluation_policy_step if evaluation_active else self.training_policy_steps
        last_push = self.evaluation_last_push_step if evaluation_active else self.last_push_policy_step
        if (not self.cfg.domain_rand.push_robots or progress == 0
                or progress % interval != 0 or progress == last_push):
            return False
        max_velocity = self.cfg.domain_rand.max_push_vel_xy
        if self.push_generator is None:
            random_velocity = torch.rand(self.num_envs, 2, device=self.device)
        else:
            random_velocity = torch.rand(self.num_envs, 2, device=self.device, generator=self.push_generator)
        self.last_push_velocity_world = (2.0 * random_velocity - 1.0) * max_velocity
        self.root_states[:, 7:9] = self.last_push_velocity_world
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
        if evaluation_active:
            self.evaluation_last_push_step = progress
        else:
            self.last_push_policy_step = progress
        return True

    def set_evaluation_policy_step(self, policy_step: int) -> None:
        """Set evaluation time without mutating any training progress counter.

        Args:
            policy_step: Absolute per-environment policy boundary, starting at A8000.

        Note:
            The first value seeds the separate evaluation push clock to prevent
            replaying a push exactly at the saved training boundary.
        """
        if self.evaluation_policy_step is None:
            self.evaluation_last_push_step = policy_step
        self.evaluation_policy_step = policy_step

    def set_evaluation_push_seed(self, seed: int) -> None:
        """Install an independent deterministic random stream for evaluation pushes.

        Args:
            seed: Non-negative evaluation push seed, independent of task/reset RNG.

        Raises:
            ValueError: The push seed is negative.
        """
        if seed < 0:
            raise ValueError("Evaluation push seed must be non-negative")
        self.push_generator = torch.Generator(device=self.device)
        self.push_generator.manual_seed(seed)

    def get_training_state(self) -> Dict[str, Any]:
        """Snapshot A09 persistent state without consuming randomness or stepping physics.

        Returns:
            Independent CPU tensors for properties, episode execution samples, push,
            training progress and Python/NumPy/Torch RNG states. No simulator snapshot.
        """
        return {
            "reproduction_part": self.cfg.run.reproduction_part,
            "num_envs": self.num_envs,
            "completed_iterations": self.completed_iterations,
            "training_policy_steps": self.training_policy_steps,
            "training_transitions": self.training_transitions,
            "mass_com_delta": self.mass_com_delta.detach().cpu().clone(),
            "friction": self.friction.detach().cpu().clone(),
            "execution_scale_delta": self.execution_scale_delta.detach().cpu().clone(),
            "last_push_policy_step": self.last_push_policy_step,
            "last_push_velocity_world": self.last_push_velocity_world.detach().cpu().clone(),
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state().clone(),
            "cuda_rng": [state.clone() for state in torch.cuda.get_rng_state_all()],
        }

    def load_training_state(self, state: Dict[str, Any]) -> None:
        """Verify creation-time physics and restore mutable state before fresh episodes.

        Args:
            state: Complete dictionary returned by get_training_state, with matching N/part.

        Raises:
            ValueError: Creation physics, identity, dimensions or training counters differ.
            KeyError: A required persistent-state field is missing.

        Note:
            Never rewrites prepared actors: GPU pipelines do not support runtime COM
            changes. Saved creation properties must be applied during construction.
            Does not reset or step. Episode-local execution scales resample on reset.
        """
        if state["num_envs"] != self.num_envs or state["reproduction_part"] != self.cfg.run.reproduction_part:
            raise ValueError("V3 training state identity mismatch")
        iterations, steps = state["completed_iterations"], state["training_policy_steps"]
        if (type(iterations) is not int or type(steps) is not int or iterations < 0
                or not iterations * 24 <= steps <= (iterations + 1) * 24
                or state["training_transitions"] != self.num_envs * steps):
            raise ValueError("V3 training state counters are inconsistent")
        for key, shape in (("mass_com_delta", (self.num_envs, 5)), ("friction", (self.num_envs, 1)),
                           ("execution_scale_delta", (self.num_envs, 14)),
                           ("last_push_velocity_world", (self.num_envs, 2))):
            if state[key].shape != shape or not torch.isfinite(state[key]).all():
                raise ValueError(f"V3 training state has invalid {key}")
        for key in ("mass_com_delta", "friction"):
            if not torch.allclose(getattr(self, key), state[key].to(self.device), atol=1e-5, rtol=1e-5):
                raise ValueError("Checkpoint creation properties must be applied before prepare_sim")
        self.execution_scale_delta.copy_(state["execution_scale_delta"].to(self.device))
        self.last_push_policy_step = state["last_push_policy_step"]
        self.last_push_velocity_world = state["last_push_velocity_world"].to(self.device).clone()
        self.completed_iterations, self.training_policy_steps = iterations, steps
        self.training_transitions = self.num_envs * steps
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        torch.cuda.set_rng_state_all(state["cuda_rng"])

    def __init__(
        self,
        cfg: Any,
        sim_params: Any,
        physics_engine: Any,
        sim_device: str,
        headless: bool,
        creation_properties: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """Initialize the official simulator base and V3 observation state.

        Args:
            cfg: V3 environment configuration with 14 actions and fixed-shape observations.
            sim_params: Isaac Gym parameters; the configured physics step is 0.005 seconds.
            physics_engine: Isaac Gym physics engine identifier.
            sim_device: Simulation device string, such as ``cuda:0`` or ``cpu``.
            headless: Whether to create the simulation without a viewer.
            creation_properties: Optional saved mass/COM deltas [N,5] in kg/m and
                friction [N,1], applied and read back before prepare_sim. None keeps
                the configured creation randomization for a fresh training run.

        Raises:
            ValueError: Saved creation properties have invalid keys, shapes or values.

        Note:
            Stages only creation-lifetime physics; Runner restores learning and RNG.
        """
        self.creation_properties = creation_properties
        if creation_properties is not None:
            if set(creation_properties) != {"mass_com_delta", "friction"}:
                raise ValueError("Invalid V3 creation property fields")
            for key, shape in (("mass_com_delta", (cfg.env.num_envs, 5)), ("friction", (cfg.env.num_envs, 1))):
                if creation_properties[key].shape != shape or not torch.isfinite(creation_properties[key]).all():
                    raise ValueError(f"Invalid V3 creation property: {key}")
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        # The configured 20 s / 20 ms contract is 1000 policy steps despite float rounding.
        self.max_episode_length = int(round(self.max_episode_length_s / self.dt))
        self.completed_iterations = 0
        self.training_policy_steps = 0
        self.training_transitions = 0
        self.last_push_policy_step = 0
        self.evaluation_policy_step: Optional[int] = None
        self.evaluation_last_push_step = 0
        self.last_push_velocity_world = torch.zeros(self.num_envs, 2, device=self.device)
        self.push_generator: Optional[torch.Generator] = None
        if len(self.dof_names) != len(ACTION_JOINT_NAMES):
            raise ValueError(f"Expected 14 actuated V3 joints, found {len(self.dof_names)}")
        self.action_dof_indices = torch.tensor(
            [self.dof_names.index(name) for name in ACTION_JOINT_NAMES],
            device=self.device,
            dtype=torch.long,
        )
        self.position_dof_indices = torch.tensor(
            [self.dof_names.index(name) for name in POSITION_JOINT_NAMES],
            device=self.device,
            dtype=torch.long,
        )
        self.wheel_dof_indices = torch.tensor(
            [self.dof_names.index(name) for name in WHEEL_JOINT_NAMES],
            device=self.device,
            dtype=torch.long,
        )
        body_names = self.gym.get_actor_rigid_body_names(self.envs[0], self.actor_handles[0])
        self.base_body_index = body_names.index("base_link")
        self.tcp_body_index = body_names.index("arm_seg6")
        self.wheel_body_indices = torch.tensor(
            [body_names.index(name) for name in ("fl_Link", "fr_Link", "bl_Link", "br_Link")],
            device=self.device,
            dtype=torch.long,
        )
        self.collision_body_indices = torch.tensor(
            [body_names.index(name) for name in COLLISION_BODY_NAMES], device=self.device,
            dtype=torch.long,
        )
        position_limits = self.dof_pos_limits[self.position_dof_indices]
        position_midpoint = position_limits.mean(dim=1)
        soft_half_width = (position_limits[:, 1] - position_limits[:, 0]) * (
            0.5 * self.cfg.rewards.soft_dof_pos_limit
        )
        self.soft_position_limits = torch.stack(
            (position_midpoint - soft_half_width, position_midpoint + soft_half_width), dim=1
        )
        self.rigid_state = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))
        self.rigid_state = self.rigid_state.view(self.num_envs, self.num_bodies, 13)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.obs_history = torch.zeros(self.num_envs, 32, 58, device=self.device)
        self.critic_history = torch.zeros(self.num_envs, 3, 98, device=self.device)
        self.obs_pred = torch.zeros(self.num_envs, 12, device=self.device)
        self.privileged_obs_buf = torch.zeros(self.num_envs, 294, device=self.device)
        self.obs_buf = torch.zeros(self.num_envs, 1856, device=self.device)
        self.commands = torch.zeros(self.num_envs, 15, device=self.device)
        self.target_start = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_end = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_move_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.target_hold_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.target_elapsed_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.last_actions = torch.zeros(self.num_envs, 14, device=self.device)
        self.external_ee_force_y = torch.zeros(self.num_envs, 3, device=self.device)
        self.external_base_force_y = torch.zeros(self.num_envs, 3, device=self.device)
        self.gripper_force_kps = torch.full((self.num_envs, 3), 200.0, device=self.device)
        self.execution_scale_delta = torch.zeros(self.num_envs, 14, device=self.device)
        self.contact_valid = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._published_observations: Dict[str, torch.Tensor] = {}
        self.policy_trace_enabled = False
        self._initialize_episode_metrics()
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self._publish_observations(torch.arange(self.num_envs, device=self.device), initial=True)

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Apply one policy action and return the four-item UniFP environment result.

        Args:
            actions: Unnormalized policy actions with shape ``[num_envs, 14]``.

        Returns:
            A new observation dictionary, reward vector, integer done vector, and step info.

        Raises:
            ValueError: If the action tensor has an incompatible shape or non-finite values.
            RuntimeError: If the simulator produces non-finite state or observation values.
        """
        if actions.shape != (self.num_envs, 14):
            raise ValueError(f"Expected actions [{self.num_envs}, 14], got {tuple(actions.shape)}")
        if not torch.isfinite(actions).all():
            raise ValueError("Actions contain non-finite values")

        if torch.count_nonzero(self.commands[:, 9:15]) or torch.count_nonzero(self.external_ee_force_y) or torch.count_nonzero(self.external_base_force_y):
            raise RuntimeError("V3 A-stage force channels must remain zero")
        base_push_applied = self._apply_scheduled_base_push()
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clamp(actions.to(self.device), -clip_actions, clip_actions)
        self.velocity_before_step = self.dof_vel.clone()
        if not self.headless:
            self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_v3_torques(self.actions)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.episode_length_buf += 1
        self._validate_simulation_state()
        self._update_base_state()
        failed = (self.rpy[:, 0].abs() > 0.8) | (self.rpy[:, 1].abs() > 1.0)
        base_contact = torch.linalg.vector_norm(self.contact_forces[:, self.base_body_index], dim=-1) > 1.0
        terminated = failed | base_contact
        time_outs = (self.episode_length_buf >= self.max_episode_length) & ~terminated
        dones = terminated | time_outs
        # Commands still describe the action interval that just ended. Settle it
        # before command resampling and before done environments lose their state.
        self.compute_reward()
        policy_trace = self._build_policy_trace(base_push_applied, terminated, time_outs)
        done_ids = dones.nonzero(as_tuple=False).flatten()
        episode_records = self._accumulate_episode_metrics(done_ids, terminated)
        self.last_actions.copy_(self.actions)
        self._advance_task_commands(torch.arange(self.num_envs, device=self.device))
        _, terminal_frame, _ = self._build_frames(done_ids)
        terminal_privileged = torch.cat((self.critic_history[done_ids, 1:], terminal_frame[:, None]), dim=1)
        terminal_privileged = terminal_privileged.reshape(done_ids.numel(), 294).clone()
        self.reset_buf.copy_(dones.long())
        if done_ids.numel():
            self.reset_idx(done_ids)
        self._publish_observations(torch.arange(self.num_envs, device=self.device))
        infos: Dict[str, Any] = {
            "terminated": terminated.clone(),
            "time_outs": time_outs.clone(),
            "terminal_env_ids": done_ids.clone(),
            "terminal_privileged_obs": terminal_privileged,
            "episode_records": episode_records,
            "base_push_applied": base_push_applied,
            "policy_trace": policy_trace,
        }
        return self._published_observations, self.rew_buf.clone(), dones.long(), infos

    def _build_policy_trace(
        self, base_push_applied: bool, terminated: torch.Tensor, time_outs: torch.Tensor
    ) -> List[Dict[str, Any]]:
        """Capture physical policy-boundary data only during explicit evaluation.

        Args:
            base_push_applied: Whether this policy boundary injected a base push.
            terminated: Boolean [N] physical-failure flags before any reset.
            time_outs: Boolean [N] pure-timeout flags before any reset.

        Returns:
            One CPU JSON-compatible trace dictionary per environment, or an empty list.

        Note:
            Disabled during training to avoid device synchronization and trace storage.
        """
        if not self.policy_trace_enabled:
            return []

        # Snapshot terminal states before reset_idx replaces completed environments.
        tcp_world = self._tcp_position()
        target_world = self._geometric_target_world()
        physical_errors = self._physical_tracking_errors()
        tensors = {
            "commands": self.commands[:, :3],
            "ee_force_command": self.commands[:, 9:12],
            "base_force_command": self.commands[:, 12:15],
            "external_ee_force": self.external_ee_force_y,
            "external_base_force": self.external_base_force_y,
            "mass_com_delta": self.mass_com_delta,
            "friction": self.friction,
            "execution_scale_delta": self.execution_scale_delta,
            "target_spherical": self.commands[:, 3:6],
            "target_world_m": target_world,
            "tcp_world_m": tcp_world,
            "base_position_world_m": self.root_states[:, :3],
            "base_orientation_world_xyzw": self.root_states[:, 3:7],
            "base_linear_velocity_body_m_s": self.base_lin_vel,
            "base_angular_velocity_body_rad_s": self.base_ang_vel,
            "state_estimate_labels_scaled": self.obs_pred,
            "physical_errors": physical_errors,
            "actions": self.actions,
            "raw_rewards": self.reward_raw,
        }
        cpu_tensors = {name: value.detach().cpu().tolist() for name, value in tensors.items()}
        steps = self.episode_length_buf.detach().cpu().tolist()
        push_velocity = self.last_push_velocity_world.detach().cpu().tolist()
        terminated_values = terminated.detach().cpu().tolist()
        timeout_values = time_outs.detach().cpu().tolist()
        target_elapsed = self.target_elapsed_steps.detach().cpu().tolist()
        target_move = self.target_move_steps.detach().cpu().tolist()
        reward_names = self.reward_names
        trace_rows = []
        for env_id in range(self.num_envs):
            trace_rows.append({
                "env_id": env_id,
                "episode_step": steps[env_id],
                "time_s": steps[env_id] * self.dt,
                "policy_input_time_s": max(steps[env_id] - 1, 0) * self.dt,
                "target_segment_elapsed_steps": target_elapsed[env_id],
                "target_segment_move_steps": target_move[env_id],
                "target_segment_complete": target_elapsed[env_id] >= target_move[env_id],
                "commands_body": cpu_tensors["commands"][env_id],
                "end_effector_force_command_yaw_n": cpu_tensors["ee_force_command"][env_id],
                "base_force_command_yaw_n": cpu_tensors["base_force_command"][env_id],
                "external_end_effector_force_yaw_n": cpu_tensors["external_ee_force"][env_id],
                "external_base_force_yaw_n": cpu_tensors["external_base_force"][env_id],
                "target_spherical_m_rad_rad": cpu_tensors["target_spherical"][env_id],
                "target_world_m": cpu_tensors["target_world_m"][env_id],
                "tcp_world_m": cpu_tensors["tcp_world_m"][env_id],
                "base_position_world_m": cpu_tensors["base_position_world_m"][env_id],
                "base_orientation_world_xyzw": cpu_tensors["base_orientation_world_xyzw"][env_id],
                "base_linear_velocity_body_m_s": cpu_tensors["base_linear_velocity_body_m_s"][env_id],
                "base_angular_velocity_body_rad_s": cpu_tensors["base_angular_velocity_body_rad_s"][env_id],
                "domain_randomization": {
                    "base_mass_delta_kg": cpu_tensors["mass_com_delta"][env_id][0],
                    "base_com_delta_local_m": cpu_tensors["mass_com_delta"][env_id][1:4],
                    "tool_mass_delta_kg": cpu_tensors["mass_com_delta"][env_id][4],
                    "friction_coefficient": cpu_tensors["friction"][env_id][0],
                    "execution_scale_delta": cpu_tensors["execution_scale_delta"][env_id],
                },
                "state_estimate_labels_scaled": cpu_tensors["state_estimate_labels_scaled"][env_id],
                "physical_errors": dict(zip(ERROR_METRIC_NAMES, cpu_tensors["physical_errors"][env_id])),
                "action": cpu_tensors["actions"][env_id],
                "reward": float(self.rew_buf[env_id]),
                "raw_rewards": dict(zip(reward_names, cpu_tensors["raw_rewards"][env_id])),
                "base_push_applied": base_push_applied,
                "base_push_velocity_world_m_s": push_velocity[env_id] if base_push_applied else None,
                "terminated": bool(terminated_values[env_id]),
                "timeout": bool(timeout_values[env_id]),
                "done_reason": "physical_failure" if terminated_values[env_id]
                else "timeout" if timeout_values[env_id] else None,
            })
        return trace_rows

    def _initialize_episode_metrics(self) -> None:
        """Allocate bounded A-stage error histories and per-environment reward sums.

        Note:
            Four float32 samples per policy boundary preserve exact episode P95.
            A-stage geometric and compensated errors share these samples.
        """
        self.episode_metric_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_raw_sums = torch.zeros(self.num_envs, len(REWARD_NAMES), device=self.device)
        self.episode_reward_returns = torch.zeros(self.num_envs, device=self.device)
        self.episode_error_samples = torch.zeros(
            self.num_envs, self.max_episode_length, len(ERROR_METRIC_NAMES), device=self.device
        )
        # Only one step's completed records are staged; Runner drains before the next step.
        self.pending_episode_records: List[Dict[str, Any]] = []

    def _accumulate_episode_metrics(
        self, done_ids: torch.Tensor, terminated: torch.Tensor
    ) -> List[Dict[str, Any]]:
        """Include the last transition and finalize each completed episode separately.

        Args:
            done_ids: Long [K] indices of physical failures or pure timeouts.
            terminated: Bool [N] mask; physical failure takes precedence over timeout.

        Returns:
            JSON-compatible records containing returns, raw means and exact error P95.

        Note:
            Stages completed records before command sampling/reset can fail; does not
            clear any accumulator. PPO fills timeout bonuses in ticket 05.
        """
        env_ids = torch.arange(self.num_envs, device=self.device)
        physical_errors = self._physical_tracking_errors()
        self.episode_error_samples[env_ids, self.episode_metric_steps] = physical_errors
        self.episode_metric_steps += 1
        self.episode_raw_sums += self.reward_raw
        self.episode_reward_returns += self.rew_buf
        records = []
        # Only completed episodes move to CPU; no per-step trace is written.
        for env_id in done_ids.tolist():
            steps = int(self.episode_metric_steps[env_id])
            duration_s = steps * self.dt
            samples = self.episode_error_samples[env_id, :steps]
            error_means = samples.mean(dim=0).tolist()
            error_p95 = torch.quantile(samples, 0.95, dim=0).tolist()
            errors = {name: {"mean": error_means[index], "p95": error_p95[index]}
                      for index, name in enumerate(ERROR_METRIC_NAMES)}
            for norm in ("l1", "l2"):
                errors[f"tcp_compensated_{norm}_m"] = dict(errors[f"tcp_geometric_{norm}_m"])
            reward_return = float(self.episode_reward_returns[env_id])
            raw_means = (self.episode_raw_sums[env_id] / steps).tolist()
            records.append({
                "completed_iterations": self.completed_iterations,
                "training_policy_steps": self.training_policy_steps,
                "training_transitions": self.training_transitions,
                "env_id": env_id, "steps": steps, "duration_s": duration_s,
                "end_type": "physical_failure" if bool(terminated[env_id]) else "timeout",
                "reward_return": reward_return, "reward_rate": reward_return / duration_s,
                "raw_reward_means": dict(zip(self.reward_names, raw_means)),
                "errors": errors, "timeout_bootstrap_bonus": None,
            })
        self.pending_episode_records = records
        return records

    def _clear_episode_metrics(self, env_ids: torch.Tensor) -> None:
        """Clear only episode accumulators for the environments being reset.

        Args:
            env_ids: Long [K] environment indices, including explicit manual resets.

        Note:
            Old error samples need no clearing; the new step count bounds all reads.
        """
        self.episode_metric_steps[env_ids] = 0
        self.episode_raw_sums[env_ids] = 0.0
        self.episode_reward_returns[env_ids] = 0.0

    def _prepare_reward_function(self) -> None:
        """Register the 23 A08 terms with immutable configured weights.

        Raises:
            ValueError: Reward names, weights, scales or soft boundaries are invalid.

        Note:
            Builds weights multiplied once by the policy period; no physical state changes.
        """
        configured_weights = class_to_dict(self.cfg.rewards.scales)
        enabled_names = {name for name, weight in configured_weights.items() if weight != 0.0}
        if enabled_names != set(REWARD_NAMES):
            raise ValueError("V3 reward configuration must enable exactly the 23 A08 terms")
        if not all(math.isfinite(weight) for weight in configured_weights.values()):
            raise ValueError("V3 reward weights must be finite")
        rewards = self.cfg.rewards
        if any(not math.isfinite(value) or value <= 0.0 for value in (
            rewards.tracking_sigma, rewards.tracking_ee_sigma,
            rewards.collision_force_threshold, rewards.base_height_target,
        )):
            raise ValueError("Reward scales and physical thresholds must be finite and positive")
        if not 0.0 < rewards.soft_dof_pos_limit <= 1.0 or not 0.0 < rewards.soft_torque_limit < 1.0:
            raise ValueError("Reward soft-limit ratios are invalid")
        if rewards.only_positive_rewards:
            raise ValueError("V3 A08 requires signed environment rewards")
        self.reward_names = REWARD_NAMES
        self.configured_reward_weights = {name: configured_weights[name] for name in REWARD_NAMES}
        self.reward_scales = {name: weight * self.dt for name, weight in self.configured_reward_weights.items()}
        self.reward_functions = [getattr(self, "_reward_" + name) for name in REWARD_NAMES]
        self.reward_raw = torch.zeros(self.num_envs, len(REWARD_NAMES), device=self.device)

    def compute_reward(self) -> None:
        """Evaluate raw physical terms and sum each signed 20 ms contribution once.

        Raises:
            RuntimeError: A raw term or the total reward is non-finite.

        Note:
            Reads current transition state before command advancement or reset.
            Updates raw diagnostics and the environment reward, without PPO bootstrap.
        """
        self.reward_raw = torch.stack([function() for function in self.reward_functions], dim=1)
        weights = self.reward_raw.new_tensor([self.reward_scales[name] for name in self.reward_names])
        self.rew_buf.copy_((self.reward_raw * weights).sum(dim=1))
        if not torch.isfinite(self.reward_raw).all() or not torch.isfinite(self.rew_buf).all():
            raise RuntimeError("V3 reward contains non-finite physical terms")

    def _geometric_target_world(self) -> torch.Tensor:
        """Convert current spherical commands into yaw-aligned world TCP targets.

        Returns:
            World positions [N,3] in meters, using the fixed nominal-height arm center.
        """
        radius, elevation, azimuth = self.commands[:, 3:6].unbind(dim=1)
        yaw = self.rpy[:, 2]
        horizontal_radius = radius * torch.cos(elevation)
        target = self.root_states[:, :3].clone()
        target[:, 0] += ARM_MOUNT_X * torch.cos(yaw) + horizontal_radius * torch.cos(azimuth + yaw)
        target[:, 1] += ARM_MOUNT_X * torch.sin(yaw) + horizontal_radius * torch.sin(azimuth + yaw)
        target[:, 2] = (self.env_origins[:, 2] + self.cfg.rewards.base_height_target
                        + ARM_MOUNT_Z + radius * torch.sin(elevation))
        return target

    def _physical_tracking_errors(self) -> torch.Tensor:
        """Measure A-stage TCP and body-velocity errors independently of rewards.

        Returns:
            [N,4] errors: TCP L1/L2 (m), planar speed L2 (m/s), yaw-rate absolute (rad/s).
        """
        tcp_error = self._tcp_position() - self._geometric_target_world()
        return torch.stack((
            tcp_error.abs().sum(dim=1), torch.linalg.vector_norm(tcp_error, dim=1),
            torch.linalg.vector_norm(self.base_lin_vel[:, :2] - self.commands[:, :2], dim=1),
            (self.base_ang_vel[:, 2] - self.commands[:, 2]).abs(),
        ), dim=1)

    def _compensated_target_world(self) -> torch.Tensor:
        """Compute the official virtual-stiffness target using actual injected forces.

        Returns:
            World TCP targets [N,3] in meters. Both force inputs are zero in A,
            so this is exactly the geometric target; contact forces never enter it.
        """
        force_yaw = self.external_ee_force_y + self.commands[:, 9:12]
        yaw = self.rpy[:, 2]
        force_world = torch.stack((
            torch.cos(yaw) * force_yaw[:, 0] - torch.sin(yaw) * force_yaw[:, 1],
            torch.sin(yaw) * force_yaw[:, 0] + torch.cos(yaw) * force_yaw[:, 1],
            force_yaw[:, 2],
        ), dim=1)
        return self._geometric_target_world() + force_world / self.gripper_force_kps

    def _reward_tracking_ee_force_world(self) -> torch.Tensor:
        """Return the zero-force compensated TCP tracking exponential.

        Returns:
            Unweighted float32 values [N]; outer aggregation applies weight and dt.
        """
        return torch.exp(-2.0 * (self._tcp_position() - self._compensated_target_world()).abs().sum(dim=1) / self.cfg.rewards.tracking_ee_sigma)

    def _reward_tracking_lin_vel(self) -> torch.Tensor:
        """Return body-frame planar velocity tracking reward.

        Returns:
            Unweighted float32 values [N]; outer aggregation applies weight and dt.
        """
        return torch.exp(-(self.base_lin_vel[:, :2] - self.commands[:, :2]).square().sum(dim=1) / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self) -> torch.Tensor:
        """Return body-frame yaw-rate tracking reward.

        Returns:
            Unweighted float32 values [N]; outer aggregation applies weight and dt.
        """
        return torch.exp(-(self.base_ang_vel[:, 2] - self.commands[:, 2]).square() / self.cfg.rewards.tracking_sigma)

    def _reward_alive(self) -> torch.Tensor:
        """Return one for every real transition, including failure and timeout.

        Returns:
            Unweighted float32 values [N]; outer aggregation applies weight and dt.
        """
        return torch.ones(self.num_envs, device=self.device)

    def _reward_torques_steering(self) -> torch.Tensor:
        """Return the final submitted torque square sum in this actuator group.

        Returns:
            Unweighted float32 values [N]; outer aggregation applies weight and dt.
        """
        return self.torques[:, self.action_dof_indices[0:4]].square().sum(dim=1)

    def _reward_torques_drive(self) -> torch.Tensor:
        """Return the final submitted torque square sum in this actuator group.

        Returns:
            Unweighted float32 values [N]; outer aggregation applies weight and dt.
        """
        return self.torques[:, self.action_dof_indices[4:8]].square().sum(dim=1)

    def _reward_torques_arm(self) -> torch.Tensor:
        """Return the final submitted torque square sum in this actuator group.

        Returns:
            Unweighted float32 values [N]; outer aggregation applies weight and dt.
        """
        return self.torques[:, self.action_dof_indices[8:14]].square().sum(dim=1)

    def _reward_action_rate_steering(self) -> torch.Tensor:
        """Return the clipped dimensionless action difference square sum.

        Returns:
            Unweighted float32 values [N]; outer aggregation applies weight and dt.
        """
        return (self.actions[:, 0:4] - self.last_actions[:, 0:4]).square().sum(dim=1)

    def _reward_action_rate_drive(self) -> torch.Tensor:
        """Return the clipped dimensionless action difference square sum.

        Returns:
            Unweighted float32 values [N]; outer aggregation applies weight and dt.
        """
        return (self.actions[:, 4:8] - self.last_actions[:, 4:8]).square().sum(dim=1)

    def _reward_action_rate_arm(self) -> torch.Tensor:
        """Return the clipped dimensionless action difference square sum.

        Returns:
            Unweighted float32 values [N]; outer aggregation applies weight and dt.
        """
        return (self.actions[:, 8:14] - self.last_actions[:, 8:14]).square().sum(dim=1)

    def _reward_dof_vel_steering(self) -> torch.Tensor:
        """Return absolute joint-speed square sum for position-controlled joints.

        Returns:
            Unweighted float32 values [N]; outer aggregation applies weight and dt.
        """
        return self.dof_vel[:, self.action_dof_indices[0:4]].square().sum(dim=1)

    def _reward_dof_vel_arm(self) -> torch.Tensor:
        """Return absolute joint-speed square sum for position-controlled joints.

        Returns:
            Unweighted float32 values [N]; outer aggregation applies weight and dt.
        """
        return self.dof_vel[:, self.action_dof_indices[8:14]].square().sum(dim=1)

    def _reward_dof_acc_steering(self) -> torch.Tensor:
        """Return squared accelerations from the two real policy-boundary velocities.

        Returns:
            Unweighted float32 values [N]; outer aggregation applies weight and dt.
        """
        return ((self.dof_vel[:, self.action_dof_indices[0:4]] - self.velocity_before_step[:, self.action_dof_indices[0:4]]) / self.dt).square().sum(dim=1)

    def _reward_dof_acc_drive(self) -> torch.Tensor:
        """Return squared accelerations from the two real policy-boundary velocities.

        Returns:
            Unweighted float32 values [N]; outer aggregation applies weight and dt.
        """
        return ((self.dof_vel[:, self.action_dof_indices[4:8]] - self.velocity_before_step[:, self.action_dof_indices[4:8]]) / self.dt).square().sum(dim=1)

    def _reward_dof_acc_arm(self) -> torch.Tensor:
        """Return squared accelerations from the two real policy-boundary velocities.

        Returns:
            Unweighted float32 values [N]; outer aggregation applies weight and dt.
        """
        return ((self.dof_vel[:, self.action_dof_indices[8:14]] - self.velocity_before_step[:, self.action_dof_indices[8:14]]) / self.dt).square().sum(dim=1)

    def _reward_torque_limits_steering(self) -> torch.Tensor:
        """Return linear torque excess above the group effort soft threshold.

        Returns:
            Unweighted float32 values [N]; outer aggregation applies weight and dt.
        """
        return (self.torques[:, self.action_dof_indices[0:4]].abs() - self.cfg.rewards.soft_torque_limit * self.torque_limits[self.action_dof_indices[0:4]]).clamp_min(0.0).sum(dim=1)

    def _reward_torque_limits_drive(self) -> torch.Tensor:
        """Return linear torque excess above the group effort soft threshold.

        Returns:
            Unweighted float32 values [N]; outer aggregation applies weight and dt.
        """
        return (self.torques[:, self.action_dof_indices[4:8]].abs() - self.cfg.rewards.soft_torque_limit * self.torque_limits[self.action_dof_indices[4:8]]).clamp_min(0.0).sum(dim=1)

    def _reward_torque_limits_arm(self) -> torch.Tensor:
        """Return linear torque excess above the group effort soft threshold.

        Returns:
            Unweighted float32 values [N]; outer aggregation applies weight and dt.
        """
        return (self.torques[:, self.action_dof_indices[8:14]].abs() - self.cfg.rewards.soft_torque_limit * self.torque_limits[self.action_dof_indices[8:14]]).clamp_min(0.0).sum(dim=1)

    def _reward_dof_pos_limits(self) -> torch.Tensor:
        """Return linear angular excess beyond the ten finite-joint soft limits.

        Returns:
            Unweighted float32 values [N]; outer aggregation applies weight and dt.
        """
        return (self.soft_position_limits[:, 0] - self.dof_pos[:, self.position_dof_indices]).clamp_min(0.0).sum(dim=1) + (self.dof_pos[:, self.position_dof_indices] - self.soft_position_limits[:, 1]).clamp_min(0.0).sum(dim=1)

    def _reward_lin_vel_z(self) -> torch.Tensor:
        """Return squared vertical velocity in the full body frame.

        Returns:
            Unweighted float32 values [N]; outer aggregation applies weight and dt.
        """
        return self.base_lin_vel[:, 2].square()

    def _reward_ang_vel_xy(self) -> torch.Tensor:
        """Return squared roll/pitch angular rates in the full body frame.

        Returns:
            Unweighted float32 values [N]; outer aggregation applies weight and dt.
        """
        return self.base_ang_vel[:, :2].square().sum(dim=1)

    def _reward_base_height(self) -> torch.Tensor:
        """Return squared base height error relative to each ground origin.

        Returns:
            Unweighted float32 values [N]; outer aggregation applies weight and dt.
        """
        return (self.root_states[:, 2] - self.env_origins[:, 2] - self.cfg.rewards.base_height_target).square()

    def _reward_collision(self) -> torch.Tensor:
        """Count non-wheel bodies above the strict contact-force threshold.

        Returns:
            Unweighted float32 values [N]; outer aggregation applies weight and dt.
        """
        return (torch.linalg.vector_norm(self.contact_forces[:, self.collision_body_indices], dim=-1) > self.cfg.rewards.collision_force_threshold).float().sum(dim=1)

    def set_training_progress(
        self, completed_iterations: int, training_policy_steps: int
    ) -> None:
        """Publish runner-owned learning progress without advancing the simulator.

        Args:
            completed_iterations: Number of fully committed PPO iterations.
            training_policy_steps: Number of synchronous policy steps consumed by
                training, including the current rollout position.

        Raises:
            ValueError: A counter is not a non-negative integer, contradicts the
                24-step rollout boundary, or moves backward.

        Note:
            This method only stores progress values. It does not step physics,
            sample randomness, update episode counters, or alter observations.
        """
        if type(completed_iterations) is not int or type(training_policy_steps) is not int:
            raise ValueError("Training progress counters must be Python integers")
        if completed_iterations < 0 or training_policy_steps < 0:
            raise ValueError("Training progress counters must be non-negative")
        lower_bound = completed_iterations * TRAINING_POLICY_STEPS_PER_ITERATION
        upper_bound = (completed_iterations + 1) * TRAINING_POLICY_STEPS_PER_ITERATION
        if not lower_bound <= training_policy_steps <= upper_bound:
            raise ValueError(
                "training_policy_steps must fall within the current 24-step rollout"
            )
        if (
            completed_iterations < self.completed_iterations
            or training_policy_steps < self.training_policy_steps
        ):
            raise ValueError("Training progress cannot move backward")

        self.completed_iterations = completed_iterations
        self.training_policy_steps = training_policy_steps
        self.training_transitions = self.num_envs * training_policy_steps

    def _validate_simulation_state(self) -> None:
        """Stop the run if refreshed core Gym state contains non-finite values.

        Raises:
            RuntimeError: A root, DOF, rigid-body, contact, or torque tensor is non-finite.
        """
        state_tensors = (
            self.root_states,
            self.dof_state,
            self.rigid_state,
            self.contact_forces,
            self.torques,
        )
        if not all(torch.isfinite(state).all() for state in state_tensors):
            raise RuntimeError("Isaac Gym produced a non-finite V3 simulation state")

    def reset(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Reset all environments without advancing the simulator.

        Returns:
            The initial observation package and an empty reset information dictionary.
        """
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.reset_idx(env_ids)
        self._publish_observations(env_ids, initial=True)
        return self._published_observations, {}

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        """Reset only the selected environments and clear their episode histories.

        Args:
            env_ids: Long tensor of environment indices with shape ``[K]``.
        """
        if env_ids.numel() == 0:
            return
        self._clear_episode_metrics(env_ids)
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        self.root_states[env_ids, 7:13] = 0.0
        self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_vel[env_ids] = 0.0
        actor_ids = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_states), gymtorch.unwrap_tensor(actor_ids), len(actor_ids)
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(actor_ids), len(actor_ids)
        )
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self._update_base_state()
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        if self.cfg.domain_rand.randomize_execution_scale:
            lower, upper = self.cfg.domain_rand.execution_scale_range
            self.execution_scale_delta[env_ids] = lower + torch.rand(env_ids.numel(), 14, device=self.device) * (upper - lower) - 1.0
        else:
            self.execution_scale_delta[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.torques[env_ids] = 0.0
        self.obs_history[env_ids] = 0.0
        self.critic_history[env_ids] = 0.0
        self.obs_pred[env_ids] = 0.0
        self.commands[env_ids] = 0.0
        # Refreshing after indexed root/DOF writes gives this reset's TCP state
        # in Preview4; the zero-velocity command must start at that same state.
        reset_tcp = self._tcp_spherical_position()[env_ids]
        self._validate_reset_state(env_ids, reset_tcp)
        self._initialize_target_segments(env_ids, reset_tcp)
        self.commands[env_ids, :3] = self._sample_base_velocity_commands(env_ids.numel())
        self.commands[env_ids, 3:6] = reset_tcp
        self.external_ee_force_y[env_ids] = 0.0
        self.external_base_force_y[env_ids] = 0.0
        self.contact_valid[env_ids] = False

    def _validate_reset_state(self, env_ids: torch.Tensor, tcp_spherical: torch.Tensor) -> None:
        """Check reset readback consistency and the candidate TCP ground margin.

        Args:
            env_ids: Long tensor of reset environment indices with shape ``[K]``.
            tcp_spherical: Reset TCP positions ``[K, 3]`` in ``[m, rad, rad]``.

        Raises:
            RuntimeError: Reset state is non-finite or its TCP violates the candidate ground margin.

        Note:
            This is a runtime state-consistency check, not a full collision or reachability certificate.
        """
        root_state = self.root_states[env_ids]
        dof_state = self.dof_state.view(self.num_envs, self.num_dof, 2)[env_ids]
        tcp_world = self._tcp_position()[env_ids]
        if not torch.isfinite(root_state).all() or not torch.isfinite(dof_state).all():
            raise RuntimeError(f"Non-finite root/DOF state after reset for env_ids={env_ids.tolist()}")
        if not torch.isfinite(tcp_spherical).all() or not torch.isfinite(tcp_world).all():
            raise RuntimeError(f"Non-finite TCP state after reset for env_ids={env_ids.tolist()}")
        min_height = self.env_origins[env_ids, 2] + self.cfg.goal_ee.min_tcp_height
        if torch.any(tcp_world[:, 2] < min_height):
            raise RuntimeError(f"Reset TCP below candidate ground margin for env_ids={env_ids.tolist()}")

    def _sample_base_velocity_commands(self, count: int) -> torch.Tensor:
        """Sample body-frame planar and yaw-rate commands for reset environments.

        Args:
            count: Number of environment command rows to sample.

        Returns:
            Uniformly sampled command tensor with shape ``[count, 3]`` in m/s and rad/s.
        """
        ranges = self.cfg.commands.ranges
        lower = torch.tensor(
            (ranges.lin_vel_x[0], ranges.lin_vel_y[0], ranges.ang_vel_yaw[0]),
            device=self.device,
        )
        upper = torch.tensor(
            (ranges.lin_vel_x[1], ranges.lin_vel_y[1], ranges.ang_vel_yaw[1]),
            device=self.device,
        )
        return lower + torch.rand(count, 3, device=self.device) * (upper - lower)

    def _initialize_target_segments(self, env_ids: torch.Tensor, start: torch.Tensor) -> None:
        """Initialize per-environment target segments from reset TCP positions.

        Args:
            env_ids: Long tensor of reset environment indices with shape ``[K]``.
            start: Actual reset TCP spherical positions with shape ``[K, 3]``.

        Raises:
            RuntimeError: No candidate endpoint passes the configured workspace and ground checks.
        """
        self.target_start[env_ids] = start
        self.target_elapsed_steps[env_ids] = 0
        pending = torch.ones(env_ids.numel(), device=self.device, dtype=torch.bool)
        accepted_end = torch.zeros_like(start)
        accepted_move_steps = torch.zeros(env_ids.numel(), device=self.device, dtype=torch.long)
        accepted_hold_steps = torch.zeros_like(accepted_move_steps)
        rejected_ground = torch.zeros_like(accepted_move_steps)
        for _ in range(self.cfg.goal_ee.max_resamples):
            local_ids = pending.nonzero(as_tuple=False).flatten()
            if local_ids.numel() == 0:
                break
            candidates = self._sample_goal_candidates(local_ids.numel())
            move_steps = self._sample_duration_steps(self.cfg.goal_ee.move_time, local_ids.numel())
            hold_steps = self._sample_duration_steps(self.cfg.goal_ee.hold_time, local_ids.numel())
            valid, ground_rejected = self._screen_target_paths(
                env_ids[local_ids], start[local_ids], candidates
            )
            rejected_ground[local_ids] += ground_rejected.long()
            accepted_local = local_ids[valid]
            accepted_end[accepted_local] = candidates[valid]
            accepted_move_steps[accepted_local] = move_steps[valid]
            accepted_hold_steps[accepted_local] = hold_steps[valid]
            pending[accepted_local] = False
        if pending.any():
            failed_envs = env_ids[pending]
            raise RuntimeError(
                "A08 target screening exhausted for "
                f"env_ids={failed_envs.tolist()}, ground_rejections={rejected_ground[pending].tolist()}"
            )
        self.target_end[env_ids] = accepted_end
        self.target_move_steps[env_ids] = accepted_move_steps
        self.target_hold_steps[env_ids] = accepted_hold_steps
        self.commands[env_ids, 3:6] = start

    def _sample_goal_candidates(self, count: int) -> torch.Tensor:
        """Sample independent spherical goal components from the A08 candidate domain.

        Args:
            count: Number of candidate endpoints to sample.

        Returns:
            Candidate spherical positions ``[count, 3]`` ordered as ``[m, rad, rad]``.
        """
        ranges = self.cfg.goal_ee
        limits = (ranges.radius, ranges.elevation, ranges.azimuth)
        values = [
            lower + torch.rand(count, device=self.device) * (upper - lower)
            for lower, upper in limits
        ]
        return torch.stack(values, dim=-1)

    def _sample_duration_steps(self, duration_range: Any, count: int) -> torch.Tensor:
        """Sample a duration uniformly and round up to policy-step boundaries.

        Args:
            duration_range: Two-element duration range in seconds.
            count: Number of durations to sample.

        Returns:
            Positive integer policy-step counts with shape ``[count]``.
        """
        dt = self.dt
        duration = duration_range[0] + torch.rand(count, device=self.device) * (
            duration_range[1] - duration_range[0]
        )
        return torch.ceil(duration / dt).long()

    def _screen_target_paths(
        self,
        env_ids: torch.Tensor,
        start: torch.Tensor,
        end: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Screen ten Cartesian TCP path samples against finite and ground constraints.

        Args:
            env_ids: Environment indices with shape ``[K]``.
            start: Segment starts ``[K, 3]`` in spherical meters/radians.
            end: Candidate endpoints ``[K, 3]`` in spherical meters/radians.

        Returns:
            A validity mask and per-candidate ground-rejection mask, each with shape ``[K]``.

        Note:
            Uses the official B2-Z1 sample count with V3's TCP ground-clearance rule.
            This check does not claim full arm collision or dynamic reachability validation.
        """
        count = env_ids.numel()
        sample_progress = torch.linspace(
            0.0,
            1.0,
            self.cfg.goal_ee.num_path_check_samples,
            device=self.device,
        )[None, :]
        progress = sample_progress.expand(count, -1)
        azimuth_delta = torch.atan2(
            torch.sin(end[:, 2] - start[:, 2]), torch.cos(end[:, 2] - start[:, 2])
        )
        path = start[:, None, :] + progress[..., None] * (end - start)[:, None, :]
        path[:, :, 2] = start[:, None, 2] + progress * azimuth_delta[:, None]
        tcp_cart_z = path[:, :, 0] * torch.sin(path[:, :, 1])
        world_z = (
            self.env_origins[env_ids, 2, None]
            + self.cfg.rewards.base_height_target
            + ARM_MOUNT_Z
            + tcp_cart_z
        )
        min_height = self.env_origins[env_ids, 2, None] + self.cfg.goal_ee.min_tcp_height
        ground_rejected = torch.any(world_z < min_height, dim=1)
        finite = torch.isfinite(path).all(dim=-1) & (path[:, :, 0] > 1.0e-6)
        valid = (~ground_rejected) & torch.all(finite, dim=1)
        return valid, ground_rejected

    def _advance_task_commands(self, env_ids: torch.Tensor) -> None:
        """Advance goal interpolation and resample base velocity commands at 20 ms boundaries.

        Args:
            env_ids: Long tensor of environment indices with shape ``[K]``.

        Raises:
            RuntimeError: A newly sampled segment exhausts the A08 candidate screening limit.
        """
        if env_ids.numel() == 0:
            return
        self.target_elapsed_steps[env_ids] += 1
        hold_done = self.target_elapsed_steps[env_ids] >= (
            self.target_move_steps[env_ids] + self.target_hold_steps[env_ids]
        )
        continuing_segments = env_ids[~hold_done]
        if continuing_segments.numel():
            elapsed = self.target_elapsed_steps[continuing_segments].float()
            alpha = torch.clamp(elapsed / self.target_move_steps[continuing_segments], max=1.0)
            azimuth_delta = torch.atan2(
                torch.sin(self.target_end[continuing_segments, 2] - self.target_start[continuing_segments, 2]),
                torch.cos(self.target_end[continuing_segments, 2] - self.target_start[continuing_segments, 2]),
            )
            current = self.target_start[continuing_segments] + alpha[:, None] * (
                self.target_end[continuing_segments] - self.target_start[continuing_segments]
            )
            current[:, 2] = self.target_start[continuing_segments, 2] + alpha * azimuth_delta
            self.commands[continuing_segments, 3:6] = current
        new_segment_ids = env_ids[hold_done]
        if new_segment_ids.numel():
            new_start = self.target_end[new_segment_ids].clone()
            self._initialize_target_segments(new_segment_ids, new_start)
        interval = int(round(self.cfg.commands.resampling_time / self.dt))
        resample_ids = env_ids[(self.episode_length_buf[env_ids] % interval) == 0]
        if resample_ids.numel():
            self.commands[resample_ids, :3] = self._sample_base_velocity_commands(resample_ids.numel())

    def get_observations(self) -> Dict[str, torch.Tensor]:
        """Return the most recently published observations without changing state.

        Returns:
            The immutable-by-convention three-key observation package.
        """
        return self._published_observations

    def _compute_v3_torques(self, actions: torch.Tensor) -> torch.Tensor:
        """Map policy-order actions to Gym-order explicit position and velocity feedback.

        Args:
            actions: Clipped dimensionless actions in the fixed 14-joint policy order.

        Returns:
            Requested joint torques in Gym DOF order, clipped by asset effort limits.

        Note:
            Per-episode execution coefficients scale positional residuals and wheel
            speed targets before their hard limits, never the final torque or gains.
        """
        scaled_actions = actions * (1.0 + self.execution_scale_delta)
        torque = torch.zeros_like(self.dof_pos)
        position_actions = actions[:, :4].new_zeros(self.num_envs, 10)
        position_actions[:, :4] = scaled_actions[:, :4]
        position_actions[:, 4:] = scaled_actions[:, 8:]
        position_target = (
            self.default_dof_pos[:, self.position_dof_indices]
            + self.cfg.control.action_scale * position_actions
        )
        position_target = torch.maximum(
            torch.minimum(position_target, self.dof_pos_limits[self.position_dof_indices, 1]),
            self.dof_pos_limits[self.position_dof_indices, 0],
        )
        position_torque = self.p_gains[self.position_dof_indices] * (
            position_target - self.dof_pos[:, self.position_dof_indices]
        ) - self.d_gains[self.position_dof_indices] * self.dof_vel[:, self.position_dof_indices]
        wheel_target = torch.clamp(
            self.cfg.control.wheel_action_scale * scaled_actions[:, 4:8],
            -self.cfg.control.wheel_velocity_limit,
            self.cfg.control.wheel_velocity_limit,
        )
        wheel_torque = self.d_gains[self.wheel_dof_indices] * (
            wheel_target - self.dof_vel[:, self.wheel_dof_indices]
        )
        torque[:, self.position_dof_indices] = position_torque
        torque[:, self.wheel_dof_indices] = wheel_torque
        return torch.maximum(torch.minimum(torque, self.torque_limits), -self.torque_limits)

    def _update_base_state(self) -> None:
        """Refresh base-frame velocities, Euler angles, and projected gravity."""
        self.base_quat = self.root_states[:, 3:7]
        self.rpy = self._quat_to_euler(self.base_quat)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        gravity = torch.zeros_like(self.base_lin_vel)
        gravity[:, 2] = -1.0
        self.projected_gravity = quat_rotate_inverse(self.base_quat, gravity)

    def _quat_to_euler(self, quaternion: torch.Tensor) -> torch.Tensor:
        """Convert Gym-order xyzw quaternions into roll, pitch, and yaw angles.

        Args:
            quaternion: World orientation tensor with shape ``[N, 4]`` in xyzw order.

        Returns:
            Euler angles in radians with shape ``[N, 3]``.
        """
        x, y, z, w = quaternion.unbind(dim=-1)
        roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x.square() + y.square()))
        pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))
        return torch.stack((roll, pitch, yaw), dim=-1)

    def _tcp_position(self) -> torch.Tensor:
        """Return the candidate contact-point position in world coordinates.

        Returns:
            Candidate TCP world positions with shape ``[N, 3]`` in meters.
        """
        body_state = self.rigid_state[:, self.tcp_body_index]
        local_offset = torch.tensor((0.0, 0.0, 0.07961206), device=self.device).expand(self.num_envs, 3)
        return body_state[:, :3] + quat_apply(body_state[:, 3:7], local_offset)

    def _tcp_spherical_position(self) -> torch.Tensor:
        """Express the candidate TCP position relative to the yaw-aligned V3 arm center.

        Returns:
            TCP spherical positions with shape ``[N, 3]`` as radius in meters and
            elevation and azimuth in radians.
        """
        self._update_base_state()
        tcp_world = self._tcp_position()
        yaw = self.rpy[:, 2]
        center = self.root_states[:, :3].clone()
        center[:, 0] += ARM_MOUNT_X * torch.cos(yaw)
        center[:, 1] += ARM_MOUNT_X * torch.sin(yaw)
        center[:, 2] = self.env_origins[:, 2] + self.cfg.rewards.base_height_target + ARM_MOUNT_Z
        delta = tcp_world - center
        local_x = torch.cos(yaw) * delta[:, 0] + torch.sin(yaw) * delta[:, 1]
        local_y = -torch.sin(yaw) * delta[:, 0] + torch.cos(yaw) * delta[:, 1]
        radius = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1.0e-6)
        elevation = torch.asin(torch.clamp(delta[:, 2] / radius, -1.0, 1.0))
        azimuth = torch.atan2(local_y, local_x)
        return torch.stack((radius, elevation, azimuth), dim=-1)

    def _build_frames(
        self, env_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build clean Actor, Critic, and auxiliary-label frames for selected environments.

        Args:
            env_ids: Long tensor of environment indices with shape ``[K]``.

        Returns:
            Actor frames ``[K, 58]``, Critic frames ``[K, 98]``, and labels ``[K, 12]``.
        """
        ids = env_ids
        self._update_base_state()
        target = self._tcp_spherical_position()
        ordered_pos = self.dof_pos[:, self.action_dof_indices]
        ordered_vel = self.dof_vel[:, self.action_dof_indices]
        pos_indices = torch.tensor(POSITION_OBS_INDICES, device=self.device)
        actor = torch.cat((
            self.rpy[:, :2], self.base_ang_vel,
            ordered_pos[:, pos_indices] - self.default_dof_pos[:, self.action_dof_indices][:, pos_indices],
            ordered_vel,
            self.last_actions,
            self.commands,
        ), dim=-1)
        labels = torch.cat((
            self.base_lin_vel * 2.0,
            target * torch.tensor((0.5, 1.0, 1.3), device=self.device),
            self.external_ee_force_y * 0.01,
            self.external_base_force_y * 0.01,
        ), dim=-1)
        actor_scaled = actor.clone()
        actor_scaled[:, 2:5] *= 0.25
        actor_scaled[:, 5:15] *= 1.0
        actor_scaled[:, 15:29] *= 0.05
        actor_scaled[:, 43:58] *= torch.tensor(COMMAND_SCALE, device=self.device)
        compensated_target = self.commands[:, 3:6]
        critic = torch.cat((
            labels,
            self.mass_com_delta,
            self.friction,
            self.execution_scale_delta,
            (self.contact_forces[:, self.wheel_body_indices, 2] > 5.0).float()
            * self.contact_valid.float()[:, None],
            self.projected_gravity,
            self.base_ang_vel * 0.25,
            actor[:, 5:15],
            ordered_vel * 0.05,
            self.last_actions,
            self.commands * torch.tensor(COMMAND_SCALE, device=self.device),
            compensated_target * torch.tensor((0.5, 1.0, 1.3), device=self.device),
        ), dim=-1)
        return actor_scaled[ids], critic[ids], labels[ids]

    def _publish_observations(self, env_ids: torch.Tensor, initial: bool = False) -> None:
        """Append one frame for selected environments and publish fresh flattened histories.

        Args:
            env_ids: Long tensor of environment indices whose histories advance.
            initial: Whether to suppress observation noise for the reset initial frame.
        """
        actor_frame, critic_frame, label_frame = self._build_frames(env_ids)
        actor_noise = torch.randn_like(actor_frame) * self._actor_noise_scale()
        if initial or not self.cfg.noise.add_noise:
            actor_noise.zero_()
        actor_frame = actor_frame + actor_noise
        if initial:
            self.obs_history[env_ids] = 0.0
            self.critic_history[env_ids] = 0.0
        else:
            self.obs_history[env_ids, :-1] = self.obs_history[env_ids, 1:].clone()
            self.critic_history[env_ids, :-1] = self.critic_history[env_ids, 1:].clone()
        self.obs_history[env_ids, -1] = actor_frame
        self.critic_history[env_ids, -1] = critic_frame
        self.obs_pred[env_ids] = label_frame
        self.contact_valid[env_ids] = True
        self.obs_buf = self.obs_history.reshape(self.num_envs, 1856).clone()
        self.privileged_obs_buf = self.critic_history.reshape(self.num_envs, 294).clone()
        if not torch.isfinite(self.obs_buf).all() or not torch.isfinite(self.privileged_obs_buf).all():
            raise RuntimeError("V3 observation contains non-finite values")
        self._published_observations = {
            "obs": torch.clamp(self.obs_buf, -100.0, 100.0),
            "privileged_obs": torch.clamp(self.privileged_obs_buf, -100.0, 100.0),
            "obs_pred": self.obs_pred.clone(),
        }

    def _actor_noise_scale(self) -> torch.Tensor:
        """Construct the single-frame Actor noise standard deviations after scaling.

        Returns:
            Standard deviations with shape ``[1, 58]`` in normalized observation units.
        """
        scale = torch.zeros(1, 58, device=self.device)
        scale[:, :2] = 0.05
        scale[:, 2:5] = 0.2 * 0.25
        scale[:, 5:15] = 0.01
        scale[:, 15:29] = 1.5 * 0.05
        return scale * self.cfg.noise.noise_level

    def _create_ground_plane(self) -> None:
        """Add the configured flat ground plane used during the A reproduction stage."""
        plane = gymapi.PlaneParams()
        plane.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane.static_friction = self.cfg.terrain.static_friction
        plane.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane)

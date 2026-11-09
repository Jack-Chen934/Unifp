"""Fixed-wall read-only V3 evaluation with explicit global actor/body mappings."""

from typing import Any, Dict, List
from itertools import product

import torch
from isaacgym import gymapi, gymtorch

from .legged_robot_v3 import V3PosForce


def classify_contact_interval(wall_n: torch.Tensor, fingers_n: torch.Tensor,
                              nonfinger_near_wall: bool, finger_near_wall: bool,
                              finger_near_ground: bool, measurement_valid: bool) -> str:
    """Conservatively classify an interval using force balance and geometric exclusion.

    Args:
        wall_n: Eight raw world-frame wall net force vectors [8,3] in N.
        fingers_n: Eight summed finger net force vectors [8,3] in N.
        nonfinger_near_wall: Any non-finger collision AABB enters the wall's combined
            20 mm contact-offset neighborhood during this interval.
        finger_near_wall: At least one finger enters that neighborhood.
        finger_near_ground: Any finger enters the 20 mm ground neighborhood.
        measurement_valid: All eight raw samples are present and finite.

    Returns:
        invalid, no_contact, ambiguous_contact or clean_expected_contact. Clean is
        consistency evidence, not a complete native contact-pair enumeration.
    """
    if not measurement_valid:
        return "invalid"
    wall_norm = torch.linalg.vector_norm(wall_n, dim=1)
    active = wall_norm > 1.
    if not active.any():
        return "no_contact"
    residual = torch.linalg.vector_norm(wall_n + fingers_n, dim=1)
    tolerance = torch.maximum(torch.ones_like(wall_norm), .1 * wall_norm)
    if (nonfinger_near_wall or not finger_near_wall or finger_near_ground
            or torch.any(residual > tolerance) or torch.any(wall_n[active, 0] <= 0.)):
        return "ambiguous_contact"
    return "clean_expected_contact"


class V3ContactEvaluation(V3PosForce):
    """Keep wall tensors separate from the original robot control/observation views."""

    def __init__(self, cfg: Any, sim_params: Any, physics_engine: Any, sim_device: str,
                 headless: bool, *, manifest: Dict[str, Any], zero_control: bool = False,
                 video: bool = False) -> None:
        """Build the fixed nominal scene before parent initialization/reset.

        Args:
            cfg: Isolated evaluation config with zero base commands and no push.
            sim_params: Actual native PhysX settings, with eight 2.5 ms steps.
            physics_engine: PhysX identifier.
            sim_device: Native physics device, independent of the tensor pipeline.
            headless: Omit the interactive viewer.
            manifest: Frozen geometry, newton amplitudes and contact thresholds.
            zero_control: Use the same trajectory with zero commanded force.
            video: Allocate graphics for representative episode recordings.

        Note:
            Creates actors and resets robots; never constructs an optimizer.
        """
        self.contact_manifest = manifest
        self.zero_control = zero_control
        self.video_enabled = video
        self.video_recorders = {}
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def create_sim(self) -> None:
        """Enable optional graphics and delegate the unchanged native simulator.

        Note:
            Allocates no viewer; cameras reuse the existing replay renderer.
        """
        if self.video_enabled:
            self.graphics_device_id = self.sim_device_id
        super().create_sim()

    def _create_envs(self) -> None:
        """Create each wall inside its robot callback, then query global indices.

        Raises:
            RuntimeError: Native actor/body maps are missing or overlap.

        Note:
            SDK requires all actors of an environment before the next environment.
            Shared base/B2 creation is untouched; all maps precede prepare_sim.
        """
        options = gymapi.AssetOptions()
        options.fix_base_link = True
        self.wall_asset = self.gym.create_box(self.sim, *self.contact_manifest["scene"]["wall_size_m"], options)
        self.wall_handles = []
        super()._create_envs()
        actor_indices, robot_bodies, wall_bodies = [], [], []
        for env_handle, robot_handle, wall_handle in zip(self.envs, self.actor_handles, self.wall_handles):
            actor_indices.append(self.gym.get_actor_index(env_handle, robot_handle, gymapi.DOMAIN_SIM))
            robot_bodies.append([self.gym.get_actor_rigid_body_index(env_handle, robot_handle, index, gymapi.DOMAIN_SIM)
                                 for index in range(self.num_bodies)])
            wall_bodies.append(self.gym.get_actor_rigid_body_index(env_handle, wall_handle, 0, gymapi.DOMAIN_SIM))
        self.robot_actor_indices = torch.tensor(actor_indices, device=self.device, dtype=torch.long)
        self.robot_body_indices = torch.tensor(robot_bodies, device=self.device, dtype=torch.long)
        self.wall_body_indices = torch.tensor(wall_bodies, device=self.device, dtype=torch.long)
        all_bodies = torch.cat((self.robot_body_indices.flatten(), self.wall_body_indices))
        if (len(actor_indices) != self.num_envs or torch.any(all_bodies < 0)
                or all_bodies.unique().numel() != self.num_envs * (self.num_bodies + 1)):
            raise RuntimeError("Invalid contact-scene native actor/body mapping")

    def _process_rigid_body_props(self, props: Any, env_id: int) -> Any:
        """Create this environment's wall while retaining robot body properties.

        Args:
            props: Imported robot rigid-body properties, returned unchanged.
            env_id: Current environment index before the next environment exists.

        Returns:
            Parent nominal robot body properties.

        Note:
            Creates one fixed wall and records its actor handle. World positions
            add the environment origin, never the robot's temporary spawn offset.
        """
        env_handle = self.gym.get_env(self.sim, env_id)
        position = self.env_origins[env_id].cpu().tolist()
        relative = self.contact_manifest["scene"]["wall_center_relative_m"]
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*(position[axis] + relative[axis] for axis in range(3)))
        wall_handle = self.gym.create_actor(env_handle, self.wall_asset, pose, "contact_wall", env_id, 0, 0)
        shape_properties = self.gym.get_actor_rigid_shape_properties(env_handle, wall_handle)
        for shape in shape_properties:
            shape.friction = self.contact_manifest["nominal_friction"]
            shape.restitution = 0.
            shape.contact_offset = self.contact_manifest["contact_offset_m"]
            shape.rest_offset = self.contact_manifest["rest_offset_m"]
        self.gym.set_actor_rigid_shape_properties(env_handle, wall_handle, shape_properties)
        self.wall_handles.append(wall_handle)
        return super()._process_rigid_body_props(props, env_id)

    def _bind_simulation_tensors(self) -> None:
        """Acquire full native tensors and allocate indexed robot-only caches.

        Note:
            Root is [2N,13], contact [N*(B+1),3], while the wall adds no DOFs.
            Advanced indexing returns copies; reset explicitly scatters them back.
        """
        self.raw_root_states = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))
        self.dof_state = gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim))
        self.raw_contact_forces = gymtorch.wrap_tensor(self.gym.acquire_net_contact_force_tensor(self.sim))
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.root_states = self.raw_root_states[self.robot_actor_indices].clone()
        self.contact_forces = self.raw_contact_forces[self.robot_body_indices].clone()
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.base_pos = self.root_states[:, :3]
        self.rpy = self._quat_to_euler(self.base_quat)

    def _bind_robot_rigid_state(self) -> None:
        """Bind full rigid-body state and populate persistent robot-only caches.

        Note:
            DOMAIN_SIM maps were acquired after all actors were created.
        """
        self.raw_rigid_state = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.rigid_state = self.raw_rigid_state[self.robot_body_indices].clone()
        self._sync_robot_state_tensors()
        names = self.gym.get_actor_rigid_body_names(self.envs[0], self.actor_handles[0])
        self.finger_local_indices = torch.tensor([names.index("gripper1"), names.index("gripper2")], device=self.device)
        self.finger_body_indices = self.robot_body_indices[:, self.finger_local_indices]
        self.wall_force_samples = torch.zeros(self.num_envs, 8, 3, device=self.device)
        self.finger_force_samples = torch.zeros_like(self.wall_force_samples)
        self.contact_samples_written = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.measurement_valid = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.last_contact_force_world = torch.zeros(self.num_envs, 3, device=self.device)
        self.contact_start_world = torch.zeros_like(self.last_contact_force_world)
        self.contact_target_world = torch.zeros_like(self.last_contact_force_world)
        self.contact_reference_world = (self.env_origins + torch.tensor(
            self.contact_manifest["scene"]["reference_tcp_m"], device=self.device))
        self.contact_episode_indices = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        self.contact_peak_n = torch.zeros(self.num_envs, device=self.device)
        # Eight corners conservatively enclose each collision link in its own frame.
        bounds = self.contact_manifest["scene"].get("body_local_collision_bounds_m", {})
        corners, present = [], []
        for name in names:
            lower, upper = bounds.get(name, [[0., 0., 0.], [0., 0., 0.]])
            corners.append([[upper[axis] if selector[axis] else lower[axis] for axis in range(3)]
                            for selector in product((0, 1), repeat=3)])
            present.append(name in bounds)
        self.collision_local_corners = torch.tensor(corners, device=self.device)
        self.collision_bounds_present = torch.tensor(present, device=self.device, dtype=torch.bool)
        center = self.env_origins + torch.tensor(self.contact_manifest["scene"]["wall_center_relative_m"], device=self.device)
        half_size = .5 * torch.tensor(self.contact_manifest["scene"]["wall_size_m"], device=self.device)
        self.wall_lower_world, self.wall_upper_world = center - half_size, center + half_size
        self.wall_proximity_samples = torch.zeros(self.num_envs, 8, self.num_bodies, device=self.device)
        self.finger_ground_proximity_samples = torch.zeros(self.num_envs, 8, device=self.device, dtype=torch.bool)
        self.robot_contact_samples = torch.zeros(self.num_envs, 8, self.num_bodies, 3, device=self.device)

    def _initialize_force_events(self) -> None:
        """Allocate full-scene zero injection storage and use real tool body indices.

        Note:
            The wall adds a body but no DOFs. Both artificial force labels remain
            zero, even when a nonzero scripted command requests pressing.
        """
        super()._initialize_force_events()
        self.rigid_body_forces = torch.zeros_like(self.raw_contact_forces)
        self.ee_force_body_indices = self.robot_body_indices[:, self.tcp_body_index]

    def _reset_force_events(self, env_ids: torch.Tensor) -> None:
        """Clear artificial forces without sampling a random training event.

        Args:
            env_ids: Selected logical robot rows [K].

        Note:
            Contact force never enters these buffers or auxiliary labels.
        """
        self.pending_ee_force_world[env_ids] = 0.
        self.applied_ee_force_world[env_ids] = 0.
        self.rigid_body_forces[self.ee_force_body_indices[env_ids]] = 0.

    def _advance_force_events(self, env_ids: torch.Tensor) -> None:
        """Retain zero injection while the world script publishes force commands.

        Args:
            env_ids: Logical robot rows [K] at the next publication boundary.
        """
        self.pending_ee_force_world[env_ids] = 0.

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        """Reset selected robots, then initialize trajectory and invalidate old force.

        Args:
            env_ids: Logical robot rows [K]; walls and other robots stay unchanged.

        Note:
            The parent's FK reads actual reset root/DOFs without advancing physics.
            Episode parity balances half/full amplitudes including early failures.
        """
        super().reset_idx(env_ids)
        self.contact_episode_indices[env_ids] += 1
        quota = self.contact_manifest["episodes_per_env"]
        full_amplitude = (env_ids * quota + self.contact_episode_indices[env_ids]) % 2
        self.contact_peak_n[env_ids] = (0. if self.zero_control else
            self.contact_manifest["contact_peak_n"] * (.5 + .5 * full_amplitude))
        self.contact_start_world[env_ids] = self._tcp_position()[env_ids]
        self.measurement_valid[env_ids] = False
        self.wall_force_samples[env_ids] = 0.
        self.finger_force_samples[env_ids] = 0.
        self.last_contact_force_world[env_ids] = 0.
        self.contact_samples_written[env_ids] = 0
        self.wall_proximity_samples[env_ids] = 0.
        self.finger_ground_proximity_samples[env_ids] = False
        self.robot_contact_samples[env_ids] = 0.
        self._advance_task_commands(env_ids)

    def _initialize_target_segments(self, env_ids: torch.Tensor, start: torch.Tensor) -> None:
        """Initialize existing task fields without random goal resampling.

        Args:
            env_ids: Selected reset rows [K].
            start: Original reset TCP spherical coordinates [K,3] in m/rad/rad.
        """
        self.target_start[env_ids] = start
        self.target_end[env_ids] = start
        self.target_move_steps[env_ids] = int(round(3. / self.dt))
        self.target_hold_steps[env_ids] = int(round(17. / self.dt))
        self.target_elapsed_steps[env_ids] = 0

    def _advance_task_commands(self, env_ids: torch.Tensor) -> None:
        """Publish the open-loop world target and yaw-frame pressing command.

        Args:
            env_ids: Robot rows [K] at their current episode clock.

        Note:
            Uses 0--3 s smooth approach, 5--7 s rise, 7--11 s hold, 11--13 s
            fall and 15--18 s smooth 5 cm retreat. No measured force feeds back.
        """
        time_s = self.episode_length_buf[env_ids].float() * self.dt
        approach = (time_s / 3.).clamp(0., 1.)
        approach = approach.square() * (3. - 2. * approach)
        retreat = ((time_s - 15.) / 3.).clamp(0., 1.)
        retreat = retreat.square() * (3. - 2. * retreat)
        self.contact_target_world[env_ids] = (self.contact_start_world[env_ids]
            + approach[:, None] * (self.contact_reference_world[env_ids] - self.contact_start_world[env_ids]))
        self.contact_target_world[env_ids, 0] -= .05 * retreat
        # Convert only the representation; the absolute world point never follows the base.
        spherical = self._world_to_task_spherical(self.contact_target_world)
        self.commands[env_ids, 3:6] = spherical[env_ids]
        self.commands[env_ids, :3] = 0.
        self.commands[env_ids, 6:9] = 0.
        self.commands[env_ids, 12:15] = 0.
        force_n = torch.minimum((time_s - 5.) / 2., (13. - time_s) / 2.).clamp(0., 1.) * self.contact_peak_n[env_ids]
        yaw = self.rpy[env_ids, 2]
        self.commands[env_ids, 9:12] = torch.stack((yaw.cos() * force_n, -yaw.sin() * force_n,
                                                  torch.zeros_like(force_n)), dim=1)
        self.target_elapsed_steps[env_ids] = self.episode_length_buf[env_ids]

    def _geometric_target_world(self) -> torch.Tensor:
        """Return the absolute scripted geometric point without force compensation.

        Returns:
            World XYZ targets [N,3] in meters, fixed independently of base motion.
        """
        return self.contact_target_world

    def _sync_robot_state_tensors(self) -> None:
        """Copy refreshed native robot rows without changing wall rows or cache views.

        Note:
            Runs only after native refresh at initialization/reset/policy boundaries.
        """
        self.root_states.copy_(self.raw_root_states[self.robot_actor_indices])
        self.contact_forces.copy_(self.raw_contact_forces[self.robot_body_indices])
        self.rigid_state.copy_(self.raw_rigid_state[self.robot_body_indices])

    def _commit_robot_reset(self, env_ids: torch.Tensor) -> None:
        """Scatter selected logical roots and submit complete native buffers.

        Args:
            env_ids: Reset logical robot rows [K], never native actor IDs.

        Note:
            Uses actual global robot actor IDs for both indexed setters. Neither
            wall state nor unselected robot rows are overwritten.
        """
        actor_ids = self.robot_actor_indices[env_ids].to(dtype=torch.int32)
        self.raw_root_states[self.robot_actor_indices[env_ids]] = self.root_states[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.raw_root_states), gymtorch.unwrap_tensor(actor_ids), len(actor_ids)
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(actor_ids), len(actor_ids)
        )

    def _after_physics_substep(self, physics_index: int) -> None:
        """Sample raw wall/finger forces once after each actual physics substep.

        Args:
            physics_index: Substep 0..7 within the existing 20 ms policy interval.

        Note:
            Stores [N,8,3] world-newton samples. The latest robot reaction is
            minus the wall force; it remains separate from artificial-force labels.
            Does not simulate, write files or feed force back to the controller.
        """
        self.gym.refresh_net_contact_force_tensor(self.sim)
        if physics_index == 0:
            self.contact_samples_written.zero_()
        self.wall_force_samples[:, physics_index] = self.raw_contact_forces[self.wall_body_indices]
        self.finger_force_samples[:, physics_index] = self.raw_contact_forces[self.finger_body_indices].sum(dim=1)
        self.robot_contact_samples[:, physics_index] = self.raw_contact_forces[self.robot_body_indices]
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        poses = self.raw_rigid_state[self.robot_body_indices]
        quaternion_vector = poses[:, :, None, 3:6].expand(-1, -1, 8, -1)
        local_corners = self.collision_local_corners[None].expand(self.num_envs, -1, -1, -1)
        cross = 2. * torch.cross(quaternion_vector, local_corners, dim=-1)
        world_corners = (local_corners + poses[:, :, None, 6:7] * cross
                         + torch.cross(quaternion_vector, cross, dim=-1) + poses[:, :, None, :3])
        lower, upper = world_corners.min(dim=2).values, world_corners.max(dim=2).values
        gap = torch.maximum(self.wall_lower_world[:, None] - upper, lower - self.wall_upper_world[:, None]).clamp_min(0.)
        self.wall_proximity_samples[:, physics_index] = torch.linalg.vector_norm(gap, dim=2)
        self.finger_ground_proximity_samples[:, physics_index] = (
            lower[:, self.finger_local_indices, 2] <= self.env_origins[:, None, 2] + .02).any(dim=1)
        self.contact_samples_written += 1
        if physics_index == 7:
            self.measurement_valid.copy_((self.contact_samples_written == 8)
                & torch.isfinite(self.wall_force_samples).all(dim=2).all(dim=1)
                & torch.isfinite(self.finger_force_samples).all(dim=2).all(dim=1))
            self.last_contact_force_world.copy_(-self.wall_force_samples[:, -1])

    def _build_policy_trace(self, base_push_applied: bool, terminated: torch.Tensor,
                            time_outs: torch.Tensor) -> List[Dict[str, Any]]:
        """Add raw measured intervals and geometry diagnostics before local reset.

        Args:
            base_push_applied: Original push marker, always false in this scene.
            terminated: Physical failure flags [N] from the original environment.
            time_outs: Timeout-only flags [N] before reset.

        Returns:
            Existing trace rows augmented with eight wall/finger samples in world N,
            conservative contact classification and fixed-script amplitude/time.

        Note:
            Sampling is finished; this method never advances physics or feeds force
            into observations, artificial-force labels, Critic or rewards.
        """
        rows = super()._build_policy_trace(base_push_applied, terminated, time_outs)
        nonfinger = self.collision_bounds_present.clone()
        nonfinger[self.finger_local_indices] = False
        for row in rows:
            env_id = row["env_id"]
            wall, fingers = self.wall_force_samples[env_id], self.finger_force_samples[env_id]
            near_wall = (self.wall_proximity_samples[env_id] <= .02).any(dim=0)
            near_nonfinger = near_wall & nonfinger
            status = classify_contact_interval(wall, fingers, bool(near_nonfinger.any()),
                bool(near_wall[self.finger_local_indices].any()),
                bool(self.finger_ground_proximity_samples[env_id].any()), bool(self.measurement_valid[env_id]))
            row.update(contact_peak_n=float(self.contact_peak_n[env_id]),
                       contact_measurement_valid=bool(self.measurement_valid[env_id]), contact_status=status,
                       wall_force_samples_world_n=wall.cpu().tolist(), finger_force_samples_world_n=fingers.cpu().tolist(),
                       wall_force_mean_world_n=wall.mean(dim=0).cpu().tolist(),
                       wall_force_last_world_n=wall[-1].cpu().tolist(),
                       wall_force_peak_norm_n=float(torch.linalg.vector_norm(wall, dim=1).max()),
                       wall_finger_residual_peak_n=float(torch.linalg.vector_norm(wall + fingers, dim=1).max()),
                       nonfinger_near_wall_local_indices=near_nonfinger.nonzero().flatten().cpu().tolist(),
                       body_wall_min_aabb_distance_m=self.wall_proximity_samples[env_id].min(dim=0).values.cpu().tolist(),
                       robot_contact_peak_norm_n=torch.linalg.vector_norm(
                           self.robot_contact_samples[env_id], dim=2).max(dim=0).values.cpu().tolist(),
                       contact_episode_index=int(self.contact_episode_indices[env_id]))
        if self.video_recorders:
            episode_index = int(self.contact_episode_indices[0])
            recorder = self.video_recorders.get(episode_index)
            if recorder is not None and (int(self.episode_length_buf[0]) % 2 == 0 or terminated[0] or time_outs[0]):
                recorder.capture()
        return rows

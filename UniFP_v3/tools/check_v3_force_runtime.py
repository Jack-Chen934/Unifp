"""Run the approved native force/contact checks and eight-condition load scan."""

import copy
import json
import platform
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Tuple
import xml.etree.ElementTree as ET

import isaacgym  # noqa: F401
from isaacgym import gymapi, gymtorch
import torch
import legged_gym.envs  # noqa: F401
from legged_gym.b2_gym_learn.ppo_cse_pf.v3_actor_critic import V3ActorCritic
from legged_gym.b2_gym_learn.ppo_cse_pf.v3_evaluation import validate_policy_evaluation_checkpoint
from legged_gym.b2_gym_learn.ppo_cse_pf.v3_run_identity import build_run_identity, file_sha256
from legged_gym.envs.v3.legged_robot_v3 import V3PosForce
from legged_gym.envs.v3.v3_contact_evaluation import V3ContactEvaluation
from legged_gym.envs.v3.v3_force_evaluation import V3ForceEvaluation
from legged_gym.scripts.evaluate_v3_contact import evaluate_contact_scenario
from legged_gym.scripts.v3_args import get_v3_args
from legged_gym.utils.helpers import class_to_dict, parse_sim_params, set_seed
from legged_gym.utils.v3_config import resolve_v3_configs
from legged_gym.utils.v3_kinematics import V3TcpKinematics


def range_force_script(time_s: torch.Tensor, env_ids: torch.Tensor,
                       peak_n: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate the frozen six-axis load scan without stepping or sampling RNG.

    Args:
        time_s: Episode-local input boundary times [N] in seconds.
        env_ids: Logical environment IDs [N]; groups of four select C/E/same/match.
            IDs 0--15 are static and 16--31 moving, with identical force scripts.
        peak_n: Single-axis force amplitude in N for this independent run.

    Returns:
        Command world forces [N,3], artificial world forces [N,3], and direction
        indices [N] ordered +X,-X,+Y,-Y,+Z,-Z, starting at 1/4/7/10/13/16 s.
    """
    direction = ((time_s - 1.) / 3.).floor().long().clamp(0, 5)
    age_s = time_s - (1. + direction * 3.)
    envelope = torch.minimum(age_s / .5, (2. - age_s) / .5).clamp(0., 1.)
    axes = torch.tensor(((1., 0., 0.), (-1., 0., 0.), (0., 1., 0.),
                         (0., -1., 0.), (0., 0., 1.), (0., 0., -1.)), device=time_s.device)
    force_world = axes[direction] * (peak_n * envelope[:, None])
    condition = (env_ids // 4) % 4
    command = force_world * (condition != 1)[:, None]
    command[condition == 3] *= -1.
    external = force_world * (condition != 0)[:, None]
    return command, external, direction


class RangeForceEvaluation(V3ForceEvaluation):
    """Keep all production dynamics while replacing only the G2 event script."""

    def _initialize_force_events(self) -> None:
        """Allocate normal event buffers and a conservative TCP root-distance bound.

        Note:
            The sum of all URDF chain translation lengths plus TCP offset is an
            upper bound under any joint/root rotation. Exceeding it proves an
            impossible target; staying inside it does not prove reachability.
        """
        super()._initialize_force_events()
        self.tcp_root_reach_upper_bound_m = (sum(float(torch.linalg.vector_norm(translation))
            for _, _, translation, _ in self.tcp_kinematics.chain)
            + float(torch.linalg.vector_norm(self.tcp_kinematics.tcp_offset_m)))

    def _advance_force_events(self, env_ids: torch.Tensor) -> None:
        """Publish deterministic world loads and static/moving base conditions.

        Args:
            env_ids: Logical robot rows [K] at their local publication boundary.

        Note:
            Command force never enters PhysX; matching forces cancel in the
            compensation equation only. The parent applies external force at COM.
        """
        command, external, _ = range_force_script(self.episode_length_buf[env_ids] * self.dt,
                                                 env_ids, self.cfg.force_events.max_force_n)
        self.pending_ee_force_world[env_ids] = external
        yaw = self.rpy[env_ids, 2]
        self.commands[env_ids, 9:12] = torch.stack((yaw.cos() * command[:, 0] + yaw.sin() * command[:, 1],
            -yaw.sin() * command[:, 0] + yaw.cos() * command[:, 1], command[:, 2]), dim=1)
        self.commands[env_ids[env_ids < 16], :3] = 0.

    def _build_policy_trace(self, base_push_applied: bool, terminated: torch.Tensor,
                            time_outs: torch.Tensor) -> List[Dict[str, Any]]:
        """Record scan condition and actual clipped torque saturation before reset.

        Args:
            base_push_applied: Existing base-push marker, disabled in this scan.
            terminated: Physical failures [N] before local reset.
            time_outs: Pure timeout flags [N] before local reset.

        Returns:
            Original policy trace plus raw root/DOF poses, artificial world force,
            six arm clipped torques and near-limit flags. No demand torque is claimed.
        """
        rows = V3PosForce._build_policy_trace(self, base_push_applied, terminated, time_outs)
        arm_ids = self.action_dof_indices[8:]
        arm_torques = self.torques[:, arm_ids]
        arm_limits = self.torque_limits[arm_ids]
        compensated_radii = self._world_to_task_spherical(self._compensated_target_world())[:, 0]
        compensated_root_distances = torch.linalg.vector_norm(self._compensated_target_world() - self.root_states[:, :3], dim=1)
        for row in rows:
            env_id = row['env_id']
            row.update(condition=('static_' if env_id < 16 else 'moving_') + ('C', 'E', 'same', 'match')[(env_id // 4) % 4],
                force_peak_n=self.cfg.force_events.max_force_n,
                root_state=self.root_states[env_id].cpu().tolist(), dof_position_rad=self.dof_pos[env_id].cpu().tolist(),
                external_force_world_n=self.applied_ee_force_world[env_id].cpu().tolist(),
                arm_clipped_torque_nm=arm_torques[env_id].cpu().tolist(),
                compensated_radius_m=float(compensated_radii[env_id]),
                compensated_root_distance_m=float(compensated_root_distances[env_id]),
                compensated_target_exceeds_chain_bound=bool(compensated_root_distances[env_id] > self.tcp_root_reach_upper_bound_m),
                arm_near_limit=(arm_torques[env_id].abs() >= .99 * arm_limits).cpu().tolist())
        return rows


def free_body_check(args: Namespace, physics_dt_s: float) -> Dict[str, Any]:
    """Verify native COM force on one unconstrained gravity-free primitive.

    Args:
        args: Explicit simulator and tensor devices from the Gym parser.
        physics_dt_s: Production physics step in seconds, nominally 0.0025.

    Returns:
        Actual mass, velocity, angular velocity and analytic F*dt/m relative error.

    Raises:
        AssertionError: Delta-v error exceeds 5% or angular increment exceeds 0.01 rad/s.

    Note:
        Creates and destroys its own simulator and primitive. No training asset,
        ground, constraint, policy or optimizer is involved.
    """
    gym = gymapi.acquire_gym()
    params = gymapi.SimParams()
    params.dt, params.up_axis = physics_dt_s, gymapi.UP_AXIS_Z
    params.gravity = gymapi.Vec3(0., 0., 0.)
    params.physx.use_gpu, params.use_gpu_pipeline = args.use_gpu, args.use_gpu_pipeline
    device_index = int(args.sim_device.split(':')[1]) if ':' in args.sim_device else 0
    sim = gym.create_sim(device_index, -1, gymapi.SIM_PHYSX, params)
    try:
        options = gymapi.AssetOptions()
        options.disable_gravity = True
        options.density = 1000.
        asset = gym.create_box(sim, .1, .1, .1, options)
        env = gym.create_env(sim, gymapi.Vec3(-1., -1., -1.), gymapi.Vec3(1., 1., 1.), 1)
        actor = gym.create_actor(env, asset, gymapi.Transform(), 'free_body', 0, 0)
        mass_kg = gym.get_actor_rigid_body_properties(env, actor)[0].mass
        gym.prepare_sim(sim)
        roots = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
        force = torch.tensor([[10., 0., 0.]], device=args.rl_device)
        steps = int(round(.1 / physics_dt_s))
        for _ in range(steps):
            gym.apply_rigid_body_force_tensors(sim, gymtorch.unwrap_tensor(force), None, gymapi.GLOBAL_SPACE)
            gym.simulate(sim)
            gym.fetch_results(sim, True)
        gym.refresh_actor_root_state_tensor(sim)
        expected_vx = 10. * steps * params.dt / mass_kg
        relative_error = abs(float(roots[0, 7]) - expected_vx) / expected_vx
        angular_increment = float(torch.linalg.vector_norm(roots[0, 10:13]))
        assert relative_error <= .05 and angular_increment <= .01
        return {'mass_kg': mass_kg, 'steps': steps, 'expected_vx_m_s': expected_vx,
                'actual_velocity_m_s': roots[0, 7:10].cpu().tolist(), 'relative_error': relative_error,
                'angular_increment_rad_s': angular_increment, 'torque_argument': None}
    finally:
        gym.destroy_sim(sim)


def check_force_interface(args: Namespace, cfg: Any, params: Any, output: Path) -> Dict[str, Any]:
    """Exercise four robots under seven five-second force conditions and local reset.

    Args:
        args: Actual physics and tensor devices; exactly four environments.
        cfg: Isolated B config retaining production physics/control and asset.
        params: Actual parsed PhysX settings.
        output: New evidence directory for the reduced native trace.

    Returns:
        Segment force labels/responses, finite-state completion and reset isolation.

    Raises:
        AssertionError: Wrong native force slot, label, reset isolation or non-finite state.

    Note:
        Uses zero policy actions and overwrites only the diagnostic input force.
        Normal physical terminations are retained and locally reset by production step.
    """
    cfg.commands.ranges.lin_vel_x = cfg.commands.ranges.lin_vel_y = cfg.commands.ranges.ang_vel_yaw = [0., 0.]
    env = V3ForceEvaluation(cfg, params, args.physics_engine, args.sim_device, True,
        scenario={'mode': 'E', 'amplitude_fraction': 1., 'moving': False, 'push': False}, episodes_per_env=1)
    segments = []
    try:
        target_forces = ((0., 0., 0.), (10., 0., 0.), (-10., 0., 0.), (0., 10., 0.),
                         (0., -10., 0.), (0., 0., 10.), (0., 0., -10.))
        with (output / 'force_trace.jsonl').open('x') as trace_file:
            for segment_id, target in enumerate(target_forces):
                expected = torch.tensor(target, device=env.device).expand(4, 3)
                failures = 0
                for step in range(250):
                    env.pending_ee_force_world.copy_(expected)
                    _, reward, _, info = env.step(torch.zeros(4, 14, device=env.device))
                    failures += len([row for row in info['episode_records'] if row['end_type'] == 'physical_failure'])
                    assert torch.isfinite(reward).all()
                    live = ~info['terminated'] & ~info['time_outs']
                    torch.testing.assert_close(env.applied_ee_force_world[live], expected[live])
                    torch.testing.assert_close(env.rigid_body_forces[env.ee_force_body_indices[live]], expected[live])
                    assert torch.count_nonzero(env.commands[:, 9:12]) == 0
                    trace_file.write(json.dumps({'segment': segment_id, 'step': step,
                        'world_force_n': target, 'root_state': env.root_states.cpu().tolist(),
                        'label_yaw_n': env.external_ee_force_y.cpu().tolist(),
                        'physical_failures': info['terminated'].cpu().tolist()}) + '\n')
                segments.append({'world_force_n': target, 'policy_steps': 250, 'physical_failures': failures})
        old_root, old_dof = env.root_states.clone(), env.dof_pos.clone()
        old_history, old_mass = env.obs_history.clone(), env.mass_com_delta.clone()
        old_events = {name: {field: value.clone() for field, value in state.items()} for name, state in env.force_state.items()}
        env.reset_idx(torch.tensor([1], device=env.device))
        keep = torch.tensor([0, 2, 3], device=env.device)
        for before, after in ((old_root, env.root_states), (old_dof, env.dof_pos), (old_history, env.obs_history)):
            torch.testing.assert_close(before[keep], after[keep], atol=0., rtol=0.)
        torch.testing.assert_close(old_mass, env.mass_com_delta, atol=0., rtol=0.)
        for name, state in old_events.items():
            for field, before in state.items():
                torch.testing.assert_close(before[keep], env.force_state[name][field][keep], atol=0., rtol=0.)
        return {'segments': segments, 'local_reset_isolated': True, 'num_envs': 4,
                'tool_body_indices': env.ee_force_body_indices.cpu().tolist(), 'torque_argument': None,
                'force_frame': 'GLOBAL_SPACE', 'policy_steps': 1750}
    finally:
        env.gym.destroy_sim(env.sim)


def nominal_com_loads(env: V3PosForce, peak_n: float) -> Dict[str, Any]:
    """Compute nominal COM Jacobian load bounds at reset and reference poses.

    Args:
        env: Real V3 robot supplying URDF path and named DOF layout.
        peak_n: Per-axis amplitude bound in N for this scan.

    Returns:
        Numerical world COM Jacobians in m/rad and six worst-case axis-box load
        bounds in N*m; excludes gravity, inertia and controller reserve.
    """
    kinematics = V3TcpKinematics(Path(env.cfg.asset.file), env.dof_names, env.device)
    urdf = ET.parse(env.cfg.asset.file).getroot()
    tool_link = next(link for link in urdf.findall('link') if link.get('name') == 'arm_seg6')
    com_xyz = [float(value) for value in tool_link.find('inertial/origin').get('xyz').split()]
    kinematics.tcp_offset_m = torch.tensor(com_xyz, device=env.device)
    roots = torch.zeros(1, 13, device=env.device)
    roots[:, 6] = 1.
    arm_ids = env.action_dof_indices[8:]
    result = {}
    for name, pose in (('reset', env.default_dof_pos.clone()), ('reference', env.default_dof_pos.clone())):
        if name == 'reference':
            pose[:, arm_ids] = torch.tensor([0., 1.4, -1., 0., 0., 0.], device=env.device)
        jacobian_columns = []
        for joint_id in arm_ids:
            plus, minus = pose.clone(), pose.clone()
            plus[:, joint_id] += .001
            minus[:, joint_id] -= .001
            jacobian_columns.append((kinematics.tcp_world(roots, plus) - kinematics.tcp_world(roots, minus))[0] / .002)
        jacobian = torch.stack(jacobian_columns, dim=1)
        result[name] = {'com_jacobian_m_rad': jacobian.cpu().tolist(),
                        'axis_box_load_bound_nm': (peak_n * jacobian.abs().sum(dim=0)).cpu().tolist()}
    return result


def check_contact_interface(args: Namespace, cfg: Any, params: Any,
                            payload: Dict[str, Any], output: Path) -> Dict[str, Any]:
    """Validate wall measurement, reset isolation and complete real-parent scripts.

    Args:
        args: Actual devices and the explicit validated A8000 parent checkpoint.
        cfg: Four-environment B native configuration, isolated from training.
        params: Actual PhysX parameters and tensor pipeline.
        payload: Provenance-validated policy checkpoint, never passed to a learner.
        output: New evidence directory for diagnostic and full-script traces.

    Returns:
        Three five-second controls, clean consecutive intervals, wall/reset checks
        and complete 20 s contact/zero-script summaries from the actual parent.

    Raises:
        AssertionError: Force sign/balance/zero, consecutive contact or reset isolation fails.

    Note:
        Diagnostic pose placement only establishes a measurement positive control;
        it is not policy force tracking and never modifies an asset or training state.
    """
    manifest = json.loads((Path(__file__).resolve().parents[1] / 'configs/v3_b_contact_v1.json').read_text())
    manifest.update(num_envs=4, episodes_per_env=1)
    for name in ('randomize_friction', 'randomize_base_mass', 'randomize_base_com',
                 'randomize_tool_mass', 'randomize_execution_scale'):
        setattr(cfg.domain_rand, name, False)
    env = V3ContactEvaluation(cfg, params, args.physics_engine, args.sim_device, True, manifest=manifest)
    controls = []
    try:
        env.policy_trace_enabled = True
        env.contact_peak_n.zero_()
        wall_ids = [env.gym.get_actor_index(handle, actor, gymapi.DOMAIN_SIM)
                    for handle, actor in zip(env.envs, env.wall_handles)]
        original_walls = env.raw_root_states[wall_ids].clone()
        arm_ids = env.action_dof_indices[8:]
        reference = torch.tensor(manifest['scene']['q_ref_rad'], device=env.device)
        actions = torch.zeros(4, 14, device=env.device)
        max_consecutive = torch.zeros(4, device=env.device, dtype=torch.long)
        consecutive = torch.zeros_like(max_consecutive)
        with (output / 'contact_probe_trace.jsonl').open('x') as trace_file:
            for name in ('no_contact', 'known_finger_contact', 'retreated'):
                if name == 'known_finger_contact':
                    env.root_states[:] = env.base_init_state
                    env.root_states[:, :3] += env.env_origins
                    env.root_states[:, 0] += .012
                    env.root_states[:, 2] = env.env_origins[:, 2] + .32
                    env.root_states[:, 7:13] = 0.
                    env.dof_pos[:, arm_ids] = reference
                    env.dof_vel.zero_()
                    env._commit_robot_reset(torch.arange(4, device=env.device))
                    actions[:, 8:] = (reference - env.default_dof_pos[:, arm_ids]) / cfg.control.action_scale
                elif name == 'retreated':
                    env.reset_idx(torch.arange(4, device=env.device))
                    env.contact_peak_n.zero_()
                    env.root_states[:, 0] -= .15
                    env._commit_robot_reset(torch.arange(4, device=env.device))
                    actions.zero_()
                peak_n, failures = 0., 0
                for step in range(250):
                    _, _, _, info = env.step(actions)
                    failures += int(info['terminated'].sum())
                    clean = torch.zeros(4, device=env.device, dtype=torch.bool)
                    for row in info['policy_trace']:
                        row.update(diagnostic_control=name, control_step=step)
                        trace_file.write(json.dumps(row, allow_nan=False) + '\n')
                        clean[row['env_id']] = row['contact_status'] == 'clean_expected_contact' and row['wall_force_mean_world_n'][0] > 1.
                    peak_n = max(peak_n, float(torch.linalg.vector_norm(env.wall_force_samples, dim=2).max()))
                    if name == 'known_finger_contact':
                        consecutive = (consecutive + 1) * clean
                        max_consecutive = torch.maximum(max_consecutive, consecutive)
                controls.append({'name': name, 'policy_steps': 250, 'peak_wall_norm_n': peak_n,
                                 'ordinary_physical_failures': failures})
                if name != 'known_finger_contact':
                    assert peak_n <= .1
            assert bool((max_consecutive >= 2).all()), max_consecutive.cpu().tolist()
        before_roots, before_dof = env.raw_root_states.clone(), env.dof_pos.clone()
        before_history, before_valid = env.obs_history.clone(), env.measurement_valid.clone()
        env.reset_idx(torch.tensor([1], device=env.device))
        keep = torch.tensor([0, 2, 3], device=env.device)
        torch.testing.assert_close(env.root_states[keep], before_roots[env.robot_actor_indices[keep]], atol=0., rtol=0.)
        torch.testing.assert_close(env.dof_pos[keep], before_dof[keep], atol=0., rtol=0.)
        torch.testing.assert_close(env.obs_history[keep], before_history[keep], atol=0., rtol=0.)
        torch.testing.assert_close(env.measurement_valid[keep], before_valid[keep])
        assert not env.measurement_valid[1]
        torch.testing.assert_close(env.raw_root_states[wall_ids], original_walls, atol=1e-6, rtol=0.)
        report = {'controls': controls, 'clean_contact_consecutive_intervals': max_consecutive.cpu().tolist(),
                  'wall_pose_preserved': True, 'local_reset_isolated': True, 'reset_invalidates_only_selected_measurement': True,
                  'diagnostic_pose_offset_x_m': .012,
                  '说明': '人为参考姿态与12mm相对穿入仅验证测量；按压力峰值不是策略性能或训练幅值。'}
    finally:
        env.gym.destroy_sim(env.sim)
    evaluation_args = copy.copy(args)
    evaluation_args.run_purpose = 'evaluation'
    for zero_control in (False, True):
        name = 'zero_control' if zero_control else 'contact'
        directory = output / name
        directory.mkdir()
        summary, config = evaluate_contact_scenario(evaluation_args, payload, manifest,
            zero_control=zero_control, output_dir=directory)
        (directory / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
        (directory / 'effective_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
        report[name] = {'complete': summary['evidence_complete'], 'policy_updates': summary['policy_updates'],
                        'assessment': summary['assessment']}
    return report


def run_range_scan(args: Namespace, cfg: Any, params: Any, payload: Dict[str, Any], output: Path) -> Dict[str, Any]:
    """Run one 32-environment, 20-second diagnostic amplitude without updates.

    Args:
        args: Source checkpoint and actual physics/tensor devices.
        cfg: Isolated B scan config; max_force_n selects the current amplitude.
        params: Actual native PhysX settings.
        payload: Validated complete A8000 checkpoint supplying policy weights only.
        output: New evidence directory for all transitions, endings and failure poses.

    Returns:
        Per-condition direction coverage, clipped arm saturation and tracking/errors.
        Outside training goal ranges is a diagnostic, not a reachability certificate.

    Note:
        Does not restore training RNG/physics or create an optimizer. All 32 robots
        run 1000 policy steps even after ordinary local reset; failures are retained.
    """
    env = RangeForceEvaluation(cfg, params, args.physics_engine, args.sim_device, True,
        scenario={'mode': 'scan', 'amplitude_fraction': 1., 'moving': True, 'push': False}, episodes_per_env=1)
    try:
        model = V3ActorCritic(env.num_obs, env.num_privileged_obs, env.cfg.env.num_pred_obs,
            env.cfg.env.num_single_obs, env.num_actions, **payload['config_snapshot']['training']['policy']).to(env.device)
        model.load_state_dict(payload['model_state_dict'], strict=True)
        model.eval()
        env.policy_trace_enabled = True
        observations = env.get_observations()
        statistics: Dict[str, Any] = {}
        endings = []
        runs = torch.zeros(32, 6, device=env.device, dtype=torch.long)
        longest = torch.zeros_like(runs)
        with (output / 'range_trace.jsonl').open('x') as trace_file:
            for step in range(1000):
                with torch.inference_mode():
                    actions = model.act_inference(observations, {})
                observations, _, _, info = env.step(actions)
                endings.extend(info['episode_records'])
                for row in info['policy_trace']:
                    condition = row['condition']
                    bucket = statistics.setdefault(condition, {'samples': 0, 'direction_samples': [0]*6,
                        'arm_near_limit_samples': [0]*6, 'tcp_geometric_l2_sum_m': 0., 'tcp_compensated_l2_sum_m': 0.,
                        'physical_failures': 0, 'compensated_radius_outside_sampled_domain': 0})
                    bucket.setdefault('compensated_target_exceeds_chain_bound_samples', 0)
                    env_id = row['env_id']
                    time_s = torch.tensor([row['policy_input_time_s']], device=env.device)
                    command, external, direction = range_force_script(time_s, torch.tensor([env_id], device=env.device), cfg.force_events.max_force_n)
                    if torch.count_nonzero(command) or torch.count_nonzero(external):
                        bucket['direction_samples'][int(direction[0])] += 1
                    bucket['samples'] += 1
                    bucket['physical_failures'] += row['terminated']
                    bucket['tcp_geometric_l2_sum_m'] += row['physical_errors']['tcp_geometric_l2_m']
                    bucket['tcp_compensated_l2_sum_m'] += row['physical_errors']['tcp_compensated_l2_m']
                    for joint_index, saturated in enumerate(row['arm_near_limit']):
                        bucket['arm_near_limit_samples'][joint_index] += saturated
                    saturated = torch.tensor(row['arm_near_limit'], device=env.device)
                    runs[env_id] = (runs[env_id] + 1) * saturated
                    longest[env_id] = torch.maximum(longest[env_id], runs[env_id])
                    radius = row['compensated_radius_m']
                    bucket['compensated_radius_outside_sampled_domain'] += bool(radius < cfg.goal_ee.radius[0] or radius > cfg.goal_ee.radius[1])
                    bucket['compensated_target_exceeds_chain_bound_samples'] += row['compensated_target_exceeds_chain_bound']
                    row.update(evaluation_step=step)
                    trace_file.write(json.dumps(row, allow_nan=False) + '\n')
                runs[info['terminated'] | info['time_outs']] = 0
        for condition, bucket in statistics.items():
            bucket['arm_near_limit_fraction'] = [value / bucket['samples'] for value in bucket['arm_near_limit_samples']]
            bucket['tcp_geometric_l2_mean_m'] = bucket['tcp_geometric_l2_sum_m'] / bucket['samples']
            bucket['tcp_compensated_l2_mean_m'] = bucket['tcp_compensated_l2_sum_m'] / bucket['samples']
            bucket['direction_coverage_complete'] = all(bucket['direction_samples'])
        return {'peak_n': cfg.force_events.max_force_n, 'conditions': statistics, 'policy_steps': 1000,
            'policy_updates': 0, 'completed_episodes': endings,
            'longest_arm_near_limit_duration_s': (longest.max(dim=0).values * env.dt).cpu().tolist(),
            'tcp_root_reach_upper_bound_m': env.tcp_root_reach_upper_bound_m,
            'nominal_com_loads': nominal_com_loads(env, cfg.force_events.max_force_n),
            '说明': '力矩为实际裁剪后值；目标超出原采样域不等同于物理不可达，普通失败不自动降幅。'}
    finally:
        env.gym.destroy_sim(env.sim)


def main() -> None:
    """Execute one named native validation in a new explicit evidence directory.

    Raises:
        ValueError: Mode, devices, counts or checkpoint provenance are invalid.
        FileExistsError: Output directory already exists.
        AssertionError: A native engineering contract fails.

    Note:
        --scenario force_interface performs G1; range_scan performs four G2
        amplitudes. Contact G3 uses the existing contact entry plus native probes.
        No formal training or asset mutation occurs.
    """
    args = get_v3_args('V3 force native checks: --scenario force_interface or range_scan')
    if args.scenario not in {'force_interface', 'range_scan', 'contact_interface'} or args.run_purpose != 'preflight':
        raise ValueError('Explicit preflight purpose and native diagnostic scenario are required')
    expected_envs = 32 if args.scenario == 'range_scan' else 4
    if args.num_envs != expected_envs or args.reproduction_part != 'B':
        raise ValueError('Native diagnostic requires B and its confirmed environment count')
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg, train_cfg = resolve_v3_configs(args, purpose='preflight')
    cfg.commands.zero_vel_cmd_prob, cfg.domain_rand.push_robots = 0., False
    snapshot = {'environment': class_to_dict(cfg), 'training': class_to_dict(train_cfg)}
    params = parse_sim_params(args, {'sim': snapshot['environment']['sim']})
    if (torch.device(args.rl_device).type == 'cuda') != params.use_gpu_pipeline:
        raise ValueError('Tensor device must match the actual pipeline')
    if torch.device(args.rl_device).type == 'cuda':
        torch.cuda.set_device(args.rl_device)
    set_seed(args.seed)
    if args.scenario == 'force_interface':
        free_body = free_body_check(args, cfg.sim.dt)
        report = check_force_interface(args, cfg, params, output)
        report['free_body'] = free_body
    else:
        if not args.checkpoint:
            raise ValueError('Range scan requires the explicit A8000 parent')
        payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if payload['identity']['reproduction_part'] != 'A':
            raise ValueError('Range scan must use A8000')
        source_root = Path(__file__).resolve().parents[1]
        parent_snapshot = copy.deepcopy(payload['config_snapshot'])
        asset_name = Path(parent_snapshot['environment']['asset']['file']).name
        asset_relative = next(name for name in payload['identity']['asset_manifest'] if name.endswith('/' + asset_name))
        parent_snapshot['environment']['asset']['file'] = str(source_root / asset_relative)
        active_identity = build_run_identity(source_root, parent_snapshot, payload['identity']['runtime'])
        compatibility = validate_policy_evaluation_checkpoint(payload, active_identity)
        if args.scenario == 'contact_interface':
            report = check_contact_interface(args, cfg, params, payload, output)
            report['source_compatibility'] = compatibility
        else:
            report = {'source_compatibility': compatibility, 'amplitudes': []}
            for peak_n in (10., 20., 40., 60.):
                set_seed(args.seed)
                cfg.force_events.max_force_n = peak_n
                directory = output / f'{int(peak_n)}N'
                directory.mkdir()
                report['amplitudes'].append(run_range_scan(args, cfg, params, payload, directory))
    report['runtime'] = {'python': platform.python_version(), 'torch': torch.__version__, 'sim_device': args.sim_device,
        'rl_device': args.rl_device, 'physx_gpu': params.physx.use_gpu, 'gpu_pipeline': params.use_gpu_pipeline}
    report['source_sha256'] = {str(path.relative_to(Path(__file__).resolve().parents[1])): file_sha256(path)
        for path in [Path(__file__).resolve(), Path(__file__).resolve().parents[1] / 'legged_gym/envs/v3/legged_robot_v3.py']}
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()

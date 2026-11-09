"""Compare analytic wheel commands with the trained policy in isolated physics."""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict

from isaacgym import gymapi, gymtorch, gymutil
import numpy as np
import torch


def advance_interval(
    environment: Any, actions: torch.Tensor, cpu_pipeline: bool
) -> torch.Tensor:
    """Advance either tensor pipeline and capture actual physics-substep torques.

    Args:
        environment: Isolated V3 environment using CPU or GPU tensors.
        actions: Policy-order actions [N,14], in dimensionless command units.
        cpu_pipeline: Fetch each native step before reading CPU tensors when true.

    Returns:
        Maximum absolute applied torque [N,14] across all native physics substeps in one policy interval.

    Note:
        Advances native physics and readback only. CPU tensors require a blocking
        fetch; GPU tensors retain the training pipeline refresh sequence. No task
        counters or optimizer state are advanced.
    """
    applied_torques = []
    for _ in range(environment.cfg.control.decimation):
        torques = environment._compute_v3_torques(actions)
        applied_torques.append(torques.detach().clone())
        environment.gym.set_dof_actuation_force_tensor(environment.sim, gymtorch.unwrap_tensor(torques))
        environment.gym.simulate(environment.sim)
        if cpu_pipeline:
            environment.gym.fetch_results(environment.sim, True)
        environment.gym.refresh_dof_state_tensor(environment.sim)
    environment.gym.refresh_actor_root_state_tensor(environment.sim)
    environment.gym.refresh_rigid_body_state_tensor(environment.sim)
    environment.gym.refresh_net_contact_force_tensor(environment.sim)
    environment._update_base_state()
    return torch.stack(applied_torques).abs().amax(dim=0)


def main() -> None:
    """Measure four body motions without training or changing source assets.

    Raises:
        RuntimeError: Frozen environment or checkpoint cannot be loaded.
        OSError: Diagnostic output cannot be written.

    Note:
        Nominal actors disable randomization, push, and observation noise only
        here. Analytic steering/speed commands are diagnostic inputs, never a
        replacement high-level controller for the reproduction project. Optional
        timing refinement changes this isolated process only, preserving policy time.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gain-scale', type=float, default=1.0)
    parser.add_argument('--disable-self-collision', action='store_true')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--asset', type=Path)
    parser.add_argument('--cpu-contacts', action='store_true')
    parser.add_argument('--steering-gain-scale', type=float, default=1.0)
    parser.add_argument('--yaw-wheel-target-scale', type=float, default=1.0)
    parser.add_argument('--motion-sign', type=int, choices=(-1, 1), default=1)
    parser.add_argument('--base-height', type=float, default=None)
    parser.add_argument('--settle-seconds', type=float, default=1.0)
    parser.add_argument('--physics-step-factor', type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root))
    sys.path.insert(0, str(args.source_root / 'tools'))
    from compare_desktop_fk import forward_kinematics
    from legged_gym.envs.v3.legged_robot_v3 import V3PosForce
    from legged_gym.envs.v3.v3_pos_force_config import V3PosForceCfg
    from legged_gym.utils.helpers import class_to_dict, set_seed
    from probe_training_collision_runtime import read_properties

    set_seed(1)
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    cfg = V3PosForceCfg()
    cfg.env.num_envs = 4
    # Isolated counterfactual: preserve the 20 ms policy interval while refining PD/physics.
    cfg.sim.dt /= args.physics_step_factor
    cfg.control.decimation *= args.physics_step_factor
    if args.base_height is not None:
        cfg.init_state.pos = [0.0, 0.0, args.base_height]
    if args.asset:
        cfg.asset.file = str(args.asset)
    if args.disable_self_collision:
        cfg.asset.self_collisions = 1
    cfg.noise.add_noise = False
    for field in ('randomize_base_mass', 'randomize_base_com', 'randomize_tool_mass',
                  'randomize_friction', 'randomize_execution_scale', 'push_robots'):
        setattr(cfg.domain_rand, field, False)
    parameters = gymapi.SimParams()
    gymutil.parse_sim_config(class_to_dict(cfg.sim), parameters)
    parameters.use_gpu_pipeline = not args.cpu_contacts
    parameters.physx.use_gpu = True
    environment = V3PosForce(cfg, parameters, gymapi.SIM_PHYSX, 'cuda:0', True)
    try:
        environment.d_gains[environment.wheel_dof_indices] *= args.gain_scale
        environment.p_gains[environment.action_dof_indices[:4]] *= args.steering_gain_scale
        base_pairs = {name: {} for name in ('forward', 'backward', 'lateral', 'yaw')}
        rigid_names = environment.gym.get_actor_rigid_body_names(environment.envs[0], environment.actor_handles[0])
        motions = np.asarray([[0.3, 0.0, 0.0], [-0.3, 0.0, 0.0],
                              [0.0, 0.2, 0.0], [0.0, 0.0, 0.3]])
        motions *= args.motion_sign
        names = ('forward', 'backward', 'lateral', 'yaw')
        manifest = json.loads((Path(cfg.asset.file).parents[1] / 'asset_manifest.json').read_text())
        poses = forward_kinematics(Path(cfg.asset.file), cfg.init_state.default_joint_angles, np.eye(4))
        wheel_names = ('fl_Link', 'fr_Link', 'bl_Link', 'br_Link')
        pivot_xy = np.asarray([poses[corner + '_steering_link'][:2, 3] for corner in ('fl', 'fr', 'bl', 'br')])
        zero_headings = np.asarray([math.atan2(-poses[name][0, 1], poses[name][1, 1]) for name in wheel_names])
        rolling_geometry = {item['link']: item for item in manifest['rolling_collision_records']}
        wheel_radius_m = np.asarray([rolling_geometry[name]['radius_m'] for name in wheel_names])
        center_y = {name: rolling_geometry[name]['center_y_m'] for name in wheel_names}
        lateral_offsets = np.asarray([
            np.dot((poses[name] @ np.array([0, center_y.get(name, 0), 0, 1]))[:3]
                   - poses[name[:2] + '_steering_link'][:3, 3], poses[name][:3, 1])
            for name in wheel_names])
        steering_target = np.zeros((4, 4))
        drive_target = np.zeros((4, 4))
        for case, (vx, vy, yaw) in enumerate(motions):
            wheel_velocity = np.column_stack((vx - yaw * pivot_xy[:, 1], vy + yaw * pivot_xy[:, 0]))
            angle = np.arctan2(wheel_velocity[:, 1], wheel_velocity[:, 0]) - zero_headings
            speed = np.linalg.norm(wheel_velocity, axis=1) / wheel_radius_m
            reverse = np.abs(angle) > math.pi / 2
            angle[reverse] -= np.sign(angle[reverse]) * math.pi
            speed[reverse] *= -1
            speed -= yaw * lateral_offsets / wheel_radius_m
            steering_target[case] = angle
            drive_target[case] = speed * (args.yaw_wheel_target_scale if names[case] == 'yaw' else 1.0)
        analytic_actions = torch.zeros(4, 14, device=environment.device)
        analytic_actions[:, :4] = torch.as_tensor(steering_target / cfg.control.action_scale, device=environment.device)
        analytic_actions[:, 4:8] = torch.as_tensor(drive_target / cfg.control.wheel_action_scale, device=environment.device)
        zeros = torch.zeros_like(analytic_actions)
        # Keep the legacy 1 s baseline; longer settling is a separate diagnostic.
        settle_trace = []
        for settle_step in range(round(args.settle_seconds / environment.dt)):
            applied_torque_peak = advance_interval(environment, zeros, args.cpu_contacts)
            settle_contacts = []
            if args.cpu_contacts:
                for contact in environment.gym.get_env_rigid_contacts(environment.envs[0]):
                    body0, body1 = int(contact['body0']), int(contact['body1'])
                    settle_contacts.append({
                        'pair': '|'.join(rigid_names[index] if index >= 0 else 'ground'
                                         for index in (body0, body1)),
                        'lambda_N': float(contact['lambda']),
                        'overlap_m': float(contact['initialOverlap']),
                    })
            settle_trace.append({
                'step': settle_step,
                'drive_velocity_rad_s': environment.dof_vel[:, environment.wheel_dof_indices].cpu().tolist(),
                'steering_position_rad': environment.dof_pos[:, environment.action_dof_indices[:4]].cpu().tolist(),
                'base_velocity_body_m_s_rad_s': torch.cat(
                    (environment.base_lin_vel, environment.base_ang_vel[:, 2:3]), dim=1
                ).cpu().tolist(),
                'base_position_world_m': environment.root_states[:, :3].cpu().tolist(),
                'base_rpy_rad': environment.rpy.cpu().tolist(),
                'wheel_position_world_m': environment.rigid_state[:, environment.wheel_body_indices, :3].cpu().tolist(),
                'wheel_contact_force_N': environment.contact_forces[:, environment.wheel_body_indices].cpu().tolist(),
                'base_contact_force_N': torch.linalg.vector_norm(
                    environment.contact_forces[:, environment.base_body_index], dim=1
                ).cpu().tolist(),
                'applied_torque_peak_Nm': (
                    applied_torque_peak.cpu().tolist()
                ),
                'contacts_env0': settle_contacts,
            })
        settled_state = {
            'drive_velocity_rad_s': environment.dof_vel[:, environment.wheel_dof_indices].detach().cpu().tolist(),
            'steering_position_rad': environment.dof_pos[:, environment.action_dof_indices[:4]].detach().cpu().tolist(),
            'base_velocity_body_m_s_rad_s': torch.cat(
                (environment.base_lin_vel, environment.base_ang_vel[:, 2:3]), dim=1
            ).detach().cpu().tolist(),
        }
        model = None
        if args.checkpoint:
            from legged_gym.b2_gym_learn.ppo_cse_pf.v3_actor_critic import V3ActorCritic

            payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
            model = V3ActorCritic(environment.num_obs, environment.num_privileged_obs,
                                  cfg.env.num_pred_obs, cfg.env.num_single_obs,
                                  environment.num_actions, **payload['config_snapshot']['training']['policy']).to(environment.device)
            model.load_state_dict(payload['model_state_dict'], strict=True)
            model.eval()
        environment.commands[:, :3] = torch.as_tensor(motions, device=environment.device)
        environment.commands[:, 3:6] = environment._tcp_spherical_position()
        environment._publish_observations(torch.arange(4, device=environment.device), initial=True)
        traces = []
        for step in range(400):
            if model is None:
                actions = analytic_actions
            else:
                with torch.inference_mode():
                    actions = model.act_inference(environment.get_observations())
            applied_torque_peak = advance_interval(environment, actions, args.cpu_contacts)
            environment.last_actions.copy_(actions)
            environment._publish_observations(torch.arange(4, device=environment.device))
            requested = actions[:, 4:8] * cfg.control.wheel_action_scale
            drive_velocity = environment.dof_vel[:, environment.wheel_dof_indices]
            steering_position = environment.dof_pos[:, environment.action_dof_indices[:4]]
            post_interval_feedback_torque = environment._compute_v3_torques(actions)[:, environment.wheel_dof_indices]
            wheel_contact = environment.contact_forces[:, environment.wheel_body_indices]
            for case in range(4):
                if args.cpu_contacts:
                    for contact in environment.gym.get_env_rigid_contacts(environment.envs[case]):
                        body0, body1 = int(contact['body0']), int(contact['body1'])
                        if environment.base_body_index not in (body0, body1):
                            continue
                        pair = '|'.join(rigid_names[index] if index >= 0 else 'ground' for index in (body0, body1))
                        entry = base_pairs[names[case]].setdefault(pair, {'count': 0, 'peak_lambda': 0.0})
                        entry['count'] += 1
                        if float(contact['lambda']) >= entry['peak_lambda']:
                            entry.update({'peak_lambda': float(contact['lambda']), 'step': step,
                                          'overlap_m': float(contact['initialOverlap']),
                                          'local_position0_m': [float(value) for value in contact['localPos0']],
                                          'local_position1_m': [float(value) for value in contact['localPos1']]})
                traces.append({'step': step, 'case': names[case],
                               'body_velocity': [float(environment.base_lin_vel[case, 0]),
                                                 float(environment.base_lin_vel[case, 1]),
                                                 float(environment.base_ang_vel[case, 2])],
                               'drive_target_rad_s': torch.clamp(requested[case], -20, 20).cpu().tolist(),
                               'steering_target_rad': steering_target[case].tolist(),
                               'actual_drive_rad_s': drive_velocity[case].cpu().tolist(),
                               'actual_steering_rad': steering_position[case].cpu().tolist(),
                               'applied_drive_torque_peak_Nm': (
                                   applied_torque_peak[case, environment.wheel_dof_indices].cpu().tolist()
                               ),
                               'post_interval_feedback_torque_Nm': post_interval_feedback_torque[case].cpu().tolist(),
                               'wheel_contact_force_N': wheel_contact[case].cpu().tolist(),
                               'base_contact_N': float(torch.linalg.vector_norm(environment.contact_forces[case, environment.base_body_index])),
                               'rpy_rad': environment.rpy[case].cpu().tolist()})
        report: Dict[str, Any] = {'gain_scale': args.gain_scale, 'controller': 'policy' if model else 'analytic',
                                  'asset': cfg.asset.file,
                                  'physics_dt_s': cfg.sim.dt, 'decimation': cfg.control.decimation,
                                  'policy_dt_s': environment.dt,
                                  'rigid_body_properties': read_properties(environment),
                                  'base_height_m': cfg.init_state.pos[2],
                                  'self_collision_disabled': args.disable_self_collision,
                                  'nominal': True, 'seed': 1, 'wheel_radius_m': wheel_radius_m.tolist(),
                                  'settle_duration_s': args.settle_seconds,
                                  'duration_s': 8, 'summary_window_s': [4, 8], 'cases': {},
                                  'settled_state_before_command': settled_state,
                                  'settle_trace': settle_trace,
                                  'cpu_contacts': args.cpu_contacts, 'base_contact_pairs': base_pairs,
                                  'steering_gain_scale': args.steering_gain_scale,
                                  'yaw_wheel_target_scale': args.yaw_wheel_target_scale,
                                  'motion_sign': args.motion_sign, 'analytic_kinematics': 'steering_pivot_v1'}
        imported_props = environment.gym.get_actor_dof_properties(environment.envs[0], environment.actor_handles[0])
        report['dof_properties'] = {
            name: {key: float(imported_props[key][index]) for key in ('driveMode', 'stiffness', 'damping', 'effort')}
            for index, name in enumerate(environment.dof_names)
        }
        report['configured_pd_gains'] = {
            name: {'stiffness': float(environment.p_gains[index]),
                   'damping': float(environment.d_gains[index])}
            for index, name in enumerate(environment.dof_names)
        }
        for case, name in enumerate(names):
            rows = [row for row in traces if row['case'] == name and row['step'] >= 200]
            velocities = np.asarray([row['body_velocity'] for row in rows])
            targets = np.asarray([row['drive_target_rad_s'] for row in rows])
            actual = np.asarray([row['actual_drive_rad_s'] for row in rows])
            report['cases'][name] = {'command': motions[case].tolist(),
                                     'body_velocity_mean': velocities.mean(axis=0).tolist(),
                                     'body_velocity_error_mean': np.abs(velocities - motions[case]).mean(axis=0).tolist(),
                                     'drive_target_mean_rad_s': targets.mean(axis=0).tolist(),
                                     'drive_actual_mean_rad_s': actual.mean(axis=0).tolist(),
                                     'drive_speed_mae_rad_s': np.abs(actual - targets).mean(axis=0).tolist(),
                                     'steering_mean_rad': np.mean([row['actual_steering_rad'] for row in rows], axis=0).tolist(),
                                     'applied_drive_torque_peak_rms_Nm': np.sqrt(np.mean(np.square([
                                         row['applied_drive_torque_peak_Nm'] for row in rows
                                     ]), axis=0)).tolist(),
                                     'peak_base_contact_N': max(row['base_contact_N'] for row in traces if row['case'] == name),
                                     'max_abs_roll_pitch_rad': np.max(np.abs([row['rpy_rad'][:2] for row in rows]), axis=0).tolist()}
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        trace_path = args.output.with_suffix('.trace.jsonl')
        trace_path.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in traces))
        print(json.dumps({"output": str(args.output), "cases": report["cases"]}, allow_nan=False), flush=True)
    finally:
        environment.gym.destroy_sim(environment.sim)


if __name__ == '__main__':
    main()

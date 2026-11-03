"""Run isolated physical and sampling checks through the actual V3 environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

from isaacgym import gymapi, gymtorch, gymutil
import numpy as np
import torch


def read_properties(environment: Any) -> list[dict]:
    """Read all bodies from the first actor after real creation randomization.

    Args:
        environment: Fully constructed V3PosForce GPU environment.

    Returns:
        Named mass (kg), COM (m) and inertia (kg m squared) records.
    """
    gym = environment.gym
    handle, actor = environment.envs[0], environment.actor_handles[0]
    names = gym.get_actor_rigid_body_names(handle, actor)
    properties = gym.get_actor_rigid_body_properties(handle, actor)
    return [{'name': name, 'mass': prop.mass,
             'com': [prop.com.x, prop.com.y, prop.com.z],
             'inertia': [[getattr(getattr(prop.inertia, row), column) for column in 'xyz'] for row in 'xyz']}
            for name, prop in zip(names, properties)]


def run_probe(arguments: argparse.Namespace) -> dict:
    """Create an isolated environment and measure the actual step/reset path.

    Args:
        arguments: Source/asset paths, environment count, steps, nominal flag
            and optional isolated physics-only timing mode.
            CUDA_VISIBLE_DEVICES must select an otherwise available physical GPU.

    Returns:
        Physical, reset-isolation and sampling timing evidence with limitations.

    Raises:
        RuntimeError: Environment import, physical validation or reset checks fail.

    Note:
        Runs no optimizer and writes no training checkpoint. Nominal diagnostics
        disable randomization/push only in their own local configuration instance.
        Other diagnostics retain the released V3 randomization settings.
    """
    sys.path.insert(0, str(arguments.source_root))
    from legged_gym.envs.v3.legged_robot_v3 import V3PosForce
    from legged_gym.envs.v3.v3_pos_force_config import V3PosForceCfg
    from legged_gym.utils.helpers import class_to_dict, set_seed

    set_seed(1)
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    cfg = V3PosForceCfg()
    cfg.asset.file = str(arguments.asset.resolve())
    cfg.env.num_envs = arguments.num_envs
    if arguments.nominal:
        for name in ('randomize_base_mass', 'randomize_base_com', 'randomize_tool_mass',
                     'randomize_friction', 'randomize_execution_scale', 'push_robots'):
            setattr(cfg.domain_rand, name, False)
    parameters = gymapi.SimParams()
    gymutil.parse_sim_config(class_to_dict(cfg.sim), parameters)
    parameters.use_gpu_pipeline = True
    parameters.physx.use_gpu = True
    started = time.perf_counter()
    environment = V3PosForce(cfg, parameters, gymapi.SIM_PHYSX, 'cuda:0', True)
    torch.cuda.synchronize()
    creation_seconds = time.perf_counter() - started
    try:
        report = {'asset': str(arguments.asset), 'source_root': str(arguments.source_root),
                  'num_envs': arguments.num_envs, 'nominal': arguments.nominal,
                  'creation_seconds': creation_seconds, 'properties': read_properties(environment),
                  'shape_count': len(environment.gym.get_actor_rigid_shape_properties(
                      environment.envs[0], environment.actor_handles[0])),
                  'mass_com_delta': environment.mass_com_delta.cpu().tolist(),
                  'friction': environment.friction.cpu().tolist()}
        actions = torch.zeros(arguments.num_envs, 14, device=environment.device)
        failure_snapshots = []
        original_reset = environment.reset_idx

        def capture_reset(env_ids: torch.Tensor) -> None:
            """Capture the first physical failure before reset replaces its state.

            Args:
                env_ids: Environment indices selected by the real step path.

            Returns:
                None.

            Note:
                Stores one diagnostic snapshot and delegates to the original
                reset without changing its inputs or simulator state.
            """
            has_physical_failure = bool((
                torch.linalg.vector_norm(environment.contact_forces[env_ids, environment.base_body_index], dim=-1) > 1.0
            ).any()) if env_ids.numel() else False
            if not failure_snapshots and has_physical_failure:
                env_id = int(env_ids[0])
                failure_snapshots.append({
                    'env_id': env_id,
                    'episode_step': int(environment.episode_length_buf[env_id]),
                    'root_state': environment.root_states[env_id].cpu().tolist(),
                    'dof_positions_rad': dict(zip(environment.dof_names,
                                                 environment.dof_pos[env_id].cpu().tolist())),
                    'body_contact_forces_N': environment.contact_forces[env_id].cpu().tolist(),
                })
            original_reset(env_ids)

        environment.reset_idx = capture_reset
        for _ in range(4):
            environment.step(actions)
        torch.cuda.synchronize()
        if arguments.physics_only:
            # This fresh-process segment starts from the same seeded creation
            # and four warm-up steps as the full-step timing. It measures PD,
            # submission, simulation and DOF refresh, without task/reset logic.
            # Diverging trajectories make it a separate benchmark, not an
            # exact subtraction-based decomposition of full environment step.
            physics_times = []
            for _ in range(arguments.steps):
                started = time.perf_counter()
                for _ in range(cfg.control.decimation):
                    environment.torques = environment._compute_v3_torques(actions)
                    environment.gym.set_dof_actuation_force_tensor(
                        environment.sim, gymtorch.unwrap_tensor(environment.torques))
                    environment.gym.simulate(environment.sim)
                    environment.gym.refresh_dof_state_tensor(environment.sim)
                torch.cuda.synchronize()
                physics_times.append(time.perf_counter() - started)
            report.update({'steps': arguments.steps,
                           'physics_policy_step_mean_s': float(np.mean(physics_times)),
                           'physics_policy_step_p95_s': float(np.percentile(physics_times, 95)),
                           'limitations_zh': '独立进程、相同初始化与4步预热；仅PD／提交／simulate／DOF刷新，轨迹与含reset的完整step可能分离，不做精确占比相减。'})
            return report
        step_times, failures, peak_base_force = [], 0, 0.0
        peak_body_forces = torch.zeros(environment.num_bodies, device=environment.device)
        for _ in range(arguments.steps):
            started = time.perf_counter()
            _, _, _, infos = environment.step(actions)
            torch.cuda.synchronize()
            step_times.append(time.perf_counter() - started)
            failures += int(infos['terminated'].sum())
            force = torch.linalg.vector_norm(environment.contact_forces[:, environment.base_body_index], dim=-1)
            peak_base_force = max(peak_base_force, float(force.max()))
            peak_body_forces = torch.maximum(
                peak_body_forces,
                torch.linalg.vector_norm(environment.contact_forces, dim=-1).max(dim=0).values)
        report.update({'steps': arguments.steps, 'physical_failure_events': failures,
                       'first_failure_snapshots': failure_snapshots,
                       'peak_base_force_N': peak_base_force,
                       'peak_body_forces_N': dict(zip(
                           environment.gym.get_actor_rigid_body_names(
                               environment.envs[0], environment.actor_handles[0]),
                           peak_body_forces.cpu().tolist())),
                       'step_mean_s': float(np.mean(step_times)), 'step_p95_s': float(np.percentile(step_times, 95)),
                       'rollout_24_estimate_s': float(np.mean(step_times) * 24),
                       'root_z_range_m': [float(environment.root_states[:, 2].min()),
                                          float(environment.root_states[:, 2].max())]})
        # Exercise indexed reset while preserving every nonselected tensor row.
        fields = ('root_states', 'dof_state', 'obs_history', 'critic_history', 'episode_length_buf',
                  'commands', 'actions', 'last_actions', 'execution_scale_delta')
        saved = {}
        for name in fields:
            value = getattr(environment, name)
            # Preview4 stores DOF state as [num_envs * 14, 2], while all
            # task tensors are indexed directly by environment.
            saved[name] = value[14:].clone() if name == 'dof_state' else value[1:].clone()
        environment.reset_idx(torch.tensor([0], device=environment.device))
        reset_differences = {
            name: float((previous - (getattr(environment, name)[14:] if name == 'dof_state'
                                   else getattr(environment, name)[1:])).abs().max().item())
            for name, previous in saved.items()
        }
        report['local_reset_other_rows_max_abs_difference'] = reset_differences
        report['local_reset_other_rows_unchanged'] = {
            name: difference <= 1e-5 for name, difference in reset_differences.items()
        }
        if not all(report['local_reset_other_rows_unchanged'].values()):
            raise RuntimeError(f'Local reset changed other environments: {reset_differences}')
        if arguments.nominal:
            # Small wheel/steering actions for 5 s; arm pose remains nominal.
            # The policy order is steering[4], drive[4], arm[6], whereas Gym
            # imports arm joints first. Never use Gym indices as policy indices.
            actions[:, :4] = 0.4
            actions[:, 4:8] = 0.1
            movement_failures = 0
            for _ in range(250):
                _, _, _, infos = environment.step(actions)
                movement_failures += int(infos['terminated'].sum())
            torch.cuda.synchronize()
            report['movement_physical_failure_events'] = movement_failures
            report['movement_dof_mean_positions_rad'] = dict(zip(environment.dof_names, environment.dof_pos.mean(0).cpu().tolist()))
            report['movement_dof_mean_velocities_rad_s'] = dict(zip(environment.dof_names, environment.dof_vel.mean(0).cpu().tolist()))
            environment.reset_idx(torch.arange(arguments.num_envs, device=environment.device))
            actions.zero_()
            actions[:, 8:] = 0.1
            arm_failures = 0
            for _ in range(250):
                _, _, _, infos = environment.step(actions)
                arm_failures += int(infos['terminated'].sum())
            torch.cuda.synchronize()
            report['arm_physical_failure_events'] = arm_failures
            report['arm_dof_mean_positions_rad'] = dict(zip(
                environment.dof_names, environment.dof_pos.mean(0).cpu().tolist()))
        report['torch_peak_memory_bytes'] = torch.cuda.max_memory_allocated()
        report['limitations_zh'] = '零动作／小动作独立诊断，无PPO；不是训练后能力验收，24步估算不是完整训练轮耗时。'
        return report
    finally:
        environment.gym.destroy_sim(environment.sim)


def main() -> None:
    """Parse isolated probe inputs and write machine-readable evidence.

    Returns:
        None.

    Note:
        Writes only the specified report path; preserves all formal run files.
    """
    parser = argparse.ArgumentParser(description='V3候选真实创建、物理与采样短测')
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--asset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num-envs', type=int, default=32)
    parser.add_argument('--steps', type=int, default=250)
    parser.add_argument('--nominal', action='store_true')
    parser.add_argument('--physics-only', action='store_true')
    arguments = parser.parse_args()
    report = run_probe(arguments)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in (
        'shape_count', 'physical_failure_events', 'step_mean_s',
        'rollout_24_estimate_s', 'physics_policy_step_mean_s') if key in report}, ensure_ascii=False))


if __name__ == '__main__':
    main()

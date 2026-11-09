"""确定性核对原V3命令槽、坐标、特权目标与发布历史，不创建仿真。"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import isaacgym  # noqa: F401
import torch

import legged_gym

SOURCE = Path(legged_gym.__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / 'tests'))
from test_v3_rewards import make_reward_environment


def root_quaternion(roll: float, pitch: float, yaw: float) -> torch.Tensor:
    """Construct a scalar-first-free xyzw quaternion from ZYX Euler angles.

    Args:
        roll: Rotation about local X, in radians.
        pitch: Rotation about local Y, in radians.
        yaw: World heading about Z, in radians.

    Returns:
        Float32 xyzw quaternion [4] for setting the tensor fixture root pose.
    """
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return torch.tensor((sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy))


def main() -> None:
    """Exercise original frame/coordinate methods against worked world formulas.

    Raises:
        AssertionError: Command indices, scaling, coordinate transforms, history,
            truth isolation, or pending/applied semantics fail independent checks.

    Note:
        Operates on mock tensor fixtures only. No native simulator or optimizer is
        created, and no production source, checkpoint or training state is written.
    """
    torch.manual_seed(1001)
    environment = make_reward_environment()
    environment.default_dof_pos = torch.zeros(1, 14)
    environment.mass_com_delta = torch.zeros(2, 5)
    environment.friction = torch.ones(2, 1)
    environment.execution_scale_delta = torch.zeros(2, 14)
    environment.wheel_body_indices = torch.arange(19, 23)
    environment.contact_valid = torch.ones(2, dtype=torch.bool)
    environment.obs_history = torch.zeros(2, 32, 58)
    environment.critic_history = torch.zeros(2, 3, 98)
    environment.obs_pred = torch.zeros(2, 12)
    environment.applied_ee_force_world = torch.zeros(2, 3)
    environment.pending_ee_force_world = torch.zeros(2, 3)
    ids = torch.arange(2)
    environment.cfg.run.reproduction_part = 'B'
    environment.cfg.noise.add_noise = False
    errors = {'actor_force_scaled': 0., 'geometric_world_m': 0., 'compensated_world_m': 0.,
        'external_label_scaled': 0., 'critic_target_world_m': 0.}
    count = 0
    examples = []
    for yaw in (-math.pi, -math.pi / 2, 0., math.pi / 2, math.pi):
        rotation = torch.tensor(((math.cos(yaw), -math.sin(yaw), 0.),
            (math.sin(yaw), math.cos(yaw), 0.), (0., 0., 1.)))
        environment.root_states[:, :3] = torch.tensor((2., -3., .47))
        environment.root_states[:, 3:7] = root_quaternion(.2, -.3, yaw)
        environment.env_origins[:, 2] = .1
        for axis in range(3):
            for sign in (-1., 1.):
                for external_enabled in (False, True):
                    command = torch.zeros(3)
                    command[axis] = sign * 30.
                    external = torch.tensor((20., -13., 7.)) if external_enabled else torch.zeros(3)
                    environment.commands.zero_()
                    environment.commands[:, 3:6] = torch.tensor((.55, .25, -.3))
                    environment.commands[:, 9:12] = command
                    environment.applied_ee_force_world[:] = external
                    environment.pending_ee_force_world[:] = torch.tensor((-59., 51., -47.))
                    environment._update_base_state()
                    environment._update_applied_force_label()
                    actor, critic, labels = environment._build_frames(ids)
                    command_world = rotation @ command
                    geometric = torch.tensor((2. + .174 * math.cos(yaw)
                        + .55 * math.cos(.25) * math.cos(-.3 + yaw),
                        -3. + .174 * math.sin(yaw) + .55 * math.cos(.25) * math.sin(-.3 + yaw),
                        .1 + environment.cfg.rewards.base_height_target + .09535 + .55 * math.sin(.25)))
                    expected_target = geometric + (external + command_world) / 200.
                    sphere = critic[:, 95:98] / torch.tensor((.5, 1., 1.3))
                    radius, elevation, azimuth = sphere.unbind(-1)
                    recovered = torch.stack((2. + .174 * math.cos(yaw) + radius * elevation.cos() * (azimuth + yaw).cos(),
                        -3. + .174 * math.sin(yaw) + radius * elevation.cos() * (azimuth + yaw).sin(),
                        .1 + environment.cfg.rewards.base_height_target + .09535 + radius * elevation.sin()), -1)
                    measured = {
                        'actor_force_scaled': (actor[:, 52:55] - command * .01).abs().max(),
                        'geometric_world_m': (environment._geometric_target_world() - geometric).abs().max(),
                        'compensated_world_m': (environment._compensated_target_world() - expected_target).abs().max(),
                        'external_label_scaled': (labels[:, 6:9] - (rotation.T @ external) * .01).abs().max(),
                        'critic_target_world_m': (recovered - expected_target).abs().max(),
                    }
                    for name, error in measured.items():
                        errors[name] = max(errors[name], float(error))
                        assert float(error) < 2e-5, (yaw, axis, sign, name, float(error))
                    torch.testing.assert_close(actor[:, 46:49], environment.commands[:, 3:6] * torch.tensor((.5, 1., 1.3)))
                    # Pending force must not leak into the current completed-state
                    # labels or the Actor through a privileged compensated target.
                    environment.pending_ee_force_world.neg_()
                    actor_pending, critic_pending, labels_pending = environment._build_frames(ids)
                    torch.testing.assert_close(actor_pending, actor, rtol=0., atol=0.)
                    torch.testing.assert_close(critic_pending, critic, rtol=0., atol=0.)
                    torch.testing.assert_close(labels_pending, labels, rtol=0., atol=0.)
                    previous_external = environment.external_ee_force_y.clone()
                    environment.external_ee_force_y[:] += 11.
                    actor_external, _, _ = environment._build_frames(ids)
                    torch.testing.assert_close(actor_external, actor, rtol=0., atol=0.)
                    environment.external_ee_force_y.copy_(previous_external)
                    environment.obs_history.fill_(.123)
                    environment.critic_history.fill_(.456)
                    environment._publish_observations(ids)
                    history = environment.get_observations()['obs'].view(2, 32, 58)
                    torch.testing.assert_close(history[:, :-1], torch.full((2, 31, 58), .123), rtol=0., atol=0.)
                    torch.testing.assert_close(history[:, -1], actor, rtol=0., atol=0.)
                    if yaw in (0., math.pi / 2) and sign > 0 and not external_enabled:
                        examples.append({'yaw_rad': yaw, 'command_axis': 'XYZ'[axis],
                            'command_yaw_n': command.tolist(), 'actor_force_slot': actor[0, 52:55].tolist(),
                            'world_target_offset_m': (expected_target - geometric).tolist()})
                    count += 1
    result = {'passed': True, 'cases': count, 'yaw_values_rad': [-math.pi, -math.pi / 2, 0., math.pi / 2, math.pi],
        'max_absolute_errors': errors, 'actor_force_slice': [52, 55], 'actor_target_slice': [46, 49],
        'critic_compensated_target_slice': [95, 98], 'force_scale_per_n': .01,
        'pending_force_isolated': True, 'external_truth_isolated_from_actor': True,
        'history_append_checked': True, 'examples': examples,
        '说明': '原帧构建与坐标方法，roll/pitch非零、不同yaw与正负命令；仅tensor fixture，不替代真实物理效果验证。'}
    output = Path(__file__).resolve().parent / 'tensor_audit.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

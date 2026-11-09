"""分解原V3控制器的动作、目标和力矩，仅作无状态诊断。"""

from __future__ import annotations

import torch


def decompose_requests(
    raw_actions: torch.Tensor,
    position: torch.Tensor,
    velocity: torch.Tensor,
    default_position: torch.Tensor,
    position_limits: torch.Tensor,
    execution_delta: torch.Tensor,
    proportional_gains: torch.Tensor,
    damping_gains: torch.Tensor,
    effort_limits: torch.Tensor,
    action_clip: float,
    position_scale: float,
    wheel_scale: float,
    wheel_limit: float,
) -> dict[str, torch.Tensor]:
    """Evaluate original requests for several actions at identical physical states.

    Args:
        raw_actions: Dimensionless candidate actions [V,N,14], in policy order:
            four steering joints, four drive joints, then six arm joints.
        position: Current joint positions [N,14], in radians and policy order.
        velocity: Joint velocities [N,14], in rad/s and policy order.
        default_position: Nominal positions [N,14], in radians.
        position_limits: Hard joint limits [14,2], in radians.
        execution_delta: Per-episode dimensionless execution deltas [N,14].
        proportional_gains: Position gains [14], in N m/rad.
        damping_gains: Velocity gains [14], in N m s/rad.
        effort_limits: Positive joint torque limits [14], in N m.
        action_clip: Original symmetric dimensionless action bound.
        position_scale: Position residual scale, in rad/action unit.
        wheel_scale: Drive speed scale, in rad/s per action unit.
        wheel_limit: Symmetric drive speed bound, in rad/s.

    Returns:
        Tensors [V,N,14] for actions, targets, position P and velocity terms,
        unclipped/clipped torques, and saturation flags. Target units are rad
        except drive slots 4:8, which are rad/s. Torque units are N m.

    Note:
        Inputs are never mutated; no physics or policy update is performed.
    """
    clipped_actions = raw_actions.clamp(-action_clip, action_clip)
    scaled_actions = clipped_actions * (1.0 + execution_delta)
    # Keep targets in policy order while respecting the two distinct units.
    raw_target = default_position + position_scale * scaled_actions
    raw_target = raw_target.clone()
    raw_target[..., 4:8] = wheel_scale * scaled_actions[..., 4:8]
    target = torch.maximum(torch.minimum(raw_target, position_limits[:, 1]), position_limits[:, 0])
    target[..., 4:8] = raw_target[..., 4:8].clamp(-wheel_limit, wheel_limit)
    position_p = proportional_gains * (target - position)
    position_p[..., 4:8] = 0.0
    velocity_term = -damping_gains * velocity
    velocity_term = velocity_term.expand_as(raw_actions).clone()
    velocity_term[..., 4:8] = damping_gains[4:8] * (target[..., 4:8] - velocity[..., 4:8])
    raw_torque = position_p + velocity_term
    torque = torch.maximum(torch.minimum(raw_torque, effort_limits), -effort_limits)
    return {
        "clipped_actions": clipped_actions,
        "raw_target": raw_target,
        "target": target,
        "position_p": position_p,
        "velocity_term": velocity_term,
        "raw_torque": raw_torque,
        "torque": torque,
        "saturated": torque.abs() >= effort_limits * 0.999,
    }

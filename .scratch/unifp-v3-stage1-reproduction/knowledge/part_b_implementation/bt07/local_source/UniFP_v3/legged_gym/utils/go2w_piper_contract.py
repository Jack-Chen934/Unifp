"""Validate and map Go2-W/Piper policy and scripted gripper commands."""

from typing import Any, Dict, Mapping, Sequence

import torch


def map_policy_actions(actions: torch.Tensor, dof_names: Sequence[str],
                       joint_map: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    """Map a batched 22-channel policy tensor to actuator names.

    Args:
        actions: Physical target tensor shaped ``[num_envs, 22]``; position targets
            use radians and wheel velocity targets use radians per second.
        dof_names: Unique simulator DOF names available on the loaded actor.
        joint_map: ``JointMapV1`` mapping stored in the combined asset manifest.

    Returns:
        Mapping from each policy-controlled joint name to its ``[num_envs]`` target.

    Raises:
        ValueError: The tensor shape, values, joint map or simulator names are invalid.
        TypeError: Policy actions are not floating-point tensors.
    """
    policy_channels = joint_map.get("policy_actions", [])
    if joint_map.get("version") != "JointMapV1" or len(policy_channels) != 22:
        raise ValueError("Expected a 22-channel JointMapV1")
    if actions.ndim != 2 or actions.shape[1] != len(policy_channels):
        raise ValueError(f"Expected action shape [num_envs, 22], got {tuple(actions.shape)}")
    if not torch.is_floating_point(actions):
        raise TypeError("Policy action tensor must use a floating-point dtype")
    if not torch.isfinite(actions).all():
        raise ValueError("Policy action tensor contains non-finite values")
    names = list(dof_names)
    if len(names) != len(set(names)):
        raise ValueError("Simulator contains duplicate DOF names")
    required_names = [channel["joint"] for channel in policy_channels]
    required_names.extend(channel["joint"] for channel in joint_map.get("scripted_gripper", []))
    missing_names = sorted(set(required_names) - set(names))
    if missing_names:
        raise ValueError(f"Simulator is missing required DOFs: {missing_names}")
    ordered_channels = sorted(policy_channels, key=lambda channel: channel["index"])
    if [channel["index"] for channel in ordered_channels] != list(range(22)):
        raise ValueError("JointMapV1 action indices must be contiguous from 0 through 21")
    if len({channel["joint"] for channel in ordered_channels}) != 22:
        raise ValueError("JointMapV1 policy joint names must be unique")
    return {channel["joint"]: actions[:, channel["index"]]
            for channel in ordered_channels}


def scripted_gripper_targets(opening_m: torch.Tensor,
                             joint_map: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    """Create symmetric position targets for the two non-policy gripper fingers.

    Args:
        opening_m: Per-environment finger travel in meters, shaped ``[num_envs]``;
            zero is closed and positive values move each finger outward equally.
        joint_map: ``JointMapV1`` scripted gripper limits from the asset manifest.

    Returns:
        Joint-name to per-environment target tensor in meters.

    Raises:
        ValueError: The mapping is incomplete, travel is invalid or limits are asymmetric.
        TypeError: Opening targets are not a floating-point tensor.
    """
    gripper_channels = {channel["joint"]: channel
                        for channel in joint_map.get("scripted_gripper", [])}
    if set(gripper_channels) != {"joint7", "joint8"}:
        raise ValueError("Scripted gripper map must contain joint7 and joint8")
    if opening_m.ndim != 1:
        raise ValueError(f"Expected gripper opening shape [num_envs], got {tuple(opening_m.shape)}")
    if not torch.is_floating_point(opening_m):
        raise TypeError("Gripper opening tensor must use a floating-point dtype")
    if not torch.isfinite(opening_m).all():
        raise ValueError("Gripper opening tensor contains non-finite values")
    joint7 = gripper_channels["joint7"]
    joint8 = gripper_channels["joint8"]
    maximum_opening_m = min(float(joint7["upper"]), -float(joint8["lower"]))
    if not torch.all((opening_m >= 0.0) & (opening_m <= maximum_opening_m)):
        raise ValueError(f"Gripper travel must be between 0 and {maximum_opening_m} m")
    return {"joint7": opening_m, "joint8": -opening_m}

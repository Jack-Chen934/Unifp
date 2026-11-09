"""Named-joint torque control matching UniFP's B2+Z1 low-level controller.

This module deliberately does not import Isaac Gym. The joint name, default pose,
gain and action scale values mirror b2z1_pos_force_config.py. The control equation
mirrors LeggedRobotB2Z1PosForce._compute_torques in the released UniFP repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import torch


ROOT = Path(__file__).resolve().parent
URDF_PATH = ROOT / "UniFP/resources/robots/b2z1/b2z1.urdf"
USD_PATH = ROOT / "assets/b2z1_merged/b2z1.usd"
PHYSICS_DT = 0.005
DECIMATION = 4
ACTION_SCALE = 0.25

# Explicit order of the released 17-dimensional policy output, independent of
# the joint ordering produced by the Isaac Lab URDF importer.
ACTION_JOINT_NAMES = (
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "z1_waist", "z1_shoulder", "z1_elbow", "z1_wrist_angle", "z1_forearm_roll",
)
HOLD_JOINT_NAMES = ("z1_wrist_rotate", "z1_jointGripper")
ALL_JOINT_NAMES = ACTION_JOINT_NAMES + HOLD_JOINT_NAMES

DEFAULT_ANGLES = {
    **{f"{leg}_hip_joint": 0.15 if leg.endswith("L") else -0.15 for leg in ("FL", "FR", "RL", "RR")},
    **{f"{leg}_thigh_joint": 0.67 for leg in ("FL", "FR", "RL", "RR")},
    **{f"{leg}_calf_joint": -1.32 for leg in ("FL", "FR", "RL", "RR")},
    "z1_waist": 0.0,
    "z1_shoulder": 1.48,
    "z1_elbow": -1.5,
    "z1_wrist_angle": 0.0,
    "z1_forearm_roll": 0.0,
    "z1_wrist_rotate": 1.57,
    "z1_jointGripper": -0.785,
}


def gains(name: str) -> tuple[float, float]:
    if "hip" in name:
        return 300.0, 7.5
    if "thigh" in name:
        return 300.0, 7.5
    if "calf" in name:
        return 500.0, 12.5
    if name == "z1_shoulder":
        return 128.0, 3.0
    if name.startswith("z1_"):
        return 64.0, 1.5
    raise ValueError(f"Unconfigured UniFP joint: {name}")


def urdf_joint_limits(path: Path = URDF_PATH) -> dict[str, tuple[float, float]]:
    root = ET.parse(path).getroot()
    result = {}
    for joint in root.findall("joint"):
        if joint.get("type") == "fixed":
            continue
        limit = joint.find("limit")
        if limit is None:
            raise ValueError(f"Missing limit in URDF joint {joint.get('name')}")
        result[joint.get("name")] = float(limit.get("effort")), float(limit.get("velocity"))
    if set(result) != set(ALL_JOINT_NAMES):
        raise ValueError(f"URDF joint mismatch: {set(result) ^ set(ALL_JOINT_NAMES)}")
    return result


@dataclass
class NamedJointPD:
    sim_joint_names: tuple[str, ...]
    action_indices: torch.Tensor
    hold_indices: torch.Tensor
    default_positions: torch.Tensor
    kp: torch.Tensor
    kd: torch.Tensor
    effort_limits: torch.Tensor
    velocity_limits: torch.Tensor

    @classmethod
    def from_robot(cls, robot) -> "NamedJointPD":
        names = tuple(robot.joint_names)
        if len(names) != 19 or set(names) != set(ALL_JOINT_NAMES):
            raise ValueError(f"Imported joint names differ from UniFP: {names}")
        device = robot.data.joint_pos.device
        limits = urdf_joint_limits()
        action_indices = torch.tensor([names.index(n) for n in ACTION_JOINT_NAMES], device=device)
        hold_indices = torch.tensor([names.index(n) for n in HOLD_JOINT_NAMES], device=device)
        defaults = torch.tensor([DEFAULT_ANGLES[n] for n in names], device=device)
        kp = torch.tensor([gains(n)[0] for n in names], device=device)
        kd = torch.tensor([gains(n)[1] for n in names], device=device)
        efforts = torch.tensor([limits[n][0] for n in names], device=device)
        velocities = torch.tensor([limits[n][1] for n in names], device=device)
        return cls(names, action_indices, hold_indices, defaults, kp, kd, efforts, velocities)

    def compute(self, actions: torch.Tensor, joint_pos: torch.Tensor, joint_vel: torch.Tensor,
                motor_strength: torch.Tensor | None = None) -> torch.Tensor:
        """Return (num_envs, 19) torques in Isaac Lab joint order."""
        if actions.ndim != 2 or actions.shape != (joint_pos.shape[0], 17):
            raise ValueError(f"Expected (num_envs, 17) actions, received {tuple(actions.shape)}")
        if joint_pos.shape != joint_vel.shape or joint_pos.shape[1] != 19:
            raise ValueError("Expected matching (num_envs, 19) joint position and velocity")
        if motor_strength is not None:
            if motor_strength.shape != actions.shape:
                raise ValueError("Motor strength must match the (num_envs, 17) policy actions")
            actions = actions * motor_strength
        offsets = torch.zeros_like(joint_pos)
        offsets[:, self.action_indices] = ACTION_SCALE * actions
        torque = self.kp * (self.default_positions + offsets - joint_pos) - self.kd * joint_vel
        return torch.clamp(torque, min=-self.effort_limits, max=self.effort_limits)

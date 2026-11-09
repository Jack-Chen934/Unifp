"""Evaluate the V3 TCP directly from current root and joint tensors."""

from pathlib import Path
from typing import List, Sequence, Tuple
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import torch


class V3TcpKinematics:
    """Keep the active URDF arm chain independent of simulator body caches."""

    def __init__(self, urdf_path: Path, dof_names: Sequence[str], device: str) -> None:
        """Load fixed transforms and named DOF indices once for the active asset.

        Args:
            urdf_path: Active V3 URDF with a base_link to arm_seg6 chain, in meters.
            dof_names: Simulator DOF names in the same order as the position tensor.
            device: Torch device holding the environment state and chain constants.

        Raises:
            OSError: The active URDF cannot be read.
            KeyError: The TCP chain is disconnected from base_link.
            ValueError: An active chain joint is absent or is not rotational/fixed.

        Note:
            Reads the asset only during initialization; evaluation does not refresh
            simulator caches, step physics, or mutate input tensors.
        """
        robot = ET.parse(urdf_path).getroot()
        parent_joints = {joint.find('child').get('link'): joint for joint in robot.findall('joint')}
        self.chain: List[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        child_link = 'arm_seg6'
        # Traverse from TCP to root so each evaluation transforms just one point.
        while child_link != 'base_link':
            joint = parent_joints[child_link]
            joint_type = joint.get('type')
            if joint_type not in ('fixed', 'revolute', 'continuous'):
                raise ValueError(f'Unsupported V3 TCP joint type: {joint_type}')
            origin = joint.find('origin')
            translation_m = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            rpy_rad = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            origin_rotation = torch.tensor(Rotation.from_euler('xyz', rpy_rad).as_matrix(),
                                           dtype=torch.float32, device=device)
            axis_element = joint.find('axis')
            axis = (np.fromstring(axis_element.get('xyz', '1 0 0'), sep=' ')
                    if joint_type != 'fixed' and axis_element is not None else np.array([1, 0, 0]))
            axis = axis / np.linalg.norm(axis)
            dof_index = -1 if joint_type == 'fixed' else dof_names.index(joint.get('name'))
            self.chain.append((dof_index, origin_rotation,
                               torch.tensor(translation_m, dtype=torch.float32, device=device),
                               torch.tensor(axis, dtype=torch.float32, device=device)))
            child_link = joint.find('parent').get('link')
        self.tcp_offset_m = torch.tensor((0.0, 0.0, 0.07961206), device=device)

    def tcp_world(self, root_states: torch.Tensor, dof_positions: torch.Tensor) -> torch.Tensor:
        """Transform the fixed TCP offset through the current arm and root poses.

        Args:
            root_states: Current [N,13] root states: position m, xyzw quaternion,
                linear velocity m/s and angular velocity rad/s; velocities unused.
            dof_positions: Current [N,D] joint positions in radians, ordered as the
                constructor's dof_names. Wheel states do not affect this arm chain.

        Returns:
            World TCP positions [N,3] in meters, valid immediately after reset writes.
        """
        point_m = self.tcp_offset_m.expand(root_states.shape[0], 3)
        for dof_index, rotation, translation_m, axis in self.chain:
            if dof_index >= 0:
                angle_rad = dof_positions[:, dof_index: dof_index + 1]
                cosine, sine = torch.cos(angle_rad), torch.sin(angle_rad)
                # Rodrigues rotation applies the current joint motion in its frame.
                point_m = (point_m * cosine + torch.cross(axis.expand_as(point_m), point_m, dim=-1) * sine
                           + axis * (point_m * axis).sum(dim=-1, keepdim=True) * (1.0 - cosine))
            point_m = point_m @ rotation.T + translation_m
        quaternion_xyz = root_states[:, 3:6]
        cross_product = torch.cross(quaternion_xyz, point_m, dim=-1)
        rotated_point_m = point_m + 2.0 * (
            root_states[:, 6:7] * cross_product + torch.cross(quaternion_xyz, cross_product, dim=-1))
        return root_states[:, :3] + rotated_point_m

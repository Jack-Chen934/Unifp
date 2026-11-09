"""Compare batched TCP positions with an independent URDF graph evaluator."""

from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'legged_gym/utils'))
from compare_desktop_fk import forward_kinematics
from v3_kinematics import V3TcpKinematics


def test_tcp_uses_current_root_and_joint_states() -> None:
    """Match independent FK before any physics step and after indexed state writes.

    Returns:
        None. Asserts TCP agreement and unaffected rows across a partial reset.

    Raises:
        AssertionError: TCP position differs from the current URDF configuration.
    """
    asset_path = (Path(__file__).resolve().parents[1] / 'resources/robots/'
                  'v3_a1_desktop_train_v6_coplanar_r2/urdf/a1arm_rgm3_desktop_fixed_gripper.urdf')
    # Deliberately use Gym-style arm-first ordering rather than policy ordering.
    dof_names = [f'arm_joint{index}' for index in range(1, 7)] + [
        f'{corner}_{kind}_joint' for corner in ('bl', 'br', 'fl', 'fr')
        for kind in ('steering', 'wheel_drive')]
    generator = np.random.default_rng(17)
    root_states = torch.zeros(8, 13)
    root_states[:, :3] = torch.as_tensor(generator.uniform(-2, 2, (8, 3)), dtype=torch.float32)
    root_states[:, 3:7] = torch.as_tensor(
        Rotation.from_euler('xyz', generator.uniform(-0.5, 0.5, (8, 3))).as_quat(),
        dtype=torch.float32)
    dof_positions = torch.zeros(8, 14)
    dof_positions[:, :6] = torch.as_tensor(generator.uniform(-0.5, 0.5, (8, 6)), dtype=torch.float32)
    dof_positions[:, 1] += 1.0
    dof_positions[:, 2] -= 1.5
    kinematics = V3TcpKinematics(asset_path, dof_names, 'cpu')
    for reset_boundary in range(2):
        expected_tcp = []
        for index in range(8):
            root_pose = np.eye(4)
            root_pose[:3, :3] = Rotation.from_quat(root_states[index, 3:7].numpy()).as_matrix()
            root_pose[:3, 3] = root_states[index, :3].numpy()
            link_poses = forward_kinematics(
                asset_path, dict(zip(dof_names, dof_positions[index].tolist())), root_pose)
            expected_tcp.append((link_poses['arm_seg6'] @ np.array([0, 0, 0.07961206, 1]))[:3])
        actual_tcp = kinematics.tcp_world(root_states, dof_positions)
        torch.testing.assert_close(actual_tcp, torch.tensor(np.asarray(expected_tcp), dtype=torch.float32),
                                   atol=1e-5, rtol=1e-5)
        if reset_boundary == 0:
            # Reset only one row without any simulator/cache refresh operation.
            untouched_tcp = actual_tcp[1:].clone()
            root_states[0, 0] += 0.7
            dof_positions[0, 1] += 0.2
        else:
            torch.testing.assert_close(actual_tcp[1:], untouched_tcp, atol=0, rtol=0)

"""Verify peak-load diagnostics respect both frozen force coordinate bounds."""

from types import SimpleNamespace

import torch
from tools.check_v3_b_capacity import PeakLoadEnvironment


def test_peak_load_remains_same_world_direction_and_inside_both_force_boxes() -> None:
    """Keep both forces aligned and bounded for every heading, including 45 degrees.

    Raises:
        AssertionError: World/yaw components exceed 60 N or frames disagree.
    """
    count = 65
    yaw_rad = torch.linspace(-torch.pi, torch.pi, count)
    environment = SimpleNamespace(
        cfg=SimpleNamespace(force_events=SimpleNamespace(max_force_n=60.)), device="cpu",
        rpy=torch.stack((torch.zeros(count), torch.zeros(count), yaw_rad), dim=1),
        pending_ee_force_world=torch.zeros(count, 3), commands=torch.zeros(count, 15))
    PeakLoadEnvironment._advance_force_events(environment, torch.arange(count))
    command_yaw_n = environment.commands[:, 9:12]
    assert bool((command_yaw_n.abs() <= 60.).all())
    assert bool((environment.pending_ee_force_world.abs() <= 60.).all())
    command_world_n = torch.stack((
        yaw_rad.cos() * command_yaw_n[:, 0] - yaw_rad.sin() * command_yaw_n[:, 1],
        yaw_rad.sin() * command_yaw_n[:, 0] + yaw_rad.cos() * command_yaw_n[:, 1],
        command_yaw_n[:, 2]), dim=1)
    torch.testing.assert_close(command_world_n, environment.pending_ee_force_world, atol=1e-5, rtol=0.)
    torch.testing.assert_close(environment.pending_ee_force_world,
                               torch.tensor((60., 0., 60.)).expand(count, 3))

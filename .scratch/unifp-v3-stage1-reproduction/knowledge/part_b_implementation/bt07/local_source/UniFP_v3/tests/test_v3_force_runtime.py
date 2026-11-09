"""Check the approved six-direction load-scan script at its pure tensor seam."""

import isaacgym  # noqa: F401
import torch


def test_load_scan_separates_command_external_same_and_matched_forces() -> None:
    """Give all eight conditions a +X plateau, then the final -Z plateau.

    Raises:
        AssertionError: A command becomes an artificial force, matched signs fail
            to cancel, or the six events drift from the frozen 1/4/.../16 s starts.
    """
    from tools.check_v3_force_runtime import range_force_script

    env_ids = torch.arange(0, 32, 4)
    command, external, direction = range_force_script(torch.full((8,), 2.), env_ids, 10.)
    torch.testing.assert_close(command[:, 0], torch.tensor([10., 0., 10., -10.] * 2))
    torch.testing.assert_close(external[:, 0], torch.tensor([0., 10., 10., 10.] * 2))
    assert direction.tolist() == [0] * 8
    command, external, direction = range_force_script(torch.full((8,), 17.), env_ids, 20.)
    torch.testing.assert_close(command[:, 2], torch.tensor([-20., 0., -20., 20.] * 2))
    assert direction.tolist() == [5] * 8
    command, external, _ = range_force_script(torch.full((8,), 19.), env_ids, 60.)
    assert torch.count_nonzero(command) == torch.count_nonzero(external) == 0

"""Command-line arguments shared by the V3 training and evaluation entry points."""

from argparse import Namespace

import isaacgym  # noqa: F401
from isaacgym import gymutil


def get_v3_args(description: str) -> Namespace:
    """Parse Isaac Gym and explicit V3 run identity arguments.

    Args:
        description: Short description shown by the Isaac Gym argument parser.

    Returns:
        Parsed simulator and V3 arguments, including the explicit A-to-B switch
        and expected parent SHA256. Both migration fields are optional for A runs.

    Raises:
        ValueError: No explicit new output directory is supplied.

    Note:
        The parser is used only by V3 entry points; official B2 argument behavior
        remains unchanged.
    """
    custom_parameters = [
        {"name": "--reproduction_part", "type": str, "required": True},
        {"name": "--run_purpose", "type": str, "required": True},
        {"name": "--seed", "type": int, "required": True},
        {"name": "--num_envs", "type": int},
        {"name": "--max_iterations", "type": int},
        {"name": "--resume", "type": str},
        {"name": "--transition_from_a", "action": "store_true"},
        {"name": "--parent_sha256", "type": str},
        {"name": "--checkpoint", "type": str},
        {"name": "--manifest", "type": str},
        {"name": "--protocol", "type": str, "default": "a8000",
         "choices": ["a8000", "zero_force_regression", "force_tracking", "fixed_wall_contact"]},
        {"name": "--scenario", "type": str},
        {"name": "--reference_archive", "type": str},
        {"name": "--smoke", "action": "store_true"},
        {"name": "--video", "action": "store_true"},
        {"name": "--episodes_per_env", "type": int},
        {"name": "--start_evidence", "type": str},
        {"name": "--output_dir", "type": str, "required": True},
        {"name": "--rl_device", "type": str, "default": "cuda:0"},
        {"name": "--headless", "action": "store_true"},
    ]
    args = gymutil.parse_arguments(
        description=description,
        custom_parameters=custom_parameters,
    )
    args.task = "v3_pos_force"
    if args.output_dir is None:
        raise ValueError("output_dir must explicitly name a new independent run directory")
    return args

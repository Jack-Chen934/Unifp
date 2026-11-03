"""V3-specific configuration resolution."""

import copy
from argparse import Namespace
from typing import Any, Tuple

from legged_gym.envs.v3.v3_pos_force_config import V3PosForceCfg, V3PosForceCfgPPO
from legged_gym.utils.task_registry import task_registry


def _copy_config_tree(source: Any) -> Any:
    """Copy config values into fresh objects, including mutable class defaults.

    Args:
        source: Config object or a leaf value from the official Python config tree.

    Returns:
        A deep copy whose nested config objects and mutable values are independent.
    """
    if not hasattr(source, "__dict__"):
        return copy.deepcopy(source)

    result = type(source)()
    for name in dir(source):
        if name.startswith("_"):
            continue
        value = getattr(source, name)
        if callable(value):
            continue
        setattr(result, name, _copy_config_tree(value))
    return result


def resolve_v3_configs(
    args: Namespace,
    *,
    purpose: str,
) -> Tuple[V3PosForceCfg, V3PosForceCfgPPO]:
    """Resolve isolated V3 task configs and synchronize explicit run parameters.

    Args:
        args: Parsed Isaac Gym and V3 arguments. ``seed`` is required; optional
            ``num_envs`` and ``max_iterations`` override their task defaults.
            Zero iterations require an explicit ``resume`` file and are verified
            as completed A8000 by the training entry before Gym construction.
        purpose: Explicit run purpose: ``debug``, ``development``, ``preflight``,
            ``formal``, or ``evaluation``.

    Returns:
        Independent environment and PPO config objects for one run.

    Raises:
        ValueError: If the run identity, seed, or supplied overrides are invalid.
    """
    allowed_purposes = {"debug", "development", "preflight", "formal", "evaluation"}
    reproduction_part = getattr(args, "reproduction_part", None)
    run_purpose = getattr(args, "run_purpose", None)
    if reproduction_part not in {"A", "B"}:
        raise ValueError("reproduction_part must be explicitly set to 'A' or 'B'")
    if run_purpose != purpose or purpose not in allowed_purposes:
        raise ValueError(f"run_purpose must explicitly match one of {sorted(allowed_purposes)}")

    seed = getattr(args, "seed", None)
    if seed is None or seed < 0:
        raise ValueError("seed must be explicitly set to a non-negative integer")

    env_cfg, train_cfg = task_registry.get_cfgs(name="v3_pos_force")
    env_cfg = _copy_config_tree(env_cfg)
    train_cfg = _copy_config_tree(train_cfg)

    train_cfg.seed = seed
    env_cfg.seed = seed
    num_envs = getattr(args, "num_envs", None)
    if num_envs is not None:
        if num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        env_cfg.env.num_envs = num_envs

    max_iterations = getattr(args, "max_iterations", None)
    if max_iterations is None:
        if purpose in {"debug", "development", "preflight"}:
            raise ValueError(f"max_iterations must be explicit for {purpose} runs")
        train_cfg.runner.max_iterations = 8000 if reproduction_part == "A" else 32000
    else:
        if max_iterations < 0 or (max_iterations == 0 and not getattr(args, "resume", None)):
            raise ValueError("max_iterations must be positive, or zero with an explicit completed checkpoint")
        train_cfg.runner.max_iterations = max_iterations

    env_cfg.run.reproduction_part = reproduction_part
    env_cfg.run.run_purpose = run_purpose
    return env_cfg, train_cfg

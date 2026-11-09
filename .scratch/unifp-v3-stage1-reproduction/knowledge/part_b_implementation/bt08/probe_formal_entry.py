"""核验真实formal入口通过准入后停在Gym创建前，不启动训练。"""

import json
from pathlib import Path
from typing import Any

import isaacgym  # noqa: F401
from legged_gym.scripts import train_v3


class VerifiedBeforeSimulation(Exception):
    """Mark that the unchanged production gate reached environment construction."""


def stop_before_simulation(*args: Any, **kwargs: Any) -> None:
    """Intercept only Gym construction after production migration and gate checks.

    Args:
        *args: Original V3PosForce constructor positional arguments, never consumed.
        **kwargs: Original creation properties, never submitted to the native SDK.

    Raises:
        VerifiedBeforeSimulation: Always, before creating any simulation or run.
    """
    raise VerifiedBeforeSimulation()


if __name__ == "__main__":
    train_v3.V3PosForce = stop_before_simulation
    try:
        train_v3.main()
    except VerifiedBeforeSimulation:
        report = {"passed": True, "formal_gate_reached_simulation_boundary": True,
                  "gym_created": False, "policy_updates": 0,
                  "说明": "实际父SHA、迁移合同和formal B准入均由原生产入口核验；仅拦截Gym构造，不替换门控。"}
        destination = Path("/mnt/data_8THDD/myl_unifp_v3_stage1/validation/b_stage_20261107_r01/bt08_formal_entry_probe.json")
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(report, ensure_ascii=False), flush=True)

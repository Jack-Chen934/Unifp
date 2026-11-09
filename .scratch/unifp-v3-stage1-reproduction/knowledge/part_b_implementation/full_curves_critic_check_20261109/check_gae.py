"""用当前真实RolloutStorage交叉核对离线GAE，不导入仿真环境。"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[4]


def main() -> None:
    """Verify all 249 full rollout blocks using the production return routine.

    Raises:
        AssertionError: Offline and production GAE or normalization disagree.

    Note:
        Allocates small CPU buffers only and writes one verification JSON.
    """
    source = PROJECT / "UniFP_v3/legged_gym/b2_gym_learn/ppo_cse_pf/v3_rollout_storage.py"
    specification = importlib.util.spec_from_file_location("checked_v3_storage", source)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    results = []
    for iteration in (9000, 12000, 16000):
        data = np.load(ROOT / f"check_{iteration}/trajectory.npz")
        expected = np.load(ROOT / f"check_{iteration}/derived_advantages.npz")
        summary = json.loads((ROOT / f"check_{iteration}/summary.json").read_text())
        gamma, lam = summary["gamma"], summary["lam"]
        storage = module.V3RolloutStorage(64, 24, [1], [1], [1], [1], "cpu")
        maximum_raw_error, maximum_normalized_error = 0.0, 0.0
        for first in range(0, len(expected["raw_gae"]), 24):
            block = slice(first, first + 24)
            terminated = torch.from_numpy(data["terminated"][block])
            timeout = torch.from_numpy(data["timeout"][block])
            next_value = torch.from_numpy(data["next_value"][block])
            rewards = torch.from_numpy(data["reward"][block]) + gamma * next_value * (timeout & ~terminated)
            storage.values.copy_(torch.from_numpy(data["value"][block]).unsqueeze(-1))
            storage.rewards.copy_(rewards.unsqueeze(-1))
            storage.dones.copy_((terminated | timeout).unsqueeze(-1))
            storage.step = 24
            last_values = next_value[-1].unsqueeze(-1) * ~storage.dones[-1]
            storage.compute_returns(last_values, gamma, lam)
            raw_error = np.max(np.abs((storage.returns - storage.values).squeeze(-1).numpy() - expected["raw_gae"][block]))
            normalized_error = np.max(np.abs(storage.advantages.squeeze(-1).numpy() - expected["normalized_gae"][block]))
            maximum_raw_error = max(maximum_raw_error, float(raw_error))
            maximum_normalized_error = max(maximum_normalized_error, float(normalized_error))
        assert maximum_raw_error < 1e-4 and maximum_normalized_error < 1e-4
        results.append({"iteration": iteration, "blocks": len(expected["raw_gae"]) // 24,
                        "maximum_raw_error": maximum_raw_error,
                        "maximum_normalized_error": maximum_normalized_error})
    output = {"storage_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "results": results, "passed": True}
    (ROOT / "gae_verification.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output))


if __name__ == "__main__":
    main()

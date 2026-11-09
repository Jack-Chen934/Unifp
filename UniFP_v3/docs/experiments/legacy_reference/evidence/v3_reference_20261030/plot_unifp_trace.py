"""Plot a trajectory emitted by play_unifp_policy.py or evaluate_unifp_policy.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from unifp_eval_protocol import validate_trace


parser = argparse.ArgumentParser(description="Plot a UniFP policy trajectory")
parser.add_argument("trace", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

payload = json.loads(args.trace.read_text(encoding="utf-8"))
samples = payload["samples"]
if not samples:
    raise ValueError("Trace contains no samples")

time_s = np.asarray([row["policy_time_s"] for row in samples], dtype=float)
base_height = np.asarray([row["base_height_m"][0] for row in samples], dtype=float)
reward = np.asarray([row["reward"][0] for row in samples], dtype=float)
action_max = np.asarray([row["action_abs_max"][0] for row in samples], dtype=float)
ee = np.asarray([row["ee_position_world_m"][0] for row in samples], dtype=float)
goal = np.asarray([row["goal_position_world_m"][0] for row in samples], dtype=float)
done = np.asarray([row["done"][0] for row in samples], dtype=bool)

is_evaluation = "position_error_m" in samples[0]
if is_evaluation:
    validate_trace(payload)
    position_error = np.asarray([row["position_error_m"][0] for row in samples], dtype=float)
    adaptation_rmse = np.asarray([row["adaptation_rmse"][0] for row in samples], dtype=float)

fig, axes = plt.subplots(4 if is_evaluation else 3, 1, figsize=(11, 11 if is_evaluation else 9), sharex=True, constrained_layout=True)
axes = np.atleast_1d(axes)

# Schema 2 records action-input states at t and transition endings at t+dt.
# Put reset boundaries at their transition time, and show true terminal states.
episode_start = 0
for index, terminal in enumerate(done):
    if terminal:
        for axis in axes:
            end_time = samples[index].get("transition", {}).get("policy_time_s", time_s[index])
            axis.axvspan(time_s[episode_start], end_time, color="tab:red", alpha=0.07)
            axis.axvline(end_time, color="tab:red", alpha=0.45, linewidth=0.8)
        episode_start = index + 1
if episode_start < len(time_s):
    for axis in axes:
        axis.axvspan(time_s[episode_start], time_s[-1], color="tab:blue", alpha=0.04)

axes[0].plot(time_s, base_height, label="base height")
axes[0].axhline(0.25, color="tab:red", linestyle="--", linewidth=1, label="0.25 m reference")
axes[0].set_ylabel("Base height (m)")
axes[0].legend(loc="best")
axes[0].grid(alpha=0.25)

for index, label in enumerate(("x", "y", "z")):
    axes[1].plot(time_s, ee[:, index], label=f"ee {label}")
    axes[1].plot(time_s, goal[:, index], linestyle="--", label=f"goal {label}")
axes[1].set_ylabel("World position (m)")
axes[1].legend(ncol=3, loc="best")
axes[1].grid(alpha=0.25)

if is_evaluation:
    axes[2].plot(time_s, position_error, label="end-effector position error")
    axes[2].axhline(0.1, color="tab:gray", linestyle="--", linewidth=1, label="0.1 m reference")
    axes[2].set_ylabel("Position error (m)")
    axes[2].legend(loc="best")
    axes[2].grid(alpha=0.25)
    axes[3].plot(time_s, adaptation_rmse, color="tab:purple", label="12-D adaptation RMSE")
    axes[3].set_ylabel("Adaptation RMSE")
    axes[3].set_xlabel("Policy time (s)")
    axes[3].legend(loc="best")
    axes[3].grid(alpha=0.25)
    if payload.get('schema_version', 0) >= 2:
        terminal_rows = [row['transition'] for row in samples if row['done'][0]]
        terminal_t = [row['policy_time_s'] for row in terminal_rows]
        for axis, field in ((axes[0], 'base_height_m'), (axes[2], 'position_error_m')):
            axis.scatter(terminal_t, [row['pre_reset_state'][field][0] for row in terminal_rows],
                         color='tab:red', marker='x', s=18, label='terminal state before reset')
            axis.legend(loc='best')
else:
    axes[2].plot(time_s, reward, label="reward")
    axes[2].plot(time_s, action_max, label="max |action|")
    axes[2].scatter(time_s[done], reward[done], color="tab:red", marker="x", label="termination")
    axes[2].set_xlabel("Policy time (s)")
    axes[2].set_ylabel("Value")
    axes[2].legend(loc="best")
    axes[2].grid(alpha=0.25)

args.output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.output, dpi=160)
print(f"UNIFP_TRACE_PLOT_PASS {args.output.resolve()}")

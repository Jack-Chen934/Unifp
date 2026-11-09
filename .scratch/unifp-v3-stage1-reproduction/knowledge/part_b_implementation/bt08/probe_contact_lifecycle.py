"""仅缩短采样循环，复现同进程接触／0N图形场景生命周期。"""

import json
import os
from pathlib import Path
from typing import Any

import isaacgym  # noqa: F401
import torch
from legged_gym.scripts import evaluate_v3_contact as entry
from legged_gym.scripts.v3_args import get_v3_args


class ProbeVideo(entry.EvaluationVideo):
    """Mark camera allocation and cleanup around the unchanged recorder."""

    def __init__(self, environment: Any, path: Path) -> None:
        """Trace native camera construction without changing recorder settings.

        Args:
            environment: Actual 32-environment contact simulator.
            path: Diagnostic MP4 on the project data volume.
        """
        print("[DEBUG-bt08-contact] camera before " + str(path), flush=True)
        super().__init__(environment, path)
        print("[DEBUG-bt08-contact] camera after", flush=True)

    def close(self) -> None:
        """Trace the real encoder cleanup without replacing its behavior.

        Note:
            Calls the original close and leaves simulation destruction unchanged.
        """
        print("[DEBUG-bt08-contact] close before", flush=True)
        super().close()
        if os.environ.get("BT08_DESTROY_CAMERA") == "1":
            self.environment.gym.destroy_camera_sensor(self.environment.sim, self.environment.envs[0], self.camera)
        print("[DEBUG-bt08-contact] close after", flush=True)


def short_range(stop: int) -> range:
    """Shorten only the long policy loop while retaining camera and scene counts.

    Args:
        stop: Original loop bound; 1000/3000 policy intervals become two intervals.

    Returns:
        A diagnostic range with camera-allocation loops unchanged.
    """
    return range(2 if stop >= 1000 else stop)


def main() -> None:
    """Run both actual evaluator lifecycles with only two sampled policy steps.

    Raises:
        Exception: The underlying SDK, recording or summarization fails.

    Note:
        This is a crash probe, not a quota or policy-performance evaluation.
    """
    args = get_v3_args("Contact lifecycle crash probe")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    manifest = json.loads(Path(args.manifest).read_text())
    output = Path(args.output_dir)
    output.mkdir(exist_ok=False)
    entry.range = short_range
    entry.EvaluationVideo = ProbeVideo
    for zero_control in (False, True):
        directory = output / ("zero_control" if zero_control else "contact")
        directory.mkdir()
        print("[DEBUG-bt08-contact] scenario before " + directory.name, flush=True)
        entry.evaluate_contact_scenario(args, payload, manifest, zero_control=zero_control, output_dir=directory)
        print("[DEBUG-bt08-contact] scenario after " + directory.name, flush=True)
    print("[DEBUG-bt08-contact] both lifecycles completed", flush=True)


if __name__ == "__main__":
    main()

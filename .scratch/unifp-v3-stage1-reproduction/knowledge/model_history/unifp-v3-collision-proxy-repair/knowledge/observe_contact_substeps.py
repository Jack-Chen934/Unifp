"""Observe completed simulation steps around failures in the existing replay."""

from __future__ import annotations

from collections import deque
import json
from typing import Any

import replay_v3
import torch

SUBSTEP_EVENTS: list[dict[str, Any]] = []
OBSERVED_BODIES = ('base_link', 'wrist_camera', 'wrist_camera_connector', 'arm_seg4')


class SubstepInspectionEnvironment(replay_v3.InspectionEnvironment):
    """Delegate physics unchanged and retain a bounded pre-reset observation ring."""

    def _capture_completed_step(self, substep: int) -> None:
        """Read synchronized poses and forces after a completed simulation call.

        Args:
            substep: Completed simulate index within this policy interval, 1 to 8.

        Note:
            Refreshes readback tensors and synchronizes GPU work. Keeps at most
            16 frames in CPU memory; no state setter or RNG operation is used.
        """
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        body_indices = [self.body_names.index(name) for name in OBSERVED_BODIES]
        self.substep_observations.append({
            'policy_step': self.inspection_step, 'substep': substep,
            'body_pose_world_m_xyzw': self.rigid_state[:, body_indices, :7].detach().cpu().clone(),
            'contact_force_world_n': self.contact_forces.detach().cpu().clone(),
        })

    def _compute_v3_torques(self, actions: torch.Tensor) -> torch.Tensor:
        """Observe the previous simulate, then delegate the original feedback law.

        Args:
            actions: Dimensionless policy actions, shape [N,14], policy ordering.

        Returns:
            Original effort-limited joint torques, shape [N,14], Gym ordering, Nm.

        Note:
            Creates an observation ring on the first policy step. The initial
            post-reset state is not interpreted as a completed simulation call.
        """
        if not hasattr(self, 'substep_observations'):
            assert self.cfg.sim.dt == 0.0025 and self.cfg.sim.substeps == 1
            assert self.cfg.control.decimation == 8
            self.substep_observations = deque(maxlen=16)
            self.observed_policy_step = -1
        if self.observed_policy_step != self.inspection_step:
            self.observed_policy_step = self.inspection_step
            self.completed_substeps = 0
        if self.completed_substeps:
            self._capture_completed_step(self.completed_substeps)
        self.completed_substeps += 1
        return super()._compute_v3_torques(actions)

    def _accumulate_episode_metrics(
        self, done_ids: torch.Tensor, terminated: torch.Tensor
    ) -> list[dict[str, Any]]:
        """Capture the final substep and flush only physically failed rows.

        Args:
            done_ids: Ending environment indices, shape [K], before local reset.
            terminated: Physical failure flags, shape [N], before local reset.

        Returns:
            Original replay episode records with unchanged termination decisions.

        Note:
            Appends JSON-compatible failure windows to SUBSTEP_EVENTS. Forces
            are net body forces, not pairwise contacts or contact-point impulses.
        """
        self._capture_completed_step(self.completed_substeps)
        for env_id in terminated.nonzero(as_tuple=False).flatten().tolist():
            frames = []
            for observation in self.substep_observations:
                frames.append({
                    'policy_step': observation['policy_step'], 'substep': observation['substep'],
                    'body_pose_world_m_xyzw': {
                        name: observation['body_pose_world_m_xyzw'][env_id, index].tolist()
                        for index, name in enumerate(OBSERVED_BODIES)},
                    'contact_force_world_n': observation['contact_force_world_n'][env_id].tolist(),
                })
            SUBSTEP_EVENTS.append({'policy_step': self.inspection_step, 'env_id': env_id,
                                   'body_names': list(self.body_names), 'frames': frames})
        return super()._accumulate_episode_metrics(done_ids, terminated)


def main() -> None:
    """Run the existing isolated replay and persist bounded failure observations.

    Note:
        Replaces only the replay's diagnostic subclass in this process. Writes
        substeps.json beside its new output; production and replay files stay intact.

    Raises:
        RuntimeError: The existing replay fails identity or simulation checks.
        OSError: Evidence cannot be saved.
    """
    args = replay_v3.parse_args()
    replay_v3.InspectionEnvironment = SubstepInspectionEnvironment
    replay_v3.main()
    report = {'events': SUBSTEP_EVENTS, 'simulation_dt_s': 0.0025,
              'policy_decimation': 8, 'physx_internal_substeps': 1,
              'note_zh': '独立零更新观测，fetch和tensor读取可能影响GPU执行时序；每帧为simulate结束后的状态，未读取内部接触流形，不能仅由净力推断穿透。'}
    (args.output_dir / 'substeps.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()

"""Init-time UniFP rigid-body and motor randomization for the Isaac Lab probe.

The released Gym code applies COM perturbation to rigid-body property index 1.
Its collapsed URDF traversal suggests FL_hip; the original Gym runtime body
order still needs direct verification. The 17 leg-mass fields stay zero because
the official switch is off.
"""

from __future__ import annotations

import torch


class UniFPDomainRandomization:
    def __init__(self, robot, controller, scene, *, enabled: bool,
                 com_body="FL_hip", gripper_body="ee_gripper_link"):
        self.enabled = enabled
        self.device = scene.device
        n = scene.num_envs
        view = robot.root_physx_view
        ids = torch.arange(n, device="cpu")
        masses_before = view.get_masses().clone()
        coms_before = view.get_coms().clone()
        materials_before = view.get_material_properties().clone()
        inertias_before = view.get_inertias().clone()
        self.base_id = robot.body_names.index("base_link")
        self.com_id = robot.body_names.index(com_body)
        self.gripper_id = robot.body_names.index(gripper_body)
        self.num_motors = len(controller.action_indices)
        self.mass_deltas = torch.zeros(n, 22, device=self.device)
        self.motor_strength = torch.ones(n, self.num_motors, device=self.device)

        if enabled:
            mass_offsets = torch.rand(n, device="cpu") * 15.0
            gripper_offsets = torch.rand(n, device="cpu") * 0.2
            com_offsets = torch.rand(n, 3, device="cpu") * 0.3 - 0.15
            masses = masses_before.clone()
            masses[:, self.base_id] += mass_offsets
            masses[:, self.gripper_id] += gripper_offsets
            view.set_masses(masses, ids)
            # Isaac Lab's init-time mass randomizer scales inertia by mass ratio.
            # Isaac Gym's recomputeInertia=True similarly updates rigid inertia.
            inertias = inertias_before.clone()
            for body_id in (self.base_id, self.gripper_id):
                inertias[:, body_id] *= (masses[:, body_id] / masses_before[:, body_id])[:, None]
            view.set_inertias(inertias, ids)
            coms = coms_before.clone()
            coms[:, self.com_id, :3] += com_offsets
            view.set_coms(coms, ids)

            # Official code draws 256 friction buckets, then assigns one bucket
            # to all collision shapes within each environment.
            buckets = 0.3 + torch.rand(256, device="cpu") * 1.7
            friction = buckets[torch.randint(0, 256, (n,), device="cpu")]
            materials = materials_before.clone()
            materials[:, :, 0] = friction[:, None]
            materials[:, :, 1] = friction[:, None]
            view.set_material_properties(materials, ids)
            self.motor_strength = 0.85 + torch.rand(n, self.num_motors, device=self.device) * 0.3

        masses_after = view.get_masses()
        coms_after = view.get_coms()
        materials_after = view.get_material_properties()
        self.mass_deltas[:, 0] = (masses_after[:, self.base_id] - masses_before[:, self.base_id]).to(self.device)
        self.mass_deltas[:, 1:4] = (coms_after[:, self.com_id, :3] - coms_before[:, self.com_id, :3]).to(self.device)
        self.mass_deltas[:, 4] = (masses_after[:, self.gripper_id] - masses_before[:, self.gripper_id]).to(self.device)
        self.friction = materials_after[:, :1, 0].to(self.device).clone()
        if not torch.allclose(materials_after[:, :, 0], materials_after[:, :1, 0].expand_as(materials_after[:, :, 0])):
            raise RuntimeError("Robot friction shapes disagree within an environment")
        if not torch.allclose(materials_after[:, :, 1], materials_after[:, :1, 1].expand_as(materials_after[:, :, 1])):
            raise RuntimeError("Robot dynamic friction shapes disagree within an environment")
        if enabled:
            assert (self.mass_deltas[:, 0] >= 0).all() and (self.mass_deltas[:, 4] >= 0).all()
            assert (self.mass_deltas[:, 1:4].abs() <= 0.151).all()
            assert (self.friction >= 0.3).all() and (self.friction <= 2.0).all()
            assert (self.motor_strength >= 0.85).all() and (self.motor_strength <= 1.15).all()
        else:
            assert torch.allclose(self.mass_deltas, torch.zeros_like(self.mass_deltas))
        self.body_names = {"base_mass": robot.body_names[self.base_id],
                           "com_index_1": robot.body_names[self.com_id],
                           "gripper_mass": robot.body_names[self.gripper_id]}

    def summary(self):
        return {
            "enabled": self.enabled,
            "body_names": self.body_names,
            "base_mass_delta_kg": self.mass_deltas[:, 0].cpu().tolist(),
            "com_delta_m": self.mass_deltas[:, 1:4].cpu().tolist(),
            "gripper_mass_delta_kg": self.mass_deltas[:, 4].cpu().tolist(),
            "leg_mass_delta_max_kg": float(self.mass_deltas[:, 5:].abs().max().item()),
            "friction": self.friction[:, 0].cpu().tolist(),
            "motor_strength_min": float(self.motor_strength.min().item()),
            "motor_strength_max": float(self.motor_strength.max().item()),
        }

    def resample_motor(self, env_ids):
        """Released reset_idx resamples motors; rigid properties stay fixed."""
        if self.enabled and len(env_ids):
            self.motor_strength[env_ids] = (
                0.85 + torch.rand(len(env_ids), self.num_motors, device=self.device) * 0.3
            )

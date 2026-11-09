"""Inspect frozen V3 policies and optionally record camera video without learning."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
from typing import Any

import isaacgym  # noqa: F401 -- The native SDK must precede Torch.
from isaacgym import gymapi, gymutil
import numpy as np
from PIL import Image, ImageDraw
import torch
import legged_gym
import legged_gym.envs  # noqa: F401
from legged_gym.b2_gym_learn.ppo_cse_pf.v3_actor_critic import V3ActorCritic
from legged_gym.b2_gym_learn.ppo_cse_pf.v3_evaluation import configure_evaluation_scenario
from legged_gym.b2_gym_learn.ppo_cse_pf.v3_run_identity import build_run_identity, file_sha256
from legged_gym.envs.v3.legged_robot_v3 import V3PosForce
from legged_gym.utils.helpers import class_to_dict, set_seed
from legged_gym.utils.v3_config import resolve_v3_configs


class InspectionEnvironment(V3PosForce):
    """Retain the original physics while observing failures before local reset."""

    camera_enabled = False

    def create_sim(self) -> None:
        """Enable a camera graphics device only for the optional recording mode.

        Note:
            Changes graphics allocation only; delegates all physics to V3.
        """
        if self.camera_enabled:
            self.graphics_device_id = self.sim_device_id
        super().create_sim()

    def _accumulate_episode_metrics(
        self, done_ids: torch.Tensor, terminated: torch.Tensor
    ) -> list[dict[str, Any]]:
        """Capture terminal predicates, then use the original episode settlement.

        Args:
            done_ids: Long [K] environment indices ending at this policy boundary.
            terminated: Boolean [N] physical failures before any local reset.

        Returns:
            Unmodified original episode records.

        Note:
            Reads terminal tensors before reset; never alters failure decisions.
        """
        failed_ids = terminated.nonzero(as_tuple=False).flatten().tolist()
        for env_id in failed_ids:
            roll, pitch = self.rpy[env_id, :2].tolist()  # Radians in world roll/pitch.
            contact = self.contact_forces[env_id, self.base_body_index].tolist()
            contact_norm = float(np.linalg.norm(contact))  # Net base-link force, N.
            predicates = {
                "roll": abs(roll) > 0.8,
                "pitch": abs(pitch) > 1.0,
                "base_contact": contact_norm > 1.0,
            }
            magnitudes = torch.linalg.vector_norm(self.contact_forces[env_id], dim=-1)
            largest = torch.topk(magnitudes, min(5, magnitudes.numel()))
            self.failure_events.append({
                "policy_step": self.inspection_step, "env_id": env_id,
                "episode_step": int(self.episode_length_buf[env_id]),
                "predicates": predicates, "roll_rad": roll, "pitch_rad": pitch,
                "base_contact_vector_n": contact, "base_contact_norm_n": contact_norm,
                "root_state": self.root_states[env_id].tolist(),
                "commands_body": self.commands[env_id, :3].tolist(),
                "action": self.actions[env_id].tolist(),
                "dof_position_rad": self.dof_pos[env_id].tolist(),
                "body_pose_world_m_xyzw": {
                    name: self.rigid_state[env_id, index, :7].tolist()
                    for index, name in enumerate(self.body_names)},
                "mass_com_delta_kg_m": self.mass_com_delta[env_id].tolist(),
                "friction": self.friction[env_id].tolist(),
                "execution_scale_delta": self.execution_scale_delta[env_id].tolist(),
                "largest_contacts": [{"body": self.body_names[int(index)], "norm_n": float(value),
                                      "vector_n": self.contact_forces[env_id, int(index)].tolist()}
                                     for index, value in zip(largest.indices, largest.values)],
            })
        return super()._accumulate_episode_metrics(done_ids, terminated)


def parse_args() -> argparse.Namespace:
    """Parse isolated inference and recording settings.

    Returns:
        Explicit checkpoint, output, scenario, action mode and simulation budget.
    """
    parser = argparse.ArgumentParser(description="V3冻结策略排查／MP4演示，不执行训练")
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--num_envs', type=int, default=128)
    parser.add_argument('--seconds', type=float, default=60.0)
    parser.add_argument('--seed', type=int, default=1001)
    parser.add_argument('--action_mode', choices=('mean', 'sample'), default='mean')
    parser.add_argument('--scenario', choices=('training', 'S0', 'S1', 'S2'), default='training')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--disable_self_collision', action='store_true')
    parser.add_argument('--unflip_visuals', action='store_true')
    parser.add_argument('--asset_override', type=Path)
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--height', type=int, default=720)
    return parser.parse_args()


def main() -> None:
    """Run new episodes with frozen weights and persist terminal evidence/video.

    Raises:
        ValueError: Checkpoint identity or requested inference settings are invalid.
        OSError: Simulation, evidence writing or video encoding fails.

    Note:
        Applies saved creation properties for the first N environments. Other RNG,
        reset state and PhysX caches are new; this is not historical trajectory replay.
        Uses no learner or optimizer and leaves the training snapshot untouched.
    """
    args = parse_args()
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    source_root = Path(legged_gym.__file__).resolve().parents[1]
    active_identity = build_run_identity(source_root, payload['config_snapshot'], payload['identity']['runtime'])
    if active_identity != payload['identity']:
        raise ValueError('冻结源码／资产身份与checkpoint不一致；请使用训练r03快照')
    if not 0 < args.num_envs <= payload['environment_training_state']['num_envs'] or args.seconds <= 0:
        raise ValueError('num_envs须位于保存规模内，seconds须为正数')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    set_seed(args.seed)
    config_args = argparse.Namespace(reproduction_part='A', run_purpose='debug', seed=args.seed,
                                     num_envs=args.num_envs, max_iterations=1)
    cfg, _ = resolve_v3_configs(config_args, purpose='debug')
    if args.asset_override is not None:
        candidate_urdf = args.asset_override.resolve()
        candidate_root = candidate_urdf.parent.parent
        manifest = json.loads((candidate_root / 'asset_manifest.json').read_text())
        if manifest.get('asset_status') != 'independent_local_base_proxy_repair_candidate':
            raise ValueError('Asset override must be an explicitly marked isolated candidate')
        cfg.asset.file = str(candidate_urdf)
    if args.scenario != 'training':
        configure_evaluation_scenario(cfg, args.scenario)
    if args.disable_self_collision:
        cfg.asset.self_collisions = 1  # Diagnostic actor flag only; not a model fix.
    if args.unflip_visuals:
        cfg.asset.flip_visual_attachments = False  # GLB vertices already use the URDF link frame.
    parameters = gymapi.SimParams()
    gymutil.parse_sim_config(class_to_dict(cfg.sim), parameters)
    parameters.use_gpu_pipeline = True
    parameters.physx.use_gpu = True
    saved_properties = {key: payload['environment_training_state'][key][:args.num_envs]
                        for key in ('mass_com_delta', 'friction')}
    InspectionEnvironment.camera_enabled = args.video
    environment = InspectionEnvironment(cfg, parameters, gymapi.SIM_PHYSX, 'cuda:0', True,
                                        creation_properties=saved_properties)
    encoder = None
    try:
        environment.failure_events = []
        environment.body_names = environment.gym.get_actor_rigid_body_names(
            environment.envs[0], environment.actor_handles[0])
        environment.set_evaluation_push_seed(2001)
        progress = payload['progress']
        environment.set_training_progress(progress['completed_iterations'], progress['training_policy_steps'])
        model = V3ActorCritic(1856, 294, 12, 58, 14,
                              **payload['config_snapshot']['training']['policy']).to('cuda:0')
        model.load_state_dict(payload['model_state_dict'], strict=True)
        model.eval()
        observations = environment.get_observations()
        frame_rate = round(1.0 / environment.dt)
        steps = round(args.seconds / environment.dt)
        camera = None
        if args.video:
            camera_properties = gymapi.CameraProperties()
            camera_properties.width, camera_properties.height = args.width, args.height
            camera = environment.gym.create_camera_sensor(environment.envs[0], camera_properties)
            if camera < 0:
                raise OSError('摄像头创建失败')
            encoder = subprocess.Popen([
                'ffmpeg', '-v', 'error', '-n', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                '-s', f'{args.width}x{args.height}', '-r', str(frame_rate), '-i', '-',
                '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20',
                str(args.output_dir / 'policy.mp4')], stdin=subprocess.PIPE)
        ended = Counter()
        saturation_counts = torch.zeros(3, device='cuda:0')  # Steering/drive/arm clipped samples.
        action_slices = (slice(0, 4), slice(4, 8), slice(8, 14))
        for step in range(steps):
            environment.inspection_step = step
            environment.set_evaluation_policy_step(progress['training_policy_steps'] + step)
            with torch.inference_mode():
                actions = model.act(observations['obs']) if args.action_mode == 'sample' else model.act_inference(observations)
            for group, action_slice in enumerate(action_slices):
                saturation_counts[group] += (actions[:, action_slice].abs() >= cfg.normalization.clip_actions).sum()
            observations, _, _, infos = environment.step(actions)
            ended.update(record['end_type'] for record in infos['episode_records'])
            if (step + 1) % 500 == 0:
                print(json.dumps({'policy_steps_done': step+1, 'physical_failures': ended['physical_failure']}), flush=True)
            if camera is not None:
                # Camera follows environment zero in world coordinates after physics.
                position = environment.root_states[0, :3].cpu().tolist()
                environment.gym.set_camera_location(camera, environment.envs[0],
                    gymapi.Vec3(position[0]+2.3, position[1]-2.3, position[2]+1.6),
                    gymapi.Vec3(position[0], position[1], position[2]+0.3))
                environment.gym.fetch_results(environment.sim, True)
                environment.gym.step_graphics(environment.sim)
                environment.gym.render_all_camera_sensors(environment.sim)
                pixels = environment.gym.get_camera_image(environment.sim, environment.envs[0], camera, gymapi.IMAGE_COLOR)
                frame = Image.fromarray(np.asarray(pixels).reshape(args.height, args.width, 4)[:, :, :3])
                error_cm = float(environment._physical_tracking_errors()[0, 1]) * 100
                commands = environment.commands[0, :3].tolist()
                annotation = f'I={progress["completed_iterations"]}  {args.scenario}  t={(step+1)*environment.dt:.2f}s  TCP L2={error_cm:.2f}cm\nvx={commands[0]:.2f} vy={commands[1]:.2f}m/s yaw={commands[2]:.2f}rad/s'
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, args.width, 42), fill=(15, 20, 30))
                draw.text((10, 6), annotation, fill=(255, 255, 255))
                encoder.stdin.write(frame.tobytes())
        if encoder is not None:
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise OSError('MP4编码失败')
        assert ended['physical_failure'] == len(environment.failure_events)
        predicate_counts = Counter(key for event in environment.failure_events
                                   for key, active in event['predicates'].items() if active)
        summary = {
            'checkpoint': str(args.checkpoint), 'checkpoint_sha256': file_sha256(args.checkpoint),
            'checkpoint_iteration': progress['completed_iterations'], 'action_mode': args.action_mode,
            'scenario': args.scenario, 'seed': args.seed, 'num_envs': args.num_envs,
            'self_collision_disabled': args.disable_self_collision,
            'flip_visual_attachments': cfg.asset.flip_visual_attachments,
            'asset_override': str(args.asset_override.resolve()) if args.asset_override else None,
            'asset_override_urdf_sha256': file_sha256(args.asset_override) if args.asset_override else None,
            'dof_names': environment.dof_names,
            'shape_contact_offsets_m': [float(shape.contact_offset) for shape in
                environment.gym.get_actor_rigid_shape_properties(environment.envs[0], environment.actor_handles[0])],
            'shape_rest_offsets_m': [float(shape.rest_offset) for shape in
                environment.gym.get_actor_rigid_shape_properties(environment.envs[0], environment.actor_handles[0])],
            'policy_steps': steps, 'seconds_per_environment': steps*environment.dt,
            'completed_episodes': dict(ended), 'failure_predicates_overlapping': dict(predicate_counts),
            'action_std': model.std.detach().cpu().tolist(),
            'action_clip_fraction_steering_drive_arm': (saturation_counts / torch.tensor(
                [4,4,6],device='cuda:0') / steps / args.num_envs).cpu().tolist(),
            'policy_updates': 0, 'source_identity_verified': True,
            'note_zh': '新episode冻结推理，采用保存物性前N行；非训练历史重放，非正式评估，未修改终止阈值。',
        }
        (args.output_dir/'failures.json').write_text(json.dumps(environment.failure_events, ensure_ascii=False, indent=2)+'\n')
        (args.output_dir/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    finally:
        if encoder is not None and encoder.poll() is None:
            encoder.stdin.close()
            encoder.wait()
        environment.gym.destroy_sim(environment.sim)


if __name__ == '__main__':
    main()

"""Compare V3 native visual import flags without changing training assets."""

import argparse
import json
from pathlib import Path
from typing import Any

import isaacgym  # noqa: F401
from isaacgym import gymapi
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from compare_desktop_fk import forward_kinematics
from legged_gym.envs.v3.v3_pos_force_config import V3PosForceCfg


def pose_matrix(position: Any, quaternion: Any) -> np.ndarray:
    """Convert native structured pose fields to a homogeneous world transform.

    Args:
        position: Structured x/y/z translation fields, in meters.
        quaternion: Structured x/y/z/w orientation fields.

    Returns:
        A float64 [4,4] link-frame pose.
    """
    matrix = np.eye(4)
    matrix[:3, 3] = [float(position[key]) for key in ('x', 'y', 'z')]
    matrix[:3, :3] = Rotation.from_quat(
        [float(quaternion[key]) for key in ('x', 'y', 'z', 'w')]).as_matrix()
    return matrix


def main() -> None:
    """Capture two visual imports and independently compare native body frames.

    Raises:
        OSError: Asset import, camera creation, or output writing fails.
        ValueError: Pose fields or the URDF joint graph are invalid.

    Note:
        Creates one isolated CPU-pipeline simulation on GPU0 as exposed by CUDA.
        Both actors have fixed bases, disabled gravity and disabled self collision;
        only the visual-flip option differs between actors. No policy is loaded.
    """
    parser = argparse.ArgumentParser(description='V3静态显示坐标单变量对照')
    parser.add_argument('--output_dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    gym = gymapi.acquire_gym()
    parameters = gymapi.SimParams()
    parameters.dt = 0.0025
    parameters.up_axis = gymapi.UP_AXIS_Z
    parameters.gravity = gymapi.Vec3(0, 0, 0)
    parameters.physx.use_gpu = True
    parameters.physx.num_position_iterations = 4
    parameters.physx.num_velocity_iterations = 1
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, parameters)
    if sim is None:
        raise OSError('无法创建独立静态仿真')
    try:
        plane = gymapi.PlaneParams()
        plane.normal = gymapi.Vec3(0, 0, 1)
        gym.add_ground(sim, plane)
        urdf_path = Path(V3PosForceCfg.asset.file)
        entries = []
        for flip in (True, False):
            options = gymapi.AssetOptions()
            for name in ('default_dof_drive_mode', 'collapse_fixed_joints',
                         'replace_cylinder_with_capsule', 'density', 'angular_damping',
                         'linear_damping', 'max_angular_velocity', 'max_linear_velocity',
                         'armature', 'thickness'):
                setattr(options, name, getattr(V3PosForceCfg.asset, name))
            options.flip_visual_attachments = flip
            options.fix_base_link = True
            options.disable_gravity = True
            asset = gym.load_asset(sim, str(urdf_path.parent.parent),
                                   'urdf/' + urdf_path.name, options)
            if asset is None:
                raise OSError('无法导入冻结V3资产')
            env = gym.create_env(sim, gymapi.Vec3(-2, -2, 0), gymapi.Vec3(2, 2, 3), 2)
            placement = gymapi.Transform()
            placement.p = gymapi.Vec3(0, 0, 0.34)
            actor = gym.create_actor(env, asset, placement, 'static_v3', len(entries), 1)
            properties = gym.get_actor_dof_properties(env, actor)
            properties['driveMode'].fill(gymapi.DOF_MODE_EFFORT)
            properties['stiffness'].fill(0)
            properties['damping'].fill(0)
            gym.set_actor_dof_properties(env, actor, properties)
            dof_names = gym.get_actor_dof_names(env, actor)
            states = np.zeros(len(dof_names), dtype=gymapi.DofState.dtype)
            states['pos'] = [V3PosForceCfg.init_state.default_joint_angles[name] for name in dof_names]
            gym.set_actor_dof_states(env, actor, states, gymapi.STATE_ALL)
            cameras = {}
            for geometry in ('visual', 'collision'):
                camera_properties = gymapi.CameraProperties()
                camera_properties.width, camera_properties.height = 960, 900
                camera_properties.use_collision_geometry = geometry == 'collision'
                camera = gym.create_camera_sensor(env, camera_properties)
                if camera < 0:
                    raise OSError('无法创建静态摄像头')
                gym.set_camera_location(camera, env, gymapi.Vec3(2.2, -3.1, 1.9),
                                        gymapi.Vec3(0, 0, 0.85))
                cameras[geometry] = camera
            entries.append((flip, env, actor, dof_names, cameras))
        # One zero-force step propagates written DOF values into native body poses.
        gym.prepare_sim(sim)
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.render_all_camera_sensors(sim)
        results = {'urdf': str(urdf_path), 'static_only': True, 'variants': []}
        for flip, env, actor, dof_names, cameras in entries:
            dof_states = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
            positions = {name: float(value) for name, value in zip(dof_names, dof_states['pos'])}
            body_names = gym.get_actor_rigid_body_names(env, actor)
            body_states = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_ALL)
            native_poses = {name: pose_matrix(state['pose']['p'], state['pose']['r'])
                            for name, state in zip(body_names, body_states)}
            expected_poses = forward_kinematics(urdf_path, positions, native_poses['base_link'])
            differences = {name: {
                'position_error_m': float(np.linalg.norm(native_poses[name][:3, 3] - expected[:3, 3])),
                'orientation_error_rad': float(Rotation.from_matrix(
                    expected[:3, :3].T @ native_poses[name][:3, :3]).magnitude())}
                for name, expected in expected_poses.items()}
            for geometry, camera in cameras.items():
                pixels = gym.get_camera_image(sim, env, camera, gymapi.IMAGE_COLOR)
                frame = np.asarray(pixels).reshape(900, 960, 4)[:, :, :3]
                Image.fromarray(frame).save(args.output_dir / f'flip_{int(flip)}_{geometry}.png')
            results['variants'].append({
                'flip_visual_attachments': flip, 'dof_names': dof_names,
                'dof_positions_rad': positions,
                'native_link_poses_m': {name: pose.tolist() for name, pose in native_poses.items()},
                'gym_to_urdf': differences,
                'max_position_error_m': max(x['position_error_m'] for x in differences.values()),
                'max_orientation_error_rad': max(x['orientation_error_rad'] for x in differences.values())})
        (args.output_dir / 'summary.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({str(x['flip_visual_attachments']): x['max_position_error_m']
                          for x in results['variants']}), flush=True)
    finally:
        gym.destroy_sim(sim)


if __name__ == '__main__':
    main()

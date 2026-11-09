"""Measure whether the approximate Ranger model translates under wheel commands."""

import argparse
import json
import math
import os
import sys
import traceback
from pathlib import Path

from ranger_force_experiment_config import ARM_EFFORT_LIMITS
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--asset-version", choices=("v2", "v3", "v3_original"), default="v2")
parser.add_argument("--output", type=Path, default=Path("ranger_approx_drive_results.json"))
parser.add_argument("--settle-steps", type=int, default=300)
parser.add_argument("--test-steps", type=int, default=500)
parser.add_argument("--steer-steps", type=int, default=600)
parser.add_argument("--screenshot", type=Path, help="Save the settled robot from the active GUI viewport.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.assets.articulation import ArticulationCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402

USD_PATH = Path(__file__).resolve().parent / "robot_description/a1arm_description/assets" / (
    "v3_original/a1arm_rgm3_original_movable.usd" if args.asset_version == "v3_original" else
    "v3/a1arm_rgm3_approx_sim_v3.usd" if args.asset_version == "v3" else "a1arm_rgm3_approx_sim_v2.usd")
ROOT_HEIGHT = .26 if args.asset_version == "v2" else .32
WHEEL_RADIUS = .153 if args.asset_version == "v2" else .1
HALF_TRACK = .182 if args.asset_version == "v2" else .185
WHEELS = ("fl", "fr", "bl", "br")


class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=str(USD_PATH)),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, ROOT_HEIGHT),
            joint_pos={
                "arm_joint1": 0.0, "arm_joint2": 1.0, "arm_joint3": -1.5,
                "arm_joint4": 0.0, "arm_joint5": 0.0, "arm_joint6": 0.0,
                ".*_axis": 0.0, ".*_steering_joint": 0.0, ".*_wheel_drive_joint": 0.0,
            },
        ),
        actuators={
            "arm": ImplicitActuatorCfg(joint_names_expr=["arm_joint[1-6]"], effort_limit_sim=ARM_EFFORT_LIMITS, stiffness=80.0, damping=4.0),
            "gripper": ImplicitActuatorCfg(joint_names_expr=["gripper.*"], effort_limit_sim=10.0, stiffness=20.0, damping=2.0),
            "steering": ImplicitActuatorCfg(joint_names_expr=[".*_steering_joint"], effort_limit_sim=35.0, stiffness=300.0, damping=20.0),
            "drive": ImplicitActuatorCfg(joint_names_expr=[".*_wheel_drive_joint"], effort_limit_sim=80.0, stiffness=0.0, damping=20.0),
        },
    )


def pose_snapshot(robot):
    pos = robot.data.root_pos_w[0].detach().cpu().tolist()
    quat = robot.data.root_quat_w[0].detach().cpu().tolist()  # w, x, y, z
    yaw = math.atan2(2.0 * (quat[0] * quat[3] + quat[1] * quat[2]), 1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2))
    return {"position_xyz": pos, "yaw_rad": yaw}


def command(robot, steering_ids, drive_ids, steer, speed):
    pos_target = robot.data.default_joint_pos.clone()
    vel_target = torch.zeros_like(robot.data.joint_vel)
    for corner, angle in zip(WHEELS, steer):
        idx = robot.find_joints([f"{corner}_steering_joint"])[0][0]
        pos_target[:, idx] = angle
    for corner, wheel_speed in zip(WHEELS, speed):
        idx = robot.find_joints([f"{corner}_wheel_drive_joint"])[0][0]
        vel_target[:, idx] = wheel_speed
    # Full position targets hold the arm and gripper at the configured pose.
    # Wheel drives have zero stiffness and therefore use velocity control only.
    robot.set_joint_position_target(pos_target)
    robot.set_joint_velocity_target(vel_target[:, drive_ids], joint_ids=drive_ids)


def wheel_targets(vx, vy, yaw_rate):
    """Swerve inverse kinematics in base coordinates; steering stays within +/-pi/2."""
    angles, speeds = [], []
    for x, y in ((0.247, HALF_TRACK), (0.247, -HALF_TRACK), (-0.247, HALF_TRACK), (-0.247, -HALF_TRACK)):
        ux, uy = vx - yaw_rate * y, vy + yaw_rate * x
        angle, speed = math.atan2(uy, ux), math.hypot(ux, uy) / WHEEL_RADIUS
        if angle > math.pi / 2:
            angle, speed = angle - math.pi, -speed
        elif angle < -math.pi / 2:
            angle, speed = angle + math.pi, -speed
        angles.append(angle)
        speeds.append(speed)
    return angles, speeds


try:
    assert USD_PATH.is_file(), USD_PATH
    print("RANGER_DRIVE_STAGE: create simulation", flush=True)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args.device))
    print("RANGER_DRIVE_STAGE: create scene", flush=True)
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))
    sim.set_camera_view(eye=(1.7, 1.7, 1.3), target=(0.0, 0.0, 0.45))
    print("RANGER_DRIVE_STAGE: reset", flush=True)
    sim.reset()
    robot = scene["robot"]
    steering_ids, steering_names = robot.find_joints(".*_steering_joint")
    drive_ids, drive_names = robot.find_joints(".*_wheel_drive_joint")
    assert len(steering_ids) == len(drive_ids) == 4, (steering_names, drive_names)

    zero_steer = [0.0] * 4
    zero_speed = [0.0] * 4
    command(robot, steering_ids, drive_ids, zero_steer, zero_speed)
    for _ in range(args.settle_steps):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())

    if args.screenshot:
        if args.headless:
            raise ValueError("Use GUI mode (omit --headless) for --screenshot.")
        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
        viewport = get_active_viewport()
        if viewport is None:
            raise RuntimeError("No active GUI viewport for screenshot.")
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(30):
            sim.render()
        capture = capture_viewport_to_file(viewport, file_path=str(args.screenshot.resolve()))
        for _ in range(60):
            app.update()
        if not args.screenshot.is_file():
            raise RuntimeError(f"Viewport capture was not written: {args.screenshot}")
        print("RANGER_DRIVE_SCREENSHOT:", args.screenshot.resolve(), flush=True)

    phases = [("forward", (0.3, 0.0, 0.0)), ("lateral", (0.0, 0.3, 0.0)), ("rotate_ccw", (0.0, 0.0, 0.6))]
    results = {"dt": sim.get_physics_dt(), "duration_per_phase_s": args.test_steps * sim.get_physics_dt(),
               "asset": str(USD_PATH), "initial": pose_snapshot(robot), "phases": []}
    for name, twist in phases:
        # Independent initial conditions keep previous phases from contaminating measurements.
        robot.write_root_state_to_sim(robot.data.default_root_state.clone())
        robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
        scene.reset()
        steering, speeds = wheel_targets(*twist)
        command(robot, steering_ids, drive_ids, steering, zero_speed)
        for _ in range(args.steer_steps):
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())
        start = pose_snapshot(robot)
        q_start = robot.data.joint_pos[0].detach().cpu().tolist()
        command(robot, steering_ids, drive_ids, steering, speeds)
        for _ in range(args.test_steps):
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())
            if not (torch.isfinite(robot.data.root_state_w).all() and torch.isfinite(robot.data.joint_pos).all()):
                raise RuntimeError(f"Non-finite robot state during {name}")
        end = pose_snapshot(robot)
        q_end = robot.data.joint_pos[0].detach().cpu().tolist()
        result = {
            "name": name, "start": start, "end": end,
            "command_base_twist": twist,
            "steering_targets_fl_fr_bl_br": steering,
            "drive_targets_fl_fr_bl_br": speeds,
            "steering_names": steering_names,
            "steering_actual": robot.data.joint_pos[0, steering_ids].detach().cpu().tolist(),
            "drive_names": drive_names,
            "drive_velocity_actual": robot.data.joint_vel[0, drive_ids].detach().cpu().tolist(),
            "delta_xyz": [b - a for a, b in zip(start["position_xyz"], end["position_xyz"])],
            "delta_yaw_rad": math.atan2(math.sin(end["yaw_rad"] - start["yaw_rad"]), math.cos(end["yaw_rad"] - start["yaw_rad"])),
            "joint_position_delta_rad": [b - a for a, b in zip(q_start, q_end)],
        }
        dx, dy, dz = result["delta_xyz"]
        yaw = start["yaw_rad"]
        body_delta = [math.cos(yaw) * dx + math.sin(yaw) * dy,
                      -math.sin(yaw) * dx + math.cos(yaw) * dy]
        result["delta_xy_in_start_base_frame"] = body_delta
        duration = results["duration_per_phase_s"]
        # Broad gates establish direction and meaningful movement, not real-robot fidelity.
        if name == "rotate_ccw":
            expected = twist[2] * duration
            motion_ok = 0.5 * expected < result["delta_yaw_rad"] < 1.5 * expected and math.hypot(dx, dy) < 0.1
        else:
            axis = 0 if name == "forward" else 1
            expected = twist[axis] * duration
            motion_ok = (0.5 * expected < body_delta[axis] < 1.5 * expected
                         and abs(body_delta[1 - axis]) < 0.1 and abs(result["delta_yaw_rad"]) < 0.15)
        result["passed"] = bool(motion_ok and abs(dz) < 0.02 and ROOT_HEIGHT-.03 < end["position_xyz"][2] < ROOT_HEIGHT+.02)
        results["phases"].append(result)
        print("RANGER_DRIVE_PHASE:", json.dumps(result), flush=True)
        command(robot, steering_ids, drive_ids, zero_steer, zero_speed)
        for _ in range(100):
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())

    results["passed"] = all(phase["passed"] for phase in results["phases"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("RANGER_DRIVE_RESULTS:", args.output.resolve(), flush=True)
    if not results["passed"]:
        raise RuntimeError("One or more drive phases failed the movement gates; inspect the result JSON.")
except BaseException:
    traceback.print_exc()
    # SimulationApp.close() can terminate the interpreter before a pending
    # exception reaches the shell. Preserve a failing exit code explicitly.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)
finally:
    app.close(skip_cleanup=True)

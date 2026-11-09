"""Bounded Isaac Lab articulation test using the imported UniFP B2+Z1 USD."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.assets.articulation import ArticulationCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

USD_PATH = Path(__file__).resolve().parent / "assets/b2z1/b2z1.usd"


class B2Z1SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=str(USD_PATH)),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.6),
            joint_pos={
                ".*L_hip_joint": 0.15,
                ".*R_hip_joint": -0.15,
                ".*_thigh_joint": 0.67,
                ".*_calf_joint": -1.32,
                "z1_waist": 0.0,
                "z1_shoulder": 1.48,
                "z1_elbow": -1.5,
                "z1_wrist_angle": 0.0,
                "z1_forearm_roll": 0.0,
                "z1_wrist_rotate": 1.57,
                "z1_jointGripper": -0.785,
            },
        ),
        actuators={
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit_sim=100.0,
                stiffness=100.0,
                damping=1.0,
            )
        },
    )


try:
    assert USD_PATH.is_file(), USD_PATH
    stage = Usd.Stage.Open(str(USD_PATH))
    mesh_count = sum(
        prim.IsA(UsdGeom.Mesh)
        for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies())
    )
    print("B2Z1_USD_MESHES_WITH_INSTANCES:", mesh_count, flush=True)
    print("B2Z1_STAGE: create simulation", flush=True)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args.device))
    print("B2Z1_STAGE: create scene", flush=True)
    scene = InteractiveScene(B2Z1SceneCfg(num_envs=1, env_spacing=2.0))
    print("B2Z1_STAGE: reset simulation", flush=True)
    sim.reset()
    print("B2Z1_STAGE: reset complete", flush=True)
    robot = scene["robot"]
    print("B2Z1_NUM_JOINTS:", robot.num_joints, flush=True)
    print("B2Z1_JOINT_NAMES:", robot.joint_names, flush=True)
    print("B2Z1_NUM_BODIES:", robot.num_bodies, flush=True)
    for _ in range(10):
        robot.set_joint_position_target(robot.data.default_joint_pos)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
    assert torch.isfinite(robot.data.root_pos_w).all().item()
    assert torch.isfinite(robot.data.joint_pos).all().item()
    print("B2Z1_ARTICULATION_PASS: 10 physics steps", flush=True)
except BaseException as exc:
    print("B2Z1_FAILURE:", type(exc).__name__, repr(exc), flush=True)
    raise
finally:
    print("B2Z1_STAGE: close app", flush=True)
    app.close()

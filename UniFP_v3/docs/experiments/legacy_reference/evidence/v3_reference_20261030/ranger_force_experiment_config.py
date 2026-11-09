"""Shared A1Arm names and torque limits for the active Ranger v3 model.

Retired independent force-estimator experiment definitions are preserved in
archive/v3_restart_cleanup_20261022/root_sources_before_cleanup.zip.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

URDF_PATH = Path(__file__).resolve().parent / "robot_description/a1arm_description/urdf/a1arm_rgm3_approx_sim_v3.urdf"
ARM_NAMES = [f"arm_joint{i}" for i in range(1, 7)]
_robot = ET.parse(URDF_PATH)
ARM_EFFORT_LIMITS = {
    name: float(_robot.find(f"joint[@name='{name}']/limit").get("effort")) for name in ARM_NAMES
}

#!/usr/bin/env python3
"""Build an explicitly approximate movable-wheel URDF from the supplied model."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


DESCRIPTION = Path(__file__).resolve().parents[1]
SOURCE = DESCRIPTION / "urdf/a1arm_rgm3.urdf"
DEFAULT_OUTPUT = DESCRIPTION / "urdf/a1arm_rgm3_approx_sim_v2.urdf"

# Public Ranger Mini 3 dimensions and total vehicle mass. Link-level inertias,
# wheel placement height, and actuator limits below remain estimates.
WHEELBASE_M = 0.494
TRACK_M = 0.364
TOTAL_MASS_KG = 75.0
WHEEL_RADIUS_M = 0.153
WHEEL_WIDTH_M = 0.14
WHEEL_CENTER_Z_M = -0.10
STEERING_EFFORT_NM = 35.0
DRIVE_EFFORT_NM = 80.0
STEERING_VELOCITY_RAD_S = 2.0
DRIVE_VELOCITY_RAD_S = 20.0

WHEELS = (
    ("fl", "fl_Link", 1.0, 1.0),
    ("fr", "fr_Link", 1.0, -1.0),
    ("bl", "bl_Link", -1.0, 1.0),
    ("br", "br_Link", -1.0, -1.0),
)


def sanitize_mesh_paths(robot: ET.Element) -> None:
    alias_dir = DESCRIPTION / "meshes/approx_sim"
    alias_dir.mkdir(parents=True, exist_ok=True)
    for mesh in robot.findall(".//mesh"):
        filename = mesh.get("filename")
        if not filename or filename.startswith(("package://", "/")):
            continue
        source = (SOURCE.parent / filename).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Mesh referenced by source URDF is missing: {source}")
        safe_name = f"{source.stem.replace('.', '_')}{source.suffix}"
        alias = alias_dir / safe_name
        if not alias.exists():
            alias.symlink_to(source)
        mesh.set("filename", f"../meshes/approx_sim/{safe_name}")


def element(parent: ET.Element, name: str) -> ET.Element:
    found = parent.find(name)
    if found is None:
        raise ValueError(f"Expected <{name}> under <{parent.tag}>")
    return found


def inertial_link(name: str) -> ET.Element:
    link = ET.Element("link", {"name": name})
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", {"value": "0.01"})
    ET.SubElement(inertial, "inertia", {
        "ixx": "0.00002", "ixy": "0", "ixz": "0",
        "iyy": "0.00002", "iyz": "0", "izz": "0.00002",
    })
    return link


def replace_wheel_geometry(link: ET.Element) -> None:
    for geometry_tag in ("visual", "collision"):
        geometry = link.find(f"{geometry_tag}/geometry")
        if geometry is None:
            raise ValueError(f"Missing {geometry_tag} geometry for {link.get('name')}")
        for child in list(geometry):
            geometry.remove(child)
        cylinder = ET.SubElement(geometry, "cylinder", {
            "radius": f"{WHEEL_RADIUS_M:.4f}",
            "length": f"{WHEEL_WIDTH_M:.4f}",
        })
        origin = link.find(f"{geometry_tag}/origin")
        if origin is None:
            origin = ET.Element("origin", {"xyz": "0 0 0", "rpy": "1.5707963 0 0"})
            link.find(geometry_tag).insert(0, origin)
        else:
            origin.set("xyz", "0 0 0")
            origin.set("rpy", "1.5707963 0 0")

    # The simplified cylinder is centered at the new drive joint. Retaining the
    # old CAD link's displaced COM would create an unintended eccentric wheel.
    inertial = element(link, "inertial")
    origin = element(inertial, "origin")
    origin.set("xyz", "0 0 0")
    origin.set("rpy", "0 0 0")
    mass = float(element(inertial, "mass").get("value"))
    transverse = mass * (3 * WHEEL_RADIUS_M ** 2 + WHEEL_WIDTH_M ** 2) / 12
    axial = mass * WHEEL_RADIUS_M ** 2 / 2
    inertia = element(inertial, "inertia")
    # Cylinder axis and drive rotation axis are both local Y.
    for name, value in {"ixx": transverse, "iyy": axial, "izz": transverse,
                        "ixy": 0.0, "ixz": 0.0, "iyz": 0.0}.items():
        inertia.set(name, f"{value:.9f}")


def build(output: Path) -> None:
    tree = ET.parse(SOURCE)
    robot = tree.getroot()
    robot.set("name", "a1arm_rgm3_approx_sim")
    robot.insert(0, ET.Comment(
        " APPROXIMATE SIMULATION ASSET. Ranger Mini 3 link inertias and actuator limits "
        "are not official per-link data; validate against the assembled robot. "
    ))

    links = {item.get("name"): item for item in robot.findall("link")}
    joints = {item.get("name"): item for item in robot.findall("joint")}

    # A floating base is the physical root. The old planar virtual joints were
    # intended for planning and would let simulation bypass wheel-ground forces.
    for joint_name in ("root_x_axis_joint", "root_y_axis_joint", "root_z_rotation_joint"):
        joint = joints.get(joint_name)
        if joint is not None:
            robot.remove(joint)
    for link_name in ("root_link1", "root_link2", "root_link3"):
        link = links.get(link_name)
        if link is not None:
            robot.remove(link)

    # Replace each rigid wheel attachment with independently steerable and
    # driven joints. Initial wheel centers use the Mini 3 published wheelbase
    # and track; z and suspension geometry are approximate.
    for prefix, wheel_link_name, x_sign, y_sign in WHEELS:
        original = joints.pop(f"{prefix}_joint")
        robot.remove(original)
        wheel_link = links[wheel_link_name]
        replace_wheel_geometry(wheel_link)

        steering_link_name = f"{prefix}_steering_link"
        steering_link = inertial_link(steering_link_name)
        robot_children = list(robot)
        robot.insert(robot_children.index(wheel_link), steering_link)
        steering_joint = ET.Element("joint", {"name": f"{prefix}_steering_joint", "type": "revolute"})
        ET.SubElement(steering_joint, "origin", {
            "xyz": f"{x_sign * WHEELBASE_M / 2:.4f} {y_sign * TRACK_M / 2:.4f} {WHEEL_CENTER_Z_M:.4f}",
            "rpy": "0 0 0",
        })
        ET.SubElement(steering_joint, "parent", {"link": "base_link"})
        ET.SubElement(steering_joint, "child", {"link": steering_link_name})
        ET.SubElement(steering_joint, "axis", {"xyz": "0 0 1"})
        ET.SubElement(steering_joint, "limit", {
            "lower": "-1.5708", "upper": "1.5708",
            "effort": f"{STEERING_EFFORT_NM:.1f}",
            "velocity": f"{STEERING_VELOCITY_RAD_S:.1f}",
        })
        ET.SubElement(steering_joint, "dynamics", {"damping": "0.2", "friction": "0.0"})
        robot_children = list(robot)
        robot.insert(robot_children.index(steering_link) + 1, steering_joint)

        drive_joint = ET.Element("joint", {"name": f"{prefix}_wheel_drive_joint", "type": "continuous"})
        ET.SubElement(drive_joint, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        ET.SubElement(drive_joint, "parent", {"link": steering_link_name})
        ET.SubElement(drive_joint, "child", {"link": wheel_link_name})
        ET.SubElement(drive_joint, "axis", {"xyz": "0 1 0"})
        ET.SubElement(drive_joint, "limit", {
            "effort": f"{DRIVE_EFFORT_NM:.1f}", "velocity": f"{DRIVE_VELOCITY_RAD_S:.1f}",
        })
        ET.SubElement(drive_joint, "dynamics", {"damping": "0.05", "friction": "0.0"})
        robot_children = list(robot)
        robot.insert(robot_children.index(steering_joint) + 1, drive_joint)

    # Lump unmodeled chassis, battery, suspension, hub motors and steering
    # hardware into base_link. Published 75 kg is the base vehicle, so retain
    # the arm, extension frame and cameras as additional payload mass.
    add_on_mass = 0.0
    platform_mass = 0.0
    for link in robot.findall("link"):
        mass = link.find("inertial/mass")
        if mass is not None:
            value = float(mass.get("value"))
            if link.get("name") in {"base_link", "bl_Link", "br_Link", "fl_Link", "fr_Link",
                                      "fl_steering_link", "fr_steering_link", "bl_steering_link", "br_steering_link"}:
                platform_mass += value
            else:
                add_on_mass += value
    base_link = links["base_link"]
    base_mass = element(base_link.find("inertial"), "mass")
    estimated_base_mass = TOTAL_MASS_KG - (platform_mass - float(base_mass.get("value")))
    base_mass.set("value", f"{estimated_base_mass:.6f}")
    base_inertial = element(base_link, "inertial")
    base_origin = element(base_inertial, "origin")
    base_origin.set("xyz", "-0.02427 -0.00303 0.05443")
    base_inertia = element(base_inertial, "inertia")
    # Box approximation for the added mass, based on the published envelope.
    mass_delta = estimated_base_mass - 0.16272
    x, y, z = 0.72, 0.50, 0.256
    base_inertia.set("ixx", f"{mass_delta * (y*y + z*z) / 12 + 0.00072:.6f}")
    base_inertia.set("iyy", f"{mass_delta * (x*x + z*z) / 12 + 0.00074:.6f}")
    base_inertia.set("izz", f"{mass_delta * (x*x + y*y) / 12 + 0.00002:.6f}")
    base_inertia.set("ixy", "0")
    base_inertia.set("ixz", "0")
    base_inertia.set("iyz", "0")

    sanitize_mesh_paths(robot)
    ET.indent(tree, space="  ")
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="utf-8", xml_declaration=True)
    print(f"Wrote approximate model: {output}")
    print(f"base_link mass: {estimated_base_mass:.3f} kg; platform mass: {TOTAL_MASS_KG:.3f} kg; "
          f"assembled model mass: {TOTAL_MASS_KG + add_on_mass:.3f} kg")
    print(f"wheel radius: {WHEEL_RADIUS_M:.3f} m; track: {TRACK_M:.3f} m; wheelbase: {WHEELBASE_M:.3f} m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    build(parser.parse_args().output.resolve())

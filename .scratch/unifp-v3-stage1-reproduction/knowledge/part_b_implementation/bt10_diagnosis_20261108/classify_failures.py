"""用终止谓词、接触合力与独立三角网格分类诊断事件并绘图。"""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation
import trimesh


PROJECT_ROOT = Path(__file__).resolve().parents[5]
LEGACY_PATH = PROJECT_ROOT / ".scratch/unifp-v3-failure-visualization/knowledge/audit_failure_geometry.py"
sys.path.insert(0, "/tmp/unifp-v3-geometry-fcl")
specification = importlib.util.spec_from_file_location("contact_geometry", LEGACY_PATH)
geometry = importlib.util.module_from_spec(specification)
specification.loader.exec_module(geometry)
URDF_PATH = PROJECT_ROOT / "UniFP_v3/resources/robots/v3_a1_desktop_train_v6_coplanar_r2/urdf/a1arm_rgm3_desktop_fixed_gripper.urdf"


def pose_matrix(pose: list[float]) -> np.ndarray:
    """Convert a native body xyz/xyzw pose into a world transform.

    Args:
        pose: Seven values, position in meters followed by quaternion xyzw.

    Returns:
        Rigid transform [4,4], with meter translations.
    """
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
    result[:3, 3] = pose[:3]
    return result


def link_mesh(link: ET.Element, kind: str) -> trimesh.Trimesh:
    """Load full-resolution link geometry in link coordinates for display/ground.

    Args:
        link: V6 URDF link containing mesh geometries.
        kind: Either visual or collision.

    Returns:
        Concatenated mesh, positions in link-local meters.

    Raises:
        OSError: A source mesh cannot be loaded.
    """
    parts = []
    for element in link.findall(kind):
        reference = element.find("geometry/mesh")
        mesh = trimesh.load((URDF_PATH.parent / reference.get("filename")).resolve(), force="scene").to_geometry()
        mesh.apply_scale(np.fromstring(reference.get("scale", "1 1 1"), sep=" "))
        if element.find("origin") is not None:
            mesh.apply_transform(geometry.pose_from_origin(element.find("origin")))
        parts.append(mesh)
    return trimesh.util.concatenate(parts)


def draw_event(name: str, meshes: list[trimesh.Trimesh], other_pose: np.ndarray,
               nearest_points: list[list[float]] | None, intersection_point_m: list[float] | None,
               output: Path, title: str) -> None:
    """Render full-resolution surfaces and a local section through the visual gap.

    Args:
        name: Stable output basename identifying the probe and terminal event.
        meshes: Base visual, counterpart visual, base collision, counterpart
            collision meshes in their respective link frames, in meters.
        other_pose: Counterpart-to-base transform [4,4].
        nearest_points: Optional two closest visual points in base meters; used
            only when the visual surfaces are disjoint.
        intersection_point_m: Optional FCL visual intersection contact position
            in base meters; anchors the section when the surfaces intersect.
        output: Existing diagnostic output directory.
        title: Chinese explanation of the event's evidence category.

    Note:
        Writes PNG and interactive HTML. Coincident display vertices are merged,
        but triangles are retained. The section is a geometric view, not a force map.
    """
    meshes = [mesh.copy() for mesh in meshes]
    meshes[1].apply_transform(other_pose)
    meshes[3].apply_transform(other_pose)
    figure, axis = plt.subplots(figsize=(9, 6))
    interactive = go.Figure()
    if nearest_points is not None:
        points = np.asarray(nearest_points)
        anchor = points.mean(axis=0)
        normal = points[1] - points[0]
        normal /= np.linalg.norm(normal)
    else:
        anchor = np.asarray(intersection_point_m) if intersection_point_m is not None else meshes[1].bounds.mean(axis=0)
        normal = np.array([0., 0., 1.])
    tangent = np.eye(3)[np.argmin(np.abs(normal))]
    tangent -= normal * np.dot(tangent, normal)
    tangent /= np.linalg.norm(tangent)
    for index, (mesh, color, label) in enumerate(zip(meshes, ("#7796b2", "#15945d", "#e45843", "#e2a62a"),
            ("机身真实视觉", "部件真实视觉", "机身碰撞代理", "部件碰撞代理"))):
        display = mesh.copy()
        display.merge_vertices(merge_tex=True, merge_norm=True, digits_vertex=12)
        vertices, faces = np.asarray(display.vertices, dtype=np.float32), np.asarray(display.faces, dtype=np.uint32)
        interactive.add_trace(go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], color=color, opacity=.8 if index < 2 else .25,
            visible=True if index < 2 else "legendonly", name=label, showlegend=True))
        section = mesh.section(plane_origin=anchor, plane_normal=np.cross(normal, tangent))
        if section is not None:
            for section_index, line in enumerate(section.discrete):
                relative = line - anchor
                axis.plot(relative @ tangent * 1000, relative @ normal * 1000, color=color,
                          linestyle="-" if index < 2 else "--", label=label if section_index == 0 else None)
    if nearest_points is not None:
        relative = points - anchor
        axis.plot(relative @ tangent * 1000, relative @ normal * 1000, "ko-", label="真实视觉最近点")
    section_note = "最近点截面" if nearest_points is not None else (
        "FCL视觉表面交点截面" if intersection_point_m is not None else "部件中心截面，非精确交点定位")
    axis.set(xlim=(-50, 50), ylim=(-30, 30), xlabel="局部切向（mm）", ylabel="局部分离法向（mm）", title=title + "\n" + section_note)
    axis.set_aspect("equal")
    axis.grid(alpha=.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / f"{name}.png", dpi=150)
    plt.close(figure)
    interactive.update_layout(title=title + "；点击图例可叠加碰撞代理", scene={"aspectmode": "data"})
    interactive.write_html(output / f"{name}.html", include_plotlyjs=True)


def main() -> None:
    """Classify all captured failures with explicit uncertainty and save examples.

    Raises:
        OSError: Probe evidence or V6 mesh references cannot be read.
        ValueError: Invalid poses or surface geometry prevent measurement.

    Note:
        Net-force pairing is heuristic. Surface intersection establishes geometry at
        the saved boundary, not contact causality or penetration depth. Separation
        cannot exclude earlier substep contact or shape containment.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
    font_manager.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
    links = {link.get("name"): link for link in ET.parse(URDF_PATH).getroot().findall("link")}
    surfaces, display_meshes = {}, {}
    base_vertices = link_mesh(links["base_link"], "collision").vertices
    rows, pictured = [], set()
    for failure_path in sorted(args.probes.glob("probe_*/failures.json")):
        for event_index, event in enumerate(json.loads(failure_path.read_text())):
            row: dict[str, Any] = {"probe": failure_path.parent.name, "event_index": event_index,
                                  "policy_step": event["policy_step"], "env_id": event["env_id"], "predicates": event["predicates"]}
            base_pose = pose_matrix(event["body_pose_world_m_xyzw"]["base_link"])
            base_world_vertices = trimesh.transform_points(base_vertices, base_pose)
            floor_z_m = event["environment_origin_world_m"][2]
            row["base_collision_lowest_above_floor_m"] = float(base_world_vertices[:, 2].min() - floor_z_m)
            row["tilt_failure"] = event["predicates"]["roll"] or event["predicates"]["pitch"]
            if not event["predicates"]["base_contact"]:
                row["contact_category"] = "无base接触终止，倾覆"
                rows.append(row)
                continue
            base_force = np.asarray(event["base_contact_vector_n"])
            candidates = []
            for body, vector in event["body_contact_vectors_n"].items():
                magnitude_n = np.linalg.norm(vector)
                if body == "base_link" or magnitude_n <= 1.:
                    continue
                residual_ratio = float(np.linalg.norm(base_force + vector) / max(np.linalg.norm(base_force), magnitude_n))
                candidates.append((residual_ratio, body, float(magnitude_n)))
            candidates.sort()
            row["opposite_net_force_candidates"] = candidates[:3]
            candidate = candidates[0][1] if candidates and candidates[0][0] < .15 else None
            row["counterpart_candidate"] = candidate
            if candidate is None:
                row["contact_category"] = "机身近地接触候选" if row["base_collision_lowest_above_floor_m"] <= .02 else "接触对未确定"
                rows.append(row)
                continue
            if links[candidate].find("visual/geometry/mesh") is None:
                row["contact_category"] = "候选部件无三角视觉，未确定"
                rows.append(row)
                continue
            pair_pose = np.linalg.inv(base_pose) @ pose_matrix(event["body_pose_world_m_xyzw"][candidate])
            for kind in ("visual", "collision"):
                for body in ("base_link", candidate):
                    if (body, kind) not in surfaces:
                        surfaces[body, kind] = geometry.load_surfaces(URDF_PATH, links[body], kind)
                row[kind] = geometry.measure_pair(surfaces["base_link", kind], surfaces[candidate, kind], pair_pose)
            if row["visual"]["surface_intersection"]:
                category = "机身与部件真实视觉表面相交"
            elif row["collision"]["surface_intersection"]:
                category = "仅碰撞代理表面相交"
            else:
                category = "终止边界表面分离，子步或余量待定"
            row["contact_category"] = category
            picture_key = (category, candidate)
            if picture_key not in pictured and len(pictured) < 10:
                for kind in ("visual", "collision"):
                    for body in ("base_link", candidate):
                        if (body, kind) not in display_meshes:
                            display_meshes[body, kind] = link_mesh(links[body], kind)
                basename = f"{failure_path.parent.name}_event{event_index}"
                closest = row["visual"]["closest_disjoint_pair"]
                nearest = closest["nearest_points_in_base_m"] if not row["visual"]["surface_intersection"] and closest else None
                intersection_point = None
                if row["visual"]["surface_intersection"]:
                    # Anchor intersecting examples at an actual full-mesh contact,
                    # rather than the counterpart center which can hide the base.
                    base_name, other_name = row["visual"]["intersecting_pairs"][0]
                    base_model = dict(surfaces["base_link", "visual"])[base_name]
                    other_model = dict(surfaces[candidate, "visual"])[other_name]
                    result = geometry.fcl.CollisionResult()
                    geometry.fcl.collide(
                        geometry.fcl.CollisionObject(base_model, geometry.fcl.Transform()),
                        geometry.fcl.CollisionObject(other_model, geometry.fcl.Transform(pair_pose[:3, :3], pair_pose[:3, 3])),
                        geometry.fcl.CollisionRequest(num_max_contacts=1, enable_contact=True), result)
                    if result.contacts:
                        intersection_point = result.contacts[0].pos.tolist()
                        row["visual_intersection_point_in_base_m"] = intersection_point
                draw_event(basename, [display_meshes[body, kind] for kind in ("visual", "collision") for body in ("base_link", candidate)], pair_pose, nearest, intersection_point,
                           args.output, f"{candidate}；{category}")
                row["visualization"] = basename
                pictured.add(picture_key)
            rows.append(row)
            if len(rows) % 10 == 0:
                print(json.dumps({"classified": len(rows), "last": category}, ensure_ascii=False), flush=True)
    summary = {"events": rows, "contact_categories": dict(Counter(row["contact_category"] for row in rows)),
               "tilt_failures": sum(row["tilt_failure"] for row in rows),
               "说明": "各probe为新episode独立采样；接触分类和倾覆可重叠，不把合力候选当成PhysX真实接触对。近地只作候选；几何只测终止边界，不能排除先前子步或实体包含。"}
    (args.output / "classification.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "events"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

"""Visualize saved native contact poses and separate terminal geometry evidence."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

PROJECT_ROOT = Path(__file__).resolve().parents[3]
COMPARISON_ROOT = PROJECT_ROOT / '.scratch/unifp-v3-stage1-reproduction/knowledge/model_history/unifp-v3-collision-proxy-repair/artifacts/visual_envelope_v8_20261106'
SUMMARY_PATH = COMPARISON_ROOT / 'comparison_summary.json'
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / 'artifacts/contact_classification_20261107'
ASSET_NAMES = {'v6': 'v3_a1_desktop_train_v6_coplanar_r2',
               'r3': 'v3_a1_desktop_train_v7_local_base_proxy_r3',
               'v8': 'v3_a1_desktop_train_v8_visual_envelope_r1'}
CATEGORY_LABELS = {'visual_surface_intersection': '真实视觉表面相交',
                   'proxy_only_surface_intersection': '仅代理表面相交',
                   'unresolved_terminal_separation': '终止边界分离，时序待定'}


def classify_terminal_event(event: dict[str, Any]) -> str:
    """Classify one terminal boundary without claiming a full contact history.

    Args:
        event: Saved summary event with counterpart_geometry visual/collision
            measurements. Multiple counterparts belong to the same failure.

    Returns:
        A category key: visual intersection takes precedence over proxy-only
        intersection; empty or disjoint evidence remains unresolved. Surface
        separation cannot rule out containment or earlier substep contact.
    """
    geometry = list(event.get('counterpart_geometry', {}).values())
    if any(item['visual']['surface_intersection'] for item in geometry):
        return 'visual_surface_intersection'
    if any(item['collision']['surface_intersection'] for item in geometry):
        return 'proxy_only_surface_intersection'
    return 'unresolved_terminal_separation'


def pose_matrix(pose_m_xyzw: list[float]) -> np.ndarray:
    """Convert a saved native body pose into a meter-valued world transform.

    Args:
        pose_m_xyzw: World xyz in meters followed by quaternion xyzw, shape [7].

    Returns:
        Body-to-world transform with shape [4,4].
    """
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat(pose_m_xyzw[3:7]).as_matrix()
    transform[:3, 3] = pose_m_xyzw[:3]
    return transform


def triangle_intersection_anchor(first_triangle_m: np.ndarray,
                                 second_triangle_m: np.ndarray) -> np.ndarray:
    """Locate a section anchor on two intersecting, noncoplanar triangles.

    Args:
        first_triangle_m: First surface triangle, shape [3,3], in base meters.
        second_triangle_m: Second surface triangle in the same frame and units.

    Returns:
        Mean of edge/triangle intersections, shape [3], on both surfaces.

    Raises:
        ValueError: No noncoplanar edge intersection can be reconstructed.
    """
    intersections_m = []
    # Reconstruct from actual vertices: FCL BVH contact.pos can fall outside the
    # intersecting triangles, so it is not a reliable section origin here.
    for edges, surface in ((first_triangle_m, second_triangle_m),
                           (second_triangle_m, first_triangle_m)):
        surface_axes = (surface[1:] - surface[0]).T
        plane_normal = np.cross(surface_axes[:, 0], surface_axes[:, 1])
        plane_normal /= np.linalg.norm(plane_normal)
        for edge_start, edge_end in zip(edges, np.roll(edges, -1, axis=0)):
            edge_delta = edge_end - edge_start
            denominator = np.dot(plane_normal, edge_delta)
            if abs(denominator) < 1e-12:
                continue
            fraction = np.dot(plane_normal, surface[0] - edge_start) / denominator
            if not 0 <= fraction <= 1:
                continue
            point_m = edge_start + fraction * edge_delta
            coordinates = np.linalg.lstsq(surface_axes, point_m - surface[0], rcond=None)[0]
            if coordinates.min() >= -1e-8 and coordinates.sum() <= 1 + 1e-8:
                intersections_m.append(point_m)
    if not intersections_m:
        raise ValueError('无法从相交三角面重建截面锚点；未使用不可靠的FCL接触位置')
    return np.mean(intersections_m, axis=0)


def load_link_mesh(urdf_path: Path, link: ET.Element, kind: str,
                   body_in_base: np.ndarray) -> trimesh.Trimesh:
    """Load all actual surfaces of a link and place them at the captured pose.

    Args:
        urdf_path: URDF resolving mesh references; no Gym visual flip is applied.
        link: XML link containing visual or collision mesh elements.
        kind: Geometry type, either visual or collision.
        body_in_base: Link-to-base_link transform, [4,4], meter translations.

    Returns:
        Concatenated original triangle mesh in base_link coordinates, in meters.

    Raises:
        OSError: A referenced mesh cannot be loaded.
    """
    parts = []
    for element in link.findall(kind):
        reference = element.find('geometry/mesh')
        mesh = trimesh.load((urdf_path.parent / reference.get('filename')).resolve(),
                            force='scene').to_geometry()
        mesh.apply_scale(np.fromstring(reference.get('scale', '1 1 1'), sep=' '))
        origin_transform = np.eye(4)
        origin = element.find('origin')
        if origin is not None:
            origin_transform[:3, :3] = Rotation.from_euler(
                'xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
            origin_transform[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
        mesh.apply_transform(body_in_base @ origin_transform)
        parts.append(mesh)
    return trimesh.util.concatenate(parts)


def save_inventory(summary: dict[str, Any], output_dir: Path) -> None:
    """Write all 71 event classifications, original metrics and a count plot.

    Args:
        summary: Existing three-asset, three-seed comparison with full evidence.
        output_dir: Diagnostic artifact directory, created by the CLI.

    Returns:
        None.

    Note:
        Writes classifications.csv, classification_summary.json and counts.png.
        Categories describe terminal surfaces, not guaranteed physical causes.
    """
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager, pyplot as plt

    totals = {variant: Counter() for variant in ASSET_NAMES}
    rows = []
    for run in summary['runs']:
        for event in run['events']:
            category = classify_terminal_event(event)
            totals[run['variant']][category] += 1
            geometry = list(event.get('counterpart_geometry', {}).values())
            rows.append({'variant': run['variant'], 'seed': run['seed'],
                         'policy_step': event['policy_step'], 'env_id': event['env_id'],
                         'category': category, 'category_zh': CATEGORY_LABELS[category],
                         'counterparts': '+'.join(event.get('counterpart_bodies', [])),
                         'base_contact_n': event['base_contact_norm_n'],
                         'minimum_visual_gap_mm': min((item['visual']['minimum_surface_distance_m'] * 1000
                                                       for item in geometry), default=None),
                         'minimum_collision_gap_mm': min((item['collision']['minimum_surface_distance_m'] * 1000
                                                          for item in geometry), default=None)})
    with (output_dir / 'classifications.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {'source_summary': str(SUMMARY_PATH),
              'source_sha256': hashlib.sha256(SUMMARY_PATH.read_bytes()).hexdigest(),
              'totals': totals, 'events': rows, 'category_labels': CATEGORY_LABELS,
              'note_zh': '仅分类终止边界。视觉表面相交支持真实自碰撞；仅代理相交包含设计裕量、近似误差及填隙，不能一概判成模型错误；边界分离不能排除先前子步或实体包含。1400轮探索策略不是8000轮成熟策略。'}
    (output_dir / 'classification_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams['font.family'] = ['Noto Sans CJK JP']
    figure, axis = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(3)
    for category, color in zip(CATEGORY_LABELS, ('#d54d43', '#e2a534', '#559fc3'), strict=True):
        values = np.array([totals[variant][category] for variant in ASSET_NAMES])
        bars = axis.bar(list(ASSET_NAMES), values, bottom=bottom, color=color, label=CATEGORY_LABELS[category])
        axis.bar_label(bars, labels=[str(value) if value else '' for value in values], label_type='center')
        bottom += values
    axis.set_ylabel('终止事件数（每资产：3seed×128环境×60秒）')
    axis.set_title('第1400轮冻结探索策略：终止边界的表面几何证据')
    axis.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
    figure.tight_layout()
    figure.savefig(output_dir / 'counts.png', dpi=160)
    plt.close(figure)


def save_event_views(summary: dict[str, Any], variant: str, seed: int, policy_step: int,
                     env_id: int, output_dir: Path) -> Path:
    """Render exact captured robot meshes as interactive HTML and a local section.

    Args:
        summary: Existing comparison identifying native failure files and assets.
        variant: One of v6, r3 or v8, choosing the actual asset geometry.
        seed: Recorded evaluation seed, 1001 through 1003.
        policy_step: Zero-based terminal policy step selecting an actual failure.
        env_id: Native environment index, needed when several fail on one step.
        output_dir: Existing diagnostic destination; no asset files are changed.

    Returns:
        Path to a standalone HTML with full robot geometry and geometry toggles.

    Raises:
        ValueError: The requested event is absent or no counterpart was resolved.
        AssertionError: Actual URDF bytes differ from the saved comparison identity.
        OSError: Evidence, meshes or output files cannot be read or written.

    Note:
        Writes per-event HTML, PNG and metadata. This is a static native snapshot,
        not a simulated trajectory or a force/penetration-depth reconstruction.
    """
    import fcl
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager, pyplot as plt
    import plotly.graph_objects as go
    from audit_failure_geometry import load_surfaces

    run = next(item for item in summary['runs'] if item['variant'] == variant and item['seed'] == seed)
    matches = [item for item in run['events'] if item['policy_step'] == policy_step and item['env_id'] == env_id]
    if not matches:
        raise ValueError('未找到该step/env事件；请先查看classifications.csv')
    event = matches[0]
    bodies = event.get('counterpart_bodies', [])
    if not bodies:
        raise ValueError('该事件未确定对侧刚体，请直接核对原始净力记录')
    urdf_path = PROJECT_ROOT / 'UniFP_v3/resources/robots' / ASSET_NAMES[variant] / 'urdf/a1arm_rgm3_desktop_fixed_gripper.urdf'
    assert hashlib.sha256(urdf_path.read_bytes()).hexdigest() == run['input_sha256'][str(urdf_path.relative_to(PROJECT_ROOT))]
    # Saved summary paths keep their historical identity after task consolidation.
    raw_directory = run['directory'].replace(
        '.scratch/unifp-v3-collision-proxy-repair/',
        '.scratch/unifp-v3-stage1-reproduction/knowledge/model_history/unifp-v3-collision-proxy-repair/')
    raw_path = PROJECT_ROOT / raw_directory / 'failures.json'
    raw_event = next(item for item in json.loads(raw_path.read_text())
                     if item['policy_step'] == policy_step and item['env_id'] == env_id)
    robot = ET.parse(urdf_path).getroot()
    links = {link.get('name'): link for link in robot.findall('link')}
    native_poses = raw_event['body_pose_world_m_xyzw']
    base_inverse = np.linalg.inv(pose_matrix(native_poses['base_link']))
    body_poses = {name: base_inverse @ pose_matrix(values) for name, values in native_poses.items()}
    visual_meshes = {name: load_link_mesh(urdf_path, link, 'visual', body_poses[name]) for name, link in links.items()}
    other_visual = trimesh.util.concatenate([visual_meshes[body] for body in bodies])
    meshes = {'base_visual': visual_meshes['base_link'], 'other_visual': other_visual,
              'context': trimesh.util.concatenate([mesh for name, mesh in visual_meshes.items()
                                                   if name != 'base_link' and name not in bodies]),
              'base_collision': load_link_mesh(urdf_path, links['base_link'], 'collision', np.eye(4)),
              'other_collision': trimesh.util.concatenate([
                  load_link_mesh(urdf_path, links[body], 'collision', body_poses[body]) for body in bodies])}
    category = classify_terminal_event(event)
    # Place the section at actual visual contact when surfaces intersect. For
    # separated surfaces use saved closest points, not the collision-proxy center.
    closest_body = min(bodies, key=lambda body: event['counterpart_geometry'][body]['visual']['minimum_surface_distance_m'])
    geometry = event['counterpart_geometry'][closest_body]
    nearest_points = None
    if geometry['visual']['surface_intersection']:
        base_model = load_surfaces(urdf_path, links['base_link'], 'visual')[0][1]
        other_models = load_surfaces(urdf_path, links[closest_body], 'visual')
        contacts = []
        for _, model in other_models:
            result = fcl.CollisionResult()
            pose = body_poses[closest_body]
            fcl.collide(fcl.CollisionObject(base_model),
                        fcl.CollisionObject(model, fcl.Transform(pose[:3, :3], pose[:3, 3])),
                        fcl.CollisionRequest(num_max_contacts=8, enable_contact=True), result)
            contacts.extend(result.contacts)
        contact = contacts[0]
        anchor_m = triangle_intersection_anchor(
            visual_meshes['base_link'].triangles[contact.b1],
            visual_meshes[closest_body].triangles[contact.b2])
        normal = other_visual.centroid - visual_meshes['base_link'].centroid
    else:
        nearest_points = np.asarray(geometry['visual']['closest_disjoint_pair']['nearest_points_in_base_m'])
        anchor_m = nearest_points[0]
        normal = nearest_points[1] - nearest_points[0]
    normal /= np.linalg.norm(normal)
    tangent = np.eye(3)[np.argmin(np.abs(normal))]
    tangent -= normal * np.dot(tangent, normal)
    tangent /= np.linalg.norm(tangent)
    title = (f'{variant} seed{seed} step{policy_step} env{env_id}：{CATEGORY_LABELS[category]}'
             f'<br>对侧 {" + ".join(bodies)}；base净力 {event["base_contact_norm_n"]:.2f} N；静态终止pose')
    colors = {'base_visual': '#8596a5', 'other_visual': '#14975c', 'context': '#c7ccd2',
              'base_collision': '#d84d4d', 'other_collision': '#f3aa30'}
    labels = {'base_visual': '机身真实视觉', 'other_visual': '接触对侧真实视觉', 'context': '机器人其他部件',
              'base_collision': '机身碰撞代理（含所有shape）', 'other_collision': '对侧碰撞代理'}
    figure = go.Figure()
    for key, mesh in meshes.items():
        # CAD exports repeat vertices. Share coincident display vertices without
        # removing triangles; quantitative sections above retain original meshes.
        display_mesh = mesh.copy()
        display_mesh.merge_vertices(merge_tex=True, merge_norm=True, digits_vertex=12)
        vertices_m = np.asarray(display_mesh.vertices, dtype=np.float32)
        faces = np.asarray(display_mesh.faces, dtype=np.uint32)
        figure.add_trace(go.Mesh3d(x=vertices_m[:, 0], y=vertices_m[:, 1], z=vertices_m[:, 2],
                         i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], name=labels[key], color=colors[key],
                         opacity=0.28 if 'collision' in key else (0.22 if key == 'context' else 0.85),
                         visible='legendonly' if 'collision' in key else True, showlegend=True))
    figure.update_layout(title=title, scene=dict(aspectmode='data', xaxis_title='base x（m）',
                         yaxis_title='base y（m）', zaxis_title='base z（m）'),
                         updatemenus=[dict(type='buttons', direction='left', y=1.07, buttons=[
                             dict(label='真实外形', method='update', args=[{'visible': [True, True, True, False, False]}]),
                             dict(label='外形＋代理叠加', method='update', args=[{'visible': [True] * 5}]),
                             dict(label='只看碰撞代理', method='update', args=[{'visible': [False, False, False, True, True]}])])])
    stem = output_dir / f'{variant}_seed{seed}_step{policy_step}_env{env_id}'
    figure.write_html(stem.with_suffix('.html'), include_plotlyjs=True, full_html=True)
    # A normal section exposes millimeter gaps that full-robot video cannot show.
    font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams['font.family'] = ['Noto Sans CJK JP']
    plot, axis = plt.subplots(figsize=(9, 6))
    for key in ('base_visual', 'other_visual', 'base_collision', 'other_collision'):
        section = meshes[key].section(plane_origin=anchor_m, plane_normal=np.cross(normal, tangent))
        if section is not None:
            for index, points_m in enumerate(section.discrete):
                relative_m = points_m - anchor_m
                axis.plot(relative_m @ tangent * 1000, relative_m @ normal * 1000,
                          color=colors[key], linestyle='--' if 'collision' in key else '-',
                          label=labels[key] if index == 0 else None)
    if nearest_points is not None:
        relative_m = nearest_points - anchor_m
        axis.plot(relative_m @ tangent * 1000, relative_m @ normal * 1000, 'ko-', label='真实视觉最近点连线')
    axis.set(xlim=(-20, 20), ylim=(-12, 16), aspect='equal', xlabel='局部切向（mm）', ylabel='局部分离法向（mm）',
             title=title.replace('<br>', '\n'))
    axis.grid(alpha=0.25)
    axis.legend(fontsize=9)
    plot.tight_layout()
    plot.savefig(stem.with_suffix('.png'), dpi=160)
    plt.close(plot)
    metadata = {'variant': variant, 'seed': seed, 'policy_step': policy_step, 'env_id': env_id,
                'category': category, 'counterparts': bodies, 'geometry': event['counterpart_geometry'],
                'visual_mesh_watertight': {name: visual_meshes[name].is_watertight for name in ['base_link', *bodies]},
                'section_anchor_in_base_m': anchor_m.tolist(), 'length_unit': 'm', 'coordinate_frame': 'base_link',
                'input_sha256': {str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in (SUMMARY_PATH, raw_path, urdf_path)},
                'note_zh': '网格应用URDF scale/origin及原生body pose各一次，不使用错误显示flip。HTML仅合并1e-12m精度下重合顶点并转float32显示，不删三角面；几何与截面仍用原网格。此为真实采样边界静态视图，非动作视频；叠加透明度不代表穿透深度，截面仅显示一个平面，不是全部接触面。视觉网格不封闭时不报告体积IoU。'}
    stem.with_suffix('.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    return stem.with_suffix('.html')


def main() -> None:
    """Export saved contact classification and one selectable native event view.

    Raises:
        ValueError: Requested event or counterpart evidence is unavailable.
        OSError: Source evidence or output files cannot be accessed.

    Note:
        Runs offline with NumPy/SciPy/Trimesh/Plotly/Matplotlib/python-fcl. No Gym,
        SSH, optimizer, asset mutation or physics-setting changes are performed.
    """
    parser = argparse.ArgumentParser(description='区分终止边界的真实外形相交／代理相交／时序待定，并绘制原生pose')
    parser.add_argument('--variant', choices=tuple(ASSET_NAMES), default='v8')
    parser.add_argument('--seed', type=int, choices=(1001, 1002, 1003), default=1003)
    parser.add_argument('--step', type=int, default=97)
    parser.add_argument('--env', type=int, default=96)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads(SUMMARY_PATH.read_text())
    save_inventory(summary, args.output_dir)
    html_path = save_event_views(summary, args.variant, args.seed, args.step, args.env, args.output_dir)
    print(json.dumps({'html': str(html_path), 'classification': str(args.output_dir / 'classifications.csv')},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

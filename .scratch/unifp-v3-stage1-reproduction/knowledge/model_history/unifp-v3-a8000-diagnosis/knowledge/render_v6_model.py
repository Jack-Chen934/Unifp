"""Render actual V6 visual meshes and highlight its revised rolling collisions."""

from pathlib import Path
import hashlib
import json
import sys
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import trimesh
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

PROJECT_ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(PROJECT_ROOT / 'UniFP_v3/tools'))
from build_desktop_asset import origin_transform
from compare_desktop_fk import forward_kinematics

ASSET_ROOT = PROJECT_ROOT / 'UniFP_v3/resources/robots/v3_a1_desktop_train_v6_coplanar_r2'
URDF_PATH = ASSET_ROOT / 'urdf/a1arm_rgm3_desktop_fixed_gripper.urdf'
OUTPUT_ROOT = Path(__file__).resolve().parents[1] / 'artifacts/model_preview'
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
robot = ET.parse(URDF_PATH).getroot()

# Match the active default joint posture without importing the Gym runtime.
joint_positions_rad = {'arm_joint2': 1.0, 'arm_joint3': -1.5}
base_pose_world = np.eye(4)
base_pose_world[2, 3] = 0.34  # Configured initial base height, in metres.
body_poses_world = forward_kinematics(URDF_PATH, joint_positions_rad, base_pose_world)
geometries = []
for link in robot.findall('link'):
    link_name = link.get('name')
    for element_kind in ('visual', 'collision'):
        for element in link.findall(element_kind):
            is_rolling_tire = 'rolling_tire' in element.get('name', '')
            if element_kind == 'collision' and not is_rolling_tire:
                continue
            reference = element.find('geometry/mesh')
            mesh_path = (URDF_PATH.parent / reference.get('filename')).resolve()
            local_mesh = trimesh.load(mesh_path, force='scene').to_geometry()
            local_mesh.apply_scale(np.fromstring(reference.get('scale', '1 1 1'), sep=' '))
            local_mesh.apply_transform(origin_transform(element.find('origin')))
            local_mesh.apply_transform(body_poses_world[link_name])
            color_element = element.find('material/color')
            # URDF colors take priority; the few uncolored links retain GLB colors.
            if color_element is not None:
                color_rgb = tuple(np.fromstring(color_element.get('rgba'), sep=' ')[:3])
            else:
                color_rgb = tuple(np.mean(local_mesh.visual.vertex_colors[:, :3], axis=0) / 255)
            geometries.append((link_name, element_kind, local_mesh, color_rgb))

view_specs = [
    ('perspective', (2.5, -3.3, 2.2), (0.0, 0.0, 0.77), 0.90, (1100, 1200), False),
    ('side', (0.0, -4.0, 0.77), (0.0, 0.0, 0.77), 0.88, (1000, 690), False),
    ('wheel_detail', (1.4, -1.9, 0.95), (0.0, 0.0, 0.16), 0.40, (1000, 560), True),
]
for view_name, camera_position, focal_point, parallel_scale, image_size, show_collision in view_specs:
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.965, 0.974, 0.985)
    for link_name, element_kind, mesh, color_rgb in geometries:
        if element_kind == 'collision' and not show_collision:
            continue
        if show_collision and not (link_name == 'base_link' or link_name.endswith('_Link')
                                   or 'steering' in link_name):
            continue
        # Convert the exact triangle surface to VTK, preserving CAD proportions.
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(numpy_to_vtk(np.asarray(mesh.vertices), deep=True))
        triangle_cells = np.column_stack((np.full(len(mesh.faces), 3), mesh.faces))
        vtk_cells = vtk.vtkCellArray()
        vtk_cells.SetCells(len(mesh.faces), numpy_to_vtkIdTypeArray(
            triangle_cells.astype(np.int64).ravel(), deep=True))
        surface = vtk.vtkPolyData()
        surface.SetPoints(vtk_points)
        surface.SetPolys(vtk_cells)
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(surface)
        normals.SetFeatureAngle(45)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(normals.GetOutputPort())
        mapper.ScalarVisibilityOff()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*(color_rgb if element_kind == 'visual' else (0.96, 0.36, 0.06)))
        actor.GetProperty().SetInterpolationToPhong()
        actor.GetProperty().SetAmbient(0.23)
        actor.GetProperty().SetDiffuse(0.70)
        actor.GetProperty().SetSpecular(0.18)
        actor.GetProperty().SetSpecularPower(30)
        if show_collision:
            actor.GetProperty().SetOpacity(0.32 if element_kind == 'visual' else 0.95)
        renderer.AddActor(actor)
    camera = renderer.GetActiveCamera()
    camera.SetPosition(*camera_position)
    camera.SetFocalPoint(*focal_point)
    camera.SetViewUp(0, 0, 1)
    camera.ParallelProjectionOn()
    camera.SetParallelScale(parallel_scale)
    renderer.ResetCameraClippingRange()
    lights = vtk.vtkLightKit()
    lights.SetKeyLightIntensity(0.85)
    lights.AddLightsToRenderer(renderer)
    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetSize(*image_size)
    window.SetMultiSamples(8)
    window.AddRenderer(renderer)
    window.Render()
    capture = vtk.vtkWindowToImageFilter()
    capture.SetInput(window)
    capture.Update()
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(str(OUTPUT_ROOT / (view_name + '.png')))
    writer.SetInputConnection(capture.GetOutputPort())
    writer.Write()
    window.Finalize()

# Compose labeled inspection panels; no geometry or dimensions are exaggerated.
canvas = Image.new('RGB', (2160, 1510), '#f6f9fc')
draw = ImageDraw.Draw(canvas)
font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
title_font = ImageFont.truetype(font_path, 43)
panel_font = ImageFont.truetype(font_path, 30)
note_font = ImageFont.truetype(font_path, 25)
draw.text((44, 24), 'V3 + A1Arm｜当前独立候选 V6 r2', fill='#152438', font=title_font)
draw.text((46, 84), '真实 URDF 网格渲染 · 默认臂位 · 视觉外形沿用桌面模型', fill='#526074', font=note_font)
canvas.paste(Image.open(OUTPUT_ROOT / 'perspective.png'), (0, 166))
canvas.paste(Image.open(OUTPUT_ROOT / 'side.png'), (1130, 154))
canvas.paste(Image.open(OUTPUT_ROOT / 'wheel_detail.png'), (1130, 910))
draw = ImageDraw.Draw(canvas)
draw.text((46, 146), '整体斜视：底盘、机械臂、夹爪与相机支架', fill='#23364d', font=panel_font)
draw.text((1150, 132), '侧视：保持真实装配与比例', fill='#23364d', font=panel_font)
draw.text((1150, 868), '轮组细节：橙色为补偿后的碰撞胎面', fill='#ad480d', font=panel_font)
draw.text((46, 1400), '14 DOF = 4 转向 + 4 轮驱动 + 6 机械臂；夹爪固定', fill='#23364d', font=note_font)
draw.text((46, 1446), '几何展示，非仿真落地截图；单轮碰撞半径调整 ≤ 2.3 mm，尚未通过训练准入。', fill='#526074', font=note_font)
canvas.save(OUTPUT_ROOT / 'v6_r2_model_overview.png')
metadata = {
    'asset': str(ASSET_ROOT.relative_to(PROJECT_ROOT)),
    'urdf_sha256': hashlib.sha256(URDF_PATH.read_bytes()).hexdigest(),
    'manifest_sha256': hashlib.sha256((ASSET_ROOT / 'asset_manifest.json').read_bytes()).hexdigest(),
    'base_height_m': 0.34,
    'joint_positions_rad': joint_positions_rad,
    'view_specs': view_specs,
    'note_zh': '从当前候选真实网格渲染，非仿真截图；橙色仅突出滚动碰撞分片，未放大补偿。',
}
(OUTPUT_ROOT / 'render_manifest.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
print(OUTPUT_ROOT / 'v6_r2_model_overview.png')

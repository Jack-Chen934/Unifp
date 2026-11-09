"""Build a versioned Mini v3 geometry asset from pinned manufacturer CAD.

Uses the existing v2 arm/control topology, replacing the chassis/wheel visuals
and correcting wheel contact geometry. The v2 asset is never overwritten.
Run with the isaaclab-unifp Python, then convert the printed URDF using the
existing Isaac Lab convert_urdf.py tool. Requires trimesh and pycollada.
"""
from pathlib import Path
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from copy import deepcopy

import numpy as np
import trimesh

DESCRIPTION = Path(__file__).resolve().parents[1]
VENDOR = DESCRIPTION / "vendor/agilex_ranger_mini_v3"
OUTPUT = DESCRIPTION / "urdf/a1arm_rgm3_approx_sim_v3.urdf"
MESHES = DESCRIPTION / "meshes/official_v3"
COMMIT = "27633a956c845903ee630538afeb17fe70afdd84"


def export_scene(name, scale):
    scene = trimesh.load(VENDOR / (name + ".dae"), force="scene")
    scene.apply_scale(scale)
    if name == "wheel_v3":
        # Manufacturer wheel has black CAD hub materials. Use silver hubs as
        # shown on the product page; preserve the source tread geometry.
        for geometry in scene.geometry.values():
            if len(geometry.faces) < 1000:
                geometry.visual.material.baseColorFactor = [175, 180, 188, 255]
                geometry.visual.material.metallicFactor = .65
    scene.export(MESHES / (name + ".glb"))
    return scene.bounds.tolist()


def visual(link, name, rpy="0 0 0"):
    for child in list(link.findall("visual")):
        link.remove(child)
    v = ET.SubElement(link, "visual")
    ET.SubElement(v, "origin", xyz="0 0 0", rpy=rpy)
    ET.SubElement(ET.SubElement(v, "geometry"), "mesh",
                  filename=f"../meshes/official_v3/{name}.glb")


def main():
    MESHES.mkdir(parents=True, exist_ok=True)
    # trimesh resolves the body DAE's millimetre units and scene transforms.
    # Its resulting body bounds are already metres; reapplying the xacro's
    # 1000 scale would incorrectly make it 750 m long. Wheel files need 10.
    bounds = {name: export_scene(name, scale) for name, scale in (
        ("ranger_base", 1.), ("steering_wheel", 10.), ("wheel_v3", 10.))}
    body = np.array(bounds["ranger_base"])
    if not (.65 < np.ptp(body[:, 0]) < .85 and .4 < np.ptp(body[:, 1]) < .6):
        raise ValueError(f"Unexpected manufacturer CAD units: {body}")
    tree = ET.parse(DESCRIPTION / "urdf/a1arm_rgm3_approx_sim_v2.urdf")
    robot = tree.getroot()
    robot.insert(0, ET.Comment(" V3 manufacturer visual geometry; approximate dynamics. See PROJECT_REPORT section 5.4. "))
    links = {l.get("name"): l for l in robot.findall("link")}
    joints = {j.get("name"): j for j in robot.findall("joint")}
    visual(links["base_link"], "ranger_base")
    # CAD bottom/top (-.223989/.016) + root .32 is 9 mm below
    # the manual's 105 mm clearance / 345 mm overall height. Register the
    # shell upward 9 mm, without stretching it or moving the arm mount.
    links["base_link"].find("visual/origin").set("xyz", "0 0 0.009")
    # Collision deliberately remains a smooth, documented approximation.
    base = links["base_link"]
    for c in list(base.findall("collision")):
        base.remove(c)
    c = ET.SubElement(base, "collision")
    ET.SubElement(c, "origin", xyz="0 0 -0.095", rpy="0 0 0")
    ET.SubElement(ET.SubElement(c, "geometry"), "box", size="0.70 0.46 0.24")
    base.find("inertial/origin").set("xyz", "0 0 -0.095")
    mass = float(base.find("inertial/mass").get("value"))
    inertia = base.find("inertial/inertia")
    for k, v in {"ixx": mass*(.50**2+.24**2)/12,
                 "iyy": mass*(.72**2+.24**2)/12,
                 "izz": mass*(.72**2+.50**2)/12}.items():
        inertia.set(k, str(v))
    for corner, sx, sy in (("fl", 1, 1), ("fr", 1, -1), ("bl", -1, 1), ("br", -1, -1)):
        # Manual wheelbase/track, symmetrical; retain positive-Z steering
        # convention used by the existing policy rather than vendor's -Z.
        joints[f"{corner}_steering_joint"].find("origin").set("xyz", f"{sx*.247} {sy*.185} -0.1")
        joints[f"{corner}_wheel_drive_joint"].find("origin").set("xyz", "0 0 -0.12")
        visual(links[f"{corner}_steering_link"], "steering_wheel",
               "0 0 3.14159265359" if sy > 0 else "0 0 0")
        wheel = links[f"{corner}_Link"]
        visual(wheel, "wheel_v3", "0 0 0" if sy > 0 else "3.14159265359 0 0")
        cyl = wheel.find("collision/geometry/cylinder")
        cyl.set("radius", "0.1")
        cyl.set("length", "0.08")
        m = float(wheel.find("inertial/mass").get("value"))
        i = wheel.find("inertial/inertia")
        for k, v in {"ixx":m*(3*.1**2+.08**2)/12,"iyy":m*.1**2/2,"izz":m*(3*.1**2+.08**2)/12}.items():
            i.set(k, str(v))
    # Fixed arm/frame attachment and all arm/control limits are inherited.
    ET.indent(tree, space="  ")
    tree.write(OUTPUT, encoding="utf-8", xml_declaration=True)
    provenance = dict(repository="https://github.com/agilexrobotics/ugv_gazebo_sim",
                      commit=COMMIT, package_license="BSD (package.xml; no root LICENSE found)",
                      converted_bounds_m=bounds, collision_radius_m=.1,
                      wheelbase_m=.494, track_m=.370, nominal_root_height_m=.32,
                      body_registration_z_m=.009,
                      note="CAD .1 m tire radius is a model assumption, not a measured real tire specification.",
                      files={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                             for p in VENDOR.iterdir() if p.suffix in (".dae", ".xacro", ".xml")})
    (VENDOR / "provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print(json.dumps(provenance, indent=2))
    print("URDF:", OUTPUT)


def original_visuals():
    """Keep source appearance, partitioning merged wheel CAD by motion frame."""
    source = DESCRIPTION / "urdf/a1arm_rgm3.urdf"
    tree = ET.parse(OUTPUT)
    robot = tree.getroot()
    original = ET.parse(source).getroot()
    originals = {l.get("name"): l for l in original.findall("link")}
    links = {l.get("name"): l for l in robot.findall("link")}
    joints = {j.get("name"): j for j in robot.findall("joint")}
    directory = DESCRIPTION / "meshes/original_movable"
    directory.mkdir(parents=True, exist_ok=True)

    def transform(node):
        t = trimesh.transformations.euler_matrix(*map(float, node.get("rpy", "0 0 0").split()))
        t[:3, 3] = list(map(float, node.get("xyz", "0 0 0").split()))
        return t

    def physics(root):
        root = deepcopy(root)
        for link in root.findall("link"):
            for v in list(link.findall("visual")):
                link.remove(v)
        for mat in list(root.findall("material")):
            root.remove(mat)
        return ET.canonicalize(ET.tostring(root, encoding="unicode"), strip_text=True)

    physical_before = physics(robot)
    # Restore every non-wheel visual, including the detailed bumper mesh.
    wheel_names = {f"{c}_Link" for c in ("fl", "fr", "bl", "br")}
    for name, link in links.items():
        for v in list(link.findall("visual")):
            link.remove(v)
        if name in originals and name not in wheel_names:
            for v in originals[name].findall("visual"):
                v = deepcopy(v)
                for mesh in v.findall("geometry/mesh"):
                    p = (source.parent / mesh.get("filename")).resolve()
                    safe = DESCRIPTION / "meshes/approx_sim" / (p.stem.replace(".", "_") + p.suffix)
                    if not safe.exists():
                        safe.symlink_to(p)
                    mesh.set("filename", "../meshes/approx_sim/" + safe.name)
                link.append(v)
    for mat in original.findall("material"):
        robot.append(deepcopy(mat))
    audit = {"source_urdf_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
             "physics_source": str(OUTPUT), "wheel_partition": {},
             "note": "All source visual triangles retained. Wheel groups assigned by connected source-face components; suspension remains rigid and contacts remain v3 approximations."}
    for corner in ("fl", "fr", "bl", "br"):
        v = originals[f"{corner}_Link"].find("visual")
        mesh_path = (source.parent / v.find("geometry/mesh").get("filename")).resolve()
        mesh = trimesh.load(mesh_path, force="scene").to_geometry()
        source_joint = original.find(f"joint[@name='{corner}_joint']/origin")
        mesh.apply_transform(transform(source_joint) @ transform(v.find("origin")))
        # Use connectivity only to locate the tire center. The source CAD is
        # an unwelded triangle soup, and trimesh.split() can duplicate/drop
        # faces while welding it, so export the original faces directly below.
        welded = mesh.copy()
        welded.merge_vertices()
        components = welded.split(only_watertight=False)
        tire = [p for p in components if .195 < p.extents[0] < .205 and .195 < p.extents[2] < .205]
        if len(tire) != 1:
            raise ValueError(f"Ambiguous tire component for {corner}: {len(tire)}")
        center = tire[0].bounds.mean(axis=0)
        steer = np.array(list(map(float, joints[f"{corner}_steering_joint"].find("origin").get("xyz").split())))
        drive = steer + np.array([0, 0, -.12])
        # Connected face indices preserve the exact source triangles (unlike
        # split(), which repairs component holes). Assign whole CAD parts so
        # wheel hubs do not remain stationary while the tire rotates.
        face_parts = trimesh.graph.connected_components(
            welded.face_adjacency, nodes=np.arange(len(mesh.faces)), min_len=1)
        groups = {name: [] for name in ("rolling", "steering", "fixed")}
        for face_ids in face_parts:
            vertices = mesh.vertices[mesh.faces[face_ids]].reshape(-1, 3)
            delta = vertices - center
            radial = np.linalg.norm(delta[:, [0, 2]], axis=1).max()
            if radial <= .101 and np.abs(delta[:, 1]).max() <= .074:
                group = "rolling"
            elif vertices[:, 2].min() >= center[2] + .09:
                group = "fixed"
            else:
                group = "steering"
            groups[group].extend(face_ids)
        groups = {name: np.asarray(ids, dtype=int) for name, ids in groups.items()}
        assigned = np.concatenate(list(groups.values()))
        assert np.array_equal(np.sort(assigned), np.arange(len(mesh.faces)))
        counts = {}
        for group, face_ids in groups.items():
            if not len(face_ids):
                continue
            out = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces[face_ids].copy(),
                                  process=False, visual=mesh.visual)
            # Align original tire axis with v3 contact axis; no scaling.
            out.apply_translation(drive - center)
            origin = {"rolling": drive, "steering": steer, "fixed": np.zeros(3)}[group]
            out.apply_translation(-origin)
            path = directory / f"{corner}_{group}.glb"
            out.export(path)
            parent = {"rolling": links[f"{corner}_Link"], "steering": links[f"{corner}_steering_link"], "fixed": links["base_link"]}[group]
            vis = ET.SubElement(parent, "visual", name=f"original_{corner}_{group}")
            ET.SubElement(vis, "origin", xyz="0 0 0", rpy="0 0 0")
            ET.SubElement(ET.SubElement(vis, "geometry"), "mesh", filename=f"../meshes/original_movable/{path.name}")
            counts[group] = len(out.faces)
        assert sum(counts.values()) == len(mesh.faces), (corner, counts, len(mesh.faces))
        audit["wheel_partition"][corner] = dict(faces=counts, source_faces=len(mesh.faces),
            registration_translation_m=(drive-center).tolist(), scale=1.)
    assert physics(robot) == physical_before, "Visual restoration changed physical parameters"
    output = DESCRIPTION / "urdf/a1arm_rgm3_original_movable.urdf"
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    audit["physical_xml_identical_to_v3"] = True
    audit["output"] = str(output)
    (directory / "provenance.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps(audit, indent=2))


def finish_usd():
    """Replace unresolved glTF MDL materials with portable USD surfaces."""
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True).app
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
    path = DESCRIPTION / ("assets/v3_original/a1arm_rgm3_original_movable.usd"
                          if "--original-visuals" in sys.argv else "assets/v3/a1arm_rgm3_approx_sim_v3.usd")
    stage = Usd.Stage.Open(str(path))
    root = str(stage.GetDefaultPrim().GetPath())
    try:
        for prim in list(stage.Traverse()):
            if prim.GetName() == "visuals":
                prim.SetInstanceable(False)
        cache = {}

        def surface(color, metallic=0.):
            key = (*color, metallic)
            if key not in cache:
                mat = UsdShade.Material.Define(stage, root+f"/OfficialLooks/m{len(cache)}")
                shader = UsdShade.Shader.Define(stage, str(mat.GetPath())+"/Shader")
                shader.CreateIdAttr("UsdPreviewSurface")
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
                shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.7 if max(color)<.2 else .4)
                shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
                mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
                cache[key] = mat
            return cache[key]

        count = 0
        for prim in list(stage.Traverse()):
            if "/visuals/" not in str(prim.GetPath()) or not prim.IsA(UsdGeom.Subset) and not prim.IsA(UsdGeom.Mesh):
                continue
            old = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
            if not old:
                continue
            shaders = [c for c in old.GetPrim().GetChildren() if c.IsA(UsdShade.Shader)]
            if not shaders:
                continue
            color = shaders[0].GetAttribute("inputs:base_color_factor").Get()
            if color is None:
                color = shaders[0].GetAttribute("inputs:diffuseColor").Get()
            if color is None:
                continue
            color = tuple(float(x) for x in color)
            if max(color) < .2:
                color = (.025, .028, .03)
            metallic = (shaders[0].GetAttribute("inputs:metallic_factor").Get()
                        or shaders[0].GetAttribute("inputs:metallic").Get() or 0.)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(surface(color, metallic))
            count += 1
        # Source CAD paints shell and lower cover together. Product images
        # distinguish the dark lower cover; split material faces only.
        xforms = UsdGeom.XformCache()
        base_inverse = xforms.GetLocalToWorldTransform(stage.GetPrimAtPath(root+"/base_link")).GetInverse()
        for prim in list(stage.Traverse()):
            if "/base_link/visuals/ranger_base/" not in str(prim.GetPath()) or not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(prim)
            points = np.asarray(mesh.GetPointsAttr().Get())
            counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
            assert np.all(counts == 3)
            faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
            relative = np.asarray(xforms.GetLocalToWorldTransform(prim) * base_inverse)
            base_points = np.column_stack((points, np.ones(len(points)))) @ relative
            lower = base_points[faces, 2].mean(axis=1) < -.10
            for subset in UsdGeom.Subset.GetAllGeomSubsets(mesh):
                if subset.GetPrim().GetName() == "LowerCover":
                    continue
                original = subset.GetPrim().GetAttribute("ranger:sourceMaterialIndices")
                if not original:
                    original = subset.GetPrim().CreateAttribute("ranger:sourceMaterialIndices", Sdf.ValueTypeNames.IntArray)
                    original.Set(subset.GetIndicesAttr().Get())
                ids = np.asarray(original.Get(), dtype=int)
                subset.GetIndicesAttr().Set(ids[~lower[ids]])
            sub = UsdGeom.Subset.CreateGeomSubset(UsdGeom.Imageable(prim), "LowerCover", UsdGeom.Tokens.face,
                                               np.flatnonzero(lower).tolist(), "materialBind")
            UsdShade.MaterialBindingAPI.Apply(sub.GetPrim()).Bind(surface((.025, .028, .03)))
        stage.GetRootLayer().Save()
        print("V3_PORTABLE_MATERIALS", count, flush=True)
    except BaseException:
        import os
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)  # SimulationApp.close can otherwise mask the exception.
    finally:
        app.close(skip_cleanup=True)


if __name__ == "__main__":
    if "--finish-usd" in sys.argv:
        finish_usd()
    elif "--original-visuals" in sys.argv:
        original_visuals()
    else:
        main()

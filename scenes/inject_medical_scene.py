"""TCP-injected into a running Isaac Sim GUI session.

Places a robot next to the operating table, adds a simple base block,
positions the CT mesh on top of the table, and frames the camera.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, Vt

PROJ = Path("/home/max/isaac_projects")
sys.path.insert(0, str(PROJ / "scenes"))
from phantom import CT_PHANTOM_MESH_OBJ

ROBOT_KIND = os.environ.get("ROBOT_KIND", "star").lower()  # star | franka
STAR_USD = "/home/max/.cache/i4h-assets/132c82d/Robots/STAR/star.usd"
FRANKA_USD = (
    "https://omniverse-content-production.s3-us-west-2."
    "amazonaws.com/Assets/Isaac/4.5/Isaac/Robots/Franka/franka.usd"
)

STAGE_PATH = "/home/max/isaac_projects/scenes/medical_scene.usd"
TABLE_PATH = "/World/Operating_table"
PHANTOM_PATH = "/World/Phantom/SpineMesh"
ROBOT_ROOT = "/World/Robot"
BASE_PATH = "/World/RobotBase"

stage = omni.usd.get_context().get_stage()
root_layer = stage.GetRootLayer()
if root_layer.realPath and os.path.realpath(root_layer.realPath) != STAGE_PATH:
    print(f"WARNING: Stage root is {root_layer.realPath}, not {STAGE_PATH}")

meters_per_unit = stage.GetMetadata("metersPerUnit") or 1.0
scale_factor = 1.0 / meters_per_unit
up_axis = (stage.GetMetadata("upAxis") or "Y").upper()
axis_index = {"X": 0, "Y": 1, "Z": 2}[up_axis]
plane_axes = [i for i in (0, 1, 2) if i != axis_index]
axis_a, axis_b = plane_axes

# --- Table bounds and center ---
if not stage.GetPrimAtPath(TABLE_PATH):
    raise RuntimeError(f"Missing table prim at {TABLE_PATH}")

bbox_cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])
world_bounds = bbox_cache.ComputeWorldBound(stage.GetPrimAtPath(TABLE_PATH))
min_pt = world_bounds.GetRange().GetMin()
max_pt = world_bounds.GetRange().GetMax()
center_a = (min_pt[axis_a] + max_pt[axis_a]) * 0.5
center_b = (min_pt[axis_b] + max_pt[axis_b]) * 0.5
center = [0.0, 0.0, 0.0]
center[axis_a] = center_a
center[axis_b] = center_b
center[axis_index] = (min_pt[axis_index] + max_pt[axis_index]) * 0.5

print("Table bounds:", min_pt, max_pt)

# --- Add base block ---
if stage.GetPrimAtPath(BASE_PATH):
    stage.RemovePrim(BASE_PATH)

base = UsdGeom.Cube.Define(stage, BASE_PATH)
base.CreateSizeAttr(1.0)
base.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.2, 0.2, 0.2)]))
def _set_or_add_op(xformable, op_name, value):
    for op in xformable.GetOrderedXformOps():
        if op.GetOpName() == op_name:
            if op_name == "xformOp:orient":
                # Coerce to the precision already on disk to avoid type mismatch.
                r, im = value.GetReal(), value.GetImaginary()
                prec = op.GetPrecision()
                coerced = (Gf.Quatd(r, Gf.Vec3d(*im))
                           if prec == UsdGeom.XformOp.PrecisionDouble
                           else Gf.Quatf(r, Gf.Vec3f(*im)))
                op.Set(coerced)
            else:
                op.Set(value)
            return
    if op_name == "xformOp:translate":
        xformable.AddTranslateOp().Set(value)
    elif op_name == "xformOp:scale":
        xformable.AddScaleOp().Set(value)
    elif op_name == "xformOp:orient":
        # New op — always use float; most Isaac Sim assets use float quats.
        r, im = value.GetReal(), value.GetImaginary()
        xformable.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(
            Gf.Quatf(r, Gf.Vec3f(*im))
        )


def set_xform(prim, translate=None, rotate=None, scale=None, orient=None):
    xformable = UsdGeom.Xformable(prim)
    if translate is not None:
        _set_or_add_op(xformable, "xformOp:translate", Gf.Vec3d(*translate))
    if orient is not None:
        _set_or_add_op(xformable, "xformOp:orient", orient)
    elif rotate is not None:
        xform_api = UsdGeom.XformCommonAPI(prim)
        xform_api.SetRotate(
            Gf.Vec3f(*rotate),
            UsdGeom.XformCommonAPI.RotationOrderXYZ,
        )
    if scale is not None:
        _set_or_add_op(xformable, "xformOp:scale", Gf.Vec3f(*scale))


base_xf = UsdGeom.Xformable(base)

# Hardcoded corrected position/scale (stage units = cm, Y-up).
BASE_TRANSLATE = (80.23421590587323, 31.597864985466003, -17.75986671447754)
BASE_SCALE     = (60.0, 60.0, 35.0)
set_xform(base, translate=BASE_TRANSLATE, scale=BASE_SCALE)

# --- Add robot reference ---
if stage.GetPrimAtPath(ROBOT_ROOT):
    stage.RemovePrim(ROBOT_ROOT)

robot_prim = stage.DefinePrim(ROBOT_ROOT, "Xform")
robot_usd = STAR_USD if ROBOT_KIND == "star" else FRANKA_USD
robot_prim.GetReferences().AddReference(robot_usd)

robot_xf = UsdGeom.Xformable(robot_prim)

# Hardcoded corrected pose (stage units = cm, Y-up).
# Orientation: 180° rotation around (0, 0.6983, 0.7158) so the arm faces the table.
ROBOT_TRANSLATE = (78.07537992456594, 61.59786605834972, -17.75986671447753)
ROBOT_ORIENT    = Gf.Quatf(0, Gf.Vec3f(0, 0.69827778804614, 0.7158268860006517))
set_xform(
    robot_prim,
    translate=ROBOT_TRANSLATE,
    orient=ROBOT_ORIENT,
    scale=(scale_factor, scale_factor, scale_factor),
)

print(f"Robot: {ROBOT_KIND} at {ROBOT_TRANSLATE}, scale={scale_factor}")

# --- Author joint drives + target positions (degrees) ---
# RemovePrim wipes any drives that were previously authored in this layer, so
# we must re-apply DriveAPI and set stiffness/damping every injection.
JOINT_TARGETS = {
    "star_joint_1":    -6.5,
    "star_joint_2":    55.5,
    "star_joint_3":     5.0,
    "star_joint_4":   -25.5,
    "star_joint_5":     8.3,
    "star_joint_6":    68.1,
    "star_joint_7":     0.0,
    "endo360_joint_1":  0.0,
}
_applied = set()
for prim in stage.Traverse():
    if not str(prim.GetPath()).startswith(ROBOT_ROOT + "/"):
        continue
    if prim.GetName() not in JOINT_TARGETS:
        continue
    deg = JOINT_TARGETS[prim.GetName()]
    drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(1e9)
    drive.CreateDampingAttr(1e7)
    drive.CreateMaxForceAttr(1e20)
    drive.CreateTargetPositionAttr(float(deg))
    _applied.add(prim.GetName())
    print(f"  drive {prim.GetPath()} → {deg}°")

_missing = set(JOINT_TARGETS) - _applied
if _missing:
    print(f"  WARNING: joints not found after reference load: {_missing}")

# --- Add CT mesh on the table ---
if not CT_PHANTOM_MESH_OBJ.exists():
    raise RuntimeError(f"Missing CT mesh at {CT_PHANTOM_MESH_OBJ}")

verts_npy = CT_PHANTOM_MESH_OBJ.with_suffix(".verts.npy")
faces_npy = CT_PHANTOM_MESH_OBJ.with_suffix(".faces.npy")
if not verts_npy.exists() or not faces_npy.exists():
    raise RuntimeError(f"Missing mesh sidecars: {verts_npy} or {faces_npy}")

verts = np.load(str(verts_npy))
faces = np.load(str(faces_npy))

if stage.GetPrimAtPath(PHANTOM_PATH):
    stage.RemovePrim(PHANTOM_PATH)

phantom = UsdGeom.Mesh.Define(stage, PHANTOM_PATH)
phantom.CreatePointsAttr(Vt.Vec3fArray(verts.tolist()))
phantom.CreateFaceVertexIndicesAttr(Vt.IntArray(faces.flatten().tolist()))
phantom.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
phantom.CreateSubdivisionSchemeAttr("none")
phantom.CreateDoubleSidedAttr(True)
phantom.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.93, 0.90, 0.82)]))

# Extent must be set so the renderer/BBoxCache use the correct world bounds.
local_min = verts.min(axis=0).tolist()
local_max = verts.max(axis=0).tolist()
phantom.CreateExtentAttr(
    Vt.Vec3fArray([Gf.Vec3f(*local_min), Gf.Vec3f(*local_max)])
)

# Hardcoded corrected position (stage units = cm, Y-up); identity orientation.
PHANTOM_TRANSLATE = (-32.84084072495667, 62.88815390170322, -17.75986671447754)

table_top = max_pt[axis_index]  # still needed for camera framing below

phantom_xf = UsdGeom.Xformable(phantom)
set_xform(
    phantom,
    translate=PHANTOM_TRANSLATE,
    orient=Gf.Quatf(1, Gf.Vec3f(0, 0, 0)),
    scale=(scale_factor, scale_factor, scale_factor),
)

print(f"Phantom placed at {PHANTOM_TRANSLATE}")

# --- Frame the camera on the table ---
camera_prim = stage.GetPrimAtPath("/OmniverseKit_Persp")
if camera_prim:
    cam_xf = UsdGeom.Xformable(camera_prim)
    cam_pos = [center[0], center[1], center[2]]
    cam_pos[axis_a] = center_a + 0.6 * scale_factor
    cam_pos[axis_b] = center_b - 0.6 * scale_factor
    cam_pos[axis_index] = table_top + 0.4 * scale_factor

    if up_axis == "Y":
        target = [center[0], center[1], center[2]]
        target[axis_index] = table_top + 0.1 * scale_factor
        dir_a = target[axis_a] - cam_pos[axis_a]
        dir_b = target[axis_b] - cam_pos[axis_b]
        dir_up = target[axis_index] - cam_pos[axis_index]
        horiz = max(np.hypot(dir_a, dir_b), 1e-6)
        yaw_deg = np.degrees(np.arctan2(dir_b, dir_a))
        pitch_deg = np.degrees(np.arctan2(dir_up, horiz))
        cam_rot = (pitch_deg, yaw_deg, 0.0)
    else:
        cam_rot = (45.0, 0.0, 135.0)

    set_xform(
        camera_prim,
        translate=cam_pos,
        rotate=cam_rot,
    )
    print(f"Camera positioned at {cam_pos}")

# --- Save the scene ---
root_layer.Save()
print(f"Saved stage: {root_layer.realPath}")
print("=== Scene update complete ===")

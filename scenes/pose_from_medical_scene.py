"""TCP-injected into a running Isaac Sim GUI session.

Extracts the STAR tool pose, phantom pose, and C-arm pose from
medical_scene.usd and writes pose.json for the registration pipeline.

The C-arm prim's rotation around the scene's up axis is also written to
pose.json as `carm_rotation_y_deg`.  The registration consumes this to set
the AP view angle (lateral defaults to AP + 90°).  Rotating the C-arm in
the GUI then changes which projection direction the registration sees.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import omni.usd
from pxr import Gf, Usd, UsdGeom

POSE_FILE = Path("/home/max/isaac_projects/output/pose.json")
POSE_TMP = POSE_FILE.with_suffix(".json.tmp")

EE_ENV = os.environ.get("EE_PRIM")
EE_CANDIDATES = [
    EE_ENV,
    "/World/Robot/endo360_needle",
    "/World/Robot/endo360_calibrated",
    "/World/Robot/star_link_ee",
    "/World/Robot/star_link_7",
    "/World/Robot/star_link_7/star_joint_ee",
]

PHANTOM_CANDIDATES = [
    "/World/Phantom/SpineMesh",
    "/World/Phantom",
]

CARM_CANDIDATES = [
    "/World/CArm",
    "/World/C_arm",
    "/World/Carm",
]


def _first_existing(stage, paths: list[str]) -> str | None:
    for p in paths:
        if not p:
            continue
        if stage.GetPrimAtPath(p):
            return p
    return None


def _world_pose(stage, prim_path: str, meters_per_unit: float):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise RuntimeError(f"Missing prim: {prim_path}")
    xf = UsdGeom.Xformable(prim)
    mat = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = mat.ExtractTranslation()
    q = mat.ExtractRotationQuat()
    pos_m = [float(t[0]) * meters_per_unit,
             float(t[1]) * meters_per_unit,
             float(t[2]) * meters_per_unit]
    quat_wxyz = [float(q.GetReal()),
                 float(q.GetImaginary()[0]),
                 float(q.GetImaginary()[1]),
                 float(q.GetImaginary()[2])]
    return pos_m, quat_wxyz


stage = omni.usd.get_context().get_stage()
meters_per_unit = stage.GetMetadata("metersPerUnit") or 1.0

print(f"Stage metersPerUnit: {meters_per_unit}")

# Resolve prims
phantom_path = _first_existing(stage, PHANTOM_CANDIDATES)
if not phantom_path:
    raise RuntimeError("Could not find phantom prim")

ee_path = _first_existing(stage, EE_CANDIDATES)
if not ee_path:
    raise RuntimeError("Could not find STAR end-effector prim")

phantom_pos, phantom_quat = _world_pose(stage, phantom_path, meters_per_unit)
ee_pos, ee_quat = _world_pose(stage, ee_path, meters_per_unit)

# ── C-arm pose ──────────────────────────────────────────────────────────────
# If a C-arm USD prim is present, read its actual world pose; otherwise fall
# back to the phantom isocenter (AP view).  The rotation around the PATIENT'S
# LONG AXIS (perpendicular to scene up — the axis the patient lies along on
# the operating table) becomes the LAO/RAO angle the registration uses for
# the AP view.  In a Y-up scene with a table whose long edge is along Z, the
# patient lies along Z, so we extract rotation around Z.
up_axis = (stage.GetMetadata("upAxis") or "Y").upper()
PATIENT_LONG_AXIS = {"Y": "Z", "Z": "Y", "X": "Z"}[up_axis]

carm_path = _first_existing(stage, CARM_CANDIDATES)
carm_rotation_deg = 0.0
if carm_path:
    carm_pos, carm_quat = _world_pose(stage, carm_path, meters_per_unit)
    # Extract rotation around the patient's long axis from the world quaternion
    w, x, y, z = carm_quat
    if PATIENT_LONG_AXIS == "X":
        # Roll: rotation around X
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        carm_rotation_deg = math.degrees(math.atan2(sinr_cosp, cosr_cosp))
    elif PATIENT_LONG_AXIS == "Y":
        # Pitch: rotation around Y
        sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
        carm_rotation_deg = math.degrees(math.asin(sinp))
    else:  # Z
        # Yaw: rotation around Z
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        carm_rotation_deg = math.degrees(math.atan2(siny_cosp, cosy_cosp))
else:
    carm_pos = list(phantom_pos)
    carm_quat = [1.0, 0.0, 0.0, 0.0]

pose = {
    "ee_pos": ee_pos,
    "ee_quat": ee_quat,
    "carm_pos": carm_pos,
    "carm_quat": carm_quat,
    "carm_rotation_y_deg": carm_rotation_deg,
    "phantom_pos": phantom_pos,
    "phantom_quat": phantom_quat,
    "timestamp": __import__("time").time(),
    "meta": {
        "ee_prim": ee_path,
        "phantom_prim": phantom_path,
        "carm_prim": carm_path or "(none — defaulting to phantom isocenter)",
        "scene_up_axis": up_axis,
    },
}

POSE_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(POSE_TMP, "w") as f:
    json.dump(pose, f, indent=2)
os.replace(POSE_TMP, POSE_FILE)

print(f"Wrote {POSE_FILE}")
print(f"  EE prim:      {ee_path}")
print(f"  Phantom prim: {phantom_path}")
print(f"  C-arm prim:   {carm_path or '(none)'}")
print(f"  EE pos (m):           {ee_pos}")
print(f"  Phantom pos (m):      {phantom_pos}")
print(f"  C-arm pos (m):        {carm_pos}")
print(f"  C-arm rotation deg:   {carm_rotation_deg:.2f}  "
      f"(around patient long axis {PATIENT_LONG_AXIS}; scene up={up_axis})")
print("=== pose.json ready ===")

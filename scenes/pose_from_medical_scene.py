"""TCP-injected into a running Isaac Sim GUI session.

Extracts the STAR tool pose and phantom pose from medical_scene.usd and writes
pose.json for the registration pipeline.
"""
from __future__ import annotations

import json
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

# C-arm pose: keep at phantom isocenter for AP view
carm_pos = list(phantom_pos)
carm_quat = [1.0, 0.0, 0.0, 0.0]

pose = {
    "ee_pos": ee_pos,
    "ee_quat": ee_quat,
    "carm_pos": carm_pos,
    "carm_quat": carm_quat,
    "phantom_pos": phantom_pos,
    "phantom_quat": phantom_quat,
    "timestamp": __import__("time").time(),
    "meta": {
        "ee_prim": ee_path,
        "phantom_prim": phantom_path,
    },
}

POSE_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(POSE_TMP, "w") as f:
    json.dump(pose, f, indent=2)
os.replace(POSE_TMP, POSE_FILE)

print(f"Wrote {POSE_FILE}")
print(f"  EE prim: {ee_path}")
print(f"  Phantom prim: {phantom_path}")
print(f"  EE pos (m): {ee_pos}")
print(f"  Phantom pos (m): {phantom_pos}")
print("=== pose.json ready ===")

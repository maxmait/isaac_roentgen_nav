"""TCP-injected: capture TWO C-arm views for multi-view registration.

Workflow:
    1. Pose the C-arm at the first view position (drag the rotate
       manipulator in the GUI viewport, or run rotate_carm.py first).
    2. Run this script — it records the current C-arm angle as SHOT 1.
    3. The script then pumps the Isaac Sim event loop for ~6 s — DURING
       this window you drag the C-arm rotation manipulator in the viewport
       to the second view position.  (Important: rotation must come from
       the GUI manipulator, NOT from another TCP injection like
       rotate_carm.py — TCP commands serialise behind this script's wait
       and won't be processed until it finishes.)
    4. It records the new C-arm angle as SHOT 2 and writes both into
       pose.json under `view_angles_deg`.
    5. Registration with USE_CARM_ROTATION=1 reads `view_angles_deg` and
       uses those two angles as the view list directly (no auto +90°).

Override the wait time with TWO_SHOT_WAIT_SEC in the injected scope:
    python3 scenes/isaacsim_client.py "TWO_SHOT_WAIT_SEC=8
    $(cat scenes/capture_two_shots.py)"

EE and phantom poses are constant across the two shots (the robot and
patient do not move during a clinical AP→lateral sweep), so they are
recorded once.
"""
import json
import math
import os
import time
from pathlib import Path

import omni.usd
from pxr import Usd, UsdGeom

POSE_FILE = Path("/home/max/isaac_projects/output/pose.json")
POSE_TMP  = POSE_FILE.with_suffix(".json.tmp")

EE_CANDIDATES = [
    "/World/Robot/endo360_needle",
    "/World/Robot/endo360_calibrated",
    "/World/Robot/star_link_ee",
    "/World/Robot/star_link_7",
]
PHANTOM_CANDIDATES = ["/World/Phantom/SpineMesh", "/World/Phantom"]
CARM_CANDIDATES = ["/World/CArm", "/World/C_arm", "/World/Carm"]

WAIT_SEC = float(globals().pop("TWO_SHOT_WAIT_SEC",
                                os.environ.get("TWO_SHOT_WAIT_SEC", 3.0)))


def first_existing(stage, paths):
    for p in paths:
        if stage.GetPrimAtPath(p):
            return p
    return None


def world_pose(stage, prim_path, mpu):
    prim = stage.GetPrimAtPath(prim_path)
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = m.ExtractTranslation()
    q = m.ExtractRotationQuat()
    pos_m = [float(t[i]) * mpu for i in range(3)]
    quat_wxyz = [float(q.GetReal())] + [float(v) for v in q.GetImaginary()]
    return pos_m, quat_wxyz


def carm_angle_around_patient_axis(stage, carm_path, up_axis):
    """Extract the C-arm's world rotation around the patient long axis."""
    _, q = world_pose(stage, carm_path, 1.0)
    w, x, y, z = q
    long_axis = {"Y": "Z", "Z": "Y", "X": "Z"}[up_axis]
    if long_axis == "X":
        return math.degrees(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    if long_axis == "Y":
        return math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))))
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


stage = omni.usd.get_context().get_stage()
mpu = stage.GetMetadata("metersPerUnit") or 1.0
up_axis = (stage.GetMetadata("upAxis") or "Y").upper()

ee_path = first_existing(stage, EE_CANDIDATES)
ph_path = first_existing(stage, PHANTOM_CANDIDATES)
ca_path = first_existing(stage, CARM_CANDIDATES)
if not (ee_path and ph_path and ca_path):
    raise RuntimeError(f"missing prim — ee={ee_path}, phantom={ph_path}, carm={ca_path}")

ee_pos,      ee_quat      = world_pose(stage, ee_path, mpu)
phantom_pos, phantom_quat = world_pose(stage, ph_path, mpu)

# ── SHOT 1 ──────────────────────────────────────────────────────────────────
shot1 = carm_angle_around_patient_axis(stage, ca_path, up_axis)
carm1_pos, carm1_quat = world_pose(stage, ca_path, mpu)
print("=" * 60)
print(f"SHOT 1 captured: C-arm angle = {shot1:+.2f}°")
print(f"  Drag the C-arm rotation manipulator in the GUI now —")
print(f"  capturing SHOT 2 in {WAIT_SEC} seconds...")
print("=" * 60)

# Pump the Isaac Sim event loop during the wait so the GUI stays responsive
# (the user needs to drag the manipulator).  Read the angle every 0.5 s and
# print so they can see their progress reflected on stdout.
import omni.kit.app
_app = omni.kit.app.get_app()
_t0 = time.time()
_last_print = 0.0
while time.time() - _t0 < WAIT_SEC:
    _app.update()
    now = time.time()
    if now - _last_print >= 0.5:
        cur = carm_angle_around_patient_axis(stage, ca_path, up_axis)
        remaining = WAIT_SEC - (now - _t0)
        print(f"  live C-arm angle: {cur:+6.2f}°  ({remaining:4.1f}s left)")
        _last_print = now

# ── SHOT 2 ──────────────────────────────────────────────────────────────────
shot2 = carm_angle_around_patient_axis(stage, ca_path, up_axis)
carm2_pos, carm2_quat = world_pose(stage, ca_path, mpu)
print(f"SHOT 2 captured: C-arm angle = {shot2:+.2f}°")
print(f"View angles list: [{shot1:+.2f}, {shot2:+.2f}]")

# ── Write pose.json ─────────────────────────────────────────────────────────
# Keep carm_pos / carm_quat = shot 1 for the world→isocenter transform
# (translation only depends on C-arm POSITION, which should be identical
# for both shots — the C-arm rotates about the isocenter).
pose = {
    "ee_pos":              ee_pos,
    "ee_quat":             ee_quat,
    "carm_pos":            carm1_pos,
    "carm_quat":           carm1_quat,
    "carm_rotation_y_deg": shot1,        # backward-compat: AP angle = shot 1
    "view_angles_deg":     [shot1, shot2],
    "phantom_pos":         phantom_pos,
    "phantom_quat":        phantom_quat,
    "timestamp":           time.time(),
    "meta": {
        "ee_prim":      ee_path,
        "phantom_prim": ph_path,
        "carm_prim":    ca_path,
        "scene_up_axis": up_axis,
        "shot1_carm_pos": carm1_pos,
        "shot2_carm_pos": carm2_pos,
    },
}

POSE_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(POSE_TMP, "w") as f:
    json.dump(pose, f, indent=2)
os.replace(POSE_TMP, POSE_FILE)
print(f"Wrote {POSE_FILE}")
print("=== Two shots ready for registration ===")
print("Now run:")
print("  DICOM_PATH=... USE_POSE_JSON=1 USE_CARM_ROTATION=1 \\")
print("      bridge/run_register_multiview.sh")

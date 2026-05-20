"""TCP-injected: append the current C-arm angle to the shots list.

Alternative to capture_two_shots.py that avoids the timed-sleep mechanic.
Workflow:

    # First call clears the list and records shot 1
    CARM_ROTATION_DEG=0   python3 .../isaacsim_client.py "CARM_ROTATION_DEG=0
                          $(cat scenes/rotate_carm.py)"
    RESET_SHOTS=1         python3 .../isaacsim_client.py "RESET_SHOTS=1
                          $(cat scenes/add_carm_shot.py)"

    # Second call appends shot 2
    CARM_ROTATION_DEG=90  python3 .../isaacsim_client.py "CARM_ROTATION_DEG=90
                          $(cat scenes/rotate_carm.py)"
                          python3 .../isaacsim_client.py < scenes/add_carm_shot.py

    # Any number of additional shots can be appended; registration will use
    # all of them as view angles.

Pose.json is updated atomically each call.  EE/phantom poses are refreshed
from the current scene state (they shouldn't move between shots, but we
record them every time anyway).
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

EE_CANDIDATES      = ["/World/Robot/endo360_needle", "/World/Robot/endo360_calibrated",
                      "/World/Robot/star_link_ee", "/World/Robot/star_link_7"]
PHANTOM_CANDIDATES = ["/World/Phantom/SpineMesh", "/World/Phantom"]
CARM_CANDIDATES    = ["/World/CArm", "/World/C_arm", "/World/Carm"]

# Pop, not get — the Isaac Sim executor keeps a persistent globals dict
# across TCP injections; without popping, RESET_SHOTS=1 from a previous
# call would leak into every subsequent call.
RESET = bool(int(globals().pop("RESET_SHOTS",
                                os.environ.get("RESET_SHOTS", "0"))))


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


def carm_angle(stage, carm_path, up_axis):
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
carm_pos,    carm_quat    = world_pose(stage, ca_path, mpu)
angle_now = carm_angle(stage, ca_path, up_axis)

# Load any existing shots, or start fresh on RESET
existing_views = []
if POSE_FILE.exists() and not RESET:
    try:
        prev = json.loads(POSE_FILE.read_text())
        existing_views = list(prev.get("view_angles_deg") or [])
    except Exception:
        existing_views = []

new_views = existing_views + [angle_now]

pose = {
    "ee_pos":              ee_pos,
    "ee_quat":             ee_quat,
    "carm_pos":            carm_pos,
    "carm_quat":           carm_quat,
    "carm_rotation_y_deg": angle_now,
    "view_angles_deg":     new_views,
    "phantom_pos":         phantom_pos,
    "phantom_quat":        phantom_quat,
    "timestamp":           time.time(),
    "meta": {
        "ee_prim":       ee_path,
        "phantom_prim":  ph_path,
        "carm_prim":     ca_path,
        "scene_up_axis": up_axis,
    },
}

POSE_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(POSE_TMP, "w") as f:
    json.dump(pose, f, indent=2)
os.replace(POSE_TMP, POSE_FILE)

print(f"{'RESET + ' if RESET else ''}Appended C-arm angle {angle_now:+.2f}°")
print(f"  Current view list: {[f'{a:+.2f}' for a in new_views]}°")
print(f"  Wrote {POSE_FILE}")

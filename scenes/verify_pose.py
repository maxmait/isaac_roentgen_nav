"""
Inject into a running Isaac Sim via isaacsim_client.py to read the live EE pose
and compare it to the last written pose.json.

Usage:
    python3 ~/isaac_projects/scenes/isaacsim_client.py \
        "$(cat ~/isaac_projects/scenes/verify_pose.py)"
"""

import json
import os
import numpy as np

POSE_FILE = os.path.expanduser("~/isaac_projects/output/pose.json")

# --- Live EE pose from Isaac Sim ---
from isaacsim.robot.manipulators.examples.franka import Franka
import omni.usd

stage = omni.usd.get_context().get_stage()

# Locate the existing Franka prim without calling world.reset()
franka_prim_path = "/World/Franka"
if not stage.GetPrimAtPath(franka_prim_path).IsValid():
    print(f"ERROR: No prim at {franka_prim_path} — is robot_scene.py running?")
else:
    from isaacsim.core.prims import XFormPrim
    from isaacsim.core.utils.xforms import get_world_pose

    ee_prim_path = "/World/Franka/panda_hand"
    ee_pos, ee_rot = get_world_pose(ee_prim_path)

    print("=== Live EE pose (from Isaac Sim stage) ===")
    print(f"  Position (xyz): {np.round(ee_pos, 4)}")
    print(f"  Orientation (quat wxyz): {np.round(ee_rot, 4)}")

    # --- Compare to pose.json ---
    if os.path.exists(POSE_FILE):
        with open(POSE_FILE) as f:
            saved = json.load(f)
        print(f"\n=== Last written pose.json ===")
        print(f"  ee_pos:  {np.round(saved['ee_pos'], 4)}")
        print(f"  ee_quat: {np.round(saved['ee_quat'], 4)}")
        print(f"  carm_pos:  {saved['carm_pos']}")
        print(f"  carm_quat: {saved['carm_quat']}")
        print(f"  timestamp: {saved['timestamp']}")

        diff = np.linalg.norm(np.array(ee_pos) - np.array(saved["ee_pos"]))
        print(f"\n  Position delta (live vs file): {diff:.6f} m")
    else:
        print(f"\nWARNING: {POSE_FILE} does not exist — is robot_scene.py running?")

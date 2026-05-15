from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import json
import os
import sys
import time

import numpy as np
from isaacsim.core.api import World
from isaacsim.robot.manipulators.examples.franka import Franka
from pxr import Gf, UsdGeom, Vt

# Make scenes/ importable when launched via ~/isaacsim/python.sh
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phantom import (
    BONE_CORE_COLOR,
    BONE_CORE_PATH,
    BONE_CORE_SEMIAXES_M,
    PHANTOM_POS_WORLD_M,
    PHANTOM_QUAT_WXYZ,
    PHANTOM_ROOT_PATH,
    SOFT_TISSUE_COLOR,
    SOFT_TISSUE_PATH,
    SOFT_TISSUE_SEMIAXES_M,
)

POSE_FILE = os.path.expanduser("~/isaac_projects/output/pose.json")
POSE_FILE_TMP = POSE_FILE + ".tmp"

# Default C-arm pose: positioned at the phantom isocenter -> AP view of the
# phantom. (carm_pos - phantom_pos) -> fluorosim translation = (0,0,0) mm,
# i.e. the C-arm's source-detector axis passes through the volume center.
CARM_POS = list(PHANTOM_POS_WORLD_M)
CARM_QUAT = [1.0, 0.0, 0.0, 0.0]

os.makedirs(os.path.dirname(POSE_FILE), exist_ok=True)

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

franka = world.scene.add(
    Franka(prim_path="/World/Franka", name="franka")
)


def add_ellipsoid(
    stage,
    path: str,
    semiaxes_m: tuple[float, float, float],
    center_world_m: tuple[float, float, float],
    color: tuple[float, float, float],
) -> None:
    """Add a UsdGeom.Sphere at `path` with non-uniform scale to form an ellipsoid.

    semiaxes_m is given in Isaac Sim (X, Y, Z) order. The sphere's unit radius
    is scaled by these semiaxes; the prim is translated to center_world_m.
    """
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(1.0)
    # Set extent so frustum culling / bbox computations match the scaled mesh.
    sphere.CreateExtentAttr([(-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)])
    sphere.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*color)]))

    xformable = UsdGeom.Xformable(sphere)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(*center_world_m))
    xformable.AddScaleOp().Set(Gf.Vec3f(*semiaxes_m))


stage = world.stage
# Parent Xform so the two ellipsoids share a single isocenter transform.
UsdGeom.Xform.Define(stage, PHANTOM_ROOT_PATH)
add_ellipsoid(stage, SOFT_TISSUE_PATH, SOFT_TISSUE_SEMIAXES_M,
              PHANTOM_POS_WORLD_M, SOFT_TISSUE_COLOR)
add_ellipsoid(stage, BONE_CORE_PATH, BONE_CORE_SEMIAXES_M,
              PHANTOM_POS_WORLD_M, BONE_CORE_COLOR)

world.reset()

step = 0
last_pose = None

while simulation_app.is_running():
    world.step(render=False)

    ee_pos, ee_rot = franka.end_effector.get_world_pose()

    pose = {
        "ee_pos": ee_pos.tolist(),
        "ee_quat": ee_rot.tolist(),
        "carm_pos": CARM_POS,
        "carm_quat": CARM_QUAT,
        "phantom_pos": list(PHANTOM_POS_WORLD_M),
        "phantom_quat": list(PHANTOM_QUAT_WXYZ),
        "timestamp": time.time(),
    }

    # Atomic write: write to .tmp then rename to avoid partial reads
    with open(POSE_FILE_TMP, "w") as f:
        json.dump(pose, f)
    os.replace(POSE_FILE_TMP, POSE_FILE)

    last_pose = pose

    if step % 100 == 0:
        joint_positions = franka.get_joint_positions()
        msg = (
            f"Step {step}\n"
            f"  EE position (xyz): {np.round(ee_pos, 4)}\n"
            f"  EE orientation (quat wxyz): {np.round(ee_rot, 4)}\n"
            f"  Joint angles: {np.round(joint_positions, 3)}\n"
            f"  pose.json written to {POSE_FILE}\n"
        )
        sys.stdout.write(msg)
        sys.stdout.flush()

    if step >= 500:
        break

    step += 1

done_msg = "=== Done ===\n"
if last_pose:
    done_msg += f"Last written pose:\n{json.dumps(last_pose, indent=2)}\n"
sys.stdout.write(done_msg)
sys.stdout.flush()
simulation_app.close()

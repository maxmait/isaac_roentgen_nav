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
    CT_PHANTOM_MESH_OBJ,
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
    """Add a UsdGeom.Sphere at `path` with non-uniform scale to form an ellipsoid."""
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(1.0)
    sphere.CreateExtentAttr([(-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)])
    sphere.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*color)]))
    xformable = UsdGeom.Xformable(sphere)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(*center_world_m))
    xformable.AddScaleOp().Set(Gf.Vec3f(*semiaxes_m))


def _load_obj(path) -> tuple[np.ndarray, np.ndarray]:
    """Load mesh from numpy arrays (fast) or parse OBJ text (fallback)."""
    from pathlib import Path
    p = Path(path)
    verts_npy = p.with_suffix(".verts.npy")
    faces_npy = p.with_suffix(".faces.npy")
    if verts_npy.exists() and faces_npy.exists():
        return np.load(str(verts_npy)), np.load(str(faces_npy))
    # Fallback: parse OBJ text
    verts, faces = [], []
    with open(p) as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:4]]
                faces.append(idx)
    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32)


def add_ct_mesh(
    stage,
    path: str,
    obj_path,
    center_world_m: tuple[float, float, float],
) -> bool:
    """Load the CT bone mesh (OBJ, metres, centred) as a UsdGeom.Mesh prim.

    Vertices are already in Isaac Sim world convention (X, Y, Z metres) and
    centred at the volume isocenter.  A single translate op places the mesh
    at center_world_m.  Returns True on success.
    """
    try:
        verts, faces = _load_obj(obj_path)
    except Exception as e:
        print(f"  WARNING: could not load CT mesh from {obj_path}: {e}",
              file=sys.stderr)
        return False

    mesh = UsdGeom.Mesh.Define(stage, path)
    # Use list() conversion — Vt arrays accept flat Python lists efficiently.
    mesh.CreatePointsAttr(Vt.Vec3fArray(verts.tolist()))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(faces.flatten().tolist()))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDisplayColorAttr(
        Vt.Vec3fArray([Gf.Vec3f(0.93, 0.90, 0.82)])   # warm bone colour
    )
    xf = UsdGeom.Xformable(mesh)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*center_world_m))
    print(f"  CT mesh loaded: {len(verts):,} verts  {len(faces):,} faces"
          f"  → {path}")
    return True


stage = world.stage
UsdGeom.Xform.Define(stage, PHANTOM_ROOT_PATH)

# Use the CT mesh when available; fall back to analytic ellipsoids.
if CT_PHANTOM_MESH_OBJ.exists():
    print(f"\n[Phantom] Loading CT mesh from {CT_PHANTOM_MESH_OBJ} ...")
    ok = add_ct_mesh(stage, PHANTOM_ROOT_PATH + "/Mesh",
                     CT_PHANTOM_MESH_OBJ, PHANTOM_POS_WORLD_M)
    if not ok:
        print("[Phantom] Mesh load failed — falling back to synthetic ellipsoids.")
        add_ellipsoid(stage, SOFT_TISSUE_PATH, SOFT_TISSUE_SEMIAXES_M,
                      PHANTOM_POS_WORLD_M, SOFT_TISSUE_COLOR)
        add_ellipsoid(stage, BONE_CORE_PATH, BONE_CORE_SEMIAXES_M,
                      PHANTOM_POS_WORLD_M, BONE_CORE_COLOR)
else:
    print(f"\n[Phantom] {CT_PHANTOM_MESH_OBJ} not found "
          f"— using synthetic ellipsoid phantom.")
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

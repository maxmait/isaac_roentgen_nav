"""TCP-injected into a running Isaac Sim GUI session.

Adds the CT spine mesh (and optionally a Franka reference) to the current
stage so we can take a viewport snapshot of the new phantom geometry.

Does NOT call world.reset() — uses pure USD API to add prims, which works
correctly in the TCP injection context.
"""
import os
import sys
from pathlib import Path

import numpy as np
import omni.usd
from pxr import Gf, Sdf, UsdGeom, Vt

PROJ = Path("/home/max/isaac_projects")
sys.path.insert(0, str(PROJ / "scenes"))
from phantom import (
    CT_PHANTOM_MESH_OBJ, PHANTOM_POS_WORLD_M, PHANTOM_ROOT_PATH,
)

stage = omni.usd.get_context().get_stage()

# ── Add Franka reference (so we can see scale next to the phantom) ─────────
FRANKA_PATH = "/World/Franka"
if not stage.GetPrimAtPath(FRANKA_PATH):
    franka_url = ("https://omniverse-content-production.s3-us-west-2."
                  "amazonaws.com/Assets/Isaac/4.5/Isaac/Robots/Franka/franka.usd")
    UsdGeom.Xform.Define(stage, "/World")
    franka_prim = stage.DefinePrim(FRANKA_PATH, "Xform")
    franka_prim.GetReferences().AddReference(franka_url)
    print(f"  Added Franka reference at {FRANKA_PATH}")
else:
    print(f"  {FRANKA_PATH} already on stage")

# ── Add CT mesh phantom ─────────────────────────────────────────────────────
verts_npy = CT_PHANTOM_MESH_OBJ.with_suffix(".verts.npy")
faces_npy = CT_PHANTOM_MESH_OBJ.with_suffix(".faces.npy")
if not verts_npy.exists() or not faces_npy.exists():
    print(f"  ERROR: {verts_npy} or {faces_npy} missing — run ct_to_mesh.py first")
else:
    verts = np.load(str(verts_npy))   # (N, 3) metres, centred at origin
    faces = np.load(str(faces_npy))   # (M, 3) int32
    print(f"  Loaded mesh: {len(verts):,} verts  {len(faces):,} faces")

    UsdGeom.Xform.Define(stage, PHANTOM_ROOT_PATH)
    mesh_path = PHANTOM_ROOT_PATH + "/SpineMesh"

    # Remove any previous version of this prim
    if stage.GetPrimAtPath(mesh_path):
        stage.RemovePrim(mesh_path)

    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    mesh.CreatePointsAttr(Vt.Vec3fArray(verts.tolist()))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(faces.flatten().tolist()))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.93, 0.90, 0.82)]))

    xf = UsdGeom.Xformable(mesh)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*PHANTOM_POS_WORLD_M))
    print(f"  Placed CT mesh at {mesh_path}  translate={PHANTOM_POS_WORLD_M}")

# ── Frame the viewport on the new geometry ─────────────────────────────────
# Move camera to look at the phantom centre from a useful angle
camera_prim = stage.GetPrimAtPath("/OmniverseKit_Persp")
if camera_prim:
    cam_xf = UsdGeom.Xformable(camera_prim)
    cam_xf.ClearXformOpOrder()
    # Camera 1m back and slightly above the phantom, looking at it
    cam_pos = (PHANTOM_POS_WORLD_M[0] + 0.8,
               PHANTOM_POS_WORLD_M[1] - 0.6,
               PHANTOM_POS_WORLD_M[2] + 0.4)
    cam_xf.AddTranslateOp().Set(Gf.Vec3d(*cam_pos))
    # Rotate roughly toward the phantom (45° around Z, then -30° around X)
    cam_xf.AddRotateXYZOp().Set(Gf.Vec3f(75.0, 0.0, 50.0))
    print(f"  Camera positioned at {cam_pos}")

print("=== Scene ready for snapshot ===")

"""TCP-injected into the running Isaac Sim GUI.

Adds a simple C-arm visualisation at carm_pos (read from pose.json):
- a torus segment shaped like a fluoroscopy C-arm
- a small "source" cube at the top of the C (where X-rays originate)
- a flat "detector" plate at the bottom of the C (where the DRR forms)

The C-arm is purely cosmetic — it does not participate in physics or
rendering of the DRR.  Its geometry is parameterised by fluorosim's
defaults (SDD = 1020 mm, SID = 510 mm) so the source/detector positions
match what the registration mathematically assumes.

The C-arm is centred at the phantom isocenter and the AP "beam axis"
points along the up-axis of the scene (Y in this medical_scene.usd).
"""
from __future__ import annotations
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import omni.usd
from pxr import Gf, UsdGeom, Vt

POSE_FILE = Path("/home/max/isaac_projects/output/pose.json")

# Fluorosim geometry constants (mm)
SID_MM = 510.0   # source to isocenter
SDD_MM = 1020.0  # source to detector
DET_HALF_MM = 256 * 0.5 / 2.0   # detector half-width (128 mm at edge)

CARM_PATH = "/World/CArm"

stage = omni.usd.get_context().get_stage()
mpu = stage.GetMetadata("metersPerUnit") or 1.0
up_axis = (stage.GetMetadata("upAxis") or "Y").upper()
up_idx = {"X": 0, "Y": 1, "Z": 2}[up_axis]
SCALE = 1.0 / mpu          # m → stage units (e.g. 100 for cm-unit stages)
MM_TO_STAGE = SCALE / 1000  # mm → stage units (so 510 mm → 51 cm in a cm-stage)

if not POSE_FILE.exists():
    raise RuntimeError(f"{POSE_FILE} missing — run pose_from_medical_scene.py first")
pose = json.loads(POSE_FILE.read_text())
carm_pos_m = pose["carm_pos"]
print(f"C-arm isocenter (m): {carm_pos_m}")

# Remove any previous C-arm
if stage.GetPrimAtPath(CARM_PATH):
    stage.RemovePrim(CARM_PATH)

# Parent Xform at the C-arm isocenter (= phantom_pos for AP view)
carm_root = UsdGeom.Xform.Define(stage, CARM_PATH)
xf = UsdGeom.Xformable(carm_root)
xf.ClearXformOpOrder()
xf.AddTranslateOp().Set(Gf.Vec3d(*(p * SCALE for p in carm_pos_m)))

# ── Build a C-shaped arc as a polyline mesh ──────────────────────────────────
# The C lies in a plane perpendicular to the "long axis" of the patient.
# For a scene with up-axis Y and patient lying on the table along Z (head-feet),
# the C-arm beam axis is Y (vertical AP).  The C arc lies in the X–Y plane.
arc_radius_mm = SID_MM * 1.1     # ~56 cm — slightly bigger than SID
arc_angle_deg = 240              # C opens 120° toward the patient
n_segments = 60
arc_thickness_mm = 60.0

theta = np.linspace(np.radians(-arc_angle_deg/2), np.radians(arc_angle_deg/2), n_segments)
arc_x = arc_radius_mm * np.sin(theta)
arc_y = -arc_radius_mm * np.cos(theta)   # bottom-open C (mouth points +Y up)

# Sweep a square cross-section along the arc to make a thick C-arm
def make_tube(centerline_xy, thickness_mm, mm_to_stage):
    """Extrude a square cross-section along the polyline (in XY plane, Z axis
    is the local "depth" of the C-arm)."""
    verts = []
    half = thickness_mm * 0.5 * mm_to_stage
    for x_mm, y_mm in centerline_xy:
        x = x_mm * mm_to_stage
        y = y_mm * mm_to_stage
        for dz, dr in [(-half, -half), (half, -half), (half, half), (-half, half)]:
            verts.append((x, y, dz))
    return verts

centerline = list(zip(arc_x, arc_y))
local_verts = make_tube(centerline, arc_thickness_mm, MM_TO_STAGE)

# Build triangular faces stitching consecutive cross-sections (quad strip)
faces = []
for i in range(n_segments - 1):
    a = i * 4
    b = (i + 1) * 4
    for k in range(4):
        k_next = (k + 1) % 4
        # Two triangles per quad face
        faces.append((a + k,      a + k_next, b + k_next))
        faces.append((a + k,      b + k_next, b + k))

# Axis remap: local (x_arc, y_arc, z_thickness) → world ... we need the C-arm
# in a plane normal to the patient's long axis.  For Y-up scenes the C arc
# should be in the X-Y plane (sagittal patient slice).  We use the local
# coords directly with the right axis mapping.
def remap(v):
    if up_axis == "Y":
        # local (x_arc, y_arc, z_thickness) → world (x_arc, y_arc, z_thickness)
        return v
    else:
        return v

mesh_path = CARM_PATH + "/Arc"
mesh = UsdGeom.Mesh.Define(stage, mesh_path)
mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*remap(v)) for v in local_verts]))
mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([i for tri in faces for i in tri]))
mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
mesh.CreateSubdivisionSchemeAttr("none")
mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.85, 0.75, 0.55)]))

# ── Source (small cube at the top of the C, SID above isocenter) ────────────
source_local = (0.0, SID_MM * MM_TO_STAGE, 0.0)
src = UsdGeom.Cube.Define(stage, CARM_PATH + "/Source")
src.CreateSizeAttr(1.0)
src.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.95, 0.3, 0.2)]))
sx = UsdGeom.Xformable(src)
sx.ClearXformOpOrder()
sx.AddTranslateOp().Set(Gf.Vec3d(*source_local))
sx.AddScaleOp().Set(Gf.Vec3f(0.06 * SCALE, 0.06 * SCALE, 0.06 * SCALE))

# ── Detector plate (flat box, opposite side of isocenter) ───────────────────
det_local = (0.0, -(SDD_MM - SID_MM) * MM_TO_STAGE, 0.0)
det = UsdGeom.Cube.Define(stage, CARM_PATH + "/Detector")
det.CreateSizeAttr(1.0)
det.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.2, 0.2, 0.25)]))
dx = UsdGeom.Xformable(det)
dx.ClearXformOpOrder()
dx.AddTranslateOp().Set(Gf.Vec3d(*det_local))
det_w = 0.30 * SCALE     # 30 cm wide
det_t = 0.04 * SCALE
dx.AddScaleOp().Set(Gf.Vec3f(det_w, det_t, det_w))

# ── Beam axis line (visual hint of the X-ray direction) ─────────────────────
beam = UsdGeom.BasisCurves.Define(stage, CARM_PATH + "/Beam")
beam.CreatePointsAttr(Vt.Vec3fArray([
    Gf.Vec3f(*source_local),
    Gf.Vec3f(*det_local),
]))
beam.CreateCurveVertexCountsAttr(Vt.IntArray([2]))
beam.CreateTypeAttr(UsdGeom.Tokens.linear)
beam.CreateWidthsAttr([0.005 * SCALE, 0.005 * SCALE])
beam.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(1.0, 0.6, 0.2)]))

print(f"C-arm added at {CARM_PATH}")
print(f"  Source at (local) {source_local}")
print(f"  Detector at (local) {det_local}")
print(f"  SID={SID_MM} mm, SDD={SDD_MM} mm  (matches fluorosim defaults)")

# Save the stage so the C-arm persists across reloads
stage.GetRootLayer().Save()
print(f"Saved {stage.GetRootLayer().realPath}")

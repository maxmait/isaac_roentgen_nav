"""TCP-injected: rotate the C-arm USD prim around the patient's long axis.

The C-arm prim ( /World/CArm ) is built by add_carm_viz.py with its source
and detector on the local Y axis (perpendicular to the patient).  Rotating
around the **patient's long axis** (world Z for Y-up scenes, world Y for
Z-up scenes — i.e. the axis the patient lies along) sweeps the source and
detector around the patient — AP → oblique → lateral.

This is the rotation that matters clinically and the rotation that
register_phantom_multiview.py reads as the AP view angle.

Usage:
    python3 scenes/isaacsim_client.py "CARM_ROTATION_DEG=30
    $(cat scenes/rotate_carm.py)"
"""
import os
import omni.usd
from pxr import Gf, UsdGeom

CARM_PATH = "/World/CArm"
# CARM_ROTATION_DEG can be set in the global scope before injecting this
# script, or via the (host) env var if the injector forwards it.
ANGLE_DEG = float(globals().get("CARM_ROTATION_DEG", os.environ.get("CARM_ROTATION_DEG", 0)))

stage = omni.usd.get_context().get_stage()
up_axis = (stage.GetMetadata("upAxis") or "Y").upper()
carm = stage.GetPrimAtPath(CARM_PATH)
if not carm:
    raise RuntimeError(f"{CARM_PATH} missing — run add_carm_viz.py first")

# Patient long axis = the axis the patient lies along on the table.
# For Y-up scenes (table top in XZ plane) the patient lies along Z.
# For Z-up scenes (table top in XY plane) the patient lies along Y.
# In either case it is the horizontal axis perpendicular to the C-arm's
# source/detector axis.
PATIENT_LONG_AXIS = {"Y": "Z", "Z": "Y", "X": "Z"}[up_axis]

xf = UsdGeom.Xformable(carm)
existing = list(xf.GetOrderedXformOps())
if not any(op.GetOpName() == "xformOp:rotateXYZ" for op in existing):
    xf.AddRotateXYZOp()

# Set rotation Euler vector: ANGLE_DEG goes on the patient-long-axis component
for op in xf.GetOrderedXformOps():
    if op.GetOpName() == "xformOp:rotateXYZ":
        if PATIENT_LONG_AXIS == "X":
            op.Set(Gf.Vec3f(ANGLE_DEG, 0.0, 0.0))
        elif PATIENT_LONG_AXIS == "Y":
            op.Set(Gf.Vec3f(0.0, ANGLE_DEG, 0.0))
        else:  # Z
            op.Set(Gf.Vec3f(0.0, 0.0, ANGLE_DEG))
        break

print(f"C-arm at {CARM_PATH} rotated to {ANGLE_DEG}° around patient long axis "
      f"({PATIENT_LONG_AXIS}) — scene up is {up_axis}")
stage.GetRootLayer().Save()
print(f"Saved {stage.GetRootLayer().realPath}")

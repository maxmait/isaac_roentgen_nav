"""Inject into a running Isaac Sim via isaacsim_client.py to confirm that the
phantom prims live at the expected world-frame isocenter.

Usage:
    python3 ~/isaac_projects/scenes/isaacsim_client.py \
        "$(cat ~/isaac_projects/scenes/verify_phantom.py)"

Prerequisites: the running Isaac Sim must have executed (or had injected) the
phantom-adding code from robot_scene.py. The headless robot_scene.py terminates,
so this script is mainly useful when the phantom prims were added to a
long-running GUI session via TCP injection.
"""

import json
import os
import sys

scenes_dir = os.path.expanduser("~/isaac_projects/scenes")
if scenes_dir not in sys.path:
    sys.path.insert(0, scenes_dir)

from phantom import (
    BONE_CORE_PATH,
    BONE_CORE_SEMIAXES_M,
    PHANTOM_POS_WORLD_M,
    SOFT_TISSUE_PATH,
    SOFT_TISSUE_SEMIAXES_M,
)

import omni.usd
from pxr import Gf, UsdGeom

stage = omni.usd.get_context().get_stage()

print("=== Phantom prim verification ===")
print(f"Expected isocenter (world m): {PHANTOM_POS_WORLD_M}")
print(f"Expected SoftTissue semiaxes (m): {SOFT_TISSUE_SEMIAXES_M}")
print(f"Expected BoneCore semiaxes (m):   {BONE_CORE_SEMIAXES_M}")
print()

for prim_path, expected_semiaxes in [
    (SOFT_TISSUE_PATH, SOFT_TISSUE_SEMIAXES_M),
    (BONE_CORE_PATH, BONE_CORE_SEMIAXES_M),
]:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"  MISSING: {prim_path}")
        continue
    xformable = UsdGeom.Xformable(prim)
    matrix = xformable.ComputeLocalToWorldTransform(0.0)
    translation = matrix.ExtractTranslation()
    # Scale comes from the SRT decomposition; extract the diagonal of the upper-left 3x3
    # after normalizing each column.
    cols = [Gf.Vec3d(matrix[i][0], matrix[i][1], matrix[i][2]) for i in range(3)]
    scale = tuple(col.GetLength() for col in cols)

    print(f"  {prim_path}")
    print(f"    world translation (m): "
          f"({translation[0]:.4f}, {translation[1]:.4f}, {translation[2]:.4f})")
    print(f"    world scale (semiaxes m): "
          f"({scale[0]:.4f}, {scale[1]:.4f}, {scale[2]:.4f})")
    print(f"    expected semiaxes (m):    {expected_semiaxes}")

# Cross-check with pose.json
pose_file = os.path.expanduser("~/isaac_projects/output/pose.json")
if os.path.exists(pose_file):
    with open(pose_file) as f:
        saved = json.load(f)
    print(f"\n  pose.json phantom_pos: {saved.get('phantom_pos')}")
    print(f"  pose.json phantom_quat: {saved.get('phantom_quat')}")

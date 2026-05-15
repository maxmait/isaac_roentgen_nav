"""Shared phantom geometry — single source of truth for the Isaac Sim scene
and the fluorosim renderer.

The synthetic phantom is the concentric-sphere CT volume that fluorosim
generates inside fluorosim_render.py:

    shape           = (128, 256, 256)   # (Z, Y, X) voxel order
    spacing_zyx_mm  = (1.0, 0.5, 0.5)
    bone core       = HU 800,  voxel radius 40   (dist function uses voxel indices)
    soft tissue     = HU  40,  voxel radius 60
    air background  = HU -900

Because spacing is anisotropic, the "spheres" in voxel space are ellipsoids
in physical space. The physical semiaxes (in Isaac Sim's X, Y, Z order, with
Z corresponding to fluorosim's slow axis) are:

    soft tissue:  (X=30, Y=30, Z=60) mm
    bone core:    (X=20, Y=20, Z=40) mm

Coordinate convention (fluorosim <-> Isaac Sim):
    fluorosim volume-local axis Z (slow)   <->  Isaac Sim world Z (up)
    fluorosim volume-local axis Y          <->  Isaac Sim world Y
    fluorosim volume-local axis X (fast)   <->  Isaac Sim world X
    fluorosim isocenter                    <->  PHANTOM_POS_WORLD_M
"""

from __future__ import annotations

# Phantom isocenter pose in Isaac Sim world frame (meters / wxyz quaternion).
# Picked to sit in the Franka workspace, above the ground plane.
PHANTOM_POS_WORLD_M: tuple[float, float, float] = (0.5, 0.0, 0.3)
PHANTOM_QUAT_WXYZ: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

# Volume-local semiaxes in meters (Isaac Sim X, Y, Z order).
SOFT_TISSUE_SEMIAXES_M: tuple[float, float, float] = (0.030, 0.030, 0.060)
BONE_CORE_SEMIAXES_M: tuple[float, float, float] = (0.020, 0.020, 0.040)

# USD prim paths for the phantom assemblies.
PHANTOM_ROOT_PATH: str = "/World/Phantom"
SOFT_TISSUE_PATH: str = "/World/Phantom/SoftTissue"
BONE_CORE_PATH: str = "/World/Phantom/BoneCore"

# Display colors (linear RGB, 0-1). Light salmon for soft tissue, white for bone.
SOFT_TISSUE_COLOR: tuple[float, float, float] = (0.85, 0.55, 0.50)
BONE_CORE_COLOR: tuple[float, float, float] = (0.95, 0.95, 0.92)

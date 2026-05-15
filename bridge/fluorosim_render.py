#!/usr/bin/env python3
"""Render a single DRR from pose.json (runs inside the fluorosim Docker container).

Expects bind mounts:
    /workspace/io   <-  ~/isaac_projects/output

Inputs (read from /workspace/io):
    pose.json   - written by Isaac Sim (scenes/robot_scene.py)

Outputs (written to /workspace/io):
    drr.png         - rendered DRR
    drr.npy         - raw float32 DRR array
    drr_meta.json   - pose used + image stats, for visual validation
    fluorosim_cache/  - preprocessed mu_volume cache (persists between runs)

The C-arm pose in pose.json is given as (carm_pos[m], carm_quat[wxyz]) in the
Isaac Sim world frame. The CT volume's isocenter is placed at phantom_pos in
the same world frame, so the fluorosim translation (which is volume-local) is

    translation_mm = (carm_pos - phantom_pos) * 1000.0

Older pose.json files without a phantom_pos field default to phantom at the
world origin (i.e. carm_pos itself is treated as the volume-local offset).
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

from fluorosim import (
    CarmGeometry,
    FluoroSimulator,
    OutputSettings,
    Pose,
    PreprocessedVolume,
    RealismSettings,
    SimulatorConfig,
    VolumePreprocessor,
)

IO_DIR = Path("/workspace/io")
POSE_FILE = IO_DIR / "pose.json"
DRR_PNG = IO_DIR / "drr.png"
DRR_NPY = IO_DIR / "drr.npy"
DRR_META = IO_DIR / "drr_meta.json"
CACHE_DIR = IO_DIR / "fluorosim_cache"


def quat_wxyz_to_euler_xyz(w: float, x: float, y: float, z: float) -> tuple[float, float, float]:
    """Convert a unit quaternion (w, x, y, z) to extrinsic XYZ Euler angles (rad)."""
    # roll (x-axis)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    # pitch (y-axis)
    sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sinp)
    # yaw (z-axis)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return (roll, pitch, yaw)


def load_or_build_synthetic_volume() -> PreprocessedVolume:
    """Reuse cached mu_volume.npy if present; otherwise build a synthetic sphere phantom."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached_mu = CACHE_DIR / "mu_volume.npy"
    if cached_mu.exists():
        print(f"  Reusing cached preprocessed volume: {CACHE_DIR}")
        return PreprocessedVolume.load(CACHE_DIR)

    print("  Building synthetic sphere phantom (one-time, then cached)...")
    shape = (128, 256, 256)
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    center = np.array(shape) / 2.0
    dist = np.sqrt(
        (z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2
    )
    hu_volume = np.full(shape, -900.0, dtype=np.float32)  # air
    hu_volume[dist < 60] = 40.0   # soft tissue shell
    hu_volume[dist < 40] = 800.0  # bone core

    preproc = VolumePreprocessor.from_numpy(
        hu_volume, spacing_zyx_mm=(1.0, 0.5, 0.5)
    )
    return preproc.preprocess(output_dir=CACHE_DIR)


def main() -> int:
    if not POSE_FILE.exists():
        print(f"ERROR: {POSE_FILE} not found. Run robot_scene.py first.", file=sys.stderr)
        return 1

    with open(POSE_FILE) as f:
        pose_data = json.load(f)

    ee_pos = pose_data["ee_pos"]
    ee_quat = pose_data["ee_quat"]
    carm_pos_m = pose_data["carm_pos"]
    carm_quat = pose_data["carm_quat"]
    phantom_pos_m = pose_data.get("phantom_pos", [0.0, 0.0, 0.0])
    phantom_quat = pose_data.get("phantom_quat", [1.0, 0.0, 0.0, 0.0])
    timestamp = pose_data["timestamp"]

    # World -> isocenter transform: express the C-arm in the volume's local frame.
    # Phantom rotation is assumed identity for now; once it's allowed to rotate
    # we'll need to compose the inverse phantom orientation into the C-arm pose.
    if phantom_quat != [1.0, 0.0, 0.0, 0.0]:
        print("WARNING: non-identity phantom rotation not yet supported; "
              "treating phantom as un-rotated.", file=sys.stderr)
    carm_local_m = [carm_pos_m[i] - phantom_pos_m[i] for i in range(3)]

    # Convert C-arm pose: quat(wxyz) -> Euler XYZ rad, pos(m) -> translation(mm)
    rot_x, rot_y, rot_z = quat_wxyz_to_euler_xyz(*carm_quat)
    tx_mm, ty_mm, tz_mm = (v * 1000.0 for v in carm_local_m)

    print("=" * 60)
    print("fluorosim_render.py")
    print("=" * 60)
    print(f"Pose file timestamp: {timestamp}")
    print(f"  EE position (m):    {ee_pos}")
    print(f"  EE quat (wxyz):     {ee_quat}")
    print(f"  Phantom pos (m):    {phantom_pos_m}")
    print(f"  C-arm pos (m):      {carm_pos_m}")
    print(f"  C-arm quat (wxyz):  {carm_quat}")
    print(f"  C-arm local (m):    {carm_local_m}  (= carm - phantom)")
    print("  --> fluorosim input:")
    print(f"      rotation (rad):     ({rot_x:.4f}, {rot_y:.4f}, {rot_z:.4f})")
    print(f"      translation (mm):   ({tx_mm:.2f}, {ty_mm:.2f}, {tz_mm:.2f})")

    print("\n[1] Loading μ-volume...")
    volume = load_or_build_synthetic_volume()
    print(f"  {volume}")

    print("\n[2] Initializing simulator...")
    config = SimulatorConfig(
        geometry=CarmGeometry(
            detector_width_px=512,
            detector_height_px=512,
            pixel_spacing_mm=0.5,
            source_to_detector_mm=1020.0,
            source_to_isocenter_mm=510.0,
        ),
        realism=RealismSettings(enabled=False),
        output=OutputSettings(save_to_disk=False),  # we save manually below
    )
    simulator = FluoroSimulator(volume, config)

    # JIT warmup so the timed render reflects steady-state
    _ = simulator.render_frame(rotation=(0.0, 0.0, 0.0), translation=(0.0, 0.0, 0.0))

    print("\n[3] Rendering frame at pose from pose.json...")
    frame = simulator.render_frame(
        rotation=(rot_x, rot_y, rot_z),
        translation=(tx_mm, ty_mm, tz_mm),
    )

    img = frame.image
    print(f"  Image shape: {img.shape}, dtype: {img.dtype}")
    print(f"  Image range: [{float(img.min()):.4f}, {float(img.max()):.4f}], mean: {float(img.mean()):.4f}")

    print("\n[4] Saving outputs...")
    frame.save(str(DRR_PNG))
    np.save(str(DRR_NPY), img)

    meta = {
        "source_pose_timestamp": timestamp,
        "ee_pos": ee_pos,
        "ee_quat": ee_quat,
        "phantom_pos_m": phantom_pos_m,
        "phantom_quat_wxyz": phantom_quat,
        "carm_pos_m": carm_pos_m,
        "carm_quat_wxyz": carm_quat,
        "carm_local_m": carm_local_m,
        "fluorosim_rotation_rad": [rot_x, rot_y, rot_z],
        "fluorosim_translation_mm": [tx_mm, ty_mm, tz_mm],
        "image": {
            "shape": list(img.shape),
            "min": float(img.min()),
            "max": float(img.max()),
            "mean": float(img.mean()),
            "std": float(img.std()),
        },
        "geometry": {
            "detector_px": [512, 512],
            "pixel_spacing_mm": 0.5,
            "source_to_detector_mm": 1020.0,
            "source_to_isocenter_mm": 510.0,
        },
    }
    with open(DRR_META, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Wrote {DRR_PNG}")
    print(f"  Wrote {DRR_NPY}")
    print(f"  Wrote {DRR_META}")

    metrics = simulator.get_metrics()
    if metrics.fps > 0:
        print(f"\n  Render FPS (single frame): {metrics.fps:.1f}")

    print("\n=== Done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())

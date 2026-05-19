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

Tool painting:
    The robot end-effector is rendered into the DRR by burning a high-μ sphere
    into the μ-volume at the EE voxel position before rendering. This is a
    first-cut abstraction — the actual Franka hand mesh is not voxelized; we
    just treat the tool as a dense metal blob centered at ee_pos. The painted
    sphere is applied to an in-memory copy of the cached μ-volume so the disk
    cache stays clean.
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

# Real-CT mode — same switch as the registration scripts.
# DICOM_PATH=<container path>  → load μ-volume from cropped (or full) CT cache.
# CT_FULL_VOLUME=1             → use the full-volume cache instead.
DICOM_PATH = os.environ.get("DICOM_PATH")
CT_FULL_VOLUME = os.environ.get("CT_FULL_VOLUME", "0") == "1"
CACHE_DIR_CT = IO_DIR / ("fluorosim_cache_ct_full" if CT_FULL_VOLUME else "fluorosim_cache_ct")

# Tool abstraction: dense sphere centered at the EE position.
# - μ ≈ 0.5 mm⁻¹ is ~28× bone (~0.018) and makes the tool effectively opaque to
#   X-rays (line-integral over a 30 mm chord ≈ 15 nepers → exp(−15) ≈ 0).
# - 15 mm radius gives a ~30 mm-diameter blob, projecting to ~120 px at the
#   detector under the default geometry (SDD/SID = 2, pixel_spacing = 0.5 mm).
TOOL_RADIUS_MM: float = 15.0
TOOL_MU_PER_MM: float = 0.5


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


def paint_tool_into_volume(
    volume: PreprocessedVolume,
    ee_pos_world: list[float],
    phantom_pos_world: list[float],
    tool_radius_mm: float = TOOL_RADIUS_MM,
    tool_mu: float = TOOL_MU_PER_MM,
) -> tuple[PreprocessedVolume, dict]:
    """Burn a dense sphere into a copy of the μ-volume at the EE voxel and
    return a new PreprocessedVolume wrapping the modified array.

    The on-disk cache (mu_volume.npy) is untouched — the tool is dynamic per
    render. PreprocessedVolume.mu_volume is a read-only property, so we
    construct a fresh volume rather than mutating the input.

    Returns:
        new_volume   - PreprocessedVolume with tool burned into μ
        info         - dict with diagnostic fields (voxels painted, voxel index,
                       whether the tool sphere was fully inside the volume)
    """
    mu = volume.mu_volume.copy()  # (Z, Y, X)
    spacing_zyx = np.asarray(volume.spacing_zyx_mm, dtype=np.float64)
    shape = np.asarray(mu.shape, dtype=np.int64)

    # World-frame offset of the EE from the phantom isocenter, in mm.
    offset_world_xyz_mm = (
        np.asarray(ee_pos_world, dtype=np.float64)
        - np.asarray(phantom_pos_world, dtype=np.float64)
    ) * 1000.0

    # Axis remap: world (X, Y, Z) -> volume-local (X, Y, Z); array axes are (Z, Y, X).
    offset_zyx_mm = offset_world_xyz_mm[[2, 1, 0]]

    # Fractional voxel coordinate of the EE.
    volume_center_voxel = (shape - 1) / 2.0
    ee_voxel = volume_center_voxel + offset_zyx_mm / spacing_zyx

    # Per-axis radius in voxels (anisotropic).
    radius_voxels = tool_radius_mm / spacing_zyx
    bb_min = np.maximum(np.zeros(3, dtype=np.int64),
                        np.floor(ee_voxel - radius_voxels).astype(np.int64))
    bb_max = np.minimum(shape,
                        (np.ceil(ee_voxel + radius_voxels) + 1).astype(np.int64))

    info: dict = {
        "ee_voxel_zyx": ee_voxel.tolist(),
        "tool_radius_mm": float(tool_radius_mm),
        "tool_mu_per_mm": float(tool_mu),
        "voxels_painted": 0,
        "fully_inside_volume": bool(
            np.all(ee_voxel - radius_voxels >= 0)
            and np.all(ee_voxel + radius_voxels <= shape - 1)
        ),
    }

    if np.any(bb_max <= bb_min):
        print(f"  WARNING: tool sphere at EE voxel "
              f"({ee_voxel[0]:.1f}, {ee_voxel[1]:.1f}, {ee_voxel[2]:.1f}) "
              f"does not overlap the volume {tuple(shape)}; no voxels painted.")
        return PreprocessedVolume(mu, volume._metadata), info

    z = np.arange(bb_min[0], bb_max[0])
    y = np.arange(bb_min[1], bb_max[1])
    x = np.arange(bb_min[2], bb_max[2])
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    dist_sq_mm = (
        ((zz - ee_voxel[0]) * spacing_zyx[0]) ** 2
        + ((yy - ee_voxel[1]) * spacing_zyx[1]) ** 2
        + ((xx - ee_voxel[2]) * spacing_zyx[2]) ** 2
    )
    mask = dist_sq_mm <= tool_radius_mm ** 2

    sub = mu[bb_min[0]:bb_max[0], bb_min[1]:bb_max[1], bb_min[2]:bb_max[2]]
    sub[mask] = tool_mu

    info["voxels_painted"] = int(mask.sum())
    print(f"  Painted {info['voxels_painted']} voxels with μ={tool_mu:.3f} mm⁻¹ at "
          f"EE voxel ({ee_voxel[0]:.1f}, {ee_voxel[1]:.1f}, {ee_voxel[2]:.1f}); "
          f"radius {tool_radius_mm:.1f} mm.")
    if not info["fully_inside_volume"]:
        print(f"  NOTE: tool sphere partially extends past the volume bounds "
              f"{tuple(shape)} — only the in-volume portion is painted.")

    return PreprocessedVolume(mu, volume._metadata), info


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
    if DICOM_PATH:
        from ct_loader import load_or_build_ct_volume
        mode = "full" if CT_FULL_VOLUME else "cropped ROI"
        print(f"  Source: real CT ({mode}) at DICOM_PATH={DICOM_PATH}")
        volume = load_or_build_ct_volume(Path(DICOM_PATH), CACHE_DIR_CT,
                                         full_volume=CT_FULL_VOLUME)
    else:
        print("  Source: synthetic ellipsoid phantom")
        volume = load_or_build_synthetic_volume()
    print(f"  {volume}")

    print("\n[1b] Painting robot tool into μ-volume (in-memory copy)...")
    volume, tool_info = paint_tool_into_volume(volume, ee_pos, phantom_pos_m)

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
        "tool": tool_info,
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

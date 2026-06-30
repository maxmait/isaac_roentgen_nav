#!/usr/bin/env python3
"""Multi-view phantom pose registration (Phase 5, 6-DOF: translation + rotation).

Clinical workflow: a mobile C-arm takes ONE shot at a time. A surgeon typically:

    1. Takes an AP (anterior-posterior) shot.
    2. Rotates the C-arm ~90° around the patient's long axis.
    3. Takes a lateral shot.
    4. Uses BOTH images together to triangulate the surgical target in 3D.

This script models that exactly. The phantom (patient) doesn't move between
shots — same world-frame phantom_pos/phantom_rot for both — only the C-arm
rotation changes. So fluorosim's translation and phantom_rot are shared across
views (the unknowns we're recovering), while the C-arm gantry angle differs
per view.

6-DOF parameterization
-----------------------
The optimizer maintains two tensors, both shared across views:

  translation  (3,) mm  — world-frame displacement (carm_pos - phantom_pos)*1000
  phantom_rot  (3,) rad — phantom orientation in world frame (ZXY Euler)

For each view i (C-arm at gantry angle ry_i), the renderer receives:

  R_phantom  = euler_zxy_to_matrix(phantom_rot)
  t_eff      = R_phantom.T @ translation        # world → phantom-local frame, mm
  R_eff      = R_phantom.T @ R_gantry_i          # world → phantom-local frame
  euler_eff  = matrix_to_euler_zxy(R_eff)
  rendered   = drr_module(euler_eff, t_eff)

Both translation and phantom_rot are sign-flip-corrected (fluorosim Slang
autodiff returns negated gradients for all parameters; negate before Adam step).

Loss = sum of MSE losses across views.

Compared to the single-view register_phantom.py:
    Single view (AP only)  →  ~0.4 mm error on Z (X-ray beam axis, depth)
    AP + Lateral           →  Z is in-plane in the lateral view; all three
                              axes converge to in-plane accuracy (~0.05 mm)

Reads pose.json from Isaac Sim if present and reports the world-frame
phantom_pos / phantom_rot recovery against the Isaac Sim ground truth.

Runs inside the fluorosim-torch Docker container.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from fluorosim import PreprocessedVolume, VolumePreprocessor
from fluorosim.rendering.diffdrr_slang_renderer import (
    SlangDiffDRRConfig,
    SlangDiffDRRRenderer,
    create_slang_diffdrr_optimizer,
)

IO_DIR = Path("/workspace/io")
CACHE_DIR = IO_DIR / "fluorosim_cache"
OUT_DIR = IO_DIR / "registration_multiview"
POSE_FILE = IO_DIR / "pose.json"
TOOL_STAMP_NPY  = IO_DIR / "tool_stamp.npy"
TOOL_STAMP_META = IO_DIR / "tool_stamp.json"

DICOM_PATH = os.environ.get("DICOM_PATH")
CT_FULL_VOLUME = os.environ.get("CT_FULL_VOLUME", "0") == "1"
CACHE_DIR_CT = IO_DIR / ("fluorosim_cache_ct_full" if CT_FULL_VOLUME else "fluorosim_cache_ct")


def _envf(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _envv(name: str, default: tuple[float, float, float]) -> tuple[float, float, float]:
    raw = os.environ.get(name)
    if not raw:
        return default
    return tuple(float(v) for v in raw.split(","))  # type: ignore[return-value]


def _env_views_deg(default: tuple[float, ...]) -> tuple[float, ...]:
    raw = os.environ.get("VIEWS_DEG_Y")
    if not raw:
        return default
    return tuple(float(v) for v in raw.split(","))


def _env_floats(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
    raw = os.environ.get(name)
    if not raw:
        return default
    return tuple(float(v) for v in raw.split(","))


# Default to 3 views (one oblique).  Two orthogonal views (0,90) leave the
# in-plane translation↔rotation (tx↔ry) ambiguity unbroken in 6-DOF, so a
# truly blind start can settle in a ~2mm/6° local minimum.  The oblique 45°
# view disambiguates it — verified: blind 28mm start → 0.003mm/0.000°.
VIEWS_DEG_Y: tuple[float, ...] = _env_views_deg((0.0, 45.0, 90.0))
USE_CARM_ROTATION = bool(int(os.environ.get("USE_CARM_ROTATION", "0")))

# Tool-in-target-DRR (clinical realism, Phase 5j follow-up).
# Paints a dense sphere into the μ-volume at the EE voxel to simulate a
# metallic surgical tool occluding the anatomy.  Tool pixels are masked
# out of the registration loss (the optimiser sees only anatomy), so the
# tool is purely visual occlusion — clinically realistic but does not
# bias the recovered pose.  μ ≈ 0.3 mm⁻¹ is approximate stainless-steel
# linear attenuation at ~60 keV; r = 15 mm gives a ~30 mm tool tip blob.
TOOL_RADIUS_MM   = _envf("TOOL_RADIUS_MM",   15.0)
TOOL_MU_PER_MM   = _envf("TOOL_MU_PER_MM",   0.3)
TOOL_MASK_THRESH = _envf("TOOL_MASK_THRESH", 0.5)  # occlusion cutoff (0=any, 1=full)
TOOL_IN_TARGET   = bool(int(os.environ.get("TOOL_IN_TARGET", "1")))

INIT_OFFSET_MM  = _envv("INIT_OFFSET_MM",  (15.0, -10.0, 8.0))
INIT_ROT_DEG    = _envv("INIT_ROT_DEG",    (5.0, 0.0, 3.0))
LR_MM           = _envf("LR_MM",     1.0)
LR_ROT_RAD      = _envf("LR_ROT_RAD", 0.005)
# Max L2 norm of phantom_rot gradient before optimizer step.  Prevents
# gimbal-lock instability when the phantom is nearly symmetric (zero
# rotation gradient) and floating-point noise accumulates in Adam state.
ROT_GRAD_CLIP   = _envf("ROT_GRAD_CLIP", 0.1)
N_ITERS         = int(_envf("N_ITERS",   100))
LOG_EVERY       = int(_envf("LOG_EVERY",   5))
USE_POSE_JSON   = bool(int(os.environ.get("USE_POSE_JSON", "1")))

# ─── Realistic image degradation (Step 1: break the "inverse crime") ──────────
# A real fluoroscope records quantum (Poisson) photon noise + detector blur
# (finite focal spot / pixel MTF) + a slowly-varying scatter pedestal.  We
# perturb ONLY the TARGET images (what the "camera" sees); the optimiser keeps
# its clean renderer, so the target is no longer exactly reproducible by the
# forward model.  The registration then solves a realistic, ill-posed problem
# instead of a self-consistent one (where target and optimiser share the same
# renderer → unrealistically µm-level "accuracy").
#
# This is a PHENOMENOLOGICAL degradation applied in the optimiser's normalised
# [0,1] display space (where the MSE loss lives), NOT a full photon Monte-Carlo
# — enough to break the inverse crime and stress-test capture range / accuracy
# under noise.  Off by default so the clean-case results stay reproducible.
DRR_NOISE         = bool(int(os.environ.get("DRR_NOISE", "0")))
DRR_BLUR_SIGMA_PX = _envf("DRR_BLUR_SIGMA_PX", 0.7)    # detector PSF (Gaussian σ, px)
DRR_PHOTON_COUNT  = _envf("DRR_PHOTON_COUNT", 1.0e4)   # photons/px; lower = noisier
DRR_SCATTER_FRAC  = _envf("DRR_SCATTER_FRAC", 0.0)     # low-freq additive scatter
DRR_NOISE_SEED    = int(_envf("DRR_NOISE_SEED", -1))   # >=0 → reproducible target

# ─── Capture-range / basin-of-attraction study (Step 2) ──────────────────────
# When CAPTURE_RANGE=1 the script does NOT do a single registration; instead it
# sweeps controlled initial offsets from the (known) GT pose and records, per
# offset radius, whether the optimiser converges.  This deliberately USES GT to
# place inits at known distances — it measures the optimiser's basin of
# attraction, not a deployment scenario.  Output: output/capture_range.json.
CAPTURE_RANGE     = bool(int(os.environ.get("CAPTURE_RANGE", "0")))
CR_TRANS_RADII_MM = _env_floats("CR_TRANS_RADII_MM", (5., 10., 20., 30., 40., 60.))
CR_N_SAMPLES      = int(_envf("CR_N_SAMPLES", 8))      # random directions per radius
CR_ROT_OFFSET_DEG = _envf("CR_ROT_OFFSET_DEG", 5.0)   # rot perturbation per sample
CR_SUCCESS_MM     = _envf("CR_SUCCESS_MM", 1.0)        # converged if ‖t_err‖ < this
CR_SUCCESS_DEG    = _envf("CR_SUCCESS_DEG", 1.0)       # and geodesic rot err < this
CR_SEED           = int(_envf("CR_SEED", 0))


# ─── ZXY Euler helpers (differentiable) ──────────────────────────────────────

def euler_zxy_to_matrix(e: torch.Tensor) -> torch.Tensor:
    """ZXY Euler (rx, ry, rz) radians → 3×3 rotation matrix.

    Matches the convention in fluorosim's Slang shader:
        R = Rz(rz) @ Rx(rx) @ Ry(ry)
    """
    cx, sx = torch.cos(e[0]), torch.sin(e[0])
    cy, sy = torch.cos(e[1]), torch.sin(e[1])
    cz, sz = torch.cos(e[2]), torch.sin(e[2])
    return torch.stack([
        cz * cy - sz * sx * sy,  -sz * cx,  cz * sy + sz * sx * cy,
        sz * cy + cz * sx * sy,   cz * cx,  sz * sy - cz * sx * cy,
        -cx * sy,                  sx,        cx * cy,
    ]).reshape(3, 3)


def matrix_to_euler_zxy(R: torch.Tensor) -> torch.Tensor:
    """3×3 rotation matrix → ZXY Euler (rx, ry, rz) radians.

    Uses atan2(sin_rx, cos_rx) rather than arcsin(sin_rx) to avoid the
    infinite gradient of arcsin at ±90° (gimbal lock).  Valid everywhere
    except pure ±90° rx where ry and rz are degenerate (sum-to-a-single
    angle), which doesn't occur for small clinical phantom rotations.
    """
    cos_rx = torch.sqrt(R[2, 0] ** 2 + R[2, 2] ** 2 + 1e-12)
    rx = torch.arctan2(R[2, 1], cos_rx)
    ry = torch.arctan2(-R[2, 0], R[2, 2])
    rz = torch.arctan2(-R[0, 1], R[1, 1])
    return torch.stack([rx, ry, rz])


def rotation_matrix_to_quat_wxyz(R: np.ndarray) -> list[float]:
    """3×3 numpy rotation matrix → unit quaternion [w, x, y, z] (Shepperd)."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    return [float(w), float(x), float(y), float(z)]


def quat_wxyz_to_euler_zxy(w: float, x: float, y: float, z: float) -> np.ndarray:
    """Unit quaternion (wxyz) → ZXY Euler (rx, ry, rz) radians."""
    n = w*w + x*x + y*y + z*z
    s = 2.0 / n if n > 0 else 0.0
    R = np.array([
        [1 - s*(y*y + z*z),     s*(x*y - z*w),     s*(x*z + y*w)],
        [    s*(x*y + z*w), 1 - s*(x*x + z*z),     s*(y*z - x*w)],
        [    s*(x*z - y*w),     s*(y*z + x*w), 1 - s*(x*x + y*y)],
    ])
    rx = np.arcsin(np.clip(R[2, 1], -1.0, 1.0))
    ry = np.arctan2(-R[2, 0], R[2, 2])
    rz = np.arctan2(-R[0, 1], R[1, 1])
    return np.array([rx, ry, rz], dtype=np.float32)


# ─── tool painting helpers ───────────────────────────────────────────────────

def ee_voxel_zyx_index(
    ee_pos_world_m,
    phantom_pos_world_m,
    R_phantom_world: np.ndarray,    # 3×3 world-frame rotation
    spacing_zyx_mm,
    volume_shape_zyx,
) -> np.ndarray:
    """Fractional (z, y, x) voxel index of the EE in the phantom volume frame.

    Accounts for non-identity phantom rotation:
        offset_world  = (EE − phantom) * 1000          mm in world XYZ
        offset_local  = R_phantom.T @ offset_world      mm in phantom-local XYZ
        offset_zyx    = permute(x,y,z → z,y,x)          axis order for volume
    """
    offset_world_xyz_mm = (
        np.asarray(ee_pos_world_m, dtype=np.float64)
        - np.asarray(phantom_pos_world_m, dtype=np.float64)
    ) * 1000.0
    # Matches the rendering math: t_eff = R_phantom.T @ t_world.
    offset_local_xyz_mm = R_phantom_world.T @ offset_world_xyz_mm
    offset_local_zyx_mm = offset_local_xyz_mm[[2, 1, 0]]
    center_voxel = (np.asarray(volume_shape_zyx, dtype=np.float64) - 1.0) / 2.0
    return center_voxel + offset_local_zyx_mm / np.asarray(spacing_zyx_mm, dtype=np.float64)


def quat_wxyz_to_matrix(q) -> np.ndarray:
    """Unit quaternion (w, x, y, z) → 3×3 column-vector rotation matrix.

    Matches the convention used elsewhere in this file (euler_zxy_to_matrix
    returns a column-vector R such that v_world_col = R @ v_local_col).
    """
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    s = 2.0 / n if n > 0 else 0.0
    return np.array([
        [1 - s*(y*y + z*z),     s*(x*y - z*w),     s*(x*z + y*w)],
        [    s*(x*y + z*w), 1 - s*(x*x + z*z),     s*(y*z - x*w)],
        [    s*(x*z - y*w),     s*(y*z + x*w), 1 - s*(x*x + y*y)],
    ], dtype=np.float64)


def load_tool_stamp() -> dict | None:
    """Load the EE-local tool stamp built by bridge/build_tool_stamp.py.

    Returns None when the stamp files are absent (the pipeline then falls
    back to the sphere painter).
    """
    if not TOOL_STAMP_NPY.exists() or not TOOL_STAMP_META.exists():
        return None
    stamp = np.load(TOOL_STAMP_NPY).astype(np.float32)
    with open(TOOL_STAMP_META) as f:
        meta = json.load(f)
    return {
        "stamp":           stamp,                      # (nz, ny, nx)
        "spacing_zyx_mm":  np.asarray(meta["spacing_zyx_mm"], dtype=np.float64),
        "origin_xyz_mm":   np.asarray(meta["origin_ee_local_xyz_mm"], dtype=np.float64),
        "tool_mu_per_mm":  float(meta["tool_mu_per_mm"]),
        "meta":            meta,
    }


def paint_stamp_into_mu(
    mu: np.ndarray,                # phantom μ-volume (Z, Y, X)
    spacing_zyx_mm,
    ee_pos_world_m,                # (3,) world frame, metres
    ee_quat_wxyz,                  # (4,)
    phantom_pos_world_m,
    R_phantom_world: np.ndarray,   # 3×3 column-vector form
    stamp_info: dict,
) -> tuple[np.ndarray, int]:
    """Splat the EE-local tool stamp into a COPY of mu via affine resampling.

    The transform (ph_idx → st_idx) is fully linear:
        v_ph_zyx_mm = (ph_idx - ph_center) * ph_spacing_zyx_mm
        v_ph_xyz_mm = P @ v_ph_zyx_mm                       # permute zyx→xyz
        v_world_mm  = R_phantom_world @ v_ph_xyz_mm + phantom_pos_mm
        v_ee_xyz    = R_ee_world.T @ (v_world_mm - ee_pos_mm)
        v_ee_zyx    = P @ v_ee_xyz                          # permute xyz→zyx
        st_idx      = (v_ee_zyx - stamp_origin_zyx_mm) / stamp_spacing_zyx
                      − 0.5                                  # voxel-corner→idx
    scipy.ndimage.affine_transform samples
        output[ph_idx] = stamp[ M @ ph_idx + b ]
    so we just expand the chain into (M, b).  An AABB of the stamp is
    computed in phantom voxel space so we resample only the relevant
    subvolume — much faster than touching the whole μ-volume.

    Tool μ is `max`-combined with anatomy μ (the steel tool occludes any
    bone/tissue it overlaps).  Returns (mu_modified, n_voxels_painted).
    """
    from scipy.ndimage import affine_transform

    stamp           = stamp_info["stamp"]
    stamp_origin    = stamp_info["origin_xyz_mm"]               # (3,) EE-local xyz mm
    stamp_spacing   = stamp_info["spacing_zyx_mm"]              # (3,) zyx mm
    R_ee_world      = quat_wxyz_to_matrix(ee_quat_wxyz)         # 3×3 col-vec

    ph_shape   = np.asarray(mu.shape, dtype=np.float64)
    ph_center  = (ph_shape - 1.0) / 2.0
    ph_spacing = np.asarray(spacing_zyx_mm, dtype=np.float64)

    phantom_pos_mm = np.asarray(phantom_pos_world_m, dtype=np.float64) * 1000.0
    ee_pos_mm      = np.asarray(ee_pos_world_m, dtype=np.float64) * 1000.0
    P = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.float64)
    stamp_origin_zyx = P @ stamp_origin                           # EE-local zyx mm

    A         = np.diag(ph_spacing)
    D_st_inv  = np.diag(1.0 / stamp_spacing)
    M         = D_st_inv @ P @ R_ee_world.T @ R_phantom_world @ P @ A
    v_ee_zyx_at_phc = P @ (R_ee_world.T @ (phantom_pos_mm - ee_pos_mm))
    st_idx_at_phc   = D_st_inv @ (v_ee_zyx_at_phc - stamp_origin_zyx) - 0.5
    b = st_idx_at_phc - M @ ph_center

    # AABB of the stamp's 8 corners in phantom voxel space (xyz → zyx permutes done explicitly)
    stamp_shape_zyx  = np.asarray(stamp.shape, dtype=np.float64)
    stamp_extent_zyx = (stamp_shape_zyx - 1.0) * stamp_spacing
    stamp_extent_xyz = stamp_extent_zyx[[2, 1, 0]]
    corners_xyz = np.array(
        [stamp_origin + np.array([cx, cy, cz])
         for cz in (0.0, stamp_extent_xyz[2])
         for cy in (0.0, stamp_extent_xyz[1])
         for cx in (0.0, stamp_extent_xyz[0])],
        dtype=np.float64,
    )                                                             # (8, 3) EE-local xyz mm
    corners_world  = (R_ee_world @ corners_xyz.T).T + ee_pos_mm
    corners_ph_xyz = (R_phantom_world.T @ (corners_world - phantom_pos_mm).T).T
    corners_ph_zyx = corners_ph_xyz[:, [2, 1, 0]]
    corners_idx    = corners_ph_zyx / ph_spacing + ph_center

    aabb_min = np.maximum(np.floor(corners_idx.min(axis=0)).astype(int), 0)
    aabb_max = np.minimum(np.ceil(corners_idx.max(axis=0)).astype(int) + 1,
                          np.asarray(mu.shape, dtype=int))
    if np.any(aabb_max <= aabb_min):
        return mu.copy(), 0

    aabb_shape  = (aabb_max - aabb_min).astype(int)
    sub_offset  = M @ aabb_min.astype(np.float64) + b
    sub_stamp   = affine_transform(
        stamp.astype(np.float32),
        M, offset=sub_offset,
        output_shape=tuple(int(s) for s in aabb_shape),
        order=1, mode="constant", cval=0.0, prefilter=False,
    )

    mu_out = mu.copy()
    sub = mu_out[aabb_min[0]:aabb_max[0],
                  aabb_min[1]:aabb_max[1],
                  aabb_min[2]:aabb_max[2]]
    np.maximum(sub, sub_stamp, out=sub)
    n_painted = int((sub_stamp > 1e-6).sum())
    return mu_out, n_painted


def paint_sphere_into_mu(
    mu: np.ndarray,
    ee_voxel: np.ndarray,
    spacing_zyx_mm,
    radius_mm: float,
    tool_mu: float,
) -> tuple[np.ndarray, int, bool]:
    """Burn an isotropic sphere of high μ into a COPY of `mu`.

    Tool voxels are SET to tool_mu (not added) — a metal tool literally
    replaces the surrounding voxel content. Returns (mu_modified, n_painted,
    fully_inside).
    """
    mu_out = mu.copy()
    spacing = np.asarray(spacing_zyx_mm, dtype=np.float64)
    shape = np.asarray(mu.shape, dtype=np.int64)
    radius_vox = radius_mm / spacing
    fully_inside = bool(
        np.all(ee_voxel - radius_vox >= 0)
        and np.all(ee_voxel + radius_vox <= shape - 1)
    )
    bb_min = np.maximum(np.zeros(3, dtype=np.int64),
                        np.floor(ee_voxel - radius_vox).astype(np.int64))
    bb_max = np.minimum(shape,
                        (np.ceil(ee_voxel + radius_vox) + 1).astype(np.int64))
    if np.any(bb_max <= bb_min):
        return mu_out, 0, fully_inside

    z = np.arange(bb_min[0], bb_max[0])
    y = np.arange(bb_min[1], bb_max[1])
    x = np.arange(bb_min[2], bb_max[2])
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    dist_sq_mm = (
        ((zz - ee_voxel[0]) * spacing[0]) ** 2
        + ((yy - ee_voxel[1]) * spacing[1]) ** 2
        + ((xx - ee_voxel[2]) * spacing[2]) ** 2
    )
    mask = dist_sq_mm <= radius_mm ** 2
    sub = mu_out[bb_min[0]:bb_max[0], bb_min[1]:bb_max[1], bb_min[2]:bb_max[2]]
    sub[mask] = tool_mu
    return mu_out, int(mask.sum()), fully_inside


def geodesic_angle_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    """Geodesic rotation distance between two rotation matrices, in degrees."""
    R_rel = R1.T @ R2
    cos_angle = (np.trace(R_rel) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


# ─── data loading ─────────────────────────────────────────────────────────────

def view_name(angle_deg: float) -> str:
    if abs(angle_deg) < 1e-3:
        return "AP"
    if abs(angle_deg - 90.0) < 1e-3:
        return "lateral"
    if abs(angle_deg + 90.0) < 1e-3:
        return "lateral_neg"
    return f"ry{angle_deg:+.0f}deg"


def load_isaac_ground_truth() -> dict | None:
    if not USE_POSE_JSON or not POSE_FILE.exists():
        return None
    with open(POSE_FILE) as f:
        pose = json.load(f)
    phantom_pos = np.asarray(pose.get("phantom_pos", [0.0, 0.0, 0.0]), dtype=np.float64)
    carm_pos    = np.asarray(pose.get("carm_pos",    [0.0, 0.0, 0.0]), dtype=np.float64)
    phantom_quat = pose.get("phantom_quat", [1.0, 0.0, 0.0, 0.0])
    gt_translation_mm = (carm_pos - phantom_pos) * 1000.0
    gt_phantom_rot_euler = quat_wxyz_to_euler_zxy(*phantom_quat)
    ee_pos = pose.get("ee_pos")    # for tool painting; None in synthetic mode
    ee_quat = pose.get("ee_quat")   # for oriented tool stamp
    return {
        "phantom_pos_world_m":    phantom_pos.tolist(),
        "carm_pos_world_m":       carm_pos.tolist(),
        "gt_translation_mm":      gt_translation_mm.tolist(),
        "gt_phantom_rot_euler":   gt_phantom_rot_euler.tolist(),
        "phantom_quat_wxyz":      phantom_quat,
        "ee_pos_world_m":         ee_pos,
        "ee_quat_wxyz":           ee_quat,
    }


def translation_mm_to_phantom_world_m(
    translation_mm: np.ndarray, carm_pos_world_m: np.ndarray
) -> np.ndarray:
    return carm_pos_world_m - np.asarray(translation_mm) / 1000.0


def load_or_build_synthetic_volume() -> PreprocessedVolume:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if (CACHE_DIR / "mu_volume.npy").exists():
        return PreprocessedVolume.load(CACHE_DIR)
    shape = (128, 256, 256)
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    center = np.array(shape) / 2.0
    dist = np.sqrt(
        (z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2
    )
    hu = np.full(shape, -900.0, dtype=np.float32)
    hu[dist < 60] = 40.0
    hu[dist < 40] = 800.0
    return VolumePreprocessor.from_numpy(
        hu, spacing_zyx_mm=(1.0, 0.5, 0.5)
    ).preprocess(output_dir=CACHE_DIR)


def load_volume() -> PreprocessedVolume:
    if DICOM_PATH:
        from ct_loader import load_or_build_ct_volume
        mode = "full" if CT_FULL_VOLUME else "cropped ROI"
        print(f"  Source: real CT ({mode}) at DICOM_PATH={DICOM_PATH}")
        return load_or_build_ct_volume(Path(DICOM_PATH), CACHE_DIR_CT, full_volume=CT_FULL_VOLUME)
    print("  Source: synthetic ellipsoid phantom")
    return load_or_build_synthetic_volume()


def degrade_target_drrs(targets_np: list[np.ndarray]) -> list[np.ndarray]:
    """Apply phenomenological fluoroscopy degradation to target images.

    Operates in the optimiser's normalised [0,1] image space (bone bright after
    invert).  Pipeline per view: detector blur (Gaussian PSF) → quantum
    (Poisson) photon noise → optional low-frequency scatter pedestal → clip.

    The Poisson model treats the [0,1] signal as a normalised photon count
    N = photon_count · signal, so the per-pixel std ≈ sqrt(signal/photon_count)
    is signal-dependent — brighter (more-attenuating) pixels are noisier, as in
    a quantum-limited detector.  Returns NEW arrays; inputs untouched.
    """
    from scipy.ndimage import gaussian_filter

    seed = DRR_NOISE_SEED if DRR_NOISE_SEED >= 0 else None
    rng = np.random.default_rng(seed)
    out: list[np.ndarray] = []
    for img in targets_np:
        x = img.astype(np.float32)
        if DRR_BLUR_SIGMA_PX > 0:
            x = gaussian_filter(x, sigma=DRR_BLUR_SIGMA_PX)
        if DRR_PHOTON_COUNT > 0:
            lam = np.clip(x, 0.0, None) * DRR_PHOTON_COUNT
            x = rng.poisson(lam).astype(np.float32) / DRR_PHOTON_COUNT
        if DRR_SCATTER_FRAC > 0:
            ped = gaussian_filter(
                rng.standard_normal(x.shape).astype(np.float32),
                sigma=max(x.shape) / 8.0,
            )
            ped = ped / (np.abs(ped).max() + 1e-8) * DRR_SCATTER_FRAC
            x = x + ped
        out.append(np.clip(x, 0.0, 1.0).astype(np.float32))
    return out


def run_optimization(
    drr_module,
    translation: "torch.Tensor",
    phantom_rot: "torch.Tensor",
    optimizer,
    rot_gantry_tensors: list,
    targets: list,
    weights: list,
    weight_norms: list,
    gt_translation_mm,
    gt_phantom_rot_euler,
    R_gt_np: np.ndarray,
    n_iters: int,
    view_names: list[str] | None = None,
    log_every: int = 0,
) -> tuple[list[dict], float]:
    """Run the 6-DOF mask-weighted-MSE optimisation loop in place.

    Mutates ``translation`` / ``phantom_rot`` (and the optimiser state).
    Returns (trace, elapsed_seconds).  Factored out of main() so the
    capture-range sweep — and, in future, a real-time tracking loop — can
    reuse it with the SAME (already-loaded) drr_module / targets, paying the
    volume + renderer setup cost only once.
    """
    trace: list[dict] = []
    t_start = time.time()
    for it in range(n_iters):
        optimizer.zero_grad()
        total_loss = torch.zeros((), dtype=torch.float32, device=translation.device)
        per_view_loss: list[float] = []

        R_ph = euler_zxy_to_matrix(phantom_rot)
        t_world = translation

        for rot_gantry, target, weight, w_norm in zip(
            rot_gantry_tensors, targets, weights, weight_norms
        ):
            R_eff = R_ph.T @ euler_zxy_to_matrix(rot_gantry)
            euler_eff = matrix_to_euler_zxy(R_eff)
            t_eff = R_ph.T @ t_world
            rendered = drr_module(euler_eff, t_eff)
            loss_v = (((rendered - target) ** 2) * weight).sum() / w_norm
            total_loss = total_loss + loss_v
            per_view_loss.append(float(loss_v.item()))

        total_loss.backward()
        # fluorosim Slang autodiff sign-flip + NaN guard (see CLAUDE.md).
        for p in [translation, phantom_rot]:
            if p.grad is None:
                continue
            if torch.isnan(p.grad).any():
                p.grad.zero_()
            else:
                p.grad.neg_()
        if phantom_rot.grad is not None and phantom_rot.grad.abs().max() > 0:
            torch.nn.utils.clip_grad_norm_([phantom_rot], max_norm=ROT_GRAD_CLIP)
        optimizer.step()

        t_np = translation.detach().cpu().numpy()
        pr_np = phantom_rot.detach().cpu().numpy()
        trans_err = t_np - np.asarray(gt_translation_mm)
        rot_err_np = pr_np - np.asarray(gt_phantom_rot_euler, dtype=np.float32)
        with torch.no_grad():
            R_rec_np = euler_zxy_to_matrix(
                torch.tensor(pr_np, dtype=torch.float32)
            ).numpy().astype(np.float64)
        rot_geo_deg = geodesic_angle_deg(R_gt_np, R_rec_np)

        trace.append({
            "iter": it,
            "loss_total": float(total_loss.item()),
            "loss_per_view": per_view_loss,
            "translation_mm": t_np.tolist(),
            "err_mm": trans_err.tolist(),
            "err_norm_mm": float(np.linalg.norm(trans_err)),
            "phantom_rot_euler_rad": pr_np.tolist(),
            "rot_err_euler_deg": [math.degrees(v) for v in rot_err_np.tolist()],
            "rot_err_geodesic_deg": rot_geo_deg,
        })
        if log_every and (it % log_every == 0 or it == n_iters - 1):
            if view_names is not None:
                pv = "  ".join(
                    f"{view_names[i]}={per_view_loss[i]:.2e}"
                    for i in range(len(view_names))
                )
            else:
                pv = ""
            print(f"  iter {it:3d}: total={total_loss.item():.6e}  {pv}  "
                  f"||t_err||={trace[-1]['err_norm_mm']:.3f} mm  "
                  f"||r_err||={rot_geo_deg:.3f}°")
    return trace, time.time() - t_start


def run_capture_range(
    drr_module,
    rot_gantry_tensors: list,
    targets: list,
    weights: list,
    weight_norms: list,
    gt_translation_mm,
    gt_phantom_rot_euler,
    R_gt_np: np.ndarray,
    device,
) -> dict:
    """Sweep controlled init offsets from GT and record convergence per radius.

    Returns a results dict (also serialised to output/capture_range.json by
    the caller).  Uses run_optimization() so the renderer + targets are reused
    across all samples — only the init tensors + optimiser are rebuilt.
    """
    rng = np.random.default_rng(CR_SEED)
    gt_t = np.asarray(gt_translation_mm, dtype=np.float32)
    gt_r = np.asarray(gt_phantom_rot_euler, dtype=np.float32)
    samples: list[dict] = []

    print(f"\n[CR] Capture-range sweep: radii (mm) = {list(CR_TRANS_RADII_MM)}, "
          f"{CR_N_SAMPLES} samples/radius, rot offset = {CR_ROT_OFFSET_DEG}°, "
          f"{N_ITERS} iters/sample")
    print(f"[CR] Success := ‖t_err‖ < {CR_SUCCESS_MM} mm AND geodesic rot err "
          f"< {CR_SUCCESS_DEG}°")

    for radius in CR_TRANS_RADII_MM:
        for s in range(CR_N_SAMPLES):
            # Random unit direction (translation) and random small rotation axis.
            u = rng.standard_normal(3).astype(np.float32)
            u /= (np.linalg.norm(u) + 1e-8)
            init_trans = gt_t + radius * u
            rax = rng.standard_normal(3).astype(np.float32)
            rax /= (np.linalg.norm(rax) + 1e-8)
            init_rot = gt_r + np.radians(CR_ROT_OFFSET_DEG) * rax

            translation = torch.tensor(init_trans, dtype=torch.float32,
                                       device=device, requires_grad=True)
            phantom_rot = torch.tensor(init_rot, dtype=torch.float32,
                                       device=device, requires_grad=True)
            optimizer = torch.optim.Adam([
                {"params": [translation], "lr": LR_MM},
                {"params": [phantom_rot], "lr": LR_ROT_RAD},
            ])
            trace, _ = run_optimization(
                drr_module, translation, phantom_rot, optimizer,
                rot_gantry_tensors, targets, weights, weight_norms,
                gt_translation_mm, gt_phantom_rot_euler, R_gt_np,
                N_ITERS, view_names=None, log_every=0,
            )
            final = trace[-1]
            init_err = float(np.linalg.norm(init_trans - gt_t))
            success = (final["err_norm_mm"] < CR_SUCCESS_MM
                       and final["rot_err_geodesic_deg"] < CR_SUCCESS_DEG)
            samples.append({
                "radius_mm": float(radius),
                "sample": s,
                "init_err_norm_mm": init_err,
                "init_rot_err_deg": float(CR_ROT_OFFSET_DEG),
                "final_err_norm_mm": final["err_norm_mm"],
                "final_rot_geodesic_deg": final["rot_err_geodesic_deg"],
                "final_loss": final["loss_total"],
                "success": bool(success),
            })
            print(f"  r={radius:5.1f}mm  s={s}  init‖t‖={init_err:5.1f}mm  "
                  f"-> final ‖t‖={final['err_norm_mm']:7.3f}mm  "
                  f"rot={final['rot_err_geodesic_deg']:6.3f}°  "
                  f"{'OK ' if success else 'FAIL'}")

    per_radius = []
    for radius in CR_TRANS_RADII_MM:
        grp = [s for s in samples if s["radius_mm"] == float(radius)]
        n_ok = sum(s["success"] for s in grp)
        fin = np.array([s["final_err_norm_mm"] for s in grp])
        per_radius.append({
            "radius_mm": float(radius),
            "n": len(grp),
            "n_success": int(n_ok),
            "success_rate": float(n_ok / max(1, len(grp))),
            "median_final_mm": float(np.median(fin)),
            "max_final_mm": float(fin.max()),
        })
        print(f"[CR] r={radius:5.1f}mm: success {n_ok}/{len(grp)} "
              f"({100*n_ok/max(1,len(grp)):.0f}%)  "
              f"median final ‖t‖={np.median(fin):.3f}mm")

    return {
        "config": {
            "trans_radii_mm": list(CR_TRANS_RADII_MM),
            "n_samples": CR_N_SAMPLES,
            "rot_offset_deg": CR_ROT_OFFSET_DEG,
            "n_iters": N_ITERS,
            "success_mm": CR_SUCCESS_MM,
            "success_deg": CR_SUCCESS_DEG,
            "seed": CR_SEED,
            "views_deg_y": list(VIEWS_DEG_Y),
            "drr_noise": DRR_NOISE,
            "lr_mm": LR_MM,
            "lr_rot_rad": LR_ROT_RAD,
        },
        "gt_translation_mm": [float(v) for v in gt_translation_mm],
        "per_radius": per_radius,
        "samples": samples,
    }


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("register_phantom_multiview.py — 6-DOF sequential C-arm shots")
    print("=" * 60)

    isaac = load_isaac_ground_truth()
    if isaac is not None:
        gt_translation_mm    = isaac["gt_translation_mm"]
        gt_phantom_rot_euler = isaac["gt_phantom_rot_euler"]
        print("Ground truth from Isaac Sim pose.json:")
        print(f"  phantom_pos  (world m):  {isaac['phantom_pos_world_m']}")
        print(f"  carm_pos     (world m):  {isaac['carm_pos_world_m']}")
        print(f"  phantom_quat (wxyz):     {isaac['phantom_quat_wxyz']}")
        print(f"  -> fluorosim translation (mm): {gt_translation_mm}")
        print(f"  -> phantom ZXY Euler (deg):    "
              f"{[round(math.degrees(v), 3) for v in gt_phantom_rot_euler]}")
    else:
        gt_translation_mm    = [0.0, 0.0, 0.0]
        gt_phantom_rot_euler = [0.0, 0.0, 0.0]
        print("No pose.json — synthetic mode, GT translation=(0,0,0), GT rot=(0,0,0).")

    # Resolve view angles
    view_angles = VIEWS_DEG_Y
    if USE_CARM_ROTATION and POSE_FILE.exists():
        with open(POSE_FILE) as f:
            _pose = json.load(f)
        if "view_angles_deg" in _pose and len(_pose["view_angles_deg"]) >= 2:
            view_angles = tuple(float(a) for a in _pose["view_angles_deg"])
            print(f"USE_CARM_ROTATION=1: view_angles_deg from pose.json = "
                  f"{[f'{a:+.1f}' for a in view_angles]}°")
        elif "carm_rotation_y_deg" in _pose:
            # Only one shot captured — synthesize an orthogonal pair plus an
            # oblique view (base, +45, +90).  The oblique view breaks the
            # in-plane tx<->ry coupling that stalls 2-view blind 6-DOF; see the
            # "Use >=3 views" note in CLAUDE.md.
            base = float(_pose["carm_rotation_y_deg"])
            view_angles = (base, base + 45.0, base + 90.0)
            print(f"USE_CARM_ROTATION=1: single-shot AP={base:+.1f}° "
                  f"-> views (ry) = {view_angles}")

    views = [
        {"name": view_name(a), "angle_deg": a,
         "rotation_rad": (0.0, math.radians(a), 0.0)}
        for a in view_angles
    ]
    init_rot_rad = np.array(
        [math.radians(d) for d in INIT_ROT_DEG], dtype=np.float32
    )
    print(f"Views (ry angles, deg):        {[v['angle_deg'] for v in views]} "
          f"-> names: {[v['name'] for v in views]}")
    print(f"Initial translation offset (mm): {INIT_OFFSET_MM}")
    print(f"Initial rotation  offset  (deg): {INIT_ROT_DEG}")
    print(f"LR translation (mm/step):        {LR_MM}")
    print(f"LR rotation    (rad/step):       {LR_ROT_RAD}")
    print(f"Iterations:                      {N_ITERS}")

    print("\n[1] Loading μ-volume...")
    volume  = load_volume()
    mu      = volume.mu_volume
    spacing = volume.spacing_zyx_mm
    print(f"  shape={mu.shape}, spacing_mm={spacing}, "
          f"μ range=[{mu.min():.4f}, {mu.max():.4f}]")

    cfg = SlangDiffDRRConfig(
        det_height_px=512, det_width_px=512,
        pixel_spacing_mm=0.5,
        source_to_detector_mm=1020.0,
        source_to_isocenter_mm=510.0,
        step_mm=0.5,
        normalize=True, invert=True,
    )

    # --- Tool painting (Phase 5 follow-up: tool-in-target-DRR) ---------------
    # Burn a dense sphere into the μ-volume at the EE voxel so the TARGET
    # images contain a metallic occlusion — matching what a real fluoroscope
    # would record with the tool inserted. A separate tool-only volume is
    # rendered per view to derive a binary mask; the registration loss is
    # weighted by (1 - mask), so the tool's pixels carry no signal and the
    # optimiser only matches anatomy. The optimiser's own renderer keeps the
    # clean μ-volume (no tool) — clinically correct: the patient's CT does
    # not contain an inserted tool.
    ee_pos_world = isaac.get("ee_pos_world_m") if isaac is not None else None
    ee_quat_world = (isaac.get("ee_quat_wxyz") if isaac is not None else None)
    paint_tool = TOOL_IN_TARGET and ee_pos_world is not None
    tool_stamp_info = load_tool_stamp() if paint_tool else None
    use_stamp = tool_stamp_info is not None and ee_quat_world is not None
    if paint_tool:
        R_phantom_gt_np = euler_zxy_to_matrix(
            torch.tensor(gt_phantom_rot_euler, dtype=torch.float32)
        ).numpy().astype(np.float64)
        ee_voxel_gt = ee_voxel_zyx_index(
            ee_pos_world, isaac["phantom_pos_world_m"], R_phantom_gt_np,
            spacing, mu.shape,
        )
        if use_stamp:
            stamp_shape = tool_stamp_info["stamp"].shape
            print(f"\n[1b] Painting tool MESH STAMP into target volume "
                  f"(stamp shape {stamp_shape}, μ={tool_stamp_info['tool_mu_per_mm']:.3f} mm⁻¹)")
            print(f"  EE world pos (m):      {ee_pos_world}")
            print(f"  EE world quat (wxyz):  {ee_quat_world}")
            print(f"  EE voxel (GT z,y,x):   "
                  f"({ee_voxel_gt[0]:.1f}, {ee_voxel_gt[1]:.1f}, {ee_voxel_gt[2]:.1f})")
            mu_target, n_painted = paint_stamp_into_mu(
                mu, spacing, ee_pos_world, ee_quat_world,
                isaac["phantom_pos_world_m"], R_phantom_gt_np, tool_stamp_info,
            )
            mu_tool_only, _ = paint_stamp_into_mu(
                np.zeros_like(mu), spacing, ee_pos_world, ee_quat_world,
                isaac["phantom_pos_world_m"], R_phantom_gt_np, tool_stamp_info,
            )
            print(f"  Painted {n_painted} voxels  "
                  f"(source mesh: {tool_stamp_info['meta'].get('source_mesh_meta', {}).get('mesh_prim', '?')})")
        else:
            print(f"\n[1b] Painting tool SPHERE into target volume "
                  f"(μ={TOOL_MU_PER_MM:.3f} mm⁻¹, r={TOOL_RADIUS_MM:.1f} mm)")
            print(f"  EE world pos (m):      {ee_pos_world}")
            print(f"  EE voxel (GT z,y,x):   "
                  f"({ee_voxel_gt[0]:.1f}, {ee_voxel_gt[1]:.1f}, {ee_voxel_gt[2]:.1f})")
            mu_target, n_painted, fully_inside = paint_sphere_into_mu(
                mu, ee_voxel_gt, spacing, TOOL_RADIUS_MM, TOOL_MU_PER_MM,
            )
            mu_tool_only, _, _ = paint_sphere_into_mu(
                np.zeros_like(mu), ee_voxel_gt, spacing,
                TOOL_RADIUS_MM, TOOL_MU_PER_MM,
            )
            print(f"  Painted {n_painted} voxels  (fully inside volume: {fully_inside})")
    else:
        if isaac is None:
            print("\n[1b] No pose.json → no EE position → tool painting disabled")
        elif ee_pos_world is None:
            print("\n[1b] pose.json missing 'ee_pos' field → tool painting disabled")
        else:
            print("\n[1b] TOOL_IN_TARGET=0 → tool painting disabled")
        mu_target = mu
        mu_tool_only = None

    # --- Target images (rendered at GT pose, with tool painted in) -----------
    print("\n[2] Simulating OR workflow — taking each shot in sequence...")
    target_renderer = SlangDiffDRRRenderer(mu_target, spacing, cfg)
    _ = target_renderer.render((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))  # JIT warmup

    # Compute GT effective rotation/translation per view (accounts for phantom rotation)
    gt_trans_t = torch.tensor(gt_translation_mm,    dtype=torch.float32)
    gt_rot_t   = torch.tensor(gt_phantom_rot_euler, dtype=torch.float32)
    R_phantom_gt = euler_zxy_to_matrix(gt_rot_t)  # no grad needed here

    targets_np: list[np.ndarray] = []
    per_view_eff_geom: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
    for view in views:
        R_gantry = euler_zxy_to_matrix(
            torch.tensor(view["rotation_rad"], dtype=torch.float32)
        )
        t_eff_gt = tuple((R_phantom_gt.T @ gt_trans_t).tolist())
        r_eff_gt = tuple(matrix_to_euler_zxy(R_phantom_gt.T @ R_gantry).tolist())
        per_view_eff_geom.append((r_eff_gt, t_eff_gt))
        img = target_renderer.render(r_eff_gt, t_eff_gt)
        targets_np.append(img)
        print(f"  [{view['name']:>8}] ry={view['angle_deg']:+.1f}°  "
              f"range=[{img.min():.4f}, {img.max():.4f}]  std={img.std():.4f}")

    # --- Realistic degradation of the TARGET images (Step 1) -----------------
    # Breaks the inverse crime: the optimiser's renderer stays clean, so the
    # noisy/blurred target is no longer exactly reproducible by the forward
    # model.  Off unless DRR_NOISE=1.
    if DRR_NOISE:
        print(f"\n[2a] Degrading target DRRs (blur σ={DRR_BLUR_SIGMA_PX}px, "
              f"photons/px={DRR_PHOTON_COUNT:.0f}, scatter={DRR_SCATTER_FRAC}, "
              f"seed={DRR_NOISE_SEED})")
        clean_std = [float(t.std()) for t in targets_np]
        targets_np = degrade_target_drrs(targets_np)
        for view, cs, t in zip(views, clean_std, targets_np):
            print(f"  [{view['name']:>8}] std {cs:.4f} -> {float(t.std()):.4f}")

    # --- Per-view tool masks (rendered from the tool-only μ-volume) ----------
    # Rendered with normalize=False + invert=False — the renderer's output
    # is then transmittance T = exp(−∫μ dl):
    #   air rays (integral = 0)             → T ≈ 1
    #   tool centre (integral ≈ 9 nepers)   → T ≈ 1.2e−4 ≈ 0
    # The tool's occlusion = 1 − T.  A pixel is "in the tool's shadow" when
    # T < some threshold (equivalently occlusion > 1 − threshold).
    # TOOL_MASK_THRESH expresses the OCCLUSION cutoff (default 0.5 → mask
    # pixels whose chord through the tool absorbs at least 50% of the beam,
    # i.e. transmittance < 0.5 → chord ≥ ln(2)/μ ≈ 2.3 mm at μ=0.3).
    masks_np: list[np.ndarray] = []
    if paint_tool:
        print("\n[2b] Building per-view tool masks (rendering tool-only volume)...")
        mask_cfg = SlangDiffDRRConfig(
            det_height_px=cfg.det_height_px, det_width_px=cfg.det_width_px,
            pixel_spacing_mm=cfg.pixel_spacing_mm,
            source_to_detector_mm=cfg.source_to_detector_mm,
            source_to_isocenter_mm=cfg.source_to_isocenter_mm,
            step_mm=cfg.step_mm,
            normalize=False, invert=False,
        )
        mask_renderer = SlangDiffDRRRenderer(mu_tool_only, spacing, mask_cfg)
        _ = mask_renderer.render((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        for view, (r_eff_gt, t_eff_gt) in zip(views, per_view_eff_geom):
            transmittance = mask_renderer.render(r_eff_gt, t_eff_gt)
            occlusion = 1.0 - transmittance
            mask = (occlusion > TOOL_MASK_THRESH).astype(np.float32)
            masks_np.append(mask)
            frac = 100.0 * mask.mean()
            print(f"  [{view['name']:>8}] tool mask: {int(mask.sum())} of "
                  f"{mask.size} pixels ({frac:.2f}%, "
                  f"max occlusion={float(occlusion.max()):.3f}, "
                  f"thresh={TOOL_MASK_THRESH:.2f})")
        del mask_renderer  # free GPU volume; we won't need it again
        mu_tool_only = None
    else:
        for view in views:
            h, w = targets_np[0].shape[:2]
            masks_np.append(np.zeros((h, w), dtype=np.float32))
    del target_renderer  # the optimiser uses its own renderer (clean μ)

    # --- Optimizer setup -----------------------------------------------------
    # Blind init: start from C-arm isocenter + planning offset, no GT knowledge.
    # INIT_OFFSET_MM is an absolute offset from the isocenter (not from GT).
    # GT is used only for error reporting, never for initialisation.
    print("\n[3] Initializing the registration at a perturbed pose...")
    init_trans = np.asarray(INIT_OFFSET_MM, dtype=np.float32)
    init_rot   = init_rot_rad.copy()

    drr_module, _, translation, _ = create_slang_diffdrr_optimizer(
        mu_volume=mu,
        spacing_zyx_mm=spacing,
        initial_rotation=np.zeros(3, dtype=np.float32),
        initial_translation=init_trans,
        cfg=cfg,
        lr=LR_MM,
    )
    translation.requires_grad_(True)
    phantom_rot = torch.tensor(init_rot, dtype=torch.float32,
                               device=translation.device, requires_grad=True)

    optimizer = torch.optim.Adam([
        {"params": [translation], "lr": LR_MM},
        {"params": [phantom_rot], "lr": LR_ROT_RAD},
    ])

    # Per-view gantry rotation tensors (constant, no grad)
    rot_gantry_tensors = [
        torch.tensor(view["rotation_rad"], dtype=torch.float32,
                     device=translation.device)
        for view in views
    ]
    targets = [torch.from_numpy(t).to(translation.device) for t in targets_np]
    # Anatomy-only weighting for the loss: 0 inside the tool footprint, 1 elsewhere.
    # When paint_tool is False all masks are zero → weights are all-1 → identical
    # to the pre-masking behaviour.
    weights = [
        (1.0 - torch.from_numpy(m).to(translation.device)) for m in masks_np
    ]
    weight_norms = [w.sum().clamp(min=1.0) for w in weights]
    if paint_tool:
        anatomy_fracs = [float(w.mean().item()) for w in weights]
        print(f"  Mask-aware loss enabled (anatomy fraction per view: "
              f"{[round(f, 3) for f in anatomy_fracs]})")

    init_trans_err = init_trans - np.asarray(gt_translation_mm, dtype=np.float32)
    init_rot_err_deg = np.degrees(init_rot) - np.degrees(np.asarray(gt_phantom_rot_euler, dtype=np.float32))
    print(f"  Init translation (mm):  {init_trans.tolist()}")
    print(f"  GT   translation (mm):  {list(gt_translation_mm)}")
    print(f"  Init rot (deg):         "
          f"{[round(math.degrees(v), 3) for v in init_rot.tolist()]}")
    print(f"  GT   rot (deg):         "
          f"{[round(math.degrees(v), 3) for v in gt_phantom_rot_euler]}")
    print(f"  Init translation ||err||: {np.linalg.norm(init_trans_err):.3f} mm")
    print(f"  Init rotation    ||err||: {np.linalg.norm(init_rot_err_deg):.3f} °")

    # Build GT rotation matrix once for error reporting
    gt_rot_np = np.asarray(gt_phantom_rot_euler, dtype=np.float64)
    with torch.no_grad():
        R_gt_np = euler_zxy_to_matrix(
            torch.tensor(gt_rot_np, dtype=torch.float32)
        ).numpy().astype(np.float64)

    # --- Capture-range study (Step 2): sweep init offsets, then exit ---------
    if CAPTURE_RANGE:
        cr = run_capture_range(
            drr_module, rot_gantry_tensors, targets, weights, weight_norms,
            gt_translation_mm, gt_phantom_rot_euler, R_gt_np,
            translation.device,
        )
        with open(OUT_DIR / "capture_range.json", "w") as f:
            json.dump(cr, f, indent=2)
        print(f"\n[CR] Wrote {OUT_DIR / 'capture_range.json'}")
        return 0

    # --- Optimize ------------------------------------------------------------
    print("\n[4] Optimizing (summed MSE across views, 6-DOF)...")
    trace, elapsed = run_optimization(
        drr_module, translation, phantom_rot, optimizer,
        rot_gantry_tensors, targets, weights, weight_norms,
        gt_translation_mm, gt_phantom_rot_euler, R_gt_np,
        N_ITERS, view_names=[v["name"] for v in views], log_every=LOG_EVERY,
    )

    # --- Save artifacts ------------------------------------------------------
    print("\n[5] Saving outputs...")
    final_trans_mm = translation.detach().cpu().numpy()
    final_rot_np   = phantom_rot.detach().cpu().numpy()

    recovered_np: list[np.ndarray] = []
    R_ph_final = euler_zxy_to_matrix(
        torch.tensor(final_rot_np, dtype=torch.float32)
    )

    # For the recovered images, paint the tool at the RECOVERED EE voxel so
    # the plotted DRR shows the tool in the position the registration places
    # it. At convergence this matches the GT position in the target DRR.
    # We render through an extra renderer; the optimiser's drr_module is
    # not used here because it carries the clean μ-volume.
    if paint_tool:
        R_phantom_recov_np = R_ph_final.numpy().astype(np.float64)
        carm_world_np = np.asarray(isaac["carm_pos_world_m"], dtype=np.float64)
        phantom_pos_recov_m = carm_world_np - final_trans_mm.astype(np.float64) / 1000.0
        ee_voxel_recov = ee_voxel_zyx_index(
            ee_pos_world, phantom_pos_recov_m, R_phantom_recov_np,
            spacing, mu.shape,
        )
        print(f"  EE voxel (recov z,y,x): "
              f"({ee_voxel_recov[0]:.1f}, {ee_voxel_recov[1]:.1f}, {ee_voxel_recov[2]:.1f})  "
              f"Δ vs GT = "
              f"({ee_voxel_recov[0] - ee_voxel_gt[0]:+.2f}, "
              f"{ee_voxel_recov[1] - ee_voxel_gt[1]:+.2f}, "
              f"{ee_voxel_recov[2] - ee_voxel_gt[2]:+.2f}) voxels")
        if use_stamp:
            mu_recov, _ = paint_stamp_into_mu(
                mu, spacing, ee_pos_world, ee_quat_world,
                phantom_pos_recov_m, R_phantom_recov_np, tool_stamp_info,
            )
        else:
            mu_recov, _, _ = paint_sphere_into_mu(
                mu, ee_voxel_recov, spacing, TOOL_RADIUS_MM, TOOL_MU_PER_MM,
            )
        recov_renderer = SlangDiffDRRRenderer(mu_recov, spacing, cfg)
        _ = recov_renderer.render((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        for rot_gantry, view, mask in zip(rot_gantry_tensors, views, masks_np):
            with torch.no_grad():
                R_eff    = R_ph_final.T @ euler_zxy_to_matrix(rot_gantry)
                euler_eff = tuple(matrix_to_euler_zxy(R_eff).tolist())
                t_eff    = tuple((R_ph_final.T @ translation).tolist())
            recovered_np.append(recov_renderer.render(euler_eff, t_eff))
            np.save(OUT_DIR / f"target_{view['name']}.npy",
                    targets_np[len(recovered_np) - 1])
            np.save(OUT_DIR / f"recovered_{view['name']}.npy", recovered_np[-1])
            np.save(OUT_DIR / f"mask_{view['name']}.npy", mask)
        del recov_renderer
    else:
        for rot_gantry, view in zip(rot_gantry_tensors, views):
            with torch.no_grad():
                R_eff    = R_ph_final.T @ euler_zxy_to_matrix(rot_gantry)
                euler_eff = matrix_to_euler_zxy(R_eff)
                t_eff    = R_ph_final.T @ translation
                recovered_np.append(drr_module(euler_eff, t_eff).cpu().numpy())
            np.save(OUT_DIR / f"target_{view['name']}.npy",
                    targets_np[len(recovered_np) - 1])
            np.save(OUT_DIR / f"recovered_{view['name']}.npy", recovered_np[-1])

    final_rot_mat = R_ph_final.numpy().astype(np.float64)
    final_rot_quat = rotation_matrix_to_quat_wxyz(final_rot_mat)

    summary: dict = {
        "views": [
            {"name": v["name"], "angle_deg_y": v["angle_deg"]}
            for v in views
        ],
        "gt_translation_mm":     list(gt_translation_mm),
        "gt_phantom_rot_euler_deg": [math.degrees(v) for v in gt_phantom_rot_euler],
        "init_offset_mm":        list(INIT_OFFSET_MM),
        "init_rot_deg":          list(INIT_ROT_DEG),
        "init_translation_mm":   init_trans.tolist(),
        "init_rot_euler_deg":    [math.degrees(v) for v in init_rot.tolist()],
        "final_translation_mm":  final_trans_mm.tolist(),
        "final_rot_euler_rad":   final_rot_np.tolist(),
        "final_rot_euler_deg":   [math.degrees(v) for v in final_rot_np.tolist()],
        "final_rot_quat_wxyz":   final_rot_quat,
        "final_err_mm":          trace[-1]["err_mm"],
        "final_err_norm_mm":     trace[-1]["err_norm_mm"],
        "final_rot_err_euler_deg": trace[-1]["rot_err_euler_deg"],
        "final_rot_err_geodesic_deg": trace[-1]["rot_err_geodesic_deg"],
        "init_err_norm_mm":      float(np.linalg.norm(init_trans_err)),
        "init_rot_err_norm_deg": float(np.linalg.norm(init_rot_err_deg)),
        "n_iters":   N_ITERS,
        "lr_mm":     LR_MM,
        "lr_rot_rad": LR_ROT_RAD,
        "wall_seconds": elapsed,
        "tool_in_target": bool(paint_tool),
        "tool_shape": ("mesh_stamp" if (paint_tool and use_stamp)
                       else ("sphere" if paint_tool else "none")),
        "tool_params": {
            "radius_mm":      TOOL_RADIUS_MM,
            "mu_per_mm":      TOOL_MU_PER_MM,
            "mask_threshold": TOOL_MASK_THRESH,
            "ee_pos_world_m": ee_pos_world if paint_tool else None,
            "ee_quat_world_wxyz": ee_quat_world if paint_tool else None,
            "stamp_source":      (tool_stamp_info["meta"].get("source_mesh_meta", {}).get("mesh_prim")
                                  if (paint_tool and use_stamp) else None),
            "ee_voxel_gt_zyx":   ee_voxel_gt.tolist() if paint_tool else None,
            "ee_voxel_recov_zyx": ee_voxel_recov.tolist() if paint_tool else None,
            "anatomy_fraction_per_view": (
                [float((1.0 - torch.from_numpy(m)).mean().item())
                 for m in masks_np] if paint_tool else None
            ),
        },
        "trace": trace,
    }

    if isaac is not None:
        carm_world   = np.asarray(isaac["carm_pos_world_m"])
        phantom_gt   = np.asarray(isaac["phantom_pos_world_m"])
        phantom_init = translation_mm_to_phantom_world_m(init_trans, carm_world)
        phantom_recov = translation_mm_to_phantom_world_m(final_trans_mm, carm_world)
        world_err_m  = phantom_recov - phantom_gt
        with torch.no_grad():
            R_gt_np_f32 = euler_zxy_to_matrix(
                torch.tensor(gt_phantom_rot_euler, dtype=torch.float32)
            ).numpy().astype(np.float64)
        rot_err_geo = geodesic_angle_deg(R_gt_np_f32, final_rot_mat)
        summary["isaac_ground_truth"] = {
            "phantom_pos_world_m":          phantom_gt.tolist(),
            "carm_pos_world_m":             carm_world.tolist(),
            "phantom_pos_init_world_m":     phantom_init.tolist(),
            "phantom_pos_recovered_world_m": phantom_recov.tolist(),
            "world_err_mm":                 (world_err_m * 1000.0).tolist(),
            "world_err_norm_mm":            float(np.linalg.norm(world_err_m) * 1000.0),
            "phantom_quat_recovered_wxyz":  final_rot_quat,
            "rot_err_geodesic_deg":         rot_err_geo,
        }

    with open(OUT_DIR / "registration_trace.json", "w") as f:
        json.dump(summary, f, indent=2)

    for view in views:
        name = view["name"]
        print(f"  Wrote {OUT_DIR / f'target_{name}.npy'} / "
              f"{OUT_DIR / f'recovered_{name}.npy'}")
    print(f"  Wrote {OUT_DIR / 'registration_trace.json'}")

    # --- Print summary -------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Summary  ({len(views)} views: {[v['name'] for v in views]})")
    print("=" * 60)
    print(f"  Init translation  ||err||:  {np.linalg.norm(init_trans_err):.3f} mm")
    print(f"  Final translation ||err||:  {trace[-1]['err_norm_mm']:.4f} mm")
    print(f"  Final per-axis trans err:   "
          f"{[round(v, 4) for v in trace[-1]['err_mm']]} mm")
    print(f"  Init  rotation    ||err||:  {np.linalg.norm(init_rot_err_deg):.3f} °")
    print(f"  Final rotation    ||err||:  "
          f"{trace[-1]['rot_err_geodesic_deg']:.4f} ° (geodesic)")
    print(f"  Final per-axis rot err:     "
          f"{[round(v, 4) for v in trace[-1]['rot_err_euler_deg']]} °")
    print(f"  Wall time:                  {elapsed:.1f} s "
          f"({elapsed / N_ITERS * 1000:.0f} ms/iter)")

    if isaac is not None:
        ig = summary["isaac_ground_truth"]
        print()
        print("=" * 60)
        print("Summary  (Isaac Sim world frame)")
        print("=" * 60)
        print(f"  Ground-truth phantom_pos (m):  {ig['phantom_pos_world_m']}")
        print(f"  Recovered    phantom_pos (m):  "
              f"{[round(v, 5) for v in ig['phantom_pos_recovered_world_m']]}")
        print(f"  World-frame trans err  (mm):   "
              f"{[round(v, 4) for v in ig['world_err_mm']]}")
        print(f"  World-frame ||t_err||  (mm):   {ig['world_err_norm_mm']:.4f}")
        print(f"  Rotation geodesic err  (deg):  {ig['rot_err_geodesic_deg']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

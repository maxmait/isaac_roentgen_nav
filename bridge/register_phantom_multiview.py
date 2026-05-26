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


VIEWS_DEG_Y: tuple[float, ...] = _env_views_deg((0.0, 90.0))
USE_CARM_ROTATION = bool(int(os.environ.get("USE_CARM_ROTATION", "0")))

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
    return {
        "phantom_pos_world_m":    phantom_pos.tolist(),
        "carm_pos_world_m":       carm_pos.tolist(),
        "gt_translation_mm":      gt_translation_mm.tolist(),
        "gt_phantom_rot_euler":   gt_phantom_rot_euler.tolist(),
        "phantom_quat_wxyz":      phantom_quat,
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
            base = float(_pose["carm_rotation_y_deg"])
            view_angles = (base, base + 90.0)
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

    # --- Target images (rendered at GT pose) ---------------------------------
    print("\n[2] Simulating OR workflow — taking each shot in sequence...")
    target_renderer = SlangDiffDRRRenderer(mu, spacing, cfg)
    _ = target_renderer.render((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))  # JIT warmup

    # Compute GT effective rotation/translation per view (accounts for phantom rotation)
    gt_trans_t = torch.tensor(gt_translation_mm,    dtype=torch.float32)
    gt_rot_t   = torch.tensor(gt_phantom_rot_euler, dtype=torch.float32)
    R_phantom_gt = euler_zxy_to_matrix(gt_rot_t)  # no grad needed here

    targets_np: list[np.ndarray] = []
    for view in views:
        R_gantry = euler_zxy_to_matrix(
            torch.tensor(view["rotation_rad"], dtype=torch.float32)
        )
        t_eff_gt = (R_phantom_gt.T @ gt_trans_t).tolist()
        r_eff_gt = matrix_to_euler_zxy(R_phantom_gt.T @ R_gantry).tolist()
        img = target_renderer.render(tuple(r_eff_gt), tuple(t_eff_gt))
        targets_np.append(img)
        print(f"  [{view['name']:>8}] ry={view['angle_deg']:+.1f}°  "
              f"range=[{img.min():.4f}, {img.max():.4f}]  std={img.std():.4f}")

    # --- Optimizer setup -----------------------------------------------------
    print("\n[3] Initializing the registration at a perturbed pose...")
    init_trans = (np.asarray(gt_translation_mm,    dtype=np.float32)
                  + np.asarray(INIT_OFFSET_MM,    dtype=np.float32))
    init_rot   = (np.asarray(gt_phantom_rot_euler, dtype=np.float32)
                  + init_rot_rad)

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

    print(f"  Init translation (mm):  {init_trans.tolist()}")
    print(f"  GT   translation (mm):  {list(gt_translation_mm)}")
    print(f"  Init rot (deg):         "
          f"{[round(math.degrees(v), 3) for v in init_rot.tolist()]}")
    print(f"  GT   rot (deg):         "
          f"{[round(math.degrees(v), 3) for v in gt_phantom_rot_euler]}")
    print(f"  Init translation ||err||: {np.linalg.norm(INIT_OFFSET_MM):.3f} mm")
    print(f"  Init rotation    ||err||: {np.linalg.norm(INIT_ROT_DEG):.3f} °")

    # Build GT rotation matrix once for error reporting
    gt_rot_np = np.asarray(gt_phantom_rot_euler, dtype=np.float64)
    with torch.no_grad():
        R_gt_np = euler_zxy_to_matrix(
            torch.tensor(gt_rot_np, dtype=torch.float32)
        ).numpy().astype(np.float64)

    # --- Optimize ------------------------------------------------------------
    print("\n[4] Optimizing (summed MSE across views, 6-DOF)...")
    trace = []
    t_start = time.time()
    for it in range(N_ITERS):
        optimizer.zero_grad()
        total_loss = torch.zeros((), dtype=torch.float32, device=translation.device)
        per_view_loss: list[float] = []

        R_ph = euler_zxy_to_matrix(phantom_rot)
        t_world = translation  # world-frame displacement (mm)

        for rot_gantry, target in zip(rot_gantry_tensors, targets):
            R_eff    = R_ph.T @ euler_zxy_to_matrix(rot_gantry)
            euler_eff = matrix_to_euler_zxy(R_eff)
            t_eff    = R_ph.T @ t_world
            rendered = drr_module(euler_eff, t_eff)
            loss_v   = ((rendered - target) ** 2).mean()
            total_loss = total_loss + loss_v
            per_view_loss.append(float(loss_v.item()))

        total_loss.backward()
        # fluorosim Slang autodiff sign-flip + NaN guard.
        # The translation backward is reliable (tested in 3-DOF).
        # The rotation backward (new code path) may produce NaN in some
        # configurations — zero it out to skip that update safely.
        for p in [translation, phantom_rot]:
            if p.grad is None:
                continue
            if torch.isnan(p.grad).any():
                p.grad.zero_()
            else:
                p.grad.neg_()  # sign-flip correction for Slang autodiff
        # Clip rotation gradient to prevent accumulation from near-zero
        # gradient noise on symmetric phantoms (sphere).
        if phantom_rot.grad is not None and phantom_rot.grad.abs().max() > 0:
            torch.nn.utils.clip_grad_norm_([phantom_rot], max_norm=ROT_GRAD_CLIP)
        optimizer.step()

        t_np  = translation.detach().cpu().numpy()
        pr_np = phantom_rot.detach().cpu().numpy()
        trans_err  = t_np - np.asarray(gt_translation_mm)
        rot_err_np = pr_np - np.asarray(gt_phantom_rot_euler, dtype=np.float32)
        with torch.no_grad():
            R_rec_np = euler_zxy_to_matrix(
                torch.tensor(pr_np, dtype=torch.float32)
            ).numpy().astype(np.float64)
        rot_geo_deg = geodesic_angle_deg(R_gt_np, R_rec_np)

        entry = {
            "iter": it,
            "loss_total": float(total_loss.item()),
            "loss_per_view": per_view_loss,
            "translation_mm":     t_np.tolist(),
            "err_mm":             trans_err.tolist(),
            "err_norm_mm":        float(np.linalg.norm(trans_err)),
            "phantom_rot_euler_rad": pr_np.tolist(),
            "rot_err_euler_deg":  [math.degrees(v) for v in rot_err_np.tolist()],
            "rot_err_geodesic_deg": rot_geo_deg,
        }
        trace.append(entry)
        if it % LOG_EVERY == 0 or it == N_ITERS - 1:
            pv = "  ".join(
                f"{view['name']}={per_view_loss[i]:.2e}"
                for i, view in enumerate(views)
            )
            print(f"  iter {it:3d}: total={total_loss.item():.6e}  {pv}  "
                  f"||t_err||={entry['err_norm_mm']:.3f} mm  "
                  f"||r_err||={rot_geo_deg:.3f}°")
    elapsed = time.time() - t_start

    # --- Save artifacts ------------------------------------------------------
    print("\n[5] Saving outputs...")
    final_trans_mm = translation.detach().cpu().numpy()
    final_rot_np   = phantom_rot.detach().cpu().numpy()

    recovered_np: list[np.ndarray] = []
    R_ph_final = euler_zxy_to_matrix(
        torch.tensor(final_rot_np, dtype=torch.float32)
    )
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
        "init_err_norm_mm":      float(np.linalg.norm(INIT_OFFSET_MM)),
        "init_rot_err_norm_deg": float(np.linalg.norm(INIT_ROT_DEG)),
        "n_iters":   N_ITERS,
        "lr_mm":     LR_MM,
        "lr_rot_rad": LR_ROT_RAD,
        "wall_seconds": elapsed,
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
    print(f"  Init translation  ||err||:  {np.linalg.norm(INIT_OFFSET_MM):.3f} mm")
    print(f"  Final translation ||err||:  {trace[-1]['err_norm_mm']:.4f} mm")
    print(f"  Final per-axis trans err:   "
          f"{[round(v, 4) for v in trace[-1]['err_mm']]} mm")
    print(f"  Init  rotation    ||err||:  {np.linalg.norm(INIT_ROT_DEG):.3f} °")
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

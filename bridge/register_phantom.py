#!/usr/bin/env python3
"""Phantom translation registration via fluorosim's Slang autodiff (Phase 5, Option A, 3-DOF).

Test of the registration loop:

  1. If pose.json from Isaac Sim is present, use its phantom_pos / carm_pos as
     ground truth (the volume placement in the actual scene). Otherwise fall
     back to a synthetic-to-synthetic run with GT_TRANSLATION_MM env var.
  2. Render a target DRR at the ground-truth translation.
  3. Initialize fluorosim's translation parameter at a perturbed value.
  4. Optimize the translation by minimizing MSE between rendered and target
     DRR. Gradients come from fluorosim's Slang shader autodiff.
  5. Convert recovered translation_mm back to world-frame phantom_pos and
     report the world-frame error against Isaac Sim's ground truth.

Rotation is held fixed at zero — we're only recovering 3 DOF in this first
cut. The phantom is the same clean synthetic sphere fluorosim_render.py uses;
the robot tool is NOT painted in (it would dominate the loss and is not what
we're trying to register).

Runs inside the fluorosim-torch Docker container; outputs land in
/workspace/io/registration/ (= ~/isaac_projects/output/registration/ on host).
"""

from __future__ import annotations

import json
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
OUT_DIR = IO_DIR / "registration"
POSE_FILE = IO_DIR / "pose.json"

# If set, load μ-volume from a DICOM CT instead of the synthetic ellipsoid.
DICOM_PATH = os.environ.get("DICOM_PATH")
# CT_FULL_VOLUME=1 → use the entire CT at native spacing (no crop/resample).
CT_FULL_VOLUME = os.environ.get("CT_FULL_VOLUME", "0") == "1"
CACHE_DIR_CT = IO_DIR / ("fluorosim_cache_ct_full" if CT_FULL_VOLUME else "fluorosim_cache_ct")

# Experiment parameters (overridable via env vars for quick sweeps from the host).
def _envf(name: str, default: float) -> float:
    return float(os.environ.get(name, default))

def _envv(name: str, default: tuple[float, float, float]) -> tuple[float, float, float]:
    raw = os.environ.get(name)
    if not raw:
        return default
    return tuple(float(v) for v in raw.split(","))  # type: ignore[return-value]

_GT_TRANSLATION_MM_ENV = _envv("GT_TRANSLATION_MM", (0.0, 0.0, 0.0))
INIT_OFFSET_MM = _envv("INIT_OFFSET_MM", (15.0, -10.0, 8.0))
LR_MM = _envf("LR_MM", 1.0)
N_ITERS = int(_envf("N_ITERS", 100))
LOG_EVERY = int(_envf("LOG_EVERY", 5))
# If USE_POSE_JSON=0, ignore pose.json even if present (forces synthetic-to-
# synthetic mode for debugging). Default: 1 (use pose.json when available).
USE_POSE_JSON = bool(int(os.environ.get("USE_POSE_JSON", "1")))


def load_isaac_ground_truth() -> dict | None:
    """If pose.json is present, return {phantom_pos_world_m, carm_pos_world_m,
    gt_translation_mm} where gt_translation_mm is fluorosim's parameter value
    that corresponds to the Isaac Sim scene state. Returns None if pose.json
    is missing or USE_POSE_JSON is disabled.
    """
    if not USE_POSE_JSON or not POSE_FILE.exists():
        return None
    with open(POSE_FILE) as f:
        pose = json.load(f)
    phantom_pos = np.asarray(pose.get("phantom_pos", [0.0, 0.0, 0.0]), dtype=np.float64)
    carm_pos = np.asarray(pose.get("carm_pos", [0.0, 0.0, 0.0]), dtype=np.float64)
    gt_translation_mm = (carm_pos - phantom_pos) * 1000.0
    return {
        "phantom_pos_world_m": phantom_pos.tolist(),
        "carm_pos_world_m": carm_pos.tolist(),
        "gt_translation_mm": gt_translation_mm.tolist(),
    }


def translation_mm_to_phantom_world_m(
    translation_mm: np.ndarray, carm_pos_world_m: np.ndarray
) -> np.ndarray:
    """Inverse of `(carm - phantom) * 1000`: given fluorosim's parameter and
    the (fixed) world-frame C-arm position, recover the phantom world position."""
    return carm_pos_world_m - np.asarray(translation_mm) / 1000.0


def load_or_build_synthetic_volume() -> PreprocessedVolume:
    """Same synthetic phantom as fluorosim_render.py — reuses the same cache."""
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


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("register_phantom.py — translation-only phantom registration")
    print("=" * 60)

    isaac = load_isaac_ground_truth()
    if isaac is not None:
        gt_translation_mm = isaac["gt_translation_mm"]
        print("Ground truth from Isaac Sim pose.json:")
        print(f"  phantom_pos (world m): {isaac['phantom_pos_world_m']}")
        print(f"  carm_pos    (world m): {isaac['carm_pos_world_m']}")
        print(f"  -> fluorosim translation (mm): {gt_translation_mm}")
    else:
        gt_translation_mm = list(_GT_TRANSLATION_MM_ENV)
        print(f"No pose.json (or USE_POSE_JSON=0) — synthetic-to-synthetic mode.")
        print(f"  Ground-truth translation (mm): {gt_translation_mm}")
    print(f"Initial offset from GT  (mm):  {INIT_OFFSET_MM}")
    print(f"Learning rate (mm/step):       {LR_MM}")
    print(f"Iterations:                    {N_ITERS}")

    print("\n[1] Loading μ-volume...")
    volume = load_volume()
    mu = volume.mu_volume
    spacing = volume.spacing_zyx_mm
    print(f"  shape={mu.shape}, spacing_mm={spacing}, μ range=[{mu.min():.4f}, {mu.max():.4f}]")

    cfg = SlangDiffDRRConfig(
        det_height_px=512,
        det_width_px=512,
        pixel_spacing_mm=0.5,
        source_to_detector_mm=1020.0,
        source_to_isocenter_mm=510.0,
        step_mm=0.5,
        normalize=True,
        invert=True,
    )

    # --- Step A: render the *target* DRR at ground-truth translation -------
    print("\n[2] Rendering target DRR at ground truth...")
    target_renderer = SlangDiffDRRRenderer(mu, spacing, cfg)
    # warm up (JIT compilation)
    _ = target_renderer.render((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    target_np = target_renderer.render((0.0, 0.0, 0.0), tuple(gt_translation_mm))
    print(f"  target shape={target_np.shape}, "
          f"range=[{target_np.min():.4f}, {target_np.max():.4f}], "
          f"std={target_np.std():.4f}")

    # --- Step B: build the optimizer at perturbed initial pose -------------
    print("\n[3] Building differentiable renderer at perturbed initial pose...")
    init_trans = np.asarray(gt_translation_mm, dtype=np.float32) \
                 + np.asarray(INIT_OFFSET_MM, dtype=np.float32)
    init_rot = np.zeros(3, dtype=np.float32)

    # The helper bundles its own Adam — we keep its renderer + tensors and
    # build our own optimizer over translation only.
    drr_module, rotation, translation, _ = create_slang_diffdrr_optimizer(
        mu_volume=mu,
        spacing_zyx_mm=spacing,
        initial_rotation=init_rot,
        initial_translation=init_trans,
        cfg=cfg,
        lr=LR_MM,  # ignored; we replace the optimizer below
    )
    rotation.requires_grad_(False)
    translation.requires_grad_(True)
    optimizer = torch.optim.Adam([translation], lr=LR_MM)

    target = torch.from_numpy(target_np).to(translation.device)
    print(f"  Initial translation:  {translation.detach().cpu().numpy().tolist()}")
    print(f"  Ground truth:         {list(gt_translation_mm)}")
    print(f"  Initial error norm:   {np.linalg.norm(INIT_OFFSET_MM):.3f} mm")

    # --- Step C: optimize --------------------------------------------------
    print("\n[4] Optimizing...")
    # NOTE: fluorosim's Slang autodiff path returns gradients with FLIPPED sign
    # relative to PyTorch's convention. Verified via finite differences: at
    # trans=(15,0,0) with MSE loss, Slang returns dL/dt_x ≈ -0.002 while the
    # numerical gradient is +0.003. We negate the gradient before optimizer.step()
    # so plain Adam converges. (Without this, the loss diverges.)
    trace = []
    t_start = time.time()
    for it in range(N_ITERS):
        optimizer.zero_grad()
        rendered = drr_module(rotation, translation)
        loss = ((rendered - target) ** 2).mean()
        loss.backward()
        if translation.grad is not None:
            translation.grad.neg_()  # workaround for fluorosim Slang sign convention
        optimizer.step()

        t_np = translation.detach().cpu().numpy()
        err = t_np - np.asarray(gt_translation_mm)
        entry = {
            "iter": it,
            "loss": float(loss.item()),
            "translation_mm": t_np.tolist(),
            "err_mm": err.tolist(),
            "err_norm_mm": float(np.linalg.norm(err)),
        }
        trace.append(entry)
        if it % LOG_EVERY == 0 or it == N_ITERS - 1:
            print(f"  iter {it:3d}: loss={loss.item():.6e}  "
                  f"t={t_np.round(3).tolist()}  ||err||={entry['err_norm_mm']:.3f} mm")
    elapsed = time.time() - t_start

    # --- Step D: save artifacts -------------------------------------------
    print("\n[5] Saving outputs...")
    recovered_np = drr_module(rotation, translation).detach().cpu().numpy()

    np.save(OUT_DIR / "target.npy", target_np)
    np.save(OUT_DIR / "recovered.npy", recovered_np)

    final_trans_mm = translation.detach().cpu().numpy()
    summary = {
        "gt_translation_mm": list(gt_translation_mm),
        "init_offset_mm": list(INIT_OFFSET_MM),
        "init_translation_mm": init_trans.tolist(),
        "final_translation_mm": final_trans_mm.tolist(),
        "final_err_mm": trace[-1]["err_mm"],
        "final_err_norm_mm": trace[-1]["err_norm_mm"],
        "init_err_norm_mm": float(np.linalg.norm(INIT_OFFSET_MM)),
        "n_iters": N_ITERS,
        "lr_mm": LR_MM,
        "wall_seconds": elapsed,
        "trace": trace,
    }
    # If we have Isaac Sim ground truth, also report the world-frame recovery.
    if isaac is not None:
        carm_world = np.asarray(isaac["carm_pos_world_m"])
        phantom_gt = np.asarray(isaac["phantom_pos_world_m"])
        phantom_init = translation_mm_to_phantom_world_m(init_trans, carm_world)
        phantom_recov = translation_mm_to_phantom_world_m(final_trans_mm, carm_world)
        world_err_m = phantom_recov - phantom_gt
        summary["isaac_ground_truth"] = {
            "phantom_pos_world_m": phantom_gt.tolist(),
            "carm_pos_world_m": carm_world.tolist(),
            "phantom_pos_init_world_m": phantom_init.tolist(),
            "phantom_pos_recovered_world_m": phantom_recov.tolist(),
            "world_err_mm": (world_err_m * 1000.0).tolist(),
            "world_err_norm_mm": float(np.linalg.norm(world_err_m) * 1000.0),
        }

    with open(OUT_DIR / "registration_trace.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Wrote {OUT_DIR / 'target.npy'}")
    print(f"  Wrote {OUT_DIR / 'recovered.npy'}")
    print(f"  Wrote {OUT_DIR / 'registration_trace.json'}")

    print("\n" + "=" * 60)
    print("Summary  (fluorosim translation_mm space)")
    print("=" * 60)
    print(f"  Initial error norm:   {np.linalg.norm(INIT_OFFSET_MM):.3f} mm")
    print(f"  Final   error norm:   {trace[-1]['err_norm_mm']:.4f} mm")
    print(f"  Final   per-axis err: "
          f"{[round(v, 4) for v in trace[-1]['err_mm']]} mm")
    print(f"  Loss reduction:       "
          f"{trace[0]['loss']:.4e} -> {trace[-1]['loss']:.4e}  "
          f"({trace[0]['loss'] / max(trace[-1]['loss'], 1e-20):.1f}x)")
    print(f"  Wall time:            {elapsed:.1f} s "
          f"({elapsed / N_ITERS * 1000:.0f} ms/iter)")

    if isaac is not None:
        ig = summary["isaac_ground_truth"]
        print()
        print("=" * 60)
        print("Summary  (Isaac Sim world frame)")
        print("=" * 60)
        print(f"  Ground-truth phantom_pos (m):  {ig['phantom_pos_world_m']}")
        print(f"  Initial guess  phantom_pos (m): "
              f"{[round(v, 5) for v in ig['phantom_pos_init_world_m']]}")
        print(f"  Recovered      phantom_pos (m): "
              f"{[round(v, 5) for v in ig['phantom_pos_recovered_world_m']]}")
        print(f"  World-frame error    (mm):     "
              f"{[round(v, 4) for v in ig['world_err_mm']]}")
        print(f"  World-frame ||err||  (mm):     "
              f"{ig['world_err_norm_mm']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

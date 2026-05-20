#!/usr/bin/env python3
"""Multi-view phantom translation registration (Phase 5, Option A, 3-DOF).

Clinical workflow: a mobile C-arm takes ONE shot at a time. A surgeon typically:

    1. Takes an AP (anterior-posterior) shot.
    2. Rotates the C-arm ~90° around the patient's long axis.
    3. Takes a lateral shot.
    4. Uses BOTH images together to triangulate the surgical target in 3D.

This script models that exactly. The phantom (patient) doesn't move between
shots — same world-frame phantom_pos for both — only the C-arm rotation
changes. So fluorosim's `translation_mm` is shared across views (it's the
unknown we're recovering), while `rotation` differs per view.

Loss = sum of MSE losses across views. A single Adam optimizer runs on a
single translation parameter, with gradients accumulating across both views.

Compared to the single-view register_phantom.py:
    Single view (AP only)  →  ~0.4 mm error on Z (X-ray beam axis, depth)
    AP + Lateral           →  Z is in-plane in the lateral view; expect
                              all three axes to converge to in-plane accuracy

Reads pose.json from Isaac Sim if present and reports the world-frame
phantom_pos recovery against the Isaac Sim ground truth.

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

# If set, load the μ-volume from a DICOM CT instead of the synthetic ellipsoid.
# The DICOM_PATH value is the in-container path to the DICOM dir (the host
# wrapper script mounts the host DICOM dir there and sets this var).
DICOM_PATH = os.environ.get("DICOM_PATH")
# CT_FULL_VOLUME=1 → use the entire CT at native spacing (no crop/resample).
# Cache dirs are kept separate so both variants can coexist on disk.
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


# Y-axis rotations (LAO/RAO) for each view, in degrees. Default: AP + Lateral.
# Surgeons may pick non-90° lateral angles in practice — e.g. "0,60" for a
# 60° oblique pair. Override via VIEWS_DEG_Y="0,90" (default), "0,45,90", etc.
VIEWS_DEG_Y: tuple[float, ...] = _env_views_deg((0.0, 90.0))

# When USE_CARM_ROTATION=1 and pose.json contains carm_rotation_y_deg, treat
# that angle (the C-arm prim's rotation around the scene's up axis) as the
# AP view, with lateral = AP + 90°.  This lets the GUI C-arm pose drive the
# DRR view direction.
USE_CARM_ROTATION = bool(int(os.environ.get("USE_CARM_ROTATION", "0")))

INIT_OFFSET_MM = _envv("INIT_OFFSET_MM", (15.0, -10.0, 8.0))
LR_MM = _envf("LR_MM", 1.0)
N_ITERS = int(_envf("N_ITERS", 100))
LOG_EVERY = int(_envf("LOG_EVERY", 5))
USE_POSE_JSON = bool(int(os.environ.get("USE_POSE_JSON", "1")))


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


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("register_phantom_multiview.py — sequential C-arm shots")
    print("=" * 60)

    isaac = load_isaac_ground_truth()
    if isaac is not None:
        gt_translation_mm = isaac["gt_translation_mm"]
        print("Ground truth from Isaac Sim pose.json:")
        print(f"  phantom_pos (world m): {isaac['phantom_pos_world_m']}")
        print(f"  carm_pos    (world m): {isaac['carm_pos_world_m']}")
        print(f"  -> fluorosim translation (mm): {gt_translation_mm}")
    else:
        gt_translation_mm = [0.0, 0.0, 0.0]
        print("No pose.json — synthetic mode, GT translation = (0, 0, 0).")

    # Optionally read the C-arm rotation from pose.json — when set, the
    # AP angle becomes the C-arm's actual rotation in the scene.
    view_angles = VIEWS_DEG_Y
    if USE_CARM_ROTATION and POSE_FILE.exists():
        with open(POSE_FILE) as f:
            _pose = json.load(f)
        if "carm_rotation_y_deg" in _pose:
            base = float(_pose["carm_rotation_y_deg"])
            view_angles = (base, base + 90.0)
            print(f"USE_CARM_ROTATION=1: AP angle from C-arm prim = {base:+.1f}°  "
                  f"-> views (ry) = {view_angles}")

    views = [
        {"name": view_name(a), "angle_deg": a,
         "rotation_rad": (0.0, math.radians(a), 0.0)}
        for a in view_angles
    ]
    print(f"Views (ry angles, deg): {[v['angle_deg'] for v in views]}  "
          f"-> names: {[v['name'] for v in views]}")
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

    # --- "Step 1: Take AP shot. Step 2: Rotate C-arm. Step 3: Take lateral." -
    print("\n[2] Simulating the OR workflow — taking each shot in sequence...")
    target_renderer = SlangDiffDRRRenderer(mu, spacing, cfg)
    _ = target_renderer.render((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))  # JIT warmup
    targets_np: list[np.ndarray] = []
    for view in views:
        img = target_renderer.render(view["rotation_rad"], tuple(gt_translation_mm))
        targets_np.append(img)
        print(f"  [{view['name']:>8}] ry={view['angle_deg']:+.1f}°  "
              f"range=[{img.min():.4f}, {img.max():.4f}]  std={img.std():.4f}")

    # --- Set up the optimizer (single translation, shared across views) ----
    print("\n[3] Initializing the registration at a perturbed pose...")
    init_trans = np.asarray(gt_translation_mm, dtype=np.float32) \
                 + np.asarray(INIT_OFFSET_MM, dtype=np.float32)
    init_rot = np.zeros(3, dtype=np.float32)

    drr_module, _, translation, _ = create_slang_diffdrr_optimizer(
        mu_volume=mu,
        spacing_zyx_mm=spacing,
        initial_rotation=init_rot,
        initial_translation=init_trans,
        cfg=cfg,
        lr=LR_MM,  # ignored; we build our own optimizer below
    )
    translation.requires_grad_(True)
    optimizer = torch.optim.Adam([translation], lr=LR_MM)

    # Per-view rotation tensors (no gradients — these are known/fixed)
    rot_tensors = [
        torch.tensor(view["rotation_rad"], dtype=torch.float32, device=translation.device)
        for view in views
    ]
    targets = [torch.from_numpy(t).to(translation.device) for t in targets_np]

    print(f"  Initial translation:  {translation.detach().cpu().numpy().tolist()}")
    print(f"  Ground truth:         {list(gt_translation_mm)}")
    print(f"  Initial error norm:   {np.linalg.norm(INIT_OFFSET_MM):.3f} mm")

    # --- Optimize -----------------------------------------------------------
    print("\n[4] Optimizing (summed MSE across views)...")
    # fluorosim Slang autodiff returns gradients with flipped sign — negate
    # before optimizer.step(). (See CLAUDE.md known-issues.)
    trace = []
    t_start = time.time()
    for it in range(N_ITERS):
        optimizer.zero_grad()
        total_loss = torch.zeros((), dtype=torch.float32, device=translation.device)
        per_view_loss = []
        for rot, target in zip(rot_tensors, targets):
            rendered = drr_module(rot, translation)
            loss_v = ((rendered - target) ** 2).mean()
            total_loss = total_loss + loss_v
            per_view_loss.append(float(loss_v.item()))
        total_loss.backward()
        if translation.grad is not None:
            translation.grad.neg_()
        optimizer.step()

        t_np = translation.detach().cpu().numpy()
        err = t_np - np.asarray(gt_translation_mm)
        entry = {
            "iter": it,
            "loss_total": float(total_loss.item()),
            "loss_per_view": per_view_loss,
            "translation_mm": t_np.tolist(),
            "err_mm": err.tolist(),
            "err_norm_mm": float(np.linalg.norm(err)),
        }
        trace.append(entry)
        if it % LOG_EVERY == 0 or it == N_ITERS - 1:
            per_view_str = "  ".join(
                f"{view['name']}={per_view_loss[i]:.2e}"
                for i, view in enumerate(views)
            )
            print(f"  iter {it:3d}: total={total_loss.item():.6e}  "
                  f"{per_view_str}  ||err||={entry['err_norm_mm']:.3f} mm")
    elapsed = time.time() - t_start

    # --- Save artifacts ----------------------------------------------------
    print("\n[5] Saving outputs...")
    recovered_np: list[np.ndarray] = []
    for rot, view in zip(rot_tensors, views):
        with torch.no_grad():
            recovered_np.append(drr_module(rot, translation).cpu().numpy())
        np.save(OUT_DIR / f"target_{view['name']}.npy", targets_np[len(recovered_np) - 1])
        np.save(OUT_DIR / f"recovered_{view['name']}.npy", recovered_np[-1])

    final_trans_mm = translation.detach().cpu().numpy()
    summary = {
        "views": [
            {"name": v["name"], "angle_deg_y": v["angle_deg"]}
            for v in views
        ],
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

    for view in views:
        name = view["name"]
        print(f"  Wrote {OUT_DIR / f'target_{name}.npy'} / "
              f"{OUT_DIR / f'recovered_{name}.npy'}")
    print(f"  Wrote {OUT_DIR / 'registration_trace.json'}")

    print("\n" + "=" * 60)
    print(f"Summary  ({len(views)} views: {[v['name'] for v in views]})")
    print("=" * 60)
    print(f"  Initial error norm:   {np.linalg.norm(INIT_OFFSET_MM):.3f} mm")
    print(f"  Final   error norm:   {trace[-1]['err_norm_mm']:.4f} mm")
    print(f"  Final   per-axis err: "
          f"{[round(v, 4) for v in trace[-1]['err_mm']]} mm")
    print(f"  Wall time:            {elapsed:.1f} s "
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
        print(f"  World-frame error     (mm):    "
              f"{[round(v, 4) for v in ig['world_err_mm']]}")
        print(f"  World-frame ||err||   (mm):    "
              f"{ig['world_err_norm_mm']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

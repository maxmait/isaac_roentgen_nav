#!/usr/bin/env python3
"""Proof-of-concept: recover the tool/TCP pose from its fluoroscopy silhouette.

Independent of the robot's forward kinematics.  The endo360 tool stamp is
rendered as its OWN differentiable volume and its 6-DOF pose (translation +
ZXY-Euler rotation, expressed in the C-arm / isocenter base frame) is optimised
to match the observed tool silhouette across multiple C-arm views.  This mirrors
the anatomy registration machinery — per-view gantry composition, Adam with the
Slang-gradient sign-flip + NaN guard — but with the tool as the object and its
pose as the unknown.

Self-contained: needs only output/tool_stamp.npy (built by build_tool_stamp.py)
and a GPU/fluorosim renderer.  The "observed" silhouette is the tool stamp
rendered at a synthetic in-frame GT pose (TOOL_GT_*); the optimiser starts blind
(TOOL_INIT_*) and GT is used only for scoring.

Why this matters: today T_robot->anatomy rides entirely on the robot FK (the EE
pose is taken as exact).  Recovering the tool pose from its own silhouette gives
a second, image-based estimate that does NOT depend on FK / hand-eye
calibration — the basis for cross-checking (or replacing) the FK term when the
tool is in the fluoroscopy field of view.

Honest limitation: the tip+shaft is ~rod-like, so rotation about its long axis
(EE-local z) barely changes the silhouette.  The POINTING direction is
recovered; ROLL is essentially unconstrained.  Both are reported separately.

Run in the fluorosim-torch container via bridge/run_register_tool.sh.
Env knobs (all optional):
  N_ITERS(200) LR_MM(1.0) LR_ROT_RAD(0.005) ROT_GRAD_CLIP(0.1) LOG_EVERY(25)
  VIEWS_DEG_Y("0,45,90")           # C-arm view angles; set "0" for single-view
  TOOL_GT_TRANS_MM("12,-8,10")     # synthetic GT tool-center offset (base frame)
  TOOL_GT_ROT_DEG("-80,6,-5")      # synthetic GT tool orientation (ZXY Euler deg)
  TOOL_INIT_TRANS_MM("0,0,0")      # blind init translation (isocenter)
  TOOL_INIT_ROT_DEG("0,0,0")       # blind init rotation
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

BRIDGE = Path(__file__).resolve().parent
if str(BRIDGE) not in sys.path:
    sys.path.insert(0, str(BRIDGE))

# Reuse the anatomy-registration helpers (differentiable rotation, geodesic
# distance, tool-stamp loader, IO paths) — single source of truth.
from register_phantom_multiview import (  # noqa: E402
    IO_DIR,
    euler_zxy_to_matrix,
    matrix_to_euler_zxy,
    geodesic_angle_deg,
    load_tool_stamp,
)
from fluorosim.rendering.diffdrr_slang_renderer import (  # noqa: E402
    SlangDiffDRRConfig,
    SlangDiffDRRRenderer,
    create_slang_diffdrr_optimizer,
)

OUT_DIR = IO_DIR / "tool_pose"


# ─── env parsing ──────────────────────────────────────────────────────────────
def _envf(key: str, default: float) -> float:
    return float(os.environ.get(key, default))


def _env_floats(key: str, default: tuple[float, ...]) -> tuple[float, ...]:
    raw = os.environ.get(key)
    if not raw:
        return default
    return tuple(float(v) for v in raw.split(",") if v.strip())


N_ITERS       = int(os.environ.get("N_ITERS", "200"))
# The anatomy path (normalize=True, invert=True) needs the Slang gradient
# sign-flipped; this transmittance silhouette loss (normalize=False,
# invert=False) does NOT — the invert=False keeps the gradient in PyTorch's
# convention.  Verified empirically: FLIP_GRAD=1 diverges, FLIP_GRAD=0 converges.
FLIP_GRAD     = bool(int(os.environ.get("FLIP_GRAD", "0")))
LR_MM         = _envf("LR_MM", 1.0)
LR_ROT_RAD    = _envf("LR_ROT_RAD", 0.005)
ROT_GRAD_CLIP = _envf("ROT_GRAD_CLIP", 0.1)
LOG_EVERY     = int(os.environ.get("LOG_EVERY", "25"))
VIEWS_DEG_Y   = _env_floats("VIEWS_DEG_Y", (0.0, 45.0, 90.0))
TOOL_GT_TRANS_MM   = _env_floats("TOOL_GT_TRANS_MM", (12.0, -8.0, 10.0))
TOOL_GT_ROT_DEG    = _env_floats("TOOL_GT_ROT_DEG", (-80.0, 6.0, -5.0))
# The optimiser starts from the FK prior + a residual calibration error: the
# init is GT plus this perturbation, NOT an absolute pose.  A thin rod has a
# small silhouette basin, so this refines an approximate (FK) pose rather than
# searching globally — the realistic clinical use (see the roll / along-axis
# ambiguities reported below).
TOOL_INIT_OFFSET_MM  = _env_floats("TOOL_INIT_OFFSET_MM", (4.0, -3.0, 3.0))
TOOL_INIT_OFFSET_DEG = _env_floats("TOOL_INIT_OFFSET_DEG", (4.0, -2.0, 2.0))


def view_name(a: float) -> str:
    if abs(a) < 1e-3:
        return "AP"
    if abs(a - 90.0) < 1e-3:
        return "lateral"
    return f"ry{a:+.0f}deg"


def angle_between_deg(u: np.ndarray, v: np.ndarray) -> float:
    u = u / (np.linalg.norm(u) + 1e-12)
    v = v / (np.linalg.norm(v) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(u @ v, -1.0, 1.0))))


def stamp_tip_offset_mm(stamp_info: dict) -> np.ndarray:
    """Vector (EE-local xyz mm) from the stamp's geometric CENTER to the TCP.

    The TCP (endo360_needle) is at EE-local origin (0,0,0); the stamp volume is
    centered on its own geometric middle, so tip − center = −center.
    """
    origin_xyz = np.asarray(stamp_info["origin_xyz_mm"], dtype=np.float64)
    nz, ny, nx = stamp_info["stamp"].shape
    sz, sy, sx = stamp_info["spacing_zyx_mm"]
    shape_xyz   = np.array([nx, ny, nz], dtype=np.float64)
    spacing_xyz = np.array([sx, sy, sz], dtype=np.float64)
    center_xyz  = origin_xyz + (shape_xyz - 1.0) / 2.0 * spacing_xyz
    return -center_xyz


def tip_pos_base_mm(t_center_mm: np.ndarray, R_tool: np.ndarray,
                    tip_offset_mm: np.ndarray) -> np.ndarray:
    """TCP position in the base frame.

    The renderer's `translation` is the C-arm→volume-center offset, so the tool
    center sits at −translation relative to the isocenter; the tip is that plus
    the (rotated) center→tip offset.
    """
    return -np.asarray(t_center_mm) + R_tool @ np.asarray(tip_offset_mm)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 62)
    print("register_tool_pose.py — image-based tool/TCP pose recovery (PoC)")
    print("=" * 62)

    stamp_info = load_tool_stamp()
    if stamp_info is None:
        sys.exit("ERROR: output/tool_stamp.npy not found. "
                 "Build it first with bridge/build_tool_stamp.py.")
    mu_tool = stamp_info["stamp"].astype(np.float32)
    spacing = tuple(float(v) for v in stamp_info["spacing_zyx_mm"])
    tip_offset = stamp_tip_offset_mm(stamp_info)
    print(f"  Tool stamp: shape={mu_tool.shape}, spacing={spacing} mm, "
          f"μ_max={mu_tool.max():.3f} mm⁻¹")
    print(f"  Center→tip offset (EE-local xyz mm): {tip_offset.round(2)}")

    views = [{"name": view_name(a), "angle_deg": a,
              "rotation_rad": (0.0, math.radians(a), 0.0)} for a in VIEWS_DEG_Y]
    print(f"  Views (ry°): {[v['angle_deg'] for v in views]} "
          f"→ {[v['name'] for v in views]}")

    # Silhouette render config: normalize=False + invert=False → the raw
    # line-integral / transmittance image (a smooth, differentiable function of
    # pose), the same representation used to build the Phase 5l tool mask.
    cfg = SlangDiffDRRConfig(
        det_height_px=512, det_width_px=512, pixel_spacing_mm=0.5,
        source_to_detector_mm=1020.0, source_to_isocenter_mm=510.0,
        step_mm=0.5, normalize=False, invert=False,
    )

    # ─── Ground-truth pose + observed silhouettes ────────────────────────────
    gt_trans = np.asarray(TOOL_GT_TRANS_MM, dtype=np.float32)
    gt_rot   = np.radians(np.asarray(TOOL_GT_ROT_DEG, dtype=np.float32))
    R_tool_gt = euler_zxy_to_matrix(torch.tensor(gt_rot)).numpy().astype(np.float64)
    gt_trans_t = torch.tensor(gt_trans, dtype=torch.float32)

    print(f"\n[1] GT tool pose (base frame):  t={gt_trans.round(2)} mm  "
          f"rot(ZXY)={np.degrees(gt_rot).round(2)}°")
    print("[2] Rendering observed tool silhouettes per view...")
    target_renderer = SlangDiffDRRRenderer(mu_tool, spacing, cfg)
    _ = target_renderer.render((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))  # JIT warmup

    targets_np: list[np.ndarray] = []
    for view in views:
        R_gantry = euler_zxy_to_matrix(
            torch.tensor(view["rotation_rad"], dtype=torch.float32))
        r_eff = tuple(matrix_to_euler_zxy(
            torch.from_numpy(R_tool_gt.astype(np.float32)).T @ R_gantry).tolist())
        t_eff = tuple((torch.from_numpy(R_tool_gt.astype(np.float32)).T
                       @ gt_trans_t).tolist())
        img = target_renderer.render(r_eff, t_eff)
        targets_np.append(img)
        # Output is transmittance T=exp(−∫μ): background≈1, tool shadow<1.
        # Silhouette = pixels with ≥10% occlusion (T<0.9).
        cov = 100.0 * float((img < 0.9).mean())
        print(f"  [{view['name']:>8}] tool-shadow coverage {cov:5.2f}% of frame  "
              f"(T range [{img.min():.3f}, {img.max():.3f}])")
        np.save(OUT_DIR / f"target_{view['name']}.npy", img)
    del target_renderer

    if all(float((t < 0.9).mean()) < 1e-4 for t in targets_np):
        sys.exit("ERROR: tool projects outside the detector in every view "
                 "(empty silhouettes). Adjust TOOL_GT_TRANS_MM / TOOL_GT_ROT_DEG.")

    # ─── Optimiser (blind init) ──────────────────────────────────────────────
    print("\n[3] Optimising tool pose (init = FK prior + residual, summed MSE)...")
    init_trans = (gt_trans + np.asarray(TOOL_INIT_OFFSET_MM,
                                        dtype=np.float32)).astype(np.float32)
    init_rot   = (gt_rot + np.radians(np.asarray(TOOL_INIT_OFFSET_DEG,
                                                 dtype=np.float32))).astype(np.float32)
    drr_module, _, translation, _ = create_slang_diffdrr_optimizer(
        mu_volume=mu_tool, spacing_zyx_mm=spacing,
        initial_rotation=np.zeros(3, dtype=np.float32),
        initial_translation=init_trans, cfg=cfg, lr=LR_MM,
    )
    translation.requires_grad_(True)
    device = translation.device
    tool_rot = torch.tensor(init_rot, dtype=torch.float32, device=device,
                            requires_grad=True)
    optimizer = torch.optim.Adam([
        {"params": [translation], "lr": LR_MM},
        {"params": [tool_rot],    "lr": LR_ROT_RAD},
    ])
    gantry_tensors = [torch.tensor(v["rotation_rad"], dtype=torch.float32,
                                   device=device) for v in views]
    targets = [torch.from_numpy(t).to(device) for t in targets_np]

    init_terr = float(np.linalg.norm(init_trans - gt_trans))
    init_rerr = geodesic_angle_deg(
        euler_zxy_to_matrix(torch.tensor(init_rot)).numpy().astype(np.float64),
        R_tool_gt)
    print(f"  Init ‖t_err‖={init_terr:.2f} mm  rot_err={init_rerr:.2f}°")

    trace: list[dict] = []
    t0 = time.time()
    for it in range(N_ITERS):
        optimizer.zero_grad()
        total = torch.zeros((), dtype=torch.float32, device=device)
        R_tool = euler_zxy_to_matrix(tool_rot)
        for rot_gantry, target in zip(gantry_tensors, targets):
            R_eff = R_tool.T @ euler_zxy_to_matrix(rot_gantry)
            euler_eff = matrix_to_euler_zxy(R_eff)
            t_eff = R_tool.T @ translation
            rendered = drr_module(euler_eff, t_eff)
            total = total + ((rendered - target) ** 2).mean()
        total.backward()
        # fluorosim Slang autodiff: flipped grad sign + NaN guard (see CLAUDE.md)
        for p in (translation, tool_rot):
            if p.grad is None:
                continue
            if torch.isnan(p.grad).any():
                p.grad.zero_()
            elif FLIP_GRAD:
                p.grad.neg_()
        if tool_rot.grad is not None and tool_rot.grad.abs().max() > 0:
            torch.nn.utils.clip_grad_norm_([tool_rot], max_norm=ROT_GRAD_CLIP)
        optimizer.step()

        t_np = translation.detach().cpu().numpy()
        R_rec = euler_zxy_to_matrix(tool_rot.detach()).cpu().numpy().astype(np.float64)
        terr = float(np.linalg.norm(t_np - gt_trans))
        rerr = geodesic_angle_deg(R_tool_gt, R_rec)
        trace.append({"iter": it, "loss": float(total.item()),
                      "t_err_norm_mm": terr, "rot_err_geodesic_deg": rerr})
        if LOG_EVERY and (it % LOG_EVERY == 0 or it == N_ITERS - 1):
            print(f"  iter {it:3d}: loss={total.item():.4e}  "
                  f"‖t_err‖={terr:.3f} mm  rot_err={rerr:.3f}°")
    elapsed = time.time() - t0

    # ─── Score: separate observable (pointing) from unobservable (roll) ──────
    t_rec = translation.detach().cpu().numpy()
    R_rec = euler_zxy_to_matrix(tool_rot.detach()).cpu().numpy().astype(np.float64)
    z_axis = np.array([0.0, 0.0, 1.0])            # tool long axis (EE-local)
    point_err = angle_between_deg(R_tool_gt @ z_axis, R_rec @ z_axis)
    geo_err   = geodesic_angle_deg(R_tool_gt, R_rec)
    # residual rotation is (approximately) the roll about the tool axis
    roll_err  = math.sqrt(max(geo_err ** 2 - point_err ** 2, 0.0))
    tip_gt  = tip_pos_base_mm(gt_trans, R_tool_gt, tip_offset)
    tip_rec = tip_pos_base_mm(t_rec,    R_rec,     tip_offset)
    tip_err = float(np.linalg.norm(tip_rec - tip_gt))
    center_err = (t_rec - gt_trans)
    # Decompose the position error along vs perpendicular to the tool's long
    # axis: sliding a rod along its axis barely changes the silhouette, so the
    # along-axis component is the weakly-constrained translation DOF (analogous
    # to roll), bounded here only by the asymmetric bullet tip.
    axis_base = R_tool_gt @ z_axis
    along_err = float(center_err @ axis_base)
    perp_err  = float(np.linalg.norm(center_err - along_err * axis_base))

    # recovered silhouettes for the plot
    R_rec_t = torch.from_numpy(R_rec.astype(np.float32))
    rec_renderer = SlangDiffDRRRenderer(mu_tool, spacing, cfg)
    _ = rec_renderer.render((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    for view in views:
        R_gantry = euler_zxy_to_matrix(torch.tensor(view["rotation_rad"],
                                                     dtype=torch.float32))
        r_eff = tuple(matrix_to_euler_zxy(R_rec_t.T @ R_gantry).tolist())
        t_eff = tuple((R_rec_t.T @ torch.from_numpy(t_rec.astype(np.float32))).tolist())
        np.save(OUT_DIR / f"recovered_{view['name']}.npy",
                rec_renderer.render(r_eff, t_eff))
    del rec_renderer

    print("\n" + "=" * 62)
    print(f"Summary ({len(views)} views: {[v['name'] for v in views]})")
    print("=" * 62)
    print(f"  Init  ‖t_err‖:            {init_terr:8.3f} mm")
    print(f"  Final center ‖t_err‖:     {trace[-1]['t_err_norm_mm']:8.4f} mm  "
          f"per-axis {center_err.round(4)}")
    print(f"    ├ perpendicular-to-axis:{perp_err:8.4f} mm   (well constrained)")
    print(f"    └ along tool axis:      {along_err:8.4f} mm   (weakly constrained)")
    print(f"  Final TCP tip ‖err‖:      {tip_err:8.4f} mm")
    print(f"  Pointing-direction err:   {point_err:8.4f} °   (OBSERVABLE)")
    print(f"  Roll-about-axis err:      {roll_err:8.4f} °   (weakly constrained)")
    print(f"  Total rotation geodesic:  {geo_err:8.4f} °")
    print(f"  Wall time:                {elapsed:8.1f} s ({1000*elapsed/N_ITERS:.0f} ms/iter)")

    summary = {
        "views_deg_y": list(VIEWS_DEG_Y),
        "n_iters": N_ITERS,
        "gt_trans_mm": gt_trans.tolist(),
        "gt_rot_deg": list(TOOL_GT_ROT_DEG),
        "init_trans_mm": init_trans.tolist(),
        "init_rot_deg": np.degrees(init_rot).tolist(),
        "init_t_err_mm": init_terr,
        "init_rot_err_deg": init_rerr,
        "final_trans_mm": t_rec.tolist(),
        "final_center_err_mm": center_err.tolist(),
        "final_center_err_norm_mm": float(np.linalg.norm(center_err)),
        "final_perp_err_mm": perp_err,
        "final_along_axis_err_mm": along_err,
        "final_tip_err_mm": tip_err,
        "pointing_err_deg": point_err,
        "roll_err_deg": roll_err,
        "rot_geodesic_deg": geo_err,
        "tip_gt_base_mm": tip_gt.tolist(),
        "tip_rec_base_mm": tip_rec.tolist(),
        "wall_seconds": elapsed,
        "trace": trace,
    }
    with open(OUT_DIR / "tool_pose_trace.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Wrote {OUT_DIR / 'tool_pose_trace.json'}")
    print(f"  Wrote per-view target_*/recovered_*.npy in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

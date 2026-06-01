#!/usr/bin/env python3
"""Plot multi-view registration results.

Reads ~/isaac_projects/output/registration_multiview/{registration_trace.json,
target_*.npy, recovered_*.npy} and writes:

    convergence.png  - total loss + per-view loss + per-axis error trajectory
    images.png       - one row per view: target | recovered | |diff|

Usage:
    python3 ~/isaac_projects/bridge/plot_registration_multiview.py [--show]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REG_DIR = Path(os.path.expanduser("~/isaac_projects/output/registration_multiview"))
TRACE_PATH = REG_DIR / "registration_trace.json"
CONVERGENCE_PNG = REG_DIR / "convergence.png"
IMAGES_PNG = REG_DIR / "images.png"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true", help="Open interactive viewer")
    args = parser.parse_args()

    if not TRACE_PATH.exists():
        print(f"ERROR: {TRACE_PATH} not found. Run run_register_multiview.sh first.",
              file=sys.stderr)
        return 1

    with open(TRACE_PATH) as f:
        summary = json.load(f)

    views = summary["views"]
    trace = summary["trace"]
    iters = np.array([e["iter"] for e in trace])
    loss_total = np.array([e["loss_total"] for e in trace])
    loss_per_view = np.array([e["loss_per_view"] for e in trace])  # (N_iters, N_views)
    err = np.array([e["err_mm"] for e in trace])
    err_norm = np.array([e["err_norm_mm"] for e in trace])
    # Rotation error — present in 6-DOF traces, absent in older 3-DOF traces
    has_rot = "rot_err_euler_deg" in trace[0]
    if has_rot:
        rot_err = np.array([e["rot_err_euler_deg"] for e in trace])
        rot_geo = np.array([e["rot_err_geodesic_deg"] for e in trace])

    print("=" * 60)
    print(f"Multi-view registration summary  ({len(views)} views)")
    print("=" * 60)
    print(f"  Views:                "
          f"{[(v['name'], v['angle_deg_y']) for v in views]}")
    print(f"  Ground truth (mm):    {summary['gt_translation_mm']}")
    print(f"  Init pose (mm):       {summary['init_translation_mm']}")
    print(f"  Final pose (mm):      "
          f"{[round(v, 4) for v in summary['final_translation_mm']]}")
    print(f"  Init err norm (mm):   {summary['init_err_norm_mm']:.3f}")
    print(f"  Final err norm (mm):  {summary['final_err_norm_mm']:.4f}")
    print(f"  Per-axis final (mm):  "
          f"{[round(v, 4) for v in summary['final_err_mm']]}")
    print(f"  Total loss reduction: "
          f"{loss_total[0]:.4e} -> {loss_total[-1]:.4e}")
    print(f"  Wall time:            {summary['wall_seconds']:.1f} s "
          f"({summary['wall_seconds']/summary['n_iters']*1000:.0f} ms/iter)")

    if has_rot:
        print(f"  Init rot  offset (deg): {summary.get('init_rot_deg', 'n/a')}")
        print(f"  Final rot ||err|| (°):  "
              f"{summary.get('final_rot_err_geodesic_deg', 'n/a'):.4f} (geodesic)")
        print(f"  Per-axis rot err  (°):  "
              f"{[round(v, 4) for v in summary.get('final_rot_err_euler_deg', [])]}")

    ig = summary.get("isaac_ground_truth")
    if ig is not None:
        print()
        print("Isaac Sim world-frame recovery:")
        print(f"  GT     phantom_pos (m): {ig['phantom_pos_world_m']}")
        print(f"  Recov. phantom_pos (m): "
              f"{[round(v, 5) for v in ig['phantom_pos_recovered_world_m']]}")
        print(f"  Per-axis world err (mm): "
              f"{[round(v, 4) for v in ig['world_err_mm']]}")
        print(f"  World ||err|| (mm):      {ig['world_err_norm_mm']:.4f}")
        if "rot_err_geodesic_deg" in ig:
            print(f"  Rotation ||err|| (°):    {ig['rot_err_geodesic_deg']:.4f}")

    try:
        import matplotlib
        if not args.show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not available — text summary only)")
        return 0

    # --- Figure 1: convergence ---------------------------------------------
    n_panels = 4 if has_rot else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.5))
    ax_loss, ax_per_view, ax_err = axes[:3]
    ax_rot = axes[3] if has_rot else None

    ax_loss.semilogy(iters, loss_total, color="black", label="total")
    ax_loss.set_xlabel("iteration")
    ax_loss.set_ylabel("MSE loss (log)")
    ax_loss.set_title("Total loss")
    ax_loss.grid(True, which="both", alpha=0.3)

    for vi, view in enumerate(views):
        ax_per_view.semilogy(iters, loss_per_view[:, vi], label=view["name"])
    ax_per_view.set_xlabel("iteration")
    ax_per_view.set_ylabel("per-view MSE (log)")
    ax_per_view.set_title("Per-view loss")
    ax_per_view.legend(loc="best", fontsize=9)
    ax_per_view.grid(True, which="both", alpha=0.3)

    for i, label in enumerate("xyz"):
        ax_err.plot(iters, err[:, i], label=f"t_err_{label}")
    ax_err.plot(iters, err_norm, "k--", lw=1.2, label="||t_err||")
    ax_err.axhline(0, color="gray", lw=0.5)
    ax_err.set_xlabel("iteration")
    ax_err.set_ylabel("translation error (mm)")
    ax_err.set_title("Translation error vs GT")
    ax_err.legend(loc="best", fontsize=9)
    ax_err.grid(True, alpha=0.3)

    if ax_rot is not None:
        for i, label in enumerate("xyz"):
            ax_rot.plot(iters, rot_err[:, i], label=f"r_err_{label}")
        ax_rot.plot(iters, rot_geo, "k--", lw=1.2, label="geodesic")
        ax_rot.axhline(0, color="gray", lw=0.5)
        ax_rot.set_xlabel("iteration")
        ax_rot.set_ylabel("rotation error (°)")
        ax_rot.set_title("Rotation error vs GT")
        ax_rot.legend(loc="best", fontsize=9)
        ax_rot.grid(True, alpha=0.3)

    suptitle = (
        f"Multi-view registration  |  views: {[v['name'] for v in views]}  |  "
        f"init t-offset = {summary['init_offset_mm']} mm  |  "
        f"final ||t_err|| = {summary['final_err_norm_mm']:.3f} mm"
    )
    if has_rot:
        suptitle += (f"  |  final ||r_err|| = "
                     f"{summary.get('final_rot_err_geodesic_deg', 0):.3f}°")
    if ig is not None:
        suptitle += f"  |  world err = {ig['world_err_norm_mm']:.3f} mm"
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    fig.savefig(CONVERGENCE_PNG, dpi=120, bbox_inches="tight")
    print(f"\nSaved {CONVERGENCE_PNG}")

    # --- Figure 2: per-view target / recovered / diff ----------------------
    def _window(img: np.ndarray) -> np.ndarray:
        """Log-compressed display for normalize=True, invert=True DRRs.

        fluorosim renders with invert=True so air=1 (bright) and bone=0
        (dark).  A plain gamma or linear stretch "overexposes" soft tissue
        because gamma < 1 brightens the mid-range.

        Instead we:
          1. Undo the invert  → bone becomes high, air becomes low
          2. Apply log1p      → matches real X-ray film (logarithmic response)
          3. p2–p98 clip      → stretch the tissue range to [0,1]

        Result: bone=bright, air=dark — same appearance as the debug DRRs
        rendered with normalize=False.
        """
        proxy = 1.0 - np.clip(img, 0.0, 1.0)       # undo invert: bone→high
        log   = np.log1p(proxy * 50.0)               # log compression
        p2, p98 = np.percentile(log, 2), np.percentile(log, 98)
        clipped = np.clip(log, p2, p98)
        return (clipped - p2) / max(p98 - p2, 1e-9)

    tool_in_target = bool(summary.get("tool_in_target", False))
    n_cols = 4 if tool_in_target else 3
    fig2, axes2 = plt.subplots(len(views), n_cols,
                                figsize=(4.3 * n_cols, 4.5 * len(views)),
                                squeeze=False)
    for vi, view in enumerate(views):
        target = np.load(REG_DIR / f"target_{view['name']}.npy")
        recovered = np.load(REG_DIR / f"recovered_{view['name']}.npy")
        diff = np.abs(target - recovered)

        # When the tool is painted into BOTH target and recovered, the diff
        # in the tool region collapses at convergence (tool position recovered
        # ≈ GT). The masked-out region was excluded from the loss but the
        # |diff| image still shows it for visual confirmation.
        t_disp = _window(target)
        r_disp = _window(recovered)
        panels = [(t_disp,    "target (with tool)" if tool_in_target else "target",
                              "gray"),
                  (r_disp,    "recovered (with tool)" if tool_in_target else "recovered",
                              "gray"),
                  (diff,      f"|diff|  max={diff.max():.4f}", "magma")]
        if tool_in_target:
            mask = np.load(REG_DIR / f"mask_{view['name']}.npy")
            # Overlay tool mask outline on the target for clarity.
            overlay = np.stack([t_disp, t_disp, t_disp], axis=-1)
            overlay[..., 0] = np.where(mask > 0.5, 1.0, overlay[..., 0])  # red tint
            overlay[..., 1] = np.where(mask > 0.5, overlay[..., 1] * 0.3,
                                       overlay[..., 1])
            overlay[..., 2] = np.where(mask > 0.5, overlay[..., 2] * 0.3,
                                       overlay[..., 2])
            panels.append((overlay,
                           f"tool mask  ({100 * mask.mean():.1f}%)",
                           None))
        for ax, (img, label, cmap) in zip(axes2[vi], panels):
            if cmap is None:  # RGB overlay
                im = ax.imshow(img)
            else:
                vmin = 0
                vmax = img.max() if cmap == "magma" else 1.0
                im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
                fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"{view['name']} ({view['angle_deg_y']:+.0f}°) — {label}",
                         fontsize=10)
            ax.axis("off")
    fig2.tight_layout()
    fig2.savefig(IMAGES_PNG, dpi=120, bbox_inches="tight")
    print(f"Saved {IMAGES_PNG}")

    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())

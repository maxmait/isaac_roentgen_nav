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

    try:
        import matplotlib
        if not args.show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not available — text summary only)")
        return 0

    # --- Figure 1: convergence ---------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    ax_loss, ax_per_view, ax_err = axes

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
        ax_err.plot(iters, err[:, i], label=f"err_{label}")
    ax_err.plot(iters, err_norm, "k--", lw=1.2, label="||err||")
    ax_err.axhline(0, color="gray", lw=0.5)
    ax_err.set_xlabel("iteration")
    ax_err.set_ylabel("translation error (mm)")
    ax_err.set_title("Per-axis error vs ground truth")
    ax_err.legend(loc="best", fontsize=9)
    ax_err.grid(True, alpha=0.3)

    suptitle = (
        f"Multi-view registration  |  views: {[v['name'] for v in views]}  |  "
        f"init offset = {summary['init_offset_mm']} mm  |  "
        f"final ||err|| = {summary['final_err_norm_mm']:.3f} mm"
    )
    if ig is not None:
        suptitle += f"  |  world err = {ig['world_err_norm_mm']:.3f} mm"
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    fig.savefig(CONVERGENCE_PNG, dpi=120, bbox_inches="tight")
    print(f"\nSaved {CONVERGENCE_PNG}")

    # --- Figure 2: per-view target / recovered / diff ----------------------
    fig2, axes2 = plt.subplots(len(views), 3, figsize=(13, 4.5 * len(views)),
                                squeeze=False)
    for vi, view in enumerate(views):
        target = np.load(REG_DIR / f"target_{view['name']}.npy")
        recovered = np.load(REG_DIR / f"recovered_{view['name']}.npy")
        diff = np.abs(target - recovered)
        vmax_img = max(target.max(), recovered.max())
        panels = [(target, "target", "gray"),
                  (recovered, "recovered", "gray"),
                  (diff, f"|diff|  max={diff.max():.4f}", "magma")]
        for ax, (img, label, cmap) in zip(axes2[vi], panels):
            vmax = vmax_img if cmap == "gray" else diff.max()
            im = ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax)
            ax.set_title(f"{view['name']} ({view['angle_deg_y']:+.0f}°) — {label}",
                         fontsize=10)
            ax.axis("off")
            fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig2.tight_layout()
    fig2.savefig(IMAGES_PNG, dpi=120, bbox_inches="tight")
    print(f"Saved {IMAGES_PNG}")

    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())

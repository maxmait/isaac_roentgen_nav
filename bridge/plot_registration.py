#!/usr/bin/env python3
"""Plot the convergence behavior of register_phantom.py.

Reads ~/isaac_projects/output/registration/{registration_trace.json,
target.npy, recovered.npy} and writes two figures:

    convergence.png   - loss vs iter (log) + per-axis translation error vs iter
    images.png        - target / recovered / abs(diff) side-by-side

Also prints a text summary so this is useful headless.

Usage:
    python3 ~/isaac_projects/bridge/plot_registration.py [--show]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REG_DIR = Path(os.path.expanduser("~/isaac_projects/output/registration"))
TRACE_PATH = REG_DIR / "registration_trace.json"
TARGET_NPY = REG_DIR / "target.npy"
RECOVERED_NPY = REG_DIR / "recovered.npy"
CONVERGENCE_PNG = REG_DIR / "convergence.png"
IMAGES_PNG = REG_DIR / "images.png"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true", help="Open interactive viewer")
    args = parser.parse_args()

    if not TRACE_PATH.exists():
        print(f"ERROR: {TRACE_PATH} not found. Run run_register.sh first.", file=sys.stderr)
        return 1

    with open(TRACE_PATH) as f:
        summary = json.load(f)
    target = np.load(TARGET_NPY)
    recovered = np.load(RECOVERED_NPY)

    trace = summary["trace"]
    iters = np.array([e["iter"] for e in trace])
    losses = np.array([e["loss"] for e in trace])
    err = np.array([e["err_mm"] for e in trace])           # (N, 3)
    err_norm = np.array([e["err_norm_mm"] for e in trace])

    print("=" * 60)
    print("Registration summary  (fluorosim translation_mm space)")
    print("=" * 60)
    print(f"  Ground truth (mm):    {summary['gt_translation_mm']}")
    print(f"  Initial pose (mm):    {summary['init_translation_mm']}")
    print(f"  Final pose (mm):      {summary['final_translation_mm']}")
    print(f"  Initial error norm:   {summary['init_err_norm_mm']:.3f} mm")
    print(f"  Final   error norm:   {summary['final_err_norm_mm']:.4f} mm")
    print(f"  Loss reduction:       {losses[0]:.4e} -> {losses[-1]:.4e}")
    print(f"  Wall time:            {summary['wall_seconds']:.1f} s "
          f"({summary['wall_seconds']/summary['n_iters']*1000:.0f} ms/iter)")

    ig = summary.get("isaac_ground_truth")
    if ig is not None:
        print()
        print("=" * 60)
        print("Registration summary  (Isaac Sim world frame)")
        print("=" * 60)
        print(f"  GT     phantom_pos (m):  {ig['phantom_pos_world_m']}")
        print(f"  Init   phantom_pos (m):  "
              f"{[round(v, 5) for v in ig['phantom_pos_init_world_m']]}")
        print(f"  Recov. phantom_pos (m):  "
              f"{[round(v, 5) for v in ig['phantom_pos_recovered_world_m']]}")
        print(f"  Per-axis err (mm):       "
              f"{[round(v, 4) for v in ig['world_err_mm']]}")
        print(f"  ||err|| (mm):            {ig['world_err_norm_mm']:.4f}")

    try:
        import matplotlib
        if not args.show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not available — text summary only)")
        return 0

    # --- Figure 1: convergence ---------------------------------------------
    fig, (ax_loss, ax_err) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_loss.semilogy(iters, losses, color="tab:blue")
    ax_loss.set_xlabel("iteration")
    ax_loss.set_ylabel("MSE loss (log)")
    ax_loss.set_title("Loss")
    ax_loss.grid(True, which="both", alpha=0.3)

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
        f"Phantom translation registration  |  "
        f"init offset = {summary['init_offset_mm']} mm  |  "
        f"final ||err|| = {summary['final_err_norm_mm']:.3f} mm"
    )
    if ig is not None:
        suptitle += (
            f"  |  world-frame ||err|| = {ig['world_err_norm_mm']:.3f} mm "
            f"vs Isaac Sim GT"
        )
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    fig.savefig(CONVERGENCE_PNG, dpi=120, bbox_inches="tight")
    print(f"\nSaved {CONVERGENCE_PNG}")

    # --- Figure 2: target / recovered / diff -------------------------------
    diff = np.abs(target - recovered)
    vmax = max(target.max(), recovered.max())
    fig2, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, img, title in zip(
        axes,
        [target, recovered, diff],
        ["target (GT pose)", "recovered (final pose)",
         f"|target - recovered|  (max={diff.max():.4f})"],
    ):
        im = ax.imshow(img, cmap="gray" if "diff" not in title else "magma",
                       vmin=0, vmax=vmax if "diff" not in title else diff.max())
        ax.set_title(title, fontsize=10)
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

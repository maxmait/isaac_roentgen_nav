#!/usr/bin/env python3
"""Visualize the latest DRR alongside the robot/C-arm pose used to render it.

Reads ~/isaac_projects/output/{drr.png, drr_meta.json} and saves a side-by-side
annotated figure to ~/isaac_projects/output/drr_annotated.png. Also prints a
text summary so it's useful headless.

Usage:
    python3 ~/isaac_projects/bridge/visualize_drr.py [--show]

--show pops up an interactive matplotlib window (requires a display).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

OUT_DIR = Path(os.path.expanduser("~/isaac_projects/output"))
DRR_NPY = OUT_DIR / "drr.npy"
DRR_PNG = OUT_DIR / "drr.png"
DRR_META = OUT_DIR / "drr_meta.json"
ANNOTATED = OUT_DIR / "drr_annotated.png"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true", help="Open interactive viewer")
    args = parser.parse_args()

    if not DRR_META.exists():
        print(f"ERROR: {DRR_META} not found. Run run_fluorosim.sh first.", file=sys.stderr)
        return 1

    with open(DRR_META) as f:
        meta = json.load(f)

    # Prefer raw NPY (full-precision) for display; fall back to PNG.
    if DRR_NPY.exists():
        img = np.load(DRR_NPY)
        src = DRR_NPY.name
    else:
        from PIL import Image
        img = np.array(Image.open(DRR_PNG))
        src = DRR_PNG.name

    print("=" * 60)
    print("DRR Visualization Summary")
    print("=" * 60)
    print(f"DRR source:       {src}  shape={img.shape}  dtype={img.dtype}")
    print(f"Pose timestamp:   {meta['source_pose_timestamp']}")
    print()
    print("Robot (Isaac Sim world frame):")
    print(f"  EE pos (m):         {meta['ee_pos']}")
    print(f"  EE quat (wxyz):     {meta['ee_quat']}")
    print()
    print("Phantom isocenter (Isaac Sim world frame):")
    print(f"  phantom pos (m):    {meta.get('phantom_pos_m')}")
    print(f"  phantom quat (wxyz):{meta.get('phantom_quat_wxyz')}")
    print()
    print("C-arm (Isaac Sim world frame -> volume-local -> fluorosim input):")
    print(f"  carm pos (m):       {meta['carm_pos_m']}")
    print(f"  carm quat (wxyz):   {meta['carm_quat_wxyz']}")
    print(f"  carm local (m):     {meta.get('carm_local_m')}  (= carm - phantom)")
    print(f"  -> rotation (rad):     {meta['fluorosim_rotation_rad']}")
    print(f"  -> translation (mm):   {meta['fluorosim_translation_mm']}")
    print()
    tool = meta.get("tool")
    if tool is not None:
        print("Tool (painted into μ-volume at EE position):")
        print(f"  voxel (z, y, x):    "
              f"({tool['ee_voxel_zyx'][0]:.1f}, "
              f"{tool['ee_voxel_zyx'][1]:.1f}, "
              f"{tool['ee_voxel_zyx'][2]:.1f})")
        print(f"  radius:             {tool['tool_radius_mm']} mm  "
              f"(μ = {tool['tool_mu_per_mm']} mm⁻¹)")
        print(f"  voxels painted:     {tool['voxels_painted']}  "
              f"(fully inside volume: {tool['fully_inside_volume']})")
        if tool["voxels_painted"] == 0:
            print("  WARNING: no voxels were painted — tool is outside the volume.")
        print()
    stats = meta["image"]
    print(f"Image stats:  min={stats['min']:.4f}  max={stats['max']:.4f}  "
          f"mean={stats['mean']:.4f}  std={stats['std']:.4f}")
    print()

    # Sanity check: a useful DRR has nonzero variance (volume actually
    # intersected the rays). A flat image means the C-arm pose missed the volume.
    if stats["std"] < 1e-4:
        print("WARNING: DRR is nearly flat (std < 1e-4).")
        print("         C-arm pose may have missed the volume.")
    else:
        print("OK: DRR has nonzero variance — volume was hit by rays.")

    try:
        import matplotlib
        if not args.show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not available — text summary only)")
        return 0

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img, cmap="gray")
    ax.set_title("DRR  (synthetic sphere phantom)")
    ax.axis("off")

    annot = (
        f"timestamp: {meta['source_pose_timestamp']:.2f}\n"
        f"EE pos (m): "
        f"[{meta['ee_pos'][0]:.3f}, {meta['ee_pos'][1]:.3f}, {meta['ee_pos'][2]:.3f}]\n"
        f"C-arm rot (deg): "
        f"[{np.degrees(meta['fluorosim_rotation_rad'][0]):.1f}, "
        f"{np.degrees(meta['fluorosim_rotation_rad'][1]):.1f}, "
        f"{np.degrees(meta['fluorosim_rotation_rad'][2]):.1f}]\n"
        f"C-arm trans (mm): "
        f"[{meta['fluorosim_translation_mm'][0]:.1f}, "
        f"{meta['fluorosim_translation_mm'][1]:.1f}, "
        f"{meta['fluorosim_translation_mm'][2]:.1f}]"
    )
    fig.text(
        0.02, 0.02, annot,
        fontsize=9, family="monospace",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray"),
    )

    fig.tight_layout()
    fig.savefig(ANNOTATED, dpi=120, bbox_inches="tight")
    print(f"\nSaved annotated figure: {ANNOTATED}")

    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())

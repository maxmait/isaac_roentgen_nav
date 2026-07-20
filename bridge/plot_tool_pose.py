#!/usr/bin/env python3
"""Plot the image-based tool-pose recovery (register_tool_pose.py).

Reads output/tool_pose/{tool_pose_trace.json, target_*.npy, recovered_*.npy}
and renders:

  * one silhouette-overlay panel per C-arm view — the observed tool shadow
    (magenta) vs the recovered tool shadow (green); where they agree the
    overlap reads white/grey.
  * a convergence panel — loss (log) + translation/rotation error vs iteration.

Run on the HOST (needs matplotlib).  Output: output/tool_pose/tool_pose.png

  python3 ~/isaac_projects/bridge/plot_tool_pose.py
  python3 ~/isaac_projects/bridge/plot_tool_pose.py --show
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(os.path.expanduser("~/isaac_projects/output/tool_pose"))
TRACE = OUT_DIR / "tool_pose_trace.json"
PNG = OUT_DIR / "tool_pose.png"


def view_name(a: float) -> str:
    if abs(a) < 1e-3:
        return "AP"
    if abs(a - 90.0) < 1e-3:
        return "lateral"
    return f"ry{a:+.0f}deg"


def occlusion(path: Path) -> np.ndarray:
    """Load a transmittance silhouette and return occlusion 1−T in [0,1]."""
    t = np.load(path)
    return np.clip(1.0 - t, 0.0, 1.0)


def main() -> int:
    if not TRACE.exists():
        raise SystemExit(f"{TRACE} not found. Run bridge/run_register_tool.sh first.")
    s = json.load(open(TRACE))
    views = [view_name(a) for a in s["views_deg_y"]]
    trace = s["trace"]

    n = len(views)
    fig, axes = plt.subplots(1, n + 1, figsize=(4.2 * (n + 1), 4.4))
    if n + 1 == 1:
        axes = [axes]

    for i, vname in enumerate(views):
        tgt = OUT_DIR / f"target_{vname}.npy"
        rec = OUT_DIR / f"recovered_{vname}.npy"
        ax = axes[i]
        if tgt.exists() and rec.exists():
            occ_t = occlusion(tgt)
            occ_r = occlusion(rec)
            # magenta = observed, green = recovered, white = agreement
            rgb = np.stack([occ_t, occ_r, occ_t], axis=-1)
            rgb = rgb / max(rgb.max(), 1e-6)
            ax.imshow(rgb, origin="lower")
        ax.set_title(f"{vname}\nobserved (magenta) vs recovered (green)",
                     fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    # Convergence panel
    ax = axes[n]
    it = [t["iter"] for t in trace]
    loss = [t["loss"] for t in trace]
    terr = [t["t_err_norm_mm"] for t in trace]
    rerr = [t["rot_err_geodesic_deg"] for t in trace]
    ax.plot(it, terr, color="tab:blue", label="‖t_err‖ (mm)")
    ax.plot(it, rerr, color="tab:green", label="rot err (°)")
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("error (mm / °)")
    ax.grid(True, which="both", alpha=0.3)
    axl = ax.twinx()
    axl.plot(it, loss, color="tab:red", alpha=0.5, ls="--", label="loss")
    axl.set_yscale("log")
    axl.set_ylabel("silhouette MSE loss", color="tab:red")
    axl.tick_params(axis="y", labelcolor="tab:red")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Convergence", fontsize=10)

    tip = s["final_tip_err_mm"]
    perp = s["final_perp_err_mm"]
    along = s["final_along_axis_err_mm"]
    point = s["pointing_err_deg"]
    fig.suptitle(
        f"Image-based tool/TCP pose recovery  |  {n} view(s)  |  "
        f"TCP tip err {tip:.3f} mm  (perp {perp:.3f} / along-axis {along:.3f} mm)  |  "
        f"pointing {point:.3f}°",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(PNG, dpi=125, bbox_inches="tight")
    print(f"Wrote {PNG}")
    if "--show" in sys.argv:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

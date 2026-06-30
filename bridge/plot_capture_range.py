#!/usr/bin/env python3
"""Plot the capture-range / basin-of-attraction study.

Reads output/registration_multiview/capture_range.json (produced by
register_phantom_multiview.py with CAPTURE_RANGE=1) and renders:

  Panel 1: success rate vs initial-offset radius (the basin curve)
  Panel 2: final ‖t_err‖ vs initial ‖t_err‖ scatter (per sample),
           colour = converged / failed, with the success threshold line.

Run on the HOST (needs matplotlib).  Output: output/registration_multiview/
capture_range.png

  python3 ~/isaac_projects/bridge/plot_capture_range.py
  python3 ~/isaac_projects/bridge/plot_capture_range.py --show
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

OUT_DIR = Path(os.path.expanduser("~/isaac_projects/output/registration_multiview"))
CR_JSON = OUT_DIR / "capture_range.json"
CR_PNG = OUT_DIR / "capture_range.png"


def main() -> int:
    if not CR_JSON.exists():
        raise SystemExit(
            f"{CR_JSON} not found. Run the study first:\n"
            f"  CAPTURE_RANGE=1 DICOM_PATH=... USE_POSE_JSON=1 "
            f"~/isaac_projects/bridge/run_register_multiview.sh"
        )
    cr = json.load(open(CR_JSON))
    cfg = cr["config"]
    per_radius = cr["per_radius"]
    samples = cr["samples"]

    radii = [p["radius_mm"] for p in per_radius]
    rates = [100.0 * p["success_rate"] for p in per_radius]
    median = [p["median_final_mm"] for p in per_radius]

    init_e = np.array([s["init_err_norm_mm"] for s in samples])
    final_e = np.array([s["final_err_norm_mm"] for s in samples])
    ok = np.array([s["success"] for s in samples], dtype=bool)
    succ_mm = cfg["success_mm"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1 — basin curve
    ax1.plot(radii, rates, "o-", color="tab:blue", label="success rate")
    ax1.set_xlabel("initial translation offset radius (mm)")
    ax1.set_ylabel("success rate (%)", color="tab:blue")
    ax1.set_ylim(-5, 105)
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)
    ax1b = ax1.twinx()
    ax1b.plot(radii, median, "s--", color="tab:red", alpha=0.7,
              label="median final ‖t_err‖")
    ax1b.set_ylabel("median final ‖t_err‖ (mm)", color="tab:red")
    ax1b.tick_params(axis="y", labelcolor="tab:red")
    noise = "ON" if cfg.get("drr_noise") else "OFF"
    ax1.set_title(f"Basin of attraction  (noise {noise}, "
                  f"{cfg['n_samples']} samples/radius, "
                  f"{cfg['n_iters']} iters)\n"
                  f"success := ‖t_err‖<{succ_mm}mm & rot<{cfg['success_deg']}°, "
                  f"views {cfg.get('views_deg_y')}")

    # Panel 2 — final vs init scatter
    ax2.scatter(init_e[ok], final_e[ok], c="tab:green", s=28,
                label=f"converged ({ok.sum()})", alpha=0.8)
    ax2.scatter(init_e[~ok], final_e[~ok], c="tab:red", s=28,
                marker="x", label=f"failed ({(~ok).sum()})", alpha=0.9)
    ax2.axhline(succ_mm, color="k", ls=":", alpha=0.6,
                label=f"success threshold ({succ_mm} mm)")
    ax2.set_xlabel("initial ‖t_err‖ (mm)")
    ax2.set_ylabel("final ‖t_err‖ (mm)")
    ax2.set_yscale("log")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(loc="upper left")
    ax2.set_title("Final vs initial translation error")

    fig.tight_layout()
    fig.savefig(CR_PNG, dpi=130)
    print(f"Wrote {CR_PNG}")
    if "--show" in sys.argv:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

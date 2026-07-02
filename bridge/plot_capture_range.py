#!/usr/bin/env python3
"""Plot the capture-range / basin-of-attraction study.

Reads capture-range JSON(s) produced by register_phantom_multiview.py with
CAPTURE_RANGE=1 and renders:

  Panel 1: success rate vs initial-offset radius (the basin curve)
  Panel 2: final ‖t_err‖ vs initial-offset radius — median line + p25–p75
           band per condition, log scale, with the success threshold.

Two modes, auto-selected:

  * COMPARISON (default when both exist): overlays
      output/registration_multiview/capture_range_clean.json  and
      output/registration_multiview/capture_range_noise.json
    so the effect of realistic fluoroscopy noise on the basin is visible.
  * SINGLE: falls back to capture_range.json (the raw last-run file).

Explicit paths may be passed as positional args (each labelled by its
filename); this overrides the auto-selection.

Run on the HOST (needs matplotlib).  Output:
  output/registration_multiview/capture_range.png

  python3 ~/isaac_projects/bridge/plot_capture_range.py
  python3 ~/isaac_projects/bridge/plot_capture_range.py --show
  python3 ~/isaac_projects/bridge/plot_capture_range.py a.json b.json
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
CR_CLEAN = OUT_DIR / "capture_range_clean.json"
CR_NOISE = OUT_DIR / "capture_range_noise.json"
CR_PNG = OUT_DIR / "capture_range.png"

# (blue, orange, green, ...) — matplotlib default cycle is fine for N series.
_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]


def _label(cr: dict, fallback: str) -> str:
    cfg = cr.get("config", {})
    if cfg.get("drr_noise"):
        return f"noise ({fallback})" if fallback not in ("noise",) else "noise"
    return "clean" if fallback in ("clean", "capture_range") else fallback


def _resolve_inputs(argv: list[str]) -> list[tuple[str, Path]]:
    paths = [a for a in argv[1:] if not a.startswith("--")]
    if paths:
        return [(Path(p).stem.replace("capture_range_", ""), Path(p)) for p in paths]
    if CR_CLEAN.exists() and CR_NOISE.exists():
        return [("clean", CR_CLEAN), ("noise", CR_NOISE)]
    if CR_JSON.exists():
        return [("capture_range", CR_JSON)]
    raise SystemExit(
        f"No capture-range JSON found in {OUT_DIR}.\n"
        f"  Run the study first with CAPTURE_RANGE=1 (see CLAUDE.md)."
    )


def _per_radius_stats(cr: dict):
    """Return (radii, success_rate%, med, p25, p75) from the samples."""
    samples = cr["samples"]
    radii = sorted({s["radius_mm"] for s in samples})
    rate, med, p25, p75 = [], [], [], []
    for r in radii:
        grp = [s for s in samples if s["radius_mm"] == r]
        fin = np.array([s["final_err_norm_mm"] for s in grp])
        rate.append(100.0 * np.mean([s["success"] for s in grp]))
        med.append(float(np.median(fin)))
        p25.append(float(np.percentile(fin, 25)))
        p75.append(float(np.percentile(fin, 75)))
    return (np.array(radii), np.array(rate), np.array(med),
            np.array(p25), np.array(p75))


def main() -> int:
    inputs = _resolve_inputs(sys.argv)
    datasets = [(_label(json.load(open(p)), tag), json.load(open(p)))
                for tag, p in inputs]

    # shared config for the title (take the first dataset's)
    cfg0 = datasets[0][1]["config"]
    succ_mm = cfg0["success_mm"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    for i, (label, cr) in enumerate(datasets):
        color = _COLORS[i % len(_COLORS)]
        radii, rate, med, p25, p75 = _per_radius_stats(cr)

        # Panel 1 — basin curve (success rate)
        ax1.plot(radii, rate, "o-", color=color, label=label, lw=2, ms=6)

        # Panel 2 — final error vs radius (median + IQR band)
        ax2.plot(radii, med, "o-", color=color, label=f"{label} (median)",
                 lw=2, ms=6)
        ax2.fill_between(radii, p25, p75, color=color, alpha=0.18,
                         label=f"{label} (p25–p75)")

    ax1.set_xlabel("initial translation offset radius (mm)")
    ax1.set_ylabel("success rate (%)")
    ax1.set_ylim(-5, 105)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower left")
    ax1.set_title(
        f"Basin of attraction  "
        f"({cfg0['n_samples']} samples/radius, {cfg0['n_iters']} iters)\n"
        f"success := ‖t_err‖<{succ_mm}mm & rot<{cfg0['success_deg']}°, "
        f"views {cfg0.get('views_deg_y')}"
    )

    ax2.axhline(succ_mm, color="k", ls=":", alpha=0.6,
                label=f"success threshold ({succ_mm} mm)")
    ax2.set_xlabel("initial translation offset radius (mm)")
    ax2.set_ylabel("final ‖t_err‖ (mm)")
    ax2.set_yscale("log")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(loc="upper left", fontsize=8)
    ax2.set_title("Final translation error vs initial offset")

    fig.tight_layout()
    fig.savefig(CR_PNG, dpi=130)
    print(f"Wrote {CR_PNG}  ({', '.join(l for l, _ in datasets)})")
    if "--show" in sys.argv:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

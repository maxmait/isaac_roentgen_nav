#!/usr/bin/env python3
"""Compute T_robot_to_anatomy (and friends) from DRR-registered phantom pose.

Combines two existing data sources:

  1. ~/isaac_projects/output/pose.json — Isaac Sim ground truth
     (ee_pos, ee_quat, carm_pos, carm_quat, phantom_pos, phantom_quat)
  2. ~/isaac_projects/output/registration_multiview/registration_trace.json
     (or the single-view fallback) — phantom pose RECOVERED via DRR registration

Computes the three clinically-relevant transforms in both their ground-truth
and recovered forms:

    T_R^A  robot end-effector in anatomy frame
    T_R^C  robot end-effector in C-arm frame
    T_A^C  anatomy in C-arm frame  (= what the registration directly recovers,
                                     just expressed in the C-arm frame)

6-DOF: both phantom translation AND rotation are recovered from the trace.
If the trace predates the 6-DOF upgrade (no phantom_quat_recovered_wxyz key),
falls back to identity rotation.

Outputs:
    ~/isaac_projects/output/robot_to_anatomy.json
    ~/isaac_projects/output/robot_to_anatomy_layout.png

Usage:
    python3 ~/isaac_projects/bridge/compute_robot_to_anatomy.py [--show]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

OUT_DIR = Path(os.path.expanduser("~/isaac_projects/output"))
POSE_FILE = OUT_DIR / "pose.json"
TRACE_MULTIVIEW = OUT_DIR / "registration_multiview" / "registration_trace.json"
TRACE_SINGLEVIEW = OUT_DIR / "registration" / "registration_trace.json"
RESULT_JSON = OUT_DIR / "robot_to_anatomy.json"
LAYOUT_PNG = OUT_DIR / "robot_to_anatomy_layout.png"

# Make scenes/phantom.py importable for the anatomy semiaxes.
SCENES_DIR = Path(__file__).resolve().parent.parent / "scenes"
if str(SCENES_DIR) not in sys.path:
    sys.path.insert(0, str(SCENES_DIR))
from phantom import (  # noqa: E402
    BONE_CORE_SEMIAXES_M,
    SOFT_TISSUE_SEMIAXES_M,
)

# Cached CT μ-volumes — if present, use them for data-driven inside/outside
# checks (more accurate than the analytic ellipsoid heuristic).
CT_CACHE_CROPPED = OUT_DIR / "fluorosim_cache_ct"
CT_CACHE_FULL    = OUT_DIR / "fluorosim_cache_ct_full"

# μ thresholds (mm⁻¹) for the data-driven clinical check.  Matches the
# bilinear HU→μ conversion in bridge/ct_loader.py.
MU_BONE_THRESH  = 0.035   # > trabecular bone
MU_TISSUE_THRESH = 0.012  # > soft tissue (air ≈ 0.0001, water ≈ 0.020)


# ─── tiny rigid-transform helper ──────────────────────────────────────────────
@dataclass
class Pose:
    """3D rigid transform: rotation R (3x3) and translation t (3,), in meters."""
    R: np.ndarray
    t: np.ndarray

    @staticmethod
    def from_pos_quat_wxyz(pos, quat_wxyz) -> "Pose":
        w, x, y, z = (float(v) for v in quat_wxyz)
        # Unit quaternion → rotation matrix
        n = w * w + x * x + y * y + z * z
        s = 2.0 / n if n > 0 else 0.0
        R = np.array([
            [1 - s * (y * y + z * z),     s * (x * y - z * w),     s * (x * z + y * w)],
            [    s * (x * y + z * w), 1 - s * (x * x + z * z),     s * (y * z - x * w)],
            [    s * (x * z - y * w),     s * (y * z + x * w), 1 - s * (x * x + y * y)],
        ], dtype=np.float64)
        return Pose(R=R, t=np.asarray(pos, dtype=np.float64))

    def inv(self) -> "Pose":
        Rt = self.R.T
        return Pose(R=Rt, t=-Rt @ self.t)

    def __matmul__(self, other: "Pose") -> "Pose":
        return Pose(R=self.R @ other.R, t=self.R @ other.t + self.t)

    def to_quat_wxyz(self) -> list[float]:
        """3x3 rotation matrix → unit quaternion (w, x, y, z). Shepperd's method."""
        R = self.R
        tr = R[0, 0] + R[1, 1] + R[2, 2]
        if tr > 0:
            s = 0.5 / np.sqrt(tr + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        return [float(w), float(x), float(y), float(z)]

    def to_dict(self) -> dict:
        return {
            "t_mm": (self.t * 1000.0).tolist(),
            "R_quat_wxyz": self.to_quat_wxyz(),
        }


# ─── data loading ─────────────────────────────────────────────────────────────
def load_pose_json() -> dict:
    if not POSE_FILE.exists():
        sys.exit(f"ERROR: {POSE_FILE} not found. Run scenes/robot_scene.py first.")
    with open(POSE_FILE) as f:
        return json.load(f)


def load_registration_trace() -> tuple[dict, str]:
    """Prefer the multi-view trace; fall back to single-view if missing."""
    for path, mode in [(TRACE_MULTIVIEW, "multiview"),
                       (TRACE_SINGLEVIEW, "single_view")]:
        if path.exists():
            with open(path) as f:
                trace = json.load(f)
            if trace.get("isaac_ground_truth") is None:
                continue
            return trace, mode
    sys.exit(
        "ERROR: no registration trace found with isaac_ground_truth.\n"
        f"  expected one of:\n    {TRACE_MULTIVIEW}\n    {TRACE_SINGLEVIEW}\n"
        "  Run bridge/run_register_multiview.sh (or run_register.sh) first."
    )


# ─── anatomy interpretability ────────────────────────────────────────────────
def normalized_ellipsoid_distance(point_m, semiaxes_m) -> float:
    """sqrt(sum((point_i / semiaxis_i)^2)) — <1 inside, >1 outside."""
    p = np.asarray(point_m, dtype=np.float64)
    s = np.asarray(semiaxes_m, dtype=np.float64)
    return float(np.sqrt(np.sum((p / s) ** 2)))


def _ct_mu_at_anatomy_point(cache_dir: Path, ee_anatomy_m: np.ndarray) -> float | None:
    """Look up μ (mm⁻¹) at a point in anatomy frame using a cached CT volume.

    The cached volume's isocenter is at the anatomy-frame origin, so an
    anatomy-frame point in mm maps directly to a voxel index via the
    volume's spacing.  Returns None if the point is outside the volume.
    """
    mu_path   = cache_dir / "mu_volume.npy"
    meta_path = cache_dir / "metadata.json"
    if not mu_path.exists() or not meta_path.exists():
        return None
    mu      = np.load(str(mu_path), mmap_mode="r")
    meta    = json.loads(meta_path.read_text())
    spacing = np.asarray(meta["spacing_zyx_mm"], dtype=np.float64)
    shape   = np.asarray(mu.shape, dtype=np.float64)

    # Anatomy frame mm (X, Y, Z) → voxel index (Z, Y, X)
    p_mm = ee_anatomy_m * 1000.0
    p_zyx_mm = np.array([p_mm[2], p_mm[1], p_mm[0]])
    voxel = p_zyx_mm / spacing + (shape - 1) / 2.0
    idx = np.round(voxel).astype(int)
    if np.any(idx < 0) or np.any(idx >= np.asarray(mu.shape)):
        return None
    return float(mu[idx[0], idx[1], idx[2]])


def clinical_summary(ee_in_anatomy_m: np.ndarray) -> dict:
    """Data-driven anatomy check when a CT cache exists, ellipsoid fallback otherwise.

    With a CT cache: look up the local μ value at the EE position in anatomy
    frame and classify by μ thresholds (bone, soft tissue, outside).  This
    uses the same μ values fluorosim integrates along its rays — perfectly
    consistent with the registration.

    Without a CT cache: fall back to the synthetic ellipsoid check.
    """
    # Prefer cropped (matches the registration ROI); fall back to full volume.
    for cache in (CT_CACHE_CROPPED, CT_CACHE_FULL):
        mu = _ct_mu_at_anatomy_point(cache, ee_in_anatomy_m)
        if mu is not None:
            inside_bone = mu > MU_BONE_THRESH
            inside_soft = mu > MU_TISSUE_THRESH
            if inside_bone:
                interp = (f"Tool tip is inside bone "
                          f"(local μ={mu:.4f} mm⁻¹ > bone threshold {MU_BONE_THRESH}).")
            elif inside_soft:
                interp = (f"Tool tip is in soft tissue "
                          f"(local μ={mu:.4f} mm⁻¹, "
                          f"thresholds: soft {MU_TISSUE_THRESH}, bone {MU_BONE_THRESH}).")
            else:
                interp = (f"Tool tip is in air or outside the CT (local μ="
                          f"{mu:.4f} mm⁻¹ < soft-tissue threshold).")
            return {
                "method": "ct_mu_lookup",
                "ct_cache": str(cache),
                "ee_pos_in_anatomy_mm_recovered": (ee_in_anatomy_m * 1000.0).tolist(),
                "local_mu_mm_inv": mu,
                "mu_bone_thresh": MU_BONE_THRESH,
                "mu_tissue_thresh": MU_TISSUE_THRESH,
                "inside_bone_core": inside_bone,
                "inside_soft_tissue": inside_soft,
                "interpretation": interp,
            }

    # Synthetic ellipsoid fallback (no CT cache available)
    d_bone = normalized_ellipsoid_distance(ee_in_anatomy_m, BONE_CORE_SEMIAXES_M)
    d_soft = normalized_ellipsoid_distance(ee_in_anatomy_m, SOFT_TISSUE_SEMIAXES_M)
    inside_bone = d_bone < 1.0
    inside_soft = d_soft < 1.0
    if inside_bone:
        interp = (f"Tool tip is inside the bone core "
                  f"(normalized distance {d_bone:.2f}).")
    elif inside_soft:
        interp = (f"Tool tip is in soft tissue (bone-core normalized "
                  f"distance {d_bone:.2f}, soft-tissue {d_soft:.2f}).")
    else:
        interp = (f"Tool tip is OUTSIDE the phantom (bone {d_bone:.2f}, "
                  f"soft {d_soft:.2f} — both > 1).")
    return {
        "method": "ellipsoid",
        "ee_pos_in_anatomy_mm_recovered": (ee_in_anatomy_m * 1000.0).tolist(),
        "bone_core_normalized_distance": d_bone,
        "soft_tissue_normalized_distance": d_soft,
        "inside_bone_core": inside_bone,
        "inside_soft_tissue": inside_soft,
        "interpretation": interp,
    }


# ─── visualization ────────────────────────────────────────────────────────────
def _wrap(text: str, width: int) -> list[str]:
    """Wrap a line to ``width`` chars, preserving its leading indent."""
    import textwrap
    indent = text[: len(text) - len(text.lstrip())]
    return textwrap.wrap(text, width=width, subsequent_indent=indent + "  ") \
        or [text]


def draw_ellipse(ax, semi_a_mm, semi_b_mm, edgecolor, label):
    """Add an ellipse at origin with the given semi-axes (mm) to the axes."""
    from matplotlib.patches import Ellipse
    e = Ellipse(xy=(0, 0), width=2 * semi_a_mm, height=2 * semi_b_mm,
                facecolor="none", edgecolor=edgecolor, lw=1.6, label=label)
    ax.add_patch(e)


def _load_ct_slices_for_layout() -> dict | None:
    """Return mu-volume slices through the origin (anatomy isocenter), or None."""
    for cache in (CT_CACHE_CROPPED, CT_CACHE_FULL):
        mu_path = cache / "mu_volume.npy"
        meta_path = cache / "metadata.json"
        if mu_path.exists() and meta_path.exists():
            mu      = np.load(str(mu_path), mmap_mode="r")
            meta    = json.loads(meta_path.read_text())
            spacing = meta["spacing_zyx_mm"]  # (sz, sy, sx) mm
            nz, ny, nx = mu.shape
            # Anatomy origin = volume centre voxel
            z0, y0, x0 = (nz - 1) // 2, (ny - 1) // 2, (nx - 1) // 2
            return {
                "axial":    np.asarray(mu[z0, :, :]),  # XY plane, +Z out of page
                "coronal":  np.asarray(mu[:, y0, :]),  # XZ plane, +Y into page
                "sagittal": np.asarray(mu[:, :, x0]),  # YZ plane, +X out of page
                "spacing":  spacing,
                "shape":    mu.shape,
                "cache":    str(cache),
            }
    return None


def make_layout_plot(ee_anatomy_mm: np.ndarray,
                     tool_seg_anatomy_mm: np.ndarray,
                     ee_anatomy_gt_mm: np.ndarray,
                     tool_seg_gt_anatomy_mm: np.ndarray,
                     phantom_err_mm: np.ndarray,
                     err_norm_mm: float,
                     summary_lines: list[str],
                     show: bool) -> None:
    """Deliverable figure: tool pose in the recovered anatomy frame.

    Three orthogonal anatomy-frame panels (axial / coronal / sagittal) show
    the recovered robot tool as an oriented needle over the CT (or the
    synthetic ellipsoid), and a fourth text panel reports the recovered
    transforms + registration error + the clinical inside/outside check.

    tool_seg_anatomy_mm is a (2, 3) array: the tool shaft endpoints (tip and
    rear) in anatomy-frame mm, so the needle's orientation is visible, not
    just its tip position (ee_anatomy_mm).  ``*_gt_*`` are the same quantities
    for the Isaac Sim ground-truth anatomy pose, drawn as a faint overlay so
    the estimate-vs-truth agreement is visible (they coincide at sub-mm error).
    """
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not available — skipping layout plot)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    ax_xy, ax_xz = axes[0, 0], axes[0, 1]
    ax_yz, ax_txt = axes[1, 0], axes[1, 1]

    ct = _load_ct_slices_for_layout()
    if ct is not None:
        # CT-backed layout: show actual μ-volume slices through the isocenter
        sz, sy, sx = ct["spacing"]
        nz, ny, nx = ct["shape"]
        half_x = (nx - 1) * sx / 2.0
        half_y = (ny - 1) * sy / 2.0
        half_z = (nz - 1) * sz / 2.0

        # Log+percentile display, like the DRR plots
        def _show(ax, sl, extent, xlabel, ylabel, title):
            log = np.log1p(sl * 50)
            p1, p99 = np.percentile(log, 1), np.percentile(log, 99)
            ax.imshow(log, cmap="bone", origin="lower", extent=extent,
                      vmin=p1, vmax=p99, aspect="equal")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.2, color="white")

        _show(ax_xy, ct["axial"],
              extent=[-half_x, half_x, -half_y, half_y],
              xlabel="X (mm) — anatomy frame",
              ylabel="Y (mm) — anatomy frame",
              title="Axial CT slice  z=0  (XY plane)")
        _show(ax_xz, ct["coronal"],
              extent=[-half_x, half_x, -half_z, half_z],
              xlabel="X (mm) — anatomy frame",
              ylabel="Z (mm) — anatomy frame",
              title="Coronal CT slice  y=0  (XZ plane)")
        _show(ax_yz, ct["sagittal"],
              extent=[-half_y, half_y, -half_z, half_z],
              xlabel="Y (mm) — anatomy frame",
              ylabel="Z (mm) — anatomy frame",
              title="Sagittal CT slice  x=0  (YZ plane)")
        plot_method = f"CT μ-slices ({Path(ct['cache']).name})"
    else:
        # Ellipsoid fallback for the synthetic phantom
        soft_x, soft_y, soft_z = (v * 1000.0 for v in SOFT_TISSUE_SEMIAXES_M)
        bone_x, bone_y, bone_z = (v * 1000.0 for v in BONE_CORE_SEMIAXES_M)
        draw_ellipse(ax_xy, soft_x, soft_y, "#d68a82", "soft tissue")
        draw_ellipse(ax_xy, bone_x, bone_y, "#888888", "bone core")
        draw_ellipse(ax_xz, soft_x, soft_z, "#d68a82", "soft tissue")
        draw_ellipse(ax_xz, bone_x, bone_z, "#888888", "bone core")
        draw_ellipse(ax_yz, soft_y, soft_z, "#d68a82", "soft tissue")
        draw_ellipse(ax_yz, bone_y, bone_z, "#888888", "bone core")
        ax_xy.set_xlim(-45, 45); ax_xy.set_ylim(-45, 45); ax_xy.set_aspect("equal")
        ax_xz.set_xlim(-45, 45); ax_xz.set_ylim(-75, 75); ax_xz.set_aspect("equal")
        ax_yz.set_xlim(-45, 45); ax_yz.set_ylim(-75, 75); ax_yz.set_aspect("equal")
        ax_xy.set_xlabel("X (mm) — anatomy frame"); ax_xy.set_ylabel("Y (mm)")
        ax_xy.set_title("Axial  (XY plane, +Z out of page)")
        ax_xz.set_xlabel("X (mm) — anatomy frame"); ax_xz.set_ylabel("Z (mm)")
        ax_xz.set_title("Coronal  (XZ plane, +Y into page)")
        ax_yz.set_xlabel("Y (mm) — anatomy frame"); ax_yz.set_ylabel("Z (mm)")
        ax_yz.set_title("Sagittal  (YZ plane, +X out of page)")
        for ax in (ax_xy, ax_xz, ax_yz):
            ax.grid(True, alpha=0.3)
        plot_method = "synthetic ellipsoid"

    # Overlay the tool on each orthogonal plane.  (xi, yi) selects which
    # anatomy axes map to each panel's horizontal / vertical axis.  The GT
    # tool is drawn first (underneath) as a faint dashed needle so the
    # estimate sits on top of it where the two agree.
    tip_mm     = tool_seg_anatomy_mm[0]
    rear_mm    = tool_seg_anatomy_mm[1]
    tip_gt_mm  = tool_seg_gt_anatomy_mm[0]
    rear_gt_mm = tool_seg_gt_anatomy_mm[1]
    tip_err_mm = float(np.linalg.norm(ee_anatomy_mm - ee_anatomy_gt_mm))
    for ax, (xi, yi) in [(ax_xy, (0, 1)), (ax_xz, (0, 2)), (ax_yz, (1, 2))]:
        # Ground-truth tool (faint, underneath)
        ax.plot([rear_gt_mm[xi], tip_gt_mm[xi]], [rear_gt_mm[yi], tip_gt_mm[yi]],
                "--", color="#ff3d3d", lw=3.5, alpha=0.55, solid_capstyle="round",
                label="tool shaft (ground truth)")
        ax.plot(tip_gt_mm[xi], tip_gt_mm[yi], "o", color="none",
                markeredgecolor="#ff3d3d", mew=2.0, ms=15, alpha=0.85,
                label=f"tool tip (GT)  Δest={tip_err_mm:.3f} mm")
        # Estimated tool (recovered, on top)
        ax.plot([rear_mm[xi], tip_mm[xi]], [rear_mm[yi], tip_mm[yi]],
                "-", color="#00e5ff", lw=2.5, solid_capstyle="round",
                label="tool shaft (estimated)")
        ax.plot(tip_mm[xi], tip_mm[yi], "o",
                color="#2ca02c", ms=10, markeredgecolor="white", mew=1.2,
                label=f"tool tip (est.) ({ee_anatomy_mm[xi]:.1f}, {ee_anatomy_mm[yi]:.1f}) mm")
        ax.plot(0, 0, "+", color="#ffeb3b", ms=14, mew=2.5,
                label="anatomy origin (GT)")
        ax.plot(phantom_err_mm[xi], phantom_err_mm[yi], "x", color="#ff7f0e",
                ms=9, mew=2, label=f"recov. anatomy Δ ({err_norm_mm:.3f} mm)")

    # One compact, shared legend (the three panels use identical markers, so a
    # legend per panel just adds clutter).  Anchored in the axial panel's
    # upper-right corner (dark background, tool sits lower-left) with shrunk
    # marker icons so the entries don't pile on top of each other.
    handles, labels = ax_xy.get_legend_handles_labels()
    ax_xy.legend(handles, labels, loc="upper right", fontsize=7,
                 markerscale=0.6, labelspacing=0.3, handlelength=1.6,
                 handletextpad=0.5, borderpad=0.4, framealpha=0.9)

    # Text panel: the deliverable transforms + clinical check
    ax_txt.axis("off")
    ax_txt.text(0.0, 1.0, "\n".join(summary_lines), transform=ax_txt.transAxes,
                fontsize=10, family="monospace", va="top", ha="left")

    fig.suptitle(
        f"Robot tool in the recovered anatomy frame  |  "
        f"registration world ||err|| = {err_norm_mm:.3f} mm  |  "
        f"background: {plot_method}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(LAYOUT_PNG, dpi=120, bbox_inches="tight")
    print(f"Saved {LAYOUT_PNG}")
    if show:
        plt.show()


# ─── main ─────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true",
                        help="Open the layout plot in an interactive window")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip the deliverable layout figure entirely")
    args = parser.parse_args()

    pose = load_pose_json()
    trace, mode = load_registration_trace()
    ig = trace["isaac_ground_truth"]
    registration_err_norm_mm = ig["world_err_norm_mm"]

    # World-frame poses
    ee_W = Pose.from_pos_quat_wxyz(pose["ee_pos"], pose["ee_quat"])
    carm_W = Pose.from_pos_quat_wxyz(pose["carm_pos"], pose["carm_quat"])
    phantom_W_gt = Pose.from_pos_quat_wxyz(
        pose["phantom_pos"], pose["phantom_quat"]
    )
    # Use recovered phantom rotation from 6-DOF registration if available.
    # Falls back to identity for traces produced before the 6-DOF upgrade.
    phantom_quat_rec = ig.get("phantom_quat_recovered_wxyz", [1.0, 0.0, 0.0, 0.0])
    phantom_W_rec = Pose.from_pos_quat_wxyz(
        ig["phantom_pos_recovered_world_m"], phantom_quat_rec
    )

    # The three transforms, in both forms where applicable
    T_R_in_A_gt = phantom_W_gt.inv() @ ee_W
    T_R_in_A_rec = phantom_W_rec.inv() @ ee_W
    T_R_in_C = carm_W.inv() @ ee_W  # carm is known either way; no GT/rec split
    T_A_in_C_gt = carm_W.inv() @ phantom_W_gt
    T_A_in_C_rec = carm_W.inv() @ phantom_W_rec

    # Errors (translation only — phantom rotation is identity by assumption)
    err_T_R_in_A_m = T_R_in_A_rec.t - T_R_in_A_gt.t
    err_T_R_in_A_mm = err_T_R_in_A_m * 1000.0
    err_T_R_in_A_norm_mm = float(np.linalg.norm(err_T_R_in_A_mm))

    err_T_A_in_C_m = T_A_in_C_rec.t - T_A_in_C_gt.t
    err_T_A_in_C_mm = err_T_A_in_C_m * 1000.0
    err_T_A_in_C_norm_mm = float(np.linalg.norm(err_T_A_in_C_mm))

    # Rotation error between GT and recovered phantom orientation
    R_rel = phantom_W_rec.R.T @ phantom_W_gt.R
    cos_a = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
    rot_err_deg = float(np.degrees(np.arccos(cos_a)))
    rot_err_from_trace_deg = ig.get("rot_err_geodesic_deg", 0.0)

    # Sanity check: T_R^A translational error should match the registration's
    # world translation error when phantom rotation is recovered accurately.
    # With imperfect rotation recovery, a small discrepancy is expected.
    if abs(err_T_R_in_A_norm_mm - registration_err_norm_mm) > 0.1:
        print(
            f"WARNING: T_R^A translation error ({err_T_R_in_A_norm_mm:.4f} mm) "
            f"differs from registration world_err_norm_mm "
            f"({registration_err_norm_mm:.4f} mm) by more than 0.1 mm — "
            "this may indicate residual rotation error coupling into translation."
        )

    # Clinical interpretation
    clin = clinical_summary(T_R_in_A_rec.t)

    # ─── stdout ──────────────────────────────────────────────────────────────
    print("=" * 60)
    print("T_robot_to_anatomy  (from DRR registration + robot forward kinematics)")
    print("=" * 60)
    print(f"  Source: registration_{mode} "
          f"(world ||err|| = {registration_err_norm_mm:.4f} mm)")
    print(f"  Sources:")
    print(f"    pose.json:           {POSE_FILE}")
    print(f"    registration trace:  "
          f"{TRACE_MULTIVIEW if mode == 'multiview' else TRACE_SINGLEVIEW}")
    print()

    def _fmt(v):
        return "[" + ", ".join(f"{x:7.3f}" for x in v) + "]"

    print(f"  T_robot_in_anatomy.t (mm):")
    print(f"           GT:        {_fmt(T_R_in_A_gt.t * 1000.0)}")
    print(f"    Recovered:        {_fmt(T_R_in_A_rec.t * 1000.0)}")
    print(f"     Error per-axis:  {_fmt(err_T_R_in_A_mm)}")
    print(f"          Error norm: {err_T_R_in_A_norm_mm:.4f} mm")
    print()
    print(f"  T_anatomy_in_carm.t  (mm):")
    print(f"           GT:        {_fmt(T_A_in_C_gt.t * 1000.0)}")
    print(f"    Recovered:        {_fmt(T_A_in_C_rec.t * 1000.0)}")
    print(f"     Error per-axis:  {_fmt(err_T_A_in_C_mm)}")
    print(f"          Error norm: {err_T_A_in_C_norm_mm:.4f} mm")
    print()
    print(f"  Phantom rotation error (geodesic): {rot_err_deg:.4f} °")
    print()
    print(f"  T_robot_in_carm.t    (mm):  {_fmt(T_R_in_C.t * 1000.0)}")
    print(f"    (No GT/recovered split — C-arm pose is known regardless.)")
    print()

    print("=" * 60)
    print(f"Clinical interpretation  (method: {clin['method']})")
    print("=" * 60)
    print(f"  EE position in anatomy frame (mm):  "
          f"{_fmt(np.asarray(clin['ee_pos_in_anatomy_mm_recovered']))}")
    inside_b = "INSIDE" if clin["inside_bone_core"] else "OUTSIDE"
    inside_s = "INSIDE" if clin["inside_soft_tissue"] else "OUTSIDE"
    if clin["method"] == "ct_mu_lookup":
        print(f"  Local μ at EE position:           "
              f"{clin['local_mu_mm_inv']:.4f} mm⁻¹")
        print(f"    Bone threshold ({clin['mu_bone_thresh']}):"
              f"      {inside_b} bone")
        print(f"    Soft-tissue threshold ({clin['mu_tissue_thresh']}):"
              f" {inside_s} soft tissue")
    else:
        print(f"  Normalized distance to bone core:   "
              f"{clin['bone_core_normalized_distance']:.3f}  -> {inside_b} bone")
        print(f"  Normalized distance to soft tissue: "
              f"{clin['soft_tissue_normalized_distance']:.3f}  -> {inside_s} soft tissue")
    print(f"  -> {clin['interpretation']}")
    print()

    # ─── JSON output ────────────────────────────────────────────────────────
    result = {
        "ground_truth": {
            "T_robot_in_anatomy": T_R_in_A_gt.to_dict(),
            "T_robot_in_carm": T_R_in_C.to_dict(),
            "T_anatomy_in_carm": T_A_in_C_gt.to_dict(),
        },
        "recovered": {
            "T_robot_in_anatomy": T_R_in_A_rec.to_dict(),
            "T_robot_in_carm": T_R_in_C.to_dict(),
            "T_anatomy_in_carm": T_A_in_C_rec.to_dict(),
        },
        "errors": {
            "T_robot_in_anatomy_t_err_mm": err_T_R_in_A_mm.tolist(),
            "T_robot_in_anatomy_t_err_norm_mm": err_T_R_in_A_norm_mm,
            "T_anatomy_in_carm_t_err_mm": err_T_A_in_C_mm.tolist(),
            "T_anatomy_in_carm_t_err_norm_mm": err_T_A_in_C_norm_mm,
            "T_robot_in_carm_t_err_mm": [0.0, 0.0, 0.0],
            "T_robot_in_carm_t_err_norm_mm": 0.0,
            "phantom_rot_err_geodesic_deg": rot_err_deg,
            "phantom_rot_err_from_trace_deg": rot_err_from_trace_deg,
        },
        "clinical": clin,
        "sources": {
            "pose_json": str(POSE_FILE),
            "registration_trace": str(
                TRACE_MULTIVIEW if mode == "multiview" else TRACE_SINGLEVIEW
            ),
            "registration_mode": mode,
        },
        "assumptions": [] if rot_err_deg < 0.1 else [
            f"Phantom rotation residual error {rot_err_deg:.3f}° — "
            "6-DOF registration recovered rotation with this geodesic error."
        ],
    }
    with open(RESULT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {RESULT_JSON}")

    # ─── plot ───────────────────────────────────────────────────────────────
    if args.no_plot:
        return 0

    phantom_err_world_m = np.asarray(ig["phantom_pos_recovered_world_m"]) \
                          - np.asarray(ig["phantom_pos_world_m"])
    phantom_err_world_mm = phantom_err_world_m * 1000.0

    # Tool shaft as an oriented needle in the recovered anatomy frame.
    # The endo360 TCP (EE prim) is the tip; the shaft extends along EE-local
    # -Z (see CLAUDE.md: mesh occupies local z ∈ [-51.5, -1.0] mm).  Draw the
    # rear of the shaft (-45 mm) to a hair past the tip (+5 mm) so orientation
    # is visible in every panel.
    ee_anatomy_mm = np.asarray(clin["ee_pos_in_anatomy_mm_recovered"])
    shaft_local_mm = np.array([[0.0, 0.0, 5.0], [0.0, 0.0, -45.0]])
    tool_seg_anatomy_mm = shaft_local_mm @ T_R_in_A_rec.R.T \
                          + (T_R_in_A_rec.t * 1000.0)
    # Ground-truth tool pose (from Isaac Sim's known anatomy pose) for the
    # overlay — at this accuracy it sits on top of the estimate; under noise
    # or a harder start the two separate, making the error visible.
    ee_anatomy_gt_mm = T_R_in_A_gt.t * 1000.0
    tool_seg_gt_anatomy_mm = shaft_local_mm @ T_R_in_A_gt.R.T \
                             + (T_R_in_A_gt.t * 1000.0)

    def _v(t):
        return "[" + ", ".join(f"{x:7.2f}" for x in t) + "]"

    def _q(q):
        return "[" + ", ".join(f"{x:6.3f}" for x in q) + "]"

    tra = T_R_in_A_rec.to_dict()
    tac = T_A_in_C_rec.to_dict()
    trc = T_R_in_C.to_dict()
    summary_lines = [
        "DELIVERABLE — recovered transforms",
        "(translation mm, rotation quat wxyz)",
        "-" * 40,
        "T_anatomy->C-arm",
        f"  t = {_v(tac['t_mm'])}",
        f"  q = {_q(tac['R_quat_wxyz'])}",
        "T_robot->anatomy",
        f"  t = {_v(tra['t_mm'])}",
        f"  q = {_q(tra['R_quat_wxyz'])}",
        "T_robot->C-arm",
        f"  t = {_v(trc['t_mm'])}",
        "",
        "Accuracy vs Isaac Sim ground truth",
        "-" * 40,
        f"  registration world ||err|| : {registration_err_norm_mm:7.4f} mm",
        f"  T_robot->anatomy   ||err|| : {err_T_R_in_A_norm_mm:7.4f} mm",
        f"  T_anatomy->C-arm   ||err|| : {err_T_A_in_C_norm_mm:7.4f} mm",
        f"  phantom rotation (geodesic): {rot_err_deg:7.4f} deg",
        "",
        f"Clinical check ({clin['method']})",
        "-" * 40,
        f"  tool tip @ {_v(ee_anatomy_mm)} mm (anatomy)",
    ]
    if clin["method"] == "ct_mu_lookup":
        summary_lines.append(
            f"  local mu = {clin['local_mu_mm_inv']:.4f} /mm")
    summary_lines += _wrap(f"  -> {clin['interpretation']}", 46)

    make_layout_plot(
        ee_anatomy_mm=ee_anatomy_mm,
        tool_seg_anatomy_mm=tool_seg_anatomy_mm,
        ee_anatomy_gt_mm=ee_anatomy_gt_mm,
        tool_seg_gt_anatomy_mm=tool_seg_gt_anatomy_mm,
        phantom_err_mm=phantom_err_world_mm,
        err_norm_mm=err_T_R_in_A_norm_mm,
        summary_lines=summary_lines,
        show=args.show,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

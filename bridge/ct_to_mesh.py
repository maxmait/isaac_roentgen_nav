#!/usr/bin/env python3
"""Generate a triangle mesh from the cached CT μ-volume.

Runs on the HOST (not in Docker).  Requires: scikit-image, scipy, numpy
(pip install scikit-image if missing).

Pipeline
--------
1. Load  mu_volume.npy from the cropped CT cache.
2. Pre-smooth the bone mask with a Gaussian to reduce staircase artefacts.
3. Extract an isosurface with skimage.measure.marching_cubes.
4. Apply Laplacian mesh smoothing (cleans marching-cubes jagginess).
5. Save an OBJ with vertices in **metres, centred at the volume isocenter**.
   Placing this mesh at PHANTOM_POS_WORLD_M in Isaac Sim aligns it with the
   fluorosim μ-volume used for registration — no extra transform needed.

OBJ coordinate convention
--------------------------
The μ-volume is stored (Z, Y, X) where Z = superior-inferior (up in Isaac
Sim), Y = anterior-posterior, X = left-right.  marching_cubes returns
vertices as (z_mm, y_mm, x_mm).  We reorder and convert to the Isaac Sim
convention (X, Y, Z metres) so the OBJ can be loaded with a single
translation op.

Usage
-----
    python3 bridge/ct_to_mesh.py                         # all defaults
    python3 bridge/ct_to_mesh.py \\
        --cache-dir ~/isaac_projects/output/fluorosim_cache_ct \\
        --out ~/isaac_projects/output/spine_mesh.obj \\
        --threshold 0.032 --smooth-sigma 0.8 --smooth-iters 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CACHE_DEFAULT = Path.home() / "isaac_projects/output/fluorosim_cache_ct"
OUT_DEFAULT   = Path.home() / "isaac_projects/output/spine_mesh.obj"

# μ threshold for bone isosurface extraction.
# 0.032 mm⁻¹ sits between soft tissue (~0.021) and trabecular bone (~0.048).
BONE_THRESHOLD_MM: float = 0.032


def laplacian_smooth(
    verts: np.ndarray,
    faces: np.ndarray,
    iterations: int = 5,
    lam: float = 0.5,
) -> np.ndarray:
    """Per-vertex Laplacian smoothing using a normalised adjacency matrix."""
    from scipy.sparse import csr_matrix

    n = len(verts)
    edges = np.vstack([
        faces[:, [0, 1]], faces[:, [1, 0]],
        faces[:, [1, 2]], faces[:, [2, 1]],
        faces[:, [0, 2]], faces[:, [2, 0]],
    ])
    data = np.ones(len(edges), dtype=np.float32)
    adj  = csr_matrix((data, (edges[:, 0], edges[:, 1])), shape=(n, n))
    deg  = np.asarray(adj.sum(axis=1)).ravel().clip(1)
    norm_adj = csr_matrix(
        (data / deg[edges[:, 0]], (edges[:, 0], edges[:, 1])), shape=(n, n)
    )
    for _ in range(iterations):
        avg   = norm_adj.dot(verts)
        verts = verts + lam * (avg - verts)
    return verts


def generate_mesh(
    cache_dir: Path,
    out_obj: Path,
    threshold: float = BONE_THRESHOLD_MM,
    smooth_sigma: float = 0.8,
    smooth_iters: int = 5,
) -> dict:
    """Extract, smooth and save the bone mesh.  Returns stats dict."""
    from scipy.ndimage import gaussian_filter

    try:
        from skimage.measure import marching_cubes
    except ImportError:
        print("ERROR: scikit-image not found.  Run: pip install scikit-image",
              file=sys.stderr)
        sys.exit(1)

    # ── Load ──────────────────────────────────────────────────────────────────
    mu_path = cache_dir / "mu_volume.npy"
    if not mu_path.exists():
        raise FileNotFoundError(
            f"{mu_path} not found.  Run run_load_ct.sh first."
        )
    mu      = np.load(mu_path)            # (Z, Y, X)  float32
    meta    = json.loads((cache_dir / "metadata.json").read_text())
    spacing = tuple(meta["spacing_zyx_mm"])   # (sz, sy, sx) mm
    print(f"  μ-volume: shape={mu.shape}  spacing={spacing} mm")
    print(f"  μ range:  [{mu.min():.5f}, {mu.max():.5f}] mm⁻¹")
    bone_frac = float((mu > threshold).mean())
    print(f"  Bone voxels (μ>{threshold}): {bone_frac*100:.2f}%")

    # ── Pre-smooth ────────────────────────────────────────────────────────────
    print(f"  Gaussian pre-smooth  σ={smooth_sigma} voxels ...")
    mu_smooth = gaussian_filter(mu.astype(np.float32), sigma=smooth_sigma)

    # ── Marching cubes ────────────────────────────────────────────────────────
    print(f"  Marching cubes at threshold={threshold} ...")
    # verts_zyx in mm (axis order: Z, Y, X); faces are triangle indices.
    verts_zyx, faces, _normals, _ = marching_cubes(
        mu_smooth, level=threshold, spacing=spacing, allow_degenerate=False
    )
    print(f"  Raw mesh: {len(verts_zyx):,} vertices  {len(faces):,} faces")

    # ── Centre at volume isocenter ────────────────────────────────────────────
    sz, sy, sx = spacing
    nz, ny, nx = mu.shape
    centre_zyx = np.array([(nz - 1) * sz / 2.0,
                            (ny - 1) * sy / 2.0,
                            (nx - 1) * sx / 2.0])
    verts_zyx -= centre_zyx          # centred: range ≈ ±64 mm each axis

    # ── Laplacian smoothing ───────────────────────────────────────────────────
    print(f"  Laplacian smoothing  {smooth_iters} iterations ...")
    verts_zyx = laplacian_smooth(verts_zyx, faces, smooth_iters)

    # ── Reorder axes: ZYX mm → XYZ metres (Isaac Sim convention) ─────────────
    # Isaac Sim: X=left-right, Y=ant-post, Z=up(=volume Z)
    verts_m = np.column_stack([
        verts_zyx[:, 2] / 1000.0,   # X: volume X
        verts_zyx[:, 1] / 1000.0,   # Y: volume Y
        verts_zyx[:, 0] / 1000.0,   # Z: volume Z (up)
    ])

    # ── Save numpy arrays (fast path for Isaac Sim) ───────────────────────────
    out_obj.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_obj.with_suffix(".verts.npy"), verts_m.astype(np.float32))
    np.save(out_obj.with_suffix(".faces.npy"), faces.astype(np.int32))
    print(f"  Saved {out_obj.with_suffix('.verts.npy')}  "
          f"({out_obj.with_suffix('.verts.npy').stat().st_size / 1e6:.1f} MB)")

    # ── Save OBJ (human-readable, for external viewers) ───────────────────────
    with open(out_obj, "w") as f:
        f.write(f"# Spine mesh from {cache_dir.name}\n")
        f.write(f"# Vertices: metres, centred at volume isocenter\n")
        f.write(f"# Place at PHANTOM_POS_WORLD_M in Isaac Sim\n")
        for v in verts_m:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for fa in faces:
            f.write(f"f {fa[0]+1} {fa[1]+1} {fa[2]+1}\n")
    print(f"  Saved {out_obj}  ({out_obj.stat().st_size / 1e6:.1f} MB)")

    # ── Save metadata ─────────────────────────────────────────────────────────
    meta_out = out_obj.with_suffix(".json")
    stats = {
        "source_cache": str(cache_dir),
        "threshold_mm_inv": threshold,
        "spacing_zyx_mm": list(spacing),
        "volume_shape_zyx": list(mu.shape),
        "volume_extent_mm": [float(nz * sz), float(ny * sy), float(nx * sx)],
        "smooth_sigma_voxels": smooth_sigma,
        "smooth_iters": smooth_iters,
        "n_vertices": len(verts_m),
        "n_faces": len(faces),
        "bounds_xyz_m": [
            [float(verts_m[:, i].min()), float(verts_m[:, i].max())]
            for i in range(3)
        ],
    }
    meta_out.write_text(json.dumps(stats, indent=2))
    print(f"  Saved {meta_out}")
    return stats


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--cache-dir",  type=Path, default=CACHE_DEFAULT)
    p.add_argument("--out",        type=Path, default=OUT_DEFAULT)
    p.add_argument("--threshold",  type=float, default=BONE_THRESHOLD_MM)
    p.add_argument("--smooth-sigma", type=float, default=0.8)
    p.add_argument("--smooth-iters", type=int,   default=5)
    args = p.parse_args()

    print("=" * 60)
    print("ct_to_mesh.py — μ-volume → triangle mesh (OBJ)")
    print("=" * 60)
    print(f"  Cache dir:  {args.cache_dir}")
    print(f"  Output OBJ: {args.out}")
    print(f"  Threshold:  {args.threshold} mm⁻¹")

    stats = generate_mesh(
        cache_dir    = args.cache_dir,
        out_obj      = args.out,
        threshold    = args.threshold,
        smooth_sigma = args.smooth_sigma,
        smooth_iters = args.smooth_iters,
    )

    print()
    print(f"  Mesh:   {stats['n_vertices']:,} vertices  {stats['n_faces']:,} faces")
    print(f"  Bounds X: {stats['bounds_xyz_m'][0]} m")
    print(f"  Bounds Y: {stats['bounds_xyz_m'][1]} m")
    print(f"  Bounds Z: {stats['bounds_xyz_m'][2]} m")
    print("\n=== Done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())

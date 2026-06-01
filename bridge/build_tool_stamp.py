#!/usr/bin/env python3
"""Voxelize the extracted endo360 tip mesh into a small μ-volume stamp.

Reads (from ~/isaac_projects/output/):
    tool_mesh.verts.npy   (N, 3) float32  EE-local mm
    tool_mesh.faces.npy   (F, 3) int32    triangle vertex indices
    tool_mesh.json                         source paths + bbox

Writes:
    tool_stamp.npy        (nz, ny, nx) float32  μ values in EE-local frame.
                          Interior voxels = TOOL_MU_PER_MM, exterior = 0.
    tool_stamp.json                              shape + spacing + origin

Stamp metadata (stamp_origin_ee_local_xyz_mm) gives the EE-local mm
coordinates of the (z=0, y=0, x=0) voxel center. The phantom-painting
code reads this to splat the stamp into a phantom μ-volume regardless of
the EE's world pose.

The voxelization uses trimesh.contains() (winding-number based, robust
for slightly non-watertight meshes). Default spacing 0.5 mm matches the
typical CT cache; pad the bbox by TOOL_STAMP_PAD_MM (default 1 mm) so the
mesh edges are reproduced cleanly.

Env vars (all optional):
    STAMP_SPACING_MM   isotropic voxel spacing       (default 0.5)
    STAMP_PAD_MM       pad around mesh bbox          (default 1.0)
    TOOL_MU_PER_MM     interior μ value              (default 0.3)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import trimesh

OUT_DIR = Path(os.path.expanduser("~/isaac_projects/output"))
VERTS_PATH = OUT_DIR / "tool_mesh.verts.npy"
FACES_PATH = OUT_DIR / "tool_mesh.faces.npy"
MESH_META = OUT_DIR / "tool_mesh.json"

STAMP_NPY = OUT_DIR / "tool_stamp.npy"
STAMP_META = OUT_DIR / "tool_stamp.json"

STAMP_SPACING_MM = float(os.environ.get("STAMP_SPACING_MM", "0.5"))
STAMP_PAD_MM     = float(os.environ.get("STAMP_PAD_MM", "1.0"))
TOOL_MU_PER_MM   = float(os.environ.get("TOOL_MU_PER_MM", "0.3"))

# Optional synthetic shaft extending in EE-local −z from the back of the
# real tip mesh. Without it the tool is ~50 mm long (just the endo360
# tip), which looks nearly axially symmetric from any C-arm angle. Adding
# a 50 mm shaft (→ total tool ~100 mm) gives a distinct silhouette in
# lateral / oblique views — the shaft projects as a long line at 90° but
# a short stub at 0°. Set SHAFT_LENGTH_MM=0 to disable (back to tip-only).
SHAFT_LENGTH_MM  = float(os.environ.get("SHAFT_LENGTH_MM", "50.0"))
SHAFT_RADIUS_MM  = float(os.environ.get("SHAFT_RADIUS_MM", "5.0"))


def main() -> int:
    if not VERTS_PATH.exists() or not FACES_PATH.exists():
        raise SystemExit(
            f"Missing {VERTS_PATH} / {FACES_PATH}. Run extract_tool_mesh.py via "
            f"TCP injection first."
        )

    verts = np.load(VERTS_PATH)               # (N, 3) float32 EE-local mm
    faces = np.load(FACES_PATH)               # (F, 3) int32
    print(f"Loaded mesh: {verts.shape[0]:,} verts, {faces.shape[0]:,} faces")

    mesh = trimesh.Trimesh(vertices=verts.astype(np.float64), faces=faces,
                            process=False)
    is_watertight = mesh.is_watertight
    print(f"  watertight: {is_watertight}, "
          f"volume(if watertight) = "
          f"{mesh.volume if is_watertight else float('nan'):.3f} mm³, "
          f"surface area = {mesh.area:.3f} mm²")
    if not is_watertight:
        print("  NOTE: mesh is not watertight. trimesh.contains() will still "
              "work via winding-number test, but may misclassify a few voxels "
              "near unsealed edges. Acceptable for this μ-volume use.")

    mesh_bbox_min = verts.min(axis=0).astype(np.float64)   # (3,) xyz mm
    mesh_bbox_max = verts.max(axis=0).astype(np.float64)

    # Combined bbox = mesh ∪ shaft extension (cylinder extending in EE-local −z
    # from the back of the tip toward the surgeon's hand).
    shaft_center_xy = (mesh_bbox_min[:2] + mesh_bbox_max[:2]) / 2.0
    add_shaft = SHAFT_LENGTH_MM > 0.0
    if add_shaft:
        # Shaft occupies x ∈ [center_x − r, center_x + r], same for y,
        # z ∈ [mesh_z_min − L, mesh_z_min].
        shaft_z_max = mesh_bbox_min[2]
        shaft_z_min = mesh_bbox_min[2] - SHAFT_LENGTH_MM
        combined_min = np.array([
            min(mesh_bbox_min[0], shaft_center_xy[0] - SHAFT_RADIUS_MM),
            min(mesh_bbox_min[1], shaft_center_xy[1] - SHAFT_RADIUS_MM),
            shaft_z_min,
        ])
        combined_max = np.array([
            max(mesh_bbox_max[0], shaft_center_xy[0] + SHAFT_RADIUS_MM),
            max(mesh_bbox_max[1], shaft_center_xy[1] + SHAFT_RADIUS_MM),
            mesh_bbox_max[2],
        ])
    else:
        combined_min, combined_max = mesh_bbox_min, mesh_bbox_max

    pad = STAMP_PAD_MM
    stamp_origin_xyz = combined_min - pad     # voxel (0,0,0) corner
    stamp_far_xyz    = combined_max + pad
    extent_xyz       = stamp_far_xyz - stamp_origin_xyz
    n_voxels_xyz     = np.ceil(extent_xyz / STAMP_SPACING_MM).astype(int)
    n_voxels_zyx     = n_voxels_xyz[[2, 1, 0]]
    spacing_zyx_mm   = (STAMP_SPACING_MM, STAMP_SPACING_MM, STAMP_SPACING_MM)
    stamp_origin_zyx = stamp_origin_xyz[[2, 1, 0]]
    print(f"  Mesh bbox (xyz mm): min={mesh_bbox_min.tolist()}, max={mesh_bbox_max.tolist()}")
    if add_shaft:
        print(f"  Synthetic shaft: z ∈ [{shaft_z_min:.1f}, {shaft_z_max:.1f}] mm  "
              f"(L={SHAFT_LENGTH_MM} mm, r={SHAFT_RADIUS_MM} mm, "
              f"axis @ xy=({shaft_center_xy[0]:.2f}, {shaft_center_xy[1]:.2f}))")
    print(f"  Stamp grid: shape (z,y,x) = {tuple(n_voxels_zyx.tolist())}, "
          f"spacing = {STAMP_SPACING_MM} mm, pad = {pad} mm "
          f"({n_voxels_zyx.prod():,} voxels)")

    # Voxel CENTER positions in EE-local xyz mm
    iz, iy, ix = np.meshgrid(
        np.arange(n_voxels_zyx[0]),
        np.arange(n_voxels_zyx[1]),
        np.arange(n_voxels_zyx[2]),
        indexing="ij",
    )
    centers_zyx_mm = (
        stamp_origin_zyx
        + np.stack([iz, iy, ix], axis=-1) * STAMP_SPACING_MM
        + STAMP_SPACING_MM / 2.0
    )
    centers_xyz_mm = centers_zyx_mm[..., [2, 1, 0]].reshape(-1, 3)

    # Restrict trimesh.contains() to voxels inside the mesh bbox (+ small
    # margin). For larger combined bboxes that's a big speedup.
    margin = STAMP_SPACING_MM
    in_mesh_bbox = np.all(
        (centers_xyz_mm >= mesh_bbox_min - margin)
        & (centers_xyz_mm <= mesh_bbox_max + margin),
        axis=1,
    )
    inside = np.zeros(centers_xyz_mm.shape[0], dtype=bool)
    bbox_idx = np.nonzero(in_mesh_bbox)[0]
    chunk = int(os.environ.get("CONTAINS_CHUNK", "50000"))
    print(f"  Running trimesh.contains() on {bbox_idx.size:,} mesh-bbox points "
          f"of {centers_xyz_mm.shape[0]:,} total, in chunks of {chunk}…")
    t0 = time.time()
    for start in range(0, bbox_idx.size, chunk):
        end = min(start + chunk, bbox_idx.size)
        sub = bbox_idx[start:end]
        inside[sub] = mesh.contains(centers_xyz_mm[sub])
        if bbox_idx.size > chunk:
            print(f"    chunk {start:>8,} – {end:>8,}  "
                  f"({100 * end / bbox_idx.size:.0f}%)  "
                  f"interior so far: {int(inside.sum()):,}", flush=True)
    dt = time.time() - t0
    print(f"  Mesh interior: {int(inside.sum()):,} voxels in {dt:.2f} s")

    if add_shaft:
        dx = centers_xyz_mm[:, 0] - shaft_center_xy[0]
        dy = centers_xyz_mm[:, 1] - shaft_center_xy[1]
        cz = centers_xyz_mm[:, 2]
        in_shaft = (
            (dx * dx + dy * dy <= SHAFT_RADIUS_MM ** 2)
            & (cz >= shaft_z_min) & (cz <= shaft_z_max)
        )
        n_shaft_only = int(in_shaft.sum() - (in_shaft & inside).sum())
        inside = inside | in_shaft
        print(f"  Synthetic shaft adds {n_shaft_only:,} voxels not already in mesh")
    print(f"  Total interior: {int(inside.sum()):,} voxels "
          f"({100 * inside.mean():.2f}% of grid)")

    stamp = np.zeros(tuple(n_voxels_zyx), dtype=np.float32)
    stamp.reshape(-1)[inside] = TOOL_MU_PER_MM

    np.save(STAMP_NPY, stamp)
    meta = {
        "shape_zyx":             [int(x) for x in n_voxels_zyx],
        "spacing_zyx_mm":         list(spacing_zyx_mm),
        "origin_ee_local_xyz_mm": stamp_origin_xyz.tolist(),
        # origin_ee_local_xyz_mm is the EE-local mm coordinate corresponding to
        # the (z=0, y=0, x=0) voxel corner (NOT center).  A voxel-center is at
        # origin + (k + 0.5) * spacing in each axis.
        "voxel_center_convention": "origin_xyz + (idx + 0.5) * spacing",
        "tool_mu_per_mm":         TOOL_MU_PER_MM,
        "pad_mm":                 STAMP_PAD_MM,
        "shaft_length_mm":        SHAFT_LENGTH_MM,
        "shaft_radius_mm":        SHAFT_RADIUS_MM if add_shaft else None,
        "shaft_axis_xy_mm":       shaft_center_xy.tolist() if add_shaft else None,
        "source_mesh":            str(VERTS_PATH),
        "source_mesh_meta":       json.load(open(MESH_META)) if MESH_META.exists() else None,
        "trimesh_watertight":     bool(is_watertight),
        "interior_voxels":        int(inside.sum()),
        "frame":                  "EE-local (axes match ee_quat); μ values in mm⁻¹",
    }
    with open(STAMP_META, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {STAMP_NPY}  ({stamp.nbytes/1024:.1f} KB)")
    print(f"Wrote {STAMP_META}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

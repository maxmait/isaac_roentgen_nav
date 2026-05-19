#!/usr/bin/env python3
"""Load a DICOM CT series and cache it as a fluorosim PreprocessedVolume.

Two modes (selected by the --full flag or CT_FULL_VOLUME=1):

  Cropped (default):
    Auto-detect the vertebral body level and crop a ROI there, then
    resample to TARGET_SHAPE_ZYX at TARGET_SPACING_ZYX_MM.
    Cache: fluorosim_cache_ct/

    The vertebral body centre is found automatically with
    find_vertebral_center() — compact-bone heuristic that works on any
    body region CT without hard-coded slice numbers.  Override via env var
    CT_CROP_CENTER_ZYX="z,y,x" (voxel indices in the full CT volume) when
    the auto-detection picks the wrong level.

  Full volume (--full / CT_FULL_VOLUME=1):
    Entire DICOM volume at native anisotropic spacing, no resampling.
    Cache: fluorosim_cache_ct_full/

Notes
-----
- Bilinear HU→μ bypasses fluorosim's VolumePreprocessor LUT, which clamps
  at μ_water and makes bone invisible in DRRs.
- Deleting the cache dir forces a rebuild (necessary if you change the
  crop centre or μ parameters).

Runs inside the fluorosim-torch container (pydicom 3.0.2 + SimpleITK 2.5.4
+ scipy already present).

CLI:
    python ct_loader.py <dicom_dir> <cache_dir> [--full] [--center=z,y,x]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from fluorosim import PreprocessedVolume, VolumePreprocessor


# ── Target geometry for cropped mode ────────────────────────────────────────
TARGET_SHAPE_ZYX: tuple[int, int, int]       = (128, 256, 256)
TARGET_SPACING_ZYX_MM: tuple[float, ...]     = (1.0, 0.5, 0.5)
HU_CLIP: tuple[float, float]                 = (-1000.0, 3000.0)

# ── Bilinear HU→μ at ~60 keV ─────────────────────────────────────────────────
# fluorosim's VolumePreprocessor clamps μ at μ_water (soft-tissue LUT).
# We bypass it with the standard piecewise-linear Hounsfield mapping so
# cortical bone reaches ~0.048 mm⁻¹ and is visible in DRRs.
MU_AIR_MM:   float = 0.0001   # air
MU_WATER_MM: float = 0.0200   # water / soft tissue
MU_BONE_MM:  float = 0.0480   # cortical bone at HU 1000


def hu_to_mu(hu: np.ndarray) -> np.ndarray:
    """Piecewise-linear HU → μ (mm⁻¹).  Extrapolates linearly above HU 1000."""
    hu = np.asarray(hu, dtype=np.float32)
    mu = np.where(
        hu <= 0,
        MU_AIR_MM   + (MU_WATER_MM - MU_AIR_MM)  * (hu + 1000.0) / 1000.0,
        MU_WATER_MM + (MU_BONE_MM  - MU_WATER_MM) * hu            / 1000.0,
    )
    return np.maximum(mu, 0.0).astype(np.float32)


# ── DICOM loading ─────────────────────────────────────────────────────────────

def load_dicom_series(
    dicom_dir: Path,
) -> tuple[np.ndarray, tuple[float, float, float], tuple[float, ...]]:
    """Read a CT DICOM series via SimpleITK.

    Returns (hu_zyx, spacing_zyx_mm, origin_xyz_mm).
    """
    import SimpleITK as sitk
    reader = sitk.ImageSeriesReader()
    files  = reader.GetGDCMSeriesFileNames(str(dicom_dir))
    if not files:
        raise RuntimeError(f"No DICOM series found under {dicom_dir}")
    reader.SetFileNames(files)
    img = reader.Execute()
    hu  = sitk.GetArrayFromImage(img).astype(np.float32)  # (Z, Y, X)
    sx, sy, sz = img.GetSpacing()
    return hu, (float(sz), float(sy), float(sx)), tuple(float(v) for v in img.GetOrigin())


# ── Vertebral-body centre detection ──────────────────────────────────────────

def find_vertebral_center(
    hu: np.ndarray,
    spacing_zyx_mm: tuple[float, float, float],
    bone_thresh_hu: float = 300.0,
) -> tuple[int, int, int]:
    """Automatically find the vertebral body level in any spine/body CT.

    Algorithm
    ---------
    Step 1 — best Z: for each axial slice count HU-bone voxels inside a
    central X-Y window (middle ⅔ of X, middle ½ of Y).  Vertebral bodies
    are compact and central; the pelvis and skull spread to the edges and
    score lower.  Search the middle 80 % of Z to avoid truncated edges.

    Step 2 — best Y: in the chosen axial slice, find the Y centroid of
    bone in the central X strip.  This places the crop on the actual
    vertebral body, not the geometric mid-body (which would miss the
    spine in the posterior half of the patient).

    X is always set to the geometric centre (spine lies on the midline).

    The result is the centre_zyx_voxel for crop_and_resample().  Override
    via CT_CROP_CENTER_ZYX="z,y,x" when auto-detection picks the wrong level.
    """
    nz, ny, nx = hu.shape
    cx, cy = nx // 2, ny // 2

    # Central window for step 1
    xlo, xhi = cx - nx // 3, cx + nx // 3
    ylo, yhi = cy - ny // 4, cy + ny // 4
    compact = (hu[:, ylo:yhi, xlo:xhi] > bone_thresh_hu).sum(axis=(1, 2))

    # Skip outer 10 % of Z (volume edges may be truncated or contain air only)
    z_margin = max(nz // 10, 1)
    z_lo, z_hi = z_margin, nz - z_margin

    if compact[z_lo:z_hi].max() == 0:
        print("  WARNING: no bone found in central window — falling back to "
              "volume centre.  Consider setting CT_CROP_CENTER_ZYX manually.")
        return nz // 2, ny // 2, cx

    best_z = z_lo + int(compact[z_lo:z_hi].argmax())

    # Step 2: Y centroid of bone in a narrow central X strip at best_z
    strip_xlo, strip_xhi = cx - nx // 6, cx + nx // 6
    y_bone = (hu[best_z, :, strip_xlo:strip_xhi] > bone_thresh_hu).sum(axis=1)
    if y_bone.sum() > 0:
        best_y = int(np.average(np.arange(ny), weights=y_bone))
    else:
        best_y = cy

    return best_z, best_y, cx


def _parse_center(raw: str) -> tuple[int, int, int]:
    """Parse 'z,y,x' string into (z, y, x) int tuple."""
    parts = [int(v.strip()) for v in raw.split(",")]
    if len(parts) != 3:
        raise ValueError(f"CT_CROP_CENTER_ZYX must be 'z,y,x', got: {raw!r}")
    return tuple(parts)  # type: ignore[return-value]


# ── Crop + resample ───────────────────────────────────────────────────────────

def crop_and_resample(
    hu: np.ndarray,
    source_spacing_zyx_mm: tuple[float, float, float],
    target_shape_zyx: tuple[int, int, int]       = TARGET_SHAPE_ZYX,
    target_spacing_zyx_mm: tuple[float, ...]      = TARGET_SPACING_ZYX_MM,
    center_zyx_voxel: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Crop a (target_shape × target_spacing) mm ROI and resample to target_shape."""
    from scipy.ndimage import zoom

    src_sp  = np.asarray(source_spacing_zyx_mm, dtype=np.float64)
    tgt_sh  = np.asarray(target_shape_zyx,       dtype=np.int64)
    tgt_sp  = np.asarray(target_spacing_zyx_mm,  dtype=np.float64)
    src_sh  = np.asarray(hu.shape,               dtype=np.int64)

    center  = (src_sh - 1) / 2.0 if center_zyx_voxel is None else \
              np.asarray(center_zyx_voxel, dtype=np.float64)

    extent_mm  = tgt_sh * tgt_sp
    half_vox   = (extent_mm / 2.0) / src_sp
    bb_min     = np.floor(center - half_vox).astype(np.int64)
    bb_max     = np.ceil( center + half_vox).astype(np.int64)
    pad_before = np.maximum(-bb_min,          0)
    pad_after  = np.maximum( bb_max - src_sh, 0)
    bb_min_c   = np.maximum(bb_min, 0)
    bb_max_c   = np.minimum(bb_max, src_sh)

    cropped = hu[bb_min_c[0]:bb_max_c[0],
                 bb_min_c[1]:bb_max_c[1],
                 bb_min_c[2]:bb_max_c[2]]
    if np.any(pad_before > 0) or np.any(pad_after > 0):
        cropped = np.pad(cropped,
                         [(int(pad_before[i]), int(pad_after[i])) for i in range(3)],
                         mode="constant", constant_values=HU_CLIP[0])

    cropped    = np.clip(cropped, HU_CLIP[0], HU_CLIP[1]).astype(np.float32)
    zoom_f     = tgt_sh / np.asarray(cropped.shape, dtype=np.float64)
    resampled  = zoom(cropped, zoom_f, order=1)

    # zoom sometimes returns shape ±1 voxel from float rounding — force exact.
    if resampled.shape != tuple(tgt_sh):
        out = np.full(tuple(tgt_sh), HU_CLIP[0], dtype=np.float32)
        n   = [min(resampled.shape[i], int(tgt_sh[i])) for i in range(3)]
        out[:n[0], :n[1], :n[2]] = resampled[:n[0], :n[1], :n[2]]
        resampled = out

    return resampled.astype(np.float32)


# ── Main loader ───────────────────────────────────────────────────────────────

def load_or_build_ct_volume(
    dicom_dir: Path,
    cache_dir: Path,
    target_shape_zyx:    tuple[int, int, int]  = TARGET_SHAPE_ZYX,
    target_spacing_zyx_mm: tuple[float, ...]   = TARGET_SPACING_ZYX_MM,
    full_volume:          bool                  = False,
    center_zyx_voxel:     tuple[int, int, int] | None = None,
) -> PreprocessedVolume:
    """Load a DICOM CT and return a cached PreprocessedVolume.

    Parameters
    ----------
    center_zyx_voxel
        Voxel index (Z, Y, X) in the *full* CT volume to centre the crop on.
        None (default) → auto-detect via find_vertebral_center().
        Ignored when full_volume=True.
        Override at runtime via env var CT_CROP_CENTER_ZYX="z,y,x".
    """
    # Env-var override always wins over programmatic argument
    env_center = os.environ.get("CT_CROP_CENTER_ZYX")
    if env_center:
        center_zyx_voxel = _parse_center(env_center)

    cache_dir.mkdir(parents=True, exist_ok=True)
    if (cache_dir / "mu_volume.npy").exists():
        print(f"  Reusing cached CT μ-volume: {cache_dir}")
        return PreprocessedVolume.load(cache_dir)

    print(f"  Loading DICOM series from {dicom_dir} ...")
    hu_full, src_spacing, src_origin = load_dicom_series(dicom_dir)
    print(f"    shape={hu_full.shape}  spacing_zyx_mm={src_spacing}")
    print(f"    DICOM origin (xyz mm)={src_origin}")
    print(f"    HU range=[{hu_full.min():.1f}, {hu_full.max():.1f}]  "
          f"mean={hu_full.mean():.1f}")

    if full_volume:
        print("  Full-volume mode — no crop/resample.")
        hu_out     = np.clip(hu_full, HU_CLIP[0], HU_CLIP[1]).astype(np.float32)
        spacing_out = src_spacing
    else:
        if center_zyx_voxel is None:
            print("  Auto-detecting vertebral body centre...")
            center_zyx_voxel = find_vertebral_center(hu_full, src_spacing)
            print(f"    detected centre (ZYX voxels): {center_zyx_voxel}")
            print(f"    world offset from volume centre: "
                  f"Z={( center_zyx_voxel[0] - hu_full.shape[0]/2)*src_spacing[0]:+.0f} mm  "
                  f"Y={( center_zyx_voxel[1] - hu_full.shape[1]/2)*src_spacing[1]:+.0f} mm  "
                  f"X={( center_zyx_voxel[2] - hu_full.shape[2]/2)*src_spacing[2]:+.0f} mm")
        else:
            print(f"  Using crop centre (ZYX voxels): {center_zyx_voxel}")

        print(f"  Cropping → shape={target_shape_zyx}  "
              f"spacing={target_spacing_zyx_mm} ...")
        hu_out      = crop_and_resample(hu_full, src_spacing,
                                        target_shape_zyx, target_spacing_zyx_mm,
                                        center_zyx_voxel)
        spacing_out = target_spacing_zyx_mm

    print(f"    output: shape={hu_out.shape}  "
          f"HU=[{hu_out.min():.0f},{hu_out.max():.0f}]  spacing={spacing_out}")

    # VolumePreprocessor writes valid metadata.json; its μ LUT is wrong for
    # bone, so we immediately overwrite mu_volume.npy with bilinear HU→μ.
    print("  Running VolumePreprocessor for metadata, then replacing μ values...")
    vol_ref = VolumePreprocessor.from_numpy(
        hu_out, spacing_zyx_mm=spacing_out
    ).preprocess(output_dir=cache_dir)

    mu_out = hu_to_mu(hu_out)
    np.save(cache_dir / "mu_volume.npy", mu_out)
    print(f"    μ range: [{mu_out.min():.5f}, {mu_out.max():.5f}] mm⁻¹")

    volume = PreprocessedVolume(mu_out, vol_ref._metadata)
    print(f"  Cached at {cache_dir}")
    return volume


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    positional  = [a for a in sys.argv[1:] if not a.startswith("--")]
    flag_set    = {a.split("=")[0] for a in sys.argv[1:] if a.startswith("--")}
    flag_values = {a.split("=")[0]: a.split("=", 1)[1]
                   for a in sys.argv[1:] if a.startswith("--") and "=" in a}

    if len(positional) < 2:
        print("Usage: ct_loader.py <dicom_dir> <cache_dir> [--full] [--center=z,y,x]",
              file=sys.stderr)
        return 1

    dicom_dir    = Path(positional[0])
    cache_dir    = Path(positional[1])
    full_volume  = "--full" in flag_set
    center       = _parse_center(flag_values["--center"]) \
                   if "--center" in flag_values else None

    print("=" * 60)
    print("ct_loader.py — DICOM CT → fluorosim PreprocessedVolume")
    print("=" * 60)
    print(f"  DICOM dir:    {dicom_dir}")
    print(f"  Cache dir:    {cache_dir}")
    print(f"  Mode:         {'full volume' if full_volume else 'cropped ROI'}")
    if center:
        print(f"  Centre (ZYX): {center}  (from --center flag)")
    elif os.environ.get("CT_CROP_CENTER_ZYX"):
        print(f"  Centre (ZYX): {os.environ['CT_CROP_CENTER_ZYX']}  (from env var)")
    else:
        print(f"  Centre (ZYX): auto-detect")

    volume = load_or_build_ct_volume(dicom_dir, cache_dir,
                                     full_volume=full_volume,
                                     center_zyx_voxel=center)
    mu = volume.mu_volume
    print()
    print(f"  Final μ shape:    {mu.shape}")
    print(f"  Spacing (zyx mm): {volume.spacing_zyx_mm}")
    print(f"  μ range:          [{mu.min():.5f}, {mu.max():.5f}] mm⁻¹")
    print(f"  μ mean:           {mu.mean():.5f} mm⁻¹")
    if mu.max() < 0.025:
        print("  WARNING: μ max very low — bone may not be visible in DRRs.")
    print("\n=== Done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())

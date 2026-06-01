"""TCP-injected into a running Isaac Sim GUI session.

Extracts the endo360 tool TIP mesh (link_1/visuals) from medical_scene.usd
and saves it in EE-local frame so the registration pipeline can splat it
into the phantom μ-volume regardless of where the robot moves the tool.

Frames:
    raw mesh local frame   — vertices as stored in the USD (meters)
    mesh world frame       — apply the link's ComputeLocalToWorldTransform
                              (this includes the metersPerUnit scaling that
                              maps the meter-scale mesh points to the cm-unit
                              stage; the result is in scene-unit cm)
    EE world frame         — endo360_needle (the TCP virtual frame; same
                              prim pose_from_medical_scene.py writes as
                              ee_pos/ee_quat in pose.json)
    EE-local frame (mm)    — what we save:
                              verts_ee_local_mm = R_ee_world.T @
                                  (verts_mesh_world_mm - ee_pos_world_mm)
                              with mm = scene-cm * 10 (after converting the
                              cm-unit stage values via metersPerUnit * 1000).

At registration time the pipeline only needs ee_pos/ee_quat (already in
pose.json); the stamp built from these EE-local verts moves rigidly with
the EE.

Outputs (in $HOME/isaac_projects/output/):
    tool_mesh.verts.npy   (N, 3) float32   EE-local mm  (x, y, z)
    tool_mesh.faces.npy   (F, 3) int32     triangle vertex indices
    tool_mesh.json                          source paths + bbox + frame info
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import omni.usd
from pxr import Gf, Usd, UsdGeom

OUT_DIR = Path("/home/max/isaac_projects/output")
# Comma-separated USD mesh prim paths.  Default = just the endo360 tip
# (link_1).  Set to e.g.
#     TOOL_MESH_PRIMS=/World/Robot/endo360_link_0/visuals,/World/Robot/endo360_link_1/visuals
# to bake the housing + tip together into one EE-local mesh (the "full tool"
# stamp scope).
MESH_PRIM_PATHS = [
    p.strip() for p in os.environ.pop(
        "TOOL_MESH_PRIMS",
        os.environ.pop("TOOL_MESH_PRIM", "/World/Robot/endo360_link_1/visuals"),
    ).split(",") if p.strip()
]
EE_PRIM_PATH = os.environ.pop("EE_PRIM",
                              "/World/Robot/endo360_needle")


def _to_np_mat4(gf_mat):
    """Convert a Gf.Matrix4d to a 4x4 numpy array (row-major, GfMatrix is also
    row-major so this is a direct copy)."""
    return np.array([[gf_mat[i][j] for j in range(4)] for i in range(4)],
                    dtype=np.float64)


def _world_pose_mat(stage, prim_path: str) -> np.ndarray:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise RuntimeError(f"Missing prim: {prim_path}")
    xf = UsdGeom.Xformable(prim)
    return _to_np_mat4(xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default()))


def _extract_one_mesh(stage, prim_path: str):
    """Read raw verts+faces from a single UsdGeom.Mesh prim. Returns (raw_verts, faces, T_mesh)."""
    mesh_prim = stage.GetPrimAtPath(prim_path)
    if not mesh_prim or not mesh_prim.IsA(UsdGeom.Mesh):
        raise RuntimeError(f"{prim_path} is not a UsdGeom.Mesh")
    mesh = UsdGeom.Mesh(mesh_prim)
    raw_verts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    fv_counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    fv_indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    if not np.all(fv_counts == 3):
        tris = []
        cursor = 0
        for n in fv_counts:
            base = fv_indices[cursor]
            for k in range(1, n - 1):
                tris.append([base, fv_indices[cursor + k], fv_indices[cursor + k + 1]])
            cursor += n
        faces = np.asarray(tris, dtype=np.int32)
    else:
        faces = fv_indices.reshape(-1, 3).astype(np.int32)
    T_mesh = _world_pose_mat(stage, prim_path)
    return raw_verts, faces, T_mesh


def main() -> None:
    stage = omni.usd.get_context().get_stage()
    mpu = stage.GetMetadata("metersPerUnit") or 1.0
    print(f"Stage metersPerUnit: {mpu}  (scene unit = {mpu*100:.2f} cm)")
    print(f"Extracting {len(MESH_PRIM_PATHS)} mesh prim(s):")
    for p in MESH_PRIM_PATHS:
        print(f"  - {p}")

    T_ee = _world_pose_mat(stage, EE_PRIM_PATH)
    scene_to_mm = mpu * 1000.0
    ee_pos_world_mm = T_ee[3, :3] * scene_to_mm
    row_norms = np.linalg.norm(T_ee[:3, :3], axis=1)
    R_ee_world = T_ee[:3, :3] / row_norms[:, None]
    print(f"  Row-norms of T_ee[:3,:3]: {row_norms}  (≈ 1/mpu = {1.0/mpu:.2f})")

    all_verts_ee_local = []
    all_faces = []
    vert_offset = 0
    per_mesh_info = []
    for prim_path in MESH_PRIM_PATHS:
        raw_verts, faces, T_mesh = _extract_one_mesh(stage, prim_path)
        raw_h = np.concatenate([raw_verts, np.ones((raw_verts.shape[0], 1))], axis=1)
        verts_world_scene = (raw_h @ T_mesh)[:, :3]
        verts_world_mm = verts_world_scene * scene_to_mm
        verts_ee_local_mm = (verts_world_mm - ee_pos_world_mm) @ R_ee_world.T

        # Reconstruction sanity per-prim
        verts_world_check = verts_ee_local_mm @ R_ee_world + ee_pos_world_mm
        rec_err = np.linalg.norm(verts_world_check - verts_world_mm, axis=1)
        if rec_err.max() > 1e-3:
            raise RuntimeError(f"{prim_path}: EE-local round-trip > 1 µm")

        all_verts_ee_local.append(verts_ee_local_mm)
        all_faces.append(faces + vert_offset)
        bb_min = verts_ee_local_mm.min(axis=0).tolist()
        bb_max = verts_ee_local_mm.max(axis=0).tolist()
        per_mesh_info.append({
            "prim":          prim_path,
            "n_verts":       int(verts_ee_local_mm.shape[0]),
            "n_faces":       int(faces.shape[0]),
            "ee_local_bbox": [bb_min, bb_max],
            "recon_err_max_mm": float(rec_err.max()),
        })
        print(f"  [{prim_path}] verts={verts_ee_local_mm.shape[0]:,} "
              f"faces={faces.shape[0]:,} "
              f"bbox z={bb_min[2]:.1f}..{bb_max[2]:.1f} mm  "
              f"recon_err_max={rec_err.max():.3e} mm")
        vert_offset += verts_ee_local_mm.shape[0]

    verts_ee_local_mm = np.concatenate(all_verts_ee_local, axis=0)
    faces = np.concatenate(all_faces, axis=0).astype(np.int32)

    # Sanity numbers
    bbox_min = verts_ee_local_mm.min(axis=0).tolist()
    bbox_max = verts_ee_local_mm.max(axis=0).tolist()
    extent  = (np.asarray(bbox_max) - np.asarray(bbox_min)).tolist()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "tool_mesh.verts.npy", verts_ee_local_mm.astype(np.float32))
    np.save(OUT_DIR / "tool_mesh.faces.npy", faces)

    meta = {
        "mesh_prims":          MESH_PRIM_PATHS,
        "per_mesh":            per_mesh_info,
        "ee_prim":              EE_PRIM_PATH,
        "stage_meters_per_unit": float(mpu),
        "n_verts":             int(verts_ee_local_mm.shape[0]),
        "n_faces":             int(faces.shape[0]),
        "ee_local_bbox_min_mm": bbox_min,
        "ee_local_bbox_max_mm": bbox_max,
        "ee_local_extent_mm":   extent,
        "frame":               "EE-local (x, y, z) in mm; rotates rigidly with ee_quat",
    }
    with open(OUT_DIR / "tool_mesh.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  ee_prim:      {EE_PRIM_PATH}")
    print(f"  combined verts: {verts_ee_local_mm.shape[0]:,}")
    print(f"  combined faces: {faces.shape[0]:,}")
    print(f"  EE-local bbox min (mm): "
          f"[{bbox_min[0]:.2f}, {bbox_min[1]:.2f}, {bbox_min[2]:.2f}]")
    print(f"  EE-local bbox max (mm): "
          f"[{bbox_max[0]:.2f}, {bbox_max[1]:.2f}, {bbox_max[2]:.2f}]")
    print(f"  EE-local extent   (mm): "
          f"[{extent[0]:.2f}, {extent[1]:.2f}, {extent[2]:.2f}]")
    print(f"Wrote {OUT_DIR / 'tool_mesh.verts.npy'}")
    print(f"Wrote {OUT_DIR / 'tool_mesh.faces.npy'}")
    print(f"Wrote {OUT_DIR / 'tool_mesh.json'}")


main()

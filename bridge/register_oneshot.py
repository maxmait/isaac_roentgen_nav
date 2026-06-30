#!/usr/bin/env python3
"""One-button registration: live scene → T_anatomy↔C-arm (and robot).

This is the "move the robot, press the button" entry point.  It chains the
three stages that were previously run by hand:

  1. CAPTURE  — inject the C-arm shot scripts into the running Isaac Sim GUI to
                sweep the C-arm to each view angle (default 0°, 45°, 90°) and
                write output/pose.json (ee / phantom / C-arm world poses +
                view_angles_deg).  Skip with --no-capture to reuse an existing
                pose.json.
  2. REGISTER — run the 6-DOF multi-view DRR registration in the fluorosim-torch
                Docker container (bridge/run_register_multiview.sh).
  3. COMPOSE  — bridge/compute_robot_to_anatomy.py composes the registered
                phantom pose with the robot EE pose → T_R^A, T_R^C, T_A^C and a
                clinical inside/outside check.

Then it prints a concise RESULT block (the transforms + the registration error).

Prereqs:
  - Isaac Sim GUI running with medical_scene.usd loaded (for the capture stage),
    the VSCode extension listening on 127.0.0.1:8226, and a /World/CArm prim
    (built by scenes/add_carm_viz.py).  Not needed with --no-capture.
  - Docker + the fluorosim-torch image (for the register stage).

Examples:
  # Full one-button run on the real spine CT (default), 3 views, with noise:
  python3 bridge/register_oneshot.py --noise

  # Re-register an already-captured pose.json without touching the scene:
  python3 bridge/register_oneshot.py --no-capture

  # Synthetic phantom, custom views, more iterations:
  python3 bridge/register_oneshot.py --synthetic --views 0,60,120 --n-iters 250
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
SCENES = PROJECT / "scenes"
BRIDGE = PROJECT / "bridge"
OUT = PROJECT / "output"
POSE_JSON = OUT / "pose.json"
TRACE_JSON = OUT / "registration_multiview" / "registration_trace.json"
RESULT_JSON = OUT / "robot_to_anatomy.json"

DEFAULT_DICOM = Path(os.path.expanduser(
    "~/medical_imaging/spine_mets_ct_seg/10250/04098/27242"))

sys.path.insert(0, str(SCENES))


def _hdr(step: str, msg: str) -> None:
    print(f"\n{'=' * 64}\n[{step}] {msg}\n{'=' * 64}", flush=True)


def capture_shots(views: list[float]) -> None:
    """Sweep the C-arm to each view angle and append a shot to pose.json."""
    from isaacsim_client import run_in_isaac  # noqa: E402

    rotate_src = (SCENES / "rotate_carm.py").read_text()
    shot_src = (SCENES / "add_carm_shot.py").read_text()

    for i, angle in enumerate(views):
        print(f"  view {i + 1}/{len(views)}: rotating C-arm to {angle:+.1f}° …",
              flush=True)
        run_in_isaac(f"CARM_ROTATION_DEG={angle}\n" + rotate_src, timeout=60.0)
        reset = "RESET_SHOTS=1\n" if i == 0 else ""
        out = run_in_isaac(reset + shot_src, timeout=60.0)
        for line in out.strip().splitlines():
            print(f"      {line}")

    if not POSE_JSON.exists():
        raise SystemExit(f"ERROR: capture did not produce {POSE_JSON}")
    pose = json.loads(POSE_JSON.read_text())
    print(f"  Captured view_angles_deg = {pose.get('view_angles_deg')}")


def run_registration(args) -> None:
    env = os.environ.copy()
    env["USE_POSE_JSON"] = "1"
    env["USE_CARM_ROTATION"] = "1"   # use the captured view_angles_deg list
    env["N_ITERS"] = str(args.n_iters)
    if not args.synthetic:
        if not args.dicom.is_dir():
            print(f"  WARNING: DICOM dir {args.dicom} not found — "
                  f"falling back to the synthetic phantom.")
            args.synthetic = True
        else:
            env["DICOM_PATH"] = str(args.dicom)
            if args.full_ct:
                env["CT_FULL_VOLUME"] = "1"
    if args.noise:
        env["DRR_NOISE"] = "1"
        if args.noise_seed is not None:
            env["DRR_NOISE_SEED"] = str(args.noise_seed)
    src = "synthetic phantom" if args.synthetic else f"CT {args.dicom}"
    print(f"  source: {src}  |  N_ITERS={args.n_iters}  |  "
          f"noise={'on' if args.noise else 'off'}", flush=True)

    wrapper = BRIDGE / "run_register_multiview.sh"
    proc = subprocess.run(["bash", str(wrapper)], env=env)
    if proc.returncode != 0:
        raise SystemExit(f"ERROR: registration failed (exit {proc.returncode})")
    if not TRACE_JSON.exists():
        raise SystemExit(f"ERROR: no registration trace at {TRACE_JSON}")


def compose_transforms(no_plot: bool) -> None:
    cmd = [sys.executable, str(BRIDGE / "compute_robot_to_anatomy.py")]
    if no_plot:
        cmd.append("--no-plot")
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(f"ERROR: compute_robot_to_anatomy failed "
                         f"(exit {proc.returncode})")


def _quat_to_axis_angle_deg(q):
    import math
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w /= n
    angle = 2.0 * math.degrees(math.acos(max(-1.0, min(1.0, abs(w)))))
    return angle


def print_result(no_plot: bool) -> None:
    res = json.loads(RESULT_JSON.read_text())
    rec = res["recovered"]
    err = res["errors"]
    trace = json.loads(TRACE_JSON.read_text())
    ig = trace.get("isaac_ground_truth", {})

    def fmt(t):
        return "[" + ", ".join(f"{v:8.3f}" for v in t) + "]"

    _hdr("RESULT", "Recovered transforms (translation mm, rotation as quat wxyz)")
    tac = rec["T_anatomy_in_carm"]
    tra = rec["T_robot_in_anatomy"]
    trc = rec["T_robot_in_carm"]
    print(f"  T_anatomy→C-arm   t={fmt(tac['t_mm'])}  q={[round(v,4) for v in tac['R_quat_wxyz']]}")
    print(f"  T_robot→anatomy   t={fmt(tra['t_mm'])}  q={[round(v,4) for v in tra['R_quat_wxyz']]}")
    print(f"  T_robot→C-arm     t={fmt(trc['t_mm'])}")
    print()
    print(f"  Registration world ‖t_err‖ : {ig.get('world_err_norm_mm', float('nan')):.4f} mm")
    print(f"  Rotation geodesic err      : {err.get('phantom_rot_err_geodesic_deg', 0.0):.4f} °")
    print(f"  T_anatomy→C-arm ‖t_err‖    : {err['T_anatomy_in_carm_t_err_norm_mm']:.4f} mm")
    print(f"  Clinical                   : {res['clinical']['interpretation']}")
    print()
    print(f"  Full detail: {RESULT_JSON}")
    print(f"  (T_anatomy→robot is the inverse of T_robot→anatomy above.)")
    if not no_plot:
        layout_png = OUT / "robot_to_anatomy_layout.png"
        if layout_png.exists():
            print(f"  Deliverable figure: {layout_png}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="One-button live-scene → anatomy/C-arm registration.")
    ap.add_argument("--views", default="0,45,90",
                    help="comma C-arm view angles in deg (default 0,45,90; "
                         "≥3 with an oblique view recommended for blind 6-DOF)")
    ap.add_argument("--no-capture", action="store_true",
                    help="skip the Isaac Sim capture; reuse existing pose.json")
    ap.add_argument("--dicom", type=lambda p: Path(os.path.expanduser(p)),
                    default=DEFAULT_DICOM, help="DICOM CT dir (default: spine CT)")
    ap.add_argument("--synthetic", action="store_true",
                    help="use the synthetic ellipsoid phantom instead of a CT")
    ap.add_argument("--full-ct", action="store_true",
                    help="use the full-volume CT cache (CT_FULL_VOLUME=1)")
    ap.add_argument("--n-iters", type=int, default=200,
                    help="optimizer iterations (default 200; 6-DOF needs ~200)")
    ap.add_argument("--noise", action="store_true",
                    help="add realistic noise/blur to the target DRRs (DRR_NOISE=1)")
    ap.add_argument("--noise-seed", type=int, default=None,
                    help="fix the noise realisation (DRR_NOISE_SEED)")
    ap.add_argument("--no-plot", action="store_true",
                    help="skip the deliverable layout figure "
                         "(robot_to_anatomy_layout.png)")
    args = ap.parse_args()

    views = [float(v) for v in args.views.split(",") if v.strip()]
    t0 = time.time()

    if args.no_capture:
        _hdr("1/3 CAPTURE", "skipped (--no-capture); using existing pose.json")
        if not POSE_JSON.exists():
            raise SystemExit(f"ERROR: --no-capture but {POSE_JSON} is missing.")
    else:
        _hdr("1/3 CAPTURE", f"sweeping C-arm to views {views}° in the live scene")
        try:
            capture_shots(views)
        except ConnectionRefusedError:
            raise SystemExit(
                "ERROR: cannot reach Isaac Sim at 127.0.0.1:8226.\n"
                "  Open the GUI with scenes/medical_scene.usd and ensure the "
                "VSCode extension is listening, or pass --no-capture.")

    _hdr("2/3 REGISTER", "6-DOF multi-view DRR registration (Docker)")
    run_registration(args)

    _hdr("3/3 COMPOSE", "composing T_robot_to_anatomy from registration + pose.json")
    compose_transforms(args.no_plot)

    print_result(args.no_plot)
    print(f"\n[done] total wall time {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

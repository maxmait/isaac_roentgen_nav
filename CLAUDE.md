# Project: Fluoroscopy-Based Robot Pose Estimation in Isaac Sim

## One-Line Summary
A simulation pipeline where a robot's pose relative to anatomy is estimated 
using simulated fluoroscopy (DRR) and 2D/3D CT registration — entirely in software.

---

## Hardware
- GPU: NVIDIA RTX 4060 Laptop (8GB VRAM, Compute Capability 8.9, RT Cores ✓)
- RAM: 32GB
- CPU: AMD Ryzen 7 7840HS
- OS: Ubuntu 22.04 LTS
- NVIDIA Driver: 590.48.01
- CUDA: 13.1 (driver), 12.6 (container)

---

## Software Stack
- Isaac Sim 4.5.0 — installed standalone at ~/isaacsim
  - Run standalone scripts with: ~/isaacsim/python.sh
  - No conda environment needed — Isaac Sim has its own embedded Python at
    /home/max/isaacsim/kit/python/bin/python3
  - CRITICAL: always launch Isaac Sim from a terminal with conda deactivated.
    If conda base is active, the wrong site-packages are used and pip installs
    go to ~/.local instead of Isaac Sim's kit Python.
  - Assets take ~10 seconds to load before script output appears, this is normal
- Docker + NVIDIA Container Toolkit — for fluorosim
- fluorosim — runs inside Docker container, isolated from Isaac Sim
- pyzmq 27.1.0 — installed into Isaac Sim's kit Python site-packages:
    /home/max/isaacsim/kit/python/lib/python3.10/site-packages

## Project Directory Structure
~/isaac_projects/                  # local working dir; GitHub repo = fluoro-pose-estimation
  README.md                        # public-facing intro
  LICENSE                          # Apache-2.0
  CLAUDE.md                        # this file — internal implementation notes
  pyproject.toml                   # host-side Python deps (pyzmq, matplotlib, numpy, Pillow)
  .gitignore                       # ignores output/, scenes/snapshots/, __pycache__, etc.
  scenes/                          # Isaac Sim side (USD scenes + host tooling)
    robot_scene.py                 # headless Franka + phantom scene, writes pose.json
    phantom.py                     # shared phantom geometry (single source of truth)
    verify_pose.py                 # TCP-injected: live EE pose vs pose.json
    verify_phantom.py              # TCP-injected: phantom prim world transform
    isaacsim_client.py             # TCP client for VSCode extension socket (port 8226)
    image_publisher.py             # TCP-injected: ZMQ viewport JPEG stream (port 5556)
    take_snapshot.py               # ZMQ SUB: save one frame from the stream
    medical_scene.usd              # Phase 5h: OR scene — table, STAR robot, phantom, C-arm
    inject_medical_scene.py        # TCP-injected: builder for medical_scene.usd contents
    pose_from_medical_scene.py     # TCP-injected: read live world poses → pose.json
    add_carm_viz.py                # TCP-injected: build /World/CArm visualisation
    rotate_carm.py                 # TCP-injected: rotate C-arm around patient long axis
    add_carm_shot.py               # TCP-injected: append C-arm angle to view_angles_deg
    capture_two_shots.py           # TCP-injected: timed two-shot capture (GUI manipulator)
    snapshots/                     # runtime viewport snapshots (gitignored)
  bridge/                          # fluorosim side (Docker-bound scripts + wrappers)
    fluorosim_render.py            # runs INSIDE the container; reads pose.json, renders DRR
    run_fluorosim.sh               # docker run wrapper with proper bind mounts
    visualize_drr.py               # host-side annotated viewer + flat-image sanity check
    fluorosim_torch.Dockerfile     # builds `fluorosim-torch` (= fluorosim + PyTorch) for registration
    register_phantom.py            # Phase 5: translation-only phantom-pose registration (single view)
    run_register.sh                # docker run wrapper for register_phantom.py
    plot_registration.py           # host-side: loss curve + parameter trajectory + image comparison
    register_phantom_multiview.py  # Phase 5: clinical sequential AP + lateral C-arm shots
    run_register_multiview.sh      # docker run wrapper for the multi-view registration
    plot_registration_multiview.py # host-side: per-view losses + per-view target/recovered/diff panels
    compute_robot_to_anatomy.py    # Phase 5 deliverable: composes registered phantom with EE pose
                                   # → T_R^A, T_R^C, T_A^C + clinical inside/outside check
    ct_loader.py                   # Phase 5d: DICOM CT → PreprocessedVolume (cropped ROI or full)
    run_load_ct.sh                 # docker wrapper to pre-warm the CT cache (one-time)
                                   # CT_FULL_VOLUME=1 → full 797×512×512 native-spacing volume
  docs/images/                     # screenshots committed to the repo (README assets)
  output/                          # runtime artifacts: pose.json, drr.*, fluorosim_cache/ (gitignored)

External repos (kept OUT of the project tree, gitignored if re-cloned in-place):
  ~/nvidia-third-party/i4h-sensor-simulation/   # fluorosim source; Docker image is built from here
  ~/nvidia-third-party/i4h-workflows/           # not used; kept around as an asset source (CT scans)

---

## Architecture

Layer 1: Isaac Sim Scene (standalone Python, ~/isaacsim/python.sh)
  - Franka robot arm loaded via isaacsim.robot.manipulators.examples.franka
  - Outputs: end effector pose (xyz + quaternion), joint angles
  - Currently: runs and exits cleanly

Layer 2: CT Phantom (future)
  - Source: synthetic numpy volume OR TCIA DICOM phantom
  - Used as: mesh in Isaac Sim scene + raw CT volume for fluorosim
  - Keep volume at ≤256³ to fit in 8GB VRAM

Layer 3: Fluoroscopy Simulator (Docker, isolated)
  - Repo: i4h-sensor-simulation/fluoro-simulator
  - Docker image: fluorosim (built locally)
  - Input: CT volume + C-arm pose + robot tool pose
  - Output: DRR image (numpy array / PNG)
  - Runs at ~155 FPS on RTX 4060 with 256³ synthetic volume ✓
  - Differentiable mode enabled (autodiff for registration) ✓
  - Communication with Isaac Sim: file-based exchange (pose in, DRR out)

Layer 4: 2D/3D Registration (future)
  - Use fluorosim's built-in Slang autodiff
  - Gradient-based pose optimization
  - Minimize image similarity loss between rendered and reference DRR
  - Recover T_robot_to_anatomy

Layer 5: Neural Network (optional, future)
  - Use pipeline as data generator (pose → DRR pairs)
  - Only pursue after classical registration is validated

## Key Transform
T_robot_to_anatomy = inv(T_anatomy_in_carm) * T_robot_in_carm

## World ↔ Isocenter Transform (Phase 4)

The CT volume in fluorosim is parameterized in a volume-local frame whose
origin is the volume center ("isocenter"). The Isaac Sim scene places that
isocenter at PHANTOM_POS_WORLD_M in the world frame. fluorosim's `translation`
input is the C-arm pose expressed *in the volume-local frame*:

    translation_mm = (carm_pos_world - phantom_pos_world) * 1000

Phantom rotation is assumed identity for now; once the phantom is allowed to
rotate, the C-arm orientation must be composed with the inverse phantom
rotation before being fed to fluorosim.

Axis mapping (fluorosim volume-local → Isaac Sim world):
    Z (slow voxel axis) ↔ Z (up)
    Y                    ↔ Y
    X (fast voxel axis)  ↔ X

Shared constants live in scenes/phantom.py (PHANTOM_POS_WORLD_M, semiaxes,
prim paths) so the Isaac Sim mesh and the fluorosim volume can never drift
out of sync.

---

## Claude Code Integration

### Purpose
Claude Code connects to the running Isaac Sim instance to execute Python,
take viewport snapshots, and visually verify simulation state without
needing a separate GUI session.

### Key Paths
- **Isaac Sim install root**: `/home/max/isaacsim/`
- **VSCode extension source**: `/home/max/isaacsim/exts/isaacsim.code_editor.vscode/isaacsim/code_editor/vscode/`
  - `executor.py`  — async Python execution engine (eval → exec, supports await)
  - `extension.py` — TCP server, reads port from Carb settings, spawns Executor
- **Extension config**: `/home/max/isaacsim/exts/isaacsim.code_editor.vscode/config/extension.toml`
- **Camera extension API**: `/home/max/isaacsim/exts/isaacsim.sensors.camera/`
- **Viewport snapshots**: `~/isaac_projects/scenes/snapshots/`

### Connection
- **VSCode extension socket**: `127.0.0.1:8226`
  - Default set in `extension.toml`: `exts."isaacsim.code_editor.vscode".port = 8226`
  - Can be overridden via Carb settings key: `/exts/isaacsim.code_editor.vscode/port`
  - Verify while Isaac Sim is running: `ss -tlnp | grep 8226`
  - Protocol: send Python source as UTF-8 over TCP; server replies with JSON then closes:
      {"status": "ok"|"error", "output": "...", "ename": "...", "evalue": "...", "traceback": [...]}
  - Do NOT call socket.shutdown(SHUT_WR) before reading — the server replies
    asynchronously via asyncio and the race will cause an empty response.
- **Default camera prim**: `/OmniverseKit_Persp`
- **Isaac Sim kit Python sys.path** (what the running process sees):
    /home/max/isaacsim/kit/python/lib/python3.10/site-packages   ← pip install target
    /home/max/isaacsim/kit/kernel/py
    /home/max/isaacsim/kit/plugins/bindings-python
    /home/max/.local/share/ov/data/Kit/Isaac-Sim Full/4.5/pip3-envs/default

### Tooling Scripts (`scenes/`)

**isaacsim_client.py** — Python helper to send code to Isaac Sim via TCP
- `run_in_isaac(code, timeout=30)` → returns stdout string, raises RuntimeError on error
- Parses the JSON reply from the VSCode extension
- CLI: `python3 isaacsim_client.py "print(1+1)"`  or  `python3 isaacsim_client.py < script.py`

**image_publisher.py** — inject into running Isaac Sim to stream viewport frames
- Inject: `python3 isaacsim_client.py "$(cat image_publisher.py)"`
- Attaches `rep.AnnotatorRegistry.get_annotator("rgb")` to the EXISTING viewport
  render product (`get_active_viewport().render_product_path`) — NOT a new
  `rep.create.render_product()`, which only renders on orchestrator.step() calls.
- Drives capture via an async loop calling `rep.orchestrator.step_async()` at 10 FPS.
- Publishes multipart ZMQ frames: `[b"frame", <JPEG bytes>]` on `tcp://127.0.0.1:5556`
- Exposes `take_snapshot(filename?)` and `stop_publisher()` in Isaac Sim's namespace.
- Detects injection vs standalone via `"omni.usd" in sys.modules`.

**take_snapshot.py** — run outside Isaac Sim to save one frame from the ZMQ stream
- `python3 take_snapshot.py [filename.jpg]`
- Connects a ZMQ SUB socket, waits up to 5s for a frame, saves to `snapshots/`
- Requires pyzmq in the system/conda Python (separate from Isaac Sim's kit Python)

---

## What Has Been Decided / Why
- NOT using i4h-workflows as base: requires ≥24GB VRAM, ≥64GB RAM, 
  RTI Connext DDS license — incompatible with available hardware
- NOT using IsaacLab: adds AppLauncher and RL training overhead, 
  caused NCCL/PyTorch crashes, not needed for this project
- NOT using RTI Connext DDS: unnecessary middleware, was causing crashes
- NOT using GR00T: wrong hardware, not relevant to registration goal
- Using Docker for fluorosim: avoids dependency conflicts with Isaac Sim,
  most stable and tested installation path
- Using file-based IPC between Isaac Sim and fluorosim: simple, 
  inspectable, no middleware required
- Using ZMQ for viewport streaming: lightweight, no middleware, avoids file-polling
  latency. pyzmq must be installed to Isaac Sim's kit Python site-packages directly.
- NOT using isaacsim.sensors.camera for viewport capture: that API creates its own
  render product and requires a full World/simulation loop to drive frame events.
  Instead, using rep.AnnotatorRegistry on the existing viewport render product driven
  by rep.orchestrator.step_async() — this works in both GUI and standalone modes.
- NOT calling world.reset() in TCP injection mode: blocks indefinitely without a
  running simulation loop. Use USD API directly to add prims in TCP context.

---

## Sequential Plan

### Phase 1: Isaac Sim Scene ✅ COMPLETE
- [x] Isaac Sim 4.5.0 installed and launches
- [x] Standalone Python script runs via ~/isaacsim/python.sh
- [x] Franka arm loads in scene
- [x] End effector pose (xyz + quaternion) readable each step
- [x] Joint angles readable each step
- [x] Script exits cleanly

### Phase 2: Fluoroscopy Simulator ✅ COMPLETE  
- [x] Docker + NVIDIA Container Toolkit working
- [x] fluorosim Docker image built successfully
- [x] Synthetic CT volume preprocessed (128x256x256)
- [x] DRR rendering working at 155 FPS on RTX 4060
- [x] Differentiable mode confirmed enabled
- [x] Output frames saved to host via volume mount

### Phase 2.5: Claude Code ↔ Isaac Sim Tooling ✅ COMPLETE
- [x] Verify VSCode extension socket is live: `ss -tlnp | grep 8226`
- [x] TCP protocol understood: JSON reply, parsed by isaacsim_client.py
- [x] isaacsim_client.py — `run_in_isaac(code)` helper, CLI and library modes
- [x] image_publisher.py — ZMQ viewport streamer, inject via isaacsim_client.py
- [x] take_snapshot.py — ZMQ SUB subscriber, saves JPEG to snapshots/
- [x] pyzmq installed to Isaac Sim's kit Python site-packages
- [x] Validated snapshot workflow: Franka arm visible in captured frame
- [x] snapshots/ directory created

### Phase 3: Connect Isaac Sim → Fluorosim ✅ COMPLETE
- [x] Pose data format defined (pose.json: ee_pos[m], ee_quat[wxyz], carm_pos[m], carm_quat[wxyz], timestamp)
- [x] robot_scene.py writes pose.json atomically each step (.tmp + os.replace)
- [x] verify_pose.py reads live EE pose via TCP injection, compares to file
- [x] fluorosim_render.py (in Docker) reads pose.json, converts quat→Euler/m→mm, renders DRR
- [x] run_fluorosim.sh wraps the Docker invocation with bind mounts (host output ↔ /workspace/io)
- [x] visualize_drr.py shows DRR + pose annotation, with std-based sanity check
- [x] Validated end-to-end: synthetic sphere phantom (bone core + soft tissue shell) visible in DRR
- [ ] DEFERRED to Phase 4: robot tool not yet in DRR (needs CT phantom + world→isocenter transform)

### Phase 4: CT Phantom Integration 🚧 IN PROGRESS
- [x] Synthetic ellipsoid phantom is shared between Isaac Sim and fluorosim
      via scenes/phantom.py (constants: position, semiaxes, prim paths, colors)
- [x] robot_scene.py creates two UsdGeom.Sphere prims (soft tissue + bone core)
      at /World/Phantom/{SoftTissue,BoneCore}, non-uniform scale = ellipsoid
      semiaxes derived from the fluorosim μ-volume voxel geometry
- [x] pose.json schema extended with phantom_pos / phantom_quat fields
- [x] World→isocenter transform applied in fluorosim_render.py:
      translation_mm = (carm_pos - phantom_pos) * 1000
- [x] Verified centered: carm_pos == phantom_pos → DRR centroid at (256, 256)
- [x] Verified scale: +20mm X shift → centroid moves exactly -80px (= 2× SDD/SID
      magnification at 0.5mm/px pixel spacing); confirms isocenter mapping
- [x] verify_phantom.py inspects /World/Phantom/* prims in live GUI stage
- [x] Viewport snapshot in snapshots/phantom_in_isaac.jpg shows prolate ellipsoid
- [x] Phantom isocenter moved to (0.43, 0, 0.42) m so the Franka rest-pose EE
      lies inside the volume bounds (±64 mm); previously the EE was 70 mm out
      in X and 120 mm out in Z, so no tool blob could ever intersect a ray.
- [x] paint_tool_into_volume() in fluorosim_render.py burns a 15 mm dense sphere
      (μ=0.5 mm⁻¹, ~28× bone) at the EE voxel before each render; the cached
      μ_volume.npy is left clean, the tool is applied to an in-memory copy.
      Returns a new PreprocessedVolume because the public mu_volume is a
      read-only property.
- [x] Verified projection: shifting ee_pos by +30 mm in world Y moved the tool
      centroid by exactly +121 px in detector Y (predicted 120 px = 30 × 2 / 0.5).
      Confirms world→volume→detector geometry end-to-end.
- [ ] Replace analytic ellipsoid + sphere abstraction with marching-cubes
      meshes (real CT phantom, real Franka hand geometry) once we have the
      source data.

### Phase 5: Registration Pipeline 🚧 IN PROGRESS
- [x] Translation-only phantom-pose registration via fluorosim Slang autodiff
      (bridge/register_phantom.py); SlangDiffDRRRenderer + custom Adam over
      translation only (rotation tensor frozen).
- [x] Derived Docker image bridge/fluorosim_torch.Dockerfile (=fluorosim-torch)
      adds PyTorch 2.5.1 + cu121 to the fluorosim base — fluorosim's torch
      autograd path is optional and not in the base image.
- [x] Synthetic-to-synthetic convergence verified: 19.7 mm initial error →
      0.41 mm final error in 100 Adam iters (5.3 s on RTX 4060); loss
      reduction 4.7e-2 → 1.6e-6 (29 000×). Per-axis err: x=0.04, y=0.02,
      z=0.41 mm. Z error dominates because Z is the X-ray beam axis and
      depth is poorly constrained by a single view (well-known single-view
      depth ambiguity in 2D/3D registration).
- [x] bridge/plot_registration.py renders convergence + image comparison plots.
- [x] Registration against Isaac Sim ground truth: register_phantom.py reads
      pose.json (phantom_pos / carm_pos), uses fluorosim translation =
      (carm-phantom)*1000 as GT, optimizes, inverts back to world-frame
      phantom_pos and reports error against Isaac Sim's value.
      Trivial case (carm==phantom, GT translation=0): world-frame ||err|| = 0.41 mm.
      Non-trivial (carm=phantom+30mm X, GT translation=(30,0,0)): ||err|| = 0.16 mm.
      Both recover the Isaac Sim phantom_pos to sub-mm in world coordinates.
- [x] Multi-view registration (AP + lateral, clinical sequential workflow).
      bridge/register_phantom_multiview.py models the OR sequence: take AP shot,
      rotate C-arm 90°, take lateral shot, register against both. Shared
      translation parameter, summed MSE across views. Depth ambiguity is
      collapsed: Z error went from 0.41 mm (single view) → 0.013 mm (8×
      improvement on ||err||: 0.41 mm → 0.05 mm = ~50 μm vs Isaac Sim GT).
      Z is now the BEST-constrained axis because BOTH views contribute to it.
- [x] bridge/compute_robot_to_anatomy.py composes the multi-view registered
      phantom_pos with the (Isaac-Sim-known) robot EE pose to produce T_R^A,
      T_R^C, T_A^C in both ground-truth and recovered forms. Error in T_R^A
      equals the registration's world_err (asserted internally) — 0.051 mm with
      multi-view, 0.411 mm with single-view fallback. Includes a clinical
      summary (EE position in anatomy frame; normalized-ellipsoid inside/outside
      check vs bone core and soft tissue) and a 2D layout PNG showing EE in
      anatomy frame.
- [x] Phase 5d — real CT integration on the registration side (no Isaac Sim yet).
      bridge/ct_loader.py loads a DICOM series via SimpleITK. Two modes:
        Cropped (default): auto-detect vertebral body level, crop 128 mm³ ROI,
          resample to (128,256,256) @ (1,0.5,0.5) mm → fluorosim_cache_ct/.
          0.096 mm error, 8.2 s (82 ms/iter) on the spine CT.
        Full volume (CT_FULL_VOLUME=1): entire scan at native anisotropic spacing
          — (797,512,512) @ (0.5,0.703,0.703) mm = 836 MB, ~1.7 GB VRAM — fits
          in 8 GB; cache at fluorosim_cache_ct_full/; 0.076 mm, 158 ms/iter.
      Both caches coexist; DICOM_PATH + CT_FULL_VOLUME + CT_CROP_CENTER_ZYX
      flags select the path. Synthetic path unchanged (no DICOM_PATH).
      Key fixes made during 2026-05-19 development (see State Log):
        1. VolumePreprocessor HU→μ LUT clamped at μ_water — bone invisible.
           Fixed with bilinear hu_to_mu() in ct_loader.py; μ now reaches
           0.065 mm⁻¹ for cortical bone vs water at 0.020.
        2. Default crop center was geometric volume center — landed in the
           head/neck for the skin-to-skin spine CT, not the spine.  Fixed
           with find_vertebral_center() auto-detection (compact-bone heuristic)
           + CT_CROP_CENTER_ZYX env var for manual override.
        3. DRR display "overexposed" — gamma=0.5 brightened mid-tones on
           already-inverted [0,1] images. Fixed: undo invert, apply
           log1p(x×50), p2–p98 clip in plot_registration_multiview.py.
           Display is completely decoupled from the registration optimizer.
- [x] Phase 5e — Isaac Sim CT mesh.  bridge/ct_to_mesh.py runs marching cubes
      on the cached μ-volume (skimage), applies Laplacian smoothing, and saves
      output/spine_mesh.{obj,verts.npy,faces.npy,json}.  scenes/robot_scene.py
      loads the .npy mesh into a UsdGeom.Mesh at /World/Phantom/Mesh; the
      synthetic ellipsoid is the fallback when the mesh is missing.  Verified
      visually via TCP injection: 238k verts / 473k faces, anatomically
      recognisable vertebral column with rib heads.  Mesh isocenter aligned
      with PHANTOM_POS_WORLD_M=(0.43,0,0.42) m, same as the synthetic phantom.
- [x] Phase 5f — close the USE_POSE_JSON=1 loop on the CT.  The CT mesh's
      isocenter in robot_scene.py is at PHANTOM_POS_WORLD_M, so pose.json
      ground-truth applies to both phantom types.  compute_robot_to_anatomy.py
      now does a data-driven inside-bone/soft-tissue check by μ-volume
      lookup at the EE position in anatomy frame (replaces the analytic
      ellipsoid distance check when a CT cache exists), and the layout PNG
      shows axial+coronal μ-slices through the isocenter with the EE marker
      overlaid.  End-to-end test on the spine CT: world ||err|| = 0.096 mm;
      EE at (0.72, 15.39, -2.91) mm in anatomy frame, local μ=0.0225 mm⁻¹
      → tool tip in soft tissue, anatomically consistent with a position
      just anterior to the vertebral body.
- [x] Phase 5h — surgical scene + end-to-end pipeline on a saved USD stage.
      scenes/medical_scene.usd is a hand-built operating room: walls/floor +
      lights + Operating_table (UsdGeom) + RobotBase cube + STAR robot
      (i4h-assets STAR + endo360 tool) + CT spine mesh /World/Phantom/SpineMesh
      on the table.  Y-up, 1 unit = 1 cm — diverges from the Z-up Franka world
      that robot_scene.py uses, so a separate ingestion script is needed
      (the registration math is frame-agnostic — only pose.json values in
      metres matter).
      STAR has UsdPhysics.DriveAPI authored on every revolute joint
      (stiffness 1e9, damping 1e7, force-type drive) so the arm holds
      whatever pose the user sets in the GUI without falling under gravity.
      scenes/pose_from_medical_scene.py is TCP-injected to extract live
      world-frame poses from the open stage and write pose.json in metres —
      no full Isaac Sim launch needed for the registration loop.
      End-to-end: pose robot in GUI → inject pose script → run register →
      compute_robot_to_anatomy.py.  World ||err|| = 0.096 mm.
- [x] Phase 5i — C-arm USD visualisation + angle-driven DRR view.
      scenes/add_carm_viz.py builds /World/CArm: source cube + detector
      plate + arc + beam-axis line, sized to fluorosim's SDD=1020 mm /
      SID=510 mm so the GUI geometry matches the math.  Source/detector
      sit on the local Y axis; rotating the C-arm xform around the patient
      long axis (world Z in Y-up scenes) sweeps source and detector around
      the patient — clinically correct AP→oblique→lateral motion.
      scenes/rotate_carm.py rotates the C-arm by a given angle around the
      patient long axis; scenes/pose_from_medical_scene.py extracts that
      angle from the C-arm world quaternion and writes it as
      `carm_rotation_y_deg` (legacy field name).
      bridge/register_phantom_multiview.py with USE_CARM_ROTATION=1 reads
      the angle and uses it as the AP view; lateral = AP + 90°.
- [x] Phase 5j — two-shot / N-shot C-arm capture workflow.
      Two scripts cover both interaction modes:
        scenes/capture_two_shots.py    — timed (~3 s wait between shots),
          pumps the Isaac Sim event loop so the GUI manipulator stays
          responsive during the wait window.  Useful for GUI-driven flows.
        scenes/add_carm_shot.py        — append-one-shot, called once per
          C-arm pose; RESET_SHOTS=1 on the first call clears the list.
          Best for command-line / scriptable workflows.
      Either script writes `view_angles_deg: [...]` to pose.json.
      bridge/register_phantom_multiview.py with USE_CARM_ROTATION=1 reads
      that list directly — no auto +90° lateral.  Tested with views
      [0°, 60°] and [0°, 90°]: 0.096–0.106 mm world error.
- [ ] 6-DOF (translation + rotation) registration — when this lands, plug the
      recovered phantom rotation into compute_robot_to_anatomy.py's phantom_W_rec
- [ ] Tool in the registration target image — currently the registration
      renders its target DRR *without* the EE blob (see "Simulation
      simplifications" in Known Issues).  Adding the tool requires either
      a tool-masking loss or robust matching; deferred.
- [ ] Blind optimiser init — currently the optimiser starts at
      `gt_translation_mm + INIT_OFFSET_MM` (simulating a planning prior
      with ~20 mm noise).  A stricter "init from carm_pos only" would be
      truly independent of GT; deferred.

### Phase 6: Neural Network Acceleration 🔲 (optional/future)
- [ ] Generate training dataset: random poses → DRR pairs
- [ ] Train pose regression network
- [ ] Compare vs classical registration baseline

---

## State Log

| Date       | Action                                          | Result                                      |
|------------|-------------------------------------------------|---------------------------------------------|
| 2026-05-12 | Attempted i4h robotic ultrasound workflow       | ❌ NCCL mismatch, RTI DDS license missing   |
| 2026-05-12 | Assessed hardware vs i4h requirements           | ❌ Incompatible — decided on custom pipeline |
| 2026-05-12 | Deleted all environments and repos              | ✅ Clean slate                              |
| 2026-05-12 | Docker + NVIDIA Container Toolkit verified      | ✅ nvidia-smi works inside container        |
| 2026-05-12 | fluorosim Docker image built                    | ✅ Successful                               |
| 2026-05-12 | fluorosim synthetic demo                        | ✅ 155 FPS, differentiable mode enabled     |
| 2026-05-12 | DRR output saved to host machine                | ✅ 20 frames rendered, LAO/RAO sweep        |
| 2026-05-12 | Isaac Sim standalone Python script              | ✅ Franka loads, EE pose readable headless  |
| 2026-05-13 | Verified VSCode ext port (extension.toml)       | ✅ Port is 8226 (not 8826)                  |
| 2026-05-13 | Located camera + VSCode extension source        | ✅ Paths confirmed, image_publisher planned |
| 2026-05-13 | Implemented isaacsim_client.py                  | ✅ TCP client with JSON reply parsing       |
| 2026-05-13 | Implemented image_publisher.py                  | ✅ ZMQ PUB at 10 FPS via rep.orchestrator  |
| 2026-05-13 | Implemented take_snapshot.py                    | ✅ ZMQ SUB subscriber, saves JPEG          |
| 2026-05-13 | Installed pyzmq into Isaac Sim kit Python       | ✅ v27.1.0 in kit/python/lib/.../site-pkgs |
| 2026-05-13 | Validated full snapshot pipeline                | ✅ Franka arm confirmed visible in frame   |
| 2026-05-15 | Modified robot_scene.py to write pose.json      | ✅ Atomic .tmp+os.replace, all 500 steps    |
| 2026-05-15 | Implemented verify_pose.py (TCP injection)      | ✅ Reads live EE pose, compares to file     |
| 2026-05-15 | Implemented fluorosim_render.py (Docker)        | ✅ pose.json → DRR + drr_meta.json output   |
| 2026-05-15 | Implemented run_fluorosim.sh wrapper            | ✅ Bind mounts host output → /workspace/io |
| 2026-05-15 | Implemented visualize_drr.py + std sanity check | ✅ drr_annotated.png with pose overlay      |
| 2026-05-15 | End-to-end Isaac Sim → fluorosim → DRR          | ✅ Synthetic sphere phantom clearly visible |
| 2026-05-15 | Factored phantom geometry into scenes/phantom.py | ✅ Shared constants for Isaac Sim + fluorosim |
| 2026-05-15 | Added phantom ellipsoids to Isaac Sim stage     | ✅ /World/Phantom/{SoftTissue,BoneCore} prims |
| 2026-05-15 | Added phantom_pos/quat to pose.json schema      | ✅ Backward-compatible (defaults if missing) |
| 2026-05-15 | World→isocenter transform in fluorosim_render   | ✅ translation_mm = (carm - phantom) × 1000  |
| 2026-05-15 | Centroid test (carm == phantom)                 | ✅ Sphere centered at (256, 256) px in DRR   |
| 2026-05-15 | Magnification test (+20mm X shift)              | ✅ Centroid at (176, 256.5) — exactly -80 px |
| 2026-05-15 | verify_phantom.py + viewport snapshot           | ✅ Prims at expected world transform + visible |
| 2026-05-15 | Phantom moved to (0.43, 0, 0.42) for EE overlap | ✅ EE rest pose now inside ±64 mm volume bounds |
| 2026-05-15 | paint_tool_into_volume() in fluorosim_render    | ✅ Tool sphere (μ=0.5, r=15mm) burns into μ-volume |
| 2026-05-15 | Tool-shift test (+30mm world Y)                 | ✅ Tool centroid moved +121 px in detector Y (vs 120 predicted) |
| 2026-05-15 | bridge/fluorosim_torch.Dockerfile (=fluorosim-torch) | ✅ Adds PyTorch 2.5.1+cu121 on top of fluorosim |
| 2026-05-15 | bridge/register_phantom.py — Phase 5 first cut  | ✅ Trans-only registration; 19.7 → 0.41 mm in 100 Adam iters |
| 2026-05-15 | Discovered Slang autodiff returns FLIPPED grad sign | ⚠️ Workaround: negate translation.grad before optimizer.step() |
| 2026-05-15 | bridge/plot_registration.py                     | ✅ Loss curve, per-axis error trace, target/recovered/diff plots |
| 2026-05-15 | register_phantom.py reads pose.json as GT       | ✅ Recovers Isaac Sim phantom_pos to 0.41 mm in world frame |
| 2026-05-15 | Non-trivial test: carm offset +30 mm X          | ✅ GT translation_mm=(30,0,0) recovered to (29.94, 0.03, 0.14) |
| 2026-05-15 | bridge/register_phantom_multiview.py (AP+lat)   | ✅ Depth ambiguity collapsed: Z err 0.41→0.013 mm; ||err||=0.05 mm |
| 2026-05-15 | bridge/compute_robot_to_anatomy.py              | ✅ T_R^A error = 0.051 mm; EE inside bone core (norm. dist 0.77) |
| 2026-05-18 | End-to-end Isaac Sim scene test (Franka+phantom) | ✅ TCP-injected Franka USD + phantom prims into GUI; snapshot in docs/images/franka_phantom_scene.jpg shows both. Headless robot_scene.py wrote pose.json. Multi-view reg + compute_robot_to_anatomy.py recovered T_R^A to 0.051 mm on live scene data. |
| 2026-05-18 | bridge/ct_loader.py + run_load_ct.sh (DICOM → cache)  | ✅ 797-slice spine CT (512² @ 0.5×0.7×0.7 mm) → SimpleITK → cropped 128 mm³ ROI, resampled to (128,256,256) @ (1,0.5,0.5) mm; cached at output/fluorosim_cache_ct/ |
| 2026-05-18 | register_phantom*.py: DICOM_PATH env switch           | ✅ Single flag selects real CT vs synthetic; wrappers conditionally mount DICOM dir + ct_loader.py at /workspace |
| 2026-05-18 | Multi-view registration on real spine CT              | ✅ 25 mm → 0.037 mm in 100 iters (9.4 s); per-axis (0.029, -0.008, 0.021) mm — beats 0.2 mm target; anatomy clearly visible in target/recovered DRRs |
| 2026-05-18 | Synthetic regression after DICOM_PATH plumbing        | ✅ 25 mm → 0.080 mm; no regression vs historical 0.05 mm baseline |
| 2026-05-19 | ct_loader.py: CT_FULL_VOLUME=1 mode (no crop/resample) | ✅ Full (797,512,512) @ (0.5,0.703,0.703) mm, 836 MB; 0.085 mm error, 156 ms/iter; fits 8 GB VRAM with room to spare |
| 2026-05-19 | Bug: VolumePreprocessor HU→μ LUT clamps at μ_water     | ❌ Bone had zero extra attenuation over water; DRRs were gray clouds with no bone visible. Fixed: bilinear hu_to_mu() in ct_loader.py bypasses VolumePreprocessor μ output; μ_bone now 0.048 mm⁻¹ |
| 2026-05-19 | Three-axis DRR survey (rx=0/90, ry=0/90, rz=0)         | ✅ Confirmed: ry=0° projects along CT Z axis (head→feet = coronal); rx=90° = true AP spine fluoroscopy view (ant→post); ry=90° = lateral (left→right) |
| 2026-05-19 | Bug: default crop center in head/neck (wrong anatomy)   | ❌ Geometric volume center (z=398) of the skin-to-skin CT is at mid-neck. Confirmed by rendering 3 orthogonal DRRs: head anatomy visible, no spine. |
| 2026-05-19 | find_vertebral_center() auto-detection in ct_loader.py  | ✅ Compact-bone heuristic: counts bone voxels in central X-Y window per axial slice, finds Y centroid of vertebral body. Auto-detected z=568 (upper lumbar) for spine CT. |
| 2026-05-19 | CT_CROP_CENTER_ZYX env var + manual --center CLI flag    | ✅ Allows per-CT override: CT_CROP_CENTER_ZYX="z,y,x" voxel indices in full CT. Propagated through run_load_ct.sh, run_register*.sh |
| 2026-05-19 | Bug: DRR display overexposed in images.png              | ❌ gamma=0.5 on already-inverted [0,1] images brightened mid-tones (soft tissue→white). Fixed: undo invert, log1p(x×50), p2–p98 clip. Display decoupled from registration. |
| 2026-05-19 | Registration with auto-detected vertebral centre        | ✅ Real spine CT cropped at z=568 (upper lumbar); AP + lateral DRRs show vertebral bodies, rib heads, disc spaces. 25 mm → 0.096 mm in 100 iters (8.2 s) |
| 2026-05-19 | Phase 5e: bridge/ct_to_mesh.py + USD UsdGeom.Mesh in robot_scene.py | ✅ Marching cubes on cached μ-volume → 238k verts / 473k faces OBJ + numpy sidecars. robot_scene.py loads mesh into /World/Phantom/Mesh; falls back to ellipsoid if absent. Verified visually via TCP injection (snapshots/ct_mesh_only.jpg) |
| 2026-05-19 | Phase 5f: USE_POSE_JSON=1 closure on the CT             | ✅ Mesh isocenter at PHANTOM_POS_WORLD_M = synthetic isocenter, so same pose.json contract applies. End-to-end test: 0.096 mm world err on CT phantom. EE at (0.72, 15.4, -2.9) mm in anatomy frame, local μ=0.0225 → soft tissue (correct anatomically) |
| 2026-05-19 | compute_robot_to_anatomy.py: data-driven inside/outside check | ✅ μ-volume lookup replaces ellipsoid distance check when CT cache exists. Layout PNG now shows axial+coronal CT slices through isocenter (replaces ellipse drawings). Ellipsoid fallback preserved for synthetic phantom |
| 2026-05-19 | Phase 5g: tool-in-DRR on the real CT via run_fluorosim.sh | ✅ Added DICOM_PATH switch to fluorosim_render.py + wrapper; renders the spine CT with the EE blob painted in. Visualisation-only path (NOT the registration target — see Phase 5j note) |
| 2026-05-20 | Phase 5h: medical_scene.usd + STAR drives + pose_from_medical_scene.py | ✅ Hand-built OR (Y-up, cm units) with STAR robot, operating table, CT spine mesh. STAR holds rest pose via authored joint drives (stiff=1e9, damp=1e7). pose_from_medical_scene.py TCP-injects to write pose.json from the live stage. End-to-end: 0.096 mm world err. |
| 2026-05-20 | Phase 5i: /World/CArm visualisation + angle-driven view  | ✅ add_carm_viz.py builds the C-arm prim (source/detector/arc/beam) at fluorosim's SDD=1020 / SID=510 geometry. rotate_carm.py rotates around patient long axis (world Z in this Y-up scene) so source and detector sweep correctly. USE_CARM_ROTATION=1 makes the registration AP-view follow the C-arm angle. Verified at 0°/45°: source visibly swings; registration converges to 0.05-0.10 mm. |
| 2026-05-20 | Bug: rotation around scene up-axis didn't move source/det  | ❌ Initial rotate_carm.py rotated around Y (the source/detector axis itself), so they stayed put while only the arc swung. Fixed: rotation axis = patient long axis (perpendicular to scene up). |
| 2026-05-20 | Phase 5j: two-shot / N-shot C-arm capture                | ✅ Two ingest paths: capture_two_shots.py (timed wait + GUI manipulator) and add_carm_shot.py (append-one-shot, command-line friendly). Both write `view_angles_deg` list to pose.json; registration uses it directly. Tested at [0°, 60°] and [0°, 90°]. |
| 2026-05-20 | Bug: stale globals in Isaac Sim TCP executor              | ❌ Executor keeps a persistent globals dict across TCP injections. `RESET_SHOTS=1` set in one call leaked into every subsequent call, causing each append to reset the list. Fixed: changed globals.get() → globals.pop() in rotate_carm.py / add_carm_shot.py / capture_two_shots.py. |
| 2026-05-20 | Bug: TCP commands serialise during sleep                  | ⚠️ time.sleep in a TCP-injected script blocks the executor — concurrent rotate_carm.py via another TCP injection queues behind it. Workaround: capture_two_shots.py pumps the event loop with app.update() during the wait so the *GUI manipulator* stays responsive, but TCP rotations still queue. add_carm_shot.py avoids this entirely. |

---

## Phase 3 Pipeline Recap (file flow)

```
Isaac Sim (headless)        host fs                        Docker container
  robot_scene.py    ──>  ~/isaac_projects/output/
                            pose.json              ──>     /workspace/io/pose.json
                                                              fluorosim_render.py
                            drr.png / drr.npy      <──     /workspace/io/drr.{png,npy}
                            drr_meta.json          <──     /workspace/io/drr_meta.json
                            fluorosim_cache/       <──>    /workspace/io/fluorosim_cache/
  visualize_drr.py  <──  drr_annotated.png
```

Commands (in order):
1. `~/isaacsim/python.sh ~/isaac_projects/scenes/robot_scene.py`  (writes pose.json)
2. `~/isaac_projects/bridge/run_fluorosim.sh`                     (renders DRR)
3. `python3 ~/isaac_projects/bridge/visualize_drr.py`             (annotates + sanity check)

---

## Known Issues / Gotchas
- Isaac Sim GUI appears unresponsive for ~10s on startup — this is normal,
  assets are loading. Always use --headless to avoid confusion.
- Force-quitting the Isaac Sim GUI kills the Python process — don't do it.
- fluorosim OptiX warnings are harmless — falls back to Slang shader path.
- CT volume VRAM budget: the full spine CT at (797,512,512) float32 uses
  ~836 MB for the μ-volume + ~836 MB for Slang gradient textures = ~1.7 GB
  total, well within 8 GB. The old "≤256³" guideline was conservative. The
  practical limit is more like ~500³ at float32 (~500 MB volume + gradients)
  before you start competing with the rest of the pipeline for VRAM.
- ALWAYS launch Isaac Sim with conda deactivated. If conda base is active,
  pip installs land in ~/.local/lib/python3.10/site-packages which is NOT
  on Isaac Sim's sys.path. Isaac Sim's kit Python site-packages are at:
  /home/max/isaacsim/kit/python/lib/python3.10/site-packages
- Isaac Sim deprecation warnings (omni.isaac.* → isaacsim.*) are harmless,
  old API still works in 4.5.0.
- VSCode extension port is 8226 (not 8826) — confirmed in extension.toml.
  Can be overridden via Carb settings; verify live with `ss -tlnp | grep 8226`.
- VSCode extension TCP protocol returns JSON, NOT raw stdout:
  {"status":"ok","output":"..."} — parse with json.loads(), don't treat as plaintext.
- Do NOT call socket.shutdown(SHUT_WR) before reading the TCP reply. The server
  executes code via asyncio.run_coroutine_threadsafe() and replies asynchronously;
  premature shutdown causes a race that produces empty responses.
- rep.create.render_product("/OmniverseKit_Persp", ...) creates a replicator-only
  render product that only renders when rep.orchestrator.step() is called explicitly.
  To capture the live GUI viewport, use get_active_viewport().render_product_path
  (e.g. /Render/OmniverseKit/HydraTextures/omni_kit_widget_viewport_ViewportTexture_0)
- rep.AnnotatorRegistry annotators return shape (0,) until rep.orchestrator.step_async()
  is called. The StageRenderingEventType.NEW_FRAME event alone does not populate them
  in GUI mode — orchestrator must step.
- world.reset() blocks indefinitely when called via TCP injection (no sim loop running).
  Never call it in injected code. Use USD API (stage.DefinePrim, etc.) instead.
- Snapshot paths: always use absolute paths (~/isaac_projects/scenes/snapshots/)
  not relative ./snapshots/ — the CWD when launching python.sh may vary.
- fluorosim `translation` is NOT a world-frame position — it's an offset from the
  CT volume's isocenter (in mm). Feeding the Isaac Sim world-frame C-arm pos
  directly produces a DRR that misses the volume entirely (all-1.0 image).
  For Phase 3 we hardcoded carm_pos=[0,0,0] (= C-arm at isocenter, AP view).
  A proper world→isocenter transform is a Phase 4 task once a phantom exists.
- DRR flat-image sanity check: `std(image) < 1e-4` means the rays never hit the
  volume — verify C-arm geometry before assuming a rendering bug.
- Docker bind-mount file ownership: files written by the fluorosim container
  land as root:root on the host. Either chown afterwards or use `--user $(id -u):$(id -g)`
  in run_fluorosim.sh if this becomes a problem.
- Isaac Sim's print() output to stdout can be swallowed by its log redirection.
  Use sys.stdout.write(...) + sys.stdout.flush() if you need to capture script
  output in a pipe (e.g. `python.sh script.py | grep ...`).
- fluorosim Slang autodiff returns gradients with FLIPPED sign relative to
  PyTorch's convention. At trans=(15,0,0) with MSE loss against a target
  rendered at (0,0,0), the analytical dL/dt_x came back as -0.002 while the
  finite-difference reference is +0.003. Workaround in
  bridge/register_phantom.py: `translation.grad.neg_()` between loss.backward()
  and optimizer.step(). Without this, plain Adam DIVERGES (error grows from
  20 mm to 110 mm over 100 iters).
- Single-view 2D/3D registration cannot recover depth-axis translation
  accurately — the volume just appears slightly larger/smaller. In our setup
  the X-ray beam is along world Z, so the Z component of recovered translation
  has ~10× the error of X/Y. Multi-view (orthogonal C-arm angles) or accepting
  the depth uncertainty is the standard fix.
- PreprocessedVolume.mu_volume is a read-only property. To inject a modified
  volume (e.g. tool-painted) into a FluoroSimulator, construct a new
  PreprocessedVolume(modified_mu, volume._metadata) rather than mutating.
- PyTorch is NOT in the base fluorosim Docker image (it's listed as "optional"
  in fluorosim's deps). The Phase 5 registration wrapper uses a derived
  image `fluorosim-torch` built from bridge/fluorosim_torch.Dockerfile. Don't
  expect `python -c "import torch"` to work inside the plain `fluorosim` image.
- fluorosim-torch image already ships pydicom 3.0.2 + SimpleITK 2.5.4 + scipy;
  no extra deps needed for the DICOM loader. They are NOT in the base
  fluorosim image — ct_loader.py only works in fluorosim-torch.
- SimpleITK's ImageSeriesReader reports a *uniform* slice spacing derived from
  SliceThickness (0.5 mm for the spine series) even when adjacent
  ImagePositionPatient z-coords suggest a finer effective spacing. Trust
  SimpleITK's spacing — pydicom-via-header arithmetic on two endpoint slices
  can mislead.
- ct_loader.py auto-detects the vertebral body centre via find_vertebral_center()
  (compact-bone heuristic in the central X-Y window, middle 80% of Z, with Y
  centroid detection so the crop lands on the actual posterior vertebral body).
  Override with CT_CROP_CENTER_ZYX="z,y,x" when the auto-detection picks the
  wrong level. The DICOM-SEG at .../04098/86171/ (Spine Segmentation) is
  available if a mask-driven bbox is ever needed but is not currently used.
- The geometric volume center is NOT a reliable crop center for long CT series.
  The skin-to-skin spine CT has its center in the head/neck (z=398 of 797).
  Confirmed by rendering 3 orthogonal DRRs: head anatomy visible, no spine.
  Always verify with a sagittal/axial slice plot of the μ-volume after first load.
- USE_POSE_JSON=1 now works with the real CT cache (Phase 5f).  The CT
  mesh's isocenter is placed at PHANTOM_POS_WORLD_M=(0.43,0,0.42) m in
  robot_scene.py — same as the synthetic phantom — so the world-frame
  ground-truth report is meaningful for both phantom types.  No need to
  set USE_POSE_JSON=0 unless you specifically want to disable the
  pose.json ground-truth comparison for debugging.
- The "Tool tip inside bone/soft tissue" clinical check in
  compute_robot_to_anatomy.py is data-driven when a CT cache exists:
  it looks up the local μ value at the EE position in anatomy frame
  and classifies by μ thresholds (bone > 0.035, soft tissue > 0.012).
  Falls back to the analytic ellipsoid distance check only when there's
  no CT cache.  The mu_volume.npy is mmapped so the lookup is essentially
  free (one voxel read).
- VolumePreprocessor HU→μ LUT clamps at μ_water (0.020 mm⁻¹) — bone gets
  zero extra attenuation over soft tissue and is invisible in DRRs. ct_loader.py
  calls VolumePreprocessor only for metadata, then immediately overwrites
  mu_volume.npy with bilinear hu_to_mu() output (μ_bone ≈ 0.048 mm⁻¹ at
  HU 1000). Never call VolumePreprocessor.preprocess() directly on CT data.
- Deleting the cache dir forces a rebuild. The cache hit check is existence-only,
  not a content hash. Delete and re-run run_load_ct.sh after changing the crop
  centre, bone threshold, or μ parameters.
- fluorosim projection axis mapping for a CT in (Z,Y,X) = (axial,AP,LR) order:
    ry=0°   → beam along CT Z axis (head→feet, coronal projection)
    ry=90°  → beam along CT X axis (left→right, lateral projection)
    rx=90°  → beam along CT Y axis (ant→post, TRUE clinical AP spine view)
  The "AP" label in register_phantom_multiview.py (ry=0°) is the coronal view.
  For clinical-style AP spine fluoroscopy, rx=90° is the correct angle. For
  registration accuracy it does not matter — depth ambiguity is collapsed by
  the orthogonal second view regardless of which axis is called "AP."
- fluorosim normalize=False output is the LINE INTEGRAL L=∫μ dl (nepers), NOT
  transmitted intensity exp(-L). Verified: raw range [0,0.63] for a lumbar AP
  view matches soft-tissue path integrals. Applying log1p(L×500) then p1–p99
  clip gives bone=bright, air=dark — identical to real X-ray film response.
- DRR display transform is completely decoupled from the registration optimizer.
  The optimizer uses normalize=True, invert=True [0,1] images internally.
  plot_registration_multiview.py applies log1p(x×50)+p2–p98 purely for
  visualization. Changing display parameters has zero effect on the loss or
  recovered translation. gamma < 1 overexposes inverted images — always use
  a log-based transform for DRR display.
- Full-volume CT (CT_FULL_VOLUME=1) gives ~0.076–0.085 mm vs ~0.096 mm for the
  cropped ROI in recent tests (similar or slightly better because the lateral
  view through the full body has richer gradient signal). Use full mode for
  anatomy-rich DRRs; cropped mode when speed matters.
- run_register*.sh sets `-w /workspace` so that `from ct_loader import ...`
  resolves against the mounted /workspace/ct_loader.py. Without this, Python
  only finds ct_loader if cwd happens to be /workspace.

### Medical scene (Phases 5h–5j)

- scenes/medical_scene.usd is **Y-up, 1 unit = 1 cm** — different from
  robot_scene.py's Z-up / 1 unit = 1 m world.  Always convert via
  `metersPerUnit` when reading prim translations.  pose_from_medical_scene.py
  and add_carm_shot.py do this automatically.
- The medical scene's STAR robot only holds pose when the joint drives
  are present.  Drives are AUTHORED in medical_scene.usd, so they survive
  save+load.  If the robot ever starts falling under gravity again, the
  drives have been lost — re-apply via the inline script in
  scenes/inject_medical_scene.py / Phase 5h commit history.
- Patient long axis in medical_scene.usd is **world Z** (the table's long
  edge runs along Z; the spine mesh's "head-feet" axis aligns with Z).
  C-arm rotation around world Z = LAO/RAO sweep — this is what
  rotate_carm.py and pose_from_medical_scene.py extract.  If you ever
  re-author the scene with a different patient orientation, update the
  PATIENT_LONG_AXIS map in those two scripts.
- TCP injections in Isaac Sim **serialise** — a TCP command issued while
  another is executing waits for the first to finish.  This means
  rotate_carm.py via TCP cannot be used to rotate the C-arm DURING a
  capture_two_shots.py wait — they queue.  Workarounds: rotate via the
  GUI manipulator (direct USD edit, no TCP), or use add_carm_shot.py
  (one shot per call, no waiting).
- The Isaac Sim TCP executor keeps a **persistent globals dict** across
  calls.  Any parameter passed via `globals()` (e.g. `CARM_ROTATION_DEG`,
  `RESET_SHOTS`, `TWO_SHOT_WAIT_SEC`) must be consumed with
  `globals().pop(...)` not `globals().get(...)` — otherwise the value
  leaks into every subsequent injection.

### Simulation simplifications (honest about what's not real)

Two known places where the simulation is more permissive than a clinical
setting.  Both are deferred fixes — listed in the Phase 5 checklist as
unchecked items:

1. **Tool not in the registration target image**.  `fluorosim_render.py`
   (run by `run_fluorosim.sh`) paints the EE blob into the μ-volume → its
   DRR shows the tool, matching a real fluoroscope.  But
   `register_phantom_multiview.py` builds its target images from the
   clean μ-volume — *no tool*.  Reason: μ=0.5 mm⁻¹ is ~28× cortical bone,
   so an opaque blob occludes the spine and dominates the MSE loss.
   Real-world fluoroscopy registration handles this by masking the tool
   region or using a robust loss.  To make our simulation stricter, we'd
   need to either paint the tool into the target AND add a mask/robust
   loss, or accept that the tool is "known geometry" the registration
   subtracts before matching.

2. **Optimizer init uses GT + INIT_OFFSET_MM**.  In
   register_phantom_multiview.py:
       init_trans = gt_translation_mm + INIT_OFFSET_MM
   The 20 mm offset simulates pre-op planning error, but the chain reads
   the ground truth to construct the starting pose.  Once the optimiser
   is running it only sees the target image — but the *capture range*
   (how far off you can start and still converge) is hidden behind this
   prior.  A stricter "blind init from carm_pos" (treating the phantom
   as roughly at the C-arm isocenter — the standard clinical assumption)
   would be independent of GT but might fail to converge if the actual
   phantom is too far from isocenter.  Worth doing as a separate
   characterisation experiment.

In both cases the registration ALGORITHM is identical to what would be
used clinically.  The simplifications affect *how realistic the input
is*, not *whether the math works*.

---

## Key Commands
# Run Isaac Sim standalone script
~/isaacsim/python.sh ~/isaac_projects/scenes/robot_scene.py

# Verify VSCode extension is listening
ss -tlnp | grep 8226

# Send Python code to running Isaac Sim (returns JSON-parsed stdout)
python3 ~/isaac_projects/scenes/isaacsim_client.py "print(1+1)"
python3 ~/isaac_projects/scenes/isaacsim_client.py < some_script.py

# Inject viewport publisher into running Isaac Sim
python3 isaacsim_client.py "$(cat image_publisher.py)"

# Take a snapshot from the ZMQ stream (run from outside Isaac Sim)
python3 ~/isaac_projects/scenes/take_snapshot.py [filename.jpg]

# Inject verify_pose.py to read live EE pose and compare to pose.json
python3 ~/isaac_projects/scenes/isaacsim_client.py < ~/isaac_projects/scenes/verify_pose.py

# Inject verify_phantom.py to inspect phantom prims in a running GUI session
python3 ~/isaac_projects/scenes/isaacsim_client.py < ~/isaac_projects/scenes/verify_phantom.py

# Render a DRR from the current pose.json (runs in fluorosim Docker)
~/isaac_projects/bridge/run_fluorosim.sh

# Visualize the rendered DRR + pose annotation
python3 ~/isaac_projects/bridge/visualize_drr.py        # saves drr_annotated.png
python3 ~/isaac_projects/bridge/visualize_drr.py --show # interactive window

# Phase 5: run phantom translation registration (uses fluorosim-torch image)
~/isaac_projects/bridge/run_register.sh
# Override defaults via env vars (all optional):
INIT_OFFSET_MM="30,0,0" LR_MM=2.0 N_ITERS=200 ~/isaac_projects/bridge/run_register.sh

# Plot the registration convergence + image comparison (host-side)
python3 ~/isaac_projects/bridge/plot_registration.py

# Phase 5 (multi-view): clinical sequential AP + lateral C-arm shots
~/isaac_projects/bridge/run_register_multiview.sh
# Override which views: angles in degrees around Y (LAO/RAO axis)
VIEWS_DEG_Y="0,45,90" ~/isaac_projects/bridge/run_register_multiview.sh
# Plot multi-view results (host-side)
python3 ~/isaac_projects/bridge/plot_registration_multiview.py

# Phase 5 deliverable: compute T_robot_to_anatomy from registration + pose.json
# (host-side, no Docker). Falls back to single-view if multi-view absent.
python3 ~/isaac_projects/bridge/compute_robot_to_anatomy.py
python3 ~/isaac_projects/bridge/compute_robot_to_anatomy.py --show  # interactive window

# Phase 5d: pre-warm the real-CT cache (one-time; auto-detects vertebral level)
~/isaac_projects/bridge/run_load_ct.sh                 # cropped ROI, auto centre
CT_FULL_VOLUME=1 ~/isaac_projects/bridge/run_load_ct.sh   # full 797×512×512
# Override DICOM source (default = spine_mets_ct_seg/10250/04098/27242):
DICOM_PATH=/path/to/dicom_dir ~/isaac_projects/bridge/run_load_ct.sh
# Override crop centre when auto-detection picks the wrong anatomy level
# (z,y,x are voxel indices in the full CT array — Z=axial slice, Y=row, X=col):
CT_CROP_CENTER_ZYX="568,252,256" ~/isaac_projects/bridge/run_load_ct.sh

# Phase 5d: run registration against the real CT (instead of synthetic).
# Must use USE_POSE_JSON=0 until phantom.py is updated for the CT geometry.
DICOM_PATH=~/medical_imaging/spine_mets_ct_seg/10250/04098/27242 \
  USE_POSE_JSON=0 INIT_OFFSET_MM="20,0,15" \
  ~/isaac_projects/bridge/run_register_multiview.sh
# Full-volume variant (836 MB, 158 ms/iter, ~0.076 mm error):
DICOM_PATH=~/medical_imaging/spine_mets_ct_seg/10250/04098/27242 \
  CT_FULL_VOLUME=1 USE_POSE_JSON=0 \
  ~/isaac_projects/bridge/run_register_multiview.sh
# Manual crop centre override also propagates into the registration wrappers:
DICOM_PATH=~/medical_imaging/spine_mets_ct_seg/10250/04098/27242 \
  CT_CROP_CENTER_ZYX="568,252,256" USE_POSE_JSON=0 \
  ~/isaac_projects/bridge/run_register_multiview.sh
# Same flags work for the single-view script:
DICOM_PATH=~/medical_imaging/spine_mets_ct_seg/10250/04098/27242 \
  USE_POSE_JSON=0 ~/isaac_projects/bridge/run_register.sh

# === Phases 5h-5j: medical scene end-to-end ===
# (Open Isaac Sim GUI with scenes/medical_scene.usd loaded first.)

# Rotate the C-arm to a specific angle around the patient long axis
python3 ~/isaac_projects/scenes/isaacsim_client.py "CARM_ROTATION_DEG=45
$(cat ~/isaac_projects/scenes/rotate_carm.py)"

# Workflow A (command-line, recommended): two-shot capture
python3 ~/isaac_projects/scenes/isaacsim_client.py "CARM_ROTATION_DEG=0
$(cat ~/isaac_projects/scenes/rotate_carm.py)"
python3 ~/isaac_projects/scenes/isaacsim_client.py "RESET_SHOTS=1
$(cat ~/isaac_projects/scenes/add_carm_shot.py)"
python3 ~/isaac_projects/scenes/isaacsim_client.py "CARM_ROTATION_DEG=90
$(cat ~/isaac_projects/scenes/rotate_carm.py)"
python3 ~/isaac_projects/scenes/isaacsim_client.py < ~/isaac_projects/scenes/add_carm_shot.py

# Workflow B (GUI): timed capture — drag the manipulator during the wait
python3 ~/isaac_projects/scenes/isaacsim_client.py "TWO_SHOT_WAIT_SEC=5
$(cat ~/isaac_projects/scenes/capture_two_shots.py)"

# Either workflow ends with pose.json containing view_angles_deg = [...]
# Run multi-view registration using those captured views:
DICOM_PATH=~/medical_imaging/spine_mets_ct_seg/10250/04098/27242 \
  USE_POSE_JSON=1 USE_CARM_ROTATION=1 \
  ~/isaac_projects/bridge/run_register_multiview.sh

# Run T_robot_to_anatomy + clinical inside/outside check
python3 ~/isaac_projects/bridge/compute_robot_to_anatomy.py

# Render a single demo DRR (with the EE tool painted in — for visualisation)
DICOM_PATH=~/medical_imaging/spine_mets_ct_seg/10250/04098/27242 \
  ~/isaac_projects/bridge/run_fluorosim.sh

# Build the derived torch-enabled image (one-time, ~5 min for the wheel download)
docker build -t fluorosim-torch -f ~/isaac_projects/bridge/fluorosim_torch.Dockerfile \
    ~/isaac_projects/bridge/

# Install a package into Isaac Sim's kit Python (NOT pip install / conda)
/home/max/isaacsim/kit/python/bin/python3 -m pip install <package> \
  --target /home/max/isaacsim/kit/python/lib/python3.10/site-packages

# Run fluorosim Docker with output mounted
docker run -it --rm --gpus all \
  -v ~/isaac_projects/output:/app/output \
  fluorosim bash

# Rebuild fluorosim Docker image
cd ~/nvidia-third-party/i4h-sensor-simulation/fluoro-simulator
docker build -t fluorosim .
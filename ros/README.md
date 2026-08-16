# ROS 2 / RViz visualization (Phase 7)

Visualise the recovered fluoroscopy-registration result live in RViz: the CT
anatomy, the recovered tool, the C-arm, and the transforms as a ROS 2 TF tree.

This is the minimal, dependency-light layer — plain ROS 2 Humble + `rviz2`, no
Isaac ROS required. It consumes the output the rest of the pipeline already
produces (`output/robot_to_anatomy.json`, `output/spine_mesh.obj`).

## What it shows

Rooted at the **anatomy** frame (the CT isocenter):

- **TF tree** — `anatomy → tool` (recovered `T_robot_in_anatomy`) and
  `anatomy → carm` (inverse of recovered `T_anatomy_in_carm`).
- **CT anatomy** — the marching-cubes surface mesh at the anatomy origin.
- **Tool** — an oriented shaft cylinder + tip sphere in the `tool` frame.
- **C-arm** — a source sphere + the beam axis to the isocenter.
- **Text panel** — the registration ‖error‖ and the clinical inside/outside check.

The bridge re-reads `robot_to_anatomy.json` at 5 Hz, so re-running the
registration updates the scene live — the hook for a future real-time tracking
loop.

## Run

```bash
# 1. Produce a result if you don't have one (writes output/robot_to_anatomy.json):
python3 bridge/register_oneshot.py --no-capture        # reuse existing pose.json

# 2. Launch the bridge + RViz (needs a display):
ros/run_rviz.sh

# Bridge only (no GUI), e.g. to inspect the TF from the terminal:
ros/run_rviz.sh --no-rviz
source /opt/ros/humble/setup.bash
ros2 run tf2_ros tf2_echo anatomy tool
```

## Files

| File | Purpose |
|---|---|
| `fluoro_tf_bridge.py` | rclpy node: TF broadcaster + MarkerArray publisher |
| `fluoro_scene.rviz`   | RViz config (fixed frame `anatomy`, TF + MarkerArray) |
| `run_rviz.sh`         | sources ROS, launches the bridge + rviz2 |

## Follow-ons

- **STAR robot URDF** — currently the robot is represented by the tool markers +
  TF frames only (no URDF was available; the STAR arm lives as USD in
  `medical_scene.usd`). Converting it to URDF + `robot_state_publisher` would
  draw the full arm.
- **Isaac ROS** — a live Isaac Sim ↔ ROS 2 bridge (stream the scene/camera,
  GPU perception), worthwhile once the real-time tracking loop exists.
- **MoveIt** — consumes this TF tree + anatomy mesh for trajectory planning
  (the navigation end goal).

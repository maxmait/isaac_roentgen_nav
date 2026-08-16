#!/usr/bin/env bash
# Launch the fluoro TF bridge + RViz to visualise the recovered registration.
#
# Shows, rooted at the anatomy (CT) frame: the CT surface mesh, the recovered
# tool (shaft + tip), the C-arm source + beam axis, the TF tree
# (anatomy → tool, anatomy → carm), and a text panel with the registration
# error + clinical check.  Re-reading output/robot_to_anatomy.json live, so
# re-running the registration updates the scene.
#
# Prereq: ROS 2 Humble (already installed at /opt/ros/humble) and a prior
# registration run (output/robot_to_anatomy.json + output/spine_mesh.obj).
#
# Usage:  ros/run_rviz.sh        (opens RViz; needs a display)
#         ros/run_rviz.sh --no-rviz   (bridge only, e.g. for tf2_echo)

set -euo pipefail

source /opt/ros/humble/setup.bash

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NODE="${HERE}/fluoro_tf_bridge.py"
RVIZ_CFG="${HERE}/fluoro_scene.rviz"
RESULT="${HOME}/isaac_projects/output/robot_to_anatomy.json"

if [[ ! -f "${RESULT}" ]]; then
    echo "WARNING: ${RESULT} not found — run a registration first" >&2
    echo "  (e.g. python3 bridge/register_oneshot.py --no-capture)" >&2
fi

# Start the TF/marker bridge in the background; stop it on exit.
python3 "${NODE}" &
BRIDGE_PID=$!
trap 'kill ${BRIDGE_PID} 2>/dev/null || true' EXIT

if [[ "${1:-}" == "--no-rviz" ]]; then
    echo "Bridge running (PID ${BRIDGE_PID}); Ctrl-C to stop."
    wait ${BRIDGE_PID}
else
    rviz2 -d "${RVIZ_CFG}"
fi

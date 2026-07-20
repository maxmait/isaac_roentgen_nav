#!/usr/bin/env bash
# Run register_tool_pose.py (image-based tool/TCP pose recovery PoC) inside the
# fluorosim-torch Docker container.
#
# Output: ~/isaac_projects/output/tool_pose/
#
# Self-contained: needs only output/tool_stamp.npy (built by build_tool_stamp.py)
# — no CT / DICOM.  register_tool_pose.py imports helpers from
# register_phantom_multiview.py, so both scripts are mounted at /workspace.
#
# Environment variables (all optional):
#   N_ITERS(200) LR_MM(1.0) LR_ROT_RAD(0.005) ROT_GRAD_CLIP(0.1) LOG_EVERY(25)
#   VIEWS_DEG_Y("0,45,90")        set "0" for a single-view depth-ambiguity demo
#   TOOL_GT_TRANS_MM("12,-8,10")  synthetic GT tool-center offset (base frame)
#   TOOL_GT_ROT_DEG("-80,6,-5")   synthetic GT tool orientation (ZXY Euler deg)
#   TOOL_INIT_TRANS_MM("0,0,0")   blind init translation (isocenter)
#   TOOL_INIT_ROT_DEG("0,0,0")    blind init rotation
#
# Usage:
#   ~/isaac_projects/bridge/run_register_tool.sh
#   VIEWS_DEG_Y="0" ~/isaac_projects/bridge/run_register_tool.sh   # single view

set -euo pipefail

HOST_IO_DIR="${HOME}/isaac_projects/output"
HOST_SCRIPT="${HOME}/isaac_projects/bridge/register_tool_pose.py"
HOST_HELPERS="${HOME}/isaac_projects/bridge/register_phantom_multiview.py"

for f in "${HOST_SCRIPT}" "${HOST_HELPERS}"; do
    if [[ ! -f "${f}" ]]; then
        echo "ERROR: ${f} not found." >&2
        exit 1
    fi
done

if [[ ! -f "${HOST_IO_DIR}/tool_stamp.npy" ]]; then
    echo "ERROR: ${HOST_IO_DIR}/tool_stamp.npy not found." >&2
    echo "  Build it first: python3 bridge/build_tool_stamp.py" >&2
    exit 1
fi

mkdir -p "${HOST_IO_DIR}/tool_pose"

DOCKER_ENV_ARGS=()
for v in N_ITERS LR_MM LR_ROT_RAD ROT_GRAD_CLIP LOG_EVERY VIEWS_DEG_Y FLIP_GRAD \
         TOOL_GT_TRANS_MM TOOL_GT_ROT_DEG TOOL_INIT_TRANS_MM TOOL_INIT_ROT_DEG; do
    if [[ -n "${!v:-}" ]]; then
        DOCKER_ENV_ARGS+=("-e" "${v}=${!v}")
    fi
done

exec docker run --rm --gpus all \
    "${DOCKER_ENV_ARGS[@]}" \
    -v "${HOST_IO_DIR}:/workspace/io" \
    -v "${HOST_SCRIPT}:/workspace/register_tool_pose.py:ro" \
    -v "${HOST_HELPERS}:/workspace/register_phantom_multiview.py:ro" \
    -w /workspace \
    fluorosim-torch \
    python /workspace/register_tool_pose.py

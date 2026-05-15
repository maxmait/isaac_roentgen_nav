#!/usr/bin/env bash
# Run fluorosim_render.py inside the fluorosim Docker container.
#
# Bind-mounts the host I/O dir (~/isaac_projects/output) to /workspace/io
# and the render script to /workspace/fluorosim_render.py, then runs it.
#
# Usage:
#   ~/isaac_projects/bridge/run_fluorosim.sh
#
# Output (on host):
#   ~/isaac_projects/output/drr.png
#   ~/isaac_projects/output/drr.npy
#   ~/isaac_projects/output/drr_meta.json
#   ~/isaac_projects/output/fluorosim_cache/  (preprocessed volume, persisted)

set -euo pipefail

HOST_IO_DIR="${HOME}/isaac_projects/output"
HOST_SCRIPT="${HOME}/isaac_projects/bridge/fluorosim_render.py"

if [[ ! -f "${HOST_IO_DIR}/pose.json" ]]; then
    echo "ERROR: ${HOST_IO_DIR}/pose.json not found." >&2
    echo "       Run robot_scene.py first to produce a pose." >&2
    exit 1
fi

if [[ ! -f "${HOST_SCRIPT}" ]]; then
    echo "ERROR: ${HOST_SCRIPT} not found." >&2
    exit 1
fi

mkdir -p "${HOST_IO_DIR}/fluorosim_cache"

exec docker run --rm --gpus all \
    -v "${HOST_IO_DIR}:/workspace/io" \
    -v "${HOST_SCRIPT}:/workspace/fluorosim_render.py:ro" \
    fluorosim \
    python /workspace/fluorosim_render.py

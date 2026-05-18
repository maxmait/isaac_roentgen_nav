#!/usr/bin/env bash
# Run register_phantom_multiview.py inside the fluorosim-torch Docker container.
#
# Output: ~/isaac_projects/output/registration_multiview/
#
# Environment variables (all optional):
#   VIEWS_DEG_Y     comma-separated LAO/RAO angles in degrees (default "0,90")
#   INIT_OFFSET_MM  perturbation in fluorosim translation space (default "15,-10,8")
#   LR_MM           Adam learning rate                          (default 1.0)
#   N_ITERS         optimizer iterations                        (default 100)
#   LOG_EVERY       progress print stride                       (default 5)
#   USE_POSE_JSON   set to 0 to ignore pose.json                (default 1)
#
# Usage:
#   ~/isaac_projects/bridge/run_register_multiview.sh
#   VIEWS_DEG_Y="0,45,90" N_ITERS=150 ~/isaac_projects/bridge/run_register_multiview.sh

set -euo pipefail

HOST_IO_DIR="${HOME}/isaac_projects/output"
HOST_SCRIPT="${HOME}/isaac_projects/bridge/register_phantom_multiview.py"

if [[ ! -f "${HOST_SCRIPT}" ]]; then
    echo "ERROR: ${HOST_SCRIPT} not found." >&2
    exit 1
fi

mkdir -p "${HOST_IO_DIR}/registration_multiview"

DOCKER_ENV_ARGS=()
for v in VIEWS_DEG_Y INIT_OFFSET_MM LR_MM N_ITERS LOG_EVERY USE_POSE_JSON; do
    if [[ -n "${!v:-}" ]]; then
        DOCKER_ENV_ARGS+=("-e" "${v}=${!v}")
    fi
done

exec docker run --rm --gpus all \
    "${DOCKER_ENV_ARGS[@]}" \
    -v "${HOST_IO_DIR}:/workspace/io" \
    -v "${HOST_SCRIPT}:/workspace/register_phantom_multiview.py:ro" \
    fluorosim-torch \
    python /workspace/register_phantom_multiview.py

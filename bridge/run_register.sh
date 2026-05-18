#!/usr/bin/env bash
# Run register_phantom.py inside the fluorosim Docker container.
#
# Mounts the host I/O dir at /workspace/io and the script at /workspace.
# Outputs land in ~/isaac_projects/output/registration/.
#
# Environment variables (all optional, defaults set in register_phantom.py):
#   GT_TRANSLATION_MM   ground-truth translation, "x,y,z" in mm (default "0,0,0")
#   INIT_OFFSET_MM      initial perturbation,     "x,y,z" in mm (default "15,-10,8")
#   LR_MM               Adam learning rate                       (default 1.0)
#   N_ITERS             optimizer iterations                     (default 100)
#   LOG_EVERY           progress print stride                    (default 5)
#
# Usage:
#   ~/isaac_projects/bridge/run_register.sh
#   INIT_OFFSET_MM="30,0,0" LR_MM=2.0 ~/isaac_projects/bridge/run_register.sh

set -euo pipefail

HOST_IO_DIR="${HOME}/isaac_projects/output"
HOST_SCRIPT="${HOME}/isaac_projects/bridge/register_phantom.py"

if [[ ! -f "${HOST_SCRIPT}" ]]; then
    echo "ERROR: ${HOST_SCRIPT} not found." >&2
    exit 1
fi

mkdir -p "${HOST_IO_DIR}/registration"

# Forward the experiment env vars into the container if set on the host.
DOCKER_ENV_ARGS=()
for v in GT_TRANSLATION_MM INIT_OFFSET_MM LR_MM N_ITERS LOG_EVERY; do
    if [[ -n "${!v:-}" ]]; then
        DOCKER_ENV_ARGS+=("-e" "${v}=${!v}")
    fi
done

exec docker run --rm --gpus all \
    "${DOCKER_ENV_ARGS[@]}" \
    -v "${HOST_IO_DIR}:/workspace/io" \
    -v "${HOST_SCRIPT}:/workspace/register_phantom.py:ro" \
    fluorosim-torch \
    python /workspace/register_phantom.py

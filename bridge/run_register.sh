#!/usr/bin/env bash
# Run register_phantom.py inside the fluorosim Docker container.
#
# Mounts the host I/O dir at /workspace/io and the script at /workspace.
# Outputs land in ~/isaac_projects/output/registration/.
#
# Environment variables (all optional, defaults set in register_phantom.py):
#   GT_TRANSLATION_MM   ground-truth translation, "x,y,z" in mm (default "0,0,0")
#   INIT_OFFSET_MM      initial perturbation,     "x,y,z" in mm (default "15,-10,8")
#   INIT_ROT_DEG        rotation perturbation (ZXY Euler deg)    (default "5,0,3")
#   LR_MM               Adam learning rate for translation       (default 1.0)
#   LR_ROT_RAD          Adam learning rate for rotation (rad/step)(default 0.01)
#   N_ITERS             optimizer iterations                     (default 100)
#   LOG_EVERY           progress print stride                    (default 5)
#   DICOM_PATH          host path to a DICOM CT dir; when set, use real CT
#                       instead of the synthetic ellipsoid phantom
#
# Usage:
#   ~/isaac_projects/bridge/run_register.sh
#   INIT_OFFSET_MM="30,0,0" LR_MM=2.0 ~/isaac_projects/bridge/run_register.sh
#   DICOM_PATH=~/medical_imaging/.../27242 ~/isaac_projects/bridge/run_register.sh

set -euo pipefail

HOST_IO_DIR="${HOME}/isaac_projects/output"
HOST_SCRIPT="${HOME}/isaac_projects/bridge/register_phantom.py"
HOST_CT_LOADER="${HOME}/isaac_projects/bridge/ct_loader.py"

if [[ ! -f "${HOST_SCRIPT}" ]]; then
    echo "ERROR: ${HOST_SCRIPT} not found." >&2
    exit 1
fi

mkdir -p "${HOST_IO_DIR}/registration"

# Forward the experiment env vars into the container if set on the host.
DOCKER_ENV_ARGS=()
for v in GT_TRANSLATION_MM INIT_OFFSET_MM INIT_ROT_DEG LR_MM LR_ROT_RAD ROT_GRAD_CLIP N_ITERS LOG_EVERY CT_FULL_VOLUME CT_CROP_CENTER_ZYX; do
    if [[ -n "${!v:-}" ]]; then
        DOCKER_ENV_ARGS+=("-e" "${v}=${!v}")
    fi
done

DOCKER_CT_ARGS=()
if [[ -n "${DICOM_PATH:-}" ]]; then
    if [[ ! -d "${DICOM_PATH}" ]]; then
        echo "ERROR: DICOM_PATH=${DICOM_PATH} not found." >&2
        exit 1
    fi
    if [[ ! -f "${HOST_CT_LOADER}" ]]; then
        echo "ERROR: ${HOST_CT_LOADER} not found (needed when DICOM_PATH is set)." >&2
        exit 1
    fi
    DOCKER_CT_ARGS=(
        -v "${DICOM_PATH}:/workspace/ct:ro"
        -v "${HOST_CT_LOADER}:/workspace/ct_loader.py:ro"
        -e "DICOM_PATH=/workspace/ct"
    )
fi

exec docker run --rm --gpus all \
    "${DOCKER_ENV_ARGS[@]}" \
    "${DOCKER_CT_ARGS[@]}" \
    -v "${HOST_IO_DIR}:/workspace/io" \
    -v "${HOST_SCRIPT}:/workspace/register_phantom.py:ro" \
    -w /workspace \
    fluorosim-torch \
    python /workspace/register_phantom.py

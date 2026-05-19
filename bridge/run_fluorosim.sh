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
HOST_CT_LOADER="${HOME}/isaac_projects/bridge/ct_loader.py"

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

# CT mode — same switch as the registration wrappers.
#   DICOM_PATH=<host dir>  → real CT volume, fluorosim-torch image (has scipy/sitk)
#   CT_FULL_VOLUME=1       → full-volume cache instead of cropped ROI
DOCKER_CT_ARGS=()
IMAGE="fluorosim"
if [[ -n "${DICOM_PATH:-}" ]]; then
    if [[ ! -d "${DICOM_PATH}" ]]; then
        echo "ERROR: DICOM_PATH=${DICOM_PATH} not found." >&2
        exit 1
    fi
    if [[ ! -f "${HOST_CT_LOADER}" ]]; then
        echo "ERROR: ${HOST_CT_LOADER} not found (needed when DICOM_PATH is set)." >&2
        exit 1
    fi
    IMAGE="fluorosim-torch"      # has scipy + SimpleITK + scikit-image
    DOCKER_CT_ARGS=(
        -v "${DICOM_PATH}:/workspace/ct:ro"
        -v "${HOST_CT_LOADER}:/workspace/ct_loader.py:ro"
        -e "DICOM_PATH=/workspace/ct"
        -e "CT_FULL_VOLUME=${CT_FULL_VOLUME:-0}"
    )
    [[ -n "${CT_CROP_CENTER_ZYX:-}" ]] && \
        DOCKER_CT_ARGS+=(-e "CT_CROP_CENTER_ZYX=${CT_CROP_CENTER_ZYX}")
fi

exec docker run --rm --gpus all \
    "${DOCKER_CT_ARGS[@]}" \
    -v "${HOST_IO_DIR}:/workspace/io" \
    -v "${HOST_SCRIPT}:/workspace/fluorosim_render.py:ro" \
    -w /workspace \
    "${IMAGE}" \
    python /workspace/fluorosim_render.py

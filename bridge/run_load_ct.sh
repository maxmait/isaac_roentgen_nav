#!/usr/bin/env bash
# One-time: load a DICOM CT and cache it as a fluorosim PreprocessedVolume.
#
# Environment variables (all optional):
#   DICOM_PATH            host path to the DICOM directory
#                         (default: ~/medical_imaging/spine_mets_ct_seg/10250/04098/27242)
#   CT_FULL_VOLUME        1 = use full CT at native spacing (no crop/resample).
#                         Default: 0 (auto-detect crop centre).
#   CT_CROP_CENTER_ZYX    "z,y,x" voxel indices in the full CT to centre the crop on.
#                         Omit to auto-detect the vertebral body level.
#                         Useful when auto-detection picks the wrong anatomy.
#   CACHE_DIR_NAME        cache subdir under ~/isaac_projects/output.
#                         Defaults to fluorosim_cache_ct or fluorosim_cache_ct_full.
#
# Usage:
#   ~/isaac_projects/bridge/run_load_ct.sh                          # auto-detect crop centre
#   CT_FULL_VOLUME=1 ~/isaac_projects/bridge/run_load_ct.sh         # full CT
#   CT_CROP_CENTER_ZYX="637,320,256" ~/isaac_projects/bridge/run_load_ct.sh  # manual centre

set -euo pipefail

HOST_IO_DIR="${HOME}/isaac_projects/output"
HOST_SCRIPT="${HOME}/isaac_projects/bridge/ct_loader.py"
DICOM_PATH="${DICOM_PATH:-${HOME}/medical_imaging/spine_mets_ct_seg/10250/04098/27242}"
CT_FULL_VOLUME="${CT_FULL_VOLUME:-0}"

if [[ "${CT_FULL_VOLUME}" == "1" ]]; then
    CACHE_DIR_NAME="${CACHE_DIR_NAME:-fluorosim_cache_ct_full}"
    FULL_FLAG="--full"
else
    CACHE_DIR_NAME="${CACHE_DIR_NAME:-fluorosim_cache_ct}"
    FULL_FLAG=""
fi

if [[ ! -f "${HOST_SCRIPT}" ]]; then
    echo "ERROR: ${HOST_SCRIPT} not found." >&2
    exit 1
fi

if [[ ! -d "${DICOM_PATH}" ]]; then
    echo "ERROR: DICOM_PATH=${DICOM_PATH} not found." >&2
    exit 1
fi

mkdir -p "${HOST_IO_DIR}/${CACHE_DIR_NAME}"

CT_ENV_ARGS=()
if [[ -n "${CT_CROP_CENTER_ZYX:-}" ]]; then
    CT_ENV_ARGS+=("-e" "CT_CROP_CENTER_ZYX=${CT_CROP_CENTER_ZYX}")
fi

exec docker run --rm --gpus all \
    "${CT_ENV_ARGS[@]}" \
    -v "${HOST_IO_DIR}:/workspace/io" \
    -v "${HOST_SCRIPT}:/workspace/ct_loader.py:ro" \
    -v "${DICOM_PATH}:/workspace/ct:ro" \
    fluorosim-torch \
    python /workspace/ct_loader.py /workspace/ct "/workspace/io/${CACHE_DIR_NAME}" ${FULL_FLAG}

#!/usr/bin/env bash
# Run register_phantom_multiview.py inside the fluorosim-torch Docker container.
#
# Output: ~/isaac_projects/output/registration_multiview/
#
# Environment variables (all optional):
#   VIEWS_DEG_Y     comma-separated LAO/RAO angles in degrees (default "0,45,90")
#                   3 views (one oblique) needed for robust blind 6-DOF — two
#                   orthogonal views leave the tx<->ry ambiguity unbroken.
#   INIT_OFFSET_MM  perturbation in fluorosim translation space (default "15,-10,8")
#   INIT_ROT_DEG    perturbation in phantom rotation (ZXY Euler, default "5,0,3")
#   LR_MM           Adam learning rate for translation          (default 1.0)
#   LR_ROT_RAD      Adam learning rate for rotation (rad/step)  (default 0.01)
#   N_ITERS         optimizer iterations                        (default 100)
#   LOG_EVERY       progress print stride                       (default 5)
#   USE_POSE_JSON   set to 0 to ignore pose.json                (default 1)
#   DICOM_PATH      host path to a DICOM CT dir; when set, use real CT
#                   instead of the synthetic ellipsoid phantom
#   TOOL_IN_TARGET  paint EE tool blob into the registration target DRR
#                   and mask its pixels out of the loss            (default 1)
#                   Auto-disabled when pose.json has no ee_pos.
#   TOOL_MU_PER_MM  tool linear attenuation                       (default 0.3,
#                                                                  ~steel @ 60 keV)
#   TOOL_RADIUS_MM  tool sphere radius                            (default 15)
#   TOOL_MASK_THRESH occlusion threshold for tool mask            (default 0.1)
#
#   --- Step 1: realistic target degradation (break the inverse crime) ---
#   DRR_NOISE         1 = add noise/blur to TARGET DRRs only       (default 0)
#   DRR_BLUR_SIGMA_PX detector PSF Gaussian sigma in px            (default 0.7)
#   DRR_PHOTON_COUNT  photons/px for Poisson noise; lower=noisier  (default 1e4)
#   DRR_SCATTER_FRAC  low-frequency additive scatter fraction      (default 0)
#   DRR_NOISE_SEED    >=0 → reproducible target realisation        (default -1)
#
#   --- Step 2: capture-range / basin-of-attraction study ---
#   CAPTURE_RANGE     1 = sweep init offsets instead of one run    (default 0)
#                     writes output/capture_range.json
#   CR_TRANS_RADII_MM comma list of init offset radii (mm)  (default "5,10,20,30,40,60")
#   CR_N_SAMPLES      random directions per radius                 (default 8)
#   CR_ROT_OFFSET_DEG rotation perturbation per sample             (default 5)
#   CR_SUCCESS_MM     converged if final ‖t_err‖ < this            (default 1.0)
#   CR_SUCCESS_DEG    and geodesic rot err < this                  (default 1.0)
#   CR_SEED           RNG seed for reproducible sampling           (default 0)
#
# Usage:
#   ~/isaac_projects/bridge/run_register_multiview.sh
#   VIEWS_DEG_Y="0,45,90" N_ITERS=150 ~/isaac_projects/bridge/run_register_multiview.sh
#   DICOM_PATH=~/medical_imaging/.../27242 ~/isaac_projects/bridge/run_register_multiview.sh

set -euo pipefail

HOST_IO_DIR="${HOME}/isaac_projects/output"
HOST_SCRIPT="${HOME}/isaac_projects/bridge/register_phantom_multiview.py"
HOST_CT_LOADER="${HOME}/isaac_projects/bridge/ct_loader.py"

if [[ ! -f "${HOST_SCRIPT}" ]]; then
    echo "ERROR: ${HOST_SCRIPT} not found." >&2
    exit 1
fi

mkdir -p "${HOST_IO_DIR}/registration_multiview"

DOCKER_ENV_ARGS=()
for v in VIEWS_DEG_Y INIT_OFFSET_MM INIT_ROT_DEG LR_MM LR_ROT_RAD ROT_GRAD_CLIP N_ITERS LOG_EVERY USE_POSE_JSON USE_CARM_ROTATION CT_FULL_VOLUME CT_CROP_CENTER_ZYX TOOL_IN_TARGET TOOL_RADIUS_MM TOOL_MU_PER_MM TOOL_MASK_THRESH DRR_NOISE DRR_BLUR_SIGMA_PX DRR_PHOTON_COUNT DRR_SCATTER_FRAC DRR_NOISE_SEED CAPTURE_RANGE CR_TRANS_RADII_MM CR_N_SAMPLES CR_ROT_OFFSET_DEG CR_SUCCESS_MM CR_SUCCESS_DEG CR_SEED; do
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
    -v "${HOST_SCRIPT}:/workspace/register_phantom_multiview.py:ro" \
    -w /workspace \
    fluorosim-torch \
    python /workspace/register_phantom_multiview.py

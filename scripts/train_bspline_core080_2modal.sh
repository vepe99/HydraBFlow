#!/usr/bin/env bash
# Two-modality model (stream B-spline tracks + Ou+2024 rotation curve) trained with the core080 B-spline
# augmentation (stream_*_streamfinder_bspline_core080): published STREAMFINDER great-circle frames, penalized
# spline fitted to the real members' central 80 % phi1 range + linear extension to the ends, lambda = 10 x GCV,
# M68 v_los = one global quadratic over all its measured stars. It is the GPU twin of the PPC
# outputs/Bsline/streamfinder_spray_p1e3_v4/ppc_spray_p1e3_v4_stream_aug_core080.
#
# Otherwise identical to outputs/Bsline/spray_p1e3_v4_smoothing_2modal: 3e5-row spray v4 set (1e3 particles),
# model stream_fusion_2modal_oldgrid, adapter stream_2modal, OLD observation model + STREAMFINDER members,
# missing_modality_prob 0.3, seed 2026.
#   [1/3] train  [2/3] evaluate sim (333 groups, base + compositional)  [3/3] evaluate real (STREAMFINDER)
# Everything runs on one GPU picked by autocvd (it waits for a free one); GPU=<id> pins it.
#
# Run:    bash scripts/train_bspline_core080_2modal.sh
# Knobs:  N_EPOCHS (1000) BATCH_SIZE (2048) N_TRAIN (300000) RUNS_DIR GPU
# Log:    ${RUNS_DIR}/run.log
set -euo pipefail
cd "$(dirname "$0")/.."

RUNS_DIR=${RUNS_DIR:-outputs/Bsline/spray_p1e3_v4_core080_2modal}
mkdir -p "${RUNS_DIR}"

export PY=${PY:-.venv/bin/python} AUTOCVD=${AUTOCVD:-.venv/bin/autocvd} GPU=${GPU:-auto}
SIM=stream_agama_spray_massloss_ibata_m200c_v4 \
DATA_DIR=data/data_jarvis/data_agama_spray_massloss_ibata_m200c_v4_p1e3_hydrabflow N_TRAIN=${N_TRAIN:-300000} \
MODEL=stream_fusion_2modal_oldgrid ADAPTER=stream_2modal \
PREPROC=stream_global_log10_sumstats_2modal REAL_PREPROC=stream_real_global_log10 \
AUG=stream_global_streamfinder_bspline_core080 REAL_AUG=stream_real_global_streamfinder_bspline_core080 \
N_EPOCHS=${N_EPOCHS:-1000} BATCH_SIZE=${BATCH_SIZE:-4096} RUNS_DIR="${RUNS_DIR}" \
bash scripts/train_v4_2modal.sh 2>&1 | tee "${RUNS_DIR}/run.log"

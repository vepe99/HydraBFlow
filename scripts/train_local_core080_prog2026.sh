#!/usr/bin/env bash
# LOCAL-level twin of scripts/train_bspline_core080_prog2026.sh (scripts/train_local_2modal.sh on the prog2026
# 3e5 set): infer each stream's [m_progenitor, t_end, vr, r, mu_ra_cosdec, mu_dec] from the core080 B-spline
# tracks + the rotation curve + the TRUE 7 globals (coupling flow, no modality dropout). Locals z-scored per
# stream on the train split. 500 epochs, batch 2048, seed 2026.
#   [1/3] train  [2/3] evaluate sim (333 groups, true globals, per-stream metrics)
#   [3/3] evaluate real by ancestral sampling with the globals of GLOBAL_RUN_DIR/posterior.npz — skipped if
#         that file does not exist yet (run the global script first, or rerun stage 3 later).
#
# Run:    bash scripts/train_local_core080_prog2026.sh
#         GPU=3 N_EPOCHS=1000 bash scripts/train_local_core080_prog2026.sh
# Knobs:  N_EPOCHS (500) BATCH_SIZE (2048) N_TRAIN (300000) GLOBAL_RUN_DIR RUNS_DIR GPU (auto)
set -euo pipefail
cd "$(dirname "$0")/.."

GLOBAL_RUN_DIR=${GLOBAL_RUN_DIR:-outputs/Bsline/spray_p1e3_v4_prog2026_core080_2modal_trial14/eval_real}
[ -f "${GLOBAL_RUN_DIR}/posterior.npz" ] || { echo "no ${GLOBAL_RUN_DIR}/posterior.npz: real-data stage will be skipped"; GLOBAL_RUN_DIR=; }
RUNS_DIR=${RUNS_DIR:-outputs/Bsline/spray_p1e3_v4_prog2026_core080_local}
mkdir -p "${RUNS_DIR}"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}
export PY=${PY:-.venv/bin/python} AUTOCVD=${AUTOCVD:-.venv/bin/autocvd} GPU=${GPU:-auto}
SIM=stream_agama_spray_massloss_ibata_m200c_v4_prog2026 \
DATA_DIR=data/data_jarvis/data_agama_spray_massloss_ibata_m200c_v4_p1e3_prog2026_hydrabflow N_TRAIN=${N_TRAIN:-300000} \
AUG=stream_global_streamfinder_bspline_core080 REAL_AUG=stream_real_global_streamfinder_bspline_core080 \
N_EPOCHS=${N_EPOCHS:-500} BATCH_SIZE=${BATCH_SIZE:-2048} \
GLOBAL_RUN_DIR="${GLOBAL_RUN_DIR}" RUNS_DIR="${RUNS_DIR}" \
bash scripts/train_local_2modal.sh 2>&1 | tee "${RUNS_DIR}/run.log"

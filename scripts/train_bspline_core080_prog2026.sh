#!/usr/bin/env bash
# Best hyperparameters of tuningtest_2modal_bspline_core080_study (tuned on the p1e3 spray v4 3e5 set),
# retrained on the prog2026 3e5 set (stream_agama_spray_massloss_ibata_m200c_v4_prog2026: 2026-09-22
# progenitor table, t_end U[2,5]). Same stack as the study: stream_fusion_2modal_oldgrid, core080 B-spline
# augmentation, STREAMFINDER members, drop prob 0.5, batch 4096, 1000 epochs, seed 2026.
# The architecture is read from the trial's params.json at run time (Pareto front: 14 = best calibration
# 0.0149 / RMSE 0.6591, 6 = best RMSE 0.6573 / calib 0.0155).
#   [1/3] train  [2/3] evaluate sim (test_multistream_333, base + compositional)  [3/3] evaluate real
#
# Run:    bash scripts/train_bspline_core080_prog2026.sh
#         TRIAL=6 GPU=3 bash scripts/train_bspline_core080_prog2026.sh
# Knobs:  TRIAL (14) N_EPOCHS (1000) BATCH_SIZE (4096) DROP_PROB (0.5) N_TRAIN (300000) RUNS_DIR GPU (auto)
set -euo pipefail
cd "$(dirname "$0")/.."

TRIAL=${TRIAL:-14}
STUDY_DIR=data/data_jarvis/data_agama_spray_massloss_ibata_m200c_v4_p1e3_hydrabflow/tuning/tuningtest_2modal_bspline_core080_study
PARAMS=${STUDY_DIR}/trials/trial_$(printf %04d "${TRIAL}")/params.json
RUNS_DIR=${RUNS_DIR:-outputs/Bsline/spray_p1e3_v4_prog2026_core080_2modal_trial${TRIAL}}
mkdir -p "${RUNS_DIR}"
cp "${PARAMS}" "${RUNS_DIR}/tuned_params.json"

# ++ because struct mode rejects dt_time_embedding_dim / dt_dropout (tune.py force-adds them)
TUNED=$(.venv/bin/python -c "import json,sys;print(' '.join(f'++{k}={v}' for k,v in json.load(open(sys.argv[1])).items()))" "${PARAMS}")

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}
export PY=${PY:-.venv/bin/python} AUTOCVD=${AUTOCVD:-.venv/bin/autocvd} GPU=${GPU:-auto}
SIM=stream_agama_spray_massloss_ibata_m200c_v4_prog2026 \
DATA_DIR=data/data_jarvis/data_agama_spray_massloss_ibata_m200c_v4_p1e3_prog2026_hydrabflow N_TRAIN=${N_TRAIN:-300000} \
MODEL=stream_fusion_2modal_oldgrid ADAPTER=stream_2modal \
PREPROC=stream_global_log10_sumstats_2modal REAL_PREPROC=stream_real_global_log10 \
AUG=stream_global_streamfinder_bspline_core080 REAL_AUG=stream_real_global_streamfinder_bspline_core080 \
N_EPOCHS=${N_EPOCHS:-1000} BATCH_SIZE=${BATCH_SIZE:-4096} DROP_PROB=${DROP_PROB:-0.5} RUNS_DIR="${RUNS_DIR}" \
EXTRA="${TUNED} ${EXTRA:-}" \
bash scripts/train_v4_2modal.sh 2>&1 | tee "${RUNS_DIR}/run.log"

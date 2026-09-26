#!/usr/bin/env bash
# Optuna study of scripts/train_bspline_core080_prog2026.sh: the tune_bspline_core080_2modal.sh study re-run on
# the prog2026 3e5 set (stream_agama_spray_massloss_ibata_m200c_v4_prog2026). Same model
# (stream_fusion_2modal_oldgrid), search space (tuningtest_2modal_bspline_core080 -> stream_2modal_oldgrid),
# core080 augmentation, STREAMFINDER real eval, drop prob 0.5, batch 4096, 1000 epochs. Each trial: train ->
# evaluate on the prog2026 test_multistream_333.npz (objectives = base RMSE + calibration) -> real Gaia.
# Study + trials: <DATA_DIR>/tuning/tuningtest_2modal_bspline_core080_prog2026_study/ (JournalStorage: launch
# this script again on another GPU to add a worker to the SAME study).
#   bash scripts/tune_bspline_core080_prog2026.sh                  # autocvd waits for a FREE GPU
#   GPU=6 N_TRIALS=25 bash scripts/tune_bspline_core080_prog2026.sh
# Knobs: N_TRIALS (25) N_EPOCHS (1000) BATCH_SIZE (4096) DROP_PROB (0.5) N_TRAIN (300000) GPU OUT_ROOT
set -euo pipefail
cd "$(dirname "$0")/.."
GPU=${GPU:-auto}
if [ "${GPU}" = "auto" ]; then eval "$(.venv/bin/autocvd -e -n 1)"; GPU=${CUDA_VISIBLE_DEVICES}; fi
export XLA_PYTHON_CLIENT_PREALLOCATE=true   # claim the card up front so nobody lands on it mid-trial
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}
OUT_ROOT=${OUT_ROOT:-outputs/Bsline/spray_p1e3_v4_prog2026_core080_2modal/tuning}
mkdir -p "${OUT_ROOT}"
TUNING=tuningtest_2modal_bspline_core080 STUDY=tuningtest_2modal_bspline_core080_prog2026_study \
SIM=stream_agama_spray_massloss_ibata_m200c_v4_prog2026 \
DATA_DIR=data/data_jarvis/data_agama_spray_massloss_ibata_m200c_v4_p1e3_prog2026_hydrabflow N_TRAIN=${N_TRAIN:-300000} \
MODEL=stream_fusion_2modal_oldgrid PREPROC=stream_global_log10_sumstats_2modal \
AUG=stream_global_streamfinder_bspline_core080 \
N_EPOCHS=${N_EPOCHS:-1000} BATCH_SIZE=${BATCH_SIZE:-4096} N_TRIALS=${N_TRIALS:-25} DROP_PROB=${DROP_PROB:-0.5} GPU=${GPU} \
OUT_ROOT="${OUT_ROOT}" bash scripts/tune_2modal_oldgrid.sh 2>&1 | tee -a "${OUT_ROOT}/worker_$(date +%Y%m%d_%H%M%S).log"

#!/usr/bin/env bash
# Test-set-scored Optuna study of the model trained by scripts/train_bspline_core080_2modal.sh: SAME training
# set (3e5-row p1e3 spray v4), SAME augmentation (stream_global_streamfinder_bspline_core080) and SAME real
# observation (STREAMFINDER members through stream_real_global_streamfinder_bspline_core080).
# Model stream_fusion_2modal_oldgrid, search space = conf/tuning/stream_2modal_oldgrid.yaml (two TST backbones
# + DiffusionTransformer subnet). Each trial: train (1000 ep) -> evaluate on test_multistream_333.npz
# (eval_sim/, objectives = base RMSE + calibration) -> real Gaia (eval_real/).
# Study + trials: <DATA_DIR>/tuning/tuningtest_2modal_bspline_core080_study/ (JournalStorage: run this script
# again on another GPU to add a worker to the SAME study). Worker logs under OUT_ROOT.
#   bash scripts/tune_bspline_core080_2modal.sh                  # autocvd waits for a FREE GPU
#   GPU=6 N_TRIALS=25 bash scripts/tune_bspline_core080_2modal.sh
# Knobs: N_TRIALS (25) N_EPOCHS (1000) BATCH_SIZE (4096, as the train script) DROP_PROB (0.3) N_TRAIN (300000)
set -euo pipefail
cd "$(dirname "$0")/.."
# A FREE gpu, not the least-used one (the shared runner's default is `autocvd -l`): block until one is free.
GPU=${GPU:-auto}
if [ "${GPU}" = "auto" ]; then eval "$(.venv/bin/autocvd -e -n 1)"; GPU=${CUDA_VISIBLE_DEVICES}; fi
# Claim the card as soon as JAX starts (default XLA 75 % preallocation) so nobody else lands on it mid-trial.
export XLA_PYTHON_CLIENT_PREALLOCATE=true
OUT_ROOT=${OUT_ROOT:-outputs/Bsline/spray_p1e3_v4_core080_2modal/tuning}
mkdir -p "${OUT_ROOT}"
TUNING=tuningtest_2modal_bspline_core080 \
SIM=stream_agama_spray_massloss_ibata_m200c_v4 \
DATA_DIR=data/data_jarvis/data_agama_spray_massloss_ibata_m200c_v4_p1e3_hydrabflow N_TRAIN=${N_TRAIN:-300000} \
MODEL=stream_fusion_2modal_oldgrid PREPROC=stream_global_log10_sumstats_2modal \
AUG=stream_global_streamfinder_bspline_core080 \
N_EPOCHS=${N_EPOCHS:-1000} BATCH_SIZE=${BATCH_SIZE:-4096} N_TRIALS=${N_TRIALS:-25} DROP_PROB=${DROP_PROB:-0.5} GPU=${GPU} \
OUT_ROOT="${OUT_ROOT}" bash scripts/tune_2modal_oldgrid.sh 2>&1 | tee -a "${OUT_ROOT}/worker_$(date +%Y%m%d_%H%M%S).log"

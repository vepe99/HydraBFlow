#!/usr/bin/env bash
# Optuna study of the scripts/train_bspline_core080_rc38_jiao26.sh model with the Eilers19 curve (+3 % systematic error): same model (stream_fusion_2modal_oldgrid), search
# space (stream_2modal_oldgrid), core080 augmentation, rc38 preprocessing, diffused prior score, drop prob 0.5,
# batch 4096, 1000 epochs, on the rc38 3e5 set with vcirc_kms on the 38 Eilers+2019 radii (errors + 3 % systematic in quadrature). Each trial: train ->
# evaluate on test_multistream_333.npz (objectives = base RMSE + calibration; compositional at d1=1.0) ->
# real Gaia (STREAMFINDER + Eilers19 curve, d1=1.0). The GPU memory is claimed (JAX preallocation, 75 %) at
# Python start-up, before the 14.7 GB training set is loaded; trial evaluations run in subprocesses on the rest.
# Study + trials: <DATA_DIR>/tuning/tuningtest_2modal_bspline_core080_rc38_eilers19_sys3_study/ (JournalStorage: launch
# again on another GPU to add a worker to the SAME study).
#   bash scripts/tune_bspline_core080_rc38_eilers19_sys3.sh            # autocvd waits for a FREE GPU
#   GPU=6 N_TRIALS=25 bash scripts/tune_bspline_core080_rc38_eilers19_sys3.sh
# Knobs: N_TRIALS (25) N_EPOCHS (1000) BATCH_SIZE (4096) DROP_PROB (0.5) N_TRAIN (300000) GPU OUT_ROOT
set -euo pipefail
cd "$(dirname "$0")/.."
GPU=${GPU:-auto}
if [ "${GPU}" = "auto" ]; then eval "$(.venv/bin/autocvd -e -n 1)"; GPU=${CUDA_VISIBLE_DEVICES}; fi
export HYDRABFLOW_EARLY_GPU_INIT=1 XLA_PYTHON_CLIENT_PREALLOCATE=true
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}
OUT_ROOT=${OUT_ROOT:-outputs/Bsline/spray_p1e3_v4_prog2026_rc38_eilers19_sys3_core080_2modal/tuning}
mkdir -p "${OUT_ROOT}"
TUNING=tuningtest_2modal_bspline_core080_rc38_eilers19_sys3 \
SIM=stream_agama_spray_massloss_ibata_m200c_v4_prog2026_rc38_eilers19_sys3 \
DATA_DIR=data/data_jarvis/data_agama_spray_massloss_ibata_m200c_v4_p1e3_prog2026_rc38_eilers19_hydrabflow N_TRAIN=${N_TRAIN:-300000} \
MODEL=stream_fusion_2modal_oldgrid PREPROC=stream_global_rc38_2modal \
AUG=stream_global_streamfinder_bspline_core080 EXTRA="eval.prior_score=diffused ${EXTRA:-}" \
N_EPOCHS=${N_EPOCHS:-1000} BATCH_SIZE=${BATCH_SIZE:-4096} N_TRIALS=${N_TRIALS:-25} DROP_PROB=${DROP_PROB:-0.5} GPU=${GPU} \
OUT_ROOT="${OUT_ROOT}" bash scripts/tune_2modal_oldgrid.sh 2>&1 | tee -a "${OUT_ROOT}/worker_$(date +%Y%m%d_%H%M%S).log"

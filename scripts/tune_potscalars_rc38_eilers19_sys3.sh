#!/usr/bin/env bash
# Optuna study of the pot_scalars model (scripts/train_potscalars_rc38_eilers19_sys3.sh: stream_fusion_2modal_oldgrid +
# an MLP after the rotation-curve summary fed [K_z, mu_l, M200]) on the rc38 x Eilers19 (+3 % sys) 3e5 set. Same setup
# as scripts/tune_bspline_core080_rc38_eilers19_sys3.sh: search space (stream_2modal_oldgrid), core080 augmentation,
# diffused prior score, drop prob 0.5, batch 4096, 1000 epochs. Each trial: train -> evaluate on test_multistream_333
# (objectives = base RMSE + calibration) -> real Gaia (pooled + per-stream) -> real Gaia with ONE modality observed
# (eval_real_only_sim_summary/, eval_real_only_vcirc_kms/ = curve + pot_scalars) -> eval_real/corner_modalities.png.
# Needs pot_scalars in the dataset (scripts/add_pot_scalars.py; already done for this set).
# Study + trials: <DATA_DIR>/tuning/tuningtest_2modal_bspline_core080_rc38_eilers19_sys3_potscalars_study/
# (JournalStorage: launch again on another GPU to add a worker to the SAME study).
#   bash scripts/tune_potscalars_rc38_eilers19_sys3.sh            # autocvd waits for a FREE GPU
#   GPU=6 N_TRIALS=25 bash scripts/tune_potscalars_rc38_eilers19_sys3.sh
# Knobs: N_TRIALS (25) N_EPOCHS (1000) BATCH_SIZE (4096) DROP_PROB (0.5) N_TRAIN (300000) GPU OUT_ROOT
set -euo pipefail
cd "$(dirname "$0")/.."
GPU=${GPU:-auto}
if [ "${GPU}" = "auto" ]; then eval "$(.venv/bin/autocvd -e -n 1)"; GPU=${CUDA_VISIBLE_DEVICES}; fi
export HYDRABFLOW_EARLY_GPU_INIT=1 XLA_PYTHON_CLIENT_PREALLOCATE=true
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}
OUT_ROOT=${OUT_ROOT:-outputs/Bsline/spray_p1e3_v4_prog2026_rc38_eilers19_sys3_potscalars_2modal/tuning}
mkdir -p "${OUT_ROOT}"
TUNING=tuningtest_2modal_bspline_core080_rc38_eilers19_sys3_potscalars \
SIM=stream_agama_spray_massloss_ibata_m200c_v4_prog2026_rc38_eilers19_sys3 \
DATA_DIR=data/data_jarvis/data_agama_spray_massloss_ibata_m200c_v4_p1e3_prog2026_rc38_eilers19_hydrabflow N_TRAIN=${N_TRAIN:-300000} \
MODEL=stream_fusion_2modal_oldgrid_potscalars ADAPTER=stream_2modal_potscalars PREPROC=stream_global_rc38_2modal_potscalars \
AUG=stream_global_streamfinder_bspline_core080_potscalars EXTRA="eval.prior_score=diffused ${EXTRA:-}" \
N_EPOCHS=${N_EPOCHS:-1000} BATCH_SIZE=${BATCH_SIZE:-4096} N_TRIALS=${N_TRIALS:-25} DROP_PROB=${DROP_PROB:-0.5} GPU=${GPU} \
OUT_ROOT="${OUT_ROOT}" bash scripts/tune_2modal_oldgrid.sh 2>&1 | tee -a "${OUT_ROOT}/worker_$(date +%Y%m%d_%H%M%S).log"

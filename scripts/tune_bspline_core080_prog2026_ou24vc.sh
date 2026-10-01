#!/usr/bin/env bash
# Optuna study of the GLOBAL model with the Ou et al. 2024 circular-velocity table (37 radii, 6.27-27.31 kpc) as the
# v_c observable: the tune_bspline_core080_prog2026.sh study on the ou24vc copy of the prog2026 3e5 set (vcirc_kms
# recomputed by scripts/recompute_vcirc_grid.py), simulator stream_agama_spray_massloss_ibata_m200c_v4_prog2026_ou24vc
# (its radii / sigma / observed curve reach add_noise_to_vcirc, mask_vcirc_radii and attach_observed_vcirc).
# Same model (stream_fusion_2modal_oldgrid), search space (tuningtest_2modal_bspline_core080), core080 augmentation,
# drop prob 0.5, batch 4096, 1000 epochs, as outputs/Bsline/spray_p1e3_v4_prog2026_ou24vc_core080_2modal_trial14.
# Study + trials: <DATA_DIR>/tuning/tuningtest_2modal_bspline_core080_prog2026_ou24vc_study/ (JournalStorage: launch
# this script again on another GPU to add a worker to the SAME study).
#   bash scripts/tune_bspline_core080_prog2026_ou24vc.sh                  # autocvd waits for a FREE GPU
#   GPU=7 N_TRIALS=25 bash scripts/tune_bspline_core080_prog2026_ou24vc.sh
# Knobs: N_TRIALS (25) N_EPOCHS (1000) BATCH_SIZE (4096) DROP_PROB (0.5) N_TRAIN (300000) GPU OUT_ROOT
set -euo pipefail
cd "$(dirname "$0")/.."
GPU=${GPU:-auto}
if [ "${GPU}" = "auto" ]; then eval "$(.venv/bin/autocvd -e -n 1)"; GPU=${CUDA_VISIBLE_DEVICES}; fi
export XLA_PYTHON_CLIENT_PREALLOCATE=true   # claim the card up front so nobody lands on it mid-trial
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}
OUT_ROOT=${OUT_ROOT:-outputs/Bsline/spray_p1e3_v4_prog2026_ou24vc_core080_2modal/tuning}
mkdir -p "${OUT_ROOT}"
TUNING=tuningtest_2modal_bspline_core080 STUDY=tuningtest_2modal_bspline_core080_prog2026_ou24vc_study \
SIM=stream_agama_spray_massloss_ibata_m200c_v4_prog2026_ou24vc \
DATA_DIR=data/data_jarvis/data_agama_spray_massloss_ibata_m200c_v4_p1e3_prog2026_ou24vc_hydrabflow N_TRAIN=${N_TRAIN:-300000} \
MODEL=stream_fusion_2modal_oldgrid PREPROC=stream_global_log10_sumstats_2modal \
AUG=stream_global_streamfinder_bspline_core080 \
N_EPOCHS=${N_EPOCHS:-1000} BATCH_SIZE=${BATCH_SIZE:-4096} N_TRIALS=${N_TRIALS:-25} DROP_PROB=${DROP_PROB:-0.5} GPU=${GPU} \
OUT_ROOT="${OUT_ROOT}" bash scripts/tune_2modal_oldgrid.sh 2>&1 | tee -a "${OUT_ROOT}/worker_$(date +%Y%m%d_%H%M%S).log"

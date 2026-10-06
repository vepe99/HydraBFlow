#!/usr/bin/env bash
# rc38 x Eilers19 (+3 % sys) 2-modal model with three extra potential constraints in the CURVE branch:
# pot_scalars = [K_z(R0, 1.1 kpc), mu_l(Sgr A*), M200 total] -> an MLP after the rotation-curve summary
# (model=stream_fusion_2modal_oldgrid_potscalars; same modality, so the masked compositional eval is unchanged).
# Training noise = symmetrized observational errors (0.16 km^2 s^-2 pc^-1, 0.026 mas/yr, 0.21e12 Msun);
# real eval attaches the observed values (2.00, -6.379, 1.17e12).
#   [0/3] scripts/add_pot_scalars.py on the training + 333 test set (CPU, appends the key in place; skipped
#         if present)  [1/3] train  [2/3] evaluate sim  [3/3] evaluate real
# Hyperparameters: trial 31 of the sys3 core080 study = Optuna Pareto front (val RMSE 0.702, calib 0.0094) with the
# best real-data MMD p_strat on the front (0.045). Front = 1/3/14/31; trial 24 = least misspecified (p 0.195) but off-front.
# Run:    bash scripts/train_potscalars_rc38_eilers19_sys3.sh          (GPU=<id> to pin; default autocvd)
# Knobs:  TRIAL (31) STUDY_DIR N_EPOCHS (1000) N_WORKERS (32, for step 0) RUNS_DIR GPU
set -euo pipefail
cd "$(dirname "$0")/.."
DATA_DIR=${DATA_DIR:-data/data_jarvis/data_agama_spray_massloss_ibata_m200c_v4_p1e3_prog2026_rc38_eilers19_hydrabflow}
TRIAL=${TRIAL:-31}

for f in test_multistream_333 training_data_${N_TRAIN:-300000}; do
  echo "[0/3] pot_scalars -> ${DATA_DIR}/${f}.npz"
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu HYDRABFLOW_NUM_GPUS=0 \
    nice .venv/bin/python scripts/add_pot_scalars.py "${DATA_DIR}/${f}.npz" --n-workers "${N_WORKERS:-32}"
done

TRIAL=${TRIAL} STUDY_DIR=${STUDY_DIR:-data/data_jarvis/data_agama_spray_massloss_ibata_m200c_v4_p1e3_prog2026_rc38_eilers19_hydrabflow/tuning/tuningtest_2modal_bspline_core080_rc38_eilers19_sys3_study} \
SIM=stream_agama_spray_massloss_ibata_m200c_v4_prog2026_rc38_eilers19_sys3 DATA_DIR=${DATA_DIR} \
MODEL=stream_fusion_2modal_oldgrid_potscalars ADAPTER=stream_2modal_potscalars \
PREPROC=stream_global_rc38_2modal_potscalars REAL_PREPROC=stream_real_global_potscalars \
AUG=stream_global_streamfinder_bspline_core080_potscalars REAL_AUG=stream_real_global_streamfinder_bspline_core080 \
EXTRA="eval.prior_score=diffused ${EXTRA:-}" N_EPOCHS=${N_EPOCHS:-1000} \
RUNS_DIR=${RUNS_DIR:-outputs/Bsline/rc38_eilers19_sys3_potscalars_trial${TRIAL}} \
bash scripts/train_bspline_core080_prog2026.sh

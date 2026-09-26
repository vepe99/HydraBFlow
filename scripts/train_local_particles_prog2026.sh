#!/usr/bin/env bash
# Particle twin of scripts/train_local_core080_prog2026.sh (as outputs/Bsline/spray_p1e3_v4_particles_local is
# of the B-spline local run): same prog2026 3e5 set, same local targets/conditions/coupling flow, but the stream
# modality is the raw star cloud through the LEGACY observation model (stream_global: RA/Dec window,
# 129/195/297 attended stars, Gaia errors, vlos zero-fill) -> masked SetTransformer. STREAMFINDER members.
#   [1/3] train  [2/3] evaluate sim (333 groups, true globals)  [3/3] ancestral real eval, skipped until
#   GLOBAL_RUN_DIR/posterior.npz exists.
# Run:    GPU=7 bash scripts/train_local_particles_prog2026.sh
# Knobs:  N_EPOCHS (1000) BATCH_SIZE (1024; SetTransformer ~13 GB at 512) N_TRAIN (300000) GLOBAL_RUN_DIR RUNS_DIR GPU
set -euo pipefail
cd "$(dirname "$0")/.."

GLOBAL_RUN_DIR=${GLOBAL_RUN_DIR:-outputs/Bsline/spray_p1e3_v4_prog2026_core080_2modal_trial14/eval_real}
[ -f "${GLOBAL_RUN_DIR}/posterior.npz" ] || { echo "no ${GLOBAL_RUN_DIR}/posterior.npz: real-data stage will be skipped"; GLOBAL_RUN_DIR=; }
RUNS_DIR=${RUNS_DIR:-outputs/Bsline/spray_p1e3_v4_prog2026_particles_local}
mkdir -p "${RUNS_DIR}"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}
export PY=${PY:-.venv/bin/python} AUTOCVD=${AUTOCVD:-.venv/bin/autocvd} GPU=${GPU:-auto}
SIM=stream_agama_spray_massloss_ibata_m200c_v4_prog2026 \
DATA_DIR=data/data_jarvis/data_agama_spray_massloss_ibata_m200c_v4_p1e3_prog2026_hydrabflow N_TRAIN=${N_TRAIN:-300000} \
MODEL=stream_fusion_2modal_particles_local ADAPTER=stream_2modal_particles_local \
AUG=stream_global REAL_AUG=stream_real_global \
EXTRA="augmentation.params.vlos_impute=zero ${EXTRA:-}" \
N_EPOCHS=${N_EPOCHS:-1000} BATCH_SIZE=${BATCH_SIZE:-1024} \
GLOBAL_RUN_DIR="${GLOBAL_RUN_DIR}" RUNS_DIR="${RUNS_DIR}" \
bash scripts/train_local_2modal.sh 2>&1 | tee "${RUNS_DIR}/run.log"

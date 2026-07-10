#!/usr/bin/env bash
# Train + evaluate the Ibata-model fusion approximator on the datasets produced by
# scripts/create_ibata_dataset.sh (full potential = fixed bulge + fixed HI/H2 gas disks +
# truncated triaxial halo + free thin/thick stellar disks; Chen+2024 spray; extended Zhou u Huang
# rotation curve + banded rejection prior; freed halo beta; and the three ancillary potential
# observables HI v_term / Sigma(1.1 kpc) / rho(z)). Dataset generation is CPU/joblib and lives in
# create_ibata_dataset.sh; this script is the GPU (JAX) half.
#
#   [1/2] train  -> ${MODEL_DIR}
#   [2/2] evaluate on the 333-group sim multistream test set -> ${EVAL_DIR}
#
# Real-data (Gaia) evaluation is intentionally NOT wired here: the Ibata ancillary observables
# (v_term, Sigma_z, rho(z)) have no real-data augmentation/preprocessing presets yet (rho(z) real
# data is a documented TODO), so there is no stream_real_global_ibata path to run against.
#
# Run:            bash scripts/train_ibata.sh
# Override knobs: N_EPOCHS=100 GPU=2 bash scripts/train_ibata.sh
#                 MODEL=stream_fusion_ibata_sigma_cond ADAPTER=stream_ibata_sigma_cond bash scripts/train_ibata.sh

set -euo pipefail
cd "$(dirname "$0")/.."

# --- GPU selection: autocvd picks a free card (org convention); GPU=<idx>|cpu overrides -------
GPU=${GPU:-auto}
if [ "${GPU}" = "cpu" ]; then
  export JAX_PLATFORMS=cpu
else
  if [ "${GPU}" = "auto" ]; then
    eval "$(uv run autocvd -n 1)"
  else
    export CUDA_VISIBLE_DEVICES="${GPU}"
  fi
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
fi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<cpu>}"

# ---- knobs -------------------------------------------------------------------------------------
SIM=${SIM:-stream_agama_ibata}
DATA_DIR=${DATA_DIR:-data_jarvis/data_agama_ibata_hydrabflow}
RES=${RES:-assets/gaia}
SEED=${SEED:-2026}
N_EPOCHS=${N_EPOCHS:-300}
BATCH_SIZE=${BATCH_SIZE:-1024}
N_TRAIN=${N_TRAIN:-100000}   # -> training_data_100000.npz
N_TEST=${N_TEST:-333}        # -> test_multistream_333.npz

MODEL=${MODEL:-stream_fusion_ibata}
ADAPTER=${ADAPTER:-stream_ibata}
AUG=${AUG:-stream_global_ibata}
PREPROC=${PREPROC:-stream_global_log10_ibata}

RUNS_DIR=${RUNS_DIR:-outputs/${SIM}/${MODEL}}
MODEL_DIR=${MODEL_DIR:-${RUNS_DIR}/train}
EVAL_DIR=${EVAL_DIR:-${RUNS_DIR}/eval_sim_${N_TEST}}

echo "=== [1/2] TRAIN  -> ${MODEL_DIR} ==="
uv run python scripts/train.py \
  simulator="${SIM}" model="${MODEL}" composition=global \
  adapter="${ADAPTER}" preprocessing="${PREPROC}" augmentation="${AUG}" \
  data.data_dir="${DATA_DIR}" data.n_simulations="${N_TRAIN}" \
  training.n_epochs="${N_EPOCHS}" training.batch_size="${BATCH_SIZE}" seed="${SEED}" \
  augmentation.params.resources_dir="${RES}" \
  hydra.run.dir="${MODEL_DIR}"

echo "=== [2/2] EVALUATE on simulated ${N_TEST}-group multistream test set -> ${EVAL_DIR} ==="
uv run python scripts/evaluate.py \
  simulator="${SIM}" model="${MODEL}" composition=global \
  adapter="${ADAPTER}" preprocessing="${PREPROC}" augmentation="${AUG}" \
  eval=stream_compositional data.data_dir="${DATA_DIR}" data.n_simulations="${N_TEST}" \
  eval.test_dataset_name=test_multistream_${N_TEST}.npz \
  model_dir="${MODEL_DIR}" augmentation.params.resources_dir="${RES}" \
  hydra.run.dir="${EVAL_DIR}"

echo "=== DONE. Outputs under ${RUNS_DIR} ==="

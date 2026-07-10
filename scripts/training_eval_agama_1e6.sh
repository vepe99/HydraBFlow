#!/usr/bin/env bash
# Train on the 1e6-row plain-agama spray set (data/streams/data_agama_hydrabflow, generated
# 2026-07-03: Fardal spray, 34-radii Zhou rotation curve, fixed beta=3, no rejection prior)
# with the missing-vlos-ablation-recommended model: masked_set_transformer stream backbone +
# constant-zero vlos fill. 300 epochs (user choice — ~880 steps/epoch at 1e6 rows, ~11-14 h),
# then evaluate on the 333-group sim test set (older filename: simulation_multistream_333.npz,
# hence the eval.test_dataset_name override) and on the real Gaia streams.
#
# Run:            bash scripts/training_eval_agama_1e6.sh
# Override knobs: N_EPOCHS=100 GPU=2 bash scripts/training_eval_agama_1e6.sh

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
DATA_DIR=${DATA_DIR:-data/streams/data_agama_hydrabflow}
RES=${RES:-assets/gaia}
REAL=${REAL:-assets/gaia/gaia_observed_streams_6Dwitherrors_cutNGC3201.npz}
SEED=${SEED:-2026}
N_EPOCHS=${N_EPOCHS:-300}
BATCH_SIZE=${BATCH_SIZE:-1024}
N_TRAIN=${N_TRAIN:-1000000}   # -> training_data_1000000.npz
N_TEST=${N_TEST:-333}         # -> simulation_multistream_333.npz (older naming, overridden below)
MODEL=${MODEL:-stream_fusion_model5_maskedvlos}

RUNS_DIR=${RUNS_DIR:-outputs/stream_agama/${MODEL}_1e6}
MODEL_DIR=${RUNS_DIR}/train
EVAL_DIR=${RUNS_DIR}/eval_sim_${N_TEST}
REAL_DIR=${RUNS_DIR}/eval_real

echo "=== [1/3] TRAIN  -> ${MODEL_DIR} ==="
uv run python scripts/train.py \
  simulator=stream_agama model="${MODEL}" composition=global \
  adapter=stream preprocessing=stream_global_log10 augmentation=stream_global \
  augmentation.params.vlos_impute=zero \
  data.data_dir="${DATA_DIR}" data.n_simulations="${N_TRAIN}" \
  training.n_epochs="${N_EPOCHS}" training.batch_size="${BATCH_SIZE}" seed="${SEED}" \
  augmentation.params.resources_dir="${RES}" \
  hydra.run.dir="${MODEL_DIR}"

echo "=== [2/3] EVALUATE on simulated ${N_TEST}-group multistream test set -> ${EVAL_DIR} ==="
uv run python scripts/evaluate.py \
  simulator=stream_agama model="${MODEL}" composition=global \
  adapter=stream preprocessing=stream_global_log10 augmentation=stream_global \
  augmentation.params.vlos_impute=zero \
  eval=stream_compositional data.data_dir="${DATA_DIR}" data.n_simulations="${N_TEST}" \
  eval.test_dataset_name=simulation_multistream_${N_TEST}.npz \
  model_dir="${MODEL_DIR}" augmentation.params.resources_dir="${RES}" \
  hydra.run.dir="${EVAL_DIR}"

echo "=== [3/3] EVALUATE REAL (Gaia Pal5/NGC3201/M68) -> ${REAL_DIR} ==="
uv run python scripts/evaluate_real.py \
  simulator=stream_agama model="${MODEL}" composition=global \
  adapter=stream preprocessing=stream_real_global_log10 augmentation=stream_real_global \
  augmentation.params.vlos_impute=zero \
  data.real_data_path="${REAL}" \
  model_dir="${MODEL_DIR}" augmentation.params.resources_dir="${RES}" \
  eval.misspecification_reference="${EVAL_DIR}" \
  hydra.run.dir="${REAL_DIR}"

echo "=== DONE. Outputs under ${RUNS_DIR} ==="

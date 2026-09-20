#!/usr/bin/env bash
# v4 (rnbody Ibata m200c, 10^4 particles, Ou+2024 rotation curve) trained with the TWO-MODALITY
# masked model: the φ1-gridded per-stream summary statistics (median + robust dispersion per bin,
# plus occupancy) and the 19-radius rotation curve, each entering the compositional posterior
# exactly once. See conf/eval/stream_compositional_masked.yaml for why that matters.
#
#   [1/3] train        -> ${MODEL_DIR}
#   [2/3] evaluate sim -> ${EVAL_DIR}   (333-group multistream test set; writes BOTH the per-stream
#                                        base_* metrics and the pooled compositional_* metrics)
#   [3/3] evaluate real-> ${REAL_DIR}   (Gaia Pal5/NGC3201/M68)
#
# Run:    bash scripts/train_v4_2modal.sh
# Smoke:  N_EPOCHS=2 BATCH_SIZE=256 RUNS_DIR=/tmp/v4smoke bash scripts/train_v4_2modal.sh
# GPU:    GPU=6 bash scripts/train_v4_2modal.sh   (GPU=cpu forces CPU; default: autocvd)

set -euo pipefail
cd "$(dirname "$0")/.."

GPU=${GPU:-auto}
if [ "${GPU}" = "cpu" ]; then
  export JAX_PLATFORMS=cpu
else
  if [ "${GPU}" = "auto" ]; then
    eval "$(${AUTOCVD:-uv run autocvd} -n 1)"
    export CUDA_VISIBLE_DEVICES   # autocvd prints a plain assignment; unexported, the Python-side autocvd waits again
  else
    export CUDA_VISIBLE_DEVICES="${GPU}"
  fi
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
fi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<cpu>}"

SIM=${SIM:-stream_agama_rnbody_ibata_m200c_v4}
DATA_DIR=${DATA_DIR:-data/data_jarvis/data_agama_rnbody_ibata_m200c_v4_hydrabflow}
RES=${RES:-assets/gaia}
REAL=${REAL:-assets/gaia/gaia_observed_streams_6Dwitherrors_cutNGC3201.npz}
SEED=${SEED:-2026}
N_EPOCHS=${N_EPOCHS:-1000}
BATCH_SIZE=${BATCH_SIZE:-1024}
N_TRAIN=${N_TRAIN:-100000}   # -> training_data_100000.npz
N_TEST=${N_TEST:-333}        # -> test_multistream_333.npz

MODEL=${MODEL:-stream_fusion_2modal}
ADAPTER=${ADAPTER:-stream_2modal}
AUG=${AUG:-stream_global_ibata_grid_v2}
REAL_AUG=${REAL_AUG:-stream_real_global_ibata_grid_v2}
PREPROC=${PREPROC:-stream_global_log10_ibata_sumstats}
REAL_PREPROC=${REAL_PREPROC:-stream_real_global_ibata_sumstats}
EVAL=${EVAL:-stream_compositional_masked}
DROP_PROB=${DROP_PROB:-0.3}   # per-group, per-sample modality dropout during training
EXTRA=${EXTRA:-}             # extra Hydra overrides applied to all three stages (space-separated)

RUNS_DIR=${RUNS_DIR:-outputs/v4_2modal/default}
MODEL_DIR=${MODEL_DIR:-${RUNS_DIR}/train}
EVAL_DIR=${EVAL_DIR:-${RUNS_DIR}/eval_sim_${N_TEST}}
REAL_DIR=${REAL_DIR:-${RUNS_DIR}/eval_real}

echo "=== [1/3] TRAIN  -> ${MODEL_DIR} ==="
${PY:-uv run python} -m hydrabflow.pipeline.train \
  simulator="${SIM}" model="${MODEL}" composition=global \
  adapter="${ADAPTER}" preprocessing="${PREPROC}" augmentation="${AUG}" \
  model.inference_network.params.missing_modality_prob="${DROP_PROB}" \
  data.data_dir="${DATA_DIR}" data.n_simulations="${N_TRAIN}" \
  training.n_epochs="${N_EPOCHS}" training.batch_size="${BATCH_SIZE}" seed="${SEED}" \
  augmentation.params.resources_dir="${RES}" \
  hydra.run.dir="${MODEL_DIR}" ${EXTRA}

echo "=== [2/3] EVALUATE sim ${N_TEST}-group multistream -> ${EVAL_DIR} ==="
${PY:-uv run python} -m hydrabflow.pipeline.evaluate \
  simulator="${SIM}" model="${MODEL}" composition=global \
  adapter="${ADAPTER}" preprocessing="${PREPROC}" augmentation="${AUG}" \
  eval="${EVAL}" eval.batch_size=8 data.data_dir="${DATA_DIR}" data.n_simulations="${N_TEST}" \
  eval.test_dataset_name=test_multistream_${N_TEST}.npz \
  model_dir="${MODEL_DIR}" augmentation.params.resources_dir="${RES}" \
  hydra.run.dir="${EVAL_DIR}" ${EXTRA}

echo "=== [3/3] EVALUATE REAL (Gaia Pal5/NGC3201/M68) -> ${REAL_DIR} ==="
${PY:-uv run python} -m hydrabflow.pipeline.evaluate \
  simulator="${SIM}" model="${MODEL}" composition=global \
  adapter="${ADAPTER}" preprocessing="${REAL_PREPROC}" augmentation="${REAL_AUG}" \
  eval="${EVAL}" eval.batch_size=8 data.real_data_path="${REAL}" \
  model_dir="${MODEL_DIR}" augmentation.params.resources_dir="${RES}" \
  eval.misspecification_reference="${EVAL_DIR}" \
  hydra.run.dir="${REAL_DIR}" ${EXTRA}

echo "=== DONE. Outputs under ${RUNS_DIR} ==="

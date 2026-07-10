#!/usr/bin/env bash
# Train model_5 (fusion) on the restricted-N-body + free-beta + Zhou∪Huang-rejection 60k set,
# then evaluate on the 333-group simulated multistream test set and on the real Gaia streams.
# Autonomous: the three stages run back-to-back and every eval automatically points at the
# training run dir produced by stage 1 (no hand-editing of <TRAIN_RUN_DIR> between runs).
#
# CPU only (no GPU on this machine): JAX_PLATFORMS=cpu. The Gaia augmentation resources come
# from the git-tracked assets/gaia/ copy (the data/ symlink may not be mounted here).
#
# Run it (from anywhere):   bash scripts/training_eval_rnbody_huand_dataset.sh
# Override any knob:        SEED=7 N_EPOCHS=300 bash scripts/training_eval_rnbody_huand_dataset.sh
# Log to a file + background:
#   nohup bash scripts/training_eval_rnbody_huand_dataset.sh > train_eval.log 2>&1 &

# -e: abort on first error   -u: error on unset var   -o pipefail: a failing stage in a pipe fails
set -euo pipefail
# Run from the repo root regardless of where the script was invoked (so relative paths resolve).
cd "$(dirname "$0")/.."

# --- GPU selection ------------------------------------------------------------------------------
# JAX (with jax[cuda12], already a dependency) auto-detects the GPU, so we do NOT set
# JAX_PLATFORMS=cpu. CUDA_VISIBLE_DEVICES pins ONE physical GPU so we don't grab a busy one;
# override with e.g.  GPU=2 bash scripts/...  . Set GPU=cpu to force CPU instead.
GPU=${GPU:-2}
if [ "${GPU}" = "cpu" ]; then
  export JAX_PLATFORMS=cpu
else
  export CUDA_VISIBLE_DEVICES="${GPU}"
  # Don't let JAX pre-grab 75% of VRAM; grow as needed so we coexist on a shared card.
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
fi

# ---- knobs (override from the environment; the ${VAR:-default} form keeps the defaults) -------
# data_dir is used verbatim relative to the repo root (Hydra does not chdir). The dataset lives
# under the `data/` symlink -> data/data_jarvis/data_agama_rnbody_huang_hydrabflow (both
# training_data_60000.npz and test_multistream_333.npz). NOTE: the bare `data_jarvis/` root
# symlink points at a DIFFERENT disk that does not hold these files, so the `data/` prefix matters.
DATA_DIR=${DATA_DIR:-data/data_jarvis/data_agama_rnbody_huang_hydrabflow}
RES=${RES:-assets/gaia}
REAL=${REAL:-assets/gaia/gaia_observed_streams_6Dwitherrors_cutNGC3201.npz}
SEED=${SEED:-2026}
N_EPOCHS=${N_EPOCHS:-300}
BATCH_SIZE=${BATCH_SIZE:-1024}
N_TRAIN=${N_TRAIN:-60000}   # -> training_data_60000.npz
N_TEST=${N_TEST:-333}       # -> test_multistream_333.npz

# All run outputs (trained model + both evals) land together under data_jarvis/runs so the whole
# deliverable lives in one place next to the datasets. Pinning hydra.run.dir (instead of the
# default timestamped dir) is what lets the eval stages reference MODEL_DIR deterministically.
RUNS_DIR=${RUNS_DIR:-outputs/stream_agama_rnbody/stream_fusion_model5_rnbody_huang/300epochs}
MODEL_DIR=${RUNS_DIR}/train
EVAL_DIR=${RUNS_DIR}/eval_sim_${N_TEST}
REAL_DIR=${RUNS_DIR}/eval_real

echo "=== [1/3] TRAIN  -> ${MODEL_DIR} ==="
uv run python scripts/train.py \
  simulator=stream_agama_rnbody_huang model=stream_fusion_model5 composition=global \
  adapter=stream preprocessing=stream_global_log10 augmentation=stream_global \
  data.data_dir="${DATA_DIR}" data.n_simulations="${N_TRAIN}" \
  training.n_epochs="${N_EPOCHS}" training.batch_size="${BATCH_SIZE}" seed="${SEED}" \
  augmentation.params.resources_dir="${RES}" \
  hydra.run.dir="${MODEL_DIR}"

echo "=== [2/3] EVALUATE on simulated ${N_TEST}-group multistream test set -> ${EVAL_DIR} ==="
uv run python scripts/evaluate.py \
  simulator=stream_agama_rnbody_huang model=stream_fusion_model5 composition=global \
  adapter=stream preprocessing=stream_global_log10 augmentation=stream_global \
  eval=stream_compositional data.data_dir="${DATA_DIR}" data.n_simulations="${N_TEST}" \
  model_dir="${MODEL_DIR}" augmentation.params.resources_dir="${RES}" \
  hydra.run.dir="${EVAL_DIR}"

echo "=== [3/3] EVALUATE REAL (Gaia Pal5/NGC3201/M68) -> ${REAL_DIR} ==="
uv run python scripts/evaluate_real.py \
  simulator=stream_agama_rnbody_huang model=stream_fusion_model5 composition=global \
  adapter=stream preprocessing=stream_real_global_log10 augmentation=stream_real_global \
  data.real_data_path="${REAL}" \
  model_dir="${MODEL_DIR}" augmentation.params.resources_dir="${RES}" \
  eval.misspecification_reference="${EVAL_DIR}" \
  hydra.run.dir="${REAL_DIR}"

echo "=== DONE. Outputs under ${RUNS_DIR} ==="

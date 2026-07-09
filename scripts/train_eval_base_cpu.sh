#!/usr/bin/env bash
# Train the BASE stream model (model=stream_fusion) for 300 epochs on the rnbody+Huang 60k set,
# then evaluate on the 333-group multistream test set and on the real Gaia streams. CPU ONLY
# (no GPU on this machine): JAX_PLATFORMS=cpu. The augmentation Gaia resources come from the
# git-tracked assets/gaia/ copy (the data/ symlink is not mounted here).
set -euo pipefail
cd "$(dirname "$0")/.."

export JAX_PLATFORMS=cpu

DATA_DIR=${DATA_DIR:-data_jarvis/data_agama_rnbody_huang_hydrabflow}
RES=${RES:-assets/gaia}
REAL=${REAL:-assets/gaia/gaia_observed_streams_6Dwitherrors_cutNGC3201.npz}
SEED=${SEED:-2026}

# All run outputs (trained model, sim eval, real eval) saved under data_jarvis, alongside the
# datasets, so the whole deliverable lives in one place.
RUNS_DIR=${RUNS_DIR:-data_jarvis/runs/stream_fusion_base_cpu_300ep}
MODEL_DIR=${RUNS_DIR}/train
EVAL_DIR=${RUNS_DIR}/eval_sim_333
REAL_DIR=${RUNS_DIR}/eval_real

# 1) TRAIN (base model, 300 epochs)
uv run python scripts/train.py \
  simulator=stream_agama_rnbody_huang model=stream_fusion composition=global \
  adapter=stream preprocessing=stream_global augmentation=stream_global \
  data.data_dir="${DATA_DIR}" data.n_simulations=60000 \
  training.n_epochs=300 seed="${SEED}" \
  augmentation.params.resources_dir="${RES}" \
  hydra.run.dir="${MODEL_DIR}"

# 2) EVALUATE on the simulated 333-group multistream test set
uv run python scripts/evaluate.py \
  simulator=stream_agama_rnbody_huang model=stream_fusion composition=global \
  adapter=stream preprocessing=stream_global augmentation=stream_global \
  eval=stream_compositional data.data_dir="${DATA_DIR}" data.n_simulations=333 \
  model_dir="${MODEL_DIR}" augmentation.params.resources_dir="${RES}" \
  hydra.run.dir="${EVAL_DIR}"

# 3) EVALUATE REAL (observed Gaia Pal5/NGC3201/M68); MMD vs the sim-eval summaries
uv run python scripts/evaluate_real.py \
  simulator=stream_agama_rnbody_huang model=stream_fusion composition=global \
  adapter=stream preprocessing=stream_real_global augmentation=stream_real_global \
  data.real_data_path="${REAL}" \
  model_dir="${MODEL_DIR}" augmentation.params.resources_dir="${RES}" \
  eval.misspecification_reference="${EVAL_DIR}" \
  hydra.run.dir="${REAL_DIR}"

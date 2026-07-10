#!/usr/bin/env bash
# Missing-vlos ablation on the restricted-N-body + Zhou∪Huang 60k set: how should stars WITHOUT
# a measured line-of-sight velocity enter the summary network?
#
#   baseline   (already trained, NOT run here): mean-impute + indicator channel — the historical
#              behavior; outputs/stream_agama_rnbody/stream_fusion_model5_rnbody_huang/
#   zerofill   arm B: same architecture, constant-0 fill + indicator (Wang et al. 2024 "E2")
#              -> model=stream_fusion_model5            augmentation.params.vlos_impute=zero
#   maskedvlos arm A: masked_set_transformer — v_los channels hard-zeroed for unmeasured stars
#              + learned missing-v_los embedding (BERT-style mask token)
#              -> model=stream_fusion_model5_maskedvlos augmentation.params.vlos_impute=zero
#
# Everything else matches scripts/training_eval_rnbody_huand_dataset.sh (same data, seed,
# epochs, batch size), so differences are attributable to the missing-vlos handling alone.
#
# Run both arms:        bash scripts/training_eval_missing_vlos_ablation.sh
# One arm / overrides:  ARMS=maskedvlos N_EPOCHS=2 bash scripts/training_eval_missing_vlos_ablation.sh

set -euo pipefail
cd "$(dirname "$0")/.."

# --- GPU selection ------------------------------------------------------------------------------
# GPU=auto (default) lets autocvd pick a currently-free GPU (org convention for JAX GPU runs).
# GPU=<idx> pins one card; GPU=cpu forces CPU.
GPU=${GPU:-auto}
if [ "${GPU}" = "cpu" ]; then
  export JAX_PLATFORMS=cpu
else
  if [ "${GPU}" = "auto" ]; then
    eval "$(uv run autocvd -n 1)"
  else
    export CUDA_VISIBLE_DEVICES="${GPU}"
  fi
  # Don't pre-grab 75% of VRAM; grow as needed so we coexist on a shared card.
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
fi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<cpu>}"

# ---- knobs (override from the environment) -----------------------------------------------------
DATA_DIR=${DATA_DIR:-data/data_jarvis/data_agama_rnbody_huang_hydrabflow}
RES=${RES:-assets/gaia}
REAL=${REAL:-assets/gaia/gaia_observed_streams_6Dwitherrors_cutNGC3201.npz}
SEED=${SEED:-2026}
N_EPOCHS=${N_EPOCHS:-1000}
BATCH_SIZE=${BATCH_SIZE:-1024}
N_TRAIN=${N_TRAIN:-60000}   # -> training_data_60000.npz
N_TEST=${N_TEST:-333}       # -> test_multistream_333.npz
ARMS=${ARMS:-"zerofill maskedvlos"}
OUT_ROOT=${OUT_ROOT:-outputs/missing_vlosexperiments}

for ARM in ${ARMS}; do
  case "${ARM}" in
    zerofill)   MODEL=stream_fusion_model5 ;;
    maskedvlos) MODEL=stream_fusion_model5_maskedvlos ;;
    *) echo "unknown arm '${ARM}' (expected zerofill|maskedvlos)"; exit 1 ;;
  esac
  RUNS_DIR=${OUT_ROOT}/${ARM}_model5
  MODEL_DIR=${RUNS_DIR}/train
  EVAL_DIR=${RUNS_DIR}/eval_sim_${N_TEST}
  REAL_DIR=${RUNS_DIR}/eval_real

  echo "=== [${ARM} 1/3] TRAIN  -> ${MODEL_DIR} ==="
  uv run python scripts/train.py \
    simulator=stream_agama_rnbody_huang model="${MODEL}" composition=global \
    adapter=stream preprocessing=stream_global_log10 augmentation=stream_global \
    augmentation.params.vlos_impute=zero \
    data.data_dir="${DATA_DIR}" data.n_simulations="${N_TRAIN}" \
    training.n_epochs="${N_EPOCHS}" training.batch_size="${BATCH_SIZE}" seed="${SEED}" \
    augmentation.params.resources_dir="${RES}" \
    hydra.run.dir="${MODEL_DIR}"

  echo "=== [${ARM} 2/3] EVALUATE on simulated ${N_TEST}-group test set -> ${EVAL_DIR} ==="
  uv run python scripts/evaluate.py \
    simulator=stream_agama_rnbody_huang model="${MODEL}" composition=global \
    adapter=stream preprocessing=stream_global_log10 augmentation=stream_global \
    augmentation.params.vlos_impute=zero \
    eval=stream_compositional data.data_dir="${DATA_DIR}" data.n_simulations="${N_TEST}" \
    model_dir="${MODEL_DIR}" augmentation.params.resources_dir="${RES}" \
    hydra.run.dir="${EVAL_DIR}"

  echo "=== [${ARM} 3/3] EVALUATE REAL (Gaia Pal5/NGC3201/M68) -> ${REAL_DIR} ==="
  uv run python scripts/evaluate_real.py \
    simulator=stream_agama_rnbody_huang model="${MODEL}" composition=global \
    adapter=stream preprocessing=stream_real_global_log10 augmentation=stream_real_global \
    augmentation.params.vlos_impute=zero \
    data.real_data_path="${REAL}" \
    model_dir="${MODEL_DIR}" augmentation.params.resources_dir="${RES}" \
    eval.misspecification_reference="${EVAL_DIR}" \
    hydra.run.dir="${REAL_DIR}"
done

echo "=== DONE. Outputs under ${OUT_ROOT} ==="

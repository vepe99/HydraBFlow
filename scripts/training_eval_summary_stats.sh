#!/usr/bin/env bash
# Summary-statistics experiment on the restricted-N-body + Zhou∪Huang 60k set: how does the global
# posterior change when the network is fed hand-crafted per-stream summary statistics instead of
# (or alongside) the raw star cloud? The rotation curve is ALWAYS an observable, so the fusion
# network is always used.
#
#   hybrid   particles + summary statistics + rotation curve
#            -> model=stream_fusion_model5_sumstats_hybrid  adapter=stream_sumstats_hybrid
#   sumonly  summary statistics + rotation curve (no raw particles to the net)
#            -> model=stream_fusion_model5_sumstats_only     adapter=stream_sumstats_only
#
# Baseline A (particles + rotation curve, NOT run here) is the existing
# outputs/stream_agama_rnbody/stream_fusion_model5_rnbody_huang/ run. Same dataset / seed / epochs /
# batch as that run's config, and DEFAULT (mean) v_los imputation on the particle side, so the only
# change vs baseline A is the summary representation.
#
# Run both arms (auto-parallel on 2 GPUs):  bash scripts/training_eval_summary_stats.sh
# One arm / overrides:  ARMS=hybrid N_EPOCHS=2 N_TRAIN=2000 bash scripts/training_eval_summary_stats.sh

set -euo pipefail
cd "$(dirname "$0")/.."

# ---- knobs (override from the environment) -----------------------------------------------------
DATA_DIR=${DATA_DIR:-data/data_jarvis/data_agama_rnbody_huang_hydrabflow}
RES=${RES:-assets/gaia}
REAL=${REAL:-assets/gaia/gaia_observed_streams_6Dwitherrors_cutNGC3201.npz}
SEED=${SEED:-2026}
N_EPOCHS=${N_EPOCHS:-1000}
BATCH_SIZE=${BATCH_SIZE:-1024}
N_TRAIN=${N_TRAIN:-60000}   # -> training_data_60000.npz
N_TEST=${N_TEST:-333}       # -> test_multistream_333.npz
ARMS=${ARMS:-"hybrid sumonly"}
OUT_ROOT=${OUT_ROOT:-outputs/summary_stats_experiments}
GPU=${GPU:-auto}            # auto: pick free GPU(s) via autocvd; <idx>: pin; cpu: force CPU

# --- Resolve one free GPU per arm (org convention: autocvd for JAX GPU runs) --------------------
# With GPU=auto and >1 arm, grab as many free GPUs as arms so the arms run on distinct cards.
declare -a GPU_IDS=()
NARMS=$(echo ${ARMS} | wc -w)
if [ "${GPU}" = "cpu" ]; then
  :
elif [ "${GPU}" = "auto" ]; then
  # autocvd -n N prints e.g. `export CUDA_VISIBLE_DEVICES=3,5`; capture and split.
  eval "$(uv run autocvd -n ${NARMS})"
  IFS=',' read -ra GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-}"
else
  IFS=',' read -ra GPU_IDS <<< "${GPU}"
fi

run_arm() {
  local ARM="$1" GPU_ID="$2"
  case "${ARM}" in
    hybrid)  MODEL=stream_fusion_model5_sumstats_hybrid; ADAPTER=stream_sumstats_hybrid ;;
    sumonly) MODEL=stream_fusion_model5_sumstats_only;   ADAPTER=stream_sumstats_only ;;
    *) echo "unknown arm '${ARM}' (expected hybrid|sumonly)"; exit 1 ;;
  esac
  local RUNS_DIR=${OUT_ROOT}/${ARM}_model5
  local MODEL_DIR=${RUNS_DIR}/train
  local EVAL_DIR=${RUNS_DIR}/eval_sim_${N_TEST}
  local REAL_DIR=${RUNS_DIR}/eval_real

  # Per-arm GPU environment.
  if [ "${GPU}" = "cpu" ]; then
    export JAX_PLATFORMS=cpu
  else
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
    export XLA_PYTHON_CLIENT_PREALLOCATE=false   # coexist on a shared card
  fi
  echo "=== [${ARM}] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<cpu>} ==="

  echo "=== [${ARM} 1/3] TRAIN -> ${MODEL_DIR} ==="
  uv run python scripts/train.py \
    simulator=stream_agama_rnbody_huang model="${MODEL}" composition=global \
    adapter="${ADAPTER}" preprocessing=stream_global_log10_sumstats augmentation=stream_global_sumstats \
    data.data_dir="${DATA_DIR}" data.n_simulations="${N_TRAIN}" \
    training.n_epochs="${N_EPOCHS}" training.batch_size="${BATCH_SIZE}" seed="${SEED}" \
    augmentation.params.resources_dir="${RES}" \
    hydra.run.dir="${MODEL_DIR}"

  echo "=== [${ARM} 2/3] EVALUATE on simulated ${N_TEST}-group test set -> ${EVAL_DIR} ==="
  uv run python scripts/evaluate.py \
    simulator=stream_agama_rnbody_huang model="${MODEL}" composition=global \
    adapter="${ADAPTER}" preprocessing=stream_global_log10_sumstats augmentation=stream_global_sumstats \
    eval=stream_compositional data.data_dir="${DATA_DIR}" data.n_simulations="${N_TEST}" \
    model_dir="${MODEL_DIR}" augmentation.params.resources_dir="${RES}" \
    hydra.run.dir="${EVAL_DIR}"

  echo "=== [${ARM} 3/3] EVALUATE REAL (Gaia Pal5/NGC3201/M68) -> ${REAL_DIR} ==="
  uv run python scripts/evaluate_real.py \
    simulator=stream_agama_rnbody_huang model="${MODEL}" composition=global \
    adapter="${ADAPTER}" preprocessing=stream_real_global_log10 augmentation=stream_real_global_sumstats \
    data.real_data_path="${REAL}" \
    model_dir="${MODEL_DIR}" augmentation.params.resources_dir="${RES}" \
    eval.misspecification_reference="${EVAL_DIR}" \
    hydra.run.dir="${REAL_DIR}"

  # Standalone sim-vs-real summary-track diagnostic (best-effort).
  uv run python scripts/ppc_summary_statistics.py \
    --sim "${DATA_DIR}/test_multistream_${N_TEST}.npz" --real "${REAL}" \
    --out "${RUNS_DIR}/ppc_summary_statistics.png" || true
}

# --- Launch arms: parallel on distinct GPUs when available, else sequential ----------------------
mkdir -p "${OUT_ROOT}"
i=0
pids=()
for ARM in ${ARMS}; do
  GPU_ID="${GPU_IDS[$i]:-${GPU_IDS[0]:-}}"
  if [ "${#GPU_IDS[@]}" -ge "${NARMS}" ] && [ "${GPU}" != "cpu" ]; then
    ( run_arm "${ARM}" "${GPU_ID}" ) > "${OUT_ROOT}/${ARM}.log" 2>&1 &
    pids+=($!)
    echo "launched arm '${ARM}' on GPU ${GPU_ID} (pid $!), logging to ${OUT_ROOT}/${ARM}.log"
  else
    run_arm "${ARM}" "${GPU_ID}"   # sequential (shared/one GPU or CPU)
  fi
  i=$((i+1))
done
for pid in "${pids[@]:-}"; do [ -n "${pid}" ] && wait "${pid}"; done

echo "=== DONE. Outputs under ${OUT_ROOT} ==="

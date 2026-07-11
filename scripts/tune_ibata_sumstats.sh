#!/usr/bin/env bash
# Launch ONE Optuna tuning process for the Ibata sumstats fusion model, pinned to a single GPU.
# The study is a concurrency-safe Optuna JournalStorage `.log` (at ${DATA_DIR}/tuning/
# stream_ibata_sumstats_study.log), so this script can be launched multiple times — once per free
# GPU, and again on a GPU freed by the training test — and every process cooperatively fills the
# SAME study. Search space (conf/tuning/stream_ibata_sumstats.yaml) includes the sim_summary
# backbone TYPE (mlp vs feature_transformer / TimeSeriesTransformer) plus the vector-observable
# backbones, the fusion head, and the diffusion subnet.
#
# Usage:  GPU=<idx> bash scripts/tune_ibata_sumstats.sh        # pin a specific card
#         bash scripts/tune_ibata_sumstats.sh                  # GPU=auto -> autocvd picks one free
# Knobs:  N_TRIALS (per process, default 50), N_EPOCHS (per trial, default 100)
# All processes append their stdout to ${OUT_ROOT}/tune_ibata_sumstats.log (the user's single log).

set -euo pipefail
cd "$(dirname "$0")/.."

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

SIM=${SIM:-stream_agama_ibata}
DATA_DIR=${DATA_DIR:-data/data_jarvis/data_agama_ibata_hydrabflow}
RES=${RES:-assets/gaia}
N_TRAIN=${N_TRAIN:-100000}
N_TRIALS=${N_TRIALS:-50}
N_EPOCHS=${N_EPOCHS:-100}
# Posterior sampling on the val split is batched as (INFER_BATCH * INFER_SAMPLES) rows through the
# diffusion integrator. The search space draws large inference nets (mlp_width up to 256, depth up
# to 8), so the default 256*1000 = 256k-row batch OOMs a 40 GB card. Keep the product small
# (32*500 = 16k) — plenty for RMSE + calibration scoring, and sampling is a negligible fraction of
# per-trial wall-clock vs the 100-epoch training.
INFER_BATCH=${INFER_BATCH:-32}
INFER_SAMPLES=${INFER_SAMPLES:-500}
MODEL=${MODEL:-stream_fusion_ibata_sumstats}
ADAPTER=${ADAPTER:-stream_ibata_sumstats}
AUG=${AUG:-stream_global_ibata_sumstats}
PREPROC=${PREPROC:-stream_global_log10_ibata_sumstats}
OUT_ROOT=${OUT_ROOT:-outputs/ibata_sumstats/tuning}
LOG=${LOG:-${OUT_ROOT}/tune_ibata_sumstats.log}
mkdir -p "${OUT_ROOT}"

echo "=== [tune] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<cpu>}  trials=${N_TRIALS} epochs=${N_EPOCHS} ===" | tee -a "${LOG}"
uv run python scripts/tune.py \
  simulator="${SIM}" model="${MODEL}" composition=global \
  adapter="${ADAPTER}" preprocessing="${PREPROC}" augmentation="${AUG}" \
  tuning=stream_ibata_sumstats \
  data.data_dir="${DATA_DIR}" data.n_simulations="${N_TRAIN}" \
  tuning.n_trials="${N_TRIALS}" tuning.n_epochs="${N_EPOCHS}" \
  inference.batch_size="${INFER_BATCH}" inference.num_samples="${INFER_SAMPLES}" \
  augmentation.params.resources_dir="${RES}" 2>&1 | tee -a "${LOG}"

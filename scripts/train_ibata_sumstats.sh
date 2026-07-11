#!/usr/bin/env bash
# Ibata sumstats fusion model (norho, sigma-condition) — the "TST test": train + evaluate (sim) +
# evaluate_real (Gaia), on the Ibata 10^5-stream dataset. The network sees the binned per-stream
# summary statistics (sim_summary, via the feature_transformer / TimeSeriesTransformer) + the
# rotation curve (vcirc_kms) + the HI terminal-velocity curve (vterm_kms) as fusion backbones, with
# the scalar surface density sigma_z (+ stream index j) as inference conditions. Raw star particles
# are NOT fed to the net; rho_z is excluded (no real datum). Unlike train_ibata.sh, real-data eval
# IS wired here (attach_observed_vterm/sigma_z make the ancillary observables available for Gaia).
#
#   [1/3] train        -> ${MODEL_DIR}
#   [2/3] evaluate sim -> ${EVAL_DIR}   (333-group multistream test set)
#   [3/3] evaluate real-> ${REAL_DIR}   (Gaia Pal5/NGC3201/M68)
#
# Run:            bash scripts/train_ibata_sumstats.sh
# Smoke:          N_EPOCHS=2 N_TRAIN=2000 N_TEST=8 bash scripts/train_ibata_sumstats.sh
# Override GPU:   GPU=3 bash scripts/train_ibata_sumstats.sh     (GPU=cpu forces CPU; default: autocvd)

set -euo pipefail
cd "$(dirname "$0")/.."

# --- GPU selection: autocvd picks a free card (org convention); GPU=<idx>|cpu overrides ---------
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
DATA_DIR=${DATA_DIR:-data/data_jarvis/data_agama_ibata_hydrabflow}
RES=${RES:-assets/gaia}
REAL=${REAL:-assets/gaia/gaia_observed_streams_6Dwitherrors_cutNGC3201.npz}
SEED=${SEED:-2026}
N_EPOCHS=${N_EPOCHS:-300}
BATCH_SIZE=${BATCH_SIZE:-1024}
N_TRAIN=${N_TRAIN:-100000}   # -> training_data_100000.npz
N_TEST=${N_TEST:-333}        # -> test_multistream_333.npz

MODEL=${MODEL:-stream_fusion_ibata_sumstats}
ADAPTER=${ADAPTER:-stream_ibata_sumstats}
AUG=${AUG:-stream_global_ibata_sumstats}
PREPROC=${PREPROC:-stream_global_log10_ibata_sumstats}
REAL_AUG=${REAL_AUG:-stream_real_global_ibata_sumstats}
REAL_PREPROC=${REAL_PREPROC:-stream_real_global_ibata_sumstats}

RUNS_DIR=${RUNS_DIR:-outputs/ibata_sumstats/tst_test}
MODEL_DIR=${MODEL_DIR:-${RUNS_DIR}/train}
EVAL_DIR=${EVAL_DIR:-${RUNS_DIR}/eval_sim_${N_TEST}}
REAL_DIR=${REAL_DIR:-${RUNS_DIR}/eval_real}

echo "=== [1/3] TRAIN  -> ${MODEL_DIR} ==="
uv run python scripts/train.py \
  simulator="${SIM}" model="${MODEL}" composition=global \
  adapter="${ADAPTER}" preprocessing="${PREPROC}" augmentation="${AUG}" \
  data.data_dir="${DATA_DIR}" data.n_simulations="${N_TRAIN}" \
  training.n_epochs="${N_EPOCHS}" training.batch_size="${BATCH_SIZE}" seed="${SEED}" \
  augmentation.params.resources_dir="${RES}" \
  hydra.run.dir="${MODEL_DIR}"

echo "=== [2/3] EVALUATE sim ${N_TEST}-group multistream -> ${EVAL_DIR} ==="
uv run python scripts/evaluate.py \
  simulator="${SIM}" model="${MODEL}" composition=global \
  adapter="${ADAPTER}" preprocessing="${PREPROC}" augmentation="${AUG}" \
  eval=stream_compositional data.data_dir="${DATA_DIR}" data.n_simulations="${N_TEST}" \
  eval.test_dataset_name=test_multistream_${N_TEST}.npz \
  model_dir="${MODEL_DIR}" augmentation.params.resources_dir="${RES}" \
  hydra.run.dir="${EVAL_DIR}"

echo "=== [3/3] EVALUATE REAL (Gaia Pal5/NGC3201/M68) -> ${REAL_DIR} ==="
uv run python scripts/evaluate_real.py \
  simulator="${SIM}" model="${MODEL}" composition=global \
  adapter="${ADAPTER}" preprocessing="${REAL_PREPROC}" augmentation="${REAL_AUG}" \
  data.real_data_path="${REAL}" \
  model_dir="${MODEL_DIR}" augmentation.params.resources_dir="${RES}" \
  eval.misspecification_reference="${EVAL_DIR}" \
  hydra.run.dir="${REAL_DIR}"

echo "=== DONE. Outputs under ${RUNS_DIR} ==="

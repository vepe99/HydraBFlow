#!/usr/bin/env bash
# LOCAL-level twin of train_v4_2modal.sh: infer each stream's local parameters
# [m_progenitor, t_end, vr, r, mu_ra_cosdec, mu_dec] conditioned on the stream summary (B-spline /
# grid) + the rotation curve (fusion backbones, always observed) + the TRUE global parameters
# (inference_conditions, standardized by BayesFlow). Locals are z-scored PER STREAM with each
# stream's own train-split mean/std (preprocessing=stream_local_*; state saved, inverted at eval).
#
#   [1/3] train         -> ${MODEL_DIR}
#   [2/3] evaluate sim  -> ${EVAL_DIR}   (333-group multistream test set, conditioned on the true
#                                         globals; per-stream Pal5_/NGC3201_/M68_ recovery,
#                                         calibration_ecdf, coverage, metrics in physical units)
#   [3/3] evaluate real -> ${REAL_DIR}   ONLY if GLOBAL_RUN_DIR is set: ancestral sampling with the
#                                         globals drawn from that run's posterior.npz
#
# Run:    bash scripts/train_local_2modal.sh
# Smoke:  N_EPOCHS=1 BATCH_SIZE=64 GPU=cpu RUNS_DIR=/tmp/localsmoke bash scripts/train_local_2modal.sh

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

SIM=${SIM:-stream_agama_spray_massloss_ibata_m200c_v4}
DATA_DIR=${DATA_DIR:-data/data_jarvis/data_agama_spray_massloss_ibata_m200c_v4_p1e3_hydrabflow}
RES=${RES:-assets/gaia}
REAL=${REAL:-assets/gaia/gaia_observed_streams_6Dwitherrors_cutNGC3201.npz}
SEED=${SEED:-2026}
N_EPOCHS=${N_EPOCHS:-300}
BATCH_SIZE=${BATCH_SIZE:-2048}
N_TRAIN=${N_TRAIN:-300000}
N_TEST=${N_TEST:-333}

MODEL=${MODEL:-stream_fusion_2modal_oldgrid_local}
ADAPTER=${ADAPTER:-stream_2modal_local}
AUG=${AUG:-stream_global_streamfinder_bspline_smoothing}
REAL_AUG=${REAL_AUG:-stream_real_global_streamfinder_bspline_smoothing}
PREPROC=${PREPROC:-stream_local_log10_sumstats_2modal}
REAL_PREPROC=${REAL_PREPROC:-stream_real_local_log10}
STANDARDIZE=${STANDARDIZE:-[inference_variables,inference_conditions,summary_variables]}
GLOBAL_RUN_DIR=${GLOBAL_RUN_DIR:-}   # e.g. outputs/Bsline/spray_p1e3_v4_smoothing_2modal/eval_real_5k
EXTRA=${EXTRA:-}

RUNS_DIR=${RUNS_DIR:-outputs/Bsline/spray_p1e3_v4_smoothing_local}
MODEL_DIR=${MODEL_DIR:-${RUNS_DIR}/train}
EVAL_DIR=${EVAL_DIR:-${RUNS_DIR}/eval_sim_${N_TEST}}
REAL_DIR=${REAL_DIR:-${RUNS_DIR}/eval_real}

COMMON="simulator=${SIM} model=${MODEL} composition=local adapter=${ADAPTER} augmentation.params.resources_dir=${RES}"

echo "=== [1/3] TRAIN (local level) -> ${MODEL_DIR} ==="
${PY:-uv run python} -m hydrabflow.pipeline.train ${COMMON} \
  preprocessing="${PREPROC}" augmentation="${AUG}" \
  data.data_dir="${DATA_DIR}" data.n_simulations="${N_TRAIN}" \
  training.n_epochs="${N_EPOCHS}" training.batch_size="${BATCH_SIZE}" \
  "training.standardize=${STANDARDIZE}" seed="${SEED}" \
  hydra.run.dir="${MODEL_DIR}" ${EXTRA}

echo "=== [2/3] EVALUATE sim ${N_TEST}-group multistream (true globals) -> ${EVAL_DIR} ==="
${PY:-uv run python} -m hydrabflow.pipeline.evaluate ${COMMON} \
  preprocessing="${PREPROC}" augmentation="${AUG}" \
  data.data_dir="${DATA_DIR}" data.n_simulations="${N_TEST}" \
  eval.test_dataset_name=test_multistream_${N_TEST}.npz \
  "training.standardize=${STANDARDIZE}" model_dir="${MODEL_DIR}" \
  hydra.run.dir="${EVAL_DIR}" ${EXTRA}

if [ -n "${GLOBAL_RUN_DIR}" ]; then
  echo "=== [3/3] EVALUATE REAL (ancestral: globals from ${GLOBAL_RUN_DIR}) -> ${REAL_DIR} ==="
  ${PY:-uv run python} -m hydrabflow.pipeline.evaluate ${COMMON} \
    preprocessing="${REAL_PREPROC}" augmentation="${REAL_AUG}" \
    data.real_data_path="${REAL}" composition.global_run_dir="${GLOBAL_RUN_DIR}" \
    "training.standardize=${STANDARDIZE}" model_dir="${MODEL_DIR}" \
    hydra.run.dir="${REAL_DIR}" ${EXTRA}
else
  echo "=== [3/3] skipped (set GLOBAL_RUN_DIR=<global eval_real dir> for ancestral real-data sampling) ==="
fi
echo "=== DONE. Outputs under ${RUNS_DIR} ==="

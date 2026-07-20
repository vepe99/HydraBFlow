#!/usr/bin/env bash
# MEDIANS-ONLY Optuna tuning for the Ibata gridded-summary fusion model on the m200_c dataset.
#
# Identical to scripts/tune_ibata_m200c.sh (summary-stats stack: sim_summary gridded
# TimeSeriesTransformer + vcirc_kms + vterm_kms backbones, sigma_z + j conditions, NO raw-particle
# SetTransformer) EXCEPT the gridded summary drops every per-bin std channel
# (augmentation=stream_global_ibata_grid_median, summary_include_std=false -> (n, K, 7) medians-only
# layout). Motivation: the 2026-07-15 misspecification localization found the real-vs-sim MMD flag
# lives entirely in the dispersion statistics (std_phi2 dominant); this arm tests whether removing
# them from the network input clears the flag / changes the real-data posterior.
#
# Runs under its OWN study name (stream_ibata_grid_m200c_median_study) so it never collides with
# the earlier m200_c study (stream_ibata_grid_study.log in the same storage dir).
#
# After the study + the delegated runner's milestone evals (evaluate + evaluate_real with the
# pooled MMD misspecification test) complete, this wrapper additionally runs the FULL
# misspecification pipeline on the best Pareto-front trial (lowest validation RMSE — always
# Pareto-optimal under the (RMSE, calibration) objectives):
#   - scripts/misspecification_per_channel.py  (per-channel MMD: which observable carries the flag)
#   - scripts/sumstat_sim_vs_real.py           (per-statistic/bin z-scores; layout-aware, handles
#                                               the 7-channel medians-only grid)
#
# Run:      bash scripts/tune_ibata_m200c_median.sh
# Smoke:    N_EPOCHS=1 N_TRAIN=2000 N_TEST=8 N_TRIALS_TOTAL=1 GPU=0 bash scripts/tune_ibata_m200c_median.sh
# Pin GPUs: GPU="0 1 3" bash scripts/tune_ibata_m200c_median.sh
set -euo pipefail
cd "$(dirname "$0")/.."

export SIM=${SIM:-stream_agama_ibata_onedisk_beta3_m200c}
export DATA_DIR=${DATA_DIR:-data/data_jarvis/data_agama_ibata_onedisk_beta3_m200c_hydrabflow}
export RUNS_DIR=${RUNS_DIR:-outputs/ibata_onedisk_grid_m200c_median/tuning}
export STUDY=${STUDY:-stream_ibata_grid_m200c_median_study}
export AUG=${AUG:-stream_global_ibata_grid_median}
export REAL_AUG=${REAL_AUG:-stream_real_global_ibata_grid_median}
export N_TRAIN=${N_TRAIN:-300000}          # the m200_c dataset is training_data_100000.npz
export N_TRIALS_TOTAL=${N_TRIALS_TOTAL:-30}

echo "=== tune_ibata_m200c_median: MEDIANS-ONLY grid on ${SIM} (data: ${DATA_DIR}) ==="
echo "=== study: ${STUDY} | ${N_TRIALS_TOTAL} trials | runs dir: ${RUNS_DIR} ==="
bash scripts/tune_ibata_onedisk_grid.sh "$@"

# ---- POST: full misspecification pipeline on the best Pareto-front trial -------------------------
STORAGE_DIR=${STORAGE_DIR:-${DATA_DIR}/tuning}
STUDY_LOG=${STORAGE_DIR}/${STUDY}.log
ARTIFACTS=${ARTIFACTS:-${STORAGE_DIR}/${STUDY}}
EVALS_DIR=${RUNS_DIR}/evals

BEST=$(uv run python scripts/_ibata_grid_select.py \
  --storage-log "${STUDY_LOG}" --study-name "${STUDY}" --artifacts-dir "${ARTIFACTS}" \
  --total "${N_TRIALS_TOTAL}" --top-k 1 --milestone full)
BEST=$(echo "${BEST}" | awk '{print $1}')
if [ -z "${BEST}" ]; then
  echo "ERROR: could not determine the best trial from ${STUDY_LOG}" >&2
  exit 1
fi
SIM_RUN="${EVALS_DIR}/trial_${BEST}/eval_sim"
REAL_RUN="${EVALS_DIR}/trial_${BEST}/eval_real"
echo "=== [POST] Misspecification pipeline on best Pareto-front trial ${BEST} ==="
echo "    sim-run:  ${SIM_RUN}"
echo "    real-run: ${REAL_RUN}"
if [ ! -e "${REAL_RUN}/posterior.npz" ]; then
  echo "ERROR: real eval for trial ${BEST} missing (${REAL_RUN}); cannot run the pipeline." >&2
  exit 1
fi

# org convention: pick a free GPU via autocvd for the JAX-backed offline scripts
if [ "${GPU:-auto}" = "cpu" ]; then
  export JAX_PLATFORMS=cpu
else
  MISSPEC_GPU=$(uv run autocvd -n 1 -o -q -l 2>/dev/null || true)
  [ -n "${MISSPEC_GPU}" ] && { export CUDA_VISIBLE_DEVICES="${MISSPEC_GPU}"; export XLA_PYTHON_CLIENT_PREALLOCATE=false; }
fi
set +e
uv run python scripts/misspecification_per_channel.py \
  --sim-run "${SIM_RUN}" --real-run "${REAL_RUN}" \
  || echo "WARN: misspecification_per_channel failed for trial ${BEST}."
uv run python scripts/sumstat_sim_vs_real.py \
  --sim-run "${SIM_RUN}" --real-run "${REAL_RUN}" \
  || echo "WARN: sumstat_sim_vs_real failed for trial ${BEST}."
set -e

echo "=== DONE (medians-only study) ==="
echo "  Best Pareto trial            : ${BEST}"
echo "  Pooled MMD test              : ${REAL_RUN}/misspecification.json"
echo "  Per-channel MMD              : ${REAL_RUN}/misspecification_per_channel.{json,png}"
echo "  Per-statistic z-maps         : ${REAL_RUN}/sumstat_sim_vs_real_{zmap,tracks}.png"

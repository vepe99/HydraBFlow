#!/usr/bin/env bash
# Hyperparameter tuning for the Ibata GRIDDED-summary fusion model on the SINGLE-DISK dataset,
# reusing the exact architecture of scripts/train_ibata_onedisk_grid.sh (model=stream_fusion_ibata_grid,
# adapter=stream_ibata_sumstats, augmentation=stream_global_ibata_grid, preprocessing=
# stream_global_log10_ibata_sumstats): sim_summary as a single φ1-gridded TimeSeriesTransformer +
# vcirc_kms + vterm_kms TimeSeriesTransformers, sigma_z + j as inference conditions.
#
# Design (50 trials × 300 epochs, all free GPUs):
#   [1] TUNE   One Optuna worker per free GPU, all sharing ONE JournalStorage study
#              (conf/tuning/stream_ibata_grid.yaml). Together they run N_TRIALS_TOTAL trials, each a
#              FULL 300-epoch train. Every trial's checkpoint + val diagnostics are kept under the
#              study's artifacts dir (${ARTIFACTS}/trials/trial_XXXX) — inspect ANY trial, not just
#              the best.
#   [2] EVAL   For the union of the best-3 trials at the 1/3, 1/2 and full milestones, run the 333-
#              group sim evaluate + the Gaia evaluate_real (each trial's sampled hyperparameters are
#              replayed as Hydra overrides so the saved weights load into the matching architecture).
#              Saved under ${EVALS_DIR}/trial_<n>/{eval_sim,eval_real} for inspection.
#   [3] REPORT At each milestone (1/3, 1/2, end) reconstruct the best-3-as-of-then (by trial-
#              completion order) and run the cross-model posterior-tension report over their Gaia
#              posteriors -> is the top-3 COHERENT on the real data? Written to
#              ${REPORTS_DIR}/<milestone>_coherence.txt.
#
# Milestones are reconstructed AFTER tuning from Optuna's per-trial completion timestamps (the "best
# 3 as of 1/3 of the run" = best 3 among the first ceil(N/3) trials to complete), so the concurrent
# workers never pause and the GPUs stay saturated.
#
# Run:          bash scripts/tune_ibata_onedisk_grid.sh
# Smoke:        N_EPOCHS=2 N_TRAIN=2000 N_TEST=8 N_TRIALS_TOTAL=4 bash scripts/tune_ibata_onedisk_grid.sh
# Pin GPUs:     GPU="0 3 5" bash scripts/tune_ibata_onedisk_grid.sh   (space-separated ids; default: all free)
#               GPU=cpu forces a single CPU worker.
# Std sigma_z:  STANDARDIZE_SIGMA_Z=1 bash scripts/tune_ibata_onedisk_grid.sh  (z-scores the sigma_z
#               inference condition only, across trials + evals; j stays raw. Off by default.)

set -euo pipefail
cd "$(dirname "$0")/.."

# ---- knobs (mirror train_ibata_onedisk_grid.sh) ------------------------------------------------
SIM=${SIM:-stream_agama_ibata_onedisk}
DATA_DIR=${DATA_DIR:-data/data_jarvis/data_agama_ibata_onedisk_hydrabflow}
RES=${RES:-assets/gaia}
REAL=${REAL:-assets/gaia/gaia_observed_streams_6Dwitherrors_cutNGC3201.npz}
SEED=${SEED:-2026}
N_EPOCHS=${N_EPOCHS:-300}
BATCH_SIZE=${BATCH_SIZE:-1024}
N_TRAIN=${N_TRAIN:-100000}        # -> training_data_100000.npz
N_TEST=${N_TEST:-333}             # -> test_multistream_333.npz
N_TRIALS_TOTAL=${N_TRIALS_TOTAL:-50}
TOP_K=${TOP_K:-3}

MODEL=${MODEL:-stream_fusion_ibata_grid}
ADAPTER=${ADAPTER:-stream_ibata_sumstats}
AUG=${AUG:-stream_global_ibata_grid}
REAL_AUG=${REAL_AUG:-stream_real_global_ibata_grid}
TUNING=${TUNING:-stream_ibata_grid}

# STANDARDIZE_SIGMA_Z=1 z-scores ONLY the sigma_z inference condition (in preprocessing, so j stays
# raw). Applies to BOTH the tuning trials and the milestone sim/Gaia evals (they share this PREPROC
# and the one fit-once preprocessing_state.npz). Off by default; explicit PREPROC/REAL_PREPROC win.
case "${STANDARDIZE_SIGMA_Z:-0}" in 1|true|yes|on) SIGMASTD=1;; *) SIGMASTD=0;; esac
if [ "${SIGMASTD}" = 1 ]; then
  PREPROC=${PREPROC:-stream_global_log10_ibata_sumstats_sigmastd}
  REAL_PREPROC=${REAL_PREPROC:-stream_real_global_ibata_sumstats_sigmastd}
  echo "sigma_z standardization: ON (preproc=${PREPROC}, real=${REAL_PREPROC})"
else
  PREPROC=${PREPROC:-stream_global_log10_ibata_sumstats}
  REAL_PREPROC=${REAL_PREPROC:-stream_real_global_ibata_sumstats}
fi

# Posterior sampling on the val split is (INFER_BATCH * INFER_SAMPLES) rows through the diffusion
# integrator; the search space draws large nets, so keep the product small (16k) to avoid OOM
# (tune.py also has OOM-backoff). Only affects per-trial scoring, not the milestone evals.
INFER_BATCH=${INFER_BATCH:-32}
INFER_SAMPLES=${INFER_SAMPLES:-500}

# Study + artifact locations (must match conf/tuning/stream_ibata_grid.yaml resolution).
STUDY=${STUDY:-stream_ibata_grid_study}
STORAGE_DIR=${STORAGE_DIR:-${DATA_DIR}/tuning}
STUDY_LOG=${STORAGE_DIR}/${STUDY}.log
ARTIFACTS=${ARTIFACTS:-${STORAGE_DIR}/${STUDY}}
PREPROC_STATE=${ARTIFACTS}/preprocessing_state.npz

RUNS_DIR=${RUNS_DIR:-outputs/ibata_onedisk_grid/tuning}
TUNE_LOG_DIR=${RUNS_DIR}/workers
EVALS_DIR=${RUNS_DIR}/evals
REPORTS_DIR=${RUNS_DIR}/milestones
mkdir -p "${TUNE_LOG_DIR}" "${EVALS_DIR}" "${REPORTS_DIR}"

# ---- GPU discovery: one tuning worker per free card (org convention: autocvd) -------------------
GPU=${GPU:-auto}
GPU_LIST=()
if [ "${GPU}" = "cpu" ]; then
  GPU_LIST=(cpu)
elif [ "${GPU}" != "auto" ]; then
  read -ra GPU_LIST <<< "${GPU}"                     # explicit space-separated ids
else
  # Count currently-free cards (low memory + low util), then let autocvd pick that many free GPUs.
  NFREE=1
  if command -v nvidia-smi >/dev/null 2>&1; then
    NFREE=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits \
      | awk -F',' '{gsub(/ /,""); if ($1+0 < 1000 && $2+0 < 10) c++} END{print (c>0?c:1)}')
  fi
  echo "Detected ${NFREE} free GPU(s); requesting them via autocvd."
  IDS=$(uv run autocvd -n "${NFREE}" -o -q)          # autocvd selects the least-contended free set
  IFS=',' read -ra GPU_LIST <<< "${IDS}"
fi
W=${#GPU_LIST[@]}
PER=$(( (N_TRIALS_TOTAL + W - 1) / W ))               # ceil: workers overshoot slightly, harmless
echo "=== Tuning on ${W} worker(s) [${GPU_LIST[*]}], ~${PER} trials each -> ${N_TRIALS_TOTAL} total, ${N_EPOCHS} epochs/trial ==="

# ---- [1] TUNE: launch one worker per GPU, all sharing the same study ----------------------------
run_worker() {  # $1 = gpu id (or "cpu"), $2 = worker index
  local gpu="$1" idx="$2"
  if [ "${gpu}" = "cpu" ]; then
    export JAX_PLATFORMS=cpu
  else
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
  fi
  echo "[worker ${idx}] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<cpu>}  trials=${PER}"
  uv run python scripts/tune.py \
    simulator="${SIM}" model="${MODEL}" composition=global \
    adapter="${ADAPTER}" preprocessing="${PREPROC}" augmentation="${AUG}" \
    tuning="${TUNING}" \
    data.data_dir="${DATA_DIR}" data.n_simulations="${N_TRAIN}" \
    tuning.n_trials="${PER}" tuning.n_epochs="${N_EPOCHS}" \
    training.batch_size="${BATCH_SIZE}" seed="${SEED}" \
    inference.batch_size="${INFER_BATCH}" inference.num_samples="${INFER_SAMPLES}" \
    augmentation.params.resources_dir="${RES}"
}

PIDS=()
i=0
for gpu in "${GPU_LIST[@]}"; do
  ( run_worker "${gpu}" "${i}" ) > "${TUNE_LOG_DIR}/worker_${i}_gpu${gpu}.log" 2>&1 &
  PIDS+=("$!")
  i=$((i + 1))
done
echo "Launched ${#PIDS[@]} tuning worker(s); logs under ${TUNE_LOG_DIR}/ . Waiting..."

set +e
FAIL=0
for pid in "${PIDS[@]}"; do
  wait "${pid}" || { echo "WARN: tuning worker PID ${pid} exited non-zero."; FAIL=1; }
done
set -e
echo "=== Tuning phase done (worker failures: ${FAIL}). Study: ${STUDY_LOG} ==="

if [ ! -f "${STUDY_LOG}" ]; then
  echo "ERROR: no study log at ${STUDY_LOG}; nothing to evaluate." >&2
  exit 1
fi

SELECT="uv run python scripts/_ibata_grid_select.py \
  --storage-log ${STUDY_LOG} --study-name ${STUDY} --artifacts-dir ${ARTIFACTS} \
  --total ${N_TRIALS_TOTAL} --top-k ${TOP_K}"

# ---- [2] EVAL the union of milestone best-3 trials (sim + Gaia), best-effort ---------------------
# The trained checkpoints live in the study artifacts dir; give each a preprocessing_state.npz
# (shared, fit once) so it is a valid model_dir, then replay the trial's hyperparameters as overrides.
EVAL_GPU=""
if [ "${GPU}" != "cpu" ]; then
  EVAL_GPU=$(uv run autocvd -n 1 -o -q -l 2>/dev/null || true)   # -l: least-used, don't block
fi
export_eval_gpu() {
  if [ "${GPU}" = "cpu" ]; then export JAX_PLATFORMS=cpu;
  elif [ -n "${EVAL_GPU}" ]; then export CUDA_VISIBLE_DEVICES="${EVAL_GPU}"; export XLA_PYTHON_CLIENT_PREALLOCATE=false; fi
}

echo "=== [2] Evaluating milestone best-${TOP_K} trials (eval GPU: ${EVAL_GPU:-<cpu>}) ==="
set +e
while IFS=$'\t' read -r NUM ADIR OVERRIDES; do
  [ -z "${NUM}" ] && continue
  echo "--- trial ${NUM}  (${ADIR}) ---"
  if [ ! -e "${ADIR}/approximator.keras" ]; then
    echo "WARN: no checkpoint for trial ${NUM} at ${ADIR}; skipping."; continue
  fi
  if [ ! -e "${ADIR}/preprocessing_state.npz" ]; then
    ln -sf "$(realpath "${PREPROC_STATE}")" "${ADIR}/preprocessing_state.npz"
  fi
  TEDIR="${EVALS_DIR}/trial_${NUM}"
  export_eval_gpu
  echo "    [sim]  -> ${TEDIR}/eval_sim"
  uv run python scripts/evaluate.py \
    simulator="${SIM}" model="${MODEL}" composition=global \
    adapter="${ADAPTER}" preprocessing="${PREPROC}" augmentation="${AUG}" \
    eval=stream_compositional data.data_dir="${DATA_DIR}" data.n_simulations="${N_TEST}" \
    eval.test_dataset_name="test_multistream_${N_TEST}.npz" \
    model_dir="${ADIR}" augmentation.params.resources_dir="${RES}" \
    ${OVERRIDES} hydra.run.dir="${TEDIR}/eval_sim" \
    || echo "WARN: sim eval failed for trial ${NUM}."
  echo "    [real] -> ${TEDIR}/eval_real"
  uv run python scripts/evaluate_real.py \
    simulator="${SIM}" model="${MODEL}" composition=global \
    adapter="${ADAPTER}" preprocessing="${REAL_PREPROC}" augmentation="${REAL_AUG}" \
    data.real_data_path="${REAL}" \
    model_dir="${ADIR}" augmentation.params.resources_dir="${RES}" \
    eval.misspecification_reference="${TEDIR}/eval_sim" \
    ${OVERRIDES} hydra.run.dir="${TEDIR}/eval_real" \
    || echo "WARN: real eval failed for trial ${NUM}."
done < <(${SELECT} --list-trials)

# ---- [3] REPORT coherence of the best-3 on the real Gaia data, per milestone --------------------
echo "=== [3] Milestone coherence reports -> ${REPORTS_DIR} ==="
for M in third half full; do
  NUMS=$(${SELECT} --milestone "${M}")
  [ -z "${NUMS}" ] && { echo "milestone ${M}: no trials"; continue; }
  OUT="${REPORTS_DIR}/${M}_coherence.txt"
  {
    echo "### Milestone '${M}' — best-${TOP_K} trials: ${NUMS}"
    echo "### (cross-model posterior tension on the real Gaia data; z<~1 => coherent)"
    echo
  } > "${OUT}"
  ARGS=()
  for N in ${NUMS}; do
    RD="${EVALS_DIR}/trial_${N}/eval_real"
    [ -e "${RD}/posterior.npz" ] && ARGS+=("trial${N}=${RD}")
    # symlink the checkpoint + evals under the milestone dir for easy inspection
    mkdir -p "${REPORTS_DIR}/${M}"
    ln -sfn "$(realpath "${ARTIFACTS}/trials/trial_$(printf '%04d' "${N}")")" "${REPORTS_DIR}/${M}/trial_${N}_checkpoint" 2>/dev/null || true
    ln -sfn "$(realpath "${EVALS_DIR}/trial_${N}")" "${REPORTS_DIR}/${M}/trial_${N}_evals" 2>/dev/null || true
  done
  if [ "${#ARGS[@]}" -ge 2 ]; then
    uv run python scripts/report_cross_model_tension.py "${ARGS[@]}" >> "${OUT}" 2>&1 \
      || echo "WARN: tension report failed for milestone ${M}."
  else
    echo "(need >=2 real-data posteriors for a tension report; have ${#ARGS[@]})" >> "${OUT}"
  fi
  echo "milestone ${M}: ${OUT}"
done
set -e

echo "=== DONE ==="
echo "  Study + all trial checkpoints : ${ARTIFACTS}/trials/"
echo "  Best-3 sim+Gaia evals         : ${EVALS_DIR}/trial_<n>/{eval_sim,eval_real}"
echo "  Milestone coherence reports   : ${REPORTS_DIR}/{third,half,full}_coherence.txt"

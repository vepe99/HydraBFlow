#!/usr/bin/env bash
# Optuna study of the LOCAL B-spline model: per-stream [m_progenitor, t_end, vr, r, mu_ra_cosdec, mu_dec] from the
# core080 B-spline tracks + the rotation curve + the TRUE globals (coupling flow), on the prog2026 3e5 set.
# The vcirc_kms TST has global trial 14's architecture and is WARM-STARTED (trainable) from its weights; the
# sim_summary TST and the coupling flow (depth, affine/spline, subnet width/depth/dropout) are searched
# (conf/tuning/stream_2modal_bspline_local.yaml). 500 epochs per trial.
# One worker per launch; launch twice (two GPUs) to run two trainings in parallel on the SAME study:
#   GPU=3 nohup bash scripts/tune_local_bspline_prog2026.sh > outputs/Bsline/spray_p1e3_v4_prog2026_bspline_local/w1.log 2>&1 &
#   GPU=5 nohup bash scripts/tune_local_bspline_prog2026.sh > outputs/Bsline/spray_p1e3_v4_prog2026_bspline_local/w2.log 2>&1 &
# (no GPU= -> autocvd waits for a free card; stagger the two launches by ~2 min.)
# Per worker: hold the GPU for the whole run; loop { train 1 trial; post-eval every finished, unclaimed trial:
#   eval_sim_333 (true globals) -> eval_real (ancestral, globals of GLOBAL_RUN_DIR) -> joint PPC (N_PPC draws, CPU)
#   -> per-stream MMD misspecification test (eval_real/mmd_local/) }.
# Study + trials: <DATA_DIR>/tuning/stream_2modal_bspline_local_study/
# Knobs: GPU (auto|<id>|cpu) N_TRIALS (25) N_EPOCHS (500) BATCH_SIZE (2048) N_TRAIN (300000) N_PPC (50)
#        GLOBAL_RUN_DIR STUDY EXTRA
cd "$(dirname "$0")/.."
PY=.venv/bin/python

# --- [0] grab the GPU and keep it until this script exits ---
GPU=${GPU:-auto}
if [ "${GPU}" = cpu ]; then
  export JAX_PLATFORMS=cpu
else
  if [ "${GPU}" = auto ]; then eval "$(.venv/bin/autocvd -n 1 -e)"; else export CUDA_VISIBLE_DEVICES=${GPU}; fi
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  ${PY} -c "import jax,time; x=jax.device_put(jax.numpy.zeros(256*2**20)); x.block_until_ready(); print('holding', jax.devices(), flush=True); time.sleep(10**9)" &
  HOLD=$!; trap 'kill ${HOLD} 2>/dev/null' EXIT
  echo "GPU ${CUDA_VISIBLE_DEVICES} held by pid ${HOLD}"
fi
set -uo pipefail
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}

SIM=stream_agama_spray_massloss_ibata_m200c_v4_prog2026
DATA_DIR=${DATA_DIR:-data/data_jarvis/data_agama_spray_massloss_ibata_m200c_v4_p1e3_prog2026_hydrabflow}
STUDY=${STUDY:-stream_2modal_bspline_local_study}
SDIR=${DATA_DIR}/tuning/${STUDY}
T14=data/data_jarvis/data_agama_spray_massloss_ibata_m200c_v4_p1e3_prog2026_hydrabflow/tuning/tuningtest_2modal_bspline_core080_prog2026_study/trials/trial_0014
GLOBAL_RUN_DIR=${GLOBAL_RUN_DIR:-${T14}/eval_real}
REAL=assets/gaia/gaia_observed_streams_6Dwitherrors_cutNGC3201.npz
OUT_ROOT=outputs/Bsline/spray_p1e3_v4_prog2026_bspline_local/tuning
mkdir -p "${OUT_ROOT}" "${SDIR}/warm"

# --- [1] trial 14's curve-TST weights (extracted by core080_local_warm14/extract_backbones.py) ---
W=${SDIR}/warm/vcirc_kms.npz
[ -f "${W}" ] || cp outputs/Bsline/spray_p1e3_v4_prog2026_core080_local_warm14/warm/vcirc_kms.npz "${W}"

# shared by train and every evaluate: trial 14's vcirc_kms architecture + warm start
V=model.summary_network.params.backbones.vcirc_kms
FIXED="++${V}.summary_dim=32 ++${V}.num_blocks=5 ++${V}.num_heads=4 ++${V}.embed_dim_per_head=8 ++${V}.dropout=0.25 \
++model.summary_network.params.warm_weights.vcirc_kms=${W} ${EXTRA:-}"
COMMON="simulator=${SIM} model=stream_fusion_2modal_oldgrid_local composition=local adapter=stream_2modal_local \
augmentation.params.resources_dir=assets/gaia training.standardize=[inference_variables,inference_conditions,summary_variables] \
data.data_dir=${DATA_DIR} ${FIXED}"

post_eval() {   # $1 = trial dir; every stage logs into the trial, a failure never stops the worker
  local t=$1 ovr
  ln -sf "$(realpath "${SDIR}/preprocessing_state.npz")" "${t}/preprocessing_state.npz"
  ovr=$(${PY} -c "import json,sys;print(' '.join(f'++{k}={v}' for k,v in json.load(open(sys.argv[1])).items()))" "${t}/params.json")
  echo "=== $(date +%F_%T) post-eval ${t} ==="
  ${PY} -m hydrabflow.pipeline.evaluate ${COMMON} ${ovr} \
    preprocessing=stream_local_log10_sumstats_2modal augmentation=stream_global_streamfinder_bspline_core080 \
    data.n_simulations=333 eval.test_dataset_name=test_multistream_333.npz \
    model_dir="${t}" hydra.run.dir="${t}/eval_sim_333" > "${t}/eval_sim_333.log" 2>&1 || { echo "!!! eval_sim failed: ${t}"; return; }
  ${PY} -m hydrabflow.pipeline.evaluate ${COMMON} ${ovr} \
    preprocessing=stream_real_local_log10 augmentation=stream_real_global_streamfinder_bspline_core080 \
    data.real_data_path="${REAL}" composition.global_run_dir="${GLOBAL_RUN_DIR}" \
    model_dir="${t}" hydra.run.dir="${t}/eval_real" > "${t}/eval_real.log" 2>&1 || { echo "!!! eval_real failed: ${t}"; return; }
  # CPU re-simulation; the holder above keeps the GPU meanwhile
  JAX_PLATFORMS=cpu HYDRABFLOW_NUM_GPUS=0 ${PY} scripts/ppc_joint_posterior.py --fit aug \
    --local-run "${t}/eval_real" --global-run "${GLOBAL_RUN_DIR}" \
    --train-config "${t}/eval_sim_333/.hydra/config.yaml" --simulator-preset ${SIM} \
    --real-aug stream_real_global_streamfinder_bspline_core080 \
    --n-samples "${N_PPC:-50}" --n-particles 1000 --n-workers 24 --out "${t}/eval_real/ppc_joint" \
    > "${t}/ppc_joint.log" 2>&1 || echo "!!! PPC failed: ${t}"
  # per-stream summary-space misspecification: LOO MMD + Mahalanobis, and bf's default MMD test
  # (mmd_hypothesis_test_bfdefault_<stream>_<slice>.pdf, slice = sim_summary | vcirc_kms | fused)
  JAX_PLATFORMS=cpu HYDRABFLOW_NUM_GPUS=0 ${PY} scripts/mmd_local_per_stream.py \
    --model-dir "${t}" --real-run "${t}/eval_real" --out "${t}/eval_real/mmd_local" \
    > "${t}/mmd_local.log" 2>&1 || echo "!!! MMD failed: ${t}"
  touch "${t}/.posteval/done"
}

for i in $(seq 1 "${N_TRIALS:-25}"); do
  echo "=== $(date +%F_%T) worker trial ${i}/${N_TRIALS:-25} (GPU ${CUDA_VISIBLE_DEVICES:-cpu}) ==="
  ${PY} -m hydrabflow.pipeline.tune ${COMMON} \
    preprocessing=stream_local_log10_sumstats_2modal augmentation=stream_global_streamfinder_bspline_core080 \
    tuning=stream_2modal_bspline_local tuning.study_name="${STUDY}" tuning.n_trials=1 tuning.n_epochs="${N_EPOCHS:-500}" \
    data.n_simulations="${N_TRAIN:-300000}" training.batch_size="${BATCH_SIZE:-2048}" seed=2026 \
    eval.batch_size=256 eval.num_samples=500 \
    hydra.run.dir="${OUT_ROOT}/worker_$(date +%Y%m%d_%H%M%S)" || echo "!!! tune run ${i} failed"
  for t in "${SDIR}"/trials/trial_*; do   # mkdir is atomic: the two workers never evaluate the same trial
    [ -f "${t}/approximator.keras" ] && [ -f "${t}/params.json" ] && mkdir "${t}/.posteval" 2>/dev/null && post_eval "${t}"
  done
done
echo "=== worker DONE ==="

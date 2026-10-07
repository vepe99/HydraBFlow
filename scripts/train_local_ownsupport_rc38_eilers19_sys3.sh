#!/usr/bin/env bash
# LOCAL model on the OWN-SUPPORT spline (stream_spline_ownsupport) + rotation curve, rc38 x Eilers19 (+3 % sys) 3e5 set:
# two TimeSeriesTransformers (sizes of global trial 24 of tuningtest_2modal_bspline_core080_rc38_eilers19_sys3_study),
# coupling flow, the 6 locals conditioned on the TRUE 8 globals, z-scored per stream on the train split.
#   [1/5] train  [2/5] eval sim (333 groups)  [3/5] eval real (ancestral, globals = global trial 24's posterior)
#   [4/5] summary-space MMD per stream (LOO null)  [5/5] joint PPC (CPU, 40 paired draws, training spline)
# Run: bash scripts/train_local_ownsupport_rc38_eilers19_sys3.sh     (GPU=<id> to pin; default waits for a free GPU)
# Knobs: AUG / REAL_AUG (default the own-support spline pair), WARM_DIR (dir with <backbone>_weights.npz from
#        scripts/export_backbone_weights.py: warm-start the summary backbones, still trained), RUNS_DIR, GPU.
# Smoke: GPU=cpu N_EPOCHS=1 BATCH_SIZE=64 N_TRAIN=2000 N_PPC=4 RUNS_DIR=/tmp/x bash scripts/train_local_ownsupport_rc38_eilers19_sys3.sh
set -euo pipefail
cd "$(dirname "$0")/.."
SIM=stream_agama_spray_massloss_ibata_m200c_v4_prog2026_rc38_eilers19_sys3
DATA_DIR=${DATA_DIR:-data/data_jarvis/data_agama_spray_massloss_ibata_m200c_v4_p1e3_prog2026_rc38_eilers19_hydrabflow}
G24=data/data_jarvis/data_agama_spray_massloss_ibata_m200c_v4_p1e3_prog2026_rc38_eilers19_hydrabflow/tuning/tuningtest_2modal_bspline_core080_rc38_eilers19_sys3_study/trials/trial_0024
RUNS_DIR=${RUNS_DIR:-outputs/Bsline/spray_p1e3_v4_prog2026_rc38_eilers19_sys3_ownsupport_local}
mkdir -p "${RUNS_DIR}"
export PY=.venv/bin/python AUTOCVD=.venv/bin/autocvd OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
GPU=${GPU:-auto}
if [ "${GPU}" = auto ]; then eval "$(${AUTOCVD} -e -n 1 -i 1)"; GPU=${CUDA_VISIBLE_DEVICES}; fi
if [ "${GPU}" = cpu ]; then export HYDRABFLOW_NUM_GPUS=0; else export CUDA_VISIBLE_DEVICES=${GPU} JAX_PLATFORMS=cuda,cpu; fi
# backbone sizes of global trial 24 (diffusion-transformer keys dropped: the local net is a coupling flow)
EXTRA=$(${PY} -c "import json,sys;print(' '.join(f'++{k}={v}' for k,v in json.load(open(sys.argv[1])).items() if 'summary_network' in k))" "${G24}/params.json")
[ -n "${WARM_DIR:-}" ] && EXTRA="${EXTRA} ++model.summary_network.params.warm_weights.sim_summary=${WARM_DIR}/sim_summary_weights.npz ++model.summary_network.params.warm_weights.vcirc_kms=${WARM_DIR}/vcirc_kms_weights.npz"
AUG=${AUG:-stream_global_streamfinder_spline_ownsupport} REAL_AUG=${REAL_AUG:-stream_real_global_streamfinder_spline_ownsupport}

GPU=${GPU} SIM=${SIM} DATA_DIR=${DATA_DIR} N_TRAIN=${N_TRAIN:-300000} N_EPOCHS=${N_EPOCHS:-300} BATCH_SIZE=${BATCH_SIZE:-2048} \
AUG=${AUG} REAL_AUG=${REAL_AUG} \
RUNS_DIR=${RUNS_DIR} MODEL_DIR=${RUNS_DIR}/train EVAL_DIR=${RUNS_DIR}/train/eval_sim_333 REAL_DIR=${RUNS_DIR}/eval_real \
GLOBAL_RUN_DIR=${G24}/eval_real EXTRA="${EXTRA}" \
  bash scripts/train_local_2modal.sh

echo "=== [4/5] summary-space MMD per stream ==="
${PY} scripts/mmd_local_per_stream.py --model-dir "${RUNS_DIR}/train" --real-run "${RUNS_DIR}/eval_real" --out "${RUNS_DIR}/mmd_local"

echo "=== [5/5] joint PPC (CPU) ==="
JAX_PLATFORMS=cpu HYDRABFLOW_NUM_GPUS=0 CUDA_VISIBLE_DEVICES= ${PY} scripts/ppc_joint_posterior.py --fit aug \
  --local-run "${RUNS_DIR}/eval_real" --global-run "${G24}/eval_real" \
  --train-config "${RUNS_DIR}/train/.hydra/config.yaml" --simulator-preset ${SIM} \
  --real-aug ${REAL_AUG} \
  --n-samples ${N_PPC:-40} --n-particles 1000 --n-workers 24 --out "${RUNS_DIR}/ppc_joint"

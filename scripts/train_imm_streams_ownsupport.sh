#!/usr/bin/env bash
# Streams-only information-maximising summary net on the own-support spline observable (trial-14 sim_summary
# backbone, CouplingFlow head, no modality dropout), prog2026 ou24vc 3e5 spray set. Train -> sim eval (base_*).
# ARM=streams (default) | vcirc (rotation curve only, trial-14 vcirc backbone).
# Run:    ARM=vcirc bash scripts/train_imm_streams_ownsupport.sh      (autocvd waits for a free GPU)
# Smoke:  GPU=cpu N_EPOCHS=1 N_TRAIN=2000 BATCH_SIZE=256 RUNS_DIR=/tmp/x bash scripts/train_imm_streams_ownsupport.sh
set -euo pipefail
cd "$(dirname "$0")/.."

GPU=${GPU:-auto}
if [ "${GPU}" = "cpu" ]; then
  export JAX_PLATFORMS=cpu HYDRABFLOW_NUM_GPUS=0
else
  if [ "${GPU}" = "auto" ]; then eval "$(.venv/bin/autocvd -n 1)"; else CUDA_VISIBLE_DEVICES="${GPU}"; fi
  export CUDA_VISIBLE_DEVICES XLA_PYTHON_CLIENT_PREALLOCATE=false
fi
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<cpu>}"

SIM=stream_agama_spray_massloss_ibata_m200c_v4_prog2026_ou24vc
DATA_DIR=${DATA_DIR:-data/data_jarvis/data_agama_spray_massloss_ibata_m200c_v4_p1e3_prog2026_ou24vc_hydrabflow}
case "${ARM:-streams}" in
  streams) MODEL=imm_streams_ownsupport ADAPTER=stream_streams_only AUG=stream_global_streamfinder_spline_ownsupport_streams_only ;;
  vcirc)   MODEL=imm_vcirc_ownsupport   ADAPTER=stream_vcirc_only   AUG=stream_global_vcirc_only ;;
  *) echo "ARM must be streams|vcirc"; exit 1 ;;
esac
RUNS_DIR=${RUNS_DIR:-outputs/Bsline/imm_${ARM:-streams}_ownsupport}
COMMON=(simulator="${SIM}" model="${MODEL}" composition=global adapter="${ADAPTER}"
        preprocessing=stream_global_log10_sumstats_2modal augmentation="${AUG}"
        data.data_dir="${DATA_DIR}" seed=2026)
PY=.venv/bin/python

echo "=== [1/2] TRAIN -> ${RUNS_DIR}/train ==="
${PY} -m hydrabflow.pipeline.train "${COMMON[@]}" \
  data.n_simulations="${N_TRAIN:-300000}" training.n_epochs="${N_EPOCHS:-1000}" training.batch_size="${BATCH_SIZE:-4096}" \
  hydra.run.dir="${RUNS_DIR}/train"

echo "=== [2/2] EVALUATE sim 333 -> ${RUNS_DIR}/eval_sim_333 ==="
${PY} -m hydrabflow.pipeline.evaluate "${COMMON[@]}" \
  eval=stream_compositional eval.batch_size=8 data.n_simulations=333 eval.test_dataset_name=test_multistream_333.npz \
  model_dir="${RUNS_DIR}/train" hydra.run.dir="${RUNS_DIR}/eval_sim_333"
echo "=== DONE -> ${RUNS_DIR} ==="

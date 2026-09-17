#!/usr/bin/env bash
# Single-modality CouplingFlow run on the Palau spray v4 set: trains an information-maximising
# summary network on ONE modality (no modality dropout), then evaluates on the 333-group test set
# (base_* per-member metrics; the coupling flow has no compositional stage).
#
#   ARM=vcirc   -> rotation curve only          (model=imm_vcirc)
#   ARM=streams -> gridded summary stats only   (model=imm_streams)
#   MODEL_OVERRIDE=imm_<arm>_mlp swaps in the MLP summary net (same adapter/augmentation).
#
# Run:  GPU=3 ARM=vcirc bash scripts/train_imm_summarynet.sh
set -euo pipefail
cd "$(dirname "$0")/.."

ARM=${ARM:-vcirc}
GPU=${GPU:-auto}
if [ "${GPU}" = "cpu" ]; then
  export JAX_PLATFORMS=cpu
else
  if [ "${GPU}" = "auto" ]; then eval "$(.venv/bin/autocvd -n 1)"; else export CUDA_VISIBLE_DEVICES="${GPU}"; fi
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
fi
echo "ARM=${ARM} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<cpu>}"

case "${ARM}" in
  vcirc)   MODEL=imm_vcirc;   ADAPTER=stream_vcirc_only;    AUG=stream_global_vcirc_only ;;
  streams) MODEL=imm_streams; ADAPTER=stream_streams_only; AUG=stream_global_sumstats_only ;;
  *) echo "ARM must be vcirc|streams"; exit 1 ;;
esac
MODEL=${MODEL_OVERRIDE:-${MODEL}}   # e.g. MODEL_OVERRIDE=imm_vcirc_mlp for the MLP twin

SIM=${SIM:-stream_agama_spray_massloss_ibata_m200c_v4_palau}
DATA_DIR=${DATA_DIR:-data/data_jarvis/data_agama_spray_massloss_ibata_m200c_v4_palau_hydrabflow}
RES=${RES:-assets/gaia}
SEED=${SEED:-2026}
N_EPOCHS=${N_EPOCHS:-1000}
BATCH_SIZE=${BATCH_SIZE:-1024}
N_TRAIN=${N_TRAIN:-100000}
N_TEST=${N_TEST:-333}
PREPROC=${PREPROC:-stream_global_log10_ibata_sumstats}
RUNS_DIR=${RUNS_DIR:-outputs/imm_summarynet/${ARM}}
MODEL_DIR=${MODEL_DIR:-${RUNS_DIR}/train}
EVAL_DIR=${EVAL_DIR:-${RUNS_DIR}/eval_sim_${N_TEST}}
PY=.venv/bin/python

echo "=== [1/2] TRAIN -> ${MODEL_DIR} ==="
${PY} -m hydrabflow.pipeline.train \
  simulator="${SIM}" model="${MODEL}" composition=global \
  adapter="${ADAPTER}" preprocessing="${PREPROC}" augmentation="${AUG}" \
  data.data_dir="${DATA_DIR}" data.n_simulations="${N_TRAIN}" \
  training.n_epochs="${N_EPOCHS}" training.batch_size="${BATCH_SIZE}" seed="${SEED}" \
  augmentation.params.resources_dir="${RES}" \
  hydra.run.dir="${MODEL_DIR}"

echo "=== [2/2] EVALUATE sim ${N_TEST}-group multistream -> ${EVAL_DIR} ==="
${PY} -m hydrabflow.pipeline.evaluate \
  simulator="${SIM}" model="${MODEL}" composition=global \
  adapter="${ADAPTER}" preprocessing="${PREPROC}" augmentation="${AUG}" \
  eval=stream_compositional eval.batch_size=8 data.data_dir="${DATA_DIR}" data.n_simulations="${N_TEST}" \
  eval.test_dataset_name=test_multistream_${N_TEST}.npz \
  model_dir="${MODEL_DIR}" augmentation.params.resources_dir="${RES}" \
  hydra.run.dir="${EVAL_DIR}"
echo "=== DONE -> ${RUNS_DIR} ==="

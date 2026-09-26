#!/usr/bin/env bash
# ONE Optuna worker for the two-modality oldgrid model on the palau v4 spray set (concurrency-safe
# JournalStorage: launch again on another GPU to add a worker to the SAME study). Objectives = base
# RMSE + calibration on the val split. Run scripts/tune_post_eval.sh alongside for the compositional
# + real-Gaia evaluation of every finished trial.
#   bash scripts/tune_2modal_oldgrid.sh            # autocvd least-used GPU
#   GPU=6 bash scripts/tune_2modal_oldgrid.sh
#   TUNING=tuningtest_2modal_oldgrid GPU=6 bash scripts/tune_2modal_oldgrid.sh   # test-set-scored study
set -euo pipefail
cd "$(dirname "$0")/.."
GPU=${GPU:-auto}
if [ "${GPU}" = "auto" ]; then eval "$(.venv/bin/autocvd -l -e -n 1)"; else export CUDA_VISIBLE_DEVICES="${GPU}"; fi
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}   # caller may set true to grab the card up front

SIM=${SIM:-stream_agama_spray_massloss_ibata_m200c_v4_palau}
DATA_DIR=${DATA_DIR:-data/data_jarvis/data_agama_spray_massloss_ibata_m200c_v4_palau_hydrabflow}
N_TRAIN=${N_TRAIN:-100000}
N_TRIALS=${N_TRIALS:-50}
N_EPOCHS=${N_EPOCHS:-1000}
BATCH_SIZE=${BATCH_SIZE:-1024}
DROP_PROB=${DROP_PROB:-0.5}
MODEL=${MODEL:-stream_fusion_2modal_oldgrid}
AUG=${AUG:-stream_global_ibata_grid}
PREPROC=${PREPROC:-stream_global_log10_ibata_sumstats}   # legacy ou24 set: stream_global_log10_sumstats_2modal
STUDY=${STUDY:-}   # optional tuning.study_name override (default: the yaml's)
TUNING=${TUNING:-stream_2modal_oldgrid}   # tuningtest_2modal_oldgrid = score on the 333 test set + real eval per trial
OUT_ROOT=${OUT_ROOT:-outputs/${TUNING/stream_/tuning_}}
mkdir -p "${OUT_ROOT}"
echo "=== [tune] GPU=${CUDA_VISIBLE_DEVICES} trials=${N_TRIALS} epochs=${N_EPOCHS} drop=${DROP_PROB} ==="
.venv/bin/python -m hydrabflow.pipeline.tune \
  simulator="${SIM}" model="${MODEL}" composition=global adapter=stream_2modal \
  preprocessing="${PREPROC}" augmentation="${AUG}" \
  ${STUDY:+tuning.study_name=${STUDY}} \
  tuning="${TUNING}" tuning.n_trials="${N_TRIALS}" tuning.n_epochs="${N_EPOCHS}" \
  model.inference_network.params.missing_modality_prob="${DROP_PROB}" \
  data.data_dir="${DATA_DIR}" data.n_simulations="${N_TRAIN}" training.batch_size="${BATCH_SIZE}" seed=2026 \
  eval.batch_size=256 eval.num_samples=500 augmentation.params.resources_dir=assets/gaia \
  ${EXTRA:-} \
  hydra.run.dir="${OUT_ROOT}/worker_$(date +%Y%m%d_%H%M%S)"

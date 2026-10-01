#!/usr/bin/env bash
# Optuna study of the LOCAL particle model of scripts/train_local_particles_prog2026.sh: same prog2026 3e5 set,
# model/adapter/preprocessing/augmentation (stream_global, vlos zero-fill), coupling flow and standardization;
# search space = conf/tuning/stream_2modal_particles_local.yaml (both summary backbones). 300 epochs per
# trial, scored on the val split (RMSE + calibration of the per-stream z-scored locals).
# Study + trials: <DATA_DIR>/tuning/stream_2modal_particles_local_study/ (JournalStorage: launch this script
# again on another GPU to add a worker to the SAME study). Worker logs under OUT_ROOT.
#   GPU=2 bash scripts/tune_local_particles_prog2026.sh
#   GPU=2 N_TRIALS=10 bash scripts/tune_local_particles_prog2026.sh
# Knobs: SIM DATA_DIR N_TRIALS (25) N_EPOCHS (300) BATCH_SIZE (1024) N_TRAIN (300000) GPU (auto = wait for a free one) OUT_ROOT
set -euo pipefail
cd "$(dirname "$0")/.."
GPU=${GPU:-auto}
if [ "${GPU}" = "auto" ]; then eval "$(.venv/bin/autocvd -e -n 1)"; else export CUDA_VISIBLE_DEVICES="${GPU}"; fi
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}
OUT_ROOT=${OUT_ROOT:-outputs/Bsline/spray_p1e3_v4_prog2026_particles_local/tuning}
mkdir -p "${OUT_ROOT}"
echo "=== [tune local particles] GPU=${CUDA_VISIBLE_DEVICES} trials=${N_TRIALS:-25} epochs=${N_EPOCHS:-300} ==="
.venv/bin/python -m hydrabflow.pipeline.tune \
  simulator="${SIM:-stream_agama_spray_massloss_ibata_m200c_v4_prog2026}" \
  model=stream_fusion_2modal_particles_local composition=local adapter=stream_2modal_particles_local \
  preprocessing=stream_local_log10_sumstats_2modal augmentation=stream_global \
  augmentation.params.vlos_impute=zero augmentation.params.resources_dir=assets/gaia \
  "training.standardize=[inference_variables,inference_conditions,summary_variables]" \
  tuning=stream_2modal_particles_local tuning.n_trials="${N_TRIALS:-25}" tuning.n_epochs="${N_EPOCHS:-300}" \
  data.data_dir="${DATA_DIR:-data/data_jarvis/data_agama_spray_massloss_ibata_m200c_v4_p1e3_prog2026_hydrabflow}" \
  data.n_simulations="${N_TRAIN:-300000}" training.batch_size="${BATCH_SIZE:-1024}" seed=2026 \
  eval.batch_size=256 eval.num_samples=500 \
  ${EXTRA:-} \
  hydra.run.dir="${OUT_ROOT}/worker_$(date +%Y%m%d_%H%M%S)" 2>&1 | tee -a "${OUT_ROOT}/worker_$(date +%Y%m%d_%H%M%S).log"

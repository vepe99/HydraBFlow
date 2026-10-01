#!/usr/bin/env bash
# Local-level twin of scripts/tune_post_eval.sh for stream_2modal_particles_local_study: the study scores
# trials on the VAL split (~30k rows), so this re-evaluates chosen trials on the 333-group test set exactly
# as scripts/train_local_particles_prog2026.sh does ([2/3]), into <trial>/eval_sim_333/.
#   bash scripts/tune_post_eval_local_particles.sh                 # every finished, not-yet-evaluated trial
#   TRIALS="trial_0003 trial_0007" GPU=5 bash scripts/tune_post_eval_local_particles.sh
set -euo pipefail
cd "$(dirname "$0")/.."
GPU=${GPU:-auto}
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}

DATA_DIR=${DATA_DIR:-data/data_jarvis/data_agama_spray_massloss_ibata_m200c_v4_p1e3_prog2026_hydrabflow}
STUDY=${STUDY:-stream_2modal_particles_local_study}
TRIALS_DIR=${DATA_DIR}/tuning/${STUDY}/trials
TRIALS=${TRIALS:-$(ls "${TRIALS_DIR}")}

for name in ${TRIALS}; do
  t=${TRIALS_DIR}/${name}
  [ -f "${t}/approximator.keras" ] && [ -f "${t}/params.json" ] || { echo "!!! ${t} not finished, skipping"; continue; }
  [ -f "${t}/eval_sim_333/M68_metrics.json" ] && { echo "--- ${name} already evaluated, skipping"; continue; }
  if [ "${GPU}" = "auto" ]; then eval "$(.venv/bin/autocvd -e -n 1 -q)"; else export CUDA_VISIBLE_DEVICES="${GPU}"; fi
  # evaluate loads preprocessing_state.npz from model_dir; the study keeps one shared copy
  ln -sf "$(realpath "${TRIALS_DIR}/../preprocessing_state.npz")" "${t}/preprocessing_state.npz"
  OVR=$(.venv/bin/python -c "import json,sys;print(' '.join(f'++{k}={v}' for k,v in json.load(open(sys.argv[1])).items()))" "${t}/params.json")
  echo "=== $(date +%F_%T) evaluating ${t} on GPU ${CUDA_VISIBLE_DEVICES} ==="
  .venv/bin/python -m hydrabflow.pipeline.evaluate \
    simulator="${SIM:-stream_agama_spray_massloss_ibata_m200c_v4_prog2026}" \
    model=stream_fusion_2modal_particles_local composition=local adapter=stream_2modal_particles_local \
    preprocessing=stream_local_log10_sumstats_2modal augmentation=stream_global \
    augmentation.params.vlos_impute=zero augmentation.params.resources_dir=assets/gaia \
    "training.standardize=[inference_variables,inference_conditions,summary_variables]" \
    data.data_dir="${DATA_DIR}" data.n_simulations=333 eval.test_dataset_name=test_multistream_333.npz \
    ${OVR} model_dir="${t}" hydra.run.dir="${t}/eval_sim_333" 2>&1 | tee "${t}/eval_sim_333.log"
done

#!/usr/bin/env bash
# Companion to scripts/tune_2modal_oldgrid.sh: evaluates every finished trial the way a train run
# is evaluated -- pooled compositional on the 333-group test set (eval_sim_333/) and real Gaia
# (eval_real/: corner, MMD, pairs) -- re-expressing the trial's sampled hyperparameters from its
# params.json as Hydra overrides. Loops until the study has N_TRIALS finished+evaluated trials.
#   bash scripts/tune_post_eval.sh          # per trial: autocvd waits for a free GPU (GPU=<id> to pin)
set -uo pipefail
cd "$(dirname "$0")/.."
GPU=${GPU:-auto}   # auto = wait for a FREE gpu before every trial (autocvd without -l blocks until one is free)
export XLA_PYTHON_CLIENT_PREALLOCATE=false

SIM=${SIM:-stream_agama_spray_massloss_ibata_m200c_v4_palau}
DATA_DIR=${DATA_DIR:-data/data_jarvis/data_agama_spray_massloss_ibata_m200c_v4_palau_hydrabflow}
STUDY=${STUDY:-stream_2modal_oldgrid_study}
TRIALS_DIR=${TRIALS_DIR:-${DATA_DIR}/tuning/${STUDY}/trials}
N_TRIALS=${N_TRIALS:-50}
DROP_PROB=${DROP_PROB:-0.5}
EVAL_BS=${EVAL_BS:-8}  
MODEL=${MODEL:-stream_fusion_2modal_oldgrid}
REAL=assets/gaia/gaia_observed_streams_6Dwitherrors_cutNGC3201.npz
COMMON="simulator=${SIM} model=${MODEL} composition=global adapter=stream_2modal eval=stream_compositional_masked eval.batch_size=${EVAL_BS}  model.inference_network.params.missing_modality_prob=${DROP_PROB} augmentation.params.resources_dir=assets/gaia"

while true; do
  done_n=0
  for t in "${TRIALS_DIR}"/trial_*/; do
    [ -f "${t}approximator.keras" ] && [ -f "${t}params.json" ] || continue
    if [ -f "${t}eval_real/posterior.npz" ]; then done_n=$((done_n+1)); continue; fi
    if [ "${GPU}" = "auto" ]; then eval "$(.venv/bin/autocvd -e -n 1 -q)"; else export CUDA_VISIBLE_DEVICES="${GPU}"; fi
    ln -sf "$(realpath "${TRIALS_DIR}/../preprocessing_state.npz")" "${t}preprocessing_state.npz"
    # params.json: {"dotted.config.path": value} -> path=value overrides
    OVR=$(.venv/bin/python -c "import json,sys;print(' '.join(f'++{k}={v}' for k,v in json.load(open(sys.argv[1])).items()))" "${t}params.json")
    echo "=== $(date +%F_%T) evaluating ${t} ==="
    .venv/bin/python -m hydrabflow.pipeline.evaluate ${COMMON} ${OVR} \
      preprocessing=stream_global_log10_ibata_sumstats augmentation=stream_global_ibata_grid \
      data.data_dir="${DATA_DIR}" data.n_simulations=333 eval.test_dataset_name=test_multistream_333.npz \
      model_dir="${t%/}" hydra.run.dir="${t}eval_sim_333" || echo "!!! sim eval failed for ${t}"
    .venv/bin/python -m hydrabflow.pipeline.evaluate ${COMMON} ${OVR} \
      preprocessing=stream_real_global_ibata_sumstats augmentation=stream_real_global_ibata_grid \
      data.real_data_path="${REAL}" eval.misspecification_reference="${t}eval_sim_333" \
      model_dir="${t%/}" hydra.run.dir="${t}eval_real" || echo "!!! real eval failed for ${t}"
    [ -f "${t}eval_real/posterior.npz" ] && done_n=$((done_n+1))
  done
  [ "${done_n}" -ge "${N_TRIALS}" ] && { echo "all ${done_n} trials evaluated"; break; }
  sleep 600
done

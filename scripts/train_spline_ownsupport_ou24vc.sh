#!/usr/bin/env bash
# Trial 14 hyperparameters (outputs/Bsline/spray_p1e3_v4_prog2026_ou24vc_core080_2modal_trial14) retrained with the
# OWN-SUPPORT spline observation space (`stream_spline_ownsupport`, the training twin of scripts/ppc_stream_splines.py):
#   per realization, after the observed-count subsample (129/195/297 stars, no extra phi1 cut), fit on its own central
#   85 % phi1 range: phi2/mu_phi1/mu_phi2 = penalized spline (lambda = 1e-4 * N_real * L_real^3), parallax = line,
#   v_los = quadratic over the measured stars; then evaluate on 20 phi1 values of the REAL members (inside their core).
# Everything else is trial 14's stack: prog2026 ou24vc 3e5 spray set, stream_fusion_2modal_oldgrid, adapter stream_2modal,
# drop prob 0.5, batch 4096, 1000 epochs, seed 2026, STREAMFINDER members.
#   [1/3] train  [2/3] evaluate sim (test_multistream_333, base + compositional)  [3/3] evaluate real
#
# Run:    bash scripts/train_spline_ownsupport_ou24vc.sh               (GPU picked by autocvd; waits for a free one)
# Smoke:  N_EPOCHS=2 BATCH_SIZE=256 N_TRAIN=300000 RUNS_DIR=/tmp/ownsupport_smoke bash scripts/train_spline_ownsupport_ou24vc.sh
# Knobs:  TUNED_FROM RUNS_DIR DATA_DIR N_EPOCHS (1000) BATCH_SIZE (4096) DROP_PROB (0.5) N_TRAIN (300000) GPU (auto|<id>|cpu) EXTRA
set -euo pipefail
cd "$(dirname "$0")/.."

TUNED_FROM=${TUNED_FROM:-outputs/Bsline/spray_p1e3_v4_prog2026_ou24vc_core080_2modal_trial14/tuned_params.json}
RUNS_DIR=${RUNS_DIR:-outputs/Bsline/spray_p1e3_v4_prog2026_ou24vc_ownsupport_2modal_trial14}
mkdir -p "${RUNS_DIR}"
cp "${TUNED_FROM}" "${RUNS_DIR}/tuned_params.json"

# ++ because struct mode rejects dt_time_embedding_dim / dt_dropout (tune.py force-adds them)
TUNED=$(.venv/bin/python -c "import json,sys;print(' '.join(f'++{k}={v}' for k,v in json.load(open(sys.argv[1])).items()))" "${TUNED_FROM}")

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}
export PY=${PY:-.venv/bin/python} AUTOCVD=${AUTOCVD:-.venv/bin/autocvd} GPU=${GPU:-auto}
SIM=stream_agama_spray_massloss_ibata_m200c_v4_prog2026_ou24vc \
DATA_DIR=${DATA_DIR:-data/data_jarvis/data_agama_spray_massloss_ibata_m200c_v4_p1e3_prog2026_ou24vc_hydrabflow} N_TRAIN=${N_TRAIN:-300000} \
MODEL=stream_fusion_2modal_oldgrid ADAPTER=stream_2modal \
PREPROC=stream_global_log10_sumstats_2modal REAL_PREPROC=stream_real_global_log10 \
AUG=stream_global_streamfinder_spline_ownsupport REAL_AUG=stream_real_global_streamfinder_spline_ownsupport \
N_EPOCHS=${N_EPOCHS:-1000} BATCH_SIZE=${BATCH_SIZE:-4096} DROP_PROB=${DROP_PROB:-0.5} SEED=2026 RUNS_DIR="${RUNS_DIR}" \
EXTRA="${TUNED} ${EXTRA:-}" \
bash scripts/train_v4_2modal.sh 2>&1 | tee "${RUNS_DIR}/run.log"

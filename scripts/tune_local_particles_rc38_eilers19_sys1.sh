#!/usr/bin/env bash
# rc38 x Eilers19 (+1 % systematic) twin of scripts/tune_local_particles_prog2026.sh (study stream_2modal_particles_local_study,
# trial_0006): same LOCAL particle model (masked SetTransformer over the stars + curve TST, coupling flow,
# stream_global vlos zero-fill), search space and per-stream data z-score of the 6 locals, on the rc38 3e5 set
# with vcirc_kms on the 38 Eilers+2019 radii. Conditions = the 8 rc38 globals (all already in log space; the
# preprocessing's log10 keys are absent from that list, so they stay in the rc38 global model's native space).
# Study + trials: <DATA_DIR>/tuning/stream_2modal_particles_local_rc38_eilers19_sys1_study/ (launch again on another
# GPU to add a worker). Test-set re-eval of finished trials:
#   PRESET=rc38_eilers19_sys1 bash scripts/tune_post_eval_local_particles.sh
#   bash scripts/tune_local_particles_rc38_eilers19_sys1.sh            # autocvd waits for a FREE GPU
#   GPU=6 N_TRIALS=10 bash scripts/tune_local_particles_rc38_eilers19_sys1.sh
# Knobs (passed through): N_TRIALS (25) N_EPOCHS (300) BATCH_SIZE (1024) N_TRAIN (300000) GPU OUT_ROOT EXTRA
set -euo pipefail
cd "$(dirname "$0")/.."
SIM=stream_agama_spray_massloss_ibata_m200c_v4_prog2026_rc38_eilers19_sys1 \
DATA_DIR=data/data_jarvis/data_agama_spray_massloss_ibata_m200c_v4_p1e3_prog2026_rc38_eilers19_hydrabflow \
OUT_ROOT=${OUT_ROOT:-outputs/Bsline/spray_p1e3_v4_prog2026_rc38_eilers19_sys1_particles_local/tuning} \
EXTRA="tuning.study_name=stream_2modal_particles_local_rc38_eilers19_sys1_study ${EXTRA:-}" \
  bash scripts/tune_local_particles_prog2026.sh

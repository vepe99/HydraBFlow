#!/usr/bin/env bash
# trial_0014 (prog2026 core080 study) on the rc38 3e5 set with the Jiao26 rotation curve
# (assets/rotation_curve_custom_v4.csv, 19 radii) instead of rc37's 37 radii. Identical to
# outputs/Bsline/spray_p1e3_v4_prog2026_rc38_core080_2modal_trial14/launch.sh except SIM / DATA_DIR / RUNS_DIR.
# Dataset = the rc38 sets with vcirc_kms recomputed per row with AGAMA on the Jiao26 radii
# (scripts/recompute_vcirc_grid.py; streams unchanged). The simulator config feeds the 19 radii, their
# sigma (add_noise_to_vcirc) and the observed curve (attach_observed_vcirc, real eval) to the pipeline.
#   [1/3] train  [2/3] evaluate sim (test_multistream_333)  [3/3] evaluate real (Gaia STREAMFINDER + Jiao26 curve)
# Run: bash scripts/train_bspline_core080_rc38_jiao26.sh      (GPU=<id> to pin; default autocvd picks a free one)
set -euo pipefail
cd "$(dirname "$0")/.."
TRIAL=${TRIAL:-14} STUDY_DIR=${STUDY_DIR:-data/data_jarvis/data_agama_spray_massloss_ibata_m200c_v4_p1e3_prog2026_hydrabflow/tuning/tuningtest_2modal_bspline_core080_prog2026_study} \
SIM=stream_agama_spray_massloss_ibata_m200c_v4_prog2026_rc38_jiao26 \
DATA_DIR=data/data_jarvis/data_agama_spray_massloss_ibata_m200c_v4_p1e3_prog2026_rc38_jiao26_hydrabflow \
PREPROC=stream_global_rc38_2modal REAL_PREPROC=stream_real_global \
EXTRA="eval.prior_score=diffused ${EXTRA:-}" N_EPOCHS=${N_EPOCHS:-1000} \
RUNS_DIR=${RUNS_DIR:-outputs/Bsline/spray_p1e3_v4_prog2026_rc38_jiao26_core080_2modal_trial${TRIAL:-14}} \
bash scripts/train_bspline_core080_prog2026.sh

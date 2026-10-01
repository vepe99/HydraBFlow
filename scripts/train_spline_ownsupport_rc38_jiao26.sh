#!/usr/bin/env bash
# scripts/train_bspline_core080_rc38_jiao26.sh with the OWN-SUPPORT spline augmentation instead of core080:
# AUG / REAL_AUG = stream_{global,real_global}_streamfinder_spline_ownsupport, the observation space of
# outputs/Bsline/spray_p1e3_v4_prog2026_ou24vc_ownsupport_2modal_trial14_frozenimm (only the augmentation is
# taken: summary backbones are trained end to end, NOT frozen at the IMM weights). Same training set (rc38 3e5,
# vcirc_kms on the 19 Jiao26 radii), test set, real data (STREAMFINDER members + Jiao26 curve), trial-14
# hyperparameters, rc38 preprocessing, diffused prior score, drop prob 0.5, batch 4096, 1000 epochs, seed 2026.
#   [1/3] train  [2/3] evaluate sim (test_multistream_333)  [3/3] evaluate real
# Run: bash scripts/train_spline_ownsupport_rc38_jiao26.sh      (GPU=<id> to pin; default autocvd picks a free one)
set -euo pipefail
# Claim the GPU memory at Python start-up, before the 14.7 GB training set is loaded.
export HYDRABFLOW_EARLY_GPU_INIT=${HYDRABFLOW_EARLY_GPU_INIT:-1} XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-true}
cd "$(dirname "$0")/.."
AUG=stream_global_streamfinder_spline_ownsupport REAL_AUG=stream_real_global_streamfinder_spline_ownsupport \
RUNS_DIR=${RUNS_DIR:-outputs/Bsline/spray_p1e3_v4_prog2026_rc38_jiao26_ownsupport_2modal_trial${TRIAL:-14}} \
bash scripts/train_bspline_core080_rc38_jiao26.sh

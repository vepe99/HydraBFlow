#!/bin/bash
# Training pipeline for the v4 RESTRICTED-N-BODY dataset with the corrected summary-grid model and
# the observation model matched to the Palau23 + Gaia DR3 member catalogue (2026-09-22).
#
# Why this combination:
#   * dataset  — `data_agama_rnbody_ibata_m200c_v4_hydrabflow` (10^5 rows, 10^4 particles). Of every
#     training set on this box it is the only one whose streams are NOT out of distribution against
#     the real members: M68 sky/pm MMD 65/56 and phi1 coverage 0.79, against 100/95 and 0.32 for the
#     particle-spray sets and 100/100 and 0.00 for the surrogates.
#   * bin model — `summary_occupancy=valid` + the masked backbones. With `counts` the in-network
#     `count >= min_count` test ran on STANDARDIZED counts and masked ~98 % of the bins, so the
#     network trained on an all-but-empty stream grid (2026-09-22).
#   * observation model — `stream_global_palau23_dr3_emperr`: member counts 126/71/95 and 17/18/8
#     velocities, the magnitude KDE built from THESE members (`magnitude_source: real_streams`) and
#     the astrometric sigmas drawn from their own (G, sigma) pairs. Fixing the magnitude source alone
#     moved the parallax MMD from 100/99/91 to 82/73/78, and the empirical errors to 68/74/75.
#   * real eval — the same catalogue, through `stream_real_global_palau23_dr3_emperr`, whose
#     `override_obs_error_with_real` puts the members' measured sigmas in the same channels the
#     training chain fills with draws from that distribution.
#
#   [1/3] train            -> ${RUNS_DIR}/train
#   [2/3] evaluate sim     -> ${RUNS_DIR}/eval_sim_333   (per-member base_* AND pooled compositional_*)
#   [3/3] evaluate real    -> ${RUNS_DIR}/eval_real      (posterior + MMD vs the sim reference)
#
# Run:    bash scripts/train_v4_rnbody_palau23.sh
# Smoke:  N_EPOCHS=2 BATCH_SIZE=256 N_TRAIN=2000 RUNS_DIR=/tmp/p23smoke bash scripts/train_v4_rnbody_palau23.sh
#         (a CPU smoke also needs `EXTRA='eval.num_samples=100' N_TEST=<small>`: the default 1000
#          diffusion samples preallocate enough on CPU to segfault under this box's strict
#          overcommit. On GPU the default is fine; if the compositional stage OOMs there, lower
#          eval.batch_size, which is already pinned to 8.)
# GPU:    GPU=6 bash scripts/train_v4_rnbody_palau23.sh   (default: autocvd picks a free card)
#
# Everything is env-overridable; see scripts/train_v4_2modal.sh, which this delegates to.
export SIM=${SIM:-stream_agama_rnbody_ibata_m200c_v4}
export DATA_DIR=${DATA_DIR:-data/data_jarvis/data_agama_rnbody_ibata_m200c_v4_hydrabflow}
export AUG=${AUG:-stream_global_palau23_dr3_emperr}
export REAL_AUG=${REAL_AUG:-stream_real_global_palau23_dr3_emperr}
export REAL=${REAL:-assets/gaia/gaia_observed_streams_palau23_dr3.npz}
export MODEL=${MODEL:-stream_fusion_2modal}
export ADAPTER=${ADAPTER:-stream_2modal}
export EVAL=${EVAL:-stream_compositional_masked}
export RES=${RES:-assets/gaia}
export RUNS_DIR=${RUNS_DIR:-outputs/v4_rnbody_palau23}
export OCC=${OCC:-valid}
export EXTRA="augmentation.params.summary_occupancy=${OCC} ${EXTRA:-}"
exec bash "$(dirname "$0")/train_v4_2modal.sh" "$@"

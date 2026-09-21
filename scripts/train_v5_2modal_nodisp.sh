#!/bin/bash
# v5 training/eval WITHOUT the per-bin dispersion channels (medians + occupancy + j + phi1 only):
# same dataset, priors, real member set and observation model as train_v5_2modal.sh, with the
# `_nodisp` augmentation presets and the matching model channel contract. Ablation for how much of
# the halo information (and of the real-data misspecification) lives in the dispersions.
export MODEL=${MODEL:-stream_fusion_2modal_nodisp}
export AUG=${AUG:-stream_global_v5_nodisp}
export REAL_AUG=${REAL_AUG:-stream_real_global_v5_nodisp}
export RUNS_DIR=${RUNS_DIR:-outputs/v5_2modal/nodisp}
exec bash "$(dirname "$0")/train_v5_2modal.sh" "$@"

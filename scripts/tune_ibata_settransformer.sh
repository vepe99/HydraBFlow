#!/usr/bin/env bash
# New tuning study = the current Ibata gridded-summary tuning (scripts/tune_ibata_onedisk_grid.sh,
# TimeSeriesTransformer backbones with their existing search ranges) PLUS an added SetTransformer
# over the raw star cloud (sim_data_projected). EVERYTHING ELSE is kept identical by delegating to
# the grid runner and only overriding the model / adapter / tuning / study / output directory.
#
# Run:      bash scripts/tune_ibata_settransformer.sh
# Smoke:    N_EPOCHS=1 N_TRAIN=100000 N_TEST=8 N_TRIALS_TOTAL=1 GPU=1 bash scripts/tune_ibata_settransformer.sh
# Pin GPUs: GPU="0 1" bash scripts/tune_ibata_settransformer.sh
set -euo pipefail
cd "$(dirname "$0")/.."

export MODEL=${MODEL:-stream_fusion_ibata_grid_settransformer}
export ADAPTER=${ADAPTER:-stream_ibata_grid_settransformer}
export TUNING=${TUNING:-stream_ibata_grid_settransformer}
export STUDY=${STUDY:-stream_ibata_grid_settransformer_study}
export RUNS_DIR=${RUNS_DIR:-outputs/ibata_onedisk_grid_beta3/tune_settransformer}

echo "=== tune_settransformer: model=${MODEL} adapter=${ADAPTER} tuning=${TUNING} study=${STUDY} ==="
echo "=== runs dir: ${RUNS_DIR} (all other config inherited from tune_ibata_onedisk_grid.sh) ==="
exec bash scripts/tune_ibata_onedisk_grid.sh "$@"

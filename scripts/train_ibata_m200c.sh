#!/usr/bin/env bash
# Train + evaluate (sim + real Gaia) the Ibata SUMMARY-STATISTICS fusion model on the m200_c dataset
# (halo reparametrized by virial mass + concentration, McMillan 2017).
#
# The network sees ONLY the binned per-stream summary statistics (sim_summary, a phi1-gridded
# TimeSeriesTransformer) + the rotation curve (vcirc_kms) + the HI terminal-velocity curve
# (vterm_kms) as fusion backbones, with sigma_z + the stream index j as inference conditions. The
# raw star particles are NOT fed to the network (no SetTransformer over the star cloud) — per the
# request, training/tuning use the summary statistics only. Inference is on the log-space halo
# globals log10_M200 / ln_cvprime (+ gamma, q, disk params); the derived rho/a saved in the npz are
# diagnostics and are ignored by the network.
#
# This is a thin wrapper: it sets the m200_c simulator + dataset dir + output/study locations and
# delegates to scripts/train_ibata_onedisk_grid.sh (the gridded summary-statistics stack:
# model=stream_fusion_ibata_grid, adapter=stream_ibata_sumstats, augmentation=stream_global_ibata_grid,
# preprocessing=stream_global_log10_ibata_sumstats, real eval wired). Every knob of that script
# (N_EPOCHS, BATCH_SIZE, GPU, STANDARDIZE_SIGMA_Z, ...) is honoured via the environment.
#
# Run:          bash scripts/train_ibata_m200c.sh
# Smoke:        N_EPOCHS=2 N_TRAIN=2000 N_TEST=8 bash scripts/train_ibata_m200c.sh
# Override GPU: GPU=3 bash scripts/train_ibata_m200c.sh    (GPU=cpu forces CPU; default: autocvd)
set -euo pipefail
cd "$(dirname "$0")/.."

export SIM=${SIM:-stream_agama_ibata_onedisk_beta3_m200c}
export DATA_DIR=${DATA_DIR:-data/data_jarvis/data_agama_ibata_onedisk_beta3_m200c_hydrabflow}
export RUNS_DIR=${RUNS_DIR:-outputs/ibata_onedisk_grid_m200c/default}

echo "=== train_ibata_m200c: summary-stats stack on ${SIM} (data: ${DATA_DIR}) ==="
echo "=== runs dir: ${RUNS_DIR} (all architecture inherited from train_ibata_onedisk_grid.sh) ==="
exec bash scripts/train_ibata_onedisk_grid.sh "$@"

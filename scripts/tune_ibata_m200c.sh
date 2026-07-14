#!/usr/bin/env bash
# Optuna hyperparameter tuning for the Ibata SUMMARY-STATISTICS fusion model on the m200_c dataset
# (halo reparametrized by virial mass + concentration, McMillan 2017).
#
# Same summary-statistics stack as scripts/train_ibata_m200c.sh (sim_summary gridded
# TimeSeriesTransformer + vcirc_kms + vterm_kms backbones, sigma_z + j conditions; NO SetTransformer
# over the raw star cloud) — the search space (conf/tuning/stream_ibata_grid.yaml) tunes the summary
# / rotation-curve / terminal-velocity backbone widths, the fusion head, and the diffusion subnet.
#
# Thin wrapper: sets the m200_c simulator + dataset dir + a dedicated study name / output dir and
# delegates to scripts/tune_ibata_onedisk_grid.sh (one Optuna worker per free GPU, all sharing ONE
# concurrency-safe JournalStorage study, then milestone sim+Gaia evals + coherence reports). Every
# knob of that script (N_TRIALS_TOTAL, N_EPOCHS, GPU="0 3 5", STANDARDIZE_SIGMA_Z, ...) is honoured.
#
# Run:      bash scripts/tune_ibata_m200c.sh
# Smoke:    N_EPOCHS=1 N_TRAIN=2000 N_TEST=8 N_TRIALS_TOTAL=1 GPU=0 bash scripts/tune_ibata_m200c.sh
# Pin GPUs: GPU="0 1 3" bash scripts/tune_ibata_m200c.sh
set -euo pipefail
cd "$(dirname "$0")/.."

export SIM=${SIM:-stream_agama_ibata_onedisk_beta3_m200c}
export DATA_DIR=${DATA_DIR:-data/data_jarvis/data_agama_ibata_onedisk_beta3_m200c_hydrabflow}
export RUNS_DIR=${RUNS_DIR:-outputs/ibata_onedisk_grid_m200c/tuning}
# NOTE: the study name is NOT overridden. tune.py takes it from conf/tuning/stream_ibata_grid.yaml
# (study_name=stream_ibata_grid_study) and the delegated runner's STUDY var must match that to find
# the study log for the eval/report phase. The JournalStorage .log lives at
# ${DATA_DIR}/tuning/stream_ibata_grid_study.log — already m200_c-specific via DATA_DIR, so it is
# fully separate from the beta3 rho/a study with no name clash.

echo "=== tune_ibata_m200c: summary-stats stack on ${SIM} (data: ${DATA_DIR}) ==="
echo "=== runs dir: ${RUNS_DIR} (all architecture/search inherited from tune_ibata_onedisk_grid.sh) ==="
exec bash scripts/tune_ibata_onedisk_grid.sh "$@"

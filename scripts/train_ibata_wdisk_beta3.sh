#!/usr/bin/env bash
# Train + evaluate the Ibata fusion model on the "wdisk_beta3" dataset:
#   halo scale radius a widened to uniform[1,50], thin-disk Sigma_Disk widened to [1e7,3e9]
#   (fixes the HI terminal-velocity deficit -- v_term probes v_circ at R=4.2-7.5 kpc, the inner
#   disk), and halo outer slope beta fixed at 3.0. Dataset built by:
#     SIM=stream_agama_ibata_wdisk_beta3 \
#     DATA_DIR=data_jarvis/data_agama_ibata_wdisk_beta3_hydrabflow \
#     bash scripts/create_ibata_dataset.sh
#
# This is a THIN wrapper around scripts/train_ibata.sh: it only points SIM + DATA_DIR at the
# wdisk_beta3 config/dataset. The model / adapter / augmentation / preprocessing defaults
# (stream_fusion_ibata -> full v_term + Sigma_z + rho_z observable set) are inherited from
# train_ibata.sh and are correct for this dataset (beta=3 is a fixed constant, so it is
# automatically excluded from the inferred parameters by the adapter derivation).
#
# On the training (GPU) node: put the two .npz under ${DATA_DIR} (scp'd from the generating node),
# then run:
#     bash scripts/train_ibata_wdisk_beta3.sh
#
# All train_ibata.sh knobs pass straight through, e.g.:
#     N_EPOCHS=300 GPU=auto bash scripts/train_ibata_wdisk_beta3.sh
#     DATA_DIR=/scratch/streams/wdisk_beta3 bash scripts/train_ibata_wdisk_beta3.sh
# To drop rho_z (no real-data datum) use the norho presets:
#     MODEL=stream_fusion_ibata_norho ADAPTER=stream_ibata_norho AUG=stream_global_ibata_norho \
#       bash scripts/train_ibata_wdisk_beta3.sh
set -euo pipefail
cd "$(dirname "$0")/.."

export SIM="${SIM:-stream_agama_ibata_wdisk_beta3}"
export DATA_DIR="${DATA_DIR:-data_jarvis/data_agama_ibata_wdisk_beta3_hydrabflow}"

exec bash scripts/train_ibata.sh "$@"

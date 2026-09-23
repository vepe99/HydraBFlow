#!/bin/bash
# Raw-PARTICLE twin of scripts/train_v4_rnbody_palau23.sh: same dataset (rnbody v4, 10^5 rows), same
# corrected Palau23+DR3 observation model, same rotation-curve modality and grouped-diffusion masking,
# but the stream modality is the 15-channel star cloud through a missingness-aware SetTransformer
# (`stream_fusion_2modal_particles`, model5_maskedvlos hyperparameters) instead of the binned summary
# grid. The augmentation presets `*_palau23_dr3_emperr_particles` are the emperr chains with
# `stream_summary_grid` removed; `vlos_impute=zero` because the masked backbone zeroes the
# unmeasured v_los channels itself (2026-07-09).
#
#   [1/3] train -> ${RUNS_DIR}/train   [2/3] evaluate sim -> eval_sim_333   [3/3] evaluate real -> eval_real
#
# Run:    N_EPOCHS=1000 BATCH_SIZE=512 bash scripts/train_v4_rnbody_palau23_particles.sh
#         BATCH_SIZE defaults to 512: 1024 OOMed on a 40 GB card for this backbone (one 15.6 GiB
#         attention buffer, 2026-09-16) and the in-process OOM backoff does not rescue it.
# Smoke:  N_EPOCHS=2 BATCH_SIZE=256 N_TRAIN=2000 RUNS_DIR=/tmp/p23psmoke bash scripts/train_v4_rnbody_palau23_particles.sh
# GPU:    GPU=6 bash scripts/train_v4_rnbody_palau23_particles.sh   (default: autocvd)
export SIM=${SIM:-stream_agama_rnbody_ibata_m200c_v4}
export DATA_DIR=${DATA_DIR:-data/data_jarvis/data_agama_rnbody_ibata_m200c_v4_hydrabflow}
export AUG=${AUG:-stream_global_palau23_dr3_emperr_particles}
export REAL_AUG=${REAL_AUG:-stream_real_global_palau23_dr3_emperr_particles}
export REAL=${REAL:-assets/gaia/gaia_observed_streams_palau23_dr3.npz}
export MODEL=${MODEL:-stream_fusion_2modal_particles}
export ADAPTER=${ADAPTER:-stream_2modal_particles}
export EVAL=${EVAL:-stream_compositional_masked_particles}
export RES=${RES:-assets/gaia}
export BATCH_SIZE=${BATCH_SIZE:-512}
export RUNS_DIR=${RUNS_DIR:-outputs/v4_rnbody_palau23_particles}
export EXTRA="augmentation.params.vlos_impute=zero ${EXTRA:-}"
exec bash "$(dirname "$0")/train_v4_2modal.sh" "$@"

#!/usr/bin/env bash
# Test-set-scored Optuna study of the 2-modal model on the v4 RESTRICTED-N-BODY set with the
# observation model matched to the Palau23 + Gaia DR3 members (the training setup of
# scripts/train_v4_rnbody_palau23.sh, but tuned): simulator stream_agama_rnbody_ibata_m200c_v4,
# augmentation stream_global_palau23_dr3_emperr, model stream_fusion_2modal (masked grid backbone,
# summary_occupancy=valid). Each trial: train -> evaluate on test_multistream_333.npz (eval_sim/)
# -> real Gaia palau23_dr3 (eval_real/). Search space: conf/tuning/stream_2modal_oldgrid.yaml.
#   GPU=<idx> N_TRIALS=25 bash scripts/tune_2modal_palau23.sh
#   MODEL=stream_fusion_2modal_oldgrid OCC=counts bash scripts/tune_2modal_palau23.sh   # plain-TST arm
cd "$(dirname "$0")/.."
OCC=${OCC:-valid}
TUNING=tuningtest_2modal_palau23 \
SIM=stream_agama_rnbody_ibata_m200c_v4 MODEL=${MODEL:-stream_fusion_2modal} \
AUG=stream_global_palau23_dr3_emperr PREPROC=stream_global_log10_ibata_sumstats \
DATA_DIR=data/data_jarvis/data_agama_rnbody_ibata_m200c_v4_hydrabflow N_TRAIN=${N_TRAIN:-100000} \
N_EPOCHS=${N_EPOCHS:-1000} BATCH_SIZE=${BATCH_SIZE:-2048} N_TRIALS=${N_TRIALS:-25} DROP_PROB=${DROP_PROB:-0.5} GPU=${GPU:-auto} \
EXTRA="augmentation.params.summary_occupancy=${OCC} ${EXTRA:-}" \
OUT_ROOT=outputs/tuningtest_2modal_palau23 bash scripts/tune_2modal_oldgrid.sh

#!/bin/bash
# v5 training/eval with the OLD grid estimator, as in outputs/v4_2modal_legacy_oldgrid: sample std
# instead of MAD for the per-bin dispersions and a plain TimeSeriesTransformer over the 14-channel grid
# (no bin masking; the occupancy counts reach the network as features). Everything else — dataset,
# priors, v5 observation model, real member set, phi1 quantile grid fitted on the recommended members —
# is train_v5_2modal.sh. `summary_scale` is declared in stream_global_ibata_grid_v2, so the override
# is accepted in struct mode and is recorded in every stage's .hydra/config.yaml.
export MODEL=${MODEL:-stream_fusion_2modal_oldgrid}
export RUNS_DIR=${RUNS_DIR:-outputs/v5_2modal_300k/oldgrid}
export N_TRAIN=${N_TRAIN:-300000}
export OCC=${OCC:-counts}   # plain TST: counts stay features (as the trained oldgrid checkpoints expect)
export EXTRA="augmentation.params.summary_scale=std ${EXTRA:-}"
exec bash "$(dirname "$0")/train_v5_2modal.sh" "$@"

#!/bin/bash
# Old-grid twin of train_v5_2modal_nodisp.sh: medians-only 9-channel grid (`_nodisp` presets) read by
# the plain TimeSeriesTransformer of stream_fusion_2modal_oldgrid (no channel contract, so the same
# model yaml serves both layouts). summary_scale=std is irrelevant without std channels but is kept
# so the two oldgrid arms differ from each other only in summary_include_std.
export AUG=${AUG:-stream_global_v5_nodisp}
export REAL_AUG=${REAL_AUG:-stream_real_global_v5_nodisp}
export RUNS_DIR=${RUNS_DIR:-outputs/v5_2modal_300k/oldgrid_nodisp}
export N_TRAIN=${N_TRAIN:-300000}
exec bash "$(dirname "$0")/train_v5_2modal_oldgrid.sh" "$@"

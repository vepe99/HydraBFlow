#!/bin/bash
# v5 training/eval defaults (2026-09-21): the 10^3-particle spray m200c dataset, the v5 observation
# model (updated counts, empirical v_los errors, magnitude-weighted v_los selection, M68 main-component
# width cut) and the recommended real member set. Thin wrapper over train_v4_2modal.sh — every
# variable can still be overridden from the environment (MODEL=, N_EPOCHS=, RUNS_DIR=, EXTRA=, ...).
export SIM=${SIM:-stream_agama_spray_massloss_ibata_m200c_v4}
export DATA_DIR=${DATA_DIR:-data/data_jarvis/data_agama_spray_massloss_ibata_m200c_v4_p1e3_hydrabflow}
export AUG=${AUG:-stream_global_v5}
export REAL_AUG=${REAL_AUG:-stream_real_global_v5}
export REAL=${REAL:-assets/gaia/gaia_observed_streams_6Dwitherrors_cutNGC3201_desi_m68palau_main.npz}
export RUNS_DIR=${RUNS_DIR:-outputs/v5_2modal/default}
# Occupancy encoding of the summary grid: the masked backbones (the default MODEL) need the ±1
# validity flags — with `counts` the in-network `count >= min_count` test ran on STANDARDIZED counts
# and masked ~98 % of the bins (2026-09-22; the v5 default/nodisp runs trained on a zeroed stream
# grid). The oldgrid wrappers, whose plain TST reads the counts as features, set OCC=counts.
export OCC=${OCC:-valid}
export EXTRA="simulator.params.n_particles=1000 augmentation.params.summary_occupancy=${OCC} ${EXTRA:-}"
exec bash "$(dirname "$0")/train_v4_2modal.sh" "$@"

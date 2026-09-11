#!/usr/bin/env bash
# 400 halo-prior draws forward-modelled with the trihedron remap, as one 20x20 phi1-vs-phi2 grid
# per stream. Reuses the fiducial 1e6-particle N-body runs cached by run_trihedron_1e6.sh, so it
# runs those three streams in one process and simulates nothing.
#
#   scripts/run_trihedron_prior_grid.sh [n_draws] [threads] [outdir]
set -euo pipefail

N_DRAWS="${1:-400}"
THREADS="${2:-32}"
OUTDIR="${3:-data_local/trihedron_1e6/prior_grid}"
CACHE_ROOT="${CACHE_ROOT:-data_local/trihedron_1e6}"
N_REMAP="${N_REMAP:-100000}"

cd "$(dirname "$0")/.."
mkdir -p "$OUTDIR"

nice -n 10 uv run python scripts/trihedron_prior_grid.py \
    --streams Pal5 NGC3201 M68 \
    --n-draws "$N_DRAWS" --n-remap "$N_REMAP" --grid-cols 20 \
    --agama-threads "$THREADS" \
    --cache-root "$CACHE_ROOT" --outdir "$OUTDIR" \
    2>&1 | tee "$OUTDIR/prior_grid.log"

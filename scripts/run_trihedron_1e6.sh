#!/usr/bin/env bash
# One stream's 1e6-particle trihedron validation: the halo-flattening sweep, then the halo-mass
# sweep, into <OUTDIR>/<stream>/. The two sweeps share the fiducial N-body via the cache, so run
# them in this order and in one process per stream.
#
#   scripts/run_trihedron_1e6.sh M68 [threads] [outdir]
set -euo pipefail

STREAM="${1:?usage: run_trihedron_1e6.sh <stream> [threads] [outdir]}"
THREADS="${2:-32}"
OUTDIR="${3:-data_local/trihedron_1e6}"
N="${N_PARTICLES:-1000000}"
NR="${NOISE_REALIZATIONS:-120}"

cd "$(dirname "$0")/.."
mkdir -p "$OUTDIR"

common=(--streams "$STREAM" --n-particles "$N" --agama-threads "$THREADS"
        --outdir "$OUTDIR" --noise --noise-realizations "$NR")

# halo flattening, at fixed rho/a
nice -n 10 uv run python scripts/compare_trihedron_vs_spray.py "${common[@]}" \
    --out q --param q_TwoPowerTriaxial_halo \
    --values 0.7 0.85 1.0 1.05 1.2 1.4

# halo mass: rho x [0.5, 0.7, 1, 1.4, 2] about the Cautun fit, reported as M200 per value
nice -n 10 uv run python scripts/compare_trihedron_vs_spray.py "${common[@]}" \
    --out rho --param rho_TwoPowerTriaxial_halo \
    --values 1.359787e7 1.903701e7 2.719573e7 3.807402e7 5.439146e7

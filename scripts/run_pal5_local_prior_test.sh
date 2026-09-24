#!/bin/bash
# Is Pal5's observation-space misspecification a LOCAL-prior effect? (2026-09-23)
#
# Two Pal5-only, 10^4-row, 10^4-particle Chen-spray training sets under the LEGACY global prior
# (base stream_agama priors, Ou+2024 grid), differing ONLY in the Pal5 progenitor:
#   legacy    stream_agama_spray_legacy_pal5            m = 4.3e3 Msun,               a = 8.43 pc
#   newlocal  stream_agama_spray_legacy_pal5_newlocal   m ~ N(1.39e4, 0.65e4) (>=1e3), a = 27.34 pc
#   bv21      stream_agama_spray_legacy_pal5_bv21       legacy m/a, Baumgardt-DB (BV21/VB21) Pal5 alpha/delta/d/v_r/pm
# both at t_end = 4 Gyr. Then the model-free observation-space misspecification check
# (ppc_observation_space.py, published STREAMFINDER frames) against the Palau23 + Gaia DR3 member
# catalogue with its matched observation model (empirical DR3 errors, catalogue magnitudes), on:
# both new sets, the orbit + DeltaTheta(phi1) B-spline set, and the rnbody v4 set as the N-body
# reference.
#
#   nohup bash scripts/run_pal5_local_prior_test.sh > logs/pal5_local_prior_test.log 2>&1 &
#
# Overrides: N_TRAIN (10000), N_WORKERS (32 — the box runs strict overcommit, see CLAUDE.md),
# ARMS ("legacy newlocal"), STAGES ("sim ppc"), N_SIM (300 simulated clouds per MMD test).
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu HYDRABFLOW_NUM_GPUS=0 \
       XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

PY=${PY:-.venv/bin/python}
N_TRAIN=${N_TRAIN:-10000}
N_WORKERS=${N_WORKERS:-32}
ARMS=${ARMS:-"legacy newlocal bv21"}
STAGES=${STAGES:-"sim ppc"}
N_SIM=${N_SIM:-300}
REAL=${REAL:-assets/gaia/gaia_observed_streams_palau23_dr3.npz}
AUG=${AUG:-stream_global_palau23_dr3_emperr}
REAL_AUG=${REAL_AUG:-stream_real_global_palau23_dr3_emperr}
OUT=${OUT:-outputs/pal5_local_prior_test}
mkdir -p logs "$OUT"

sim_of() { [ "$1" = legacy ] && echo stream_agama_spray_legacy_pal5 || echo stream_agama_spray_legacy_pal5_$1; }
dir_of() { echo "data_jarvis/data_agama_spray_legacy_pal5_${1}_hydrabflow"; }

ppc() {  # label sim_npz simulator
    echo "=== ppc: $1 ==="
    nice -n 10 $PY scripts/ppc_observation_space.py --sim "$2" --out "$OUT/$1" \
        --simulator "$3" --frame streamfinder --aug "$AUG" --real-aug "$REAL_AUG" --real "$REAL" \
        --n-sim "$N_SIM" --title "$1 vs Palau23+DR3" > "$OUT/$1.log" 2>&1 \
        || echo "ppc $1 FAILED (see $OUT/$1.log)"
    tail -3 "$OUT/$1.log"
}

for arm in $ARMS; do
    SIM=$(sim_of "$arm"); DIR=$(dir_of "$arm"); mkdir -p "$DIR/logs"
    if [[ " $STAGES " == *" sim "* ]] && [ ! -f "$DIR/training_data_${N_TRAIN}.npz" ]; then
        echo "=== simulate $arm ($SIM, $N_TRAIN Pal5 rows) $(date) ==="
        nice -n 10 $PY -m hydrabflow.pipeline.simulate simulator="$SIM" \
            '~simulator.params.target_streams.NGC3201' '~simulator.params.target_streams.M68' \
            data.data_dir="$DIR" data.dataset_name="training_data_${N_TRAIN}.npz" \
            data.n_simulations="$N_TRAIN" data.chunk_size=1000 \
            simulator.params.n_workers="$N_WORKERS" seed=2026 > "$DIR/logs/train.log" 2>&1
        tail -2 "$DIR/logs/train.log"
    fi
done

if [[ " $STAGES " == *" ppc "* ]]; then
    for arm in $ARMS; do
        ppc "spray_legacyglobal_${arm}" "$(dir_of "$arm")/training_data_${N_TRAIN}.npz" "$(sim_of "$arm")"
    done
    ppc bspline_orbit_offset data_jarvis/data_orbit_offset_mcmillan17_v4_hydrabflow/training_data_10000.npz \
        stream_orbit_offset_mcmillan17_v4
    ppc rnbody_v4 data_jarvis/data_agama_rnbody_ibata_m200c_v4_hydrabflow/training_data_100000.npz \
        stream_agama_rnbody_ibata_m200c_v4
fi
echo "done $(date)"

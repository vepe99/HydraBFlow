#!/usr/bin/env bash
# Build a trihedron template from a finished restricted-N-body run, then generate the flat training
# set + the grouped test set with it, then run the prior-predictive checks.
#
#   scripts/create_trihedron_dataset.sh
#   N_TRAIN=100000 N_WORKERS=48 scripts/create_trihedron_dataset.sh
#
# The remap is ~40 ms per stream (vs ~10-30 s for the restricted N-body at 10^4 particles), so the
# whole thing is minutes. Launch the stages with .venv/bin/python, NOT `uv run`, which recreates the
# venv and breaks agama.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv/bin/python}"
FIDUCIAL="${FIDUCIAL:-data_local/mcmillan17_mean_tend_best/mcmillan17_mean.npz}"
TEMPLATE="${TEMPLATE:-data_local/mcmillan17_mean_tend_best/trihedron_template.npz}"
SIM="${SIM:-stream_trihedron_mcmillan17_v4}"
DATA_DIR="${DATA_DIR:-data_jarvis/data_trihedron_mcmillan17_v4_hydrabflow}"
T_GYR="${T_GYR:-0.6}"
N_TRAIN="${N_TRAIN:-10000}"
N_TEST="${N_TEST:-333}"
N_WORKERS="${N_WORKERS:-24}"
REAL="${REAL:-assets/gaia/gaia_observed_streams_6Dwitherrors_cutNGC3201_desi_m68palau_main.npz}"
AUG="${AUG:-stream_global_v5}"
REAL_AUG="${REAL_AUG:-stream_real_global_v5}"

export HYDRABFLOW_NUM_GPUS=0 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$DATA_DIR/logs"

# The box enforces strict overcommit (vm.overcommit_memory=2): allocations fail with MemoryError
# while hundreds of GB sit physically free, because OTHER users' jobs fill the box-wide limit.
head=$(( ($(awk '/CommitLimit/{print $2}' /proc/meminfo) - $(awk '/Committed_AS/{print $2}' /proc/meminfo)) / 1048576 ))
echo "commit headroom: ${head} GiB, using ${N_WORKERS} workers"
[ "$head" -lt 5 ] && { echo "less than 5 GiB of commit headroom — wait or lower N_WORKERS"; exit 1; }

echo "=== 1/5 template (window +-${T_GYR} Gyr) ==="
[ -f "$TEMPLATE" ] || $PY scripts/build_trihedron_template.py \
    --sim "$FIDUCIAL" --out "$TEMPLATE" --T-gyr "$T_GYR"

echo "=== 2/5 pilot (24 rows) ==="
$PY -m hydrabflow.pipeline.simulate simulator="$SIM" \
    data.data_dir="$DATA_DIR/pilot" data.dataset_name=pilot.npz \
    data.n_simulations=24 data.chunk_size=24 \
    simulator.params.n_workers="$N_WORKERS" seed=1 2>&1 | tail -2
$PY - "$DATA_DIR/pilot/pilot.npz" <<'EOF'
import sys, numpy as np
d = np.load(sys.argv[1]); s = d["sim_data_projected"]
nan = ~np.isfinite(s).all(axis=(1, 2))
inw = (s[:, :, 0] != -999).sum(axis=1)
print(f"pilot: {s.shape}  NaN rows {nan.sum()}/{len(s)}  in-window median {np.median(inw):.0f}")
if nan.mean() > 0.05:
    raise SystemExit("pilot NaN fraction above 5% — investigate before the full run")
EOF

echo "=== 3/5 test set (${N_TEST} groups) ==="
nice -n 10 $PY -m hydrabflow.pipeline.simulate_multistream simulator="$SIM" \
    data.data_dir="$DATA_DIR" data.dataset_name="test_multistream_${N_TEST}.npz" \
    data.n_simulations="$N_TEST" data.chunk_size=111 \
    simulator.params.n_workers="$N_WORKERS" seed=7 > "$DATA_DIR/logs/test.log" 2>&1
tail -2 "$DATA_DIR/logs/test.log"

echo "=== 4/5 training set (${N_TRAIN} rows) ==="
nice -n 10 $PY -m hydrabflow.pipeline.simulate simulator="$SIM" \
    data.data_dir="$DATA_DIR" data.dataset_name="training_data_${N_TRAIN}.npz" \
    data.n_simulations="$N_TRAIN" data.chunk_size=1000 \
    simulator.params.n_workers="$N_WORKERS" seed=2026 > "$DATA_DIR/logs/train.log" 2>&1
tail -2 "$DATA_DIR/logs/train.log"

echo "=== 5/5 prior-predictive checks ==="
# Observation space in the PUBLISHED STREAMFINDER frames (Ibata+2024 Table 3) — a fixed external
# ruler, unlike the catalogue-fitted great circle the other ppc_* scripts use.
nice -n 10 $PY scripts/ppc_observation_space.py \
    --sim "$DATA_DIR/test_multistream_${N_TEST}.npz" --out "$DATA_DIR/ppc/observation_space" \
    --simulator "$SIM" --frame streamfinder --aug "$AUG" --real-aug "$REAL_AUG" --real "$REAL" \
    --title "trihedron remap"
nice -n 10 $PY scripts/ppc_observation_space.py \
    --sim "$DATA_DIR/test_multistream_${N_TEST}.npz" --out "$DATA_DIR/ppc/observation_space_icrs" \
    --simulator "$SIM" --frame icrs --aug "$AUG" --real-aug "$REAL_AUG" --real "$REAL" \
    --title "trihedron remap, catalogue coordinates"
nice -n 10 $PY scripts/ppc_rotation_curve_prior.py \
    --sim "$DATA_DIR/training_data_${N_TRAIN}.npz" --simulator "$SIM" \
    --out "$DATA_DIR/ppc/rotation_curve"

cat <<EOM

Datasets in $DATA_DIR. Train with (GPU -> autocvd):

  PY=.venv/bin/python AUTOCVD=.venv/bin/autocvd \\
  DATA_DIR=$DATA_DIR N_TRAIN=$N_TRAIN SIM=$SIM \\
  RUNS_DIR=outputs/trihedron_2modal bash scripts/train_v5_2modal.sh
EOM

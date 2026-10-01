#!/bin/bash
# 10^6-row training set (+ 333-group test set) of the v4 Chen spray + linear mass loss, 10^3 particles,
# 2026-09-22 progenitor table, t_end U[2,5] — i.e. data_agama_spray_massloss_ibata_m200c_v4_p1e3_prog2026 —
# with vcirc_kms on the NEW 37-radius rotation curve (2026-09-30; see
# conf/simulator/stream_agama_spray_massloss_ibata_m200c_v4_prog2026_rc37.yaml). Own dataset dir because the
# old dir's test set is on the 19-radius Ou+2024 grid.
#   seed 2026, chunk 1000 -> the streams are row-for-row identical to training_data_300000 of the old dir
#   (verified bit-identical on chunk 0), only vcirc_kms differs.
# Size ~49 GB (sim_data_projected (1e6,2000,6) float32). io.run_chunked assembles the chunks streaming
# (one chunk in memory), so the final write needs no extra commit headroom. ETA ~10-11 h at 100 workers.
# Strict-overcommit watchdog: wait for commit headroom, size the worker pool to it, re-launch on failure
# (resumes from <stem>.chunks/).
#
# Run:   nohup bash scripts/create_spray_v4_p1e3_prog2026_rc37_dataset.sh > /dev/null 2>&1 &
# Log:   $D/logs/spray_v4_p1e3_prog2026_rc37.log
# Knobs: N_TRAIN (1000000), MAX_WORKERS (100), MIN_WORKERS (16), MIN_HEADROOM_GB (8), D (dataset dir)
set -u
cd /home/giuseppe/HydraBFlow
export HYDRABFLOW_NUM_GPUS=0 HYDRABFLOW_SIM_QUIET=1 JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
D=${D:-data_jarvis/data_agama_spray_massloss_ibata_m200c_v4_p1e3_prog2026_rc37_hydrabflow}
SIM=stream_agama_spray_massloss_ibata_m200c_v4_prog2026_rc37
N_TRAIN=${N_TRAIN:-1000000}
MAX_WORKERS=${MAX_WORKERS:-100}; MIN_WORKERS=${MIN_WORKERS:-16}; MIN_HEADROOM_GB=${MIN_HEADROOM_GB:-8}
mkdir -p "$D/logs"
exec >> "$D/logs/spray_v4_p1e3_prog2026_rc37.log" 2>&1

headroom_gb() { awk '/CommitLimit/{l=$2} /Committed_AS/{c=$2} END{printf "%d", (l-c)/1048576}' /proc/meminfo; }
run_stage() {  # $1 = stage module, rest = overrides; retries until success
  local attempt=0 h w rc
  while true; do
    h=$(headroom_gb)
    if (( h < MIN_HEADROOM_GB )); then echo "[$(date +%T)] headroom ${h} GB < ${MIN_HEADROOM_GB}; waiting"; sleep 180; continue; fi
    w=$(( (h - 3) * 5 )); (( w > MAX_WORKERS )) && w=$MAX_WORKERS; (( w < MIN_WORKERS )) && w=$MIN_WORKERS
    attempt=$((attempt+1)); echo "[$(date +%T)] $1 attempt ${attempt}: headroom ${h} GB -> n_workers=${w}"
    nice -n 10 .venv/bin/python -m hydrabflow.pipeline.$1 simulator=$SIM data.data_dir="$D" \
      simulator.params.n_workers=$w simulator.params.n_particles=1000 "${@:2}"
    rc=$?
    (( rc == 0 )) && { echo "[$(date +%T)] $1 DONE"; return 0; }
    echo "[$(date +%T)] $1 attempt ${attempt} failed (rc=${rc}); retry in 120 s"
    pkill -9 -u giuseppe -f "^/home/giuseppe/HydraBFlow/.venv/bin/python3? -m joblib.externals.loky" 2>/dev/null; sleep 120
  done
}

echo "[$(date +%T)] START  sim=$SIM  dir=$D  n_train=$N_TRAIN"
[ -f $D/test_multistream_333.npz ] || run_stage simulate_multistream data.n_simulations=333 data.chunk_size=111 \
  data.dataset_name=test_multistream_333.npz seed=7
[ -f $D/training_data_${N_TRAIN}.npz ] || run_stage simulate data.n_simulations=$N_TRAIN data.chunk_size=1000 \
  data.dataset_name=training_data_${N_TRAIN}.npz seed=2026
echo "[$(date +%T)] ALL DONE"

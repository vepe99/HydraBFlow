#!/bin/bash
# Re-run of data_jarvis/data_agama_spray_massloss_ibata_m200c_v4_p1e3_hydrabflow/training_data_300000
# (v4 Chen spray + linear mass loss, 10^3 particles, seed 2026, chunk 1000) with the updated
# progenitor table + t_end <= 5 Gyr — see conf/simulator/stream_agama_spray_massloss_ibata_m200c_v4_prog2026.yaml:
#   Pal5    a=27.34 pc  m_final=1.39e4  m ~ U[1.39e4, 7e4]    t_end ~ U[2,5]
#   NGC3201 a=4.75 pc   m_final=1.49e5  m ~ U[1.49e5, 3.5e5]  t_end ~ U[2,5]
#   M68     a=5.91 pc   m_final=1.23e5  m ~ U[1.28e5, 5.46e5] t_end ~ U[2,5]  (final mass below the old floor -> prior unchanged)
# Stages: 333-group multistream test set (seed 7), then the 3e5-row flat training set (seed 2026).
# Strict-overcommit watchdog as in run_spray_v4_p1e3_300k.sh: wait for commit headroom, size the worker
# pool to it, re-launch on failure (io.run_chunked resumes from the saved chunks).
#
# Run:   nohup bash scripts/create_spray_v4_p1e3_prog2026_dataset.sh > /dev/null 2>&1 &
# Log:   $D/logs/spray_v4_p1e3_prog2026.log
# Knobs: MAX_WORKERS (100), MIN_WORKERS (16), MIN_HEADROOM_GB (8), N_TRAIN (300000)
set -u
cd /home/giuseppe/HydraBFlow
export HYDRABFLOW_NUM_GPUS=0 HYDRABFLOW_SIM_QUIET=1 JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
D=data_jarvis/data_agama_spray_massloss_ibata_m200c_v4_p1e3_prog2026_hydrabflow
SIM=stream_agama_spray_massloss_ibata_m200c_v4_prog2026
N_TRAIN=${N_TRAIN:-300000}
MAX_WORKERS=${MAX_WORKERS:-100}; MIN_WORKERS=${MIN_WORKERS:-16}; MIN_HEADROOM_GB=${MIN_HEADROOM_GB:-8}
mkdir -p "$D/logs"
exec >> "$D/logs/spray_v4_p1e3_prog2026.log" 2>&1

headroom_gb() { awk '/CommitLimit/{l=$2} /Committed_AS/{c=$2} END{printf "%d", (l-c)/1048576}' /proc/meminfo; }
run_stage() {  # $1 = stage module, rest = overrides; retries until success
  local attempt=0 h w
  while true; do
    h=$(headroom_gb)
    if (( h < MIN_HEADROOM_GB )); then echo "[$(date +%T)] headroom ${h} GB < ${MIN_HEADROOM_GB}; waiting"; sleep 180; continue; fi
    w=$(( (h - 3) * 5 )); (( w > MAX_WORKERS )) && w=$MAX_WORKERS; (( w < MIN_WORKERS )) && w=$MIN_WORKERS
    attempt=$((attempt+1)); echo "[$(date +%T)] $1 attempt ${attempt}: headroom ${h} GB -> n_workers=${w}"
    nice -n 10 .venv/bin/python -m hydrabflow.pipeline.$1 simulator=$SIM data.data_dir="$D" \
      simulator.params.n_workers=$w simulator.params.n_particles=1000 "${@:2}" && { echo "[$(date +%T)] $1 DONE"; return 0; }
    echo "[$(date +%T)] $1 attempt ${attempt} failed (rc=$?); retry in 120 s"
    pkill -9 -u giuseppe -f "^/home/giuseppe/HydraBFlow/.venv/bin/python3? -m joblib.externals.loky" 2>/dev/null; sleep 120
  done
}

echo "[$(date +%T)] START  sim=$SIM  dir=$D  n_train=$N_TRAIN"
[ -f $D/test_multistream_333.npz ] || run_stage simulate_multistream data.n_simulations=333 data.chunk_size=111 \
  data.dataset_name=test_multistream_333.npz seed=7
[ -f $D/training_data_${N_TRAIN}.npz ] || run_stage simulate data.n_simulations=$N_TRAIN data.chunk_size=1000 \
  data.dataset_name=training_data_${N_TRAIN}.npz seed=2026
echo "[$(date +%T)] ALL DONE"

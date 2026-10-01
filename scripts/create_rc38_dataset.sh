#!/bin/bash
# rc38 training set: 3e5 rows, one stream per potential (j uniform over Pal5/NGC3201/M68), 10^3
# particles, seed 2026, chunks of 1000 (resumable: io.run_chunked keeps <stem>.chunks/ until the
# streaming assembly finishes). Simulator: conf/simulator/stream_agama_spray_massloss_ibata_m200c_v4_prog2026_rc38.yaml
# (Moster-conditional stellar mass, thin + thick exponential disks, log10_M200 U[11.6,12.4], no rejection).
# Never touches rc37 data; refuses to run if the output already exists.
# Strict-overcommit watchdog as in the rc37 runner: wait for commit headroom, size the pool, relaunch on
# failure. After success a summary (rows, NaN rows, floored fraction) is appended to the log.
#
# Run:   nohup bash scripts/create_rc38_dataset.sh > /dev/null 2>&1 &
# Log:   $D/logs/rc38_train.log
# Knobs: N_TRAIN (300000), SEED (2026), MAX_WORKERS (100), MIN_WORKERS (16), MIN_HEADROOM_GB (8), D
# Test set (not run here): add a simulate_multistream stage with data.n_simulations=333 seed=7.
set -u
cd /home/giuseppe/HydraBFlow
export HYDRABFLOW_NUM_GPUS=0 HYDRABFLOW_SIM_QUIET=1 JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
D=${D:-data_jarvis/data_agama_spray_massloss_ibata_m200c_v4_p1e3_prog2026_rc38_hydrabflow}
SIM=stream_agama_spray_massloss_ibata_m200c_v4_prog2026_rc38
N_TRAIN=${N_TRAIN:-300000}; SEED=${SEED:-2026}
MAX_WORKERS=${MAX_WORKERS:-100}; MIN_WORKERS=${MIN_WORKERS:-16}; MIN_HEADROOM_GB=${MIN_HEADROOM_GB:-8}
OUT=$D/training_data_${N_TRAIN}.npz
mkdir -p "$D/logs"
exec >> "$D/logs/rc38_train.log" 2>&1
if [ -f "$OUT" ]; then echo "[$(date +%T)] $OUT exists; refusing to overwrite"; exit 1; fi

headroom_gb() { awk '/CommitLimit/{l=$2} /Committed_AS/{c=$2} END{printf "%d", (l-c)/1048576}' /proc/meminfo; }
echo "[$(date '+%F %T')] START sim=$SIM dir=$D n_train=$N_TRAIN seed=$SEED chunk=1000 n_particles=1000 git=$(git rev-parse --short HEAD)"
attempt=0
while true; do
  h=$(headroom_gb)
  if (( h < MIN_HEADROOM_GB )); then echo "[$(date +%T)] headroom ${h} GB < ${MIN_HEADROOM_GB}; waiting"; sleep 180; continue; fi
  w=$(( (h - 3) * 5 )); (( w > MAX_WORKERS )) && w=$MAX_WORKERS; (( w < MIN_WORKERS )) && w=$MIN_WORKERS
  attempt=$((attempt+1)); echo "[$(date +%T)] simulate attempt ${attempt}: headroom ${h} GB -> n_workers=${w}"
  nice -n 10 .venv/bin/python -m hydrabflow.pipeline.simulate simulator=$SIM data.data_dir="$D" \
    simulator.params.n_workers=$w simulator.params.n_particles=1000 data.n_simulations=$N_TRAIN \
    data.chunk_size=1000 data.dataset_name=training_data_${N_TRAIN}.npz seed=$SEED \
    hydra.run.dir="$D/logs/hydra_train_${N_TRAIN}"
  rc=$?
  (( rc == 0 )) && break
  echo "[$(date +%T)] simulate attempt ${attempt} failed (rc=${rc}); retry in 120 s"
  pkill -9 -u giuseppe -f "^/home/giuseppe/HydraBFlow/.venv/bin/python3? -m joblib.externals.loky" 2>/dev/null; sleep 120
done
echo "[$(date '+%F %T')] simulate DONE"
.venv/bin/python - "$OUT" <<'EOF'
import sys, zipfile
import numpy as np
path = sys.argv[1]
z = np.load(path)
x = z["sim_data_projected"]  # (n, 2000, 6) float32 — loaded once for the NaN count
nan = np.isnan(x).all(axis=(1, 2))
fl = z["disk_mass_floored_derived"][:, 0]
j = z["j"][:, 0].astype(int)
print(f"SUMMARY rows={len(nan)}  all-NaN rows={int(nan.sum())} ({nan.mean():.4%})  "
      f"floored={int(fl.sum())} ({fl.mean():.4%})  streams={np.bincount(j).tolist()}  "
      f"dtype={x.dtype}  vcirc_kms={z['vcirc_kms'].shape}")
EOF
echo "[$(date '+%F %T')] ALL DONE"

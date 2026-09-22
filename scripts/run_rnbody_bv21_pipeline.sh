#!/bin/bash
# Generate a fresh 10^4-particle restricted-N-body dataset with the v4 GLOBAL priors and the
# UPDATED (BV21/VB21) progenitor phase-space priors, then train and evaluate on it.
#
# What differs from `data_agama_rnbody_ibata_m200c_v4_hydrabflow`: only NGC3201's and M68's observed
# distance / v_los / proper motion, which that dataset predates (it was generated 2026-09-12, the
# priors were updated 2026-09-21). Globals, progenitor masses, radii, stripping ages, particle count,
# storage cap and rotation-curve grid are identical — see
# conf/simulator/stream_agama_rnbody_ibata_m200c_v4_bv21.yaml, which restates the numbers so this
# dataset's provenance cannot drift if conf/simulator/stream_agama.yaml is edited again.
#
# Stages (STAGES selects; default all three):
#   test   -> ${DATA_DIR}/test_multistream_333.npz   (seed 7,    ~40 min)
#   train  -> ${DATA_DIR}/training_data_100000.npz   (seed 2026, ~20-25 h at 100 workers)
#   fit    -> ${RUNS_DIR}/{train,eval_sim_333,eval_real}   (GPU; delegates to
#             scripts/train_v4_rnbody_palau23.sh, i.e. the corrected bin model + the Palau23+DR3
#             observation model + real evaluation on that catalogue)
#
# Simulation is CPU/joblib and runs under a strict-overcommit watchdog: this box has
# vm.overcommit_memory=2, and other users routinely fill the box-wide CommitLimit, so the pool is
# sized to the free commit headroom (~0.2 GB/worker) and every stage is retried — io.run_chunked
# resumes from its saved chunks, so a crash costs only the chunk in flight. Check
# `grep Committed_AS /proc/meminfo` against CommitLimit before launching anything large.
#
# Run it detached; it is a multi-day job:
#   nohup bash scripts/run_rnbody_bv21_pipeline.sh > rnbody_bv21.log 2>&1 &
#   tail -f rnbody_bv21.log
#
# Useful overrides:
#   STAGES="test train"      generate only, fit later on the GPU box
#   STAGES=fit               train on an already-generated dataset
#   N_TRAIN=300000           a bigger training set (the 10^5 default is on the small side)
#   MAX_WORKERS=64           cap the pool (default 100; each worker is ~0.2 GB of commit charge)
#   GPU=6                    pin a card for the fit stage (default: autocvd picks a free one)

set -uo pipefail
cd "$(dirname "$0")/.."

SIM=${SIM:-stream_agama_rnbody_ibata_m200c_v4_bv21}
DATA_DIR=${DATA_DIR:-data_jarvis/data_agama_rnbody_ibata_m200c_v4_bv21_hydrabflow}
RUNS_DIR=${RUNS_DIR:-outputs/v4_rnbody_bv21_palau23}
N_TRAIN=${N_TRAIN:-100000}
N_TEST=${N_TEST:-333}
CHUNK=${CHUNK:-1000}
SEED_TRAIN=${SEED_TRAIN:-2026}
SEED_TEST=${SEED_TEST:-7}
STAGES=${STAGES:-"test train fit"}
PY=${PY:-.venv/bin/python}

MAX_WORKERS=${MAX_WORKERS:-100}
MIN_WORKERS=${MIN_WORKERS:-16}
MIN_HEADROOM_GB=${MIN_HEADROOM_GB:-8}

export HYDRABFLOW_NUM_GPUS=0 HYDRABFLOW_SIM_QUIET=1 JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

mkdir -p "${DATA_DIR}/logs"
headroom_gb() { awk '/CommitLimit/{l=$2} /Committed_AS/{c=$2} END{printf "%d", (l-c)/1048576}' /proc/meminfo; }

run_stage() {   # $1 = pipeline module; rest = Hydra overrides. Retries until it succeeds.
  local attempt=0 h w
  while true; do
    h=$(headroom_gb)
    if (( h < MIN_HEADROOM_GB )); then
      echo "[$(date +%F' '%T)] commit headroom ${h} GB < ${MIN_HEADROOM_GB} GB — waiting"; sleep 180; continue
    fi
    w=$(( (h - 3) * 5 )); (( w > MAX_WORKERS )) && w=$MAX_WORKERS; (( w < MIN_WORKERS )) && w=$MIN_WORKERS
    attempt=$((attempt+1))
    echo "[$(date +%F' '%T)] $1 attempt ${attempt}: headroom ${h} GB -> n_workers=${w}"
    nice -n 10 ${PY} -m hydrabflow.pipeline."$1" \
      simulator="${SIM}" data.data_dir="${DATA_DIR}" simulator.params.n_workers="${w}" "${@:2}" \
      && { echo "[$(date +%F' '%T)] $1 DONE"; return 0; }
    echo "[$(date +%F' '%T)] $1 attempt ${attempt} failed (rc=$?) — retrying in 120 s (chunks are resumed)"
    sleep 120
  done
}

if [[ " ${STAGES} " == *" test "* ]]; then
  if [ -f "${DATA_DIR}/test_multistream_${N_TEST}.npz" ]; then
    echo "=== [test] ${DATA_DIR}/test_multistream_${N_TEST}.npz exists — skipping"
  else
    echo "=== [test] ${N_TEST}-group multistream test set (seed ${SEED_TEST}) ==="
    run_stage simulate_multistream data.n_simulations="${N_TEST}" data.chunk_size=111 \
      data.dataset_name="test_multistream_${N_TEST}.npz" seed="${SEED_TEST}"
  fi
fi

if [[ " ${STAGES} " == *" train "* ]]; then
  if [ -f "${DATA_DIR}/training_data_${N_TRAIN}.npz" ]; then
    echo "=== [train] ${DATA_DIR}/training_data_${N_TRAIN}.npz exists — skipping"
  else
    echo "=== [train] ${N_TRAIN}-row flat training set (seed ${SEED_TRAIN}) ==="
    run_stage simulate data.n_simulations="${N_TRAIN}" data.chunk_size="${CHUNK}" \
      data.dataset_name="training_data_${N_TRAIN}.npz" seed="${SEED_TRAIN}"
  fi
  ${PY} - <<PYEOF
import numpy as np
d = np.load("${DATA_DIR}/training_data_${N_TRAIN}.npz")
sd, j = d["sim_data_projected"], np.asarray(d["j"]).reshape(-1).astype(int)
nan = ~np.isfinite(sd).all(axis=(1, 2))
mb = d["m_bound_final"].reshape(len(j), -1)[:, 0] if "m_bound_final" in d.files else None
print(f"[report] rows {len(j)}  NaN {int(nan.sum())} ({100*nan.mean():.2f} %)  shape {sd.shape}")
for i, n in enumerate(["Pal5", "NGC3201", "M68"]):
    m = j == i
    stars = np.isfinite(sd[m]).all(-1).sum(1)
    line = f"[report] {n:9s} rows {m.sum():6d}  in-window stars median {int(np.median(stars))}"
    if mb is not None:
        line += f"  progenitor survives {100*np.mean(mb[m] > 0):.0f} %"
    print(line)
PYEOF
fi

if [[ " ${STAGES} " == *" fit "* ]]; then
  echo "=== [fit] train -> evaluate sim -> evaluate real (GPU) ==="
  unset JAX_PLATFORMS HYDRABFLOW_NUM_GPUS
  SIM="${SIM}" DATA_DIR="${DATA_DIR}" RUNS_DIR="${RUNS_DIR}" \
    N_TRAIN="${N_TRAIN}" N_TEST="${N_TEST}" PY="${PY}" AUTOCVD="${AUTOCVD:-.venv/bin/autocvd}" \
    bash scripts/train_v4_rnbody_palau23.sh
fi

echo "=== ALL DONE. dataset: ${DATA_DIR}   run: ${RUNS_DIR} ==="

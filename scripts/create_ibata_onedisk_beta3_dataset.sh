#!/usr/bin/env bash
# Create the Ibata SINGLE-DISK / beta=3 particle-spray stream datasets, using the config
#   conf/simulator/stream_agama_ibata_onedisk_beta3.yaml
#
# Same full Ibata observable set as create_ibata_dataset.sh (Chen+2024 spray; extended
# Zhou u Huang rotation-curve observable + banded rejection prior; ancillary potential
# observables v_term / Sigma_z / rho_z; truncated r_t=1000 kpc halo; exponential stellar-disk
# vertical profile; fixed HI/H2 gas disks) BUT with:
#   - a SINGLE free thin stellar disk (thick_disk: false),
#   - the fixed McMillan (2017) bulge (bulge_density_norm: 9.93e10),
#   - the halo outer slope FIXED at beta = 3.0 (identity prior on beta_..._halo).
#
#   1. 10^5 flat particle-spray training set -> training_data_100000.npz
#   2. 333-group multistream test set        -> test_multistream_333.npz
#
# CPU/joblib only — NO GPU (agama). Both stages are resumable (per-chunk checkpoints in a
# .chunks/ sidecar), so it is safe to Ctrl-C and re-run. A fast PILOT batch + prior-predictive
# check runs FIRST so you can eyeball all observations before committing to the multi-hour full
# run (set RUN_PILOT=0 to skip).
#
# Run it yourself:   bash scripts/create_ibata_onedisk_beta3_dataset.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# Dataset generation is CPU-only (AGAMA/joblib) — never touch a GPU. This stops every joblib
# worker from probing for GPUs via autocvd (which, on a GPU-less box, prints a warning per worker
# and slows worker startup). AGAMA's own C-level chatter is silenced by HYDRABFLOW_SIM_QUIET
# (default on; set to 0 to see it when debugging a simulator). Net effect: a single tqdm bar.
export HYDRABFLOW_NUM_GPUS=${HYDRABFLOW_NUM_GPUS:-0}
export HYDRABFLOW_SIM_QUIET=${HYDRABFLOW_SIM_QUIET:-1}

SIM=${SIM:-stream_agama_ibata_onedisk_beta3}
DATA_DIR=${DATA_DIR:-data_jarvis/data_agama_ibata_onedisk_beta3_hydrabflow}
SEED=${SEED:-2026}
N_WORKERS=${N_WORKERS:-60}
N_FULL=${N_FULL:-300000}
N_GROUPS=${N_GROUPS:-333}
RUN_PILOT=${RUN_PILOT:-1}
PPC_DIR="${DATA_DIR}/ppc"

echo ">>> Ibata onedisk/beta3 dataset generation | sim=${SIM} data_dir=${DATA_DIR} seed=${SEED} workers=${N_WORKERS}"

# --------------------------------------------------------------------------------------------- #
# STEP 0 (optional): fast PILOT batch + prior-predictive check on ALL observables.
# ~2000 spray rows through the rejection prior — a few minutes — so misspecified priors/potential
# show up before the long run. Inspect ${PPC_DIR}/pilot/*.png, then let the full run proceed.
# --------------------------------------------------------------------------------------------- #
if [[ "${RUN_PILOT}" == "1" ]]; then
  echo ">>> [pilot] 2000 flat + 30 multistream for the pre-flight PPC"
  uv run python scripts/simulate.py \
    simulator=${SIM} composition=global \
    data.data_dir="${DATA_DIR}/pilot" data.n_simulations=2000 data.chunk_size=2000 \
    simulator.params.n_workers="${N_WORKERS}" seed="${SEED}"
  uv run python scripts/simulate_multistream.py \
    simulator=${SIM} composition=global \
    data.data_dir="${DATA_DIR}/pilot" data.n_simulations=30 data.chunk_size=30 \
    data.dataset_name=test_multistream_30.npz \
    simulator.params.n_workers="${N_WORKERS}" seed="${SEED}"

  echo ">>> [pilot] prior-predictive checks"
  uv run python scripts/ppc_ancillary_observables.py \
    "${DATA_DIR}/pilot/training_data_2000.npz" "${PPC_DIR}/pilot" \
    --sim-multistream "${DATA_DIR}/pilot/test_multistream_30.npz"
  uv run python scripts/ppc_prior_predictive.py \
    "${DATA_DIR}/pilot/training_data_2000.npz" "${PPC_DIR}/pilot" || true
  echo ">>> [pilot] PPC figures in ${PPC_DIR}/pilot  — inspect before the full run continues."
fi

# --------------------------------------------------------------------------------------------- #
# STEP 1: full 10^5 flat training set.
# --------------------------------------------------------------------------------------------- #
echo ">>> [full] ${N_FULL} flat particle-spray rows"
uv run python scripts/simulate.py \
  simulator=${SIM} composition=global \
  data.data_dir="${DATA_DIR}" data.n_simulations="${N_FULL}" data.chunk_size=5000 \
  simulator.params.n_workers="${N_WORKERS}" seed="${SEED}"

# --------------------------------------------------------------------------------------------- #
# STEP 2: 333-group multistream test set (one shared potential per group, 3 streams each).
# --------------------------------------------------------------------------------------------- #
echo ">>> [full] ${N_GROUPS}-group multistream test set"
uv run python scripts/simulate_multistream.py \
  simulator=${SIM} composition=global \
  data.data_dir="${DATA_DIR}" data.n_simulations="${N_GROUPS}" data.chunk_size="${N_GROUPS}" \
  data.dataset_name=test_multistream_${N_GROUPS}.npz \
  simulator.params.n_workers="${N_WORKERS}" seed="${SEED}"

# --------------------------------------------------------------------------------------------- #
# STEP 3: prior-predictive checks on the full sets (all observations).
# --------------------------------------------------------------------------------------------- #
echo ">>> [full] prior-predictive checks -> ${PPC_DIR}/full"
uv run python scripts/ppc_ancillary_observables.py \
  "${DATA_DIR}/training_data_${N_FULL}.npz" "${PPC_DIR}/full" \
  --sim-multistream "${DATA_DIR}/test_multistream_${N_GROUPS}.npz"
uv run python scripts/ppc_prior_predictive.py \
  "${DATA_DIR}/training_data_${N_FULL}.npz" "${PPC_DIR}/full" || true

echo ">>> DONE."
echo "    training set : ${DATA_DIR}/training_data_${N_FULL}.npz"
echo "    test set     : ${DATA_DIR}/test_multistream_${N_GROUPS}.npz"
echo "    PPC figures  : ${PPC_DIR}/full/"

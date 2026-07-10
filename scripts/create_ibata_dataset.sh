#!/usr/bin/env bash
# Create the Ibata-model particle-spray stream datasets (see new_constrains.md + CLAUDE.md):
#   full potential = fixed bulge + fixed HI/H2 gas disks + truncated (r_t=1000 kpc) triaxial halo
#   + free thin + free thick stellar disks; Chen+2024 spray; extended Zhou u Huang rotation-curve
#   observable + banded rejection prior; freed halo outer slope beta; and the three new ancillary
#   potential observables (HI terminal velocity, Sigma(1.1 kpc), vertical stellar density rho(z)).
#
#   1. 10^5 flat particle-spray training set -> training_data_100000.npz
#   2. 333-group multistream test set        -> test_multistream_333.npz
#
# CPU/joblib only — NO GPU (agama). n_workers=24 lives in the simulator config (override below if
# your box has more cores). Both stages are resumable (per-chunk checkpoints in a .chunks/ sidecar),
# so it is safe to Ctrl-C and re-run. A fast PILOT batch + prior-predictive check runs FIRST so you
# can eyeball all observations before committing to the multi-hour full run (set RUN_PILOT=0 to skip).
#
# VCIRC REJECTION PRIOR (active): stream_agama_ibata inherits the banded vcirc_rejection from
# stream_agama_spray_huang, so both stages reject every prior draw whose MODEL rotation curve fails
#   - Zhou band  (5.5 < r <= 24 kpc): median |vc - vc_Zhou| / vc_Zhou   < 0.20   (fractional)
#   - Huang band (r > 24 kpc):        median |vc - vc_Huang| / sigma_Huang < 2.0  (sigma-based)
# BEFORE the expensive stream integrator runs (adaptive batch, aborts only after 5M draws). The new
# gas + thick disks shift the inner curve, so the acceptance rate may differ from the spray_huang
# model — the pilot batch surfaces this early. To disable the cut (e.g. a quick test), append
# `~simulator.params.vcirc_rejection` to a simulate command below.
#
# Run it yourself:   bash scripts/create_ibata_dataset.sh
set -euo pipefail
cd "$(dirname "$0")/.."

SIM=stream_agama_ibata
DATA_DIR=${DATA_DIR:-data_jarvis/data_agama_ibata_hydrabflow}
SEED=${SEED:-2026}
N_WORKERS=${N_WORKERS:-30}
N_FULL=${N_FULL:-100000}
N_GROUPS=${N_GROUPS:-333}
RUN_PILOT=${RUN_PILOT:-1}
PPC_DIR="${DATA_DIR}/ppc"

echo ">>> Ibata dataset generation | sim=${SIM} data_dir=${DATA_DIR} seed=${SEED} workers=${N_WORKERS}"

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
  # New: HI terminal velocity + Sigma(1.1) + rho(z) shape + per-stream summary-statistic tracks.
  uv run python scripts/ppc_ancillary_observables.py \
    "${DATA_DIR}/pilot/training_data_2000.npz" "${PPC_DIR}/pilot" \
    --sim-multistream "${DATA_DIR}/pilot/test_multistream_30.npz"
  # Existing: rotation curve band + stream sky/PM loci vs real Gaia.
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
echo "    PPC figures  : ${PPC_DIR}/full/  (ppc_ancillary_observables.png,"
echo "                   ppc_stream_summary_statistics.png, ppc_rotation_curve.png, ppc_streams.png)"
echo
echo "Next (training, GPU): run  bash scripts/train_ibata.sh"

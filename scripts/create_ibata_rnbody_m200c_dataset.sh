#!/usr/bin/env bash
# Create the RESTRICTED N-BODY Ibata single-disk / beta=3 / (M200, c_v') datasets, config
#   conf/simulator/stream_agama_rnbody_ibata_onedisk_beta3_m200c.yaml
#
# Identical to create_ibata_m200c_dataset.sh in every respect (same m200_c halo priors, same
# banded Zhou u Huang rotation-curve REJECTION PRIOR screening every draw, same extended 50-radii
# vcirc observable, same ancillary observables v_term / Sigma_z / rho_z, gamma capped [-2,1.5],
# Sigma_Disk ceiling 3e9, derived rho/a stored per row) EXCEPT:
#   * the stream forward model is RESTRICTED N-BODY (self-consistent Plummer progenitor +
#     periodically refit moving progenitor potential) instead of Chen particle spray —
#     t_end = 4 Gyr for all three streams (NGC3201/M68 raised per the 2026-07-05 PPC finding);
#   * default size is 10^4 rows (rnbody is ~3-5x slower per row than spray).
#
# Purpose: test whether self-consistent stripping produces streams whose per-bin summary-statistic
# dispersion matches the real Gaia streams — the spray streams look TOO COLD on most tracks (see
# the cold-stream table/figure printed by scripts/ppc_summary_statistics.py, now emitted as
# ppc_stream_summary_statistics_std.png by the PPC step below).
#
# Products:
#   1. 10^4 flat rnbody training set   -> training_data_10000.npz
#   2. 333-group multistream test set  -> test_multistream_333.npz
#   3. PPC figures (all observables + summary tracks + PER-BIN STD cold-stream check) + a corner
#      of the effective prior after the vcirc cut: prior_after_cut_corner.png
#
# CPU/joblib only — NO GPU. Resumable (per-chunk checkpoints). A fast PILOT batch + PPC runs FIRST
# (set RUN_PILOT=0 to skip once inspected).
#
# Run it yourself:   bash scripts/create_ibata_rnbody_m200c_dataset.sh
# Training/tuning afterwards on the GPU box: scp the dataset dir under data/data_jarvis/ and use
#   scripts/train_ibata_m200c.sh / scripts/tune_ibata_m200c.sh with
#   SIM=stream_agama_rnbody_ibata_onedisk_beta3_m200c DATA_DIR=data/data_jarvis/<this dataset dir>.
set -euo pipefail
cd "$(dirname "$0")/.."

# Dataset generation is CPU-only (AGAMA/joblib) — never touch a GPU. This stops every joblib
# worker from probing for GPUs via autocvd; AGAMA's C-level chatter is silenced.
export HYDRABFLOW_NUM_GPUS=${HYDRABFLOW_NUM_GPUS:-0}
export HYDRABFLOW_SIM_QUIET=${HYDRABFLOW_SIM_QUIET:-1}

SIM=${SIM:-stream_agama_rnbody_ibata_onedisk_beta3_m200c}
DATA_DIR=${DATA_DIR:-data_jarvis/data_agama_rnbody_ibata_onedisk_beta3_m200c_hydrabflow}
SEED=${SEED:-2026}
N_WORKERS=${N_WORKERS:-180}
N_FULL=${N_FULL:-10000}
N_GROUPS=${N_GROUPS:-333}
RUN_PILOT=${RUN_PILOT:-1}
PPC_DIR="${DATA_DIR}/ppc"

# Inferred globals + the derived (rho, a) AGAMA received — the columns the "prior after the cut"
# corner plots. rho/a themselves are identity constants (auto-skipped); their *_derived twins vary.
CORNER_PARAMS=(
  log10_M200_TwoPowerTriaxial_halo ln_cvprime_TwoPowerTriaxial_halo
  gamma_TwoPowerTriaxial_halo q_TwoPowerTriaxial_halo
  r_Disk z_Disk Sigma_Disk
  rho_TwoPowerTriaxial_halo_derived a_TwoPowerTriaxial_halo_derived
)

echo ">>> rnbody Ibata m200_c dataset generation | sim=${SIM} data_dir=${DATA_DIR} seed=${SEED} workers=${N_WORKERS}"

# --------------------------------------------------------------------------------------------- #
# STEP 0 (optional): fast PILOT batch + PPC (all observables + cold-stream std check) + corner.
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

  echo ">>> [pilot] prior-predictive checks (incl. per-bin std cold-stream check) + prior corner"
  uv run python scripts/ppc_ancillary_observables.py \
    "${DATA_DIR}/pilot/training_data_2000.npz" "${PPC_DIR}/pilot" \
    --sim-multistream "${DATA_DIR}/pilot/test_multistream_30.npz"
  uv run python scripts/ppc_prior_predictive.py \
    "${DATA_DIR}/pilot/training_data_2000.npz" "${PPC_DIR}/pilot" || true
  uv run python scripts/corner_parameters.py \
    "${DATA_DIR}/pilot/training_data_2000.npz" "${PPC_DIR}/pilot" \
    --params "${CORNER_PARAMS[@]}" --name prior_after_cut_corner.png \
    --title "Effective prior after vcirc cut (rnbody m200_c, pilot n=2000)" || true
  echo ">>> [pilot] PPC + corner figures in ${PPC_DIR}/pilot — inspect before the full run continues."
fi

# --------------------------------------------------------------------------------------------- #
# STEP 1: full 10^4 flat rnbody training set (the vcirc rejection prior is applied per draw).
# --------------------------------------------------------------------------------------------- #
echo ">>> [full] ${N_FULL} flat restricted-N-body rows"
uv run python scripts/simulate.py \
  simulator=${SIM} composition=global \
  data.data_dir="${DATA_DIR}" data.n_simulations="${N_FULL}" data.chunk_size=2000 \
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
# STEP 3: PPC (all observables + cold-stream std check) + prior-after-cut corner on the full set.
# --------------------------------------------------------------------------------------------- #
echo ">>> [full] prior-predictive checks + prior-after-cut corner -> ${PPC_DIR}/full"
uv run python scripts/ppc_ancillary_observables.py \
  "${DATA_DIR}/training_data_${N_FULL}.npz" "${PPC_DIR}/full" \
  --sim-multistream "${DATA_DIR}/test_multistream_${N_GROUPS}.npz"
uv run python scripts/ppc_prior_predictive.py \
  "${DATA_DIR}/training_data_${N_FULL}.npz" "${PPC_DIR}/full" || true
uv run python scripts/corner_parameters.py \
  "${DATA_DIR}/training_data_${N_FULL}.npz" "${PPC_DIR}/full" \
  --params "${CORNER_PARAMS[@]}" --name prior_after_cut_corner.png \
  --title "Effective prior after vcirc cut (rnbody m200_c, n=${N_FULL})" || true

echo ">>> DONE."
echo "    training set : ${DATA_DIR}/training_data_${N_FULL}.npz"
echo "    test set     : ${DATA_DIR}/test_multistream_${N_GROUPS}.npz"
echo "    PPC + corner : ${PPC_DIR}/full/ (cold-stream check: ppc_stream_summary_statistics_std.png)"

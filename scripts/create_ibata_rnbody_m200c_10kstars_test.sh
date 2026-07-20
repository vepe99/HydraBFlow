#!/usr/bin/env bash
# TEST run of the RESTRICTED N-BODY Ibata single-disk / beta=3 / (M200, c_v') simulator with a
# HIGH-RESOLUTION progenitor: 1e4 stars per stream (n_particles=10000) instead of the default 1000.
# Based on create_ibata_rnbody_m200c_dataset.sh (same m200_c halo priors, same banded Zhou u Huang
# rotation-curve REJECTION PRIOR screening every draw, same extended 50-radii vcirc observable, same
# ancillary observables v_term / Sigma_z / rho_z, gamma capped [-2,1.5], Sigma_Disk ceiling 3e9,
# derived rho/a stored per row, t_end=4 Gyr for all three streams) EXCEPT:
#   * n_particles = 10000 (10x the default) — resolve the streams more finely so the per-bin
#     summary-statistic DISPERSION (track std) is not resolution-limited;
#   * only 1e3 flat simulations (this is a TEST to see whether the streams LOOK BETTER — i.e. whether
#     the higher star count fixes the "streams look too cold" prior-predictive-check problem — NOT a
#     training set).
#
# Purpose: check whether pushing the restricted N-body progenitor to 1e4 stars produces streams whose
# per-bin summary-statistic dispersion matches the real Gaia streams (the cold-stream PPC problem).
# Inspect ppc_stream_summary_statistics_std.png in the PPC dir.
#
# CPU/joblib only — NO GPU. Resumable (per-chunk checkpoints). rnbody at 1e4 stars is ~10x slower
# per row than at 1e3, so 1000 rows is deliberately small.
#
# Run it yourself:   bash scripts/create_ibata_rnbody_m200c_10kstars_test.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# Dataset generation is CPU-only (AGAMA/joblib) — never touch a GPU. This stops every joblib
# worker from probing for GPUs via autocvd; AGAMA's C-level chatter is silenced.
export HYDRABFLOW_NUM_GPUS=${HYDRABFLOW_NUM_GPUS:-0}
export HYDRABFLOW_SIM_QUIET=${HYDRABFLOW_SIM_QUIET:-1}

SIM=${SIM:-stream_agama_rnbody_ibata_onedisk_beta3_m200c}
DATA_DIR=${DATA_DIR:-data_jarvis/data_agama_rnbody_ibata_onedisk_beta3_m200c_10kstars_hydrabflow}
SEED=${SEED:-2026}
N_WORKERS=${N_WORKERS:-180}
N_PARTICLES=${N_PARTICLES:-10000}   # stars per stream (default sim value is 1000)
N_FULL=${N_FULL:-1000}
N_GROUPS=${N_GROUPS:-30}             # small multistream test set for the summary-track PPC
CHUNK=${CHUNK:-250}                  # per-chunk checkpoints (resumable)
PPC_DIR="${DATA_DIR}/ppc"

# Inferred globals + the derived (rho, a) AGAMA received — the columns the "prior after the cut"
# corner plots. rho/a themselves are identity constants (auto-skipped); their *_derived twins vary.
CORNER_PARAMS=(
  log10_M200_TwoPowerTriaxial_halo ln_cvprime_TwoPowerTriaxial_halo
  gamma_TwoPowerTriaxial_halo q_TwoPowerTriaxial_halo
  r_Disk z_Disk Sigma_Disk
  rho_TwoPowerTriaxial_halo_derived a_TwoPowerTriaxial_halo_derived
)

echo ">>> rnbody Ibata m200_c 10k-star TEST | sim=${SIM} data_dir=${DATA_DIR} seed=${SEED}"
echo "    n_particles=${N_PARTICLES} n_full=${N_FULL} n_groups=${N_GROUPS} workers=${N_WORKERS}"

# --------------------------------------------------------------------------------------------- #
# STEP 1: 1e3 flat restricted-N-body rows at n_particles=10000 (vcirc rejection prior per draw).
# --------------------------------------------------------------------------------------------- #
echo ">>> [flat] ${N_FULL} rows, ${N_PARTICLES} stars/stream"
uv run python scripts/simulate.py \
  simulator=${SIM} composition=global \
  simulator.params.n_particles="${N_PARTICLES}" \
  data.data_dir="${DATA_DIR}" data.n_simulations="${N_FULL}" data.chunk_size="${CHUNK}" \
  simulator.params.n_workers="${N_WORKERS}" seed="${SEED}"

# --------------------------------------------------------------------------------------------- #
# STEP 2: small multistream test set (one shared potential per group, 3 streams each) for the
#         per-stream summary-statistic tracks used by the cold-stream PPC.
# --------------------------------------------------------------------------------------------- #
echo ">>> [multistream] ${N_GROUPS} groups, ${N_PARTICLES} stars/stream"
uv run python scripts/simulate_multistream.py \
  simulator=${SIM} composition=global \
  simulator.params.n_particles="${N_PARTICLES}" \
  data.data_dir="${DATA_DIR}" data.n_simulations="${N_GROUPS}" data.chunk_size="${N_GROUPS}" \
  data.dataset_name=test_multistream_${N_GROUPS}.npz \
  simulator.params.n_workers="${N_WORKERS}" seed="${SEED}"

# --------------------------------------------------------------------------------------------- #
# STEP 3: PPC (all observables + PER-BIN STD cold-stream check) + prior-after-cut corner.
# --------------------------------------------------------------------------------------------- #
echo ">>> [ppc] prior-predictive checks + prior-after-cut corner -> ${PPC_DIR}"
uv run python scripts/ppc_ancillary_observables.py \
  "${DATA_DIR}/training_data_${N_FULL}.npz" "${PPC_DIR}" \
  --sim-multistream "${DATA_DIR}/test_multistream_${N_GROUPS}.npz"
uv run python scripts/ppc_prior_predictive.py \
  "${DATA_DIR}/training_data_${N_FULL}.npz" "${PPC_DIR}" || true
uv run python scripts/corner_parameters.py \
  "${DATA_DIR}/training_data_${N_FULL}.npz" "${PPC_DIR}" \
  --params "${CORNER_PARAMS[@]}" --name prior_after_cut_corner.png \
  --title "Effective prior after vcirc cut (rnbody m200_c, ${N_PARTICLES} stars, n=${N_FULL})" || true

echo ">>> DONE."
echo "    flat set   : ${DATA_DIR}/training_data_${N_FULL}.npz  (${N_PARTICLES} stars/stream)"
echo "    test set   : ${DATA_DIR}/test_multistream_${N_GROUPS}.npz"
echo "    PPC        : ${PPC_DIR}/ (cold-stream check: ppc_stream_summary_statistics_std.png)"

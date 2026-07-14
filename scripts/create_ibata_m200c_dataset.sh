#!/usr/bin/env bash
# Create the Ibata single-disk / beta=3 particle-spray datasets with the DARK HALO REPARAMETRIZED
# BY VIRIAL MASS + CONCENTRATION (McMillan 2017), config
#   conf/simulator/stream_agama_ibata_onedisk_beta3_m200c.yaml
#
# Identical to create_ibata_onedisk_beta3_dataset.sh in every respect (Chen+2024 spray; extended
# Zhou u Huang rotation curve + BANDED REJECTION PRIOR — the circular-velocity cut IS applied here,
# it lives in the simulator's vcirc_rejection.bands and screens every prior draw; ancillary
# potential observables v_term / Sigma_z / rho_z; truncated r_t=1000 kpc halo; exponential stellar
# disk; fixed McMillan bulge + HI/H2 gas disks; single free thin disk) EXCEPT the halo prior
# COORDINATES:
#   * the INFERRED halo globals are  log10_M200_TwoPowerTriaxial_halo ~ U[11.699, 12.398]
#     (M200 log-uniform [0.5, 2.5]e12 Msun) and ln_cvprime_TwoPowerTriaxial_halo ~ N(2.56, 0.272)
#     (McMillan c_v' prior at Delta_c ~ 94), both sampled IN LOG SPACE (already log for inference);
#   * gamma is capped at [-2, 1.5]; beta is fixed at 3.0;
#   * the old rho / a (densityNorm / scaleRadius) become UNUSED identity constants and are NOT
#     inferred. Per row the simulator converts (M200, c_v', gamma) -> the (densityNorm, scaleRadius)
#     it hands to AGAMA and STORES THEM in the .npz under
#     rho_TwoPowerTriaxial_halo_derived / a_TwoPowerTriaxial_halo_derived (diagnostics only — the
#     network never sees them; inference is on log10_M200 / ln_cvprime).
#
# Products:
#   1. 10^5 flat particle-spray training set -> training_data_100000.npz
#   2. 333-group multistream test set        -> test_multistream_333.npz
#   3. PPC figures (all observables) + a CORNER PLOT of the effective prior AFTER the vcirc cut
#      over all inferred globals + the derived (rho, a): prior_after_cut_corner.png
#
# CPU/joblib only — NO GPU (agama). Both stages are resumable (per-chunk checkpoints in a .chunks/
# sidecar), so it is safe to Ctrl-C and re-run. A fast PILOT batch + PPC + prior corner runs FIRST
# so you can eyeball everything before the multi-hour full run (set RUN_PILOT=0 to skip).
#
# Run it yourself:   bash scripts/create_ibata_m200c_dataset.sh
# Then scp the dataset dir to the GPU box under data/data_jarvis/ and run
#   scripts/train_ibata_m200c.sh   (train + sim/real eval)   and
#   scripts/tune_ibata_m200c.sh    (Optuna tuning).
set -euo pipefail
cd "$(dirname "$0")/.."

# Dataset generation is CPU-only (AGAMA/joblib) — never touch a GPU. This stops every joblib worker
# from probing for GPUs via autocvd; AGAMA's C-level chatter is silenced by HYDRABFLOW_SIM_QUIET.
export HYDRABFLOW_NUM_GPUS=${HYDRABFLOW_NUM_GPUS:-0}
export HYDRABFLOW_SIM_QUIET=${HYDRABFLOW_SIM_QUIET:-1}

SIM=${SIM:-stream_agama_ibata_onedisk_beta3_m200c}
DATA_DIR=${DATA_DIR:-data_jarvis/data_agama_ibata_onedisk_beta3_m200c_hydrabflow}
SEED=${SEED:-2026}
N_WORKERS=${N_WORKERS:-60}
N_FULL=${N_FULL:-100000}
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

echo ">>> Ibata m200_c dataset generation | sim=${SIM} data_dir=${DATA_DIR} seed=${SEED} workers=${N_WORKERS}"

# --------------------------------------------------------------------------------------------- #
# STEP 0 (optional): fast PILOT batch + PPC (all observables) + prior corner after the vcirc cut.
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

  echo ">>> [pilot] prior-predictive checks + prior-after-cut corner"
  uv run python scripts/ppc_ancillary_observables.py \
    "${DATA_DIR}/pilot/training_data_2000.npz" "${PPC_DIR}/pilot" \
    --sim-multistream "${DATA_DIR}/pilot/test_multistream_30.npz"
  uv run python scripts/ppc_prior_predictive.py \
    "${DATA_DIR}/pilot/training_data_2000.npz" "${PPC_DIR}/pilot" || true
  uv run python scripts/corner_parameters.py \
    "${DATA_DIR}/pilot/training_data_2000.npz" "${PPC_DIR}/pilot" \
    --params "${CORNER_PARAMS[@]}" --name prior_after_cut_corner.png \
    --title "Effective prior after vcirc cut (m200_c, pilot n=2000)" || true
  echo ">>> [pilot] PPC + corner figures in ${PPC_DIR}/pilot — inspect before the full run continues."
fi

# --------------------------------------------------------------------------------------------- #
# STEP 1: full 10^5 flat training set (the vcirc rejection prior is applied per draw).
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
# STEP 3: PPC (all observations) + prior-after-cut corner on the full training set.
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
  --title "Effective prior after vcirc cut (m200_c, n=${N_FULL})" || true

echo ">>> DONE."
echo "    training set : ${DATA_DIR}/training_data_${N_FULL}.npz"
echo "    test set     : ${DATA_DIR}/test_multistream_${N_GROUPS}.npz"
echo "    PPC + corner : ${PPC_DIR}/full/  (prior_after_cut_corner.png)"
echo "    scp ${DATA_DIR} to the GPU box under data/data_jarvis/, then run"
echo "    scripts/train_ibata_m200c.sh and scripts/tune_ibata_m200c.sh."

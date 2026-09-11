#!/usr/bin/env bash
# Dataset generation for the v2 restricted-N-body Ibata / m200_c model.
#
# What is new relative to create_ibata_rnbody_m200c_dataset.sh (see the two simulator configs'
# headers for the full reasoning and the literature behind every number):
#
#   1. CORRECTED PROGENITORS (stream_agama_rnbody_ibata_m200c_prog): the inherited m_progenitor values
#      were PRESENT-DAY masses being fed as the INITIAL mass at t = -t_end, under-massing the
#      progenitors ~3x against modern present-day values and ~9-16x against literature initial masses;
#      and Pal 5's a_progenitor was a King CORE radius used as a Plummer scale radius (8.43 vs ~21 pc).
#      Both are corrected and freed.
#   2. NO ROTATION-CURVE REJECTION PRIOR: the curve is already an observable, so the cut was
#      double-counting it and truncating the halo-shape prior. Prior sampling is now free.
#   3. FREED HALO/BULGE KNOBS: alpha (transition sharpness), p (axisRatioY) and a tilt angle, all
#      previously hardcoded, plus a ~10% bulge amplitude. These give the model legitimate routes to an
#      inner-steep / outer-declining halo and a non-axisymmetric inner halo, instead of forcing gamma
#      to the edge of its prior.
#   4. FREED STRIPPING AGE: t_end ~ U[2, 10] Gyr for all three streams (a nuisance at
#      composition=global).
#
# NOT included: member contamination. A `contaminate_members` augmentation exists and sits in the v2
# augmentation chain, but with contamination_max_frac=0.0 it is an exact no-op — the training set is
# generated and trained WITHOUT it, by decision. Being a per-batch augmentation it never affected the
# stored npz in any case; enabling it later needs only
# `+augmentation.params.contamination_max_frac=0.25` at train time, no regeneration.
#
# Also new in the output npz: `m_bound_final`, the present-day BOUND mass of each progenitor remnant.
# This is a diagnostic (the adapter drops it), but an important one — a row whose cluster dissolved
# entirely cannot describe the streams we observe, and MEASURED SURVIVAL RATES ARE LOW: on a 96-row
# probe, 41.7% of rows left a surviving remnant with the rotation-curve cut, and only 19.8% without
# it. So expect ~80% of v2 rows to be uninformative about stream morphology.
#
# CAVEAT when reading that number: the bound-mass estimate is RESOLUTION-DEPENDENT, because the
# progenitor's self-gravity is a Multipole refit from the particles themselves. The probes above used
# n_particles=400 (19.8%) and a 150-particle smoke run gave 8.3% — so only trust survival fractions
# from a run at the real n_particles (1000 by default), and never compare across resolutions.
#
#   >>> READ THE PILOT REPORT BEFORE LETTING THE FULL RUN PROCEED. <<<
# If the survival fraction is as low as the probe suggests, the efficient fix is a survival screen
# (keep rows whose remnant mass is within a factor of the observed cluster mass). Like the
# rotation-curve cut it is a hard indicator, so the analytic compositional prior score stays valid —
# but it is a *stream-based* criterion rather than a second use of the rotation curve. It is NOT
# enabled here because it is a prior truncation and that was explicitly what we set out to remove;
# it is a deliberate choice for you to make on the pilot numbers.
#
# CPU/joblib only — NO GPU. Resumable (per-chunk checkpoints). A fast PILOT batch + PPC runs FIRST
# (set RUN_PILOT=0 to skip once inspected).
#
# Run it yourself:   bash scripts/create_ibata_rnbody_m200c_v2_dataset.sh
# TRAINING AND VALIDATION ARE DELIBERATELY NOT PART OF THIS SCRIPT — they run on the other cluster.
# The commands are printed at the end; scp the dataset dir under data/data_jarvis/ first.
set -euo pipefail
cd "$(dirname "$0")/.."

# Dataset generation is CPU-only (AGAMA/joblib) — never touch a GPU. This stops every joblib
# worker from probing for GPUs via autocvd; AGAMA's C-level chatter is silenced.
export HYDRABFLOW_NUM_GPUS=${HYDRABFLOW_NUM_GPUS:-0}
export HYDRABFLOW_SIM_QUIET=${HYDRABFLOW_SIM_QUIET:-1}

SIM=${SIM:-stream_agama_rnbody_ibata_m200c_v2}
DATA_DIR=${DATA_DIR:-data_jarvis/data_agama_rnbody_ibata_m200c_v2_hydrabflow}
SEED=${SEED:-2026}
N_WORKERS=${N_WORKERS:-180}
N_FULL=${N_FULL:-10000}
N_GROUPS=${N_GROUPS:-333}
N_PILOT=${N_PILOT:-2000}
N_PILOT_GROUPS=${N_PILOT_GROUPS:-30}
RUN_PILOT=${RUN_PILOT:-1}
# Stars per stream; the simulator default is 1000. Resolution matters in BOTH directions here, so
# this is a real knob, not only a smoke-test lever:
#   * RAISING it (e.g. 10000) better resolves the progenitor's self-gravity — which is a Multipole
#     refit from the particles themselves, so a sparse progenitor gets a noisy, weak potential and
#     dissolves too easily (measured remnant survival rose 19.8% -> 23.7% going from 400 -> 1000) —
#     and it supplies enough in-window stars to fill `observed_n_stars` and the phi1 bins, which is
#     the occupancy confound behind the per-bin dispersion comparison.
#   * LOWERING it is only for smoke tests; every per-bin statistic becomes occupancy-limited and
#     survival fractions from different resolutions must never be compared.
N_PARTICLES=${N_PARTICLES:-}
PARTICLE_OVERRIDE=()
if [[ -n "${N_PARTICLES}" ]]; then
  PARTICLE_OVERRIDE=(simulator.params.n_particles="${N_PARTICLES}")
  echo ">>> n_particles overridden to ${N_PARTICLES} (default 1000; see the note above — do not"
  echo "    compare survival fractions or per-bin dispersions across resolutions)"
fi
PPC_DIR="${DATA_DIR}/ppc"

# Inferred globals + the derived (rho, a) AGAMA received. v2 adds alpha / p / tilt / rho_Bulge.
# Overridable (space-separated in CORNER_PARAMS_LIST) so a variant that pins some of these — e.g. the
# NFW halo, which fixes gamma and alpha — does not ask the corner plot for constant columns.
CORNER_PARAMS_LIST=${CORNER_PARAMS_LIST:-"\
log10_M200_TwoPowerTriaxial_halo ln_cvprime_TwoPowerTriaxial_halo \
gamma_TwoPowerTriaxial_halo q_TwoPowerTriaxial_halo \
alpha_TwoPowerTriaxial_halo p_TwoPowerTriaxial_halo tilt_TwoPowerTriaxial_halo \
rho_Bulge r_Disk z_Disk Sigma_Disk \
rho_TwoPowerTriaxial_halo_derived a_TwoPowerTriaxial_halo_derived"}
read -r -a CORNER_PARAMS <<< "${CORNER_PARAMS_LIST}"
LABEL=${LABEL:-"v2 prior (no vcirc cut, rnbody m200_c"}
# Training augmentation chain, used by the noise-convolved cold-stream check so it mimics the
# observation model the network will actually be trained under.
AUG_PRESET=${AUG_PRESET:-stream_global_ibata_grid_v2}

echo ">>> rnbody Ibata m200_c dataset | sim=${SIM} data_dir=${DATA_DIR} seed=${SEED} workers=${N_WORKERS}"

# Report the progenitor survival + NaN rates of a generated npz: the two numbers that decide whether
# the run is worth continuing at full scale.
report_survival () {
  uv run python - "$1" <<'PY'
import sys
import numpy as np

d = np.load(sys.argv[1])
proj = d["sim_data_projected"]
nan_rows = np.isnan(proj).any(axis=tuple(range(1, proj.ndim)))
print(f"    rows {len(nan_rows)}   NaN rows {100 * nan_rows.mean():.1f}%")
if "m_bound_final" in d:
    mb = np.asarray(d["m_bound_final"], float).reshape(-1)
    m0 = np.asarray(d["m_progenitor"], float).reshape(-1)
    ok = np.isfinite(mb)
    surv = ok & (mb > 0)
    print(f"    progenitor remnant survives: {100 * surv.mean():.1f}%   (m_bound NaN {100 * (~ok).mean():.1f}%)")
    if surv.any():
        frac = mb[surv] / m0[surv]
        q = np.percentile(frac, [10, 50, 90])
        print(f"    retained mass fraction among survivors: p10 {q[0]:.2f}  median {q[1]:.2f}  p90 {q[2]:.2f}")
        print("    -> if survival is low, consider a survival screen; see this script's header.")
else:
    print("    (no m_bound_final in this npz — not a restricted-N-body run?)")
PY
}

# --------------------------------------------------------------------------------------------- #
# STEP 0 (optional): fast PILOT batch + PPC + corner + the survival report.
# --------------------------------------------------------------------------------------------- #
if [[ "${RUN_PILOT}" == "1" ]]; then
  echo ">>> [pilot] ${N_PILOT} flat + ${N_PILOT_GROUPS} multistream for the pre-flight PPC"
  uv run python -m hydrabflow.pipeline.simulate \
    simulator=${SIM} composition=global \
    data.data_dir="${DATA_DIR}/pilot" data.n_simulations="${N_PILOT}" data.chunk_size="${N_PILOT}" \
    simulator.params.n_workers="${N_WORKERS}" seed="${SEED}" "${PARTICLE_OVERRIDE[@]}"
  uv run python -m hydrabflow.pipeline.simulate_multistream \
    simulator=${SIM} composition=global \
    data.data_dir="${DATA_DIR}/pilot" data.n_simulations="${N_PILOT_GROUPS}" data.chunk_size="${N_PILOT_GROUPS}" \
    data.dataset_name=test_multistream_${N_PILOT_GROUPS}.npz \
    simulator.params.n_workers="${N_WORKERS}" seed="${SEED}" "${PARTICLE_OVERRIDE[@]}"

  echo ">>> [pilot] progenitor survival / NaN report"
  report_survival "${DATA_DIR}/pilot/training_data_${N_PILOT}.npz"

  echo ">>> [pilot] prior-predictive checks (incl. per-bin std cold-stream check) + prior corner"
  uv run python scripts/ppc_ancillary_observables.py \
    "${DATA_DIR}/pilot/training_data_${N_PILOT}.npz" "${PPC_DIR}/pilot" \
    --sim-multistream "${DATA_DIR}/pilot/test_multistream_${N_PILOT_GROUPS}.npz"
  # --noise convolves the sim with the TRAINING observation model before comparing to real Gaia; the
  # robust (MAD) column is the one to read, and the comparison is only fair with both. NOTE this
  # script takes NAMED arguments (--sim / --out), and --out is a FILE, not a directory — it was being
  # called positionally, so argparse rejected it and the `|| true` swallowed the failure, silently
  # leaving only the NOISELESS table from ppc_ancillary_observables above. --aug/--simulator must name
  # the v2 chain too, or the noise model is the wrong one.
  uv run python scripts/ppc_summary_statistics.py \
    --sim "${DATA_DIR}/pilot/test_multistream_${N_PILOT_GROUPS}.npz" \
    --out "${PPC_DIR}/pilot/ppc_stream_summary_statistics_noise.png" \
    --aug "${AUG_PRESET}" --simulator "${SIM}" --noise || true
  uv run python scripts/ppc_prior_predictive.py \
    "${DATA_DIR}/pilot/training_data_${N_PILOT}.npz" "${PPC_DIR}/pilot" || true
  uv run python scripts/corner_parameters.py \
    "${DATA_DIR}/pilot/training_data_${N_PILOT}.npz" "${PPC_DIR}/pilot" \
    --params "${CORNER_PARAMS[@]}" --name prior_corner.png \
    --title "${LABEL}, pilot n=${N_PILOT})" || true
  echo ">>> [pilot] figures in ${PPC_DIR}/pilot — INSPECT the survival report and the cold-stream"
  echo "    std table before the full run continues (set RUN_PILOT=0 to skip next time)."
fi

# --------------------------------------------------------------------------------------------- #
# STEP 1: full flat rnbody training set.
# --------------------------------------------------------------------------------------------- #
echo ">>> [full] ${N_FULL} flat restricted-N-body rows"
uv run python -m hydrabflow.pipeline.simulate \
  simulator=${SIM} composition=global \
  data.data_dir="${DATA_DIR}" data.n_simulations="${N_FULL}" data.chunk_size=2000 \
  simulator.params.n_workers="${N_WORKERS}" seed="${SEED}" "${PARTICLE_OVERRIDE[@]}"

# --------------------------------------------------------------------------------------------- #
# STEP 2: multistream test set (one shared potential per group, 3 streams each).
# --------------------------------------------------------------------------------------------- #
echo ">>> [full] ${N_GROUPS}-group multistream test set"
uv run python -m hydrabflow.pipeline.simulate_multistream \
  simulator=${SIM} composition=global \
  data.data_dir="${DATA_DIR}" data.n_simulations="${N_GROUPS}" data.chunk_size="${N_GROUPS}" \
  data.dataset_name=test_multistream_${N_GROUPS}.npz \
  simulator.params.n_workers="${N_WORKERS}" seed="${SEED}" "${PARTICLE_OVERRIDE[@]}"

# --------------------------------------------------------------------------------------------- #
# STEP 3: PPC + prior corner + survival report on the full set.
# --------------------------------------------------------------------------------------------- #
echo ">>> [full] progenitor survival / NaN report"
report_survival "${DATA_DIR}/training_data_${N_FULL}.npz"

echo ">>> [full] prior-predictive checks + prior corner -> ${PPC_DIR}/full"
uv run python scripts/ppc_ancillary_observables.py \
  "${DATA_DIR}/training_data_${N_FULL}.npz" "${PPC_DIR}/full" \
  --sim-multistream "${DATA_DIR}/test_multistream_${N_GROUPS}.npz"
uv run python scripts/ppc_summary_statistics.py \
  --sim "${DATA_DIR}/test_multistream_${N_GROUPS}.npz" \
  --out "${PPC_DIR}/full/ppc_stream_summary_statistics_noise.png" \
  --aug "${AUG_PRESET}" --simulator "${SIM}" --noise || true
uv run python scripts/ppc_prior_predictive.py \
  "${DATA_DIR}/training_data_${N_FULL}.npz" "${PPC_DIR}/full" || true
uv run python scripts/corner_parameters.py \
  "${DATA_DIR}/training_data_${N_FULL}.npz" "${PPC_DIR}/full" \
  --params "${CORNER_PARAMS[@]}" --name prior_corner.png \
  --title "${LABEL}, n=${N_FULL})" || true

echo ">>> DONE (dataset generation only)."
echo "    training set : ${DATA_DIR}/training_data_${N_FULL}.npz"
echo "    test set     : ${DATA_DIR}/test_multistream_${N_GROUPS}.npz"
echo "    PPC + corner : ${PPC_DIR}/full/ (cold-stream check: ppc_stream_summary_statistics_std.png)"
echo
echo "    Training / validation run on the OTHER cluster. After scp'ing the dataset dir under"
echo "    data/data_jarvis/, the v2 stack is:"
echo "      simulator=${SIM} \\"
echo "      model=stream_fusion_ibata_grid_masked \\"
echo "      adapter=stream_ibata_sumstats \\"
echo "      augmentation=stream_global_ibata_grid_v2 \\"
echo "      preprocessing=stream_global_log10_ibata_sumstats composition=global"
echo "    (real-data eval uses augmentation=stream_real_global_ibata_grid_v2; GPU runs pick a free"
echo "     device with autocvd.)"

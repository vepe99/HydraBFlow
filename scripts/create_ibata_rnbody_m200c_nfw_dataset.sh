#!/usr/bin/env bash
# Dataset generation for the NFW-halo restricted-N-body Ibata / m200_c model.
#
# Thin wrapper over create_ibata_rnbody_m200c_v2_dataset.sh: the pilot, the survival / NaN report, the
# PPCs and the resumable chunking are all identical — only the simulator config and the output dir
# differ. Read the v2 script's header for what the pipeline does and what to look for in the pilot.
#
# The one difference in the MODEL: the dark halo is a FLATTENED NFW instead of a free two-power
# family. AGAMA's Spheroid with (alpha, beta, gamma) = (1, 3, 1) IS NFW, and AGAMA's native `type=NFW`
# is spherical-only, so the flattening (axisRatioZ = q, plus axisRatioY = p and the tilt inherited
# from v2) requires the Spheroid form. Free halo parameters: (M200, c_v', q, p, tilt) — 9 global
# parameters in total vs v2's 11. Full rationale in conf/simulator/stream_agama_rnbody_ibata_m200c_nfw.yaml;
# the short version is that under m200_c the relation r_h = r200 / [c200 (2 - gamma)] makes gamma
# rescale the halo rather than reshape it, which is why its posterior railed, and at gamma = 1 that
# relation collapses to the textbook r_s = r200 / c200.
#
# Generate this ALONGSIDE the v2 set, not instead of it. Same progenitors, same nuisances, same
# observables, same seed => the two trained models are directly comparable, which is what tells you
# whether the free halo exponents bought real fit quality or only absorbed a parameterization artefact.
#
# Run it yourself:   bash scripts/create_ibata_rnbody_m200c_nfw_dataset.sh
set -euo pipefail
cd "$(dirname "$0")/.."

export SIM=${SIM:-stream_agama_rnbody_ibata_m200c_nfw}
export DATA_DIR=${DATA_DIR:-data_jarvis/data_agama_rnbody_ibata_m200c_nfw_hydrabflow}
export LABEL=${LABEL:-"NFW-halo prior (rnbody m200_c, gamma=alpha=1"}
# gamma and alpha are identity constants here, so drop them from the corner plot.
export CORNER_PARAMS_LIST=${CORNER_PARAMS_LIST:-"\
log10_M200_TwoPowerTriaxial_halo ln_cvprime_TwoPowerTriaxial_halo \
q_TwoPowerTriaxial_halo p_TwoPowerTriaxial_halo tilt_TwoPowerTriaxial_halo \
rho_Bulge r_Disk z_Disk Sigma_Disk \
rho_TwoPowerTriaxial_halo_derived a_TwoPowerTriaxial_halo_derived"}

exec bash scripts/create_ibata_rnbody_m200c_v2_dataset.sh "$@"

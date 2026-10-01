#!/usr/bin/env bash
# Trial-14 own-support 2-modal diffusion model with BOTH summary backbones frozen at the information-maximising
# single-modality runs (scripts/train_imm_streams_ownsupport.sh, ARM=streams and ARM=vcirc). Only the
# DiffusionTransformer is trained; everything else = scripts/train_spline_ownsupport_ou24vc.sh (train -> eval sim
# base+compositional -> eval real).
# Run (after both IMM runs finished):  bash scripts/train_frozen_imm_ownsupport_2modal.sh
# Knobs: IMM_STREAMS IMM_VCIRC (run dirs) + everything train_spline_ownsupport_ou24vc.sh takes.
set -euo pipefail
cd "$(dirname "$0")/.."

IMM_STREAMS=${IMM_STREAMS:-outputs/Bsline/imm_streams_ownsupport}
IMM_VCIRC=${IMM_VCIRC:-outputs/Bsline/imm_vcirc_ownsupport}
W_STREAMS="$(realpath "${IMM_STREAMS}/train/summary_network_weights.npz")"
W_VCIRC="$(realpath "${IMM_VCIRC}/train/summary_network_weights.npz")"

RUNS_DIR=${RUNS_DIR:-outputs/Bsline/spray_p1e3_v4_prog2026_ou24vc_ownsupport_2modal_trial14_frozenimm} \
EXTRA="++model.summary_network.params.frozen_weights.sim_summary=${W_STREAMS} ++model.summary_network.params.frozen_weights.vcirc_kms=${W_VCIRC} ${EXTRA:-}" \
bash scripts/train_spline_ownsupport_ou24vc.sh

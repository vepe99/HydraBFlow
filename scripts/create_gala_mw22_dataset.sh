#!/usr/bin/env bash
# Dataset generation for the gala particle-spray / MilkyWayPotential2022-style model
# (conf/simulator/stream_gala_spray_mw22.yaml — read its header for every physics decision):
#
#   0. PILOT   24 flat rows (+ a 6-group multistream) with timing / memory / NaN / in-window report
#              and the noise-convolved cold-stream figure. Inspect before letting the full run go
#              (RUN_PILOT=0 to skip once seen).
#   1. TEST    333-group compositional test set (seed 7)            -> $D/test_multistream_333.npz
#   2. TRAIN   10^4-row flat training set (seed 2026, chunks of 1000) -> $D/training_data_10000.npz
#   3. PPC     prior-predictive coverage vs the real Gaia streams in THREE representations, all
#              model-free and CPU-only, into $D/ppc/:
#                - summary TRACKS (PPC estimator)      ppc_summary_statistics.py --noise, ppc_summary_coverage.py
#                - the network's 14-channel GRID       ppc_summary_grid_coverage.py
#                - raw PARTICLES                        ppc_particle_coverage.py (new)
#              + a prior corner incl. the derived c_phi.
#   4. Print the two GPU training commands (particles arm / summary-statistics arm). NOT run here.
#
# CPU/joblib only (gala in the loky workers) — NO GPU. Resumable: io.run_chunked checkpoints every
# chunk, so a crashed/killed run resumes. Strict-overcommit watchdog (CLAUDE.md 2026-09-12): waits
# for commit headroom, sizes the worker pool to it, relaunches on failure. gala workers were measured
# at ~140 MB RSS and ~15 s/row (leapfrog, 1e4 stars, 5000 steps) -> ~1 h for 10^4 rows at 48 workers.
#
# Run:   bash scripts/create_gala_mw22_dataset.sh
# Env:   SIM D N_FULL N_GROUPS N_PILOT RUN_PILOT PILOT_ONLY MAX_WORKERS MIN_WORKERS MIN_HEADROOM_GB PY
set -euo pipefail
cd "$(dirname "$0")/.."

export HYDRABFLOW_NUM_GPUS=${HYDRABFLOW_NUM_GPUS:-0}
export HYDRABFLOW_SIM_QUIET=${HYDRABFLOW_SIM_QUIET:-1}
export JAX_PLATFORMS=${JAX_PLATFORMS:-cpu}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

SIM=${SIM:-stream_gala_spray_mw22}
D=${D:-data_jarvis/data_gala_spray_mw22_hydrabflow}
N_FULL=${N_FULL:-10000}
N_GROUPS=${N_GROUPS:-333}
N_PILOT=${N_PILOT:-24}
RUN_PILOT=${RUN_PILOT:-1}
MAX_WORKERS=${MAX_WORKERS:-48}; MIN_WORKERS=${MIN_WORKERS:-8}; MIN_HEADROOM_GB=${MIN_HEADROOM_GB:-6}
PY=${PY:-.venv/bin/python}          # NOT `uv run` (it has rebuilt the venv and broken agama before)
REAL=assets/gaia/gaia_observed_streams_6Dwitherrors_cutNGC3201.npz
mkdir -p "$D/logs" "$D/ppc"

headroom_gb() { awk '/CommitLimit/{l=$2} /Committed_AS/{c=$2} END{printf "%d", (l-c)/1048576}' /proc/meminfo; }
run_stage() {  # $1 = stage module, rest = overrides; retries until success (chunks resume)
  local attempt=0
  while true; do
    h=$(headroom_gb)
    if (( h < MIN_HEADROOM_GB )); then echo "[$(date +%T)] headroom ${h} GB < ${MIN_HEADROOM_GB}; waiting"; sleep 180; continue; fi
    w=$(( (h - 3) * 4 ))   # ~0.25 GB committed per gala worker (measured 140 MB RSS) + margin
    (( w > MAX_WORKERS )) && w=$MAX_WORKERS; (( w < MIN_WORKERS )) && w=$MIN_WORKERS
    attempt=$((attempt+1)); echo "[$(date +%T)] $1 attempt ${attempt}: headroom ${h} GB -> n_workers=${w}"
    nice -n 10 $PY -m hydrabflow.pipeline.$1 simulator=$SIM data.data_dir="$D" \
      simulator.params.n_workers=$w "${@:2}" && { echo "[$(date +%T)] $1 DONE"; return 0; }
    echo "[$(date +%T)] $1 attempt ${attempt} failed (rc=$?); retry in 120 s"
    sleep 120
  done
}

# ------------------------------------------------------------------ 0. pilot
if [[ "$RUN_PILOT" == "1" ]]; then
  echo "== PILOT: ${N_PILOT} flat rows"
  /usr/bin/time -v nice -n 10 $PY -m hydrabflow.pipeline.simulate simulator=$SIM data.data_dir="$D/pilot" \
      data.n_simulations=$N_PILOT data.chunk_size=$N_PILOT data.dataset_name=pilot_${N_PILOT}.npz \
      simulator.params.n_workers=$N_PILOT seed=11 2> "$D/logs/pilot_time.log"
  grep -E "Elapsed|Maximum resident" "$D/logs/pilot_time.log" || true
  $PY - "$D/pilot/pilot_${N_PILOT}.npz" "$N_PILOT" << 'EOF'
import sys, numpy as np
d = np.load(sys.argv[1]); n = int(sys.argv[2])
x = d["sim_data_projected"]; ok = np.isfinite(x).all(axis=(1, 2))
print(f"rows {len(x)}  NaN rows {int((~ok).sum())}  ({100*(~ok).mean():.1f} %)")
inwin = (x[..., 0] > -900).sum(-1); j = d["j"].ravel().astype(int)
for jj, name in enumerate(("Pal5", "NGC3201", "M68")):
    s = inwin[(j == jj) & ok]
    if s.size: print(f"  {name:8s} stored in-window stars: median {np.median(s):.0f} min {s.min()} capped(2000) {np.mean(s==2000):.2f}  (real members 129/195/297)")
print(f"c_phi_halo_derived range [{d['c_phi_halo_derived'].min():.3f}, {d['c_phi_halo_derived'].max():.3f}]  "
      f"q_rho [{d['q_rho_halo'].min():.2f}, {d['q_rho_halo'].max():.2f}]")
neg = d["halo_rho_neg_r_kpc_derived"].ravel()
print(f"negative-density rows {int(np.isfinite(neg).sum())}/{n}; min radius {np.nanmin(np.where(np.isfinite(neg), neg, np.nan)):.1f} kpc" if np.isfinite(neg).any() else "no negative-density rows")
v = d["vcirc_kms"][:, :, 0]; print(f"vcirc finite {np.isfinite(v).all()}  v_c(8.2 kpc) range [{v[:,2].min():.0f}, {v[:,2].max():.0f}] km/s")
EOF
  echo "== PILOT: 6-group multistream + cold-stream figure"
  nice -n 10 $PY -m hydrabflow.pipeline.simulate_multistream simulator=$SIM data.data_dir="$D/pilot" \
      data.n_simulations=6 data.chunk_size=6 data.dataset_name=pilot_multistream_6.npz simulator.params.n_workers=18 seed=12
  $PY scripts/ppc_summary_statistics.py --sim "$D/pilot/pilot_multistream_6.npz" --real $REAL \
      --out "$D/pilot/ppc_pilot_summary_statistics.png" --aug stream_global --simulator $SIM --noise --n-sim 6 || true
  echo ">>> Inspect $D/pilot/ (timing above, ppc_pilot_summary_statistics*.png). Re-run with RUN_PILOT=0 to skip."
  [[ "${PILOT_ONLY:-0}" == "1" ]] && exit 0
fi

# ------------------------------------------------------------------ 1. test set, 2. training set
[[ -f "$D/test_multistream_${N_GROUPS}.npz" ]] || run_stage simulate_multistream \
    data.n_simulations=$N_GROUPS data.chunk_size=111 data.dataset_name=test_multistream_${N_GROUPS}.npz seed=7
[[ -f "$D/training_data_${N_FULL}.npz" ]] || run_stage simulate \
    data.n_simulations=$N_FULL data.chunk_size=1000 data.dataset_name=training_data_${N_FULL}.npz seed=2026

# ------------------------------------------------------------------ 3. prior-predictive coverage
T="$D/test_multistream_${N_GROUPS}.npz"
echo "== PPC: summary tracks (noise-convolved cold-stream table)"
$PY scripts/ppc_summary_statistics.py --sim "$T" --real $REAL --out "$D/ppc/ppc_summary_statistics_noise.png" \
    --aug stream_global --simulator $SIM --noise --n-sim 40 || true
echo "== PPC: summary-track coverage"
$PY scripts/ppc_summary_coverage.py --sim "$T" --real $REAL --out "$D/ppc/coverage_track" \
    --aug stream_global_ibata_grid_v2 --simulator $SIM --title "gala MW22 spray, ${N_GROUPS} groups" || true
echo "== PPC: network-input GRID coverage (summary-statistics representation)"
$PY scripts/ppc_summary_grid_coverage.py --sim "$T" --real $REAL --out "$D/ppc/coverage_grid" --simulator $SIM \
    --aug stream_global_ibata_grid_v2 --real-aug stream_real_global_ibata_grid_v2 --title "gala MW22 spray" || true
echo "== PPC: raw PARTICLE coverage (particle representation)"
$PY scripts/ppc_particle_coverage.py --sim "$T" --real $REAL --out "$D/ppc/coverage_particles" --simulator $SIM \
    --aug stream_global --real-aug stream_real_global --n-sim 60 || true
echo "== prior corner"
$PY scripts/corner_parameters.py "$D/training_data_${N_FULL}.npz" "$D/ppc" --name prior_corner.png \
    --params log10_M200_halo c200_halo q_rho_halo m_disk h_R_disk h_z_disk c_phi_halo_derived || true

# ------------------------------------------------------------------ 4. training (GPU) — not run here
cat << EOF

== DONE. Train later (GPU via autocvd; both arms read the same dataset):
# summary-statistics arm (gridded per-stream summaries + Ou+2024 curve, grouped diffusion)
SIM=$SIM DATA_DIR=$D N_TRAIN=$N_FULL N_TEST=$N_GROUPS PREPROC=stream_global_log10_gala_2modal REAL_PREPROC=stream_real_global_log10_gala \\
  RUNS_DIR=outputs/gala_mw22_2modal/sumstats PY=$PY AUTOCVD=.venv/bin/autocvd bash scripts/train_v4_2modal.sh
# particles arm (masked SetTransformer over the star cloud + Ou+2024 curve)
SIM=$SIM DATA_DIR=$D N_TRAIN=$N_FULL N_TEST=$N_GROUPS ADAPTER=stream_2modal_particles MODEL=stream_fusion_2modal_particles \\
  EVAL=stream_compositional_masked_particles AUG=stream_global REAL_AUG=stream_real_global \\
  PREPROC=stream_global_log10_gala_2modal REAL_PREPROC=stream_real_global_log10_gala \\
  EXTRA="augmentation.params.vlos_impute=zero" BATCH_SIZE=512 RUNS_DIR=outputs/gala_mw22_2modal/particles \\
  PY=$PY AUTOCVD=.venv/bin/autocvd bash scripts/train_v4_2modal.sh
EOF

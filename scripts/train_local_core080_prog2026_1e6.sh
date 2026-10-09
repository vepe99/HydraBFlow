#!/usr/bin/env bash
# LOCAL-level twin of scripts/train_bspline_core080_prog2026_1e6.sh: the SAME network as trial 14 of
# tuningtest_2modal_bspline_core080_prog2026_study (stream_fusion_2modal_oldgrid = core080 B-spline TST +
# rotation-curve TST -> grouped diffusion with a DiffusionTransformer, trial 14's sizes, drop prob 0.5) but
# targeting each stream's locals [m_progenitor, t_end, vr, r, mu_ra_cosdec, mu_dec], conditioned on the
# stream + curve summaries and the TRUE globals (an always-observed condition group; standardized).
# Locals z-scored per stream on the train split. Trained on the 1e6-row prog2026 set, into
# <DATA_DIR>/tuning/1e6local_14/:
#   train/      approximator, history, convergence
#   eval_sim/   333-group test set conditioned on the true globals: per-stream Pal5_/NGC3201_/M68_
#               recovery, calibration_ecdf, coverage, z_score_contraction, metrics (physical units)
# GPU: each stage picks the least-used card itself (autocvd in hydrabflow's backend, waits for a free one)
# and claims its memory at `import hydrabflow`, before Hydra composes or the 48 GB set loads
# (HYDRABFLOW_EARLY_GPU_INIT + preallocation), so nobody grabs the card in between.
#   bash scripts/train_local_core080_prog2026_1e6.sh
#   GPU=3 DROP_PROB=0 bash scripts/train_local_core080_prog2026_1e6.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
D=${DATA_DIR:-data/data_jarvis/data_agama_spray_massloss_ibata_m200c_v4_p1e3_prog2026_hydrabflow}
PARAMS=$D/tuning/tuningtest_2modal_bspline_core080_prog2026_study/trials/trial_0014/params.json
R=${RUNS_DIR:-$D/tuning/1e6local_14}
N_TRAIN=${N_TRAIN:-1000000} N_EPOCHS=${N_EPOCHS:-1000} BATCH_SIZE=${BATCH_SIZE:-4096} DROP_PROB=${DROP_PROB:-0.5}
mkdir -p "$R"
cp "$PARAMS" "$R/tuned_params.json"

[ "${GPU:-auto}" = auto ] || export CUDA_VISIBLE_DEVICES=$GPU
export HYDRABFLOW_EARLY_GPU_INIT=1 XLA_PYTHON_CLIENT_PREALLOCATE=true
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}

# ++ because struct mode rejects dt_time_embedding_dim / dt_dropout (tune.py force-adds them)
mapfile -t TUNED < <($PY -c "import json,sys;[print(f'++{k}={v}') for k,v in json.load(open(sys.argv[1])).items()]" "$PARAMS")
COMMON=(simulator=stream_agama_spray_massloss_ibata_m200c_v4_prog2026 model=stream_fusion_2modal_oldgrid
        composition=local adapter=stream_2modal_local
        preprocessing=stream_local_log10_sumstats_2modal augmentation=stream_global_streamfinder_bspline_core080
        "training.standardize=[inference_variables,inference_conditions,summary_variables]"
        model.inference_network.params.missing_modality_prob=$DROP_PROB
        data.data_dir=$D seed=2026 augmentation.params.resources_dir=assets/gaia "${TUNED[@]}")

echo "=== [1/2] TRAIN local level ($N_TRAIN rows) -> $R/train ==="
[ -f "$R/train/approximator.keras" ] || $PY -m hydrabflow.pipeline.train "${COMMON[@]}" \
  data.n_simulations=$N_TRAIN training.n_epochs=$N_EPOCHS training.batch_size=$BATCH_SIZE hydra.run.dir=$R/train

echo "=== [2/2] EVALUATE sim per stream (true globals) -> $R/eval_sim ==="
[ -f "$R/eval_sim/M68_metrics.json" ] || $PY -m hydrabflow.pipeline.evaluate "${COMMON[@]}" model_dir=$R/train \
  data.n_simulations=333 eval.test_dataset_name=test_multistream_333.npz eval.num_samples=1000 eval.batch_size=32 \
  hydra.run.dir=$R/eval_sim

echo "=== DONE. Outputs under $R ==="

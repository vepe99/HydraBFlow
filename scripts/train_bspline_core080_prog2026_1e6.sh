#!/usr/bin/env bash
# Trial 14 of tuningtest_2modal_bspline_core080_prog2026_study, retrained on the 1e6-row prog2026 set
# (training_data_1000000.npz, same simulator config as the 300k set the study tuned on), then the same
# evaluation suite as the original trial_0014, into <DATA_DIR>/tuning/1e6global_14/:
#   train/                          approximator, history, convergence
#   eval_sim/                       333-group test set: base (each stream + curve) and compositional
#   eval_sim_only_sim_summary/      base, streams alone (curve masked)
#   eval_sim_only_vcirc_kms/        base, rotation curve alone (streams masked)
#   eval_real_10k/                  real Gaia, 1e4 samples: each stream + curve, and compositional
#   eval_real_10k_only_sim_summary/ real Gaia, each stream alone (+ compositional)
#   eval_real_10k_only_vcirc_kms/   real Gaia, curve alone (+ compositional)
#   eval_real_10k/real_global_vs_streams_corner_latex.png  each stream + curve, with the compositional
#   eval_real_10k/real_modalities_corner_latex.png         each stream alone + curve alone, with the compositional
# Each stage is skipped if its output exists, so a crashed run resumes.
#   bash scripts/train_bspline_core080_prog2026_1e6.sh          # autocvd waits for a free GPU
#   GPU=3 bash scripts/train_bspline_core080_prog2026_1e6.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
D=data/data_jarvis/data_agama_spray_massloss_ibata_m200c_v4_p1e3_prog2026_hydrabflow
PARAMS=$D/tuning/tuningtest_2modal_bspline_core080_prog2026_study/trials/trial_0014/params.json
R=${RUNS_DIR:-$D/tuning/1e6global_14}
REAL=assets/gaia/gaia_observed_streams_6Dwitherrors_cutNGC3201.npz
mkdir -p "$R"
cp "$PARAMS" "$R/tuned_params.json"

GPU=${GPU:-auto}
if [ "$GPU" = auto ]; then eval "$(.venv/bin/autocvd -n 1)"; else CUDA_VISIBLE_DEVICES=$GPU; fi
export CUDA_VISIBLE_DEVICES XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# ++ because struct mode rejects dt_time_embedding_dim / dt_dropout (tune.py force-adds them)
mapfile -t TUNED < <($PY -c "import json,sys;[print(f'++{k}={v}') for k,v in json.load(open(sys.argv[1])).items()]" "$PARAMS")
COMMON=(simulator=stream_agama_spray_massloss_ibata_m200c_v4_prog2026 model=stream_fusion_2modal_oldgrid
        composition=global adapter=stream_2modal eval=stream_compositional_masked
        model.inference_network.params.missing_modality_prob=0.5
        data.data_dir=$D seed=2026 augmentation.params.resources_dir=assets/gaia "${TUNED[@]}")
SIM=("${COMMON[@]}" preprocessing=stream_global_log10_sumstats_2modal augmentation=stream_global_streamfinder_bspline_core080)
EVAL_SIM=("${SIM[@]}" model_dir=$R/train data.n_simulations=333 eval.test_dataset_name=test_multistream_333.npz
          eval.num_samples=1000 eval.batch_size=8)
EVAL_REAL=("${COMMON[@]}" preprocessing=stream_real_global_log10 augmentation=stream_real_global_streamfinder_bspline_core080
           model_dir=$R/train data.real_data_path=$REAL eval.misspecification_reference=$R/eval_sim
           eval.num_samples=10000 eval.batch_size=32)

echo "=== [1/5] TRAIN (1e6 rows) -> $R/train ==="
[ -f "$R/train/approximator.keras" ] || $PY -m hydrabflow.pipeline.train "${SIM[@]}" \
  data.n_simulations=1000000 training.n_epochs=1000 training.batch_size=4096 hydra.run.dir=$R/train

echo "=== [2/5] EVALUATE sim, streams + curve (base + compositional) -> $R/eval_sim ==="
[ -f "$R/eval_sim/compositional_metrics.json" ] || $PY -m hydrabflow.pipeline.evaluate "${EVAL_SIM[@]}" hydra.run.dir=$R/eval_sim

echo "=== [3/5] EVALUATE sim, one modality at a time (base) ==="
# The compositional stage ignores eval.observed_groups (its own mask plan), i.e. it would redo eval_sim's
# ~35 min run, so the process is stopped once the base posterior is saved (as in trial_0014).
for G in sim_summary vcirc_kms; do
  OUT=$R/eval_sim_only_$G
  [ -f "$OUT/base_metrics.json" ] && continue
  $PY -m hydrabflow.pipeline.evaluate "${EVAL_SIM[@]}" "eval.observed_groups=[$G]" hydra.run.dir=$OUT &
  PID=$!
  until { [ -f "$OUT/base_metrics.json" ] && grep -q "Compositional" "$OUT/evaluate.log" 2>/dev/null; } || ! kill -0 $PID 2>/dev/null; do sleep 10; done
  kill $PID 2>/dev/null || true; wait $PID 2>/dev/null || true
  [ -f "$OUT/base_metrics.json" ] || { echo "eval_sim_only_$G failed"; exit 1; }
done

echo "=== [4/5] EVALUATE REAL, 1e4 samples: streams + curve, streams only, curve only ==="
[ -f "$R/eval_real_10k/posterior.npz" ] || $PY -m hydrabflow.pipeline.evaluate "${EVAL_REAL[@]}" hydra.run.dir=$R/eval_real_10k
for G in sim_summary vcirc_kms; do
  OUT=$R/eval_real_10k_only_$G
  [ -f "$OUT/single_stream_posterior.npz" ] || $PY -m hydrabflow.pipeline.evaluate "${EVAL_REAL[@]}" "eval.observed_groups=[$G]" hydra.run.dir=$OUT
done

echo "=== [5/5] CORNERS (compositional always shown) ==="
$PY scripts/replot_global_corner.py $R/eval_real_10k                       # each stream + curve
$PY scripts/replot_global_corner.py $R/eval_real_10k --out real_modalities_corner_latex.png \
  --stream-only $R/eval_real_10k_only_sim_summary --curve-only $R/eval_real_10k_only_vcirc_kms --no-joint

echo "=== DONE. Outputs under $R ==="

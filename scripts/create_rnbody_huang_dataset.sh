#!/usr/bin/env bash
# Create the restricted-N-body + free-beta + Zhou∪Huang-rejection stream datasets:
#   1. 60k flat training set  -> training_data_60000.npz
#   2. 333-group multistream  -> test_multistream_333.npz
# CPU/joblib only (no GPU). n_workers=180 (~70% of 256 cores) lives in the simulator config;
# batch_size=1 (load balancing) is in stream_agama.py. Both stages are resumable.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_DIR=${DATA_DIR:-data_jarvis/data_agama_rnbody_huang_hydrabflow}
SEED=${SEED:-2026}

# 1) training set
uv run python scripts/simulate.py \
  simulator=stream_agama_rnbody_huang \
  composition=global \
  data.data_dir="${DATA_DIR}" \
  data.n_simulations=60000 \
  data.chunk_size=10000 \
  seed="${SEED}"

# 2) multistream test set
uv run python scripts/simulate_multistream.py \
  simulator=stream_agama_rnbody_huang \
  composition=global \
  data.data_dir="${DATA_DIR}" \
  data.n_simulations=333 \
  data.chunk_size=333 \
  data.dataset_name=test_multistream_333.npz \
  seed="${SEED}"

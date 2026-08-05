# Running

## Install

```bash
uv sync            # .venv + all deps (incl. pytest, ruff)
uv run pytest -q   # optional sanity check, CPU only
```

## The four stages

Each stage is a Hydra app: `uv run hydrabflow-<stage>` (or `python -m hydrabflow.pipeline.<stage>`).
Any config value can be overridden on the command line.

| Stage      | Reads                              | Writes                                                        |
| ---------- | ---------------------------------- | ------------------------------------------------------------- |
| `simulate` | prior + forward model              | dataset `.npz` in `data.data_dir` (+ config snapshot)         |
| `train`    | that `.npz`                        | `approximator.keras`, `preprocessing_state.npz`, `loss.png`, `history.json` |
| `evaluate` | `model_dir` + test (or real) data  | `posterior.npz`, `metrics.json`, diagnostic plots             |
| `tune`     | that `.npz`                        | Optuna study + `best_trials.json`                             |

Everything lands in `outputs/${simulator.name}/${run_name}/<timestamp>/`, together with Hydra's
`.hydra/` copy of the resolved config — a run is reproducible from its own output folder. `run_name`
only labels that output dir; `model_dir` is the opposite, an *input* pointing at a finished `train`
dir for `evaluate` to load ([details](configuration.md#run_name-vs-model_dir)).

## Two Moons, end to end

```bash
# 1. training + test datasets (the test set only differs in name and seed)
uv run hydrabflow-simulate
uv run hydrabflow-simulate data.dataset_name=test_data_10000.npz seed=7

# 2. train
uv run hydrabflow-train

# 3. evaluate the run you just trained
uv run hydrabflow-evaluate model_dir=outputs/two_moons/set_transformer+flow_matching/<timestamp>
```

Smoke version (~1 min, CPU): append
`data.n_simulations=200 data.chunk_size=100 training.n_epochs=2 training.batch_size=32
eval.num_samples=50 data.data_dir=/tmp/hbf` to each command.

The multimodal example shows two observables fused into one summary:

```bash
uv run hydrabflow-simulate simulator=multimodal model/summary_network=fusion
uv run hydrabflow-train    simulator=multimodal model/summary_network=fusion
```

## Your own observed data

Set `data.real_data_path` and `evaluate` skips everything truth-aware (no metrics, no
resimulation) and writes posterior pair plots only:

```bash
uv run hydrabflow-evaluate model_dir=<train run> data.real_data_path=data/observed.npz
```

The `.npz` needs the same observable keys, shapes and units as the training data — the fitted
preprocessing from `model_dir` is replayed on it.

## A dataset with no simulator

`adapter.inference_variables` / `summary_variables` are normally derived from the simulator. For
data no simulator produced, name the keys yourself in `conf/config.yaml` (or on the CLI):

```yaml
adapter:
  inference_variables: [theta1, theta2]
  summary_variables: [x]
```

Then run `train` / `evaluate` directly on your `.npz` (skip `simulate`).

## Tuning

```bash
uv run hydrabflow-tune          # multi-objective: RMSE + calibration error
```

The study lives in `tuning.storage_dir` as a concurrency-safe log, so running the command N times
in parallel extends one study. Search space: `tuning.search_space` in `conf/config.yaml`.

## GPU / CPU

The backend is JAX with `KERAS_BACKEND=jax`, pinned before any keras import. Before Hydra exists,
so these are env vars:

- `CUDA_VISIBLE_DEVICES=2` — explicit choice, disables auto-selection.
- `HYDRABFLOW_NUM_GPUS=0` — CPU only. Default `1`, picked by `autocvd` (least-used GPU).
- `KERAS_BACKEND=tensorflow` — other backend.

On GPU OOM, training halves `training.batch_size` and retries, down to 16.

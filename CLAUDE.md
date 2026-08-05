# HydraBFlow: SBI pipeline template (BayesFlow + Hydra)

A cookiecutter-style repo for Simulation-Based Inference. Infrastructure (dataset generation,
training, inference, tuning, tracing) is fixed; a new user writes a simulator and picks networks.

## Design principles

- **Full traceability**: every run writes its resolved Hydra config into its output dir. A run is
  valid only if it can be reconstructed from that folder.
- **Hydra-native**: all entry points are Hydra apps, no argparse.
- **Structured configs + registries, not `_target_`**: typed dataclasses in `config.py` (only
  `RootConfig` is in the ConfigStore, as `base_config`); YAML fills in values; a name string bridges
  config to code through the five registries in `hydrabflow/registry.py`. Components self-register
  (`@register_simulator`, `@register_step`, `@register_augmentation`, `@register_summary_network`,
  `@register_inference_network`), and each registry lazily imports its whole package on the first
  failed lookup (`Registry.discover`), so adding a component = drop a file + a config entry. No
  `__init__.py` edit.
- **The simulator owns the variable names**: empty `adapter.inference_variables` /
  `summary_variables` are filled from `parameter_names` / `observable_keys`
  (`pipeline.adapter.fill_adapter_from_simulator`). Explicit config wins (bring-your-own-data).
- **Single-level inference only**: one summary + one inference network via `bf.BasicWorkflow`. No
  hierarchical global/local split, no compositional scoring.
- **Preprocessing ≠ augmentation**: preprocessing is deterministic, whole-dataset, fit on the train
  split, state saved and replayed (`preprocessing/`); augmentation is stochastic and per-batch inside
  `fit_offline` (`augmentation/`).

## Stack

BayesFlow 2.x (Keras 3) on JAX — `KERAS_BACKEND=jax` pinned by `utils/backend.py`, imported first by
`hydrabflow/__init__.py`, GPU chosen by `autocvd`. `uv` for packaging (src-layout, `hydrabflow-*`
console scripts). Optuna for multi-objective tuning. Marimo for `notebooks/explore.py`.

## Layout

```
conf/
  config.yaml                  # everything: seed, run_name, model_dir, data, training,
                               #   preprocessing, augmentation, adapter, eval, tuning, hydra
  simulator/                   # two_moons, multimodal (+ yours)
  model/summary_network/       # set_transformer | deep_set | time_series_transformer | fusion
  model/inference_network/     # flow_matching | diffusion
src/hydrabflow/
  config.py                    # all dataclass schemas + register_configs()
  registry.py                  # all 5 registries + decorators + the 3 builders
  simulators/                  # USER: base.py, two_moons.py, multimodal.py
  networks/factory.py          # shipped network builders
  preprocessing/               # base, standardize, steps
  augmentation/noise.py
  pipeline/                    # INFRA: _app, adapter, workflow, io, artifacts, simulate,
                               #   train, evaluate, tune
  utils/                       # backend (JAX/GPU pin), seed, paths, oom
tests/  docs/  outputs/ (gitignored)
```

## Stages

`hydrabflow-<stage>` (or `python -m hydrabflow.pipeline.<stage>`), output dir
`outputs/${simulator.name}/${run_name}/<timestamp>`:

- `simulate` — prior + forward model in chunks → `.npz` + a `<stem>.hydra/` config snapshot next to it.
- `train` — `.npz` → preprocessing (fit on train, save state) → `fit_offline` with augmentations →
  `approximator.keras`, `approximator_best.weights.h5`, `preprocessing_state.npz`, `loss.png`,
  `history.json`.
- `evaluate` — loads model + preprocessing state from `model_dir`, samples the posterior. One code
  path, two modes: simulated test set (truth-aware diagnostics + `metrics.json`) or
  `data.real_data_path` set (posterior pair plots only).
- `tune` — Optuna multi-objective (RMSE + calibration error) over `tuning.search_space`;
  `best_trials.json`, study in `tuning.storage_dir` (concurrency-safe log → N parallel launches
  extend one study).

## What the user modifies

`conf/simulator/<name>.yaml` + `src/hydrabflow/simulators/<name>.py`; the network group YAMLs;
knobs in `conf/config.yaml`. Optionally custom preprocessing steps, augmentations, or network
builders. The `adapter:` block stays untouched unless there is no simulator. Nothing else.

**Fixed infrastructure**: `pipeline/` (stages, `make_cli`, adapter/workflow builders, io,
artifacts), `config.py`, `registry.py`, `utils/backend.py`, the `hydra:` block in `config.yaml`.

## Notes worth keeping

- `training.learning_rate` is the peak LR passed as `initial_learning_rate`; `BasicWorkflow`
  wraps it in cosine decay with 5% warmup + AdamW. Passing an explicit optimizer disables that
  schedule, so `training.optimizer` deliberately does not exist.
- `training.standardize` is `[inference_variables]` only — the summary side is handled by the
  `standardize` preprocessing step, whose fitted state is saved and replayed.
- Networks take `embed_dim_per_head` (attention width = `num_heads * embed_dim_per_head`), so any
  tuner draw stays divisible by the head count.
- Fusion is implemented, not a seam: >1 `summary_variables` → one backbone per key behind
  `bf.networks.FusionNetwork`; per-key type via `model.summary_network.params.backbones`.
- `load_approximator` passes `compile=False` (evaluation never resumes training) — without it,
  loading a TimeSeriesTransformer fails on AdamW slot-variable shapes.
- The preprocessing framework cannot be replaced by adapter transforms: `Adapter.standardize`
  requires caller-supplied mean/std, and the adapter is per-batch, so `drop_nan` (row filtering)
  and `train_val_split` are impossible there.
- Open: no `fit_online` support — the pipeline is offline-only, so a cheap simulator cannot skip
  the `simulate` stage.

## docs/

Three files, deliberately short: `running.md` (install, stages, walkthroughs, real data, tuning,
GPU env vars), `configuration.md` (config blocks + the three groups), `extending.md` (the
drop-a-module-and-decorate pattern for all five extension points).

## graphify

Knowledge graph at `graphify-out/`.

- For codebase questions, run `graphify query "<question>"` first (also `graphify path "<A>" "<B>"`,
  `graphify explain "<concept>"`) — a scoped subgraph, far smaller than `GRAPH_REPORT.md` or grep.
- `graphify-out/wiki/index.md`, if present, beats raw source browsing for navigation.
- Read `GRAPH_REPORT.md` only for broad architecture review.
- After changing code, run `graphify update .` (AST-only, no API cost).

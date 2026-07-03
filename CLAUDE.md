# HydraBFlow: SBI Pipeline Template with BayesFlow 

## Goal

A reusable, cookiecutter-style repository for setting up Simulation-Based Inference (SBI)
pipelines using BayesFlow and Hydra. The template handles all infrastructure (training,
inference, dataset generation, experiment tracking) so that a new user only needs to:

1. Write their simulator (forward model)
2. Choose and configure their SBI components (summary network, inference network, etc.)

Everything else — config management, output tracing, reproducibility — is fixed infrastructure.

## Core Design Principles

- **Full traceability**: every run (training, inference, dataset generation) must save its
  Hydra config to the output directory. A run is only valid if it can be fully reconstructed
  from its output folder.
- **Hydra-native**: all entry points are Hydra apps. No argparse. Config composition via
  config groups covers all axes of variation (model, simulator, training, data).
- **Modularity via structured configs + registries** (NOT `_target_`): every config group has a
  typed dataclass schema registered in Hydra's `ConfigStore`; YAML files fill in values. Factory
  functions read those dataclasses and resolve names through registries (`networks.factory`,
  `simulators.registry`, `preprocessing.registry`, `augmentation.registry`, `pipeline.adapter`).
  Components self-register by name (`@register_simulator`, `@register_step`,
  `@register_augmentation`, `@register_summary_network`, `@register_inference_network`), and each
  package auto-imports its modules (`utils.discovery`), so adding a component = dropping a file +
  a config entry, no infrastructure edits (not even `__init__.py`).
- **The simulator is the single source of truth for variable names**: empty
  `adapter.inference_variables` / `summary_variables` are derived from the simulator's
  `parameter_names` / `observable_keys` at CLI entry (`pipeline.adapter.fill_adapter_from_simulator`).
  Explicit adapter config overrides (required for bring-your-own-data, where no class exists).
- **Separation of concerns**: infrastructure code (training loop, logging, checkpointing)
  is never modified by the end user. User-facing code lives in clearly marked locations
  (`src/hydrabflow/simulators/`, plus optional custom `networks`/`preprocessing`/`augmentation`).
- **Single-level inference only**: one summary network + one inference network via
  `bf.BasicWorkflow`. The reference project's hierarchical global/local split and compositional
  vs non-compositional score modeling are deliberately removed.
- **Preprocessing vs augmentation are distinct stages**: preprocessing is deterministic,
  whole-dataset, applied once and fitted on the train split (`src/hydrabflow/preprocessing/`);
  augmentation is stochastic and per-batch, applied inside `fit_offline`
  (`src/hydrabflow/augmentation/`).

## Tech Stack

- **SBI framework**: BayesFlow 2.x (Keras 3)
- **Compute backend**: JAX. `KERAS_BACKEND=jax` is pinned by `hydrabflow.utils.backend` (imported
  first via `hydrabflow/__init__.py`) before any keras/bayesflow import. Override via env var.
- **Packaging / env**: `uv` (`pyproject.toml`, src-layout, console scripts `hydrabflow-*`).
- **Config management**: Hydra with structured dataclass configs (`ConfigStore`) + config groups.
- **Neural architectures**: SetTransformer / DeepSet / TimeSeriesTransformer (summary network),
  FlowMatching / DiffusionModel (inference network) — user-swappable via config. Summary defaults
  to a single observable; multi-observable FusionNetwork is a documented seam in
  `pipeline.adapter` + `networks.factory`.
- **Hyperparameter tuning**: Optuna (multi-objective: RMSE + calibration error).
- **Notebooks**: Marimo (`notebooks/explore.py`).

## Folder Structure (finalized)

HydraBFlow/
├── pyproject.toml               # uv-managed; deps + console scripts (hydrabflow-*)
├── conf/                        # Hydra config groups (YAML values; schemas live in code)
│   ├── config.yaml              # Root: defaults list, seed, model_dir, hydra.run.dir
│   ├── simulator/               # skeleton.yaml (+ your simulators)
│   ├── model/                   # default.yaml -> summary_network/ + inference_network/
│   ├── training/  data/  preprocessing/  augmentation/
│   ├── adapter/   inference/    eval/   tuning/
├── src/hydrabflow/
│   ├── config/schema.py         # ALL dataclass schemas + register_configs()
│   ├── simulators/              # USER MODIFIES: base.py, registry.py, skeleton.py
│   ├── networks/factory.py      # build_summary_network / build_inference_network
│   ├── preprocessing/           # base, standardize, steps, registry (deterministic, once)
│   ├── augmentation/            # base/registry + examples (stochastic, per-batch)
│   ├── pipeline/                # INFRASTRUCTURE: adapter, workflow, io, checkpoint,
│   │                            #   simulate, train, evaluate, evaluate_real, tune, _app
│   └── utils/                   # backend (JAX pin), seed, logging, paths
├── scripts/                     # thin Hydra entry points -> pipeline.<stage>.cli
│   ├── simulate.py  train.py  evaluate.py  evaluate_real.py  tune.py
├── tests/                       # config-compose, registries, preprocessing, workflow smoke tests
├── notebooks/explore.py         # Marimo
├── outputs/                     # Hydra run dirs (gitignored)
└── CLAUDE.md

### Run stages (5 entry points)
- `simulate`  — sample prior + run forward model in chunks -> aggregated `.npz`.
- `train`     — load `.npz` -> preprocessing (fit on train, save state) -> `fit_offline` with
                augmentations -> save approximator + loss curve.
- `evaluate`  — load model + preprocessing state from `model_dir`, sample posterior on a
                simulated test set, write truth-aware diagnostics (RMSE/calibration, recovery,
                calibration ECDF, z-score contraction).
- `evaluate_real` — same, but on a user-provided real-data `.npz` (no truth, no resimulation).
- `tune`      — Optuna multi-objective study over a config-driven search space.

## What the User Modifies

- `conf/simulator/<name>.yaml` + `src/hydrabflow/simulators/<name>.py`: the forward model
  (a `@register_simulator`-decorated `BaseSimulator` subclass; auto-imported, self-registers).
- `conf/adapter/*`: normally untouched — variables derive from the simulator. Explicit config
  only for bring-your-own-data or to override the derivation (subset inference, fusion).
- `conf/model/...`: choose/configure summary + inference networks.
- Optionally: custom preprocessing steps, augmentations, or network architectures (drop a module
  in the package; each self-registers; no infra edits).
- Nothing else should need to change for a new problem.

## What Is Fixed Infrastructure (do not modify)

- Entry point scripts (`scripts/`) and the `pipeline.*.cli` wrappers (`pipeline/_app.py`).
- The five run stages, adapter/workflow builders, IO, checkpointing (`src/hydrabflow/pipeline/`).
- Config schema + registration (`src/hydrabflow/config/schema.py`).
- Hydra output directory setup and config saving; JAX backend pin (`utils/backend.py`).

## Output Directory Convention

Hydra's `hydra.run.dir` is set to:
`outputs/${simulator.name}/${model.name}/${now:%Y-%m-%d_%H-%M-%S}`

Every run saves:
- `.hydra/` folder with full config (Hydra does this automatically)
- `simulate`: dataset `.npz` in `data.data_dir`, plus a `<dataset_stem>.hydra/` config snapshot
  next to it (copied from Hydra's `.hydra/` via `utils.paths.save_config_snapshot`) so each
  dataset is traceable to the config that generated it. Keyed by the dataset filename so
  training and test sets in the same `data_dir` don't overwrite each other's snapshot.
- `train`: `approximator.keras`, `preprocessing_state.npz`, `loss.png`
- `evaluate`: `posterior.npz`, `metrics.json`, diagnostic plots
- `evaluate_real`: `posterior.npz`, posterior pair plots
- `tune`: `best_trials.json` (Optuna study in `tuning.storage_dir`)

`evaluate` / `evaluate_real` load the trained model + fitted preprocessing from `model_dir`
(set it to a completed `train` run dir).

## Decisions Log

*(Update this section after each Claude Code session)*

- [x] Folder structure finalized
- [x] Config group schema defined (structured dataclasses in `config/schema.py`, no `_target_`)
- [x] Base simulator interface defined (`simulators/base.py` + registry; skeleton stub shipped)
- [x] Training loop scaffold written (`pipeline/train.py` via `bf.BasicWorkflow.fit_offline`)
- [x] Five run stages implemented + verified end-to-end on a temporary Gaussian simulator
- [x] Preprocessing module (deterministic, fit-on-train, save/load) separate from augmentation
- [x] Optuna multi-objective tuning wired
- Session 1 decisions: JAX backend; structured-dataclass configs (overrides original `_target_`
  plan); skeleton-only example simulator; single-observable summary, fusion-ready; single-level
  inference (no global/local, no compositional scoring).
- Session 2026-07-03 (user-friendliness pass): default simulator = `two_moons` (first run succeeds
  out of the box; skeleton stays as the copyable stub); restored timestamped `hydra.run.dir`
  (removed leftover debug value); packages auto-import their modules via `utils.discovery` so
  dropped components self-register without `__init__.py` edits; adapter variables derive from the
  simulator when left empty (`fill_adapter_from_simulator` in `pipeline/_app.py`), explicit config
  wins; summary/inference network builders moved from if/elif to registries
  (`@register_summary_network` / `@register_inference_network`) with free-form `params` in both
  network schemas for custom builders; dev deps moved to `[dependency-groups]` so `uv sync`
  installs pytest/ruff by default.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

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
├── conf/                        # ONE config file + the 3 groups with real alternatives
│   ├── config.yaml              # EVERYTHING: seed, run_name, model_dir, data, training,
│   │                            #   preprocessing, augmentation, adapter, eval, tuning, hydra
│   ├── simulator/               # two_moons.yaml (+ your simulators)
│   ├── model/summary_network/   # set_transformer | deep_set | time_series_transformer
│   └── model/inference_network/ # flow_matching | diffusion
├── src/hydrabflow/
│   ├── config.py                # ALL dataclass schemas + register_configs() (one node)
│   ├── simulators/              # USER MODIFIES: base.py, registry.py, two_moons.py
│   ├── networks/factory.py      # build_summary_network / build_inference_network
│   ├── preprocessing/           # base, standardize, steps, registry (deterministic, once)
│   ├── augmentation/            # registry + noise.py (stochastic, per-batch)
│   ├── pipeline/                # INFRASTRUCTURE: adapter, workflow, io, checkpoint, artifacts,
│   │                            #   simulate, train, evaluate, tune, _app
│   └── utils/                   # backend (JAX pin), registry (shared by all 5 extension
│                                #   points), seed, paths, oom
├── tests/                       # config-compose, registries, preprocessing, workflow smoke tests
├── notebooks/explore.py         # Marimo
├── outputs/                     # Hydra run dirs (gitignored)
└── CLAUDE.md

### Run stages (4 entry points: `hydrabflow-<stage>`, or `python -m hydrabflow.pipeline.<stage>`)
- `simulate`  — sample prior + run forward model in chunks -> aggregated `.npz`.
- `train`     — load `.npz` -> preprocessing (fit on train, save state) -> `fit_offline` with
                augmentations -> save approximator + loss curve.
- `evaluate`  — load model + preprocessing state from `model_dir` and sample the posterior. Two
                modes, one code path: the simulated test set (writes truth-aware diagnostics —
                RMSE/calibration, recovery, calibration ECDF, z-score contraction), or, when
                `data.real_data_path` is set, a user-provided real-data `.npz` (no truth, no
                resimulation: posterior pair plots only).
- `tune`      — Optuna multi-objective study over a config-driven search space.

## What the User Modifies

- `conf/simulator/<name>.yaml` + `src/hydrabflow/simulators/<name>.py`: the forward model
  (a `@register_simulator`-decorated `BaseSimulator` subclass; auto-imported, self-registers).
- `conf/config.yaml`: every knob (training, data, preprocessing, augmentation, eval, tuning). The
  `adapter:` block is normally untouched — variables derive from the simulator; set it explicitly
  only for bring-your-own-data or to override the derivation (subset inference, fusion).
- `conf/model/summary_network/*`, `conf/model/inference_network/*`: choose/configure the networks.
- Optionally: custom preprocessing steps, augmentations, or network architectures (drop a module
  in the package; each self-registers; no infra edits).
- Nothing else should need to change for a new problem.

## What Is Fixed Infrastructure (do not modify)

- The `cli = make_cli(run_fn)` entry points (`pipeline/_app.py`); there is no `scripts/` folder.
- The four run stages, adapter/workflow builders, IO, checkpointing (`src/hydrabflow/pipeline/`).
- Config schema + registration (`src/hydrabflow/config.py`); the shared `utils/registry.py`.
- Hydra output directory setup and config saving; JAX backend pin (`utils/backend.py`).

## Output Directory Convention

Hydra's `hydra.run.dir` is set to:
`outputs/${simulator.name}/${run_name}/${now:%Y-%m-%d_%H-%M-%S}`
(`run_name` defaults to `<summary_type>+<inference_type>`; override it to label an experiment.)

Every run saves:
- `.hydra/` folder with full config (Hydra does this automatically)
- `simulate`: dataset `.npz` in `data.data_dir`, plus a `<dataset_stem>.hydra/` config snapshot
  next to it (copied from Hydra's `.hydra/` via `utils.paths.save_config_snapshot`) so each
  dataset is traceable to the config that generated it. Keyed by the dataset filename so
  training and test sets in the same `data_dir` don't overwrite each other's snapshot.
- `train`: `approximator.keras`, `approximator_best.weights.h5`, `preprocessing_state.npz`,
  `loss.png`, `history.json`
- `evaluate`: `posterior.npz` + `metrics.json` and diagnostic plots (simulated test set), or
  `posterior.npz` + posterior pair plots (`data.real_data_path` set)
- `tune`: `best_trials.json` (Optuna study in `tuning.storage_dir`)

`evaluate` loads the trained model + fitted preprocessing from `model_dir` (set it to a completed
`train` run dir).

## Decisions Log

*(Update this section after each Claude Code session)*

- [x] Folder structure finalized
- [x] Config group schema defined (structured dataclasses in `config.py`, no `_target_`)
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
- Session 2026-08-03 (lightweight pass, ~1000 lines cut from src+conf): **conf/ is now one
  `config.yaml` plus three groups** (`simulator`, `model/summary_network`,
  `model/inference_network`) — no `default.yaml` files anywhere and no group YAML inheriting from
  another via a `defaults:` list. Only `RootConfig` is stored in the ConfigStore (as `base_config`);
  every group is a typed field of it, so group YAMLs are still validated with no per-group `base_*`
  node. `model.name` -> root `run_name` (defaults to `<summary_type>+<inference_type>`); the
  `inference` group merged into `eval` (both stages share `eval.num_samples`/`batch_size`).
  Removed: `utils/reporting.py` (report.md + heuristic metric ratings), `utils/quiet.py` and
  `utils/progress.py` (both callerless; the latter monkeypatched joblib globally), the skeleton
  simulator + its YAML (two_moons is the copyable example), `io.run_chunked`'s resume machinery,
  `PreprocessStep.inverse_transform` / `get_step`, the `cast_dtype` / `select_keys` steps, the
  `available_*()` registry wrappers, and two of three example augmentations. Added
  `pipeline/artifacts.py` (public `run_diagnostics` / `save_history` / `restore_best_weights` /
  `save_posterior*`) so stages no longer import each other's underscore functions. Networks take
  `embed_dim_per_head` instead of `embed_dim` + `params.embed_dim_multiplier` (one way to set
  attention width). Default `training.standardize` is now `[inference_variables]` only — the summary
  side is standardized by the preprocessing step whose fitted state is saved and replayed.
  All 5 stages re-verified end to end on two_moons (CPU); 36 tests pass.
- Session 2026-08-03b (src+scripts structural pass, ~470 more lines cut): **`scripts/` is gone** —
  each file was a 7-line re-export of a `cli` that pyproject already exposes as `hydrabflow-<stage>`;
  every stage is also `python -m hydrabflow.pipeline.<stage>`. **One shared `utils/registry.py`
  (`Registry` + `discover`)** replaces the five hand-rolled name->object dicts (simulators,
  preprocessing steps, augmentations, and the two network registries); each package's `registry.py`
  is now ~12 lines and its `__init__.py` is one `discover(__name__, __path__)` call, so `discovery.py`
  is gone too. Registries expose `.items` and raise `KeyError` (not `ValueError`) on unknown names.
  **`evaluate_real` merged into `evaluate`**: one stage, switching on `data.real_data_path` — 4 run
  stages, not 5. `config/schema.py` -> flat `config.py` (the package held one module).
  `utils/logging.py` deleted (Hydra installs the handlers; stages use `logging.getLogger(__name__)`).
  Also removed: `paths.ensure_dir` (= `os.makedirs(exist_ok=True)`) and 3 of 4 filename constants,
  the `SplitStep` marker ABC (now a `splits: bool` attribute) and the `name#occurrence.` state-key
  scheme, `BaseSimulator`'s abstract `parameter_names`/`observable_keys` properties (plain class
  attributes now — user simulators get shorter), tune's single-objective code path, and ~70 lines of
  docstring that restated this file. src+conf: 2312 -> 1969 lines.
  **docs/ rewritten** (2161 -> 840 lines, 5 files -> 3): `running.md` (install, the 4 stages, the
  two_moons walkthrough + smoke run, real-data mode, tuning, bring-your-own-dataset, artifacts, GPU
  env vars), `configuration.md` (every block of `config.yaml` + the 3 groups), `extending.md` (the
  one drop-a-module-and-decorate pattern for all 5 extension points). No longer stale.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

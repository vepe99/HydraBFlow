# HydraBFlow — Configuration Reference

Everything HydraBFlow does is driven by [`conf/`](../conf). This document is the in-depth
reference for that folder: what each config group means, every field it exposes, how to add a new
config, and how to point the pipeline at a master config of your own.

It complements the task-oriented walkthrough in
[`end_to_end_guide.md`](./end_to_end_guide.md) — read that to *do* a pipeline; read this to
*understand the knobs*.

---

## Table of contents

1. [How the config system works](#1-how-the-config-system-works)
2. [The root master config — `config.yaml`](#2-the-root-master-config--configyaml)
3. [Config groups, one by one](#3-config-groups-one-by-one)
   - [`simulator/`](#31-simulator) · [`model/`](#32-model) (+ `summary_network/`, `inference_network/`) ·
     [`data/`](#33-data) · [`training/`](#34-training) · [`preprocessing/`](#35-preprocessing) ·
     [`augmentation/`](#36-augmentation) · [`adapter/`](#37-adapter) ·
     [`inference/`](#38-inference) · [`eval/`](#39-eval) · [`tuning/`](#310-tuning)
4. [Overriding configs from the CLI](#4-overriding-configs-from-the-cli)
5. [Step-by-step: adding a new config file](#5-step-by-step-adding-a-new-config-file)
6. [Using your own master config](#6-using-your-own-master-config)
7. [Splitting configs per task / per folder](#7-splitting-configs-per-task--per-folder)

---

## 1. How the config system works

There are **three layers** behind every value the pipeline reads. Knowing which layer owns what
is the key to using `conf/` confidently.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 1 — Schema (code)   src/hydrabflow/config/schema.py                │
│    Typed @dataclass per group. Declares the fields, their types and       │
│    defaults, and which are MISSING (???) and must be supplied.            │
│    register_configs() stores each as `base_<group>` in Hydra's            │
│    ConfigStore.                                                            │
└───────────────────────────────┬─────────────────────────────────────────┘
                                 │ inherited via `defaults:` list
┌───────────────────────────────▼─────────────────────────────────────────┐
│  LAYER 2 — YAML values     conf/<group>/<choice>.yaml                     │
│    Each file starts with `- base_<group>` then fills in concrete values.  │
│    Multiple files per group = swappable *choices* (e.g. set_transformer   │
│    vs deep_set). config.yaml picks one choice per group.                  │
└───────────────────────────────┬─────────────────────────────────────────┘
                                 │ read by
┌───────────────────────────────▼─────────────────────────────────────────┐
│  LAYER 3 — Factories (code)                                               │
│    networks.factory, simulators.registry, preprocessing.registry,        │
│    augmentation.registry, pipeline.adapter — turn the validated config    │
│    into live objects. They read the dataclass fields by name.            │
└─────────────────────────────────────────────────────────────────────────┘
```

**Why structured dataclasses instead of `_target_`?** Every value you set in YAML is checked
against the schema *before* anything runs: a typo in a field name, a string where an int is
expected, or a forgotten mandatory value fails fast with a clear Hydra error rather than deep
inside Keras. The schema is the single source of truth for "what is configurable"; the YAML is
just values.

**Composition.** [`conf/config.yaml`](../conf/config.yaml) has a `defaults:` list that names one
choice from each group. Hydra merges them — plus the `base_config` schema — into one config tree
(`cfg`). Inside any YAML you can reference any other value with `${...}` interpolation
(e.g. `data_dir: data/${simulator.name}`); these resolve *after* the whole tree is composed.

**The `base_<group>` line.** Every group YAML begins with:

```yaml
defaults:
  - base_<group>      # pull in the dataclass schema for this group
```

This is what binds the YAML to its schema. Without it, Hydra would treat the file as an
unvalidated dict. Keep it as the first entry.

**`_self_`.** When a YAML composes sub-groups *and* sets its own values (e.g.
[`conf/model/default.yaml`](../conf/model/default.yaml)), `_self_` controls ordering. Listed
last, it means "my own values win over what the sub-groups set."

> The schema file [`src/hydrabflow/config/schema.py`](../src/hydrabflow/config/schema.py) and the
> registration in `register_configs()` are **fixed infrastructure**. You add YAML *choices*; you
> only touch the schema when introducing a genuinely new field (rare — see
> [§5](#5-step-by-step-adding-a-new-config-file)).

---

## 2. The root master config — `config.yaml`

[`conf/config.yaml`](../conf/config.yaml) is the entry point Hydra loads (`config_name="config"`,
wired in [`pipeline/_app.py`](../src/hydrabflow/pipeline/_app.py)). It does three things:

```yaml
defaults:
  - base_config          # the RootConfig schema (validates the whole tree)
  - simulator: two_moons # ── pick ONE choice from each group ───────────────
  - model: default
  - data: default
  - training: default
  - preprocessing: default
  - augmentation: default
  - adapter: default
  - inference: default
  - eval: default
  - tuning: default
  - _self_               # values below win over the group defaults

seed: 42                 # global RNG seed (utils.seed)

hydra:
  run:
    dir: outputs/${simulator.name}/${model.name}/${now:%Y-%m-%d_%H-%M-%S}
  sweep:
    dir: multirun/${simulator.name}/${model.name}/${now:%Y-%m-%d_%H-%M-%S}
    subdir: ${hydra.job.num}
```

| Part | Meaning |
| --- | --- |
| `defaults:` | The composition recipe. Each `group: choice` line selects `conf/group/choice.yaml`. Change a default here to change the project-wide default; override on the CLI for a one-off. |
| `seed` | Top-level field on `RootConfig`. Seeds NumPy/JAX for reproducibility. |
| `model_dir` | Not set here (defaults to `null`). `evaluate` / `evaluate_real` **require** it — point it at a completed `train` run dir. Pass it on the CLI: `model_dir=outputs/.../<timestamp>`. |
| `hydra.run.dir` | Where a single run writes its outputs (and its `.hydra/` config snapshot). Timestamped so runs never overwrite each other — the traceability convention documented in CLAUDE.md. |
| `hydra.sweep.dir` | Where `--multirun` sweeps land, one subdir per job. |

---

## 3. Config groups, one by one

For each group below: **what it controls**, the **schema** (from
[`schema.py`](../src/hydrabflow/config/schema.py)), the **shipped YAML choices**, and **how to
override** it.

---

### 3.1 `simulator/`

**Controls** the forward model — which simulator class runs, and the parameters passed to its
constructor. Resolved through the simulator **registry** (`@register_simulator("name")`).

**Schema — `SimulatorConfig`:**

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `name` | `str` | `???` (MISSING) | Registry key of the simulator class. Also feeds `${simulator.name}` in `data_dir` / output paths. |
| `params` | `dict` | `{}` | Free-form mapping passed verbatim to the simulator constructor. Declare whatever your simulator needs — no schema change required. |

`params` is intentionally untyped: the simulator class owns its own argument contract. Parameter
names and observable keys are reported by the class itself
(`BaseSimulator.parameter_names` / `observable_keys`), **not** by this config.

**Shipped choices:**

- [`two_moons.yaml`](../conf/simulator/two_moons.yaml) — the Two Moons benchmark (bimodal
  posterior). `params`: `prior_low/high`, `n_obs`, `mean_radius`, `std_radius`.
- [`skeleton.yaml`](../conf/simulator/skeleton.yaml) — placeholder that raises a clear
  `NotImplementedError`. Copy it to start a new simulator.

```yaml
# conf/simulator/two_moons.yaml
defaults:
  - base_simulator
name: two_moons
params:
  prior_low: -1.0
  prior_high: 1.0
  n_obs: 1
  mean_radius: 0.1
  std_radius: 0.01
```

**Override:** `simulator=skeleton` (switch choice) or `simulator.params.n_obs=10` (tweak one param).
The [`adapter`](#37-adapter) follows automatically: with the default (empty) adapter config its
variables are derived from the selected simulator's `parameter_names` / `observable_keys`.

---

### 3.2 `model/`

**Controls** the neural architecture: exactly **one summary network + one inference network**
(`bf.BasicWorkflow`, single-level inference). Built by
[`networks.factory`](../src/hydrabflow/networks/factory.py).

`model/default.yaml` is itself a *composing* config — it selects nested choices from two
sub-groups:

```yaml
# conf/model/default.yaml
defaults:
  - base_model
  - summary_network: set_transformer    # sub-group choice
  - inference_network: flow_matching    # sub-group choice
  - _self_
name: default
```

**Schema — `ModelConfig`:** `name` (`???`), `summary_network` (nested), `inference_network` (nested).

#### `model/summary_network/`

Encodes the raw observable into a fixed-length summary vector. **Schema —
`SummaryNetworkConfig`:**

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `type` | `str` | `set_transformer` | Registry key of the builder. Shipped: `set_transformer` \| `deep_set` \| `time_series_transformer`; custom builders self-register via `@register_summary_network` (see [end_to_end_guide.md §4](./end_to_end_guide.md#4-adding-a-summarynetwork-that-isnt-shipped)). |
| `summary_dim` | `int` | `32` | Output summary vector length. |
| `num_blocks` | `int` | `2` | Stacked encoder blocks (attention/MLP). |
| `num_heads` | `int` | `4` | Attention heads (ignored by `deep_set`). |
| `embed_dim` | `int` | `64` | Per-block embedding width (ignored by `deep_set`). |
| `mlp_depth` | `int` | `2` | MLP depth inside a block. |
| `mlp_width` | `int` | `128` | MLP width inside a block. |
| `dropout` | `float` | `0.05` | Dropout rate. |
| `params` | `dict` | `{}` | Free-form extras for custom builders (like `simulator.params`); shipped builders ignore it. |

Block-wise architectures expand the scalar fields into `num_blocks`-length tuples inside the
builder.

**Shipped choices:** [`set_transformer`](../conf/model/summary_network/set_transformer.yaml)
(permutation-invariant, attention — good default for unordered sets/point clouds),
[`deep_set`](../conf/model/summary_network/deep_set.yaml) (lighter, no attention — drops
`num_heads`/`embed_dim`), [`time_series_transformer`](../conf/model/summary_network/time_series_transformer.yaml)
(ordered sequences / curves).

#### `model/inference_network/`

The posterior (generative) network. **Schema — `InferenceNetworkConfig`:**

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `type` | `str` | `flow_matching` | Registry key of the builder. Shipped: `flow_matching` \| `diffusion`; custom builders self-register via `@register_inference_network` (see [end_to_end_guide.md §5](./end_to_end_guide.md#5-adding-an-inferencenetwork-that-isnt-shipped)). |
| `mlp_depth` | `int` | `4` | MLP depth. |
| `mlp_width` | `int` | `128` | MLP width. |
| `dropout` | `float` | `0.05` | Dropout rate. |
| `time_embedding_dim` | `int` | `32` | Used by the diffusion network only. |
| `params` | `dict` | `{}` | Free-form extras for custom builders; shipped builders ignore it. |

**Shipped choices:** [`flow_matching`](../conf/model/inference_network/flow_matching.yaml) (fast
sampling, strong default), [`diffusion`](../conf/model/inference_network/diffusion.yaml) (more
expressive, typically more sampling steps).

**Override examples:**

```bash
model/summary_network=deep_set                  # swap summary architecture
model/inference_network=diffusion               # swap inference architecture
model.summary_network.summary_dim=64            # tweak a single field
model.inference_network.mlp_width=256
```

> Note the slash vs dot: `model/inference_network=diffusion` selects a **group choice** (a file);
> `model.inference_network.mlp_width=256` overrides a **value** inside the composed tree.

---

### 3.3 `data/`

**Controls** dataset generation volume and on-disk layout (consumed by `simulate`, and the file
names `train`/`evaluate` look for).

**Schema — `DataConfig`:**

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `data_dir` | `str` | `data` | Root dir for generated `.npz`. Shipped YAML sets `data/${simulator.name}`. |
| `n_simulations` | `int` | `10000` | Number of prior draws / forward-model runs. |
| `chunk_size` | `int` | `1000` | Simulations per chunk (memory control during `simulate`). |
| `dataset_name` | `str` | `???` | Output filename. Shipped: `training_data_${data.n_simulations}.npz`. |
| `real_data_path` | `str?` | `null` | Path to a real-data `.npz` for `evaluate_real`. Null until you have one. |

```yaml
# conf/data/default.yaml
defaults:
  - base_data
data_dir: data/${simulator.name}
n_simulations: 10000
chunk_size: 1000
dataset_name: training_data_${data.n_simulations}.npz
real_data_path: null
```

**Note the interpolation pattern.** `dataset_name` embeds `${data.n_simulations}` so the file name
encodes its size, and the [`eval`](#39-eval) group's `test_dataset_name` uses the same trick. This
is why a training set and a test set can share one `data_dir` without colliding.

**Override:** `data.n_simulations=50000`, `data.real_data_path=/abs/path/real.npz`.

---

### 3.4 `training/`

**Controls** the BayesFlow training loop (`workflow.fit_offline`).

**Schema — `TrainingConfig`:**

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `n_epochs` | `int` | `50` | Training epochs. |
| `batch_size` | `int` | `512` | Minibatch size. |
| `learning_rate` | `float` | `1e-3` | Optimizer LR. |
| `optimizer` | `str` | `adam` | Optimizer name. |
| `validation_fraction` | `float` | `0.1` | Held-out fraction. Referenced by the preprocessing `train_val_split` step via `${training.validation_fraction}`. |
| `standardize` | `list[str]` | `[inference_variables, summary_variables]` | Keys BayesFlow z-scores **internally** during fit. |
| `verbose` | `int` | `2` | Keras verbosity. |

> **Two standardizations, don't confuse them.** `training.standardize` is BayesFlow's internal,
> in-graph z-scoring of inference targets + summary outputs. The
> [`preprocessing`](#35-preprocessing) `standardize` step is a separate, deterministic z-scoring of
> the **raw dataset on disk**, fitted on the train split and saved for replay at evaluation.

**Override:** `training.n_epochs=200 training.batch_size=256 training.learning_rate=5e-4`.

---

### 3.5 `preprocessing/`

**Controls** deterministic, whole-dataset transforms applied **once** before training and
**replayed** at inference from saved state. Each step is `{name: <registry key>, ...params}`;
**order is preserved**. Resolved through
[`preprocessing.registry`](../src/hydrabflow/preprocessing).

**Schema — `PreprocessingConfig`:** `steps: list[dict]` (empty by default).

```yaml
# conf/preprocessing/default.yaml
defaults:
  - base_preprocessing
steps:
  - name: drop_nan                                   # remove failed simulations
    keys: ${adapter.summary_variables}
  - name: train_val_split                            # hold out validation (fit on train only)
    validation_fraction: ${training.validation_fraction}
  - name: standardize                                # per-feature z-score; stats saved to run dir
    keys: ${adapter.summary_variables}
```

Each step's extra keys (`keys`, `validation_fraction`, …) are passed to the registered step.
Interpolations let preprocessing follow the adapter/training config automatically. The fitted
state (e.g. mean/std) is written to the run dir and reloaded by `evaluate` / `evaluate_real` — so
real-data inference applies the exact same transform.

**Override (list replacement):** to change steps from the CLI you replace the whole list, e.g.
`'preprocessing.steps=[{name:drop_nan,keys:[x]}]'`. For anything non-trivial, prefer adding a new
[`preprocessing` choice file](#5-step-by-step-adding-a-new-config-file).

---

### 3.6 `augmentation/`

**Controls** stochastic, **per-batch** transforms applied *inside* `fit_offline` (e.g.
observational noise). Distinct from preprocessing: random, re-drawn every batch, never saved.
Resolved through [`augmentation.registry`](../src/hydrabflow/augmentation).

**Schema — `AugmentationConfig`:** `steps: list[str]` (names) + `params: dict`. Both empty by
default.

```yaml
# conf/augmentation/default.yaml
defaults:
  - base_augmentation
steps: []     # e.g. [add_observation_noise], applied in order
params: {}    # shared params keyed for those steps
```

**Override:** `'augmentation.steps=[add_observation_noise]' 'augmentation.params={sigma:0.1}'`.

---

### 3.7 `adapter/`

**Controls** the BayesFlow structural transform that maps **raw dataset keys → BayesFlow roles**.
This is the glue between your simulator's output dict and the network. Built by
[`pipeline.adapter`](../src/hydrabflow/pipeline).

**Schema — `AdapterConfig`:**

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `inference_variables` | `list[str]` | `[]` | Parameter keys concatenated into the inference target. **Empty = derived from the simulator's `parameter_names`.** |
| `summary_variables` | `list[str]` | `[]` | Observable key(s) fed to the summary network. **Empty = derived from the simulator's `observable_keys`.** One entry = single observable; multiple = the fusion seam. |
| `inference_conditions` | `list[str]` | `[]` | Direct (non-summarized) conditions. |
| `drop` | `list[str]` | `[]` | Keys to discard. |

With the shipped [`default.yaml`](../conf/adapter/default.yaml) (all lists empty) the adapter
wires itself from the simulator class — the class is the single source of truth for its
parameter/observable names, so for a registered simulator you never repeat them in config
(`pipeline.adapter.fill_adapter_from_simulator` runs at every CLI entry).

Set the lists explicitly only to override the derivation:

```yaml
# conf/adapter/two_moons.yaml — the explicit form (equivalent to what two_moons derives)
defaults:
  - base_adapter
inference_variables: [theta1, theta2]   # the simulator's parameters
summary_variables: [x]                  # the observable -> summary network
inference_conditions: []
drop: []
```

The two cases that need the explicit form: training on data no registered simulator produced
([bring your own data](./bring_your_own_data.md) — with nothing to derive from, the pipeline
fails fast with an actionable error), and departing from the simulator's declaration (inferring a
parameter subset, multi-observable fusion).

**Override:** `'adapter.inference_variables=[theta1,theta2]'`,
`'adapter.summary_variables=[x,y]'` (enable fusion).

---

### 3.8 `inference/`

**Controls** posterior sampling at evaluation time.

**Schema — `InferenceConfig`:** `num_samples` (`1000`), `batch_size` (`256`).

```yaml
# conf/inference/default.yaml
defaults:
  - base_inference
num_samples: 1000
batch_size: 256
```

**Override:** `inference.num_samples=5000`.

---

### 3.9 `eval/`

**Controls** the `evaluate` stage: which held-out test set to score and which diagnostics to
produce.

**Schema — `EvalConfig`:**

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `test_dataset_name` | `str` | `???` | File `evaluate` loads from `data_dir`. Shipped: `test_data_${data.n_simulations}.npz`. |
| `num_samples` | `int` | `1000` | Posterior draws per test point. |
| `batch_size` | `int` | `256` | Sampling batch size. |
| `diagnostics` | `list[str]` | `[metrics, recovery, calibration_ecdf, z_score_contraction]` | Which diagnostic plots/metrics to emit. |

```yaml
# conf/eval/default.yaml
defaults:
  - base_eval
test_dataset_name: test_data_${data.n_simulations}.npz
num_samples: 1000
batch_size: 256
diagnostics: [metrics, recovery, calibration_ecdf, z_score_contraction]
```

> **The test file naming convention** is config-driven: `evaluate` looks for
> `${data.data_dir}/test_data_${data.n_simulations}.npz`, i.e.
> `data/<simulator.name>/test_data_<n_simulations>.npz`. Generate it by running `simulate` with a
> different seed and `data.dataset_name=test_data_${data.n_simulations}.npz`.

**Override:** `'eval.diagnostics=[metrics,recovery]'`,
`eval.test_dataset_name=my_test.npz`.

---

### 3.10 `tuning/`

**Controls** the Optuna multi-objective study (`tune` stage) — minimize RMSE *and* calibration
error over a config-driven search space.

**Schema — `TuningConfig`:**

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `study_name` | `str` | `hydrabflow_study` | Optuna study name. |
| `storage_dir` | `str` | `data/tuning` | Where the study log lives. Shipped YAML uses `${data.data_dir}/tuning`. |
| `n_trials` | `int` | `50` | Trials to run. |
| `directions` | `list[str]` | `[minimize, minimize]` | One per objective (RMSE, calibration). |
| `n_epochs` | `int` | `10` | Short training budget **per trial**. |
| `search_space` | `dict` | `{}` | `config.path -> {type, low, high, step, log, choices}`. |
| `save_artifacts` | `bool` | `True` | Persist each trial's model/posterior/plots + shared preprocessing state. |
| `artifacts_dir` | `str` | `${tuning.storage_dir}/${tuning.study_name}` | Where artifacts land, keyed by trial number. |

**The search space** maps a dotted config path to a sampling spec. The objective applies each
suggestion onto a *copy* of the config before building the model:

```yaml
search_space:
  model.summary_network.summary_dim:   {type: int,   low: 16,  high: 64}
  model.summary_network.num_heads:     {type: int,   low: 2,   high: 8}
  model.summary_network.embed_dim:     {type: int,   low: 32,  high: 128, step: 16}
  model.inference_network.mlp_depth:   {type: int,   low: 2,   high: 8}
  model.inference_network.mlp_width:   {type: int,   low: 32,  high: 256, step: 16}
  training.learning_rate:              {type: float, low: 1e-4, high: 1e-2, log: true}
```

Spec types: `int` / `float` (use `low`, `high`, optional `step`, `log: true` for log-scale) and
`categorical` (use `choices: [...]`). Add or remove entries freely — the key just has to be a real
dotted path into the config. The study itself is stored in a concurrency-safe JournalStorage
`.log` so many processes can fill one study cooperatively.

**Override:** `tuning.n_trials=100 tuning.n_epochs=20`.

---

## 4. Overriding configs from the CLI

Hydra exposes every group and value on the command line. Two distinct syntaxes:

| Goal | Syntax | Example |
| --- | --- | --- |
| **Select a group choice** (a different file) | `group=choice` (use `/` for nested groups) | `simulator=two_moons` · `model/inference_network=diffusion` |
| **Override a value** in the composed tree | `dotted.path=value` | `training.n_epochs=200` · `data.n_simulations=50000` |
| **Override a list** | quote it | `'adapter.inference_variables=[theta1,theta2]'` |
| **Add a key not in schema** | `+path=value` | `+training.clipnorm=1.0` |
| **Sweep (multirun)** | `--multirun` + comma lists | `--multirun model/inference_network=flow_matching,diffusion` |

```bash
# Single run, several overrides at once
uv run python scripts/train.py \
  simulator=two_moons \
  model/summary_network=deep_set \
  training.n_epochs=100 data.n_simulations=20000

# Sweep two inference networks × two learning rates (4 jobs)
uv run python scripts/train.py --multirun \
  model/inference_network=flow_matching,diffusion \
  training.learning_rate=1e-3,5e-4
```

CLI overrides always win over the `defaults:` in `config.yaml`. Inspect the fully-composed result
without running anything by appending `--cfg job`:

```bash
uv run python scripts/train.py --cfg job          # print resolved config and exit
uv run python scripts/train.py --help             # Hydra's auto-generated group/option help
```

---

## 5. Step-by-step: adding a new config file

Three cases, in increasing order of effort — and **most of the time it's Case A** (a new
value-only variant, zero code) or Case B (a dropped, self-registering module).

### Case A — a new *choice* in an existing group (no schema change)

Example: a bigger set-transformer summary network.

1. **Copy an existing choice** in the group:
   ```bash
   cp conf/model/summary_network/set_transformer.yaml \
      conf/model/summary_network/big_set_transformer.yaml
   ```
2. **Keep the `base_<group>` line**, edit the values:
   ```yaml
   # conf/model/summary_network/big_set_transformer.yaml
   defaults:
     - base_summary_network      # ← do not remove
   type: set_transformer
   summary_dim: 64
   num_blocks: 4
   num_heads: 8
   embed_dim: 128
   mlp_depth: 3
   mlp_width: 256
   dropout: 0.1
   ```
   Only fields declared on the group's dataclass are allowed — anything else fails validation.
3. **Use it** — either as a one-off override or as the new project default:
   ```bash
   # one-off
   uv run python scripts/train.py model/summary_network=big_set_transformer
   ```
   ```yaml
   # or make it the default in conf/model/default.yaml
   defaults:
     - base_model
     - summary_network: big_set_transformer
     - inference_network: flow_matching
     - _self_
   ```

The same recipe works for `simulator/`, `data/`, `training/`, `preprocessing/`, `augmentation/`,
`adapter/`, `inference/`, `eval/`, `tuning/`, and the two `model/*` sub-groups.

### Case B — a new registered component (still no schema change)

New simulators, preprocessing steps, augmentations, and summary / inference networks all follow
the same pattern: drop a module into the component's package, decorate it with the family's
`@register_*` decorator (it is auto-imported), and select it by name in YAML. The free-form
`params` mappings (`simulator.params`, `model.*.params`) carry any custom hyperparameters, so
the schema stays untouched. Fully worked network examples:
[`end_to_end_guide.md` §4–5](./end_to_end_guide.md); simulators: the project README and
[`bring_your_own_data.md`](./bring_your_own_data.md).

### Case C — a new typed field in the schema

Only when you want Hydra-side type checking for a new knob (instead of `params`): add the field
to the relevant dataclass in
[`src/hydrabflow/config/schema.py`](../src/hydrabflow/config/schema.py) (with a default), read it
in your registered builder/component, then create the YAML choice as in Case A.

> **After editing code, refresh the knowledge graph:** `graphify update .` (AST-only, no API
> cost), per the project's `CLAUDE.md`.

---

## 6. Using your own master config

The entry points load `config.yaml` from the repo's `conf/` because
[`pipeline/_app.py`](../src/hydrabflow/pipeline/_app.py) pins
`config_path=<repo>/conf` and `config_name="config"`. You do **not** edit that file to use your
own master config — Hydra lets you redirect both from the CLI.

### 6.1 A different master *name*, same `conf/` folder

Drop another root file beside `config.yaml`, e.g. `conf/config_prod.yaml`, then:

```bash
uv run python scripts/train.py --config-name config_prod
```

`config_prod.yaml` is a full root config — it needs its own `defaults:` list and `hydra:` block
(copy `config.yaml` and edit). This is the cleanest way to keep several named project profiles
(`config.yaml`, `config_prod.yaml`, `config_smoke.yaml`) in one place.

### 6.2 A master config in *your own* folder

Keep your configs outside the repo (e.g. in your experiment repo) and add that folder to Hydra's
search path with `--config-dir`:

```bash
uv run python scripts/train.py \
  --config-dir /path/to/my_experiment/conf \
  --config-name my_master
```

`--config-dir` is *prepended* to the search path, so your folder can:
- provide its own `my_master.yaml` root, **and**
- override or add group choices (a file at `/path/to/my_experiment/conf/simulator/mine.yaml` is
  selectable as `simulator=mine`), while still falling back to the repo's `conf/` for everything
  it doesn't define.

Your master config still composes against the **same schema** (the dataclasses are registered in
code, independent of folder), so validation and interpolation work exactly as before. A minimal
custom master:

```yaml
# /path/to/my_experiment/conf/my_master.yaml
defaults:
  - base_config
  - simulator: mine          # from your folder
  - adapter: mine            # from your folder
  - model: default           # falls back to the repo's conf/model/default.yaml
  - data: default
  - training: default
  - preprocessing: default
  - augmentation: default
  - inference: default
  - eval: default
  - tuning: default
  - _self_

seed: 7
hydra:
  run:
    dir: /path/to/my_experiment/outputs/${simulator.name}/${model.name}/${now:%Y-%m-%d_%H-%M-%S}
```

> `--config-dir` (add a folder to the search path) is the recommended, non-invasive route.
> `--config-path` also exists but is interpreted relative to the app file and is meant for app
> authors, not end users — prefer `--config-dir`.

---

## 7. Splitting configs per task / per folder

You don't need one giant master config. Two complementary strategies:

### 7.1 One root file per task (recommended)

Keep a separate, fully-formed root config per task and select it with `--config-name`:

```
conf/
├── config.yaml          # default / dev
├── config_train.yaml    # tuned for a long training run
├── config_tune.yaml     # bigger search space, more trials
└── config_eval.yaml     # pins model_dir, eval-only diagnostics
```

```bash
uv run python scripts/train.py    --config-name config_train
uv run python scripts/tune.py     --config-name config_tune
uv run python scripts/evaluate.py --config-name config_eval
```

Each is independent — it can pick different group choices and set different `hydra.run.dir`s.
Shared values still come from the same group files, so there's no duplication of the actual
knobs, only of the (small) `defaults:` recipe.

### 7.2 Profile sub-folders inside a group

When several knobs always move together, bundle them into a named choice rather than overriding
each on the CLU. For example, group all "quick smoke test" training settings into
`conf/training/smoke.yaml`:

```yaml
# conf/training/smoke.yaml
defaults:
  - base_training
n_epochs: 2
batch_size: 64
validation_fraction: 0.2
```

Then `training=smoke` swaps the whole bundle in one token. The same applies to bundling
eval-diagnostic sets, augmentation stacks, or preprocessing pipelines as named choices. This keeps
the CLI short and makes the *intent* (a named profile) explicit and reusable.

### 7.3 Per-task directories outside the repo

For fully separate experiments, give each its own `conf/` folder and point at it with
`--config-dir` (see [§6.2](#62-a-master-config-in-your-own-folder)). The repo's `conf/` remains
the shared base; each experiment folder only contains what it changes. This keeps experiment
configs versioned alongside *their* results, not inside the template.

---

## See also

- [`end_to_end_guide.md`](./end_to_end_guide.md) — run a full pipeline, add networks (Case B).
- [`two_moons_pipeline.md`](./two_moons_pipeline.md) — the shipped benchmark walkthrough.
- [`bring_your_own_data.md`](./bring_your_own_data.md) — wiring a new simulator + adapter.
- [`hyperparameter_tuning.md`](./hyperparameter_tuning.md) — the `tune` stage in depth.
- [`src/hydrabflow/config/schema.py`](../src/hydrabflow/config/schema.py) — the authoritative schema.
- Project `CLAUDE.md` — design principles, output-directory convention, what is fixed infrastructure.

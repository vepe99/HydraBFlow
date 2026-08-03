# Configuration reference

All configuration is **one file plus three groups**:

```
conf/
├── config.yaml                    # every knob: seed, data, training, preprocessing,
│                                  #   augmentation, adapter, eval, tuning, hydra
├── simulator/two_moons.yaml       # + your simulators
├── model/summary_network/         # set_transformer | deep_set | time_series_transformer
└── model/inference_network/       # flow_matching | diffusion
```

There are no `default.yaml` files and no group YAML inherits from another via a `defaults:` list.
The only `defaults:` list in the repo is at the top of `config.yaml`, which picks which group file to
use:

```yaml
defaults:
  - base_config                      # the dataclass schema (hydrabflow/config.py)
  - simulator: two_moons
  - model/summary_network: set_transformer
  - model/inference_network: flow_matching
  - _self_                           # this file's values win over the groups
```

**Validation.** `base_config` is `RootConfig` from `src/hydrabflow/config.py`, the single node in
Hydra's `ConfigStore`. Every group is a typed field of it, so a group YAML is validated just by being
selected — a misspelled key or a string where an int belongs fails at startup, not mid-run:

```
$ uv run hydrabflow-train training.no_such_knob=1
Could not override 'training.no_such_knob'. Key 'no_such_knob' is not in struct
```

Adding a component never requires touching the schema; adding a genuinely new *field* does (see
[Adding a typed field](#adding-a-typed-field)).

- [Overriding values](#overriding-values)
- [The blocks in config.yaml](#the-blocks-in-configyaml)
- [The three groups](#the-three-groups)
- [Interpolation](#interpolation)
- [Sweeps](#sweeps)
- [Adding a typed field](#adding-a-typed-field)
- [Your own master config](#your-own-master-config)

## Overriding values

Three ways, in increasing permanence:

```bash
# 1. On the command line (nothing is edited; the run still records what you used)
uv run hydrabflow-train training.n_epochs=5 model/summary_network=deep_set

# 2. Edit conf/config.yaml — the new default for every future run
# 3. Add a group file (conf/simulator/my_sim.yaml) and select it: simulator=my_sim
```

Syntax notes: `key=value` overrides, `+key=value` adds a key the schema does not define,
`group=choice` swaps a group file (note the `/` in `model/summary_network=deep_set` — that is a group
*path*, not a dotted key). Lists are `adapter.drop=[a,b]`; dicts are `simulator.params={n_obs:5}`.

## The blocks in `config.yaml`

### `seed`, `run_name`, `model_dir`

| Key         | Default                           | Meaning                                              |
| ----------- | --------------------------------- | ---------------------------------------------------- |
| `seed`      | `42`                              | Seeds Python, NumPy, and Keras; also threads an explicit generator through simulators and preprocessing |
| `run_name`  | `<summary_type>+<inference_type>` | The middle path element of the output dir; override to label an experiment |
| `model_dir` | `null`                            | A completed `train` run dir. Required by `evaluate`  |

### `data`

| Key              | Default                                   | Meaning                                     |
| ---------------- | ----------------------------------------- | ------------------------------------------- |
| `data_dir`       | `data`                                    | Where datasets are written and read         |
| `n_simulations`  | `10000`                                   | Rows to generate in `simulate`              |
| `chunk_size`     | `1000`                                    | Rows per forward-model call; each chunk is seeded from `(seed, offset)` so the chunk size never changes the dataset |
| `dataset_name`   | `training_data_${data.n_simulations}.npz` | The file `simulate` writes and `train`/`tune` read |
| `real_data_path` | `null`                                    | Set it and `evaluate` runs on your observed data instead of the test set |

### `training`

| Key                   | Default                 | Meaning                                         |
| --------------------- | ----------------------- | ----------------------------------------------- |
| `n_epochs`            | `50`                    | Offline training epochs                         |
| `batch_size`          | `512`                   | Halved automatically on GPU OOM                 |
| `learning_rate`       | `1e-3`                  | Optimizer learning rate                         |
| `optimizer`           | `adam`                  |                                                 |
| `validation_fraction` | `0.1`                   | Read by the `train_val_split` preprocessing step |
| `standardize`         | `[inference_variables]` | BayesFlow's *internal* z-scoring. Deliberately only the parameter side — the observable side is handled by the `standardize` preprocessing step, whose fitted state is saved and replayed at inference. Adding `summary_variables` here would standardize it twice |
| `verbose`             | `2`                     | Keras verbosity (`0` silences the epoch bar)    |
| `save_best_weights`   | `true`                  | Checkpoint the best val loss and restore it before saving |

### `preprocessing`

Deterministic, whole-dataset transforms: applied **once**, fit on the train split, state saved to the
run dir and replayed by `evaluate`. An ordered list; each entry is `{name: <registry key>, ...params}`
where the extra keys are the step's constructor arguments.

```yaml
preprocessing:
  steps:
    - name: drop_nan                # drop simulations containing NaN/Inf (failed forward runs)
      keys: ${adapter.summary_variables}
    - name: train_val_split         # steps listed AFTER this one are fit on the train split only
      validation_fraction: ${training.validation_fraction}
    - name: standardize             # per-feature z-score; mean/std -> preprocessing_state.npz
      keys: ${adapter.summary_variables}
```

Order matters: `drop_nan` before the split sees the whole dataset; `standardize` after it is fit on
train and applied to both splits. Custom steps: [extending.md](extending.md#a-preprocessing-step).

### `augmentation`

Stochastic, per-batch transforms applied **inside** `fit_offline` — re-drawn every batch and every
epoch, so they never touch the dataset on disk. Empty by default:

```yaml
augmentation:
  steps: [gaussian_noise]           # registry keys, applied in order
  params: {noise_key: x, noise_scale: 0.05}
```

`gaussian_noise` at the default `noise_scale: 0.0` is a no-op, which is why the shipped config trains
without augmentation. Each step gets its own child RNG derived from `seed`, so a step's random stream
does not depend on which other steps are enabled. Validation data passes through the same chain once,
with a fixed draw.

### `adapter`

Maps dataset keys to the roles BayesFlow expects. **Normally leave this alone** — empty lists are
filled from the selected simulator's `parameter_names` / `observable_keys` at startup:

```yaml
adapter:
  inference_variables: []           # <- simulator.parameter_names
  summary_variables: []             # <- simulator.observable_keys
  inference_conditions: []          # conditions fed directly, bypassing the summary network
  drop: []                          # keys the network must not see (but augmentations may read)
```

Set them explicitly for data no simulator produced (see
[running.md](running.md#bring-your-own-dataset-no-simulator)), or to infer a subset of parameters.
Explicit values always win over the derivation. Several `summary_variables` switches the adapter from
`rename` to `group`, which is the seam for multi-observable fusion
(`bayesflow.networks.FusionNetwork` in `networks/factory.py`).

### `eval`

| Key                 | Default                               | Meaning                                     |
| ------------------- | ------------------------------------- | ------------------------------------------- |
| `test_dataset_name` | `test_data_${data.n_simulations}.npz` | Read from `data.data_dir` unless `data.real_data_path` is set |
| `num_samples`       | `1000`                                | Posterior draws per observation              |
| `batch_size`        | `256`                                 | Observations per sampling batch (halved on OOM) |
| `diagnostics`       | `[metrics, recovery, calibration_ecdf, coverage, z_score_contraction]` | Drop names to skip them; `metrics` writes `metrics.json`. Ignored in real-data mode (no truth) |

### `tuning`

| Key              | Default                                      | Meaning                                  |
| ---------------- | -------------------------------------------- | ---------------------------------------- |
| `study_name`     | `hydrabflow_study`                           | Change it to start a new study            |
| `storage_dir`    | `${data.data_dir}/tuning`                    | Holds the study's `.log`                  |
| `artifacts_dir`  | `${tuning.storage_dir}/${tuning.study_name}` | Per-trial models/posteriors/plots         |
| `save_artifacts` | `true`                                       | Off = objectives only, no per-trial files |
| `n_trials`       | `50`                                         | Trials *this* process runs                |
| `n_epochs`       | `10`                                         | Short per-trial budget                    |
| `directions`     | `[minimize, minimize]`                       | RMSE, then calibration error              |
| `search_space`   | see below                                    | What is sampled                           |

`search_space` maps a **dotted config path** to a sampling spec, so anything in the config can be
tuned — including a simulator parameter:

```yaml
search_space:
  model.summary_network.summary_dim:        {type: int, low: 16, high: 64}
  model.summary_network.num_heads:          {type: int, low: 2, high: 8}
  model.summary_network.embed_dim_per_head: {type: int, low: 8, high: 32, step: 8}
  model.inference_network.mlp_depth:        {type: int, low: 2, high: 8}
  model.inference_network.mlp_width:        {type: int, low: 32, high: 256, step: 16}
  training.learning_rate:                   {type: float, low: 1e-4, high: 1e-2, log: true}
  model.inference_network.type:             {type: categorical, choices: [flow_matching, diffusion]}
```

Spec fields: `type` (`int` | `float` | `categorical`), `low`/`high`, `step`, `log` (float only),
`choices` (categorical only). Attention width is sampled *per head*, so `num_heads` and
`embed_dim_per_head` can be drawn independently and their product is always divisible by the head
count — no invalid trials.

### `hydra`

```yaml
hydra:
  run:
    dir: outputs/${simulator.name}/${run_name}/${now:%Y-%m-%d_%H-%M-%S}
```

Changing this changes where runs land; the `.hydra/` config snapshot inside is what makes a run
reproducible, so keep the timestamp (or something equally unique) in the path.

## The three groups

### `simulator/`

```yaml
# conf/simulator/two_moons.yaml
name: two_moons                     # must match @register_simulator("two_moons")
params:                             # free-form; reaches the simulator as self.params
  n_obs: 1                          # i.i.d. observations per parameter draw (the "set size")
  prior_low: -1.0
  prior_high: 1.0
  mean_radius: 0.1
  std_radius: 0.01
```

`params` is intentionally untyped so a new simulator needs no schema change. Select with
`simulator=two_moons`, override a knob with `simulator.params.n_obs=10`.

### `model/summary_network/`

Compresses one observation (or a set of them) into a fixed-size vector.

| File                           | `type`                    | Input it expects                      |
| ------------------------------ | ------------------------- | ------------------------------------- |
| `set_transformer.yaml`         | `set_transformer`         | exchangeable set, `(batch, n_obs, d)` |
| `deep_set.yaml`                | `deep_set`                | exchangeable set, cheaper             |
| `time_series_transformer.yaml` | `time_series_transformer` | ordered sequence, `(batch, time, d)`  |

Shared fields: `summary_dim` (output width), `num_blocks`, `num_heads`, `embed_dim_per_head`
(attention width **per head** — the total is `num_heads × embed_dim_per_head`), `mlp_depth`,
`mlp_width`, `dropout`, plus free-form `params` for custom builders.

### `model/inference_network/`

Learns the posterior over parameters given the summary.

| File                 | `type`          | Notes                                   |
| -------------------- | --------------- | --------------------------------------- |
| `flow_matching.yaml` | `flow_matching` | default; fast to train, fast to sample  |
| `diffusion.yaml`     | `diffusion`     | adds `time_embedding_dim`               |

Shared fields: `mlp_depth`, `mlp_width`, `dropout`, `time_embedding_dim` (diffusion only), `params`.

## Interpolation

`${...}` references another config value and is resolved at read time, so overriding the source
updates every use:

```bash
uv run hydrabflow-simulate data.n_simulations=2000   # -> data/training_data_2000.npz
```

Chains work (`tuning.artifacts_dir` -> `tuning.storage_dir` -> `data.data_dir`), as do
cross-references from a list entry (`keys: ${adapter.summary_variables}` picks up whatever the
simulator declared). `${now:%Y-%m-%d_%H-%M-%S}` is Hydra's timestamp resolver.

## Sweeps

`--multirun` (`-m`) runs the cross product, one job per combination, into
`multirun/<simulator>/<run_name>/<timestamp>/<job_num>/`:

```bash
uv run hydrabflow-train -m model/inference_network=flow_matching,diffusion \
                           training.learning_rate=1e-3,3e-4        # 4 jobs
```

Use a sweep to compare a handful of named alternatives; use `tune` for a *search* over continuous
ranges.

## Adding a typed field

Only needed for a genuinely new knob — a new simulator parameter belongs in `simulator.params`, and a
new network hyperparameter can go in the network's `params`. When you do need one, add it to the
dataclass in `src/hydrabflow/config.py` with a default, then set it in `conf/config.yaml`:

```python
@dataclass
class TrainingConfig:
    ...
    gradient_clip: float = 0.0      # new field, defaults to off
```

A default keeps every existing config and saved run valid.

## Your own master config

To keep experiment configs outside the repo, write a master file that includes the shipped one and
point Hydra at its folder:

```yaml
# /path/to/experiments/conf/my_run.yaml
defaults:
  - /config                         # the shipped conf/config.yaml
  - _self_

run_name: my_experiment
training:
  n_epochs: 200
data:
  data_dir: /scratch/my_data
```

```bash
uv run python -m hydrabflow.pipeline.train \
    --config-dir /path/to/experiments/conf --config-name my_run
```

`--config-dir` is searched *in addition to* the repo's `conf/`, so groups like
`model/summary_network=deep_set` keep working and your file only states what differs.

# Hyperparameter tuning

HydraBFlow's `tune` stage runs an [Optuna](https://optuna.org) study over a config-driven search
space. By default it is **multi-objective**, minimizing both posterior **RMSE** and **calibration
error** on the validation split, and it **saves every trial** — the trained model, posterior
samples, and all evaluation plots — so any trial can be inspected or reused later, not just the
best one.

Two properties make it production-ready out of the box:

- **Every run is saved by default** (`tuning.save_artifacts: true`). Each trial gets its own
  directory with `approximator.keras`, `posterior.npz`, `metrics.json`, and the diagnostic plots.
  The preprocessing is fit **once** and **shared** across every trial/model (it is data-level, not
  model-level), so it is stored a single time at the study root.
- **Concurrency-safe storage.** The study lives in a single `.log` file
  ([Optuna `JournalStorage`](https://optuna.readthedocs.io/en/stable/reference/storages.html)),
  which many processes can append to at once. You can launch the same command N times and they
  cooperatively run trials of **one shared study** — no SQLite lock contention, works on NFS.

For the conceptual background (config system, networks, simulators) see the
[end-to-end guide](end_to_end_guide.md). This document is the concrete "how to tune" recipe; it
uses the shipped [`two_moons`](two_moons_pipeline.md) example throughout.

---

## 1. Prerequisites

You need a **training dataset** to tune on (the study loads it, preprocesses it once, and splits it
into train/validation). Generate one exactly as for normal training:

```bash
uv sync
uv run python scripts/simulate.py simulator=two_moons data.n_simulations=10000
```

This writes `data/two_moons/training_data_10000.npz`. (`adapter.inference_variables` must list your
parameter keys — it ships set for `two_moons`.)

---

## 2. Run a study

```bash
uv run python scripts/tune.py \
  simulator=two_moons \
  data.n_simulations=10000 \
  tuning.n_trials=50 tuning.n_epochs=10
```

What happens, in order ([`pipeline/tune.py`](../src/hydrabflow/pipeline/tune.py)):

1. Load `training_data_10000.npz`, fit the preprocessing pipeline **once**, split into
   `train` / `val`. The fitted state is written to
   `data/two_moons/tuning/hydrabflow_study/preprocessing_state.npz` (shared by all trials).
2. Open/create the study in `data/two_moons/tuning/hydrabflow_study.log` (`load_if_exists=True`).
3. For each of `n_trials`: sample hyperparameters from the search space, apply them onto a config
   copy, build a fresh workflow, train for the short `n_epochs` budget, sample the posterior on
   `val`, and score `RMSE` + `calibration_error`.
4. With `save_artifacts: true`, persist the full trial under `trials/trial_<number>/`.
5. Write `best_trials.json` (the Pareto-optimal trials for a multi-objective study).

> **Short budget.** `tuning.n_epochs` (default `10`) is deliberately smaller than
> `training.n_epochs` — tuning compares architectures cheaply. After you pick a winner, retrain it
> fully with `scripts/train.py` for the real model.

The console entry point `uv run hydrabflow-tune` is equivalent (same Hydra app).

---

## 3. What gets saved

```
data/two_moons/tuning/                         #  = ${data.data_dir}/tuning  = ${tuning.storage_dir}
├── hydrabflow_study.log                        # Optuna JournalStorage (concurrency-safe)
└── hydrabflow_study/                            #  = ${tuning.artifacts_dir}
    ├── preprocessing_state.npz                 # fit once, SHARED across every trial/model
    ├── best_trials.json                        # best / Pareto trials (+ their artifact dirs)
    └── trials/
        ├── trial_0000/
        │   ├── approximator.keras              # trained model for this trial
        │   ├── posterior.npz                   # posterior samples on the val split
        │   ├── metrics.json                    # RMSE + calibration error (values + means)
        │   ├── loss.png
        │   ├── recovery.png
        │   ├── calibration_ecdf.png
        │   └── z_score_contraction.png
        ├── trial_0001/
        └── ...
```

Trial folders are keyed by the **study-global Optuna trial number**, so concurrent processes fill
one `trials/` directory without ever colliding. The per-process Hydra run dir
(`outputs/two_moons/default/<timestamp>/`) also receives a copy of `best_trials.json` for full
traceability of that specific invocation.

The diagnostics are the **same truth-aware diagnostics the [evaluate](../src/hydrabflow/pipeline/evaluate.py)
stage produces** — the validation split carries ground-truth parameters, so RMSE, calibration
ECDF, recovery, and z-score contraction are all meaningful per trial. Which plots are produced is
controlled by `eval.diagnostics`.

To disable artifact saving and keep only `best_trials.json`:

```bash
uv run python scripts/tune.py ... tuning.save_artifacts=false
```

To put artifacts somewhere else, override `tuning.artifacts_dir` (defaults to
`${tuning.storage_dir}/${tuning.study_name}`).

---

## 4. Run many processes at once (parallel tuning)

Because the study is a `.log` `JournalStorage`, simply launch the **same command** multiple times —
in several terminals, or as array jobs on a cluster. They share `study_name` + `storage_dir`, so
they extend one study and divide the trials between them:

```bash
# terminal 1
uv run python scripts/tune.py simulator=two_moons \
  data.n_simulations=10000 tuning.n_trials=25 &

# terminal 2 (identical) — joins the SAME study
uv run python scripts/tune.py simulator=two_moons \
  data.n_simulations=10000 tuning.n_trials=25 &
```

Together they complete ~50 trials of `hydrabflow_study`, all writing into the same `trials/`
directory keyed by trial number. Each process trains on its own GPU — the
[`autocvd` GPU pin](../README.md#design-at-a-glance) gives each worker a different free GPU
automatically (`HYDRABFLOW_NUM_GPUS=1` per process, the default).

> **SLURM tip.** Submit an array job where every task runs the identical `tune.py` command with the
> same `study_name`/`storage_dir` on a shared filesystem. `n_trials` is *per task*; total trials =
> `n_trials × array_size`. Use a distinct `study_name` per experiment to avoid mixing studies.

---

## 5. Reading the results

`best_trials.json` lists the winning trial(s), their hyperparameters, objective values, and the
path to each trial's saved artifacts:

```json
[
  {
    "number": 17,
    "values": [0.42, 0.031],
    "params": {"model.summary_network.summary_dim": 48, "training.learning_rate": 0.0021, ...},
    "artifact_dir": "data/two_moons/tuning/hydrabflow_study/trials/trial_0017"
  }
]
```

You can load any trial's model directly from its directory (it is a normal training-run layout —
`approximator.keras` + the shared `preprocessing_state.npz` one level up). Or query the study
programmatically:

```python
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

storage = JournalStorage(JournalFileBackend("data/two_moons/tuning/hydrabflow_study.log"))
study = optuna.load_study(study_name="hydrabflow_study", storage=storage)
for t in study.best_trials:           # Pareto front (multi-objective)
    print(t.number, t.values, t.params)
```

**To get a production model, retrain the winning config fully.** Read the best `params` and pass
them as overrides to `train.py`, e.g.:

```bash
uv run python scripts/train.py simulator=two_moons \
  data.n_simulations=10000 training.n_epochs=100 \
  model.summary_network.summary_dim=48 training.learning_rate=0.0021
```

---

## 6. Changing what is tuned (the search space)

The search space maps a **dotted config path** to a sampling spec. Edit
[`conf/tuning/default.yaml`](../conf/tuning/default.yaml) (or override on the CLI). Any config path
is fair game — network fields, training fields, even simulator knobs.

```yaml
search_space:
  model.summary_network.summary_dim:      # -> trial.suggest_int(16, 64)
    type: int
    low: 16
    high: 64
  model.summary_network.embed_dim:        # int with a step
    type: int
    low: 32
    high: 128
    step: 16
  training.learning_rate:                  # log-uniform float
    type: float
    low: 1e-4
    high: 1e-2
    log: true
  model.inference_network.type:            # categorical
    type: categorical
    choices: [flow_matching, diffusion]
```

Spec keys, by `type` ([`_suggest`](../src/hydrabflow/pipeline/tune.py)):

| `type`        | required        | optional            | maps to                       |
|---------------|-----------------|---------------------|-------------------------------|
| `int`         | `low`, `high`   | `step` (default 1)  | `trial.suggest_int`           |
| `float`       | `low`, `high`   | `step`, `log`       | `trial.suggest_float`         |
| `categorical` | `choices`       | —                   | `trial.suggest_categorical`   |

Add or remove entries freely — anything **not** in the search space keeps its configured value.
A one-off space can be passed on the CLI:

```bash
uv run python scripts/tune.py ... \
  'tuning.search_space={training.learning_rate:{type:float,low:1e-4,high:1e-2,log:true}}'
```

### Single-objective instead of RMSE + calibration

`directions` controls the objective(s). The default `[minimize, minimize]` optimizes RMSE **and**
calibration error (Pareto). For a single objective (RMSE only), set one direction:

```bash
uv run python scripts/tune.py ... tuning.directions=[minimize]
```

With one direction the objective returns RMSE alone and `best_trials.json` holds the single best
trial; with two it returns `(rmse, calibration_error)` and holds the Pareto front. (To change
*which* metrics are optimized, edit the `bf_metrics` calls in
[`_objective`](../src/hydrabflow/pipeline/tune.py).)

---

## 7. Key config reference (`tuning` group)

| Field | Default | Meaning |
|-------|---------|---------|
| `study_name` | `hydrabflow_study` | Optuna study name; also the artifacts subdir and `.log` name. |
| `storage_dir` | `${data.data_dir}/tuning` | Where the `<study_name>.log` study file lives. |
| `n_trials` | `50` | Trials to run **this process** (total = sum over concurrent processes). |
| `n_epochs` | `10` | Short per-trial training budget. |
| `directions` | `[minimize, minimize]` | Objectives: RMSE, then calibration error. |
| `search_space` | (see YAML) | Dotted-path → sampling spec. |
| `save_artifacts` | `true` | Save model + posterior + plots per trial, and shared preprocessing. |
| `artifacts_dir` | `${tuning.storage_dir}/${tuning.study_name}` | Root of per-trial artifacts. |

---

## 8. Command recap

```bash
uv sync

# 1. dataset to tune on
uv run python scripts/simulate.py simulator=two_moons data.n_simulations=10000

# 2. run the study (repeat the SAME command in parallel to share it)
uv run python scripts/tune.py simulator=two_moons \
  data.n_simulations=10000 tuning.n_trials=50 tuning.n_epochs=10

# 3. inspect data/two_moons/tuning/hydrabflow_study/best_trials.json, then retrain the winner fully
uv run python scripts/train.py simulator=two_moons \
  data.n_simulations=10000 training.n_epochs=100 <best params as overrides>
```

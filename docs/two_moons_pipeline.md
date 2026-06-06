# Running a full pipeline with the Two Moons simulator

This is a copy-pasteable walkthrough of a **complete HydraFlow run — training *and* evaluation —**
on the shipped [`two_moons`](../src/hydraflow/simulators/two_moons.py) simulator (the classic
bimodal SBI benchmark). Nothing here needs code changes: the simulator, its config, and a matching
adapter all ship with the template. You only run the stages.

For the conceptual background (config system, adding your own simulator / networks, tuning) see the
[end-to-end guide](end_to_end_guide.md). This document is the concrete "press these buttons" recipe.

---

## 0. What you're running

The Two Moons forward model has two parameters `theta1`, `theta2` (uniform prior on `[-1, 1]`) and
produces a 2-D observation whose posterior is famously crescent-shaped / bimodal — a good stress
test for the inference network.

Two config groups make it work, both already in the repo:

| File | Role |
|------|------|
| [`conf/simulator/two_moons.yaml`](../conf/simulator/two_moons.yaml) | selects the `two_moons` forward model + its prior / noise knobs |
| [`conf/adapter/two_moons.yaml`](../conf/adapter/two_moons.yaml) | maps `theta1,theta2` → inference variables and `x` → summary variable |

Observable shape is `(n, n_obs, 2)`: `n_obs` is the number of i.i.d. observations per parameter
(the summary-network "set size", default `1`). The default model
([`conf/model/default.yaml`](../conf/model/default.yaml)) — a SetTransformer summary network + a
FlowMatching inference network — consumes that directly.

Every command below selects both groups with `simulator=two_moons adapter=two_moons`.

---

## 1. Prerequisites

```bash
uv sync          # create .venv and install BayesFlow / JAX / Hydra / ...
```

The JAX backend for Keras is pinned automatically (`hydraflow.utils.backend`), so you don't set
`KERAS_BACKEND` yourself. All commands are run with `uv run` so they use the project venv.

---

## 2. The four commands (full run)

The pipeline is **simulate (train set) → simulate (test set) → train → evaluate**. Datasets land in
`data/${simulator.name}/` (i.e. `data/two_moons/`); model artifacts land in a timestamped
`outputs/two_moons/default/<timestamp>/` directory.

> The file names are config-interpolated from `data.n_simulations`. Keep `data.n_simulations` the
> **same** across all four commands so the train stage writes `training_data_<N>.npz` and the
> evaluate stage looks for `test_data_<N>.npz` with the matching `<N>`.

### 2.1 Generate the training set

```bash
uv run python scripts/simulate.py \
  simulator=two_moons adapter=two_moons \
  data.n_simulations=10000
```

Writes `data/two_moons/training_data_10000.npz` (keys: `theta1`, `theta2`, `x`).

### 2.2 Generate a held-out test set

Same simulator, but a **different output name** (`eval` looks for `test_data_<N>.npz`) and a
**different `seed`** so the test draws are independent of the training draws:

```bash
uv run python scripts/simulate.py \
  simulator=two_moons adapter=two_moons \
  data.n_simulations=10000 \
  data.dataset_name=test_data_10000.npz \
  seed=123
```

Writes `data/two_moons/test_data_10000.npz`.

### 2.3 Train

```bash
uv run python scripts/train.py \
  simulator=two_moons adapter=two_moons \
  data.n_simulations=10000 \
  training.n_epochs=50
```

This loads the training set, runs preprocessing (NaN drop → train/val split → z-score
standardization, fitted on train and saved), builds the workflow, and trains via
`fit_offline`. It writes a **timestamped run directory**, printed at the end:

```
Training complete. Artifacts in outputs/two_moons/default/2026-06-06_15-31-32
```

Copy that path — the evaluate stage needs it. Capture it programmatically if you prefer:

```bash
RUN_DIR=$(ls -dt outputs/two_moons/default/*/ | head -1)
echo "$RUN_DIR"
```

The run dir contains `approximator.keras`, `preprocessing_state.npz`, `loss.png`, and a `.hydra/`
folder with the fully resolved config (full traceability).

### 2.4 Evaluate

Point `model_dir` at the train run dir. The evaluate stage reloads the approximator **and** the
fitted preprocessing, replays it on the test set (no re-fit), samples the posterior, and writes
truth-aware diagnostics:

```bash
uv run python scripts/evaluate.py \
  simulator=two_moons adapter=two_moons \
  data.n_simulations=10000 \
  model_dir="$RUN_DIR"
```

(If you didn't capture `$RUN_DIR`, paste the literal path:
`model_dir=outputs/two_moons/default/2026-06-06_15-31-32`.)

This writes, into its own timestamped `outputs/two_moons/default/<timestamp>/`:

- `posterior.npz` — posterior samples for every test observation
- `metrics.json` — RMSE + calibration error
- `recovery.png`, `calibration_ecdf.png`, `z_score_contraction.png` — diagnostic plots

For Two Moons, expect the recovery plot to show good capture of both parameters and the calibration
ECDF to sit near the diagonal once trained long enough.

---

## 3. Fast smoke run (≈1 minute)

To confirm everything is wired before committing to a long run, shrink every axis. Use a small
dataset, few epochs, and few posterior samples:

```bash
# small train + test sets
uv run python scripts/simulate.py simulator=two_moons adapter=two_moons \
  data.n_simulations=2000 data.chunk_size=1000
uv run python scripts/simulate.py simulator=two_moons adapter=two_moons \
  data.n_simulations=2000 data.chunk_size=1000 \
  data.dataset_name=test_data_2000.npz seed=123

# quick train
uv run python scripts/train.py simulator=two_moons adapter=two_moons \
  data.n_simulations=2000 training.n_epochs=3

# evaluate against the run just produced
RUN_DIR=$(ls -dt outputs/two_moons/default/*/ | head -1)
uv run python scripts/evaluate.py simulator=two_moons adapter=two_moons \
  data.n_simulations=2000 eval.num_samples=200 model_dir="$RUN_DIR"
```

(Diagnostics from a 3-epoch run will be poor — this only proves the plumbing.)

---

## 4. Optional: train with observational noise (augmentations)

The augmentation module adds **per-batch, stochastic-but-seed-controlled** perturbations during
training (re-drawn every epoch, reproducible from `cfg.seed`). To train Two Moons with additive
observational noise on `x`, enable a step and set its strength on the CLI:

```bash
uv run python scripts/train.py \
  simulator=two_moons adapter=two_moons \
  data.n_simulations=10000 training.n_epochs=50 \
  'augmentation.steps=[gaussian_noise]' \
  +augmentation.params.noise_scale=0.02
```

Available example augmentations (see [`augmentation/examples.py`](../src/hydraflow/augmentation/examples.py)):
`gaussian_noise` (additive), `multiplicative_noise` (gain jitter, `+augmentation.params.mult_scale=`),
`feature_dropout` (random masking, `+augmentation.params.dropout_prob=`). Each is a no-op at its
default strength. Evaluate exactly as in step 2.4 (augmentations apply at training time only).

---

## 5. Tuning the prior / observation knobs (optional)

The simulator exposes its knobs through `simulator.params`. For example, turn the single-observation
benchmark into a 10-observation set (sharper posterior) by overriding `n_obs` on **every** stage:

```bash
... simulator=two_moons adapter=two_moons simulator.params.n_obs=10 ...
```

The observable then has shape `(n, 10, 2)` and the SetTransformer pools over the 10 observations.
Other knobs: `simulator.params.prior_low` / `prior_high`, `mean_radius`, `std_radius`
(see [`conf/simulator/two_moons.yaml`](../conf/simulator/two_moons.yaml)).

For Optuna hyperparameter search over network / training settings, see
[§6–7 of the end-to-end guide](end_to_end_guide.md).

---

## 6. Command recap

```bash
uv sync

# 1. train set
uv run python scripts/simulate.py simulator=two_moons adapter=two_moons data.n_simulations=10000
# 2. test set (different name + seed)
uv run python scripts/simulate.py simulator=two_moons adapter=two_moons data.n_simulations=10000 \
  data.dataset_name=test_data_10000.npz seed=123
# 3. train
uv run python scripts/train.py simulator=two_moons adapter=two_moons data.n_simulations=10000 \
  training.n_epochs=50
# 4. evaluate
RUN_DIR=$(ls -dt outputs/two_moons/default/*/ | head -1)
uv run python scripts/evaluate.py simulator=two_moons adapter=two_moons data.n_simulations=10000 \
  model_dir="$RUN_DIR"
```

| Stage | Reads | Writes |
|-------|-------|--------|
| simulate (train) | — | `data/two_moons/training_data_10000.npz` |
| simulate (test) | — | `data/two_moons/test_data_10000.npz` |
| train | `training_data_10000.npz` | `outputs/two_moons/default/<ts>/` (`approximator.keras`, `preprocessing_state.npz`, `loss.png`) |
| evaluate | `test_data_10000.npz` + `model_dir` | `outputs/two_moons/default/<ts>/` (`posterior.npz`, `metrics.json`, diagnostic plots) |

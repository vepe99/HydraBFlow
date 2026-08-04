# Running HydraBFlow

Install, then run the four stages. Everything here works out of the box on the shipped
**Two Moons** benchmark — no code changes, no simulator to write yet.

- [Install](#install)
- [The four stages](#the-four-stages)
- [Full walkthrough (Two Moons)](#full-walkthrough-two-moons)
- [Smoke run (~1 minute)](#smoke-run-1-minute)
- [Inference on your own observed data](#inference-on-your-own-observed-data)
- [Hyperparameter tuning](#hyperparameter-tuning)
- [Bring your own dataset (no simulator)](#bring-your-own-dataset-no-simulator)
- [What lands where](#what-lands-where)
- [GPUs, CPU-only, and OOM](#gpus-cpu-only-and-oom)

## Install

```bash
uv sync          # creates .venv and installs everything, including pytest + ruff
uv run pytest -q # optional: 36 tests, no GPU needed
```

## The four stages

Each stage is a Hydra app. There are two equivalent ways to call one:

```bash
uv run hydrabflow-train                        # console script
uv run python -m hydrabflow.pipeline.train     # module form (identical)
```

| Stage      | Command               | Reads                                   | Writes                                          |
| ---------- | --------------------- | --------------------------------------- | ----------------------------------------------- |
| `simulate` | `hydrabflow-simulate` | prior + forward model                   | a dataset `.npz` in `data.data_dir`             |
| `train`    | `hydrabflow-train`    | that `.npz`                             | `approximator.keras`, preprocessing state, loss |
| `evaluate` | `hydrabflow-evaluate` | a trained `model_dir` + test/real data  | `posterior.npz` + diagnostics                   |
| `tune`     | `hydrabflow-tune`     | that `.npz`                             | an Optuna study + per-trial artifacts           |

Any config value can be overridden on the command line (`training.n_epochs=5`,
`model/summary_network=deep_set`, `simulator=my_sim`). See
[configuration.md](configuration.md).

## Full walkthrough (Two Moons)

The defaults are 10 000 simulations and 50 epochs — a few minutes on a laptop.

**1. Training set.** Writes `data/training_data_10000.npz` (from `data.dataset_name`, which
interpolates `data.n_simulations`):

```bash
uv run hydrabflow-simulate
```

**2. Held-out test set.** Same command with a different name *and* a different seed, so it is not
the data you trained on. The name must match `eval.test_dataset_name`, which defaults to
`test_data_${data.n_simulations}.npz`:

```bash
uv run hydrabflow-simulate data.dataset_name=test_data_10000.npz seed=123
```

Each dataset gets a `<name>.hydra/` config snapshot next to it, so you can always tell which config
produced which file.

**3. Train.**

```bash
uv run hydrabflow-train
```

This creates `outputs/two_moons/set_transformer+flow_matching/<timestamp>/` — the path is
`outputs/<simulator>/<run_name>/<timestamp>`, and `run_name` defaults to the two network types.
Override it to label an experiment: `run_name=wider_summary`. The final log line prints the run
directory; that is the `model_dir` the next step needs.

**4. Evaluate.**

```bash
uv run hydrabflow-evaluate model_dir=outputs/two_moons/set_transformer+flow_matching/<timestamp>
```

Or grab the newest run automatically:

```bash
uv run hydrabflow-evaluate model_dir=$(ls -dt outputs/two_moons/*/*/ | head -1)
```

You get `metrics.json` (RMSE + calibration error, per parameter and averaged) plus
`recovery.png`, `calibration_ecdf.png`, `coverage.png`, `z_score_contraction.png`. For Two Moons
after a full run, expect an RMSE around 0.3–0.5 per parameter; the posterior is deliberately
bimodal, so `recovery.png` will *not* be a tight diagonal — check `calibration_ecdf.png` instead,
which should stay inside its confidence band.

## Smoke run (~1 minute)

Same four commands, tiny numbers — use this to check an installation or a new simulator end to end.
`data.data_dir` keeps the throwaway data out of your real one:

```bash
D=/tmp/hbf_smoke
uv run hydrabflow-simulate data.n_simulations=600 data.data_dir=$D
uv run hydrabflow-simulate data.n_simulations=600 data.data_dir=$D data.dataset_name=test_data_600.npz seed=7
uv run hydrabflow-train    data.n_simulations=600 data.data_dir=$D training.n_epochs=2 training.batch_size=64
uv run hydrabflow-evaluate data.n_simulations=600 data.data_dir=$D eval.num_samples=200 \
    model_dir=$(ls -dt outputs/two_moons/*/*/ | head -1)
```

The metrics from a 2-epoch run are meaningless; that all four stages complete and write their
artifacts is the point.

## Inference on your own observed data

`evaluate` is one stage with two modes. Set `data.real_data_path` and it reads your `.npz` instead
of the simulated test set:

```bash
uv run hydrabflow-evaluate model_dir=outputs/two_moons/.../<timestamp> \
    data.real_data_path=/path/to/observations.npz
```

Differences in real mode: there is no ground truth, so the truth-aware diagnostics are skipped and
you get `posterior_pairs.png` per observation instead (`posterior_pairs_obs0.png`,
`posterior_pairs_obs1.png`, … when the file holds several observations). No resimulation happens.

Your file must contain the observable keys the model was trained on (for Two Moons: `x`, shaped
`(n_observations, n_obs, 2)`) — the same keys the simulator declares in `observable_keys`. The
fitted preprocessing from `model_dir` is replayed on it, so your data is scaled exactly like the
training data was.

## Hyperparameter tuning

An Optuna study minimizing **RMSE and calibration error** together (a Pareto front, not one
"best"). It needs a training dataset, nothing else:

```bash
uv run hydrabflow-tune                       # 50 trials × 10 epochs by default
uv run hydrabflow-tune tuning.n_trials=10 tuning.n_epochs=5
```

The dataset is loaded and preprocessed once and shared by every trial; each trial applies its
sampled hyperparameters, trains a fresh workflow, and is scored on the validation split.

**Results.** `best_trials.json` (in the run dir *and* next to the trial folders) lists each Pareto
trial's `values`, `params`, and `artifact_dir`. With `tuning.save_artifacts=true` (the default),
every trial keeps its own model, posterior, and diagnostics under
`data/tuning/hydrabflow_study/trials/trial_0007/`. Retrain a winner properly by copying its params
onto a normal `train` run:

```bash
uv run hydrabflow-train model.summary_network.summary_dim=48 model.inference_network.mlp_depth=6
```

**Parallel workers.** The study lives in a Journal-backed `.log` file that is safe for many
processes to append to, so running the *same* command in N terminals cooperatively fills one study:

```bash
uv run hydrabflow-tune &   # terminal 1
uv run hydrabflow-tune &   # terminal 2 — joins the SAME study
```

Keep `tuning.study_name` and `tuning.storage_dir` identical across workers; change `study_name` to
start a fresh study rather than extend the old one. Trial numbers are study-global, so concurrent
workers never collide on an artifact directory.

Which knobs are searched is `tuning.search_space` in `conf/config.yaml` — see
[configuration.md](configuration.md#tuning).

## Bring your own dataset (no simulator)

If your simulations already exist, you can skip `simulate` entirely. Write one `.npz` whose keys are
arrays with the number of simulations as the leading axis — parameters shaped `(n, 1)` and
observables shaped `(n, ...)`:

```python
import numpy as np
np.savez("data/my_train.npz", theta1=t1, theta2=t2, x=x)  # t1: (n,1)  x: (n, n_obs, 2)
```

Because no simulator class exists to declare the names, the adapter cannot derive them — set them
explicitly (this is the one case where the `adapter:` block in `conf/config.yaml` matters):

```bash
uv run hydrabflow-train \
    data.dataset_name=my_train.npz \
    adapter.inference_variables=[theta1,theta2] \
    adapter.summary_variables=[x] \
    simulator.name=my_data                     # any label; names the output folder
```

Then evaluate as usual — with a test `.npz` that includes the ground-truth parameters for
truth-aware diagnostics, or via `data.real_data_path` for observations without truth. To read a
format other than `.npz`, `pipeline/io.py:load_dataset` is the single seam: it returns
`{key: array}` and nothing else in the pipeline touches files.

## What lands where

Every run writes a `.hydra/` folder with its fully resolved config, so a run directory alone is
enough to reconstruct the run:

```
outputs/<simulator>/<run_name>/<timestamp>/
├── .hydra/                       # resolved config (Hydra writes this)
├── <stage>.log                   # train.log, evaluate.log, …
├── approximator.keras            # train
├── approximator_best.weights.h5  # train: best val loss, restored before saving
├── preprocessing_state.npz       # train: fitted transforms, replayed by evaluate
├── history.json  loss.png        # train
├── posterior.npz                 # evaluate
├── metrics.json + 4 plots        # evaluate, simulated test set
├── posterior_pairs*.png          # evaluate, real data
└── best_trials.json              # tune
```

Datasets go to `data.data_dir` (not the run dir), each with a `<name>.hydra/` snapshot.

## GPUs, CPU-only, and OOM

The Keras backend is JAX, pinned before any import. GPU selection happens at import too, via
`autocvd`, which picks the least-used device:

```bash
HYDRABFLOW_NUM_GPUS=2 uv run hydrabflow-train   # expose two GPUs
HYDRABFLOW_NUM_GPUS=0 uv run hydrabflow-train   # force CPU-only
CUDA_VISIBLE_DEVICES=3 uv run hydrabflow-train  # pick it yourself; autocvd is skipped
KERAS_BACKEND=torch uv run hydrabflow-train     # override the backend
```

If a batch does not fit on the card, the batch size is halved and retried automatically (down to 16;
in tuning, a trial that still OOMs at 512 is pruned instead of crawling). A diverged run is stopped
by `TerminateOnNaN`, and the best-val-loss weights are restored before saving — a late NaN spike
cannot destroy a converged model.

## See also

- [configuration.md](configuration.md) — every knob in `conf/config.yaml` and how to override it
- [extending.md](extending.md) — add a simulator, network, preprocessing step, or augmentation
- [hydra.md](hydra.md) — how config, registries, and `@register_*` fit together; which file owns what

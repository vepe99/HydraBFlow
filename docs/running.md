# Running

## Install

```bash
uv sync            # .venv + all deps (incl. pytest, ruff)
uv run pytest -q   # optional sanity check, CPU only
```

## The four stages

Each stage is a Hydra app: `uv run hydrabflow-<stage>` (or `python -m hydrabflow.pipeline.<stage>`).
Any config value can be overridden on the command line.

| Stage      | Reads                              | Writes                                                        | Where |
| ---------- | ---------------------------------- | ------------------------------------------------------------- | ----- |
| `simulate` | prior + forward model              | dataset `.npz` (+ `<stem>.hydra/` config snapshot)            | `data.data_dir` |
| `train`    | that `.npz`                        | `approximator.keras`, `preprocessing_state.npz`, `loss.png`, `history.json` | run dir |
| `evaluate` | `model_dir` + test (or real) data  | `posterior.npz`, `metrics.json`, diagnostic plots             | run dir |
| `tune`     | that `.npz`                        | `best_trials.json` → run dir; study + per-trial models → `tuning.storage_dir` / `artifacts_dir` | both |

### Where output goes

There is **no per-stage `output_dir` config key**. One path setting, `hydra.run.dir`, decides it for
every stage:

```yaml
hydra:
  run:
    dir: outputs/${simulator.name}/${run_name}/${now:%Y-%m-%d_%H-%M-%S}
```

Each launch gets its own timestamped dir, Hydra drops the resolved config into its `.hydra/`
subfolder, and the stage writes its artifacts beside it — so a run is reproducible from its own
folder, and reruns never overwrite each other. Adding explicit output dirs per config block would
mean copying that config snapshot by hand in every stage and losing the append-only timestamping.

Two things are deliberately *not* in the run dir, because they must outlive a single launch:

- **datasets** (`data.data_dir`) — one `.npz` is reused by many training runs; `simulate` pays for
  this by copying the config snapshot next to the file itself.
- **the Optuna study and trial artifacts** (`tuning.storage_dir`, `tuning.artifacts_dir`) — N
  parallel `hydrabflow-tune` launches have N different run dirs but must append to *one* study log.

`run_name` only labels the run dir; `model_dir` is the opposite, an *input* pointing at a finished
`train` dir for `evaluate` to load, and `evaluate` never writes into it
([details](configuration.md#run_name-vs-model_dir)).

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

## One config file per experiment

Instead of a long override string on every command, put the whole experiment in one file and pass
its name. Drop it in `conf/experiments/<name>.yaml`:

```yaml
# @package _global_
defaults:
  - /base_config
  - override /simulator: <your_simulator>
  - override /model/summary_network: <your_summary_net>
  - override /model/inference_network: <your_inference_net>
  - _self_

seed: 42
data:
  data_dir: data_experiments/<name>
  n_simulations: 60_000
# ... every other block from conf/config.yaml you want to change
```

Then every stage takes the same flag:

```bash
uv run hydrabflow-simulate +experiments=<name>
uv run hydrabflow-train    +experiments=<name>
uv run hydrabflow-evaluate +experiments=<name> model_dir=<train run>
```

Two things the file needs because it is not the root config:

- `# @package _global_` — without it the keys land under `experiments.` instead of the config root.
- `/`-prefixed group paths in `defaults` (`/simulator`, not `simulator`), resolved against `conf/`
  rather than `conf/experiments/`; and `override` because `conf/config.yaml` already selects those
  groups, and this file is composed on top of it.

`+experiments=<name>` searches `conf/`. For a file kept outside the repo, prepend its parent to the
search path — the layout below it is the same:

```bash
uv run hydrabflow-train --config-dir /path/to/myconf +experiments=<name>   # /path/to/myconf/experiments/<name>.yaml
```

CLI overrides still win over the file, so `+experiments=<name> training.n_epochs=2` is a valid smoke
run. Check what you composed without launching anything:

```bash
uv run hydrabflow-train +experiments=<name> --cfg job
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


## The protoplanetary-disk arm

```bash
uv run hydrabflow-train    experiment=protoplan
uv run hydrabflow-evaluate experiment=protoplan model_dir=outputs/protoplan/protoplan_npe/<timestamp>
```

`experiment=protoplan` selects the simulator and both networks *and* fills in the plain blocks a
group file cannot reach (`preprocessing`, `augmentation`, `adapter`, `training`, `eval`). There is no
`simulate` stage: the forward model is an external radiative-transfer code, and
`simulator.params.data_path` points at the `.npy` caches it produced.

Needs `$STPSF_PATH` (the JWST PSF reference data) and the caches. The beam, the per-band noise, the
distance and the extinction are `augmentation.params` knobs; the image cache you pick must be paired
with a matching `augmentation.params.px_arcsec_mod`.

For a **real** disk, the generic `data.real_data_path` recipe above is not enough on its own -- the
posterior is conditional on a discrete indicator the observation does not carry, and a disk may be
missing a band. `docs/protoplanetary_disk.md` §9 has the three-command version
(`make_real_npz.py`, then `evaluate` with `eval.mask_condition_groups`).

Full write-up — the encoding, the five per-band branches, every knob, how to read the per-branch
diagnostics, troubleshooting: [**protoplanetary_disk.md**](protoplanetary_disk.md).

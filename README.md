# HydraBFlow

A reusable, cookiecutter-style template for **Simulation-Based Inference (SBI)** pipelines built
on [BayesFlow](https://bayesflow.org) + [Hydra](https://hydra.cc). The template owns all the
infrastructure — dataset generation, training, evaluation, real-data application, hyperparameter
tuning, preprocessing, checkpointing, and full config traceability. To start a new project you
only:

1. **Write your simulator** (forward model) in `src/hydrabflow/simulators/`.
2. **Pick & configure SBI components** (summary network, inference network, training, etc.) by
   editing YAML under `conf/`.

Everything else is fixed infrastructure you should not need to touch.

> **New here?** The [end-to-end pipeline guide](docs/end_to_end_guide.md) is a runnable,
> step-by-step tutorial covering a worked example simulator, the `conf/` system, adding summary /
> inference networks, and hyperparameter tuning. See also the
> [Two Moons pipeline](docs/two_moons_pipeline.md) (a ready-to-run example),
> [configuration](docs/configuration.md) (every config group, field by field, and how to
> override/extend them), [bring your own data](docs/bring_your_own_data.md) (train/evaluate on
> pre-existing simulations with no simulator, and how to support file formats other than `.npz`),
> and [hyperparameter tuning](docs/hyperparameter_tuning.md) (Optuna studies that save every trial
> and run concurrently across processes).

## Design at a glance

- **Single-level SBI.** One summary network + one inference network, `bf.BasicWorkflow`. No
  hierarchical global/local split and no compositional score modeling (deliberately removed from
  the reference project this template generalizes).
- **Hydra config groups + structured dataclass configs.** Every config group (`simulator`,
  `model`, `training`, `data`, `preprocessing`, `augmentation`, `adapter`, `inference`, `eval`,
  `tuning`) has a typed dataclass schema registered in Hydra's `ConfigStore`. Networks and
  simulators are built by **factory functions** that read these dataclasses (no `_target_`).
- **Everything extensible is a self-registering registry.** Simulators, preprocessing steps,
  augmentations, and summary / inference network builders all follow one pattern: drop a module
  into the component's package, decorate it with `@register_<component>("name")` (the package
  auto-imports it), and select it by name in YAML. No infrastructure edits, ever.
- **The adapter wires itself.** The simulator class declares its `parameter_names` and
  `observable_keys`; the BayesFlow adapter derives its variables from them, so you never repeat
  the names in config (explicit adapter config remains available as an override, e.g. for
  [bring your own data](docs/bring_your_own_data.md)).
- **JAX backend + GPU pin.** Before keras/bayesflow/JAX are imported, `hydrabflow.utils.backend`
  pins `KERAS_BACKEND=jax` and uses [`autocvd`](https://pypi.org/project/autocvd) to limit the
  visible GPUs (picking available/free ones). Defaults to one GPU; override with `HYDRABFLOW_NUM_GPUS`
  (`0` = CPU-only), or set `CUDA_VISIBLE_DEVICES` yourself to take full control (autocvd is then
  skipped). Falls back gracefully when there are no NVIDIA GPUs.
- **Preprocessing vs augmentation split.**
  - *Preprocessing* = deterministic, whole-dataset transforms applied **once** (NaN cleaning,
    train/val split, z-score standardization). Fitted on train, saved to the run dir, reused at
    inference. Lives in `src/hydrabflow/preprocessing/`.
  - *Augmentation* = stochastic, per-batch transforms applied **inside** `fit_offline`. Lives in
    `src/hydrabflow/augmentation/`.
- **Full traceability.** Every run writes its resolved Hydra config (`.hydra/`), checkpoints,
  metrics, and (for inference) posterior samples into
  `outputs/<simulator>/<model>/<timestamp>/`.

## Quickstart

The default config runs the shipped **Two Moons** benchmark end-to-end, no code changes needed.
The four commands below are copy-pasteable and take a few minutes on a laptop:

```bash
uv sync                                                        # create .venv, install everything

uv run hydrabflow-simulate                                     # training set -> data/two_moons/
uv run hydrabflow-simulate data.dataset_name=test_data_10000.npz seed=123   # held-out test set
uv run hydrabflow-train                                        # -> outputs/two_moons/default/<timestamp>/
uv run hydrabflow-evaluate model_dir=outputs/two_moons/default/<timestamp>  # posterior + diagnostics
```

`train` prints its timestamped run directory at the end — that is the `model_dir` you hand to
`evaluate` (or grab it with `ls -dt outputs/two_moons/default/*/ | head -1`). Then, optionally:

```bash
uv run hydrabflow-tune                                         # Optuna hyperparameter search
uv run hydrabflow-evaluate-real model_dir=... data.real_data_path=...   # your observed data
```

Equivalently, the Hydra apps under `scripts/` can be run directly, e.g.
`uv run python scripts/train.py training.n_epochs=5 data.n_simulations=2000`. The
[Two Moons walkthrough](docs/two_moons_pipeline.md) explains each step (including a ~1-minute
smoke-run variant).

## Adding your own simulator

The simulator is the only Python you must write. There is a `SkeletonSimulator` stub
(`src/hydrabflow/simulators/skeleton.py`) you can copy; the shipped `two_moons` is a worked
example.

1. Drop `src/hydrabflow/simulators/my_sim.py` into the package:

   ```python
   from hydrabflow.simulators.base import BaseSimulator
   from hydrabflow.simulators.registry import register_simulator

   @register_simulator("my_sim")
   class MySimulator(BaseSimulator):
       @property
       def parameter_names(self): return ["theta1", "theta2"]
       @property
       def observable_keys(self): return ["x"]
       def sample_prior(self, n, rng): ...
       def simulate(self, params, rng): ...
   ```

2. Create `conf/simulator/my_sim.yaml` with `name: my_sim` and any simulator-specific params
   (copy `conf/simulator/skeleton.yaml`).
3. Run with `simulator=my_sim`.

That's all: the module is auto-imported (no `__init__.py` edit), and the adapter derives its
variables from `parameter_names` / `observable_keys` (no adapter config). No infrastructure code
changes are required. The [end-to-end guide §2](docs/end_to_end_guide.md#2-changing-the-simulator)
walks through a complete example including the shape contract for each summary network.

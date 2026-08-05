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

> **New here?** Three docs, in reading order:
> [**running.md**](docs/running.md) — install, the four stages, a full Two Moons walkthrough and a
> 1-minute smoke run, real-data inference, tuning studies, bring-your-own-dataset;
> [**configuration.md**](docs/configuration.md) — every knob in `conf/config.yaml`, the three groups,
> overrides and sweeps; [**extending.md**](docs/extending.md) — add a simulator, network,
> preprocessing step, or augmentation.

## Design at a glance

- **Single-level SBI.** One summary network + one inference network, `bf.BasicWorkflow`. No
  hierarchical global/local split and no compositional score modeling (deliberately removed from
  the reference project this template generalizes).
- **One config file, three config groups.** Everything you tune (`data`, `training`,
  `preprocessing`, `augmentation`, `adapter`, `eval`, `tuning`) is a plain block in
  [`conf/config.yaml`](conf/config.yaml) — no `default.yaml` files and no YAML inheriting from
  another YAML. Only the axes with real alternatives stay swappable groups: `simulator`,
  `model/summary_network`, `model/inference_network`. Types come from one dataclass schema
  (`config.py`), so a typo in the YAML fails loudly; networks and simulators are built by
  **factory functions** reading those dataclasses (no `_target_`).
- **Everything extensible is a self-registering registry.** Simulators, preprocessing steps,
  augmentations, and summary / inference network builders all share one `Registry`
  (`registry.py`): drop a module into the component's package, decorate it with
  `@register_<component>("name")` (the package auto-imports it), and select it by name in YAML.
  No infrastructure edits, ever.
- **The adapter wires itself.** The simulator class declares its `parameter_names` and
  `observable_keys`; the BayesFlow adapter derives its variables from them, so you never repeat
  the names in config (explicit adapter config remains available as an override, e.g. for
  [bring your own dataset](docs/running.md#a-dataset-with-no-simulator)).
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
  `outputs/<simulator>/<run_name>/<timestamp>/`. `run_name` defaults to the two network types;
  override it to label an experiment (`run_name=wider_summary`).

## Quickstart

The default config runs the shipped **Two Moons** benchmark end-to-end, no code changes needed.
The four commands below are copy-pasteable and take a few minutes on a laptop:

```bash
uv sync                                                        # create .venv, install everything

uv run hydrabflow-simulate                                     # training set -> data/training_data_10000.npz
uv run hydrabflow-simulate data.dataset_name=test_data_10000.npz seed=123   # held-out test set
uv run hydrabflow-train                                        # -> outputs/two_moons/<run_name>/<timestamp>/
uv run hydrabflow-evaluate model_dir=outputs/two_moons/<run_name>/<timestamp>  # posterior + diagnostics
```

`train` prints its timestamped run directory at the end — that is the `model_dir` you hand to
`evaluate` (or grab it with `ls -dt outputs/two_moons/*/*/ | head -1`). Then, optionally:

```bash
uv run hydrabflow-tune                                         # Optuna hyperparameter search
uv run hydrabflow-evaluate model_dir=... data.real_data_path=...   # your observed data
```

`evaluate` is one stage with two modes: by default it scores the simulated test set against its
known truth, and setting `data.real_data_path` points it at your observed data instead (posterior
pair plots only — no truth to score against).

Each stage is also runnable as a module, e.g.
`uv run python -m hydrabflow.pipeline.train training.n_epochs=5 data.n_simulations=2000`. The
[Two Moons walkthrough](docs/running.md#two-moons-end-to-end) explains each step (including a
~1-minute smoke-run variant).

## Adding your own simulator

The simulator is the only Python you must write. Copy the shipped worked example,
[`src/hydrabflow/simulators/two_moons.py`](src/hydrabflow/simulators/two_moons.py).

1. Drop `src/hydrabflow/simulators/my_sim.py` into the package:

   ```python
   from hydrabflow.simulators.base import BaseSimulator
   from hydrabflow.registry import register_simulator

   @register_simulator("my_sim")
   class MySimulator(BaseSimulator):
       parameter_names = ["theta1", "theta2"]
       observable_keys = ["x"]

       def sample_prior(self, n, rng): ...
       def simulate(self, params, rng): ...
   ```

2. Create `conf/simulator/my_sim.yaml` with `name: my_sim` and any simulator-specific params
   (copy `conf/simulator/two_moons.yaml`).
3. Run with `simulator=my_sim`.

That's all: the module is auto-imported (no `__init__.py` edit), and the adapter derives its
variables from `parameter_names` / `observable_keys` (no adapter config). No infrastructure code
changes are required. [extending.md](docs/extending.md#your-simulator) walks through a complete example
including the shape contract for each summary network.

# HydraFlow

A reusable, cookiecutter-style template for **Simulation-Based Inference (SBI)** pipelines built
on [BayesFlow](https://bayesflow.org) + [Hydra](https://hydra.cc). The template owns all the
infrastructure — dataset generation, training, evaluation, real-data application, hyperparameter
tuning, preprocessing, checkpointing, and full config traceability. To start a new project you
only:

1. **Write your simulator** (forward model) in `src/hydraflow/simulators/`.
2. **Pick & configure SBI components** (summary network, inference network, training, etc.) by
   editing YAML under `conf/`.

Everything else is fixed infrastructure you should not need to touch.

> **New here?** The [end-to-end pipeline guide](docs/end_to_end_guide.md) is a runnable,
> step-by-step tutorial covering a worked example simulator, the `conf/` system, adding summary /
> inference networks, and hyperparameter tuning. See also the
> [Two Moons pipeline](docs/two_moons_pipeline.md) (a ready-to-run example) and
> [bring your own data](docs/bring_your_own_data.md) (train/evaluate on pre-existing simulations
> with no simulator, and how to support file formats other than `.npz`).

## Design at a glance

- **Single-level SBI.** One summary network + one inference network, `bf.BasicWorkflow`. No
  hierarchical global/local split and no compositional score modeling (deliberately removed from
  the reference project this template generalizes).
- **Hydra config groups + structured dataclass configs.** Every config group (`simulator`,
  `model`, `training`, `data`, `preprocessing`, `augmentation`, `adapter`, `inference`, `eval`,
  `tuning`) has a typed dataclass schema registered in Hydra's `ConfigStore`. Networks and
  simulators are built by **factory functions** that read these dataclasses (no `_target_`).
- **JAX backend + GPU pin.** Before keras/bayesflow/JAX are imported, `hydraflow.utils.backend`
  pins `KERAS_BACKEND=jax` and uses [`autocvd`](https://pypi.org/project/autocvd) to limit the
  visible GPUs (picking available/free ones). Defaults to one GPU; override with `HYDRAFLOW_NUM_GPUS`
  (`0` = CPU-only), or set `CUDA_VISIBLE_DEVICES` yourself to take full control (autocvd is then
  skipped). Falls back gracefully when there are no NVIDIA GPUs.
- **Preprocessing vs augmentation split.**
  - *Preprocessing* = deterministic, whole-dataset transforms applied **once** (NaN cleaning,
    train/val split, z-score standardization). Fitted on train, saved to the run dir, reused at
    inference. Lives in `src/hydraflow/preprocessing/`.
  - *Augmentation* = stochastic, per-batch transforms applied **inside** `fit_offline`. Lives in
    `src/hydraflow/augmentation/`.
- **Full traceability.** Every run writes its resolved Hydra config (`.hydra/`), checkpoints,
  metrics, and (for inference) posterior samples into
  `outputs/<simulator>/<model>/<timestamp>/`.

## Quickstart

```bash
uv sync                      # create .venv and install everything
uv run hydraflow-simulate    # generate a dataset (skeleton sim raises NotImplementedError)
uv run hydraflow-train       # train the approximator
uv run hydraflow-evaluate    # diagnostics on a simulated test set
uv run hydraflow-tune        # Optuna hyperparameter search
uv run hydraflow-evaluate-real  # apply the trained model to real data
```

Equivalently, the Hydra apps under `scripts/` can be run directly, e.g.
`uv run python scripts/train.py training.n_epochs=5 data.n_simulations=2000`.

The shipped simulator (`conf/simulator/skeleton.yaml` →
`hydraflow.simulators.skeleton.SkeletonSimulator`) is an intentional stub: running the pipeline
raises a clear `NotImplementedError` telling you where to plug in your forward model. Replace it
with your own `BaseSimulator` subclass and a matching `conf/simulator/<name>.yaml`.

## Adding your own simulator

1. Create `src/hydraflow/simulators/my_sim.py`:

   ```python
   from hydraflow.simulators.base import BaseSimulator
   from hydraflow.simulators.registry import register_simulator

   @register_simulator("my_sim")
   class MySimulator(BaseSimulator):
       @property
       def parameter_names(self): return ["theta1", "theta2"]
       @property
       def observable_keys(self): return ["x"]
       def sample_prior(self, n, rng): ...
       def simulate(self, params, rng): ...
   ```

2. Create `conf/simulator/my_sim.yaml` with `name: my_sim` and any simulator-specific params.
3. Run with `simulator=my_sim`.

No infrastructure code changes are required.

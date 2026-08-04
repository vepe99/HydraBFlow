# Extending HydraBFlow

Every extension point works the same way: **drop a module into a package, decorate it, select it by
name in YAML.** The package auto-imports its modules (`utils/registry.py:discover`), so the decorator
runs and the component registers itself — you never edit an `__init__.py`, a factory, or an if/elif
chain.

| To add…               | Put a module in                   | Decorate with                 | Select with                        |
| --------------------- | --------------------------------- | ----------------------------- | ---------------------------------- |
| a simulator           | `src/hydrabflow/simulators/`      | `@register_simulator`         | `simulator=my_sim`                 |
| a summary network     | `src/hydrabflow/networks/`        | `@register_summary_network`    | `model.summary_network.type=my_net` |
| an inference network  | `src/hydrabflow/networks/`        | `@register_inference_network`  | `model.inference_network.type=my_net` |
| a preprocessing step  | `src/hydrabflow/preprocessing/`   | `@register_step`              | an entry in `preprocessing.steps`  |
| an augmentation       | `src/hydrabflow/augmentation/`    | `@register_augmentation`      | an entry in `augmentation.steps`   |

All five are the same `Registry` class. Two consequences worth knowing: registering a name twice
raises immediately (no silent shadowing), and an unknown name raises a `KeyError` that lists what *is*
registered.

- [A simulator](#a-simulator)
- [A summary network](#a-summary-network)
- [An inference network](#an-inference-network)
- [A preprocessing step](#a-preprocessing-step)
- [An augmentation](#an-augmentation)
- [Another workflow type (CompositionalWorkflow)](#another-workflow-type-compositionalworkflow)
- [Reading a format other than .npz](#reading-a-format-other-than-npz)

## A simulator

The only Python you *must* write. Copy
[`src/hydrabflow/simulators/two_moons.py`](../src/hydrabflow/simulators/two_moons.py) as a worked
example.

**1. The class.** Two class attributes and two methods:

```python
# src/hydrabflow/simulators/gaussian.py
from typing import Dict, Mapping

import numpy as np

from hydrabflow.simulators.base import BaseSimulator
from hydrabflow.simulators.registry import register_simulator


@register_simulator("gaussian")
class GaussianSimulator(BaseSimulator):
    parameter_names = ["mu", "sigma"]   # become adapter.inference_variables
    observable_keys = ["x"]             # become adapter.summary_variables

    def sample_prior(self, n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        # one (n, 1) column per parameter
        return {
            "mu": rng.uniform(-3.0, 3.0, size=(n, 1)),
            "sigma": rng.uniform(0.1, 2.0, size=(n, 1)),
        }

    def simulate(self, params: Mapping[str, np.ndarray], rng: np.random.Generator):
        n_obs = int(self.params.get("n_obs", 10))       # from conf/simulator/gaussian.yaml
        mu, sigma = params["mu"], params["sigma"]       # both (n, 1)
        x = rng.normal(mu, sigma, size=(mu.shape[0], n_obs))
        return {"x": x[..., None]}                      # (n, n_obs, 1)
```

Rules that matter:

- **Batched.** Both methods take/return arrays whose leading axis is the number of simulations `n`.
  Parameters are `(n, 1)`; observables are `(n, *event_shape)`.
- **All randomness comes from `rng`.** Never `np.random.*` directly — that is what makes a run
  reproducible from `seed`, and what lets `simulate` chunk the work deterministically.
- **Configuration comes from `self.params`**, the free-form `simulator.params` mapping.
- The dataset written to disk is the union of both dicts, one row per (parameters, observation) pair.

**2. The config.** `name` must match the decorator:

```yaml
# conf/simulator/gaussian.yaml
name: gaussian
params:
  n_obs: 10
```

**3. Run it.** Nothing else to wire — the adapter derives its variables from the class:

```bash
uv run hydrabflow-simulate simulator=gaussian
uv run hydrabflow-train    simulator=gaussian
```

### Shape contract per summary network

The observable shape must match the summary network you pick:

| Summary network           | Expects                    | Meaning of the middle axis |
| ------------------------- | -------------------------- | -------------------------- |
| `set_transformer`         | `(n, n_obs, d)`            | exchangeable set           |
| `deep_set`                | `(n, n_obs, d)`            | exchangeable set           |
| `time_series_transformer` | `(n, time, d)`             | ordered sequence           |

A single scalar observation is `(n, 1, 1)`. If you get a shape error from the summary network, this
table is the first place to look.

## A summary network

A builder is a function taking the network config and returning a Keras/BayesFlow layer:

```python
# src/hydrabflow/networks/my_summary.py
from hydrabflow.networks.factory import register_summary_network


@register_summary_network("my_summary")
def build_my_summary(cfg):
    import keras

    return keras.Sequential([
        keras.layers.GlobalAveragePooling1D(),                  # pool the set axis
        keras.layers.Dense(int(cfg.mlp_width), activation="gelu"),
        keras.layers.Dense(int(cfg.summary_dim)),
    ])
```

Reuse the typed fields (`summary_dim`, `mlp_width`, `num_heads`, `embed_dim_per_head`, …) where they
fit, and put anything else in the untyped `params` mapping — no schema change needed:

```python
    activation = cfg.params.get("activation", "gelu")
```

Then either select it inline or add a group file so it is swappable:

```yaml
# conf/model/summary_network/my_summary.yaml
type: my_summary
summary_dim: 32
mlp_width: 128
params:
  activation: relu
```

```bash
uv run hydrabflow-train model/summary_network=my_summary
uv run hydrabflow-train model.summary_network.type=my_summary   # inline, reusing the current file
```

Keep `bayesflow`/`keras` imports *inside* the builder: the package is imported in config-only contexts
(and most of the test suite) where the backend should not be loaded.

## An inference network

Identical, with the other decorator. It must be a BayesFlow inference network — the workflow hands it
to `bf.BasicWorkflow`:

```python
# src/hydrabflow/networks/my_flow.py
from hydrabflow.networks.factory import register_inference_network


@register_inference_network("my_flow")
def build_my_flow(cfg):
    import bayesflow as bf

    return bf.networks.FlowMatching(
        subnet_kwargs={
            "widths": [int(cfg.mlp_width)] * int(cfg.mlp_depth),
            "dropout": float(cfg.dropout),
        },
        integrate_kwargs={"steps": int(cfg.params.get("integration_steps", 100))},
    )
```

```bash
uv run hydrabflow-train model.inference_network.type=my_flow
```

## A preprocessing step

Deterministic, whole-dataset, applied once. Subclass `PreprocessStep`, implement `transform`, and add
`fit` + `state`/`load_state` if the step learns something that must be replayed at inference:

```python
# src/hydrabflow/preprocessing/log_transform.py
from typing import Dict, Iterable

import numpy as np

from hydrabflow.preprocessing.base import Dataset, PreprocessStep
from hydrabflow.preprocessing.registry import register_step


@register_step("log_transform")
class LogTransform(PreprocessStep):
    name = "log_transform"          # also the prefix under which state is saved

    def __init__(self, keys: Iterable[str]) -> None:
        self.keys = list(keys)      # constructor args come from the YAML entry

    def transform(self, data: Dataset) -> Dataset:
        out = dict(data)
        for key in self.keys:
            out[key] = np.log10(np.asarray(data[key]))
        return out
```

A step with fitted state adds:

```python
    def fit(self, data: Dataset) -> None:            # called on the TRAIN split only
        self._mean = {k: np.asarray(data[k]).mean(axis=0) for k in self.keys}

    def state(self) -> Dict[str, np.ndarray]:        # -> preprocessing_state.npz
        return {f"{k}__mean": v for k, v in self._mean.items()}

    def load_state(self, state: Dict[str, np.ndarray]) -> None:   # <- at evaluate time
        self._mean = {k: state[f"{k}__mean"] for k in self.keys}
```

Enable it, remembering that position relative to `train_val_split` decides whether it sees the whole
dataset or is fit on train only:

```yaml
preprocessing:
  steps:
    - name: drop_nan
      keys: ${adapter.summary_variables}
    - name: log_transform         # before the split: applied to everything
      keys: [x]
    - name: train_val_split
      validation_fraction: ${training.validation_fraction}
    - name: standardize           # after the split: fit on train, applied to both
      keys: ${adapter.summary_variables}
```

`evaluate` rebuilds the pipeline from config and loads `preprocessing_state.npz`, so **the step list
must match the one used for training** — which the run's `.hydra/config.yaml` records for you.

## An augmentation

Stochastic and per-batch: a factory returning a `batch -> batch` closure. Copy
[`augmentation/noise.py`](../src/hydrabflow/augmentation/noise.py).

```python
# src/hydrabflow/augmentation/dropout.py
import numpy as np

from hydrabflow.augmentation.registry import Augmentation, register_augmentation


@register_augmentation("random_masking")
def random_masking(params, rng: np.random.Generator, context: dict) -> Augmentation:
    key = params.get("mask_key", "x")
    fraction = float(params.get("mask_fraction", 0.0))

    def _apply(batch: dict) -> dict:
        if fraction > 0.0 and key in batch:
            x = np.asarray(batch[key])
            keep = rng.random(x.shape[:2]) > fraction    # re-drawn every batch
            batch[key] = x * keep[..., None]
        return batch

    return _apply
```

The three arguments are always the same: the shared `augmentation.params` mapping, a generator
private to this step (seeded from `seed`, so it is reproducible but independent of the other steps),
and `context` — run-level objects, currently the fitted `PreprocessPipeline` under `"pipeline"`, which
is how an augmentation can work in the same scaled space the network sees.

```yaml
augmentation:
  steps: [random_masking]
  params: {mask_key: x, mask_fraction: 0.1}
```

Make the default a **no-op** (as `noise_scale: 0.0` is) so the shipped config stays unaugmented.

## Another workflow type (`CompositionalWorkflow`)

This is the one extension point with **no registry**: `pipeline/workflow.py` hardcodes
`bf.BasicWorkflow`, because single-level inference was a deliberate design decision. BayesFlow 2.x
also ships `bf.CompositionalWorkflow` (and `bf.EnsembleWorkflow`), and adding one is a small,
self-contained edit. Worked example below.

**What compositional inference is.** Given `K` independent datasets that share the same parameters,
sample from the *joint* posterior `p(θ | x₁ … x_K)` instead of running one posterior per dataset and
hoping they agree. It is a **sampling-time** capability: training is unchanged — the same
`fit_offline` on single-dataset examples — so you get both modes from one trained model.

Two hard requirements before you start:

- **A score-based inference network.** Only `bf.networks.DiffusionModel` implements
  `_inverse_compositional`; `FlowMatching` raises `NotImplementedError`. Use
  `model/inference_network=diffusion`.
- **A prior score function.** Naively multiplying `K` posteriors counts the prior `K` times, so
  compositional sampling subtracts it: you must supply `compute_prior_score(params) -> ∇_θ log p(θ)`.

**1. A config knob.** Add one typed field to `TrainingConfig` in `src/hydrabflow/config.py` (this is
the "genuinely new field" case from [hydra.md](hydra.md), so the schema does change):

```python
    workflow: str = "basic"          # "basic" | "compositional"
```

**2. One branch in `build_workflow`.** `CompositionalWorkflow.__init__` takes the same keyword
arguments this repo already passes (`adapter`, `summary_network`, `inference_network`, `standardize`,
`checkpoint_*`), so the change is the class, nothing else:

```python
# src/hydrabflow/pipeline/workflow.py
    WORKFLOWS = {"basic": bf.BasicWorkflow, "compositional": bf.CompositionalWorkflow}
    return WORKFLOWS[str(cfg.training.workflow)](**kwargs)
```

Train exactly as before:

```bash
uv run hydrabflow-train training.workflow=compositional model/inference_network=diffusion
```

**3. Sample compositionally.** The saved `approximator.keras` deserializes back into a
`CompositionalApproximator`, so `evaluate`'s existing `workflow.sample(...)` still works
(one dataset at a time) and the compositional method is simply also available:

```python
# in a notebook, or behind a flag in pipeline/evaluate.py next to the workflow.sample call
posterior = workflow.approximator.compositional_sample(
    num_samples=int(cfg.eval.num_samples),
    conditions=conditions,               # each array (n_datasets, n_compositional, ...)
    compute_prior_score=prior_score,
    batch_size=int(cfg.eval.batch_size),
)
```

**Shapes.** `compositional_sample` adds one axis in front of what `sample` expects: conditions are
`(n_datasets, n_compositional, *event_shape)` with `n_compositional >= 2`. `n_datasets` is the usual
batch axis (independent compositional problems, each returning one posterior); `n_compositional` is the
number of datasets fused within a problem. For two_moons, `x` goes from `(n, n_obs, 2)` to
`(n_datasets, K, n_obs, 2)`.

**The prior score.** A callable returning one array per inference variable, shaped like the parameters:

```python
def prior_score(params: dict, *_args) -> dict:
    # ∇_θ log p(θ) for the uniform two_moons prior is 0 inside the support
    return {k: np.zeros_like(v) for k, v in params.items()}
```

For a non-uniform prior, differentiate its log-density (analytically, or with `jax.grad`).

> **Gotcha — the score must live in the network's space.** The network sees parameters *after*
> `preprocessing` and after BayesFlow's `standardize`. A prior score computed in physical units is
> simply wrong there (a missing Jacobian factor), and it fails silently: sampling runs and returns
> plausible-looking, biased posteriors. Either express the prior in the transformed space, or drop the
> parameter transforms for compositional runs and compare against the single-dataset posterior first —
> if the `K = 1` compositional result does not match `workflow.sample`, the score is wrong.

**Should this be a registry?** With one alternative, the two-entry dict above is the right size. If you
end up with several (`EnsembleWorkflow`, a hierarchical variant, your own), promote it: a
`Registry("workflow")` in `pipeline/workflow.py` plus `@register_workflow("compositional")` builders
makes it the sixth extension point and matches every other section of this file.

## Reading a format other than `.npz`

`pipeline/io.py:load_dataset` is the only place the pipeline reads a dataset. It returns
`{key: array}`; every stage consumes that dict. To support HDF5, Parquet, or your own layout, branch
on the extension there:

```python
def load_dataset(path: str) -> Dataset:
    if path.endswith(".h5"):
        import h5py

        with h5py.File(path, "r") as f:
            return {k: f[k][...] for k in f}
    ...
```

Then point the config at it (`data.dataset_name=my_train.h5`, `data.real_data_path=obs.h5`). Only
`simulate` writes datasets, so if you generate data elsewhere you can leave `save_dataset` alone.

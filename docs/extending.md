# Extending

One pattern for all five extension points: **drop a module into the right package, decorate it,
select it by name in config.** No `__init__.py` edit, no infrastructure change — each registry
imports every module in its package on the first lookup.

```python
from hydrabflow.registry import register_simulator          # simulators/
from hydrabflow.registry import register_summary_network    # networks/
from hydrabflow.registry import register_inference_network  # networks/
from hydrabflow.registry import register_step              # preprocessing/
from hydrabflow.registry import register_augmentation      # augmentation/
```

| Decorator                       | Package          | Selected by                    |
| ------------------------------- | ---------------- | ------------------------------ |
| `@register_simulator`           | `simulators/`    | `simulator=<name>`             |
| `@register_summary_network`     | `networks/`      | `model.summary_network.type`   |
| `@register_inference_network`   | `networks/`      | `model.inference_network.type` |
| `@register_step`                | `preprocessing/` | `preprocessing.steps[].name`   |
| `@register_augmentation`        | `augmentation/`  | `augmentation.steps[]`         |

## Your simulator

Two files. Copy `src/hydrabflow/simulators/two_moons.py` and
`conf/simulator/two_moons.yaml` and edit.

```python
# src/hydrabflow/simulators/my_model.py
import numpy as np
from hydrabflow.registry import register_simulator
from hydrabflow.simulators.base import BaseSimulator


@register_simulator("my_model")
class MyModel(BaseSimulator):
    parameter_names = ["alpha", "beta"]   # -> adapter.inference_variables
    observable_keys = ["y"]               # -> adapter.summary_variables

    def sample_prior(self, n, rng):
        return {
            "alpha": rng.uniform(0.0, 1.0, size=(n, 1)),
            "beta": rng.normal(0.0, 1.0, size=(n, 1)),
        }

    def simulate(self, theta, rng):
        alpha = np.asarray(theta["alpha"])                      # (n, 1)
        y = alpha * np.arange(50) + rng.normal(0, 0.1, (len(alpha), 50))
        return {"y": y}                                         # (n, 50)
```

```yaml
# conf/simulator/my_model.yaml
name: my_model
params:                 # free-form; reaches the instance as self.params
  n_steps: 50
```

Rules:

- Leading axis is always `n` (one row = one parameter/observation pair). Parameters are `(n, 1)`;
  observables are `(n, *event_shape)`. `BaseSimulator.sample` checks both, plus that the returned
  keys match the two class attributes, so a mis-shaped forward model fails at the source instead of
  writing a corrupt `.npz`.
- **All randomness must come from the passed `rng`** — never global `np.random` — so a run
  reproduces from `cfg.seed`.
- If the forward model cannot vectorize over `n` — a parameter that sets a loop length, an ODE
  solver, an external binary — set `is_batched = False` and write `simulate` for a single draw:
  it then receives `{param_name: (1,)}` and returns `{observable_key: event_shape}`, and `sample`
  loops and stacks (an epidemiological model whose reporting delay sets the number of timesteps is
  a typical case). Everything else is unchanged, so this is a one-line switch.
- The class attributes are the single source of truth for variable names: the adapter is derived
  from them, so `conf/config.yaml` needs no edit.
- Several `observable_keys` automatically switches the summary side to a `FusionNetwork` (see
  `multimodal.py` + `conf/model/summary_network/fusion.yaml`).

Then: `uv run hydrabflow-simulate simulator=my_model` and `hydrabflow-train simulator=my_model`.

## Your network

A builder is a function from the network config to a Keras/BayesFlow network. Append it to
`networks/factory.py` for a thin wrapper, or drop a new module for something substantial.

```python
# src/hydrabflow/networks/my_net.py
from hydrabflow.registry import register_summary_network


@register_summary_network("my_net")
def _my_net(cfg):
    import bayesflow as bf          # import inside, so config-only contexts skip the backend
    return MyNetwork(
        summary_dim=int(cfg.summary_dim),
        depth=int(cfg.mlp_depth),
        **cfg.params,               # anything not in the schema
    )
```

Select it with `model.summary_network.type=my_net`; extra hyperparameters go under
`model.summary_network.params`. Inference networks are identical with
`@register_inference_network`, and must be a BayesFlow inference network (`FlowMatching`,
`DiffusionModel`, or a subclass).

## Preprocessing step (deterministic, once, fit on train)

```python
from hydrabflow.preprocessing.base import PreprocessStep
from hydrabflow.registry import register_step


@register_step("clip")
class Clip(PreprocessStep):
    name = "clip"                       # prefix for its saved state keys

    def __init__(self, keys, lo=-5.0, hi=5.0):
        self.keys, self.lo, self.hi = list(keys), lo, hi

    def transform(self, data):
        return {k: (np.clip(v, self.lo, self.hi) if k in self.keys else v)
                for k, v in data.items()}
```

```yaml
preprocessing:
  steps:
    - name: clip
      keys: ${adapter.summary_variables}
```

Stateful steps additionally implement `fit(data)`, `state()` and `load_state(state)` — that state is
saved to `preprocessing_state.npz` and replayed at inference time. See `standardize.py`.

## Augmentation (stochastic, per batch)

A factory `(params, rng, context) -> (batch -> batch)`. `rng` is a seeded child generator;
`context["pipeline"]` is the fitted preprocessing pipeline.

```python
from hydrabflow.registry import register_augmentation


@register_augmentation("dropout_obs")
def dropout_obs(params, rng, context):
    p = float(params.get("dropout_p", 0.1))

    def _apply(batch):
        mask = rng.random(batch["x"].shape[:2]) > p
        batch["x"] = batch["x"] * mask[..., None]
        return batch

    return _apply
```

```yaml
augmentation:
  steps: [dropout_obs]
  params: {dropout_p: 0.1}
```

Draw only from the injected `rng`. Augmentations may change the observation layout — validation data
goes through the same chain once with a fixed draw.


## A forward model you cannot run in-process

Some forward models are an external code (a radiative-transfer run, an instrument pipeline, a
cluster job) whose output already exists as files. Such a simulator implements two optional hooks
instead of `simulate`, and the `simulate` stage is skipped entirely:

```python
@register_simulator("mine")
class MySimulator(BaseSimulator):
    parameter_names = [...]      # still declared: they fill the adapter's key lists
    observable_keys = [...]      # and drive select_adapter_keys

    def load_dataset(self):                 # -> {key: array}, leading axis = rows
        ...
    def build_adapter(self, cfg):           # optional, only if AdapterConfig cannot express it
        ...
    def sample_prior(self, n, rng): raise NotImplementedError
    simulate = sample_prior
```

`pipeline.io.load_config_dataset(cfg)` prefers `load_dataset()` over the `.npz` on disk, and
`pipeline.adapter.build_adapter` prefers `build_adapter()` over the generic four-key-list path.
Neither is declared on `BaseSimulator`, so `hasattr` is the test and no existing simulator changes.
Reach for `build_adapter` only when the layout genuinely needs it -- grouped inputs, per-key
transforms, conditions routed into the summary network. `simulators/protoplan.py` is the worked
example.

If your summary network assembles its own multi-backbone graph (rather than one backbone per
summary key), set `handles_fusion = True` on the builder so `registry.build_summary_network` does not
wrap it in the generic `FusionNetwork`.

# How Hydra is wired here (and what `@register_*` actually does)

This is the mental model behind [configuration.md](configuration.md) (what the knobs mean) and
[extending.md](extending.md) (how to add a component). Read it once and you will know **which file to
open** for any change.

- [The two halves](#the-two-halves)
- [What `@register_*` does](#what-register_-does)
- [The five registries](#the-five-registries)
- [Who owns the Hydra infrastructure](#who-owns-the-hydra-infrastructure)
- [Life of one run, step by step](#life-of-one-run-step-by-step)
- [Where do I change X?](#where-do-i-change-x)

## The two halves

Every configurable thing in this repo is split in two:

| half | lives in | answers |
|---|---|---|
| **the value** | `conf/*.yaml`, typed by a dataclass in `src/hydrabflow/config.py` | *what did the user ask for?* |
| **the code** | a class/function in a package, entered into a registry by a decorator | *what runs when they ask for it?* |

The bridge between them is always a **name string**. `simulator.name: two_moons` in YAML, and
`@register_simulator("two_moons")` on the class. A factory looks the string up in the registry and
instantiates. That is the whole pattern — repeated five times.

Deliberately **not** used: Hydra's `_target_` / `instantiate`. Nothing in `conf/` names a Python
import path, so configs stay readable and typo-proof (the dataclass validates them) and refactoring
a module never breaks a YAML.

## What `@register_*` does

```python
# src/hydrabflow/simulators/my_sim.py
@register_simulator("my_sim")          # <- this line
class MySimulator(BaseSimulator):
    parameter_names = ["a", "b"]
    observable_keys = ["x"]
```

Mechanically it is three lines of `src/hydrabflow/utils/registry.py`:

```python
def add(self, name):                   # register_simulator = SIMULATORS.add
    def _wrap(obj):
        self.items[name] = obj         # dict entry: "my_sim" -> MySimulator
        return obj                     # the class is returned unchanged
    return _wrap
```

So the decorator **does not modify your class at all**. It only inserts `name -> object` into a
module-level dict. Duplicate names raise at import time; unknown names raise a `KeyError` that lists
what *is* registered.

The part that makes it feel magic is `discover()`, called by each package's `__init__.py`:

```python
# src/hydrabflow/simulators/__init__.py
discover(__name__, __path__)           # imports every non-underscore module in this folder
```

Importing a module runs its decorators. So **dropping a file into the package is the registration** —
you never edit an `__init__.py`, a factory, an `if/elif`, or the schema. A new component costs exactly
one Python file plus (for simulators/networks) one YAML.

## The five registries

| decorator | registry object | selected by config key | consumed by |
|---|---|---|---|
| `@register_simulator` | `simulators/registry.py` | `simulator.name` | `get_simulator(cfg.simulator)` |
| `@register_step` | `preprocessing/registry.py` | `preprocessing.steps[].name` | `build_pipeline(cfg.preprocessing)` |
| `@register_augmentation` | `augmentation/registry.py` | `augmentation.steps[]` | `build_augmentations(cfg.augmentation)` |
| `@register_summary_network` | `networks/factory.py` | `model.summary_network.type` | `build_summary_network(cfg...)` |
| `@register_inference_network` | `networks/factory.py` | `model.inference_network.type` | `build_inference_network(cfg...)` |

All five are instances of the same `Registry` class. The builders also decide **what else the config
carries into your code**: a simulator gets `cfg.simulator.params` as constructor kwargs, a
preprocessing step gets the leftover keys of its `{name: ..., ...}` mapping as kwargs, an augmentation
gets `cfg.augmentation.params` plus a seeded RNG, and a network builder gets its whole typed config
node (extras via `params`).

## Who owns the Hydra infrastructure

One file each — this is the useful list:

- **`src/hydrabflow/config.py`** — *the schema owner.* Every dataclass (`RootConfig` and its typed
  fields) plus `register_configs()`, which stores exactly one node in Hydra's `ConfigStore` under the
  name `base_config`. Because each group is a typed field of `RootConfig`, group YAMLs are validated
  by being selected; there are no per-group schema nodes. **Touch this only to add a genuinely new
  field**, never to add a component.
- **`conf/config.yaml`** — *the composition root.* The repo's only `defaults:` list: it pulls in
  `base_config`, picks one file from each of the three groups, then `_self_` so its own values win.
  It also owns the `hydra:` block (`hydra.run.dir`, the timestamped output convention).
- **`src/hydrabflow/pipeline/_app.py`** — *the entry-point owner.* `make_cli(run_fn)` calls
  `register_configs()`, wraps `run_fn` in `hydra.main(config_path=conf/, config_name="config")`, and
  fixes the job name so logs land in `train.log` rather than `_app.log`. Each stage module ends with
  `cli = make_cli(run_*)`; `pyproject.toml` exposes those as `hydrabflow-<stage>`. No argparse
  anywhere.
- **`src/hydrabflow/pipeline/adapter.py`** — *the name-derivation owner.* `fill_adapter_from_simulator`
  runs inside `make_cli` before your stage: if `adapter.inference_variables` /
  `summary_variables` are empty, they are filled from the selected simulator's `parameter_names` /
  `observable_keys`. The simulator class is the single source of truth for variable names; an explicit
  `adapter:` block overrides it (needed for bring-your-own-data, where no simulator exists).
- **`src/hydrabflow/utils/registry.py`** — *the extension-point owner.* `Registry` + `discover`,
  shared by all five decorators above.
- **the four stage modules** (`pipeline/simulate.py`, `train.py`, `evaluate.py`, `tune.py`) — each is
  a plain `run(cfg)` that reads the resolved config and calls the builders. They never parse
  arguments, never construct paths by hand (`utils/paths.get_run_dir()` returns Hydra's run dir), and
  never import each other's private helpers (shared artifact writing lives in `pipeline/artifacts.py`).

`tune.py` is the one stage that *writes* config: for each Optuna trial it sets the dotted paths in
`tuning.search_space` on a copy of the config and re-runs training. That is why any knob is tunable
without extra plumbing — the search space is just config paths.

## Life of one run, step by step

`uv run hydrabflow-train model/summary_network=deep_set training.n_epochs=5`

1. `pyproject.toml` console script → `pipeline/train.py:cli`, which is `make_cli(run_training)`.
2. `register_configs()` stores `RootConfig` as `base_config`.
3. `hydra.main` composes `conf/config.yaml`: `base_config` defaults ← group files (with your
   `deep_set` override) ← `_self_` ← your command-line overrides. Type errors and unknown keys fail
   **here**, before any compute.
4. Hydra creates `outputs/two_moons/deep_set+flow_matching/<timestamp>/`, `chdir`s into it, writes
   the fully resolved config to `.hydra/`, and installs the logging handlers.
5. `fill_adapter_from_simulator(cfg)` derives the adapter variable names from the simulator class.
6. `run_training(cfg)` runs: `build_pipeline` → `build_workflow` (which calls the two network
   builders) → `build_augmentations` → `fit_offline` → artifacts.

Step 4 is the traceability guarantee: the output folder alone is enough to reconstruct the run.

## Where do I change X?

| I want to… | edit |
|---|---|
| change a value for one run | nothing — pass `key=value` on the command line |
| change a default for all runs | the relevant block in `conf/config.yaml` |
| add a forward model | new file in `simulators/` with `@register_simulator` + a `conf/simulator/<name>.yaml` |
| add a network architecture | new file in `networks/` with `@register_summary_network` / `@register_inference_network` + a group YAML; extras go in `params` |
| add a preprocessing step / augmentation | new file in `preprocessing/` or `augmentation/` with `@register_step` / `@register_augmentation`, then list it in `config.yaml` |
| add a new *typed field* (a knob no dataclass has) | the dataclass in `src/hydrabflow/config.py`, then the YAML — or, for a one-off, `+my.key=value` / the free-form `params` dicts |
| change the output directory layout | the `hydra:` block in `conf/config.yaml` |
| change what a stage *does* | the stage module in `pipeline/` — but note this is infrastructure; prefer a registered component |
| point evaluate at a trained model | `model_dir=<a completed train run dir>` |

Rule of thumb: **if you are adding a capability, you are writing a new file and a decorator. If you
are editing `config.py`, `_app.py`, or a stage module, stop and check whether a registry already
gives you the seam.**

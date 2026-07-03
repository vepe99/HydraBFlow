# Bring your own data (no simulator) & supporting other file formats

HydraBFlow's `simulate` stage exists to *generate* a dataset from a forward model. But the
downstream stages don't depend on it: **`train`, `evaluate`, `evaluate_real`, and `tune` only ever
touch a dataset through one function pair** — [`io.load_dataset` / `io.save_dataset`](../src/hydrabflow/pipeline/io.py).
They never instantiate a simulator. That makes two things easy:

- **Part A** — train and evaluate on simulations you already ran elsewhere, with no simulator code.
- **Part B** — swap the on-disk format (`.npz` today) for HDF5, Parquet, CSV, FITS, … by editing
  one file.

For the conceptual tour of the config system and stages, see the
[end-to-end guide](end_to_end_guide.md); for the shipped worked example, the
[Two Moons pipeline](two_moons_pipeline.md).

---

## Part A — Use a pre-existing dataset (no simulator)

### A.1 The data contract

A dataset is a flat mapping `key -> array` where **every array shares the same leading axis** = the
number of simulations (one (parameters, observation) pair per row). This is exactly what
[`io.load_dataset`](../src/hydrabflow/pipeline/io.py#L25) returns and what the BayesFlow adapter
consumes.

| Kind of key | Shape convention | Example |
|-------------|------------------|---------|
| each **parameter** (inference variable) | `(n, 1)` | `theta1: (n, 1)`, `theta2: (n, 1)` |
| each **observable** (summary variable) | `(n, ...)` — set-shaped `(n, set_size, features)` for SetTransformer | `x: (n, n_obs, 2)` |

The key *names* are arbitrary — they just have to match what you put in the **adapter** (next
step). For the **truth-aware** `evaluate` stage your test file must also contain the parameter keys
(it needs ground truth to compute RMSE / calibration). For **`evaluate_real`** only the observables
are required.

> Shape note: parameters are conventionally 2-D `(n, 1)`, not 1-D `(n,)`. If your saved params are
> 1-D, reshape them (`theta[:, None]`) when you build the file, or add a preprocessing step.

### A.2 Convert your existing arrays into the dataset file

If your prior runs live in some other structure, write them once into the format the loader
expects. With the default `.npz` backend that's simply:

```python
import numpy as np

# however your past run is stored — load it into plain numpy arrays:
theta1 = ...   # (n, 1)
theta2 = ...   # (n, 1)
x      = ...   # (n, n_obs, 2)   your observable(s)

np.savez("data/mydata/training_data_50000.npz", theta1=theta1, theta2=theta2, x=x)
# a held-out test file WITH ground-truth params, for truth-aware evaluate:
np.savez("data/mydata/test_data_50000.npz",     theta1=..., theta2=..., x=...)
```

(If you'd rather keep your native format, do **Part B** instead of converting, then point the
stages straight at your files.)

### A.3 Tell the pipeline about it (config only)

Two pieces of config. Note that **no simulator class is needed** — `simulator.name` is only a
*label* used to build output paths (`outputs/${simulator.name}/...`) and the default
`data_dir: data/${simulator.name}`; the simulator group stays selected but is never instantiated
outside `simulate`.

Create `conf/simulator/mydata.yaml` (a label + empty params):

```yaml
defaults:
  - base_simulator
name: mydata
params: {}
```

Create `conf/adapter/mydata.yaml` mapping **your file's keys** to BayesFlow roles
(see [`AdapterConfig`](../src/hydrabflow/config/schema.py#L142)):

```yaml
defaults:
  - base_adapter
inference_variables: [theta1, theta2]   # your parameter keys
summary_variables: [x]                   # your observable key(s)
inference_conditions: []
drop: []
```

The explicit adapter is **required** here: with a registered simulator the empty default adapter
derives these lists from the class, but `mydata` is only a label with no class behind it, so
there is nothing to derive from. Skip this file and the run fails fast with an error pointing
back to this document.

> You can skip the two YAML files and pass everything on the CLI
> (`simulator.name=mydata 'adapter.inference_variables=[theta1,theta2]' 'adapter.summary_variables=[x]'`),
> but version-controlled config files keep the run reproducible and self-documenting.

### A.4 Run train + evaluate

With the files placed under `data/mydata/` (the default `data_dir: data/${simulator.name}`),
`data.n_simulations` only controls the interpolated file *name* (`training_data_<N>.npz`,
`test_data_<N>.npz`), so set it to match what you saved:

```bash
# train on your pre-existing simulations
uv run python scripts/train.py \
  simulator=mydata adapter=mydata \
  data.n_simulations=50000 \
  training.n_epochs=100

# truth-aware evaluate (test file contains ground-truth params)
RUN_DIR=$(ls -dt outputs/mydata/default/*/ | head -1)
uv run python scripts/evaluate.py \
  simulator=mydata adapter=mydata \
  data.n_simulations=50000 \
  model_dir="$RUN_DIR"
```

If instead your data are **real observations with no ground truth**, use `evaluate_real` and point
it at the file directly:

```bash
uv run python scripts/evaluate_real.py \
  simulator=mydata adapter=mydata \
  data.real_data_path=data/mydata/observed.npz \
  model_dir="$RUN_DIR"
```

If your files live outside the convention, override the location instead of relying on
interpolation:

```bash
... data.data_dir=/abs/path/to/dir data.dataset_name=my_train.npz ...      # train
... data.data_dir=/abs/path/to/dir eval.test_dataset_name=my_test.npz ...  # evaluate
```

### A.5 What reads what

| Stage | Reads | Needs ground truth? |
|-------|-------|---------------------|
| `train` | `data.data_dir/data.dataset_name` | — |
| `evaluate` | `data.data_dir/eval.test_dataset_name` + `model_dir` | **yes** (params in the test file) |
| `evaluate_real` | `data.real_data_path` + `model_dir` | no |
| `tune` | `data.data_dir/data.dataset_name` | yes (uses an internal split) |

> Caveat: nothing currently validates that your file's keys match the adapter. A typo surfaces as a
> BayesFlow error at workflow-build / sample time, not a friendly message. A tiny key/shape check is
> a natural addition if you ingest external data often — see the end of Part B.

---

## Part B — Support a file format other than `.npz`

### B.1 The single seam

All dataset IO funnels through two functions in
[`src/hydrabflow/pipeline/io.py`](../src/hydrabflow/pipeline/io.py):

```python
def save_dataset(path: str, data: Dataset) -> None: ...   # used only by simulate
def load_dataset(path: str) -> Dataset: ...               # used by train / evaluate / evaluate_real / tune
```

`Dataset = Dict[str, np.ndarray]`. Every stage imports `io` and calls these — verified call sites:
[train.py:34](../src/hydrabflow/pipeline/train.py#L34), [evaluate.py:48](../src/hydrabflow/pipeline/evaluate.py#L48),
[evaluate_real.py:47](../src/hydrabflow/pipeline/evaluate_real.py#L47), [tune.py:89](../src/hydrabflow/pipeline/tune.py#L89),
[simulate.py:44](../src/hydrabflow/pipeline/simulate.py#L44). **Change these two functions and the
whole pipeline changes format** — no stage code is touched.

What is **not** part of this seam (separate `.npz` uses, leave alone unless you specifically want
to change them):

- **Preprocessing state** — fitted state is persisted by the preprocessing module itself in
  [`preprocessing/base.py`](../src/hydrabflow/preprocessing/base.py#L85) (`preprocessing_state.npz`).
  Internal artifact, not your dataset.
- **Posterior output** — written with a direct `np.savez` in [evaluate.py:59](../src/hydrabflow/pipeline/evaluate.py#L59)
  and [evaluate_real.py:56](../src/hydrabflow/pipeline/evaluate_real.py#L56) (`posterior.npz`).
  Output, not input. If you want posteriors in another format, change those two lines too.
- **Model** — saved as `.keras` in [`checkpoint.py`](../src/hydrabflow/pipeline/checkpoint.py)
  (BayesFlow's own serialization). Unrelated to dataset format.

### B.2 Option 1 — Quick swap (one format, replace the body)

The minimal change: keep the signatures, swap the implementation. Example for HDF5
(`pip install h5py`):

```python
# src/hydrabflow/pipeline/io.py
import h5py
import numpy as np

def save_dataset(path: str, data: Dataset) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with h5py.File(path, "w") as f:
        for k, v in data.items():
            f.create_dataset(k, data=np.asarray(v))

def load_dataset(path: str) -> Dataset:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}.")
    with h5py.File(path, "r") as f:
        return {k: f[k][()] for k in f.keys()}
```

Then point the config at the new extension (so interpolated names line up):

```yaml
# conf/data/default.yaml
dataset_name: training_data_${data.n_simulations}.h5
# conf/eval/default.yaml
test_dataset_name: test_data_${data.n_simulations}.h5
```

That's the whole change. Note the loader returns the **same `Dict[str, np.ndarray]` contract** from
Part A.1 — that contract is what every downstream stage relies on, regardless of format.

### B.3 Option 2 — A format registry (support several formats by extension)

If you want `.npz`, `.h5`, `.parquet`, … to coexist, dispatch on the file extension. This keeps the
public `load_dataset` / `save_dataset` signatures identical (so no stage changes) while making new
formats a small, self-contained addition — mirroring how simulators/augmentations self-register
elsewhere in the template.

Replace the body of [`io.py`](../src/hydrabflow/pipeline/io.py) with:

```python
"""Dataset IO. A dataset is Dict[str, np.ndarray]; every array shares the leading (row) axis.
Format is chosen by file extension via the loader/saver registries below.
"""
from __future__ import annotations
import os
from typing import Callable, Dict

import numpy as np
from hydrabflow.utils.logging import get_logger

log = get_logger(__name__)
Dataset = Dict[str, np.ndarray]

_LOADERS: Dict[str, Callable[[str], Dataset]] = {}
_SAVERS: Dict[str, Callable[[str, Dataset], None]] = {}

def register_format(ext: str, *, load=None, save=None) -> None:
    """Register a loader and/or saver for files ending in `ext` (e.g. '.npz')."""
    if load is not None:
        _LOADERS[ext] = load
    if save is not None:
        _SAVERS[ext] = save

def _ext(path: str) -> str:
    return os.path.splitext(path)[1].lower()

def load_dataset(path: str) -> Dataset:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}. Generate it or convert your data.")
    ext = _ext(path)
    if ext not in _LOADERS:
        raise KeyError(f"No loader for '{ext}'. Registered: {sorted(_LOADERS)}.")
    data = _LOADERS[ext](path)
    n = len(next(iter(data.values()))) if data else 0
    log.info("Loaded dataset (%d rows, keys=%s) <- %s", n, list(data), path)
    return data

def save_dataset(path: str, data: Dataset) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ext = _ext(path)
    if ext not in _SAVERS:
        raise KeyError(f"No saver for '{ext}'. Registered: {sorted(_SAVERS)}.")
    _SAVERS[ext](path, data)
    n = len(next(iter(data.values()))) if data else 0
    log.info("Saved dataset (%d rows, keys=%s) -> %s", n, list(data), path)

def concatenate_chunks(chunks: list[Dataset]) -> Dataset:
    if not chunks:
        return {}
    return {k: np.concatenate([c[k] for c in chunks], axis=0) for k in chunks[0]}

# ---- built-in formats -------------------------------------------------------------------------- #
def _load_npz(path: str) -> Dataset:
    raw = np.load(path, allow_pickle=True)
    return {k: raw[k] for k in raw.files}

def _save_npz(path: str, data: Dataset) -> None:
    np.savez(path, **data)

register_format(".npz", load=_load_npz, save=_save_npz)
```

Now add a new format anywhere that gets imported (a new `io_formats.py`, or the bottom of `io.py`).
For example **HDF5** and **Parquet** (row-oriented; good for tabular params + flat observables):

```python
# HDF5 — arbitrary array shapes
import h5py
register_format(
    ".h5",
    load=lambda p: {k: h5py.File(p, "r")[k][()] for k in h5py.File(p, "r").keys()},
    save=lambda p, d: _h5_save(p, d),
)
def _h5_save(p, d):
    import h5py
    with h5py.File(p, "w") as f:
        for k, v in d.items():
            f.create_dataset(k, data=np.asarray(v))

# Parquet — one column per key; reshape on the way in/out if observables are multi-dim
import pandas as pd
register_format(
    ".parquet",
    load=lambda p: {k: pd.read_parquet(p)[k].to_numpy()[:, None] for k in pd.read_parquet(p).columns},
    save=lambda p, d: pd.DataFrame({k: np.asarray(v).reshape(len(v), -1).squeeze() for k, v in d.items()}).to_parquet(p),
)
```

> Tabular formats (Parquet/CSV) flatten naturally only when each key is 1-D or `(n, 1)`. For
> set-shaped observables `(n, set_size, features)` either keep `.npz`/`.h5`, or store the flattened
> columns plus a preprocessing/adapter step that reshapes — HDF5 is the least-friction choice for
> multi-dimensional observables.

Point the config at whichever extension you're using (`dataset_name: ...h5`,
`test_dataset_name: ...h5`, `real_data_path: ....h5`). Because dispatch is by extension, you can
mix: keep `.npz` for generated data and load a colleague's `.h5` by just naming it.

### B.4 (Optional) validate keys/shapes on load

Since external data isn't produced by a known simulator, a guard turns silent BayesFlow errors into
clear ones. Add to `load_dataset` (or call from your stage):

```python
def check_dataset(data: Dataset, cfg) -> None:
    keys = set(data)
    required = set(cfg.adapter.inference_variables) | set(cfg.adapter.summary_variables)
    missing = required - keys
    if missing:
        raise KeyError(f"Dataset missing adapter keys {sorted(missing)}; has {sorted(keys)}.")
    n = {k: len(v) for k, v in data.items()}
    if len(set(n.values())) != 1:
        raise ValueError(f"Inconsistent leading axis across keys: {n}.")
```

---

## Checklist

**Pre-existing dataset, no simulator:**

- [ ] Arrays written to one file with a shared leading axis (params `(n,1)`, observables `(n,...)`)
- [ ] `conf/simulator/<label>.yaml` (just `name` + empty `params`) — or `simulator.name=` on CLI
- [ ] `conf/adapter/<label>.yaml` with `inference_variables` / `summary_variables` = your keys
- [ ] File placed at `data/<label>/` (or override `data.data_dir` / `data.dataset_name`)
- [ ] Truth-aware `evaluate` only if the test file carries ground-truth params; else `evaluate_real`

**New file format:**

- [ ] Edit only [`io.py`](../src/hydrabflow/pipeline/io.py) — `load_dataset` / `save_dataset` (Option 1)
      or register by extension (Option 2)
- [ ] Update `dataset_name` / `test_dataset_name` / `real_data_path` extensions in `conf/`
- [ ] (Optional) also change posterior output (`np.savez` in evaluate / evaluate_real) and
      preprocessing-state IO ([`preprocessing/base.py`](../src/hydrabflow/preprocessing/base.py)) if
      you want those off `.npz` too — they're independent of the dataset seam

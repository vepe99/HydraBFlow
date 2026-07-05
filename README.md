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

## Stream project (compositional score modeling)

The `stream_project` branch adds an opt-in **compositional** inference mode on top of the
single-level template above, ported from a hierarchical stellar-stream inference study
(particle-spray tidal streams around the Milky Way, simulated with
[AGAMA](https://github.com/GalacticDynamics-Oxford/Agama)). It infers two levels of parameters
from a group of streams that share a global Galactic potential:

- **Global**: potential parameters shared by all streams (halo `rho`/`gamma`/`a`/`q`, disk
  `r`/`z`/`Sigma`), conditioned only on the stream index.
- **Local**: per-stream phase-space parameters (`vr`, `r`, `mu_ra_cosdec`, `mu_dec`), conditioned
  on the global parameters (fixed at the truth during training, or a previously-inferred posterior
  at real-data evaluation time).

Each level is trained and evaluated as a **separate model** (`bf.CompositionalWorkflow` instead of
`bf.BasicWorkflow`); the config group `composition` (`none` / `global` / `local`) switches which
one you're building. `composition=none` is the template default and leaves single-level SBI
untouched.

Two observables are fused per stream: the observed particle set (sky position, proper motion,
parallax, radial velocity — `sim_data_projected`) and a model rotation curve (`vcirc_kms`), summarized
by a `SetTransformer` + `TimeSeriesTransformer` combined with `MaskedFusionNetwork`
(`model=stream_fusion`). Observations are standardized **per stream** (not globally) — parameters
via preprocessing (`per_stream_parameter_standardize`, z-scored by each stream's own prior
mean/std, invertible so posteriors come back in physical units), observations via a per-batch
augmentation (`per_stream_standardize`) fed by stats fitted once in preprocessing
(`stream_observation_stats`).

### Configuration used explicitly

Every stream run composes the same six overrides on top of `conf/config.yaml`'s defaults, plus one
of the `composition` levels:

| Config group | Value | What it selects |
|---|---|---|
| `simulator` | `stream_agama` | AGAMA particle-spray simulator; declares the global/local/context split and both observables |
| `model` | `stream_fusion` | `MaskedFusionNetwork` (SetTransformer + TimeSeriesTransformer) + `DiffusionModel` |
| `adapter` | `stream` | Routes the particle attention mask to `summary_attention_mask`; variables still derive from the simulator, gated by `composition.level` |
| `composition` | `global` \| `local` | Which level's workflow/adapter derivation to build (see table below) |
| `preprocessing` | `stream_global` \| `stream_local` | NaN cleanup, rotation-curve trim, split, (+ per-stream parameter/observation stats for `local`) |
| `augmentation` | `stream_global` \| `stream_local` | Gaia-like observation model (selection window, photometric errors, DR3 astrometric errors, `v_los` masking) run per batch during training |
| `training` | `base_training` (global) \| `stream_local` (local) | `stream_local` standardizes only targets + conditions (observations are already per-stream normalized) |
| `eval` | `stream_compositional` | `compositional_sample` / `ancestral_sample` kwargs (`method: two_step_adaptive`, `compositional_bridge_d1: 0.166667`) |

`composition.level` changes what the adapter derives automatically:

| `composition.level` | `inference_variables` | `inference_conditions` |
|---|---|---|
| `global` | global parameters (halo + disk) | `j` (stream index) |
| `local` | local parameters (`vr`, `r`, `mu_ra_cosdec`, `mu_dec`) | global parameters + `j` |

### Data generation

Two dataset shapes are needed: a flat training set (one draw per row, used by both levels) and a
grouped multistream set (one shared global draw + one member per stream, used for compositional
evaluation and for the real-data flow):

```bash
uv run hydrabflow-simulate simulator=stream_agama data.n_simulations=100000
uv run hydrabflow-simulate-multistream simulator=stream_agama data.n_simulations=2000
```

### Training

Train the global and local models separately (they are different networks with different adapters):

```bash
uv run hydrabflow-train simulator=stream_agama model=stream_fusion adapter=stream \
  composition=global preprocessing=stream_global augmentation=stream_global

uv run hydrabflow-train simulator=stream_agama model=stream_fusion adapter=stream \
  composition=local preprocessing=stream_local augmentation=stream_local training=stream_local
```

Each prints its `outputs/stream_agama/stream_fusion/<timestamp>/` run dir — the `model_dir` used
below.

### Evaluation (plots + metrics)

**On simulated test data** (truth available, per-level diagnostics):

```bash
uv run hydrabflow-evaluate model_dir=<global_run_dir> simulator=stream_agama model=stream_fusion \
  adapter=stream composition=global preprocessing=stream_global augmentation=stream_global \
  eval=stream_compositional

uv run hydrabflow-evaluate model_dir=<local_run_dir> simulator=stream_agama model=stream_fusion \
  adapter=stream composition=local preprocessing=stream_local augmentation=stream_local \
  training=stream_local eval=stream_compositional
```

Global evaluation loads the grouped multistream test set and runs `compositional_sample` (pooling
exchangeable stream members with the simulator's prior score); local evaluation samples each
stream separately, conditioned on its true globals, and writes one set of diagnostics **per
stream** (filename-prefixed). Both write into the run's `.hydra/`-adjacent output dir:

- `posterior.npz` — posterior samples
- `metrics.json` (local: `<stream>_metrics.json`) — RMSE + calibration error
- `recovery.png`, `calibration_ecdf.png`, `z_score_contraction.png` (local: `<stream>_*.png`)

**On real (observed) data** — no truth, chained global → local:

```bash
uv run hydrabflow-evaluate-real model_dir=<global_run_dir> data.real_data_path=<real.npz> \
  simulator=stream_agama model=stream_fusion adapter=stream composition=global \
  preprocessing=stream_real_global augmentation=stream_real_global eval=stream_compositional

uv run hydrabflow-evaluate-real model_dir=<local_run_dir> data.real_data_path=<real.npz> \
  simulator=stream_agama model=stream_fusion adapter=stream composition=local \
  preprocessing=stream_real_local augmentation=stream_real_local training=stream_local \
  eval=stream_compositional composition.global_run_dir=<global_eval_run_dir>
```

The real-data preprocessing/augmentation presets (`stream_real_*`) drop the synthetic
selection/noise steps (the real data is already observed) but keep the per-stream normalization,
loading its fitted stats from the training run's `preprocessing_state.npz`. The local pass needs
`composition.global_run_dir` — the *evaluation* run dir of a completed global real-data pass — to
draw the ancestral global conditions its posterior sampling needs. Output: `posterior.npz` +
`posterior_pairs.png` (local: one pair plot per stream).

### Hyperparameter tuning

`conf/tuning/stream.yaml` searches both fusion backbones (dotted paths into
`model.summary_network.params.backbones.<observable>.*`, including
`params.embed_dim_multiplier` so attention widths stay divisible by `num_heads`), the fusion head,
and the diffusion subnet:

```bash
uv run hydrabflow-tune simulator=stream_agama model=stream_fusion adapter=stream \
  composition=global preprocessing=stream_global augmentation=stream_global tuning=stream
```

Every trial's model, posterior samples, and diagnostics are saved under
`${tuning.artifacts_dir}/trials/trial_<number>/`; `best_trials.json` lists the Pareto-optimal
trials (RMSE + calibration error, multi-objective by default). Swap `composition=local
preprocessing=stream_local augmentation=stream_local training=stream_local` to tune the local
model instead.

### GPU

The AGAMA simulator is CPU-only (joblib-parallel across rows); training/evaluation/tuning run on
GPU if available — `jax[cuda12]` is a project dependency, and `hydrabflow.utils.backend` pins one
free GPU by default (`HYDRABFLOW_NUM_GPUS` to change).

## Future work (TODO)

*Roadmap from the 2026-07-04 misspecification-diagnosis session (see the `CLAUDE.md` Decisions
Log): cross-model posteriors agree on simulated test sets but diverge on the real Gaia streams —
the upgrades below address the suspected model misspecification, and should be promoted into
training only as the synthetic misspecification tests warrant.*

Both questions now have solid, evidence-grounded answers — and there's a neat closure: the member table you already use (`apjad382dt1_mrt.txt`) *is* the Ibata et al. 2024 STREAMFINDER atlas table, and their fit with essentially **your current family** (double power law, β fixed at −3) gives q = 0.75±0.03 — consistent with model_5's oblate solution, while your other models land at 1.1–1.3.

### 1. What more flexible potential family

Staged, so each step is falsifiable with the diagnostics we built:

**Stage A — free what's pinned, same family (YAML + prior edits only).** Free `beta_halo` (outer slope; Ibata+24 pinned it too, but your vcirc channel reaches ~25 kpc and has some grip on it), free the halo transition sharpness `alpha`, and give the **bulge one free amplitude** with a tight literature prior instead of pinning all five parameters — the bulge–disk trade-off at R ≈ 5–8 kpc feeds directly into Sigma_Disk and r_Disk, two of your three worst-tension parameters. Add R0/v_sun as **marginalized nuisances** (varied in simulation, excluded from `inference_variables` via the explicit adapter override — supported today). Priors can be centered on the shipped `McMillan17.ini` / `Cautun20.ini`.

**Stage B — radius-dependent halo flattening: the q-tension killer.** q is your most discordant parameter (6.7σ, oblate↔prolate), and a constant-q ellipsoid is the strongest rigidity in the model: if the real halo goes from oblate inside to rounder/tilted outside (as several analyses suggest, and as the [tilted-halo GD-1 result](https://arxiv.org/pdf/2504.07187) dramatizes), each stream at its own radius wants a different q, and no constant-q model satisfies all three streams + the curve — precisely your pathology. Two agama-native implementations: (i) **two Spheroid halo components** (inner/outer with independent q and scale radii — their sum gives a smooth q(r) at +2 parameters), or (ii) a user-defined Python density with q(r) = q₀ + (q∞−q₀)·r/(r+r_q) fed to `agama.Potential(type='Multipole', density=...)` — agama builds multipole potentials from arbitrary densities fast enough for dataset generation. I'd stop here before anything like free basis-expansion coefficients: with 3 streams the data can't constrain dozens of coefficients (Ibata needed 87 streams for a global fit), and morphology-based shape inference has known degeneracies ([the 2026 "illusion of morphology" analysis](https://arxiv.org/pdf/2604.06585)).

**Stage C — non-axisymmetric & time-dependent, as test-set physics first.** Agama ships three ready-made bars: `example_mw_bar_potential.py` (Portail+17/Sormani+22 CylSpline), `example_mw_potential_hunter24.py` (same bar, tuned MW-wide), and `example_mw_potential_khalil25.py` (bar **plus spiral arms with growth histories**, fitted to Gaia DR3-RVS; Khalil et al. 2025, A&A 699, 263) — plus the LMC rewind machinery in `example_lmc_mw_interaction.py` (rigid moving MW+LMC, reflex included). Pal5 is the known bar victim; the LMC matters at the few-km/s coherent-PM level for the outer stream parts. Don't put these in training yet — generate **misspecified test sets** with them and measure whether your trained models' posteriors actually move (the new misspecification stage + tension script give you the numbers). Promote to marginalized training physics only what bites.

### 2. More realistic stream simulation with agama — straight from their bundled examples

Your installed agama 1.0.157 ships `py/tutorial_streams.ipynb` + `example_tidal_stream.py`, which together give you a full realism ladder:

1. **Chen+25 spray instead of Fardal15** — the tutorial's spray section implements both variants (gala-style initial conditions). The [improved algorithm](https://arxiv.org/abs/2408.01496) ([ApJS 276, 32](https://iopscience.iop.org/article/10.3847/1538-4365/ad9904), [code](https://github.com/ybillchen/particle_spray)) is calibrated on N-body and reproduces stream morphology/kinematics to ~10%. Cheap drop-in for the `stream_agama` worker — and you can **mix recipes across training rows** so the network can't key on recipe-specific micro-structure.
2. **Progenitor self-gravity during stripping** — tutorial section "Include additional potential components during stream generation": add the moving progenitor potential to the host while integrating stripped particles. Changes width/epicyclic structure — exactly the small-scale features your SetTransformer sees.
3. **Self-consistent mass loss (restricted N-body)** — `example_tidal_stream.py`: progenitor represented by test particles, bound mass and its potential recomputed on the fly (dynamical friction included, negligible for GC masses). Seconds-to-minutes per stream — too slow for the 1M training set, ideal for **synthetic misspecification test sets** and for calibrating how wrong the spray recipe is at your S/N.
4. **Perturbation-theory upsampling** (tutorial, adopted from galax) — densify a coarse stream cheaply if the 1000-particle budget ever binds.
5. **Full N-body rung** — pyfalcon inside `example_tidal_stream.py`, or the gadget4/arepo patches in `example_nbody_simulation.py`: a handful of gold-standard streams for final validation.
6. **Time-dependent hosts** — all of the above compose with Khalil+25's growing bar+arms, the LMC rewind, and agama's Tilted/Rotating/Evolving modifiers, since agama potentials stack.

**How I'd sequence it:** leave-one-stream-out on real data (free) → synthetic misspecification test sets, one per component: Chen-vs-Fardal recipe, progenitor self-gravity, bulge/β/solar perturbations, bar (Khalil25), LMC (each is a small `simulate_multistream` config variant + one evaluation with the new stage) → rank by posterior shift and MMD → regenerate the 1M training set once with Stage A + whatever bit, plus Stage B's q(r) → retrain and re-run the real-data battery. That turns "new simulator?" into a ranked, measured shopping list instead of a leap of faith.

Sources: [Chen et al. 2025, ApJS 276, 32](https://iopscience.iop.org/article/10.3847/1538-4365/ad9904) ([arXiv:2408.01496](https://arxiv.org/abs/2408.01496), [code](https://github.com/ybillchen/particle_spray)) · [Ibata et al. 2024, ApJ 967, 89](https://iopscience.iop.org/article/10.3847/1538-4357/ad382d) · [tilted-halo GD-1 accelerations](https://arxiv.org/pdf/2504.07187) · [non-spherical-halo morphology degeneracies](https://arxiv.org/pdf/2604.06585) · [Bonaca & Price-Whelan, streams-in-the-Gaia-era review](https://www.sciencedirect.com/science/article/pii/S1387647324000204) · Khalil+25 / Hunter+24 / Sormani+22 / LMC examples: shipped in your `agama/py/`.

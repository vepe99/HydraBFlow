# HydraBFlow: SBI pipeline template (BayesFlow + Hydra)

A cookiecutter-style repo for Simulation-Based Inference. Infrastructure (dataset generation,
training, inference, tuning, tracing) is fixed; a new user writes a simulator and picks networks.

## Design principles

- **Full traceability**: every run writes its resolved Hydra config into its output dir. A run is
  valid only if it can be reconstructed from that folder.
- **Hydra-native**: all entry points are Hydra apps, no argparse.
- **Structured configs + registries, not `_target_`**: typed dataclasses in `config.py` (only
  `RootConfig` is in the ConfigStore, as `base_config`); YAML fills in values; a name string bridges
  config to code through the five registries in `hydrabflow/registry.py`. Components self-register
  (`@register_simulator`, `@register_step`, `@register_augmentation`, `@register_summary_network`,
  `@register_inference_network`), and each registry lazily imports its whole package on the first
  failed lookup (`Registry.discover`), so adding a component = drop a file + a config entry. No
  `__init__.py` edit.
- **The simulator owns the variable names**: empty `adapter.inference_variables` /
  `summary_variables` are filled from `parameter_names` / `observable_keys`
  (`pipeline.adapter.fill_adapter_from_simulator`). Explicit config wins (bring-your-own-data).
- **Single-level inference only**: one summary + one inference network via `bf.BasicWorkflow`. No
  hierarchical global/local split, no compositional scoring.
- **Preprocessing ≠ augmentation**: preprocessing is deterministic, whole-dataset, fit on the train
  split, state saved and replayed (`preprocessing/`); augmentation is stochastic and per-batch inside
  `fit_offline` (`augmentation/`).

## Stack

BayesFlow 2.x (Keras 3) on JAX — `KERAS_BACKEND=jax` pinned by `utils/backend.py`, imported first by
`hydrabflow/__init__.py`, GPU chosen by `autocvd`. `uv` for packaging (src-layout, `hydrabflow-*`
console scripts). Optuna for multi-objective tuning. Marimo for `notebooks/explore.py`.

## Layout

```
conf/
  config.yaml                  # everything: seed, run_name, model_dir, data, training,
                               #   preprocessing, augmentation, adapter, eval, tuning, hydra
  simulator/                   # two_moons, multimodal, protoplan (+ yours)
  model/summary_network/       # set_transformer | deep_set | time_series_transformer | fusion
                               #   | protoplan_fusion
  model/inference_network/     # flow_matching | diffusion | protoplan_flow_matching
  experiment/                  # protoplan: sets several plain blocks at once
src/hydrabflow/
  config.py                    # all dataclass schemas + register_configs()
  registry.py                  # all 5 registries + decorators + the 3 builders
  simulators/                  # USER: base.py, two_moons.py, multimodal.py,
                               #   protoplan.py + _protoplan_spec.py
  networks/factory.py          # shipped network builders
  networks/protoplan.py        # the disk's conditioned CNNs + modality dropout
  preprocessing/               # base, standardize, steps
  augmentation/noise.py
  augmentation/protoplan_instrument.py   # the JWST+ALMA instrument model
  pipeline/                    # INFRA: _app, adapter, workflow, io, artifacts, simulate,
                               #   train, evaluate, tune
  utils/                       # backend (JAX/GPU pin), seed, paths, oom
assets/protoplan/               # extinction law + real disks' SED error bars
notebooks/prior_predictive_checks/     # _realdisk.py + three marimo "is this real disk in the
                               #   training population?" notebooks (SED / amplitude / morphology)
tests/  docs/  outputs/ (gitignored)
```

## Stages

`hydrabflow-<stage>` (or `python -m hydrabflow.pipeline.<stage>`), output dir
`outputs/${simulator.name}/${run_name}/<timestamp>`:

- `simulate` — prior + forward model in chunks → `.npz` + a `<stem>.hydra/` config snapshot next to it.
- `train` — `.npz` → preprocessing (fit on train, save state) → `fit_offline` with augmentations →
  `approximator.keras`, `approximator_best.weights.h5`, `preprocessing_state.npz`, `loss.png`,
  `history.json`.
- `evaluate` — loads model + preprocessing state from `model_dir`, samples the posterior. One code
  path, two modes: simulated test set (truth-aware diagnostics + `metrics.json`) or
  `data.real_data_path` set (posterior pair plots only).
- `tune` — Optuna multi-objective (RMSE + calibration error) over `tuning.search_space`;
  `best_trials.json`, study in `tuning.storage_dir` (concurrency-safe log → N parallel launches
  extend one study).

## What the user modifies

`conf/simulator/<name>.yaml` + `src/hydrabflow/simulators/<name>.py`; the network group YAMLs;
knobs in `conf/config.yaml`. Optionally custom preprocessing steps, augmentations, or network
builders. The `adapter:` block stays untouched unless there is no simulator. Nothing else.

**Fixed infrastructure**: `pipeline/` (stages, `make_cli`, adapter/workflow builders, io,
artifacts), `config.py`, `registry.py`, `utils/backend.py`, the `hydra:` block in `config.yaml`.

## Notes worth keeping

- `training.learning_rate` is the peak LR passed as `initial_learning_rate`; `BasicWorkflow`
  wraps it in cosine decay with 5% warmup + AdamW. Passing an explicit optimizer disables that
  schedule, so `training.optimizer` deliberately does not exist.
- `training.standardize` is `[inference_variables]` only — the summary side is handled by the
  `standardize` preprocessing step, whose fitted state is saved and replayed.
- Networks take `embed_dim_per_head` (attention width = `num_heads * embed_dim_per_head`), so any
  tuner draw stays divisible by the head count.
- Fusion is implemented, not a seam: >1 `summary_variables` → one backbone per key behind
  `bf.networks.FusionNetwork`; per-key type via `model.summary_network.params.backbones`.
- `load_approximator` passes `compile=False` (evaluation never resumes training) — without it,
  loading a TimeSeriesTransformer fails on AdamW slot-variable shapes.
- **Any fusion network breaks `keras.models.load_model`** — stock `bf.networks.FusionNetwork`
  included (reproduce with `simulator=multimodal`). Keras walks the layer tree by attribute name:
  the branches are saved under `layers/<fusion>/backbones/...` (reached via the model's `layers`
  property) and looked for under `summary_network/backbones/...` on load, so each reports
  "expected 2 variables, but received 0". Hence `load_approximator(run_dir, workflow=,
  probe_batch=)`: it falls back to rebuild-from-config + `load_weights`, which is keyed by
  construction order and immune. Single-backbone models (`two_moons`) load fine.
  `tests/test_protoplan.py::test_saved_approximator_reloads_with_its_weights` pins it.
- The preprocessing framework cannot be replaced by adapter transforms: `Adapter.standardize`
  requires caller-supplied mean/std, and the adapter is per-batch, so `drop_nan` (row filtering)
  and `train_val_split` are impossible there.
- Open: no `fit_online` support — the pipeline is offline-only, so a cheap simulator cannot skip
  the `simulate` stage.

## docs/

Three files, deliberately short: `running.md` (install, stages, walkthroughs, real data, tuning,
GPU env vars), `configuration.md` (config blocks + the three groups), `extending.md` (the
drop-a-module-and-decorate pattern for all five extension points).

## graphify

Knowledge graph at `graphify-out/`.

- For codebase questions, run `graphify query "<question>"` first (also `graphify path "<A>" "<B>"`,
  `graphify explain "<concept>"`) — a scoped subgraph, far smaller than `GRAPH_REPORT.md` or grep.
- `graphify-out/wiki/index.md`, if present, beats raw source browsing for navigation.
- Read `GRAPH_REPORT.md` only for broad architecture review.
- After changing code, run `graphify update .` (AST-only, no API cost).


## The protoplanetary-disk arm (`protoplan_sbi` branch)

Multimodal NPE for protoplanetary disks: an external radiative-transfer code produced ~49k
(parameters, clean image, clean SED) rows, and a posterior over 17 physical disk parameters is
trained from *mock observations* of them. **Full write-up:
[docs/protoplanetary_disk.md](docs/protoplanetary_disk.md)** -- read it before changing anything
here. Run it with

```bash
uv run hydrabflow-train experiment=protoplan
uv run hydrabflow-evaluate experiment=protoplan model_dir=outputs/protoplan/protoplan_npe/<ts>
```

`hydrabflow-simulate` is **not** used: the forward model is not runnable in-process, so
`ProtoplanetaryDiskSimulator` implements the two optional `BaseSimulator` hooks instead --
`load_dataset` (reads the `.npy` caches) and `build_adapter` (a layout the four `AdapterConfig` key
lists cannot express). Those hooks are the only additions to the framework itself, plus
`registry.load_dataset`, the `tail_split` preprocessing step, `eval.diagnostics:
per_discrete_branch`, and one bug fix (`standardize: [all]` used to reach BayesFlow as a list,
which standardizes nothing -- see `pipeline/workflow.py`).

### The instrument model is an augmentation

`augmentation/protoplan_instrument.py` turns a clean RT row into a mock JWST + ALMA observation on
the fly: extinction, distance rescaling, Jy/px -> MJy/sr, the JWST stpsf PSF and per-band ALMA
Gaussian beams, a random in-plane rotation and parity flip, resampling to each instrument's own
pixel grid, then instrument noise. It is fully vectorised over the batch axis, all randomness flows
through one JAX PRNG key, and the heavy functions are `@staticmethod @jax.jit` *on purpose* --
defining them as closures inside methods would make a new function object per call and force full
retracing. Two invariants worth keeping: the pipeline's images stay **linear** (compression is the
adapter's job, where it is recorded per run and invertible), and no step may overwrite an adapter
input in place.

Every constructor argument is an `augmentation.params` knob, which is how the beam and the noise
are configured -- either one fixed measured setup (`randomize_alma_setup: false`, with
`alma_beams: [[maj, min, PA], ...]` and `alma_noises_jy_beam` / `jwst_noise_mjy_sr`) or a prior to
marginalise over (`true`, with the `*_range` knobs; there is no separate minor-axis range, since
minor = `axis_ratio * major`). Requires `stpsf` reference data via `$STPSF_PATH`.

**`px_arcsec_mod` must match the image cache's grid.** The cached image variants all cover the same
3" field and differ only in sampling, but the PSF/beam kernels are sized from `px_arcsec_mod` in
`__init__`. A mismatch is a *silent beam error*, not a shape error -- `preprocess` therefore checks
it against the batch's own H on the first call.

### The encoding: three coupled choices

- **Discrete-as-conditions.** `sil_id` and a derived `has_cavity` are `inference_conditions`, not
  targets, so the estimator is `p(theta_17 | data, sil_id, has_cavity)`. A flow cannot represent a
  delta atom; the alternative was smearing both indicators into continuous clusters. `log_r_cav`
  stays a target and on the no-cavity rows is drawn from the *real* cavity branch, so "posterior =
  prior there" is an honest statement about an unidentified parameter. What comes out is four
  *conditional* posteriors -- nothing estimates `p(sil_id, has_cavity | data)`, so they cannot be
  mixed into one marginal without separately estimated weights.
- **Conditions in the summary networks.** The geometry and beam/noise scalars are spliced into the
  CNN trunks (`ConditionedConvolutionalNetwork`), after the conv stages and pooling head but before
  the projection to `summary_dim`. Whether a faint ring survived the beam is a question about the
  image, and the CNN is the only network that sees the image. Routing is per instrument and defined
  once, in `_protoplan_spec.summary_condition_routing`.
- **asinh images.** `asinh(x / sigma_c)` per band, `sigma` from `simulator.params.asinh_sigma_*`.
  Without it the only preprocessing is one global mean/sd per channel, which cannot serve a channel
  spanning five decades: the sd is set by the inner rim, so the background's p10-p90 span collapses
  to 0.004 sd on JWST. asinh and not log because ~40-48% of post-noise pixels are negative.
  **Change the knees when you change the noise levels** -- far enough off and the transform
  degenerates into a near-identity or a pure log.

### Four image branches, one per band

JWST plus ALMA B9 (450um), B7 (880um), B6 (1300um) each get their own CNN rather than being stacked
as channels of one, because each band is observed with its own beam: a shared trunk would have to
serve three point-spread functions at once, and the per-band beam/noise conditions could only be
handed to it pooled. One `*_alma` hyperparameter configures all three; add `*_b9`/`*_b7`/`*_b6` to
differ per band (`networks/protoplan.py::_tag`).

`missing_modality_prob > 0` swaps `FlowMatching` for `GroupedFlowMatching`, which drops each of
{B6, B7, B9, JWST, SED} from the condition vector as a *whole group*. Upstream's
`missing_conditions_prob` drops each element independently, so a modality's slice essentially never
goes together (prob `p ** summary_dim`) -- that trains resilience to feature dropout, not to a
missing modality. It requires `fuse_head: false` (an MLP head mixes the branches, leaving no
boundary to drop) and never drops the two discrete indicators, which are conditions on the density
rather than a measurement.

### Prior-predictive checks on a real disk

`notebooks/prior_predictive_checks/` asks whether a real observation is inside the training
population, in three spaces: `check_sed_range.py` (SED; CPU-only, needs nothing but `assets/`),
`check_obs_range.py` (amplitude, which is what `standardize: [all]` makes load-bearing) and
`check_shape_range.py` (morphology through the disk's own beam, at its measured PA). Figures land in
`plots/<disk>/`, gitignored. `docs/protoplanetary_disk.md` §8 is the write-up.

`_realdisk.py` holds everything shared -- the per-disk `DISKS` registry, `build_real_batch` (the only
reader of a real observation), and the scoring toolkit ported from upstream so the numbers stay
comparable. Nothing under `src/` changed for it.

Three traps, all documented in §8: the real images are **not** in this repo (upstream `dill` pickles,
`$PROTOPLAN_OBS_DIR`); the regrid target must match the forward model's ALMA grid (this disk's own
measured beams give 122 px, the randomized training prior 162 -- `check_obs_range` therefore passes
`alma_px=randomized_alma_obs_px(cfg)`, and both notebooks assert rather than trust); and the
distance, `A_V` and JWST noise floor per disk are **assumptions**, declared once in `DISKS` and
echoed at the top of every notebook. A disk missing a band is a missing modality, not an error --
oph163131 has no B7, so nothing b7 is plotted or scored.

Fixed-PA observing needs no new code: `apply_rotation` already collapses to one angle when
`rot_range_deg=(theta, theta)`. Neither does sampling extinction: an `(lo, hi)` `av` already means
"one A_V per disk, marginalised". All three notebooks take an `A_V` override (`$CHECK_AV`, or the
widget) and write to `plots/<disk>_av<lo>-<hi>/` so they cannot clobber the as-trained figures.

**What A_V can reach.** Extinction is one multiplicative scalar per channel, so over A_V in [1,5] the
swing is 1.72 dex at 0.5um, 0.12 at 3.9um, and **under 0.001 dex at 450/880/1300um**. An ALMA finding
is therefore not an extinction problem, and no amount of A_V sampling will move it -- pinned by
`test_extinction_is_negligible_at_alma_wavelengths_over_any_av`. Every consumer of `av` must handle
both a scalar and a range (`parse_av` / `av_ref` / `extinction_correction`); a scalar-only assumption
fails silently by broadcasting or loudly on `np.isnan(tuple)`.

`check_summary_range` is the fourth check, in the network's own summary space; it needs a finished
run, scores per instrument (a disk's absent band is its positive control), and caches the embeddings
to `<plot_dir>/summaries.npz`. `check_summary_pca` decomposes its Mahalanobis distance over the PCA
basis -- judge in the 99%-variance subspace, since most of the raw d2 sits in near-degenerate
directions. `notebooks/overlay_corner.py` overlays several evaluate runs on one corner plot.

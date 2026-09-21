# HydraBFlow: SBI pipeline template (BayesFlow + Hydra)

A cookiecutter-style repo for Simulation-Based Inference. Infrastructure (dataset generation,
training, inference, tuning, tracing) is fixed; a new user writes a simulator and picks networks.
This branch (`stream_project`) hosts the stellar-stream / Milky-Way-potential project on top of it
(hierarchical compositional inference, AGAMA stream simulators, the Gaia observation model); the
research history is the Decisions Log at the end of this file.

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
  (`pipeline.adapter.fill_adapter_from_simulator`); under `composition=global|local` from the
  simulator's `global_parameter_names` / `local_parameter_names` / `context_keys`. Explicit config
  wins (bring-your-own-data). `fill_stream_grid_from_simulator` does the same for the stream
  simulators' rotation-curve grid.
- **Single-level inference by default, compositional as an opt-in**: `composition=none` is
  `bf.BasicWorkflow`; `global` / `local` switch to `bf.CompositionalWorkflow` and train/evaluate one
  level of a hierarchical simulator (see `docs/streams.md`).
- **Preprocessing ≠ augmentation**: preprocessing is deterministic, whole-dataset, fit on the train
  split, state saved and replayed (`preprocessing/`); augmentation is stochastic and per-batch inside
  `fit_offline` (`augmentation/`).

## Stack

BayesFlow 2.x (Keras 3) on JAX — `KERAS_BACKEND=jax` pinned by `utils/backend.py`, imported first by
`hydrabflow/__init__.py`, GPU chosen by `autocvd`. `uv` for packaging (src-layout, `hydrabflow-*`
console scripts). Optuna for multi-objective tuning. Marimo for `notebooks/explore.py`. Streams:
`agama` (CPU/joblib), `astropy`, `corner`.

BayesFlow stays on the PyPI **2.0.12** release: `pipeline/_bf_patches.py` fixes its
compositional-conditions reshape bug and is version-specific. `main` tracks the bayesflow git HEAD
(2.0.13) — re-check the patch before following.

## Layout

```
conf/
  config.yaml                  # everything: seed, run_name, model_dir, data, training,
                               #   preprocessing, augmentation, adapter, composition, eval, tuning
  simulator/                   # two_moons, multimodal, stream_agama*, ...
  model/summary_network/       # set_transformer | deep_set | time_series_transformer | fusion | stream_fusion_*
  model/inference_network/     # flow_matching | diffusion
  model/                       # whole-model presets (stream_fusion_model5, ...): model=<name>
  adapter/ augmentation/ preprocessing/ training/ tuning/ eval/ composition/
                               # stream PRESETS only; the defaults are the blocks in config.yaml
src/hydrabflow/
  config.py                    # all dataclass schemas + register_configs()
  registry.py                  # all 5 registries + decorators + the 3 builders
  simulators/                  # USER: base.py, two_moons.py, multimodal.py; stream_agama*.py, stream_common.py
  networks/                    # factory.py (shipped builders), fusion.py, masked_*_transformer.py
  preprocessing/               # base, standardize, steps, streams
  augmentation/                # noise.py; streams.py (Gaia observation model), stream_summary.py
  pipeline/                    # INFRA: _app, adapter, workflow, io, artifacts, simulate,
                               #   simulate_multistream, train, evaluate (+ evaluate_real helpers),
                               #   tune, compositional, misspecification, _bf_patches
  utils/                       # backend (JAX/GPU pin), seed, paths, oom, progress, quiet, reporting
scripts/                       # stream dataset/training/tuning runners + PPC & diagnostics scripts
assets/gaia/                   # small static Gaia inputs (git-tracked); datasets live in data*/ (gitignored)
tests/  docs/  outputs/ (gitignored)
```

## Stages

`hydrabflow-<stage>` (or `python -m hydrabflow.pipeline.<stage>`), output dir
`outputs/${simulator.name}/${run_name}/<timestamp>`:

- `simulate` — prior + forward model in chunks → `.npz` + a `<stem>.hydra/` config snapshot next to
  it. Chunks are checkpointed to a sidecar dir, so a crashed multi-hour stream run resumes.
- `simulate_multistream` — compositional datasets: one shared global draw per row, one observation
  per group member (`sample_compositional`) → grouped `.npz` (globals `(n,1)`, members `(n,m,...)`).
- `train` — `.npz` → preprocessing (fit on train, save state) → `fit_offline` with augmentations →
  `approximator.keras`, `approximator_best.weights.h5`, `preprocessing_state.npz`, `loss.png`,
  `history.json`, `convergence.json`.
- `evaluate` — loads model + preprocessing state from `model_dir`, samples the posterior. One entry
  point: simulated test set (truth-aware diagnostics + `metrics.json`; at `composition=global` both
  `base_*` per-member and `compositional_*` pooled, plus `summaries.npz` for the MMD test) or
  `data.real_data_path` set (posterior pair plots; at `composition=global` also
  `single_stream_posterior.npz`, `real_global_vs_streams_corner.png`, `misspecification.json`;
  at `local` ancestral sampling from `composition.global_run_dir`). Real-data code:
  `pipeline/evaluate_real.py` (a module, not a stage).
- `tune` — Optuna multi-objective (RMSE + calibration error) over `tuning.search_space`;
  `best_trials.json`, study in `tuning.storage_dir` (concurrency-safe log → N parallel launches
  extend one study); OOM backoff on fit and sampling.

## What the user modifies

`conf/simulator/<name>.yaml` + `src/hydrabflow/simulators/<name>.py`; the network group YAMLs;
knobs in `conf/config.yaml` or a preset under `conf/<group>/`. Optionally custom preprocessing
steps, augmentations, or network builders. The `adapter:` block stays untouched unless there is no
simulator. Nothing else.

**Fixed infrastructure**: `pipeline/` (stages, `make_cli`, adapter/workflow builders, io,
artifacts), `config.py`, `registry.py`, `utils/backend.py`, the `hydra:` block in `config.yaml`.

## Notes worth keeping

- `training.learning_rate` is the peak LR passed as `initial_learning_rate`; `BasicWorkflow`
  wraps it in cosine decay with 5% warmup + AdamW. Passing an explicit optimizer disables that
  schedule, so `training.optimizer` deliberately does not exist.
- `training.standardize` is `[inference_variables, summary_variables]` here (main: inference only):
  the stream observables are standardized per batch *after* augmentation, not by a preprocessing
  step, and the trained stream models depend on it. For the plain `standardize`-preprocessed path
  the second pass is a near-identity. `training=stream_local` uses `inference_conditions` instead.
- Networks take `embed_dim_per_head` (attention width = `num_heads * embed_dim_per_head`), so any
  tuner draw stays divisible by the head count — also inside the fusion `params.backbones` and in
  the `conf/tuning/stream*.yaml` search paths (was `params.embed_dim_multiplier`).
- Fusion: >1 `summary_variables` → one backbone per key behind `bf.networks.FusionNetwork` with
  per-key type via `params.backbones={key: type}`; `type: fusion` (`networks/fusion.py`) instead
  takes full per-key specs, an MLP head and `mask_backbone` (the one backbone that receives the
  `summary_attention_mask`); its builder is flagged `consumes_grouped_inputs` so the registry does
  not wrap it again.
- The stream presets are Hydra groups with **no default file**: `conf/config.yaml` lists them as
  `- adapter: null` etc. and puts `_self_` *before* the groups so a selected preset wins over the
  inline block. Whole-model presets (`conf/model/*.yaml`) select their nets with
  `- override summary_network: <name>` (relative; an absolute `/model/...` path gets a doubled
  package). Preset YAMLs may inherit other presets in the same group (`defaults: - stream_global`).
- `load_approximator` passes `compile=False` (evaluation never resumes training) — without it,
  loading a TimeSeriesTransformer fails on AdamW slot-variable shapes.
- `BaseSimulator.sample` checks the declared names are present with leading axis `n` but allows
  extra keys — the stream simulators return fixed constants, `j`, derived diagnostics
  (`*_derived`, `m_bound_final`) that the adapter drops (`select_adapter_keys`).
- Augmentation factories take `(params, rng, context)`; `context["pipeline"]` is the fitted
  preprocessing pipeline (`per_stream_standardize` reads `stream_observation_stats` from it).
- Stream augmentations read Gaia tables from `augmentation.params.resources_dir` (default `data/`,
  a symlink to shared storage); on a fresh clone pass `++augmentation.params.resources_dir=assets/gaia`.
- Shared box: default OpenBLAS/XLA thread pools hit the per-user thread limit on the 256-core host
  when other users run large jobs (`pthread_create failed`, `mmap error 12`); run tests with
  `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false
  intra_op_parallelism_threads=1" JAX_PLATFORMS=cpu HYDRABFLOW_NUM_GPUS=0`. GPU runs: `autocvd`.
- Open: no `fit_online` support — the pipeline is offline-only.

## docs/

`running.md` (install, stages, walkthroughs, real data, tuning, GPU env vars), `configuration.md`
(config blocks + groups), `extending.md` (the drop-a-module-and-decorate pattern),
`streams.md` (the compositional stream project: levels, stages, presets, resources).

## graphify

Knowledge graph at `graphify-out/`.

- For codebase questions, run `graphify query "<question>"` first (also `graphify path "<A>" "<B>"`,
  `graphify explain "<concept>"`) — a scoped subgraph, far smaller than `GRAPH_REPORT.md` or grep.
- `graphify-out/wiki/index.md`, if present, beats raw source browsing for navigation.
- Read `GRAPH_REPORT.md` only for broad architecture review.
- After changing code, run `graphify update .` (AST-only, no API cost).

## Decisions Log

*(Update this section after each Claude Code session)*

- [x] Folder structure finalized
- [x] Config group schema defined (structured dataclasses in `config/schema.py`, no `_target_`)
- [x] Base simulator interface defined (`simulators/base.py` + registry; skeleton stub shipped)
- [x] Training loop scaffold written (`pipeline/train.py` via `bf.BasicWorkflow.fit_offline`)
- [x] Five run stages implemented + verified end-to-end on a temporary Gaussian simulator
- [x] Preprocessing module (deterministic, fit-on-train, save/load) separate from augmentation
- [x] Optuna multi-objective tuning wired
- Session 1 decisions: JAX backend; structured-dataclass configs (overrides original `_target_`
  plan); skeleton-only example simulator; single-observable summary, fusion-ready; single-level
  inference (no global/local, no compositional scoring).
- Session 2026-07-03 (user-friendliness pass): default simulator = `two_moons` (first run succeeds
  out of the box; skeleton stays as the copyable stub); restored timestamped `hydra.run.dir`
  (removed leftover debug value); packages auto-import their modules via `utils.discovery` so
  dropped components self-register without `__init__.py` edits; adapter variables derive from the
  simulator when left empty (`fill_adapter_from_simulator` in `pipeline/_app.py`), explicit config
  wins; summary/inference network builders moved from if/elif to registries
  (`@register_summary_network` / `@register_inference_network`) with free-form `params` in both
  network schemas for custom builders; dev deps moved to `[dependency-groups]` so `uv sync`
  installs pytest/ruff by default.
- Session 2026-07-03 (stream_project branch — compositional stream port): reference =
  `/export/data/vgiusepp/latest_bayesflow/diffusion-experiments/case_study5/project_stream`
  (read-only; the `*_agama`/`rotationcurve` variants are the current generation; the
  prototyping names — `jonas_streamnorm`, `_gaiastreams_`, `nocomposition`, `_new` — do not
  survive here). Ported so far:
  - `stream_agama` simulator (particle-spray streams + model rotation curve `vcirc_kms` as a
    second observable; CPU/joblib; hierarchical global potential / local per-stream phase-space
    parameters; `identity` prior entries = fixed constants; inferred = non-identity). Other
    stream simulators (gala CPU, odisseo/galax/StreaMax GPU) still to port on this seam.
  - `composition` config group (none|global|local) + `bf.CompositionalWorkflow`; adapter
    derivation is level-aware; `simulate_multistream` stage for grouped test sets.
  - Fusion: `MaskedFusionNetwork` (summary type `fusion`, one backbone per grouped observable,
    attention mask routed to `mask_backbone` only); stock bf SetTransformer is mask-aware, only
    the fusion wrapper is custom. `adapter.attention_mask_key` renames the batch mask to
    BayesFlow's `summary_attention_mask` role.
  - "jonas_streamnorm" split into preprocessing: `per_stream_parameter_standardize` (local
    params z-scored by their stream's prior mean/std; invertible, replayed on posteriors) +
    `stream_observation_stats` (per-stream obs stats + log10-vcirc bin stats fitted on train,
    applied per batch by the `per_stream_standardize` augmentation *after* the physical-unit
    augmentations). Preprocessing state keys are name+occurrence based so training state loads
    under the `stream_real_*` presets.
  - Gaia observation model as registered augmentations (`augmentation/streams.py`, NumPy port
    of AugmentationsClass); resources (member magnitudes, DR3 error tables, id mapping) read
    from `data/` (a symlink to shared storage — never modify existing `.npz` there; simulator
    tests write to `data/hydrabflow_testsimulator/`).
  - Eval: global = `compositional_sample` (+ prior score from the simulator's prior spec,
    `eval.sample_kwargs` for method/steps/compositional_bridge_d1); local simulated =
    per-member conditioned on true globals with per-stream diagnostics; real data = global
    posterior then local `ancestral_sample` via `composition.global_run_dir`.
    `pipeline/_bf_patches.py` fixes the bayesflow 2.0.12 compositional-conditions reshape bug.
  - Tuning searches nested fusion params via dotted paths; `embed_dim_multiplier` keeps
    attention widths divisible by head counts; trials train with the augmentation chain.
  - agama dependency builds without isolation and with auto-yes prompts (`[tool.uv]` settings);
    `.gitignore` `data/` pattern root-anchored (it was swallowing `conf/data/`).
- Session 2026-07-04 (misspecification diagnostics): quantified that the reference tuned models
  agree on shared simulated test sets (pairwise tension z≲0.2 median) but diverge on the real Gaia
  streams (q_halo up to 6.7σ, Sigma_Disk 5.7σ, r_Disk 3.9σ) — misspecification-driven
  extrapolation, not training instability. Added a summary-space misspecification stage
  (`pipeline/misspecification.py`, Schmitt+21 MMD): `evaluate composition=global` saves
  `summaries.npz` (reference set, best-effort hook), `evaluate_real composition=global` MMD-tests
  the observed members against it when `eval.misspecification_reference` points at that run dir —
  with a stratified-by-`j` bootstrap null (uniform trios distort the null since summaries encode
  stream identity) and per-stream Mahalanobis percentiles that localize which member is OOD.
  Artifacts: `misspecification.json` + `mmd_hypothesis_test.png`; all hooks defensive (never abort
  chained runs). `scripts/report_cross_model_tension.py` = offline cross-run posterior tension
  report (real vs sim). Verified end-to-end on the 2026-07-03 smoke model (GPU 1).
- Session 2026-07-05 (restricted N-body simulator + prior predictive check): new
  `stream_agama_rnbody` simulator (subclasses `stream_agama`; agama example_tidal_stream method —
  self-consistent Plummer progenitor particles + periodically refit moving Multipole potential, no
  dynamical friction (GC masses); NaN-guarded workers, per-worker `agama.setNumThreads`). ~10-25
  s/row at 1 thread vs ~4-7 s spray. PPC on 120 matched prior draws vs the real Gaia members
  found a **t_end inconsistency**: at the spray-era t_end=1.5 Gyr, restricted N-body cannot grow
  the observed ~100° arms (real-locus reach NGC3201 13% / M68 8% vs spray 93% / 56%; median 0
  in-window particles for NGC3201, whose progenitor lies outside its RA window) — spray fabricates
  stripping uniformly over t_end so it never noticed; Pal 5 (t_end=4) is perfect in both. t_end=4
  Gyr recovers reach ~42% / 73% ⇒ per user decision, `conf/simulator/stream_agama_rnbody.yaml` now
  overrides NGC3201/M68 t_end to 4.0 (documented in the yaml). 30k-row training set generating to
  `data/streams/data_agama_rnbody_hydrabflow/training_data_30000.npz` (20 nice'd workers).
  agama>=1.0.157 gotcha: `getUnits()` returns astropy Quantities once astropy is imported in the
  process — both simulators' workers harden `time_unit_gyr` against it.
- Session 2026-07-07 (rotation-curve rejection prior): optional `params.vcirc_rejection` on the
  `stream_agama` BASE class truncates the global-potential prior by rejecting draws whose model
  rotation curve is grossly incompatible with the observed **Zhou et al. (2023)** curve
  (== `stream_common.OBS_VC_KMS`; its header comment previously mislabeled "Eilers", now fixed).
  Accept iff `stat`(median|max) of `|vc_model - vc_obs|/vc_obs` over the r>`r_min_kpc` bins is <
  `max_frac_dev`. Screens only the potential (~18 ms/draw), so the expensive stream integrator is
  never spent on wild galaxies. Impl in `simulators/stream_agama.py`: `_vcirc_accept_worker`
  (joblib), `_vcirc_accept_mask`, `_rejection_sample` (adaptive batch, aborts >5M draws); wired
  into `sample_prior` + `sample_compositional`, both no-op when the key is absent (spray/two_moons/
  base configs unaffected; all tests pass). `conf/simulator/stream_agama_rnbody.yaml` sets
  `stat: median, max_frac_dev: 0.20, r_min_kpc: 5.5`. **Why this criterion**: strict 5σ of the Zhou
  observational errors accepts 0/100k prior draws (inner-disk σ~0.2 km/s demands <1 km/s agreement);
  a physical fractional cut is required. PPC (100k draws): median-20% acceptance ~24.7%; accepted
  draws' max-radius deviation ≤46% at the 90th pct. **Compositional-score validity**: a hard
  indicator cut contributes zero score inside the accepted region, so the compositional
  `prior_score_from_spec` stays correct UNCHANGED — the networks learn the truncation implicitly
  from rejection-sampled training data (no change to `pipeline/compositional.py`; optional future
  add: post-hoc filter + report the fraction of compositional draws that leak outside the cut).
- Session 2026-07-08 (real-data global-vs-streams corner): `evaluate_real composition=global` now
  always emits `real_global_vs_streams_corner.png` + `single_stream_posterior.npz`
  (`_save_global_vs_streams_corner` in `pipeline/evaluate_real.py`). Overlays the pooled global
  (compositional) posterior and each per-member `workflow.sample` single-stream posterior over the
  shared global parameters, in physical units, with 68/95% contours + marginals (ChainConsumer
  `shade=False` look from the reference `main_eval_gaiastreams.py` `global_cornerplot`). Reuses the
  sim-eval "base" per-member sampling mechanism + `target_streams` name map; shared robust
  percentile ranges so overlays register; best-effort (never aborts). On the model_5 real-data run
  it localizes the misspecification: M68 alone pulls q_halo low (~1.0 vs ~1.5) and Sigma_Disk high
  (~1e9 vs ~5e8), while the pooled Global tracks the Pal5/NGC3201 consensus.
  Stream-level PPC (30 accepted potentials, rnbody, seed=2026): real-locus reach 100%/93%/90%
  (Pal5/NGC3201/M68), 0% NaN — the cut REMOVES wild potentials that threw M68 off-locus, lifting
  its reach above the unconstrained t_end=4 check (~60%). Planned rejection-prior datasets (30k flat
  + 333 multistream) and model_5 training were scoped this session but NOT run yet.
- Session 2026-07-08 (Chen+2024 spray + Zhou∪Huang rejection + portable assets + 30k dataset):
  goal = a spray training set consistent with BOTH observed rotation curves. Three additions, all
  config-driven on the existing `stream_agama` class (no new simulator class):
  - **Chen, Gnedin & Li (2024) release recipe** as `params.spray_method: chen` (`_ic_chen_spray`,
    a 1:1 port of gala `ChenStreamDF._sample` — 6D (r,φ,θ,v,α,β) Lagrange-point draw with the
    r–α covariance; verified agama's R-matrix/`einsum` frame == gala's transform, so it is a clean
    IC-only swap). Fardal+2015 stays the default (`spray_method` absent ⇒ fardal).
  - **Extended rotation-curve observable** (`params.obs_r_grid: extended`, `obs_r_split_kpc: 24`):
    the stored `vcirc_kms` AND the rejection grid become the Zhou(2023)∪Huang(2016) union (50
    radii, single source of truth `stream_common.extended_rotation_curve`). `beta_..._halo` freed
    to uniform[2,4] (was identity 3.0) so the model can grow Huang's declining outer curve.
  - **Banded `vcirc_rejection`** (`params.vcirc_rejection.bands`, per-band criterion/stat/thresh):
    zhou band 5.5–24 kpc fractional median<0.20; huang band r>24 sigma median<2.0 (respects
    Huang's heteroscedastic 7–50 km/s errors — a flat fractional cut can't). Still a hard
    indicator ⇒ compositional prior score stays valid unchanged. `_build_accept_bands` +
    rewritten `_vcirc_accept_worker` (list of bands; row accepted iff every band passes).
  Config: `conf/simulator/stream_agama_spray_huang.yaml` (`name: stream_agama`). Helper scripts:
  `compare_spray_methods.py` (Fardal vs Chen vs real overlay), `probe_vcirc_acceptance.py`
  (per-band + combined acceptance), `extend_vcirc_huang.py`; `ppc_prior_predictive.py` generalized
  to the 50-radii grid (Zhou+Huang overlays, log-x, split line). Pilot: Zhou band 24.6%, Huang
  15.4%, combined 7.9% (~12.6 screens/accepted row). **30k dataset generated** (Chen spray,
  rejection prior, seed=2026, 24 workers, 40.5 min) →
  `data/streams/data_agama_spray_huang_hydrabflow/training_data_30000.npz` (30000 rows,
  `vcirc_kms` (30000,50,1), beta spanning [2,4] mean 3.10, streams 9922/10039/10039, 83 NaN rows
  0.28%). Chunk-0 PPC: prior band brackets both curves, prior-median fracdev 5.6% vs Zhou / 5.3%
  vs Huang (old fixed-β was ~21% off Huang); real-locus reach 100/99/97%.
  **Portability (`assets/gaia/`, git-tracked, ~3.6 MB)**: copied the small static inputs (Gaia
  member/error tables from the `data/` symlink; real observed-stream npz + reference tracks from
  the reference project) so the repo is self-contained for moving to a GPU-less cluster.
  README documents provenance + new-cluster wiring (`+augmentation.params.resources_dir=assets/gaia`).
  Dataset creation needs NONE of these (rotation curves are hardcoded in `stream_common`); they
  serve the PPC scripts + training/real eval. `ppc_prior_predictive` `DEFAULT_REAL` now prefers
  the in-repo copy. Committed as b3692a1 (not pushed).
  **Extended-grid training wiring** (was deferred, now done — commits b1599cc/b681ed2): the
  50-radii observable needed wiring before `train`/`evaluate*` would run, since `mask_vcirc_radii`
  (raised on 50≠34), `add_noise_to_vcirc` σ, and (real data) `attach_observed_vcirc` all hardcoded
  the 34-Zhou grid. Fix mirrors `fill_adapter_from_simulator`: `stream_agama` gained
  `obs_sigma_vc` + `obs_vc_kms` properties (extended → Zhou∪Huang per-bin σ / observed curve),
  and `adapter.fill_stream_grid_from_simulator` (wired into `_app.py`) injects the simulator's
  radii + per-bin σ + observed curve into those three config nodes when `obs_r_grid=extended` and
  the user left them unset — no-op for the default Zhou path. Gotcha fixed: `getattr(DictConfig,
  "values")` returns the bound `.values()` method, not the key — use `step.get("values")`.
- Session 2026-07-08 (model_5 train/eval/eval_real on the spray+Huang 30k): trained
  `stream_fusion_model5` (composition=global, plain stream_global preproc+aug, 300 epochs, GPU 0,
  23.6 min, val_loss 1.20, convergence OK) on `training_data_30000.npz`; generated the 333-group
  test set `simulation_multistream_333.npz`. **Sim eval** (base=per-member, compositional=pooled):
  base RMSE 0.628 / calib 0.029; compositional RMSE 0.561 / calib 0.048 (well-calibrated; pooling
  improves accuracy). **Real eval** (Gaia Pal5/NGC3201/M68): global q_halo 1.15 [0.99,1.28],
  beta_halo **3.26 [2.97,3.55]** (data-constrained, not railing the freed [2,4] prior),
  Sigma_Disk 4.98e8, gamma_halo 1.06, r_Disk 2.83. **Key win**: the per-stream disagreement that
  plagued the earlier rnbody model_5 (M68 pulling q_halo→1.0, Sigma_Disk high) is largely gone —
  per-stream q_halo 1.32/0.97/1.12 and Sigma_Disk 5.24/6.57/4.51e8 now overlap and track the
  pooled Global (rejection prior + freed β + Chen spray ⇒ coherent joint fit). **MMD** still flags
  residual misspecification (mmd 2.84, p_plain 0.015, p_stratified 0.0; per-member Mahalanobis pct
  M68 100 / Pal5 99.7 / NGC3201 81.5) — parameters now agree but M68/Pal5 summary features remain
  atypical vs the sim reference; much milder than before. Runs: train
  `outputs/stream_agama/stream_fusion_model5_spray_huang/2026-07-08_12-12-40`, sim-eval
  `…/2026-07-08_14-33-47`, real-eval `…/2026-07-08_14-42-49`. Next (not run): local level via
  `ancestral_sample` with `composition.global_run_dir` → the real-eval run.
- Session 2026-07-08 (rotation-curve-only PPC + next-dataset plan): added
  `scripts/ppc_rotation_curve.py` — a posterior-predictive check on the **model rotation curve
  alone**. It REUSES the already-sampled posterior draws from an `evaluate_real composition=global`
  run (`posterior.npz` = pooled global, `single_stream_posterior.npz` = per-member — the same draws
  behind `real_global_vs_streams_corner.png`; NO network re-sampling), subsamples N per group
  (default 100 → 100×4=400 curves for Combined+Pal5+NGC3201+M68), builds each draw's host potential
  and evaluates `v_circ(r)` on the observed grid, and overlays median + 68/95% bands on the observed
  Zhou∪Huang curve. **Standalone** (agama+numpy+matplotlib only; the rotation-curve constants +
  `_host_potential`/`_vcirc` are inlined verbatim from `stream_agama`/`stream_common`) — importing
  `hydrabflow.simulators` triggers auto-discovery that imports BayesFlow→JAX and stalled ~11 min on
  a busy GPU, so the PPC deliberately avoids the package import; also runs on the GPU-less cluster.
  On model_5 (spray_huang real-eval run): all four groups reproduce the curve to **<1% median frac
  dev** (Combined 0.8 / Pal5 0.7 / NGC3201 0.6 / M68 0.9%); 95% band covers 68–86% of obs points
  (bands slightly narrower than Zhou's point-to-point scatter). Parameter-level per-stream tension
  from the corner plot does NOT show up as a rotation-curve mismatch — vcirc pins enclosed mass, on
  which all three streams agree. Artifact: `ppc_rotation_curve.png` in the run dir.
- **PLANNED next dataset (not yet run)**: a **larger** training set that combines all three prior/
  model improvements at once — (1) the banded **Zhou∪Huang rotation-curve rejection prior**
  (`vcirc_rejection.bands`, `obs_r_grid: extended`), (2) the **restricted N-body** stream simulator
  (`stream_agama_rnbody`, self-consistent progenitor + refit Multipole, t_end=4 Gyr for NGC3201/M68),
  and (3) the **freed halo β** (`beta_..._halo` uniform[2,4] instead of fixed 3.0) so the model can
  match Huang's declining outer curve. Requires a new `conf/simulator/*.yaml` (`name:
  stream_agama_rnbody`) carrying the spray_huang extras (chen IC is spray-only, so rnbody keeps its
  own IC; the rejection prior + extended grid + free β are class-level and already config-driven on
  the base `stream_agama`, which `stream_agama_rnbody` subclasses). Dataset creation to run on the
  GPU-less cluster (rnbody is CPU/joblib, ~10-25 s/row × rejection screening) then scp'd back.
- Session 2026-07-09 (missing-vlos handling — implemented + ablation): literature check first
  (user request): Wang et al. PLOS CompBiol 2024 (missing data in BayesFlow NPE) find constant-fill
  + binary indicator ("E2") the most robust encoding — our mean-fill was nonstandard (fill = a
  statistic of the observed subset); Le Morvan et al. 2020 prove constant+mask asymptotically
  Bayes-optimal (so this is an efficiency fix, not correctness); NAIM 2024 = precedent for the
  learned missing-embedding design below. Implemented per the 2026-07-08 TODO design:
  `mask_vlos` gained `params.vlos_impute: mean|zero` (declared in `conf/augmentation/
  stream_global.yaml` — Hydra struct mode rejects undeclared overrides); new `impute_vlos` step in
  `stream_real_global` re-applies the fill on real data from its `vlos_mask` (REQUIRED for
  zero-fill arms: the real npz ships pre-imputed with the per-stream mean — verified exactly equal;
  mean mode is a value-preserving no-op, so existing real evals unchanged); new
  `networks/masked_set_transformer.py` (`masked_set_transformer`) = zero vlos value/sigma channels
  where mask=0 → Dense(embed_dim) → + learned missing-vlos embedding (BERT mask token) → stock
  bf SetTransformer, attention_mask forwarded — channel zeroing inside the net makes sim/real
  consistent regardless of upstream fill. Configs `stream_fusion_model5_maskedvlos` (only the
  stream backbone differs from model_5; channels value=[5], sigma=[11], mask=13 of the 15-ch
  stream_global layout). 12 tests in `tests/test_masked_vlos.py` (incl. unmeasured-vlos invariance
  + serialization round-trip). Ablation harness
  `scripts/training_eval_missing_vlos_ablation.sh` (autocvd GPU pick), smoke-verified end-to-end;
  full runs launched on the rnbody+Huang 60k set (baseline mean-impute =
  `outputs/stream_agama_rnbody/stream_fusion_model5_rnbody_huang`, finished 2026-07-09: base RMSE
  0.540/calib 0.020, comp RMSE 0.511/calib 0.041, real MMD 2.65 p_strat 0.015) vs
  `outputs/missing_vlosexperiments/{zerofill,maskedvlos}_model5`. **Results** (see
  `outputs/missing_vlosexperiments/README.md`): sim accuracy indistinguishable (base RMSE
  0.540/0.536/0.534 for mean/zero/masked); maskedvlos best base calibration (0.015 vs 0.019/0.020)
  and the only run passing the overfit check (1.08x vs 1.13-1.14x). Real-data **MMD unchanged**
  (2.65/2.67/2.69, p_strat 0.010-0.015, all members ≳98th pct) ⇒ the residual misspecification is
  NOT the vlos imputation. Real posteriors shift ~1-2σ between arms (maskedvlos: q_halo 1.39±0.02
  tightest+highest, gamma 1.03, Sigma_Disk 7.8e8) — within the known cross-model real-data
  scatter; one seed per arm. **Recommendation: use `model=stream_fusion_model5_maskedvlos` +
  `augmentation.params.vlos_impute=zero` going forward.** Same day, user-directed: a joint
  Gaussian-KDE prior score for the vcirc-truncated prior (swap for `prior_score_from_spec` in
  compositional sampling) was implemented, tested and then **removed — "the kde approximation of
  the prior does not work well, do not use it"** (user's own check); the analytic spec score
  stays, per the 2026-07-07 rationale.
- **TODO — summary-statistics observables (designed 2026-07-10, user-deferred, not implemented)**:
  replace/augment the star-level stream input with hand-crafted per-stream summary statistics, to
  make real-data inference robust to the misspecification-driven cross-model divergence (models
  agree on sim, diverge on real ⇒ learned summaries extrapolate arbitrarily off-manifold; cf. the
  four-generation comparison of 2026-07-10 where spray_huang 30k — NOT rnbody_huang 60k — is the
  most real-data-coherent model: per-stream Sigma_Disk spread 17% vs 39-53% elsewhere).
  **Design**: new augmentation `stream_summary_statistics` inserted after `log10_vcirc` (before
  the concatenations) reading the 6-channel observable + `vlos_mask` + `attention_mask` + `j`,
  writing a new batch key `sim_summary` `(n, n_stats)` — per-stream fixed RA-bin (from
  `observational_window`) weighted stats: count fraction, dec/parallax/pm tracks + dec dispersion
  (~8 bins), vlos mean/dispersion from MEASURED stars only (~4 coarse bins — natively solves
  missing-vlos, no imputation), plus scalars (attended fraction, extent, arm asymmetry);
  optionally linear-density power-spectrum modes (Bovy+2017). Two arms: summaries-only (adapter
  `summary_variables=[sim_summary, vcirc_kms]`, run vs the maskedvlos/baseline ablation) and
  hybrid (3-key fusion `[sim_data_projected, sim_summary, vcirc_kms]`). Precedents: Albatross
  (Alvey+23), Hermans+21; robust-SBI motivation Ward+22/Huang+23. Payoff: per-statistic sim-vs-real
  z-scores localize WHICH physics is off.
  **Seams verified 2026-07-10** (all file:line refs checked): per-batch augmentations run BEFORE
  the adapter (bf `offline_dataset.py:134`) so `sim_summary` needn't exist in stored npz;
  `select_adapter_keys`/`condition_keys`/`_prepare_real_members` all tolerate the later-created
  key; bf `Standardize` handles grouped dicts per-leaf; fusion `mask_backbone` optional and
  2-D backbone inputs fine (outputs all rank-2). **Two real blockers**: (1) `drop_nan` preprocessing
  uses `keys: ${adapter.summary_variables}` with a bare `data[key]` → KeyError before augmentation;
  pin its keys to npz-resident observables or make the step skip missing keys. (2) no registered
  2-D summary backbone — add `@register_summary_network("mlp")` = `bf.networks.MLP(widths=[mlp_width]*mlp_depth)`
  + `Dense(summary_dim)` (schema already has mlp_depth/mlp_width). New configs: adapter presets
  with explicit `summary_variables` (survives `fill_adapter_from_simulator`), augmentation presets
  `stream_global_sumstats`/`stream_real_global_sumstats`(+hybrid variants keeping concatenations),
  model yamls `stream_fusion_model5_sumstats`/`_hybrid` (mask_backbone null for summaries-only).
- **Native missing-vlos handling in the SetTransformer (designed 2026-07-08, IMPLEMENTED
  2026-07-09 — see that session entry; original design notes kept below)**: replace the
  `mask_vlos` mean imputation with a missingness-aware summary network.
  Current state: `mask_vlos` (`augmentation/streams.py:445`) overwrites unmeasured stars' vlos with
  the per-stream mean of the kept values and their sigma with the sample std, then
  `concatenate_vlos_mask` (`streams.py:629`) appends the binary indicator channel — so the
  SetTransformer sees fabricated, artificially coherent values and must learn to ignore them via
  the indicator alone. The bf attention mask can't help: `summary_attention_mask` (routed by the
  adapter rename `adapter.py:191-193` → `MaskedFusionNetwork` → only the `sim_data_projected`
  backbone, `fusion.py:80-83`) masks whole set elements (padding), never a single feature channel.
  **Plan (recommended)**: new self-registering module
  `src/hydrabflow/networks/masked_set_transformer.py` — a `SummaryNetwork` subclass registered as
  `masked_set_transformer` that (a) reads the vlos mask from its feature channel, (b) zeros the
  vlos value + sigma channels where mask=0, (c) projects features through a `Dense(embed_dim)`
  input layer and **adds a learned "missing-vlos" embedding vector** to stars without vlos
  (BERT-style mask token), then (d) runs a stock `bf.networks.SetTransformer`, forwarding
  `attention_mask` unchanged. Channel indices for the current `stream_global` layout (6 obs +
  6 sigma + magnitude + vlos_mask + stream index = 15): `value_channels: [5]`,
  `sigma_channels: [11]`, `mask_channel: 13` (config `params`). Also: `mask_vlos` gains
  `params.impute: mean|zero` (default mean for back-compat; use zero with the new net so true
  simulated vlos never leaks), and `conf/model/summary_network/stream_fusion*.yaml` swaps the
  `sim_data_projected` backbone type. Adapter/preprocessing/real-data path untouched — real Gaia
  npz already carries its own `vlos_mask` (`evaluate_real.py:122`), so sim and real missingness go
  through the identical forward pass. Rejected alternative: two-set fusion (astrometry set +
  vlos-only set, per-backbone attention masks) — bf's approximator forwards exactly ONE
  `summary_attention_mask`, so it needs mask-stacking hacks or a custom approximator and breaks the
  star-level pm↔vlos joint unless coords are duplicated. Caveats: architecture change ⇒ existing
  checkpoints (model_5 …) can't be reused, retrain required; if `per_stream_standardize` is ever
  added to the chain, fit its vlos stats on measured values only (today they include imputed ones).
- Session 2026-07-10 (KDE compositional prior score — bug found + fixed + rerun): the 2026-07-09
  KDE prior score (removed as "does not work well") was diagnosed as a **space bug, not a KDE
  limitation**. `bayesflow…helpers.compositional.build_prior_score_fn` calls `compute_prior_score`
  on parameters in the network's native space (un-standardized + adapter-inverse, which requires
  **zero log_det_jac** — so the log10 reparam lives in *preprocessing*, not the adapter): i.e.
  `log10(x)` for the `log10_transform` keys, physical otherwise; bayesflow re-applies the
  standardization Jacobian itself afterward, and since our callable names a `time` arg it must
  apply the `(1-t)` decay itself. The old KDE was fit on the **raw physical-unit** npz arrays but
  evaluated on log10-space θ ⇒ wrong gradients, worst for large-magnitude log10 keys — base
  (no prior score) stayed pristine while compositional blew up (RMSE 0.51→3.49, calib 0.04→0.40;
  rho 7.38, Sigma 4.74). **Fix**: `pipeline/compositional.py::prior_score_from_kde` fits the KDE in
  the SAME network space (log10 on `log10_keys`), so `grad log p_KDE` is directly the score — no
  +ln10 term (that corrects the analytic *closed-form* density, not a density fit in the
  transformed space). Diagonal-bandwidth Gaussian KDE (per-dim Scott factor²×var),
  softmax-weighted closed-form gradient in `keras.ops`, applies its own `(1-t)`; selector
  `build_prior_score` reads `eval.prior_score` (spec|kde) + `eval.prior_kde_{samples,max_points,
  bandwidth}` (added to `EvalConfig`), wired into evaluate.py + evaluate_real.py. Analytic spec
  stays the default. Gradient verified vs finite differences (~4e-4). **Rerun** (same model, GPU
  via autocvd) `outputs/stream_agama_rnbody/stream_fusion_model5_rnbody_huang/kdeprior_fixed/
  eval_sim_333`: compositional RMSE **0.520** / calib **0.027** (base 0.540/0.020; pooling improves
  q/a/Sigma/r_Disk; z_Disk the lone "poor" RMSE, tightest normal prior — same as the analytic
  path). Real-data eval with `prior_score=kde` (diagonal) then run:
  `.../kdeprior_fixed/eval_real` — global q_halo 1.26 [1.23,1.29], beta_halo 2.62 [2.42,2.92]
  (freed [2,4], data pulls it toward the declining outer curve), Sigma_Disk 7.6e8, gamma 1.23;
  MMD 2.653 / p_strat 0.015 — **identical** to the analytic-spec baseline real eval (MMD is on
  the summary space, independent of the prior score → clean sanity check the swap touched only
  the score).
- Session 2026-07-10 (KDE prior score: linear-space guard + numerical-stability fix + jax
  full-cov comparison): three follow-ups to the KDE fix, all before committing.
  - **Numerical stability**: adding a linear-space (no `log10_keys`) regression test exposed
    float32 catastrophic cancellation in the expanded Mahalanobis form `q - 2*cross + r` (large
    on GPU, ~0.1 gradient error, GPU≠CPU) whenever a dimension's magnitude ≫ its bandwidth (the
    log10(rho)~7 case). Fixed in `prior_score_from_kde` by **centering** the data by its per-dim
    mean before the kernel (`u = theta - mu`, `Xc = X - mu`); the gradient `theta - weighted_x`
    is invariant under the shift, so it's provably unchanged but GPU-precise. Tests now pass at
    tight tol on GPU. Two new tests: `test_prior_score_from_kde_linear_space` (finite-diff, all
    physical params incl. a large-magnitude one) + the existing log10 one.
  - **Second KDE implementation** (user-requested, kept separate): `prior_score_from_kde_jax` =
    `jax.scipy.stats.gaussian_kde` (full covariance, Scott) + `jax.grad` (jax imported lazily,
    only on this path). Selector knob `eval.prior_kde_impl` (diagonal|jax), **default diagonal**.
    Same log10-space contract + `(1-t)` decay. Tests: jax-vs-finite-diff correctness, and
    `test_kde_jax_and_diagonal_agree_when_uncorrelated` (the two coincide only when the training
    draws are uncorrelated).
  - **Comparison — jax full-cov is UNUSABLE here** (important finding): on the same model,
    `kdeprior_jax/eval_sim_333` compositional RMSE **3.49** / calib 0.40 (≈ the old space-bug
    numbers) vs diagonal 0.52/0.03; real-Gaia posterior collapses/rails (gamma→-2.6, beta→4.7,
    q→0.17, all outside their priors). Diagnosed (not a wrapper bug): the log10-space parameter
    covariance has tightly-constrained directions (eigenvalues 0.004-0.006, condition number
    ~280, from the rejection prior pinning r_Disk/z_Disk/a); the full-covariance bandwidth whitens
    by it, so a 1σ move along a low-variance direction is many bandwidth-units away and the score
    explodes (max |score| 272 vs 15 for diagonal at mean+1σ), dominating the compositional term
    `(1-n)(1-t) grad log p` and collapsing the posterior. **The diagonal closed-form is the
    correct estimator for this rejection-truncated, near-degenerate prior**; jax full-cov is kept
    only as a documented, selectable alternative. Runs preserved side by side: `kdeprior_fixed/`
    (diagonal) vs `kdeprior_jax/`.
- Session 2026-07-10 (summary-statistics observables — IMPLEMENTED + run; was the long-standing
  TODO): hand-crafted per-stream summary statistics in a data-driven stream frame, to test whether
  physically-motivated summaries stabilise/change the real-data posterior vs the learned particle
  embedding. **New augmentation** `stream_summary_statistics` (`augmentation/stream_summary.py`):
  fits a great-circle frame per stream from the REAL Gaia members (pole = smallest-eigval eigvec of
  Σ n nᵀ), projects positions + proper motions into (φ1,φ2,μ_φ1,μ_φ2) in JAX per batch (astropy/gala
  too slow per batch; gala NOT needed/installed), and writes `sim_summary` (n,91) = per-φ1-bin
  **median+std** of {φ2,parallax,μ_φ1,μ_φ2} (10 track bins) + v_los from MEASURED stars only (3 bins,
  native missing-vlos) + scalars (measured frac, attended frac, φ1 extent, arm asymmetry) + stream
  index `j` (user-requested: the summary MLP must be stream-aware). Bin counts derived from the real
  member/vlos counts (Pal5 129/69, NGC3201 195/37, M68 297/29 → K_track=10, K_vlos=3), NOT guessed;
  validated in a prior-predictive check (`scripts/ppc_summary_statistics.py`, standalone numpy) that
  the sim tracks bracket the real Gaia data. **Infra**: `adapter_keys()` now includes `adapter.drop`
  (so summaries-only retains the raw star cloud as the augmentation input but drops it before the
  net); new `mlp` summary backbone (`networks/factory.py`). **Always-on checkpointing** (user
  request, also fixes the NaN below): `build_workflow(cfg, run_dir)` enables BayesFlow's built-in
  best-val-loss `ModelCheckpoint` (`approximator_best.weights.h5`, `training.save_best_weights`
  default True) + train.py adds `TerminateOnNaN` and restores best weights before saving — a late
  divergence can no longer destroy a run. Configs: `stream_global_sumstats`/`stream_real_global_sumstats`
  augmentation, `stream_sumstats_{hybrid,only}` adapter, `stream_fusion_model5_sumstats_{hybrid,only}`
  model, `stream_global_log10_sumstats` preprocessing (pins `drop_nan.keys` to npz observables since
  `sim_summary` is batch-only). Runner `scripts/training_eval_summary_stats.sh` (2 arms, 2 GPUs via
  autocvd). Tests `tests/test_stream_summary.py` (8) + checkpoint-wiring test. **Results**
  (`outputs/summary_stats_experiments/README.md`; A=particles baseline, B=hybrid, C=summaries-only):
  sim base RMSE 0.540/0.537/0.611, comp 0.511/0.507/0.570 — **hybrid ≈ baseline** (summaries add
  nothing on in-distribution sim), summaries-only only ~13% worse (a tiny MLP on 91 numbers recovers
  most of the particle SetTransformer's info). **Real-data key finding: the input representation
  drives halo flattening** — particles (A,B) rail to prolate q_halo≈1.26–1.33, summaries-only gives
  oblate **q_halo≈0.76 [0.69,0.82]**; disk params agree; hybrid tracks the particle q (raw particles
  dominate halo-shape when both present). MMD (each in its own summary space): summaries-only makes
  Pal5 look typical (69th pct vs 100th for particles), NGC3201 stays 100th everywhere. **NaN gotcha
  (documented)**: at 1000 epochs the summaries-only diffusion net deterministically NaN'd at epoch
  761 (heavy-tailed per-bin std features → large standardized value → inf loss → NaN grad; bf's
  default clipnorm=1.5 can't catch an inf-loss NaN); converged by ~epoch 300, so C was run at 300
  epochs. The always-on checkpointing above is the durable fix. Runs:
  `outputs/summary_stats_experiments/{hybrid,sumonly}_model5/`. **TODO not yet done**: the hybrid
  3-key variant keeps particles in ICRS (per user) — a stream-frame-particles arm and the
  summaries-only-without-rotation-curve ablation are natural follow-ups; sumonly used the
  pre-checkpointing train.py (saved epoch-300 weights, mild 1.20x overfit) so a re-run with
  best-weights restore would be marginally cleaner.
- Session 2026-07-10 (Ibata 2023 ancillary observables + full potential model — IMPLEMENTED, dataset
  gen deferred to user): per `new_constrains.md`, added three potential-derived observables (HI
  terminal velocity `v_term(l)`, local surface density `Sigma(1.1 kpc)`, vertical stellar-density
  profile `rho(z)`) computed as pure functions of each row's AGAMA potential (no stream sim). All
  config-driven on the base `stream_agama` class (legacy configs/tests unchanged; `pot_cfg=None`
  reproduces the old potential bit-for-bit, so `stream_agama_rnbody` etc. are untouched).
  - **Full Ibata potential** (user chose "full", not minimal): `_host_potential(agama, p, pot_cfg)`
    now assembles fixed bulge (already a separate compact Spheroid — the brief's bulge/halo split
    was already satisfied) + fixed McMillan HI & H2 gas disks (`GAS_HI/H2_PARAMS`) + **halo
    truncated at r_t=1000 kpc** (`params.halo_r_t_kpc`, was `outerCutoffRadius=inf`) + free thin +
    **free thick stellar disk**. Thick coupling `zd_thick>zd_thin` enforced by reparametrization:
    thick scale height = `z_Disk + dz_thick_Disk` (dz>0), so NO extra rejection. New free globals:
    `r_thick_Disk` U[1,10], `dz_thick_Disk` U[0.05,4.5], `Sigma_thick_Disk` U[1e7,1e9]. Stellar
    disks switched to **exponential** vertical profile (`disk_vertical: exponential`, negative agama
    scaleHeight) per McMillan/Ibata; gas stay sech^2, bulge unchanged. **M200 hard-bound rejection
    was implemented then REMOVED per user** ("too slow" — the per-draw enclosed-mass root-find).
  - Helpers in `stream_common.py`: `terminal_velocity`, `surface_density`, `vertical_density_profile`,
    `vcirc_from_potential`; constants `R0_KPC=8.178`, `G_KPC_KMS2_MSUN` (== agama.G, verified to
    ~1e-8). Grids `VTERM_L_DEG` (first-quadrant l=31..67 deg, 2 deg — matches the observed CSV) and
    `RHO_Z_KPC` (0.1..5, 20) are the single source of truth, NOT stored in the npz (only per-row
    VALUES are, like vcirc_kms). Sim stores `vterm_kms (n,n_l,1)`, `sigma_z (n,1)`, `rho_z (n,n_z,1)`
    (group-level in compositional, one per dataset, like vcirc). Enabled by
    `params.ancillary_observables: [vterm, sigma_z, rho_z]` (empty by default -> zero overhead).
  - **Physics validated** before wiring: flat-curve limits exact (`v_term(30 deg)=0.5 V`); McMillan
    (2017) cross-check `Sigma(1.1)=71.7` (~71 obs) confirms G + units + disk-height sign; tests in
    `tests/test_ancillary_observables.py` (10, all pass; 89 total green).
  - **Observational-error augmentations** (`augmentation/streams.py`, mirror `add_noise_to_vcirc`,
    each a no-op if its key is absent): `add_noise_to_vterm` (6.2 km/s), `add_noise_to_sigma_z`
    (6.0), `add_noise_to_rho_z` (rel_err*|rho|), `log10_rho_z`.
  - **Wired into fusion, selectable** (user request — as summary backbones OR as a condition):
    configs `simulator/stream_agama_ibata.yaml` (inherits `stream_agama_spray_huang`: Chen spray,
    extended Zhou u Huang grid + banded rejection, freed beta), `augmentation/stream_global_ibata`,
    `preprocessing/stream_global_log10_ibata` (+ thick keys to log10), `adapter/stream_ibata`
    (all 3 as summary backbones) + `adapter/stream_ibata_sigma_cond` (sigma_z as inference_condition,
    vectors stay backbones), `model/summary_network/stream_fusion_ibata{,_sigma_cond}` (v_term/rho_z
    = TimeSeriesTransformer, sigma_z = mlp), `model/stream_fusion_ibata{,_sigma_cond}`. Both variants
    compose + build (MaskedFusionNetwork). `simulate`/`simulate_multistream` CLIs verified end-to-end
    at tiny scale.
  - **Assets**: `assets/terminal_velocity.csv` added (first-quadrant HI v_term, McClure-Griffiths &
    Dickey 2016, l=31..67 deg, sigma=6.2; grid == VTERM_L_DEG). Rotation-curve CSVs NOT added
    (already hardcoded in `stream_common`). **rho(z) real data = TODO** (Ibata 2017b Fig 12f, digitize);
    `Sigma_z=71+/-6` hardcoded as the real datum; v_term real values live in the CSV.
  - **PPC**: `scripts/ppc_ancillary_observables.py` (standalone-ish, imports only `stream_common`):
    prior bands of v_term vs observed CSV, Sigma_z hist vs 71+/-6, rho(z) shape band; `--sim-multistream`
    also renders the **per-stream summary-statistic tracks** (reuses a refactored `render()` in
    `ppc_summary_statistics.py`). Smoke-tested (24 rows): observed v_term sits at the TOP edge of the
    prior band (prior median ~15-20% low — watch on the full run), Sigma_z median ~70 brackets 71.
  - **Deliverable for the user to run**: `scripts/create_ibata_dataset.sh` — a fast pilot batch +
    full PPC first, then the **10^5 flat spray training set + 333-group multistream** test set
    (CPU/joblib, resumable, n_workers=24). NOT yet run (user runs it). Training later uses GPU ->
    autocvd; the script prints the train command (`model=stream_fusion_ibata adapter=stream_ibata ...`).
- Session 2026-07-11 (Ibata norho + summary-statistics + sigma-condition model — trained/eval'd on
  the 10^5 dataset + concurrent Optuna): the Ibata 10^5 dataset was generated (by the user) to
  `data/data_jarvis/data_agama_ibata_hydrabflow/` (`training_data_100000.npz` + `test_multistream_333.npz`;
  all 3 ancillary observables present). Built the variant the user asked for: **exclude rho_z** (the
  last-commit norho line, since rho_z is the only ancillary observable with no real datum), feed the
  network the **binned stream-frame summary statistics** (`sim_summary`) instead of raw particles,
  route the **scalar `sigma_z` as an inference condition** (like `j`), keep the **vector observables
  `vcirc_kms`/`vterm_kms` as fusion `time_series_transformer` backbones**.
  - **New backbone** `feature_transformer` (`networks/factory.py`): reshapes a flat rank-2
    `(batch, F)` summary vector to `(batch, F, 1)` feature tokens and runs `bf.TimeSeriesTransformer`
    — lets a Transformer stand in for `mlp` on `sim_summary` (the "TST test"). Drop-in with `mlp`
    (both consume rank-2), so the Optuna study makes `sim_summary.type` a **categorical `[mlp,
    feature_transformer]`**.
  - **Real-data eval enabled** (what `train_ibata.sh` skipped): new `attach_observed_vterm` /
    `attach_observed_sigma_z` preprocessing steps (`preprocessing/streams.py`) + `OBS_VTERM_KMS`
    constant (`stream_common`; `sigma_z`=71 reuses `SIGMA_Z_OBS_MSUN_PC2`); excluding rho_z is what
    makes this feasible. Presets `stream_real_global_ibata_sumstats` (preproc + aug).
  - **Config quartet**: `adapter/stream_ibata_sumstats` (`summary_variables=[sim_summary, vcirc_kms,
    vterm_kms]`, `inference_conditions=[j, sigma_z]` — **j MUST be first**, `evaluate.py:125` derives
    the member count m from `inference_conditions[0]`; a group-level scalar first collapses m→1),
    `augmentation/stream_global_ibata_sumstats`, `preprocessing/stream_global_log10_ibata_sumstats`
    (drop_nan.keys pinned to npz-resident obs, not `${adapter.summary_variables}`, since sim_summary
    is batch-only), `model[/summary_network]/stream_fusion_ibata_sumstats` (`mask_backbone: null`).
  - **Tuning** `conf/tuning/stream_ibata_sumstats.yaml` (study `stream_ibata_sumstats_study`,
    n_epochs=100, n_trials=50/process): categorical sim_summary.type + sim_summary/vcirc/vterm
    backbone dims + fusion head + diffusion subnet. Objectives = RMSE + calibration (Pareto).
    Concurrency-safe JournalStorage `.log`; launched **one process per free GPU** (0,3,5), all
    sharing the one study — plus the test-GPU freed a 4th slot. Runners `scripts/train_ibata_sumstats.sh`
    (test: train 300ep → eval sim → eval real) + `scripts/tune_ibata_sumstats.sh` (one GPU/process).
  - **OOM resilience** (user request): `utils/oom.py::run_with_oom_backoff` catches JAX
    `RESOURCE_EXHAUSTED`, halves the batch size and retries (to min 16). Wired into `train.py`
    (fit_offline) and `tune.py` (fit + posterior sample). The tuning OOM was in the **sampling** step
    (`inference.batch_size*num_samples`=256*1000=256k rows through the diffusion integrator with the
    search space's large nets); the tune runner also pins `inference.batch_size=32 num_samples=500`
    (16k rows) as the primary fix, backoff as the safety net.
  - **Results (TST test, 300 epochs, sim_summary=feature_transformer)**: sim **base RMSE 0.733 /
    calib 0.022**, **compositional RMSE 0.669 / calib 0.053** (pooling improves accuracy; well
    calibrated). Real (Gaia) MMD 3.09, per-member percentile Pal5 94.6 / NGC3201 90.1 / **M68 100**
    (M68 most atypical — consistent with every prior generation). Run:
    `outputs/ibata_sumstats/tst_test/{train,eval_sim_333,eval_real}`. Mild overfit flagged
    (val_loss 1.11x best) but best-weights restore handles it. Tests: `feature_transformer` forward,
    `attach_observed_*` shapes, 5 OOM-backoff tests — full suite green (97).
- Session 2026-07-14 (halo prior reparameterized by virial mass + concentration — McMillan 2017):
  optional halo parameterization by (M200, c_v') instead of (densityNorm rho, scaleRadius a), from
  McMillan (2017, MNRAS 465, 76 = the paper the gas disks already come from). Config-driven on the
  base `stream_agama` class via `params.halo_parameterization: rho_a|m200_c` (default `rho_a`
  reproduces the old halo bit-for-bit; legacy configs/tests untouched). **Physics/validation**: the
  halo is McMillan's exact profile (`rho = rho0/[x^g (1+x)^(3-g)]`, `x=r/r_h`), so the mapping
  reproduces his Table 3 to <1% — `(M200=1.30e12, c_v'=15.4, gamma=1) -> r_h=19.6 kpc,
  rho0=8.54e6`. Both new globals are **sampled in log space with the stock uniform/normal prior
  types**, so NO new prior type and the analytic compositional prior score is unchanged:
  `log10_M200_TwoPowerTriaxial_halo ~ U[11.699,12.398]` (= M200 log-U[0.5,2.5]e12, Delta=200 x
  rho_crit, H0=70.4) and `ln_cvprime_TwoPowerTriaxial_halo ~ N(2.56,0.272)` (McMillan eq. 8 /
  Boylan-Kolchin 2010 c-M, at the Delta_c~94 x rho_crit "virial" overdensity). Per-row conversion
  in `stream_agama._halo_params_m200c(agama,p,cfg)`: `r200` from M200; `c200 =
  convert_concentration(c_v', 94->200)` (`stream_common`, bisection on invariant Delta c^3/m(c),
  NFW m — applied for all gamma per McMillan, exact only at gamma=1); `r_h = r200/[c200 (2-gamma)]`
  (r_-2=(2-gamma)r_h); `densityNorm` from the **unit-norm `enclosedMass(r200)` solve** (exact incl.
  the 1000 kpc cutoff taper + flattening q). **gamma capped [-2,1.5]** (was [-2,2]; user decision)
  because (2-gamma)->0 sends r_h->inf as gamma->2 (~20% of draws had r_h>30 kpc otherwise). Cosmology
  threaded through `_pot_cfg` (`halo_H0_kms_mpc`/`halo_Delta_mass`/`halo_Delta_c`) + `_DEFAULT_POT_CFG`.
  Config `conf/simulator/stream_agama_ibata_onedisk_beta3_m200c.yaml` (inherits onedisk_beta3, beta
  identity 3.0; rho/a set to identity = unused constants, so NOT inferred; adapter derives the
  inferred set -> infers log10_M200/ln_cvprime in place of rho/a, existing train/eval stack unchanged).
  Tests `tests/test_m200c_halo.py` (concentration roundtrip, McMillan Table 3, enclosedMass==M200
  across gamma/mass, host-potential dispatch matches equivalent rho_a); full suite green (112).
  Verified end-to-end: tiny `simulate` produces log10_M200 in-range, ln_cvprime ~ N(2.56,0.272),
  rho/a stored as identity constants, finite physical vcirc. **Dataset gen deferred to user** (CPU/
  joblib; `model=stream_fusion_ibata adapter=stream_ibata` etc. work unchanged). **Convention notes**:
  M200 at 200 rho_crit but c_v' at ~94 rho_crit is McMillan's own split; the 94->200 conversion +
  (2-gamma) factor reconcile them. This is a genuine prior reparameterization (new training set); the
  rejection prior now carves the (M200,c) plane (still a hard indicator, score valid).
- Session 2026-07-14 (m200_c: save AGAMA-passed rho/a + dataset/train/tune scripts): follow-up to
  the above so the user can scp-and-run. (1) **Derived halo diagnostics saved**: `_simulate_one`
  now also returns the `(densityNorm, scaleRadius)` AGAMA actually received per row (only when
  `halo_parameterization=m200_c`; `_resolve_pot_cfg` guard, computed via `_halo_params_m200c`), and
  `simulate` stores them as `rho_TwoPowerTriaxial_halo_derived` / `a_TwoPowerTriaxial_halo_derived`
  `(n,1)`; `sample_compositional` reshapes them per-group `(n,1)` like `vcirc_kms`. These are
  diagnostics ONLY — the adapter drops them (verified in the smoke log: "Dropping dataset keys the
  adapter does not use: [... rho/a_..._derived ...]"), inference stays on log10_M200/ln_cvprime; the
  identity rho/a stay fixed constants alongside. 2 new tests in `test_m200c_halo.py` (derived keys
  match `_halo_params_m200c` per row; absent for rho_a); suite green (114). (2) **Corner script**:
  `corner_parameters.py` NON_PARAM_KEYS now also excludes the ancillary observables (vterm_kms,
  sigma_z, rho_z) so they aren't mistaken for scalar params. (3) **Three scripts** (per user, the
  training/tuning use the SUMMARY-STATISTICS stack, NOT the raw-particle SetTransformer):
  `scripts/create_ibata_m200c_dataset.sh` (standalone; pilot+full 10^5 flat + 333 multistream, the
  vcirc rejection cut applied per draw, PPC + `prior_after_cut_corner.png` over all inferred globals
  + derived rho/a); `scripts/train_ibata_m200c.sh` + `scripts/tune_ibata_m200c.sh` (thin wrappers
  delegating to `train_ibata_onedisk_grid.sh` / `tune_ibata_onedisk_grid.sh` — the gridded
  sim_summary TimeSeriesTransformer + vcirc + vterm backbones, sigma_z+j conditions, raw particles
  dropped; NOT overriding STUDY since tune.py reads study_name from the yaml and DATA_DIR already
  isolates the study log). Dataset dir convention mirrors the onedisk pair: gen under
  `data_jarvis/data_agama_ibata_onedisk_beta3_m200c_hydrabflow`, train/tune read
  `data/data_jarvis/...`. Smoke-verified end-to-end on CPU (64-row flat + 8-group multistream, 1
  epoch): train → eval_sim (base+compositional) → eval_real all produce full artifacts; derived
  keys land in both flat and multistream npz. **Full dataset gen + GPU train/tune deferred to user.**
- Session 2026-07-15 (rho_a Ibata onedisk_beta3 grid model on 3e5 streams — best-2 eval + PPC):
  evaluated the best 2 completed trials (of 21) of the LIVE Optuna study `stream_ibata_grid_300k_study`
  and ran both PPCs on the Gaia streams (GPU 2 via autocvd, tuning workers undisturbed; isolated
  output under `outputs/ibata_onedisk_grid/ppc_best2/trial_{2,15}/`). **Setup**: simulator
  `stream_agama_ibata_onedisk_beta3` with **`halo_parameterization=rho_a`** (the ORIGINAL densityNorm+
  scaleRadius halo prior, NOT the m200_c reparam; beta identity 3.0, single exponential stellar disk,
  gas disks on, halo r_t=1000 kpc), gridded summary-statistics fusion model
  `stream_fusion_ibata_grid` (sim_summary/vcirc_kms/vterm_kms as TimeSeriesTransformer backbones,
  sigma_z+j inference conditions, raw particles dropped, sigma_z standardization ON), trained on the
  **3e5-stream** single-disk training set (`data_agama_ibata_onedisk_beta3_hydrabflow`). **Sim eval**
  (333-group test): trial 2 base RMSE 0.474/calib 0.016, comp 0.449/0.042; trial 15 base 0.470/0.014,
  comp 0.452/0.043 — well calibrated, pooling improves accuracy. **Key result — the two best models
  agree closely and land on an OBLATE halo** (this prior parameterization + 3e5 streams): global
  q_halo **0.80 [0.73,0.88]** (trial 2) / **0.78 [0.70,0.87]** (trial 15), gamma_halo 1.78/1.75,
  a_halo ~22 kpc, rho_halo ~2.7e6, r_Disk ~3.2 kpc, z_Disk ~0.30 kpc, Sigma_Disk ~5e8. This q≈0.78–0.80
  is the **summary-statistics representation's signature** (cf. the raw-particle models that rail to
  prolate q≈1.3, and the earlier standalone summaries-only q≈0.76). **MMD** still flags real-data
  misspecification (mmd ~2.96, p_strat 0.0, all 3 members ~98–100th Mahalanobis pct) — parameters
  agree but Gaia summaries remain atypical vs the sim reference, as in every prior generation.
  **Rotation-curve PPC** (`ppc_rotation_curve.png`): both models reproduce the Zhou∪Huang curve to
  **<1% median |frac dev|** (Combined/Pal5/NGC3201/M68 0.5–1.0%); 95% band covers ~40–60% of obs
  points (bands slightly narrower than the observed scatter). **Ancillary PPC**
  (`ppc_ancillary_posterior.png`): Sigma_z medians ~70–81 Msun/pc^2 (obs 71±6; within 1sigma for
  Combined/M68, low coverage for NGC3201); **HI terminal velocity UNDER-predicted** — only ~5–32% of
  observed v_term points fall in the 95% posterior band (a genuine, consistent mild misspecification
  across both best models). Full artifacts (posterior_pairs, real_global_vs_streams_corner,
  mmd_hypothesis_test, both PPC figures + summary JSONs) under each trial's `eval_real/`.
- Session 2026-07-15 (MMD misspecification LOCALIZED to the stream channel — per-channel + per-
  statistic diagnostics): two new offline scripts answer WHERE the fused-summary MMD flag comes
  from, run on m200c best_trial36 (`outputs/ibata_onedisk_grid_m200c/tuning/best_trial36/`, fused
  MMD 2.86 p=0) and rho_a trial_2 — both give the SAME verdict, so it's a property of the
  data/simulator pair, not the halo parameterization.
  - `scripts/misspecification_per_channel.py`: runs the Schmitt+21 MMD test SEPARATELY on each raw
    projected observable channel (`sim_summary`, `vcirc_kms`, `vterm_kms`, `sigma_z`) — model-free,
    in data space, both sides through the exact eval pipelines (`_load_test_data`+`flatten_members`
    / `_prepare_real_members`+transform) + one augmentation draw, replaying the CLI fill_* steps
    (saved .hydra configs are pre-fill). **Result: the flag is entirely `sim_summary`** (p_strat
    0.046/0.032; Pal5/NGC3201/M68 at 97.6/99.4/100th Mahalanobis pct). Potential channels are clean
    when tested correctly at ONE ROW PER POTENTIAL (group-level): observed MW vcirc at 97th pct
    (expected — the rejection prior truncates around it), vterm 25th, sigma_z 7th, p 0.6–0.99.
    **Gotcha (bug fixed)**: group-level channel detection must use the RAW grouped test-set shapes
    (`flatten_members`' rule, ndim>=3 & shape[1]==m), NOT post-augmentation within-group variance —
    the noise augmentations run after flattening so the m copies differ; member-treating a group
    channel duplicates the observed row m× and artificially drives MMD p→0.
  - `scripts/sumstat_sim_vs_real.py`: per (stream, statistic, φ1-bin) robust z-scores of the real
    `sim_summary` cells vs the 996-stream sim reference (zmap + track overlays). **Dominant
    offender: `std_phi2` — the real streams are WIDER on-sky than any sim** (aggregate median |z|
    2.08, max 10.4; Pal5 central bins z +6..+8). **M68 is fluffier in EVERY dispersion** (φ2/pm/vlos
    std elevated in all bins, plus leading-edge track offsets med_phi2 +8.7 — member contamination
    or under-heated sims). **NGC3201**: medians excellent, but in the last φ1 bin the SIMS grow a
    hot dispersed edge population the real data lacks (std_mu_phi1 z −9.7, std_vlos −5.2).
  - Interpretation: the residual misspecification is in stream MORPHOLOGY (sims too cold/thin —
    missing heating from progenitor internal dispersion / GMC-bar-spiral perturbations — or real
    member-selection contamination), NOT in the potential-derived observables.
  - Also new (untested-at-scale helpers, committed): `mmd_{stream,potential}_vs_training.py`
    (training-set-sized references instead of the thin 333-group test set),
    `ppc_ancillary_posterior.py` + `ppc_rotation_curve_ibata.py` (the PPCs used in the best-2 eval
    above). `optuna_results.py` now points at the m200c tuning log — NOTE: it still loads
    `study_name='stream_ibata_grid_study'`; verify that matches the m200c study before trusting it.
    `tune_ibata_onedisk_grid.sh`: N_TRAIN default 100000→300000 + passes `tuning.study_name`
    explicitly. Artifacts: `misspecification_per_channel.{json,png}` +
    `sumstat_sim_vs_real_{zmap,tracks}.png` in each run's `eval_real/`.

- Session 2026-07-16 (rnbody m200_c dataset + "too cold streams" diagnosis -> freed-prior fix):
  goal = the m200c config/rejection/priors with the RESTRICTED N-BODY forward model, plus a
  per-bin summary-stat std PPC ("are the sim streams too cold?"). Five pieces:
  - **rnbody x Ibata/m200c wiring**: base `stream_agama.simulate` refactored around a `_row_jobs`
    hook (the forward-model seam — subclasses swap only the joblib worker, dispatch + output
    assembly shared); `_simulate_one_rnbody` now takes `pot_cfg`/`ancillary`, computes the
    potential-only outputs (vcirc, vterm/sigma_z/rho_z, derived m200_c rho/a) BEFORE the stream
    try-block so they survive failed orbits, returns the same 4-tuple as the spray worker; the
    duplicated rnbody `simulate` override was deleted. Test
    `test_rnbody_simulate_m200c_ancillary_and_derived_keys`; suite green.
  - **Configs** `stream_agama_rnbody_ibata_onedisk_beta3_m200c[.yaml]` (inherits the spray m200c
    file wholesale — same priors incl. Sigma_Disk<=3e9, banded Zhou u Huang rejection, extended
    grid, ancillary observables; `name: stream_agama_rnbody`; the inherited `spray_method: chen`
    is ignored by the rnbody worker; t_end=4 identity for all three streams) and, after the
    probes below, `..._m200c_wide` (Pal5 `m_progenitor ~ U[4.3e3, 3.44e4]`, M68
    `t_end ~ U[4, 10]` Gyr). Script `create_ibata_rnbody_m200c_dataset.sh` (SIM/DATA_DIR
    env-overridable, N_FULL=10000). Datasets in `data_jarvis/data_agama_rnbody_ibata_onedisk_
    beta3_m200c{,_wide}_hydrabflow/` (10k flat + 333 multistream each).
  - **Cold-stream std check, prior + posterior**: `ppc_summary_statistics.render` now also emits
    `*_std.png` (per-bin std tracks) + a printed table (real raw / robust 1.4826*MAD / vlos
    error-deconvolved / sim median / P(sim<real)); new
    `scripts/ppc_posterior_summary_statistics.py` = posterior twin (reuses an
    `evaluate_real composition=global` run's saved `posterior.npz`/`single_stream_posterior.npz`
    — native-space, log10 keys inverted via the run's preprocessing — pairs global draws with
    PRIOR-drawn locals (not inferred at the global level), re-simulates with the run's own
    simulator, same renderer; `--source pooled|per-stream`, `--noise`). **`--noise` mode** (both
    scripts): `augment_sim()` pushes the raw (N,S,P,6) sim through the TRAINING observation-model
    prefix (`observational_window -> observed_n_stars -> compact -> sample_magnitudes ->
    sample_obs_error -> apply_obs_error -> mask_vlos`; resources fall back to `assets/gaia` when
    `data/` lacks the tables) so the comparison vs real Gaia is apples-to-apples; posterior
    variant reads the TRAIN augmentation from `model_dir`'s config (the eval_real run's own node
    is the real-data preset).
  - **Diagnosis (key session finding)**: the raw check made everything look cold; noise
    convolution + robust stats dissolved most of it. Explained away: Pal5 pm (Gaia noise floor
    ~0.2 mas/yr dominates), Pal5 vlos (real 6.0 -> robust 2.5-3.0, outlier-inflated; sim ~5
    brackets), NGC3201 everything (rnbody even slightly hot in phi2; spray-era NGC3201 phi2
    coldness was already fixed by rnbody). GENUINE gaps: Pal5 phi2 (sim 0.13 vs robust 0.20 deg)
    and M68 all four quantities (phi2 0.57 vs 1.38, vlos 11 vs 18.6). Detrended-per-bin check
    ruled out great-circle-curvature inflation; sim M68 is CLOSER than real (7.3 vs ~10 kpc) so
    geometry can't explain it. **Probes** (24-32 groups each, noise-convolved): Pal5 fixed by
    m_progenitor x2 (present-day 4.3e3 is the wrong mass for tails shed earlier); M68 mass
    SATURATES (x8=4.6e5 Msun still 40% short in phi2, mu_phi2 unmoved) and a_progenitor x3 does
    nothing — **M68's lever is stripping age: t_end=8 Gyr brackets all four (P~0.54-0.58)**.
    Freed-prior validation (the _wide config): every stream/quantity bracketed (P 0.09-0.84,
    nothing pinned at 1.0). The freed locals are nuisance-marginalized at composition=global, so
    the training manifold now COVERS the real widths instead of extrapolating.
  - Cold-table snapshots (noise-convolved, robust targets): baseline rnbody m200c — Pal5 phi2
    0.127/P=.90, M68 phi2 0.566/P=.94, M68 vlos 11.1/P=.89; wide priors — Pal5 phi2 0.196/P=.66,
    M68 phi2 0.79/P=.72, M68 vlos 14.0/P=.70. Probe npz/pngs in the session scratchpad only;
    dataset-level figures under each dataset's `ppc/` dir.

- Session 2026-07-28 (v2: progenitor-input correction, occupancy-aware summaries, freed halo/bulge/
  t_end, rejection prior dropped): a literature + code audit of the "streams still look unrealistic"
  and "gamma sits on its prior boundary" symptoms found that **three of the four leading causes were
  input/estimator errors, not missing physics** — so the earlier `_wide` prior-freeing had been
  compensating for them. Nothing here is trained; dataset generation is the deliverable (training
  runs on the other cluster).
  - **Progenitor masses were PRESENT-DAY masses used as INITIAL masses.** `mass_sat` is the mass at
    t = -t_end (agama's example treats `initmass` the same way). All three values trace to Table 2 of
    Palau & Miralda-Escude (2023), which holds cluster mass fixed: Pal5 4.3e3 = bound REMNANT only
    (Ibata, Lewis & Martin 2017), NGC3201 6.47e4 = LUMINOUS mass (Sollima & Baumgardt 2017; their
    dynamical mass is 1.20e5), M68 5.7e4 = a +-50% Plummer dynamical fit (Lane et al. 2010). Against
    Baumgardt & Hilker 2018 present-day masses that is ~3x/2.2x/3.1x low, and ~9-16x low against
    literature initial masses. **Pal5's `a_progenitor` = 8.43 pc was also the wrong KIND of radius** —
    a King core radius fed in as a Plummer scale radius (r_h,m/1.305 gives ~21 pc), so its progenitor
    was ~2.5x too compact as well as far too light. Confirmed by the new bound-mass diagnostic: at
    4.3e3 the Pal5 progenitor DISSOLVES COMPLETELY in 4 Gyr (densest surviving clump ~11/400
    particles) though the real cluster still exists. Config `stream_agama_rnbody_ibata_m200c_prog`
    corrects and frees both (masses from B&H18 present-day up to M_Ini; radii bracketing both
    conventions). Freeing the radius doubles as the grounded proxy for BH-retention cluster inflation
    (Weatherford & Bonaca 2025: sigma 1.2->2.2 km/s, "similar in magnitude to heating by Galactic
    substructure" and explicitly a selection effect for DM inference).
  - **`m_bound_final` saved** (rnbody only): `_bound_mass` re-estimates the remnant centre from the
    particles (shrinking sphere + bound-set iteration) because our centre trajectory is the massless
    test-particle orbit and a GC well is only ~2 km/s deep for Pal5 — evaluated at the orbit centre
    the criterion declares everything unbound. Per-stream `(n,m,1)` in compositional, a diagnostic the
    adapter drops. **Measured survival is low: 41.7% of rows leave a remnant WITH the rotation-curve
    cut, 19.8% without** (96-row probe) — a physical, stream-based screen the rotation-curve cut never
    applied. Deliberately NOT turned into a rejection prior (that is what we set out to remove); the
    dataset script reports it so the choice is made on real numbers.
  - **Summary-statistic estimator fixed** (`augmentation/stream_summary.py`). The phi1 bin edges are
    equal-count quantiles of the REAL members, so real bins hold N/K stars by construction and
    simulated ones do not; combined with `jnp.nanstd`'s ddof=0 (0.72x at N=3, 0.57x at N=2, exactly 0
    at N=1) and zero-filling of empty bins, simulated dispersions were biased LOW — and for phi2 a
    fabricated 0 *is* the on-track value, so an empty bin passed for an on-track, zero-dispersion one.
    Now: ddof=1 or a MAD scale (`summary_scale`), NaN below `summary_min_count`, per-bin **occupancy
    channels** (`n_track`, `n_vlos`), out-of-range stars excluded instead of clipped into the end bins
    (plus an out-of-range-fraction scalar). Layouts: grid 12 -> **14** channels, flat 91 -> **105**.
    **NOTE the scope of this bug:** `ppc_summary_statistics.py` already used ddof=1 + min_count=3, so
    the published cold-stream TABLE was never biased by it; what was biased is what the **network
    trains on** and `sumstat_sim_vs_real.py` (the z-map that localized `std_phi2`, |z| up to 10.4).
  - **`masked_time_series_transformer`** (`networks/`): reads the occupancy channels, zeroes
    under-populated bins, masks them out of attention AND — the reason it must exist — pools only over
    valid bins. **BayesFlow's own nets cannot do this**: `SetTransformer.call` ends with an unmasked
    `pooling_by_attention` and `TimeSeriesTransformer.call` with an unmasked `self.pooling(inp)`, so
    masked positions leak in (measured: padded-plus-masked differs from the true subset by ~0.39/0.48
    against output scales 1.1/2.0, i.e. 20-35%). With the wrapper the same test gives 4.8e-7, so
    **padding is transparent and per-stream adaptive phi1 bin counts become possible** — K no longer
    has to be tuned down to the sparsest stream (M68's 29 vlos stars currently cap K_vlos=3 for all).
    Implemented via `TimeSeriesTransformer(return_sequences=True)` + our own masked pooling, so no
    BayesFlow internals are forked. Required mask shape is `(B,T,T)`/broadcastable; `(B,T)` raises.
  - **q(r) considered and REJECTED on the literature** (the user required grounding). Published radial
    variation in MW halo shape is in ORIENTATION and sets in at r >~ 30-150 kpc — outside these
    streams' 5-30 kpc reach (Shao+2021 twist radius 30-150; Vasiliev+2021's r_q is Sgr-driven at
    15-100 and called "tentative"); inside 30 kpc simulations find axis ratios almost independent of
    radius for a 10^12 Msun halo (Chua+2019 verbatim; Shao+2021 b/a~0.95, c/a~0.85 stable). The only
    in-range support is Vera-Ciro & Helmi 2013's ~10 kpc transition (a modelling device to reconcile
    Sgr with a disc-aligned inner halo) and Bovy+2016's explicit "hint" at +-0.14. Meanwhile the
    scatter BETWEEN constant-q inner-halo analyses (0.75 Ibata+24, 1.05 Bovy+16, 1.06 Palau+23, 1.20
    Woudenberg & Helmi 24, our own 0.78 vs 1.3) dwarfs any predicted gradient — that is systematics.
    Measured costs on the full potential (20k forces, the quantity scaling the 10-25 s/row rnbody
    cost): two superposed Spheroids ~1.0x but **ineffective** (q_eff only 1.016->1.082; superposition
    washes the contrast out), explicit q(r) via `Multipole` 1.29x and effective (q_eff 0.72->1.09,
    reproduces the analytic Spheroid to ~1e-6 when q0==q_inf), **triaxial 1.92x**.
  - **Freed instead: `alpha` (transition sharpness), `p` (axisRatioY) and a `tilt` angle** — all three
    were literal constants — plus a ~10% `rho_Bulge`. Grounded in Nibauer & Bonaca 2025 (axis ratios
    1:0.75:0.70, major axis tilted 18-20 deg at r~12 kpc, from GD-1), Woudenberg & Helmi 2024
    (p=1.013+-0.006, q=1.204 inside 20 kpc), Emami+2021/Auriga (tilt generic, 19+-20 deg). Each
    defaults to its old value, verified bit-identical (force diff 0.000e+00). **Why gamma rails:**
    under m200_c `r_h = r200/[c200 (2-gamma)]` hard-couples gamma to the scale radius (gamma 1.0 ->
    r_h 14.6 kpc, 1.5 -> 29.1, 1.75 -> 58.3, 1.9 -> 145.6), and at fixed (M200,c) raising gamma
    1.0->1.9 adds ~30 km/s of halo v_c at R=4 kpc while LOWERING it beyond 15 kpc — exactly what the
    under-predicted HI terminal velocity (R=4.2-7.5 kpc) and Huang's declining outer curve jointly
    demand, so gamma was the only knob that could do both. gamma=1 with alpha=2 reaches 183 km/s at
    4 kpc and still declines. Also: gamma~1.75 is unphysical (empirical MW values cluster at 0.4-1.0;
    Palau & Miralda-Escude 2023 fit OUR three streams with both slopes free and get inner slope
    0.06+-0.22 with a heavy (6-8)e10 disc — the opposite corner of the disc-halo degeneracy).
    **NOT done, deliberately:** tightening Sigma_Disk to Bland-Hawthorn & Gerhard 2016 as they did —
    it directly conflicts with this project's own finding that the 1.5e9 ceiling capped inner-disc
    mass below the observed v_term and that 3.0e9 was needed. Revisit only if gamma still rails.
  - **Rejection prior dropped** (`vcirc_rejection: null`, per user): the curve is already an observable
    (`vcirc_kms` + `mask_vcirc_radii` + `attach_observed_vcirc`), so the cut double-counted it and
    truncated the halo-shape prior; removing it also makes the analytic compositional prior score
    exactly correct rather than correct-only-inside-the-accepted-region. Prior sampling becomes free
    (was ~12.6 screens/accepted row); measured NaN rate 3.1% vs 6.2% (n=96, so consistent).
  - **t_end freed to U[2,10] Gyr for ALL THREE streams.** Structural reason to do all three:
    `local_parameter_names` reads the FIRST stream only (`stream_agama.py:667-669`), so in `_wide`
    M68's freed t_end varied per row but was silently NOT an inference variable. Tension to keep in
    view: Palau, Wang & Han 2025 put M68's stream age at 3.04 (+5.63/-0.29) Gyr, and a genuinely long
    t_end pushes the integration through the Gaia-Enceladus merger where a static halo is hard to
    defend — so if the corrected progenitors remove the need for long ages, prefer short.
  - **`contaminate_members` augmentation added but DISABLED** (`contamination_max_frac: 0.0`, per
    user: no contamination in the training set). Interlopers are modelled as broadened look-alikes of
    that row's own members, not uniform field stars, because real contaminants are exactly the objects
    that passed the stream-finder cuts (Ibata+2020 reject 1 of 5 spectroscopic Gjoll targets on RV and
    find a metallicity spread ~30x the cluster's intrinsic sigma; M68 has no published width at all).
    At 0.0 it is an exact no-op, so enabling it later is one override and needs no regeneration; it is
    a per-batch augmentation so it never touched the stored npz either way.
  - Configs: `simulator/stream_agama_rnbody_ibata_m200c_{prog,v2}.yaml`,
    `augmentation/stream_{global,real_global}_ibata_grid_v2.yaml`,
    `model[/summary_network]/stream_fusion_ibata_grid_masked.yaml`; script
    `scripts/create_ibata_rnbody_m200c_v2_dataset.sh` (pilot + survival/NaN report + PPCs, prints the
    train commands rather than running them). Suite 114 -> 132 tests, all green.
  - **Solar phase-space frame varied as a MARGINALIZED nuisance** (added same session, per user):
    `R0_Sun ~ N(8.178, 0.026)` kpc (GRAVITY 2019) and the Schonrich, Binney & Dehnen (2010) PECULIAR
    velocities `U_Sun ~ N(11.1,1.25)`, `V_Sun ~ N(12.24,2.05)`, `W_Sun ~ N(7.25,0.62)` km/s. This also
    **fixes the R0 inconsistency**: `sky_projection` used astropy's default Galactocentric frame
    (R0=8.122, v_sun=(12.9,245.6,7.78)) while the potential observables used `R0_KPC = 8.178`.
    `stream_agama._solar_frame` now builds ONE per-row frame used for the projection, the progenitor's
    observed-ICRS -> Galactocentric conversion, AND the R0-anchored ancillary observables (v_term
    tangent points, Sigma(1.1), rho(z)), so they cannot desynchronise. Because V is peculiar, the
    frame's azimuthal velocity is `v_circ(R0) + V_Sun` **in that row's own potential** — the Sun's
    motion is not independent of the mass model (sanity: ~245 km/s, matching astropy's default).
    Implementation notes: the worker returns its frame as a 6th tuple element so `simulate` reuses it
    rather than rebuilding it; `sky_projection(xv, frames)` takes the AGAMA path only when a solar
    prior is declared, so **existing configs keep the astropy path byte-for-byte**. The two paths
    differ by ~1e-4 deg / ~1e-4 mas/yr (different ICRS<->Galactic rotation matrices; distance and
    v_los agree to machine precision) — far below Gaia errors but real, hence the conservative
    default. Round-trip ICRS -> Galactocentric -> ICRS through a varied frame is exact to ~1e-14.
    **New `params.marginalize` seam** (`stream_agama._marginalized`): names listed there are drawn per
    row and fed to the forward model but excluded from `global_parameter_names` — and therefore from
    `prior_spec_global`, so the compositional prior score gains no term, which is correct for a
    dimension that is not inferred. The v2 config marginalizes all four; the global posterior stays
    11-dimensional. This is the mechanism the README roadmap asked for ("varied in simulation,
    excluded from inference_variables") without an explicit hand-maintained adapter list.
  - **NOT done / follow-ups**: (a) no bin-count sensitivity sweep
    was added to the PPC scripts (the occupancy report was added to `sumstat_sim_vs_real.py`), so the
    "is the verdict stable in K?" check is still manual; (c) with padding now transparent, raising
    K_phi1 / going per-stream adaptive is unexploited; (d) whether to add a remnant-survival screen is
    an open decision resting on the pilot numbers.

- Session 2026-07-29 (streams in a FIXED Cautun+2020 potential — 10-realization observation-space
  check): a controlled forward-model check with the Galaxy held fixed, so the only variation is the
  progenitors' present-day phase-space coordinates within their measurement errors. Nothing trained.
  - **`conf/simulator/stream_agama_rnbody_cautun_fixed.yaml`** (inherits `stream_agama_rnbody`):
    every global is `identity`, so `global_parameter_names == []` and `local_parameter_names ==
    [vr, r, mu_ra_cosdec, mu_dec]` (ra/dec stay identity, m_progenitor/a_progenitor/t_end pinned).
    `vcirc_rejection: null` (a fixed potential makes the cut all-or-nothing).
  - **Cautun mapping.** `agama/data/Cautun20.ini` and this project's Ibata potential are the SAME
    component family for every baryonic component, so those transfer exactly (thin 7.31e8/2.63/0.30,
    thick 1.01e8/3.80/0.90 via `dz_thick_Disk=0.60`, bulge amplitude 1.03e11 into
    `bulge_density_norm` — the rest of `BULGE_PARAMS` already matches Cautun's Spheroid — and
    Cautun's HI/H2 gas disks ARE `GAS_HI/H2_PARAMS` to <1%). `disk_vertical: isothermal` because
    the ini uses positive scaleHeight. The ONE component that does not transfer is the dark halo
    (Cautun's is adiabatically contracted, tabulated as a spherical `Multipole`), so it was FITTED:
    least squares on the fractional TOTAL v_circ residual over 2-60 kpc with the baryons fixed on
    both sides → **rho = 2.7196e7, a = 9.8011 kpc, gamma = 1.0188** (beta 3, q 1, alpha 1, r_t 1000).
    Median |frac dev| 1.28% / max 3.09%; M(<r) +3.9% at 10 kpc, +5.4% at 20, -3.4% at 50, **-15.1% at
    100 kpc** (Cautun's CGM Spheroid r_s=219 kpc — deliberately unfitted, no orbit here reaches it).
    Verified through the full simulator path (`_host_potential` reproduces the fit exactly).
  - **`scripts/plot_fixed_potential_samples.py`**: N-samples x 3-streams panel grid + an
    all-samples overlay (phi2 / parallax / mu_phi1 / mu_phi2 / v_los vs phi1) in each stream's
    data-driven great-circle frame, reusing `ppc_summary_statistics` (`fit_frame`/`project`/
    `augment_sim`). Defaults to the noise-convolved training observation model; `--no-noise` applies
    the chain's own `1/d` so channel 2 is a parallax on both sides (the real npz stores parallax,
    `sky_projection` emits distance — an easy trap).
  - **Results** (10 rows, seed 2026, 95 s; `data_local/cautun_fixed/`). Yardstick = the sample-to-
    sample scatter, which IS the measurement-error-only spread: Pal5 tracks are reproduced within
    ~1.3x that scatter; **NGC3201 is off in the mean** (phi2 +1.32 deg vs 0.39 scatter = 3.4x,
    mu_phi1 -2.92 vs 1.28 mas/yr = 2.3x); and **both NGC3201 and M68 come out far too LONG** —
    phi1 extent 107 vs 67 deg and 132 vs 102 deg against a realization scatter of only ~1.5 deg.
    Cautun+t_end=4 Gyr over-strips those two. Parallax panels are pure Gaia noise (as expected).
  - **Progenitor survival is the sharpest result: Pal 5 dissolves completely in 10/10 rows**
    (`m_bound_final = 0`) even at the CORRECTED B&H18 present-day mass 1.34e4 with a 21 pc Plummer
    radius, while the real cluster still exists. NGC3201 survives 6/10, M68 4/10 (2 NaN each). So the
    2026-07-28 dissolution finding is NOT an artifact of the old 4.3e3 remnant mass, and not
    potential-specific — at 1.34e4 Pal 5 still cannot survive 4 Gyr. That points at the mass
    (literature INITIAL 4.7-7e4, i.e. the upper edge of the freed `_prog` prior) rather than at the
    Galaxy, and is worth checking before the freed-prior dataset is taken as settled.
  - **A/B against the LEGACY progenitors** (user-requested; sibling config
    `stream_agama_rnbody_cautun_fixed_legacyprog.yaml`, identical potential, `priors_local` restored
    to the base `stream_agama.yaml` values: m 4.3e3/6.47e4/5.7e4, a 8.43/4.9/6.4 pc, t_end
    4/1.5/1.5 Gyr — note `stream_agama_rnbody` overrides NGC3201/M68 t_end to 4.0, so 1.5 has to be
    written out explicitly, not merely un-overridden). Same seed, so the phase-space draws match
    row-for-row and the two runs differ ONLY in the progenitor inputs. Results:
    * **t_end = 1.5 Gyr makes NGC3201 and M68 unobservable.** In-window particles (of 1000) collapse
      from a median 222 / 472 to **3 / 24**, against 195 / 297 real members — every row falls below
      the observed count, so there is no stream to compare and their track statistics are
      meaningless at that t_end. Both clusters survive 10/10 retaining ~99% of their initial mass
      (m_bound 64053/64700 and 55746/57000): at 1.5 Gyr the restricted N-body model has barely
      stripped anything. This reproduces the 2026-07-05 t_end finding in the Cautun potential and
      confirms it is a property of the rnbody forward model, not of the halo parameterization —
      spray fabricated stripping uniformly over t_end and so never exposed it.
    * **Pal 5 (t_end = 4 in both) is the clean comparison, and the corrected progenitor wins on
      length while the legacy one wins on thinness**: legacy tails are tighter (realization scatter
      0.083 vs 0.140 deg) but too SHORT — phi1 extent 22.2 vs real 28.9 deg (**-6.7**, against the
      corrected run's -1.1) and a phi2 offset of +0.23 deg = 2.8x the realization scatter (corrected:
      1.3x). The light, compact legacy progenitor under-produces tail length as well as width, which
      is the documented "too cold" symptom seen from the other side.
    * **Pal 5 dissolves in 10/10 rows under BOTH progenitor sets** — so its non-survival is
      insensitive to the 4.3e3 -> 1.34e4 mass and 8.43 -> 21 pc radius correction, and points at the
      literature INITIAL mass (4.7-7e4) as the only remaining lever.
    Artifacts: `data_local/cautun_fixed{,_legacyprog}/*_{panels,overlay}.png` + the datasets and
    their `.hydra` snapshots.
  - **Scaled up to 100 realizations x 10^4 particles** (corrected-progenitor config, seed 2026,
    `data_local/cautun_fixed_1e4/`): 42 min wall at `n_workers=48` nice'd (~8 s per stream-sim;
    300 stream-sims), 216 MB npz. `plot_fixed_potential_samples.py` reworked to what this needs —
    ONE 10x10 phi2-only grid PER STREAM (`--grid-cols`, `<out>_<stream>_phi2.png`) plus the
    all-observables overlay (phi2/parallax/mu_phi1/mu_phi2/v_los), with marker size/alpha and the
    legend auto-scaled above 12 samples. 0 NaN rows; in-window particles median 8292/2229/4692 of
    10^4, so no row is member-starved.
    * With 100 draws the sim envelope resolves into a proper band, and the verdict from the 10-row
      run **holds with better-determined scatter**: NGC3201 phi2 **+1.00 deg = 2.7x** the realization
      scatter and mu_phi1 **-2.38 mas/yr = 2.6x** (real sits ABOVE the whole sim band); phi1 extent
      +38.6 deg for NGC3201 and +30.0 deg for M68 — but those two extent numbers are
      **window-saturated and must not be quoted as arm-length excesses** (retraction + the correct
      edge/centre-density diagnostic in the freed-t_end entry below). Pal 5 remains consistent, and
      its mu_phi1 offset went from -0.064 to **-0.008** mas/yr — the 10-row value was noise, so do
      not read single-digit-sample offsets as measurements.
    * New at this sample size: the real members of Pal 5 and M68 sit at the **upper edge** of the
      simulated phi2 band rather than in its middle, i.e. the model's spread is asymmetric about the
      observed track — a shape mismatch that only the 100-draw band makes visible.
    * **Survival at 10^4 particles: Pal 5 0/100** (83 zero, 17 NaN), NGC3201 78/100, M68 43/100.
      Remember `m_bound_final` is resolution-dependent, so these are their own measurement and NOT
      comparable to the 10^3-particle fractions above; the one robust statement across both is that
      **Pal 5 never survives**.
  - **Same 100 x 10^4 run with the PARTICLE-SPRAY forward model** (`stream_agama_spray_cautun_fixed`
    .yaml — inherits the rnbody Cautun config wholesale so the potential is byte-identical, changing
    only `name: stream_agama` + `spray_method: chen`; same seed 2026, so the phase-space draws match
    the rnbody run row-for-row and the ONLY difference is the forward model).
    **2.5 min vs 42 min** — spray is ~17x cheaper here (no per-update Multipole refit). Artifacts in
    `data_local/cautun_fixed_spray_1e4/`. NaN rows 3/2/2 (rnbody had 0); in-window medians
    9330/1749/3891 vs 8292/2229/4692.
    Offsets in units of the realization scatter, rnbody -> spray:
    | | Pal5 phi2 | Pal5 vlos | NGC3201 phi2 | NGC3201 mu_phi1 | phi1 extent NGC/M68 |
    |---|---|---|---|---|---|
    | rnbody | 1.1x | 0.5x | 2.7x | 2.6x | +38.6 / +30.0 deg |
    | spray  | **2.6x** | **2.3x** | **5.6x** | **3.9x** | +39.0 / +30.6 deg |
    * **Spray is not better — it is worse wherever the two differ**, and for a specific reason: its
      realization scatter is uniformly SMALLER (Pal5 phi2 0.144 -> 0.080 deg, Pal5 v_los 4.23 -> 1.54
      km/s), i.e. spray tails are colder/thinner, so the same mean offset becomes far more
      significant. This is the known "sims too cold" symptom isolated cleanly: at identical potential,
      progenitor and phase-space draws, the stripping model alone sets the tail width.
    * Proper-motion medians are essentially perfect in BOTH (Pal5 mu_phi1 -0.008 rnbody / -0.001
      spray), so the pm agreement is not diagnostic of the forward model.
    * The arm-length excess came out identical in the two models (+39/+31 deg for NGC3201/M68).
      **RETRACTED — see the freed-t_end entry below**: for those two streams the in-window phi1 extent
      is saturated by the observation window, so the agreement was an artifact of the statistic and
      carries no information about the stripping physics. Only Pal 5's extent (-1.7 rnbody / -3.4
      spray) is unclipped and readable.
    * Caveat when reading spray here: `m_progenitor` only sets the Jacobi radius and `a_progenitor` is
      unused, so there is no `m_bound_final` and the "Pal 5 dissolves in every row" result cannot even
      be posed — spray fabricates stripping over `t_end` regardless of survivability (the 2026-07-05
      blind spot). Do not read spray's healthy in-window counts as evidence the progenitor is viable.
  - **500-row spray run with the STRIPPING AGE FREED** (`stream_agama_spray_cautun_fixed_tend26.yaml`
    — inherits the spray Cautun config, `t_end ~ U[2,6]` Gyr for all three streams; 10^4 particles,
    seed 2026, `data_local/cautun_fixed_spray_tend26/`, ~10 min / 1.44 GB).
    Motivated by the rnbody-vs-spray result above: both stripping models give the SAME arm-length
    excess, so `t_end` and the potential own it — this run tests whether any single stripping age in
    the Cautun potential reproduces all three observed arm lengths at once. Range narrowed from the
    v2 config's U[2,10] to U[2,6]: below ~2 Gyr NGC3201/M68 have essentially no in-window stars
    (measured: 3 and 24 of 10^4 at 1.5 Gyr), and 6 Gyr keeps the rewind short of the Gaia-Enceladus
    merger where a static halo gets hard to defend; Palau, Wang & Han (2025) put M68's stream age at
    3.04 (+5.63/-0.29) Gyr, inside the window.
    * **Freeing t_end changes what the realization scatter MEANS**: `local_parameter_names` goes from
      `[vr, r, mu_ra_cosdec, mu_dec]` to `[t_end, vr, r, mu_ra_cosdec, mu_dec]`, so the spread now
      mixes measurement error with the t_end prior and is NOT the measurement-error-only yardstick the
      t_end=4 runs provided. Do not compare "x scatter" significances across the two.
    * Spray caveat that bites harder here: no `m_bound_final`, and `m_progenitor` only sets the Jacobi
      radius, so spray will strip a cluster for 6 Gyr that could not have survived it. A long-t_end row
      matching the observed arms is NOT evidence the age is physical — that needs rnbody.
    * **RESULT — no single stripping age fits all three, and for two of them t_end is not even the
      relevant knob.** Median in-window phi1 extent by t_end bin (deg; observed in brackets):
      | stream | 2-3 | 3-4 | 4-5 | 5-6 | observed |
      |---|---|---|---|---|---|
      | Pal5 | 18.2 | 23.4 | 27.4 | 29.6 | 28.9 |
      | NGC3201 | 106.3 | 106.5 | 106.3 | 106.5 | 67.3 |
      | M68 | 130.9 | 132.7 | 132.9 | 133.0 | 102.2 |
      Pal 5 grows monotonically and crosses the observed extent at **t_end ~ 5.1 Gyr**. NGC3201 and
      M68 are **completely flat in t_end** and too long at every age sampled.
    * **CORRECTION to the earlier "arm-length excess" numbers (this session, 100-row runs above):
      the +39 deg / +31 deg excesses quoted for NGC3201 and M68 were measuring the OBSERVATION
      WINDOW, not the stream.** Diagnostic (`scratchpad/window_check.py`, raw projection, no
      augmentation): for NGC3201 the RAW pre-window extent grows 208 -> 242 deg from t_end 2-3 to
      5-6 Gyr while the in-window extent is pinned at 107.0 -> 107.1; M68 raw 173 -> 210 vs in-window
      131.8 -> 133.2. The statistic is saturated, which is exactly why it looked identical between
      rnbody and spray. **So the attribution I recorded from that A/B — "over-long arms are set by
      t_end and the potential, not by the stripping physics" — is wrong on both halves:** t_end
      demonstrably does not move it, and the agreement between models was a window artifact, not
      physics. The extent statistic is only meaningful for Pal 5 (raw 20.1 vs in-window 19.4 at
      2-3 Gyr, i.e. unclipped; mild clipping sets in above ~4 Gyr, raw 49.2 vs in-window 30.9 at
      5-6 Gyr, so even the 5.1 Gyr crossing is slightly contaminated).
    * **What IS diagnostic instead: the phi1 edge/centre number density.** A stream that overflows its
      window piles up at the edges. Sim gives 2.30 (NGC3201) and 2.13-2.48 (M68) versus **0.81 / 0.77
      for the real members** — the real streams fall off towards their ends, the simulated ones do not.
      So NGC3201 and M68 genuinely are far more extended than observed, at EVERY age in [2,6] Gyr,
      which points at the orbit/potential rather than the stripping age. Pal 5's ratio is 0.27 at
      2-3 Gyr (real 0.30) rising to 0.59 by 5-6 Gyr. **Use this ratio, not the extent, for any future
      arm-length claim on the two wide-window streams.**
  - **Plot-script generalization for large sample counts** (`plot_fixed_potential_samples.py`):
    `--grid-cols 0` (new default) picks a roughly square grid (500 -> 23x22 instead of 10x50); new
    `--panel-w/--panel-h` (default 2.4x2.0 in) and `--dpi` (auto: 150, stepping to 110 above 120
    panels, to stay inside matplotlib's pixel limit); panels packed edge to edge via
    `subplots_adjust` — NOT `tight_layout`, which recomputes spacing and undoes the packing; x labels
    go on the last *populated* axis of each column since the final row can be partial. Titles now
    name the varying inputs, read off the stored draws. **Gotcha**: that detection must reduce over
    the ROW axis (`np.ptp(arr, axis=0).max()`) — a ptp over the whole `(n_rows, n_streams, 1)` array
    reports every per-stream constant (m_progenitor, ra, dec) as varying, since those differ between
    streams by construction.
  - **Note on shared-box etiquette**: a user-owned `hydrabflow-simulate` run (`freemass_t26_norej`,
    1000 rows, `n_workers=180`) was live throughout. Budget workers against it (48 nice'd here, not
    the 96 first attempted) and scope any `pkill` to the dataset name in your own command line —
    `pkill -f hydrabflow-simulate-multistream` does not match a plain `hydrabflow-simulate`, but the
    margin is uncomfortably thin.

- Session 2026-08-28 (Frenet-Serret stream remapping — Palau & Miralda-Escude 2023 Appendix D,
  implemented + validated on M68): a cheap surrogate forward model. Simulate ONCE in a fiducial
  potential, freeze every star into the moving Frenet-Serret trihedron of the progenitor's orbit,
  and for any new potential re-integrate only the progenitor orbit and drop the stored offsets onto
  the new trihedra. Same paper (MNRAS 524, 2124) this project's legacy progenitor table came from.
  Diagnostic only — nothing registered as a simulator, no config added, per user.
  - **`scripts/trihedron.py`** (portable core, numpy + an `agama` handle, no hydrabflow imports so
    it lifts into `simulators/` unchanged): `orbit_window` (rewind to -T then one dense forward
    integration over 2T, the `_rnbody_stream` centre-trajectory idiom), `nearest_time` (blocked
    argmin over the knot grid + parabolic sub-knot refinement), `trihedron`, `build_template`,
    `remap`, `auto_window`. Two implementation choices worth keeping: (1) the paper's
    `a = d/dt(v/|v|)` equals `(g - (g.v^)v^)/|v|`, so **e2 is the normalized perpendicular
    acceleration taken analytically from `pot.force()`** — no finite differencing of the orbit;
    (2) interpolation is **cubic Hermite using the orbit's own (x,v) and (v,g)** at the knots, which
    needs no scipy and makes the fiducial round-trip exact (measured max |dx| = **1.3e-11 kpc** at
    1e5 stars — the `--check-fiducial` self-test).
  - **`scripts/compare_trihedron_vs_spray.py`**: three arms at identical potential/progenitor/seed
    — `rnbody` (ground truth), `trihedron` (template built once at q_fid), `spray` (Chen+2024).
    Caches every expensive arm under `<outdir>/cache` keyed by (stream, arm, q, N, seed), so
    re-running only redraws figures. Reuses `stream_frame`/`project_sample`/`robust_lim`/`QTY` from
    `plot_fixed_potential_samples` and `fit_frame`/`binned_median`/`binned_std`/`WINDOW` from
    `ppc_summary_statistics`. Note the solar frame is potential-dependent (`_solar_frame` uses
    `v_circ(R0)+V_Sun`), so the progenitor's Galactocentric state is recomputed per q from its
    FIXED observed ICRS coordinates — both arms see the same one.
  - **Run**: M68, **1e5 particles**, Cautun+2020 fixed potential (q_fid = 1.0), q in
    {0.7, 0.85, 1.0, 1.05, 1.2, 1.4}, T = 200 Myr / 2001 knots, seed 2026. Template diagnostics
    clean at that T (boundary_frac = 0, ambiguous_frac = 0), so the window contains the whole
    stream — the paper's own T = 40-60 Myr was NOT enough of a prior to trust here, it was checked.
  - **RESULT — the remap beats particle spray against the N-body truth at every q.** Median
    |d phi2 track| over 10 phi1 bins (deg): q=0.7 **0.313** vs spray 0.551; 0.85 **0.259**/0.413;
    1.0 **0.000**/0.548; 1.05 **0.100**/0.386; 1.2 **0.131**/0.485; 1.4 **0.071**/0.388. The error
    is **asymmetric about the fiducial** — it grows steadily on the oblate side (q<1) and stays flat
    and small out to q=1.4 — so plot it against q, never |q - q_fid|, which folds the asymmetry away.
  - **Cost**: remap **56 ms** per potential for 1e5 stars vs **85-103 s** for the restricted N-body
    and 11 s for spray, i.e. ~1600x the ground truth and ~200x spray. Template build (excluding its
    one N-body run) 7.4 s.
  - **Per-star comparison does NOT work, and that is itself the finding.** `_rnbody_stream` draws
    the Plummer progenitor before anything potential-dependent, so star i is nominally the same star
    at every q — but the per-star |dx| is ~10 kpc and **saturates**, identical at dq=0.05 and
    dq=0.4, decomposing into ~9.5 kpc along-track and only ~1.9 kpc perpendicular. ~10 kpc is about
    half M68's stream length, i.e. individual stars decorrelate completely over 4 Gyr even in a
    barely-changed potential. Judge these models distributionally; the script reports the
    decomposition so the trap is visible rather than mistaken for remap error.
  - **The method's real limitation is along-track structure, by construction.** `t_hat` is frozen at
    the fiducial, so the remap CANNOT change how stars are distributed along the orbit. Measured via
    the phi1 edge/centre density ratio: rnbody swings 2.10-4.82 across the sweep while trihedron is
    pinned at 1.98-2.18 (spray likewise 1.95-2.10). **Caveat on that statistic here**: M68's
    progenitor sits at phi1 ~ -85, inside an edge bin, so the ratio is tracking progenitor SURVIVAL
    (it correlates with `m_bound_final`: 4.82 at q=1.05 where m_bound = 4.6e4, 2.10 at q=1.0 where
    m_bound = 0), not window overflow as in the 2026-07-29 usage. The remap inherits the fiducial's
    remnant and so cannot represent q-dependent survival either.
  - Confirms the 2026-07-29 warning independently: **in-window phi1 extent is useless for M68** —
    134.4-135.1 deg for all three arms at all six q, fully window-saturated.
  - **NOT done**: only M68 was run (user scoped it); Pal5/NGC3201, other parameters than q_halo, and
    the question of whether a per-q `t_hat` rescaling could restore the along-track density are open.

- Session 2026-08-28 (trihedron remap at 1e6 particles, all three streams, q AND mass sweeps —
  **the "beats spray everywhere" claim does NOT generalize**): the follow-up the previous entry
  scoped. Same fixed Cautun potential, seed 2026, 1e6 particles per arm, noise-convolved with 120
  pooled realizations. Artifacts + full tables in `data_local/trihedron_1e6/README.md`;
  `summary.json` + `combined_error_vs_param.png` are the headline.
  - **Script generalized** (`compare_trihedron_vs_spray.py`): `--param`/`--values` sweep ANY global
    (was a hardcoded `Q_KEY`), per-stream output subdirs `<outdir>/<stream>/`, cache tags keyed by
    parameter with the untouched fiducial tagged `fid` so it is SHARED between sweeps (saves one
    N-body per stream), `halo_m200()` (bisection on `enclosedMass(r) = (4pi/3) 200 rho_crit r^3`,
    H0=70.4 as in `_halo_params_m200c`) reporting M200/r200/c200 per value, `--noise-subsample`
    (default 20000: tiling 1e6 particles x 120 realizations is ~6 GB, and the chain keeps only
    `observed_n_stars` anyway), and a new **observable-space corner** per swept value (6 observables,
    3 arms as 68/95% contours, Gaia members overplotted, measured-v_los stars only — `corner` was
    already a dep). Runner `scripts/run_trihedron_1e6.sh`, collator `scripts/summarize_trihedron.py`.
  - **BUG FOUND AND FIXED — `sky_projection` is all-or-nothing on NaN**: agama returns an ALL-NaN
    projection if a SINGLE input row is non-finite. At 1e5 it never fired; at 1e6, two NaN stars out
    of a million (routine for spray and for a dissolved progenitor) silently zeroed whole arms
    (`in-window spray=0`, `track offset=nan`) — which reads as "this model produces nothing" rather
    than "the projection failed". `project_xv` projects the finite subset and writes NaN back,
    preserving the row indexing `per_star_error` needs. **Any other caller of `sky_projection` on
    unfiltered particles has this bug.** All numbers below are post-fix.
  - **q sweep** (median |d phi2 track| vs the N-body truth, remap / spray, deg): M68 0.323/0.558,
    0.267/0.432, —, 0.123/0.397, 0.139/0.524, 0.066/0.380 (remap wins at ALL six, 2-6x — and
    reproduces the 1e5 run to <=0.02 deg, so that result was NOT resolution-limited); **NGC3201**
    0.697/0.422, 0.494/0.467, —, 0.045/0.749, 0.367/1.005, 1.159/1.464 (**loses on the oblate
    side**); **Pal5** 0.096/0.057, 0.051/0.058, —, 0.016/0.087, 0.056/0.092, 0.308/0.097 (**loses at
    both extremes**). So the previous session's headline held only for the one stream it was tested
    on. The q-asymmetry direction is also stream-specific: Pal5 degrades on the PROLATE side, M68 and
    NGC3201 on the oblate side.
  - **Mass sweep** (`rho` x[0.5,0.7,1,1.4,2] = halo M200 2.84e11-1.44e12; NOTE this moves M200 AND
    concentration together, c200 14.0->24.1, since `a` is held): the remap **loses almost everywhere**
    — Pal5 5.1x/3.2x/—/1.7x/1.2x worse than spray, NGC3201 up to 6.5x, M68 2.2x/1.4x/—/0.8x/1.3x
    (its one win). **Why: `t_hat` is frozen, so the remap cannot redistribute stars ALONG the track,
    and a mass change is mostly an orbital-period rescaling — i.e. exactly that. A `q` change bends
    the orbit sideways, which the stored perpendicular offsets do capture.** Spray's error is roughly
    FLAT in both parameters (it re-derives release conditions per potential) while the remap is exact
    at the fiducial and degrades away from it, so the remap only wins inside a window around the
    fiducial.
  - **Usable range** (beats spray AND < ~0.15 deg, the measurement-error-only realization scatter of
    the 2026-07-29 runs): M68 q in [1.0,1.4]; Pal5 q in [0.85,1.2]; NGC3201 q in [1.0,1.05]; **mass:
    none, on any stream**. Far too narrow for a prior of q in [0.7,1.4] x M200 in [0.5,2.5]e12 — a
    single template cannot cover it. Tiled/local templates (one per prior cell) are the only route to
    a training-set generator.
  - **Along-track density, quantified** (phi1 edge/centre ratio over the q sweep): rnbody moves
    2.07-4.85 (M68), 0.99-2.72 (NGC3201), 0.38-1.38 (Pal5) while the remap is pinned at 1.94-2.16 /
    1.11-1.16 / 0.60-0.75 and spray is likewise flat. Any inference leaning on along-track density
    (arm length; the edge/centre statistic this project uses for the wide-window streams) is biased
    by EITHER surrogate.
  - **Neither surrogate knows whether the progenitor survives.** At M200=2.84e11 the N-body says
    NGC3201 is never stripped — `m_bound_final` = 1.92e5 = its full initial mass, all 1e6 particles
    in a clump spanning 8.66-9.41 kpc at RA 142-164, dec -49..-44, i.e. OUTSIDE its window, so there
    is no stream. Spray still puts 214,882 stars in the window and the remap 23,400. Spray's
    blindness is already documented (it fabricates stripping over `t_end`); the remap inherits it
    structurally by replaying the fiducial's already-stripped population. M68 shows it mildly at the
    same mass (rnbody edge/centre 140 = barely-stripped clump).
  - Cost unchanged and still overwhelming: ~8-15 min per potential for the N-body at 1e6 (32 agama
    threads) vs milliseconds per potential for the remap after one template build. It is the accuracy
    envelope, not the speed, that limits use.
  - **NOT done**: no tiled-template test (the obvious next step); whether a per-value `t_hat`
    rescaling could restore the along-track density is still open and is now clearly THE lever, since
    it is the same defect behind both the mass-axis failure and the edge/centre pinning.

- Session 2026-09-11 (merge of main's codebase slimming into stream_project): pulled `origin/main`
  (commits d7b265e..44f016c: "Strip the package to bare bones", "Consolidate five registries into
  one", "Fold checkpoint.py into artifacts.py", ...) and ported the stream work onto its structure
  instead of resurrecting the old one. Adopted from main: single `registry.py` (lazy per-package
  discovery; the old `simulators/preprocessing/augmentation/registry.py`, `utils/discovery.py`,
  `utils/logging.py` are gone — stdlib `logging.getLogger`), `config.py` with only `RootConfig`
  registered (all `- base_<group>` defaults stripped from 56 preset YAMLs), `pipeline/artifacts.py`
  (absorbs `checkpoint.py` + the train/eval save helpers; `load_approximator(compile=False)`),
  `evaluate` with the real-data mode on `data.real_data_path` (`evaluate_real.py` kept as the
  helper module; the `hydrabflow-evaluate-real` console script and the `inference` config group are
  gone — `eval.num_samples/batch_size`), `run_name`-based run dirs (`run_name: ${model.name}`;
  `ModelConfig.name` defaults to `<summary>+<inference>` so whole-model presets keep labelling runs),
  `embed_dim_per_head` (replaces `params.embed_dim_multiplier` and raw `embed_dim` everywhere incl.
  tuning search paths), `BaseSimulator.is_batched`, `Adapter.create_default`, the thin `scripts/*.py`
  entry points deleted (shell scripts call `python -m hydrabflow.pipeline.<stage>`), main's tests and
  docs (`running/configuration/extending.md`) + a new `docs/streams.md`. Kept from the branch:
  everything stream-specific (simulators, augmentations, preprocessing steps, fusion/masked nets,
  compositional + misspecification pipeline, `_bf_patches`, resumable chunked `io.run_chunked`,
  convergence report, `report.md`), the group-preset mechanism (now `- <group>: null` placeholders
  in `config.yaml` with `_self_` first; model presets use `override summary_network:`),
  `training.standardize` incl. `summary_variables`, and the **bayesflow 2.0.12 PyPI pin** (main moved
  to the git HEAD 2.0.13; the compositional patch is version-specific — not followed, flagged).
  `BaseSimulator.sample` now validates declared keys/leading axis but tolerates the extra arrays the
  stream simulators emit. Verified: every preset composes; the full test suite passes (run
  single-threaded on the shared box, see Notes); two_moons simulate→train→evaluate→evaluate(real)
  end-to-end on CPU; three stream workflows (model5/global, ibata_grid_masked/global,
  maskedvlos/local) build with the fusion nets + compositional patch. Not done: no GPU training rerun
  of a stream model on the merged code (the architectures are unchanged, so old checkpoints remain
  loadable only if their configs are re-expressed with `embed_dim_per_head`).

- Session 2026-09-12 (galpy `StreamTrack.from_particles` as a smooth track+covariance summary —
  feasibility on the REAL Gaia members + sim rows; nothing adopted yet): galpy 1.12 (not a project
  dep; run with `uv run --with galpy`; do NOT import it in the same process as JAX — static-TLS
  clash aborts with `pthread_create failed`, so the observation model is dumped from a separate
  process). Scripts + figures + `report.json` in `outputs/streamtrack_feasibility/`. Method needs
  full 6D Galactocentric phase space per star and a densely sampled progenitor orbit ("ruler"; tp =
  closest-point time on it, offsets smoothed in Cartesian and added back), fits ONE arm at a time
  and its tp grid always reaches tp=0 (extrapolates to the progenitor even with no stars there —
  mask to the phi1 support). Real data therefore needs: an assumed potential+progenitor orbit (used
  galpy Cautun20; median sky residual to members Pal5 0.3 / NGC3201 0.8 / M68 1.4 deg, max 7 deg),
  imputed distances (Gaia parallaxes useless; sky-closest point on the ruler vs constant distance
  changes the track by 0.03 deg for Pal5 but 0.1-0.2 deg / 0.1-0.3 mas/yr / 2-5 km/s for
  NGC3201/M68), and a v_los policy: galpy has NO missing-data support (`numpy.cov` over all six
  columns, no weights). Stars without v_los need NOT be dropped for the sky+pm track (impute v_los,
  the phi2/pm track is unaffected at the 0.01-0.07 deg level on sim truth), but the v_los track and
  every v_los covariance entry of that fit are then the assumption (sim: 2-40 km/s error, sigma
  2.3 -> 0.3 km/s); v_los must come from a separate measured-only fit (Pal5 25+44 stars OK,
  NGC3201 37 marginal, M68 29 -> galpy falls back to linear interpolation below 5 valid bins).
  Gotchas: default `ntp=max(21,sqrt N)` leaves 6/21 empty bins per Pal5 arm and empty bins are
  ZERO-filled in the covariance smoother (same bias as our 2026-07-28 summary fix) — use
  ntp ~ N/8; wrong-wrap imputation assigned 18 M68 stars to a 0.5 Gyr "trailing" arm (restrict
  the ruler window per stream); rows whose true potential is far from the ruler kink (galpy
  warns; ~half the M68/NGC3201 prior rows), so sim and real must use the SAME fixed ruler to be
  the same estimator. The covariance is the 6D Cartesian scatter per tp bin mapped by a Jacobian
  at the mean: along-track spread leaks into phi2/pm when the track is inclined, sig_phi2 changed
  up to 2x between true and imputed 6D while the mean moved <0.15 deg — not a drop-in for the
  per-bin std. Verdict: a smoother mean track for Pal5 (and NGC3201 with care), not for M68 with
  the Cautun ruler, and not for the covariance.
- Session 2026-09-12 (prior review for halo-shape inference — `docs/priors_literature.md`): three
  web literature sweeps (stream-inference priors, cosmological mass/concentration/shape priors,
  baryonic + solar-frame priors) diffed against the `_v2` prior table. Findings: the halo priors are
  supported (uniform density q in [0.5,1.5], log-uniform M200, McMillan's ln c ~ N(2.56,0.272)
  which ≡ c200≈10 and is right for an UNcontracted profile mimicking the contracted MW); no stream
  paper uses a simulation shape prior and doing so (c/a≈0.72±0.12) would exclude half the
  observational literature (0.56–1.40). Four changes recommended: gamma U[0,1.7] (negative gamma
  has no support); thin-disc Sigma U[1e7,3e9] Msun/kpc² admits a 1.3e11 disc — parameterize by mass
  U[1.5e10,7e10] and RESTORE the thick disc (its absence is why 3e9 was needed for v_term);
  v_phi,sun should be pinned by Sgr A* (4.74047×6.411×R0 = 248.5±0.8 km/s at 8.178) instead of
  v_c(R0)+V_pec(12.24±2.05) computed in each row's potential, which imposes the low-V_sun solution;
  R0 N(8.23,0.06) to span GRAVITY 2019 vs 2021 (3σ apart). Minor: widen tilt to [0,45]°, add a
  marginalized gas factor N(1,0.25), widen log10 M200 low edge to 11.6. Conventions to state: our
  M200 is the HALO ONLY inside its own r200 (baryons add 4–17 %, measured on 6 rows); q,p are density
  axis ratios ((1−q_rho)≈3(1−q_Phi)). Nothing changed in configs yet.
- Session 2026-09-12 (halo tilt: units BUG fixed + stream-position sensitivity): agama's
  `orientation` Euler angles are RADIANS (verified: (0, pi/2, 0) swaps the y/z densities exactly);
  `_halo_shape_extras` passed the config's degree-valued `tilt_TwoPowerTriaxial_halo` straight
  through, so the v2 prior U[0,30] "deg" was U[0,30] rad = a near-random orientation of the minor
  axis in the y-z plane. Fixed (`np.radians`), test `test_halo_tilt_is_declared_in_degrees` (90 deg
  swaps y<->z; 20 deg puts the density minimum 20 deg from z). No dataset had been generated with
  v2, so nothing stored/trained is affected. Geometry: the second Euler angle rotates about the x
  (Sun-centre) axis, tipping the minor axis toward the direction of rotation while the halo x axis
  stays on the Sun-centre line — it cannot represent Nibauer & Bonaca's major-axis-toward-the-Sun
  solution (needs pitch+yaw). **Stream sensitivity** (spray 1e4, v2 fiducial potential with q=0.8,
  p=0.9, same seed; `outputs/tilt_check/`): median |dphi2| of the binned track vs tilt 0 —
  Pal5 0.015 deg (20 deg) / 0.027 (45 deg); **NGC3201 0.27 / 0.68 deg** (+0.15/0.19 mas/yr mu_phi1,
  larger than the q 0.8->1.0 effect of 0.18 deg, and it shifts phi2 UP at phi1<0, toward the real
  members' +1 deg offset seen 2026-07-29); M68 0.04 / 0.09 deg (q dominates: spherical = 0.89 deg).
  The old bug (20 rad ~ 66 deg effective) gave 0.07 / 0.68 / 0.21 deg. So a tilt about the
  Sun-centre axis is a real, NGC3201-identifiable degree of freedom at 20 deg, not a nuisance.
- Session 2026-09-12 (v3 prior config): `conf/simulator/stream_agama_rnbody_ibata_m200c_v3.yaml`
  (inherits v2) = the "minimal changes" prior to investigate next: tilt identity 0 (no tilted
  halo), `vcirc_rejection: null` (no rotation-curve cut), gamma U[0,1.5], Sigma_Disk U[3e8,1.6e9],
  log10 M200 (halo-only) U[11.6,12.4], R0_Sun N(8.23,0.06), V_Sun N(12,5) (code-free fallback;
  the Sgr A*-pinned v_phi,sun is still a deferred code change). 10 inferred globals (gamma, q, p,
  alpha, log10 M200, ln c', r/z/Sigma_Disk, rho_Bulge); solar frame marginalized; locals from
  `_prog` + t_end U[2,10]. Composes; 2000-draw prior sanity OK. Generate with
  `SIM=stream_agama_rnbody_ibata_m200c_v3 DATA_DIR=data_jarvis/data_agama_rnbody_ibata_m200c_v3_hydrabflow
  scripts/create_ibata_rnbody_m200c_v2_dataset.sh`. Not generated/trained.
- Session 2026-09-12 (v3 pilot recovered; NaN diagnosis; a_progenitor pinned to 30 pc): the
  previous session's 333-group v3 pilot had crashed twice with `MemoryError` — NOT our workers'
  RAM: the box runs `vm.overcommit_memory=2` (strict) and other users' jobs held Committed_AS at
  the 532 GB CommitLimit while 900 GB sat physically free, so every new allocation failed with
  Errno 12. Killed the hung 128-worker tree (~24 GB of commit charge; `pkill -f` with the dataset
  name matches your own shell — kill by PID list instead) and reran at `n_workers=64`, which fit
  in the ~10 GB headroom. Check `grep Committed_AS /proc/meminfo` against CommitLimit before any
  large joblib launch. **NaN diagnosis** on that pilot (12.8 % of member sims, 35 % of groups):
  confined to NGC3201/M68 with COMPACT progenitors — a<5 pc & t_end>=5 Gyr failed 88 %/94 %, a>=5
  & t<5 failed 0-2 %, Pal5 (a in [10,35]) never; the failure is the per-update orbit step cap
  (steps/update ~ (t_end/12)/P_inner, P_inner ~ sqrt(a^3/m)). User decision: **a_progenitor =
  30 pc fixed for all three streams** (Ibata+2024, "Charting the Galactic acceleration field II",
  single Plummer scale length), written into `stream_agama_rnbody_ibata_m200c_v3.yaml`. Regenerated
  pilot: **0/999 NaN**, 6x faster (7.7 vs 1.3 stream-sims/s at 64 workers); survival Pal5/NGC3201/
  M68 = 21/41/38 % overall, 34/79/85 % at t_end<5 vs 14/18/13 % at t_end>=5. The a-free pilot is
  archived under the dataset's `superseded_a_free/`. Capping t_end at 5 Gyr was NOT needed for
  the NaNs; it remains a physics choice (survival). **Coverage check** (`scripts/ppc_summary_coverage.py`
  from the previous session, on the 333 set → `outputs/v3_prior_coverage/`): every real
  (stream, statistic, phi1-bin) cell inside the sims' central 99 % — the observed streams are NOT
  out of distribution under v3 — but one-sided: NGC3201 std_phi2 at the 4-19th pct in ALL bins
  (real thinner than sims), M68 std_phi2 77-98th (real wider), Pal5 centred. **10k flat training
  set generated** (`training_data_10000.npz`, 730 MB, seed 2026, user kept t_end U[2,10]): 0 NaN,
  survival Pal5/NGC3201/M68 21/40/36 % (t<5: 35/75/79 %; t>=5: 12/18/10 %). Launch needed a
  headroom watchdog (`logs/run_full_10k.sh`: polls Committed_AS, sizes n_workers ~5/GiB cap 64,
  resumes chunks on crash) — the box was pinned at the CommitLimit; the user's idle vscode-server
  (~85 GB VSZ; keeps running after the window closes) was what freed it. Ran at 40 workers, 36 min.
  PPCs in `<dataset>/ppc/full/`: v_term obs points inside the prior 5-95 % band 42 %, Sigma_z
  prior median 64.8 (obs 71+/-6, 15 % of rows within 1 sigma); noise-convolved cold-stream table
  P(sim<real) Pal5 0.20-0.50, NGC3201 phi2 0.08 / mu_phi1 0.12 (sims too WIDE/hot), M68 0.76-0.78
  (sims too cold, as before). **NGC3201 width diagnosis** (noise-convolved per-bin std_phi2 over
  the 3273 NGC3201 training rows, `scratchpad/ngc_width_drivers.py`): real 0.87 deg sits at the
  2nd pct of the prior predictive; width is driven by t_end (Spearman +0.53; median 1.6 deg at
  2-4 Gyr -> 4.8 at 8-10) and survival (thin 19 % if the progenitor survives vs 2 % if dissolved),
  NOT by any global (|rho|<=0.11) — so it is a local-nuisance mismatch, not a halo-bias mechanism.
  Only 48 % of NGC3201 rows have >=20 in-window stars (median 19 vs 195 real) at 1000 particles.
  **Probes** (`<dataset>/probes/`, 1000 NGC3201-only rows via `~simulator.params.target_streams.
  {Pal5,M68}` + `priors_local.NGC3201.t_end.prior_parameters=[2,5]`; note a plain
  `target_streams={NGC3201: 1}` override MERGES, it does not replace): t_end U[2,5] lifts survival
  40 -> 78 % (95/83/57 % at 2-3/3-4/4-5 Gyr), 0 NaN, but width barely moves (median 1.72, real at
  3rd pct). At **1e4 particles** (`ngc3201_tend25_1e4`, 23 min/64 workers, 720 MB) the resolution
  problem disappears (median in-window = 195 = real, 80 % usable) while the width gets WORSE:
  median 2.05 deg, 3 % as thin as real, real at the 0.1 pct — sparse 1e3-particle sims sampled only
  the dense core and UNDER-estimated the sim width. Conclusion: the simulated NGC3201 is
  intrinsically ~2.3x too wide at every t_end in [2,5]; t_end is the survival lever only; the width
  lever must be the fixed 30 pc progenitor radius / mass (not probed yet) or missing physics.
  **Radius probe** (`ngc3201_tend25_1e4_a15`: same seed/t_end/1e4 particles, a_progenitor 15 pc;
  52 min, 2.3x slower than 30 pc): the radius is NOT the width lever — median std_phi2 2.05 -> 1.86
  deg (real 0.87), thin 3.0 -> 4.7 %, real at the 1.2 pct; survival unchanged (77 %); the compact
  cluster strips LESS (median in-window 195 -> 102). Stream width is set by the tidal radius
  (~ M^{1/3}), not by the Plummer scale — the untested lever is `m_progenitor` (U[1.93e5,5.61e5],
  B&H18 present-day 1.6e5) and/or missing physics. **Mass probe** (`ngc3201_tend25_1e4_m1e5`:
  a=30 pc, t_end U[2,5], 1e4 particles, seed 2026, `m_progenitor` pinned identity 1.0e5 Msun via
  `priors_local.NGC3201.m_progenitor.type=identity`; 13.5 min): mass IS a lever, but a partial one —
  median std_phi2 2.05 -> **1.69 deg** (real 0.87), thin (<=1.08) 3.0 -> **14 %**, real at the 2.0 pct
  (was 0.1); per t_end bin 1.68/1.56/1.88 deg, thin 7/21/14 %. Survival drops 78 -> 66 % (86/68/44 %
  at 2-3/3-4/4-5 Gyr; the lighter cluster dissolves) while the in-window count stays at the real 195
  median and >=100-star rows RISE to 80 %. M^{1/3} scaling from the prior median 3.8e5 predicts ~1.3
  deg, so the response is weaker than tidal-radius scaling and the factor ~2 is not closed by any
  progenitor knob: radius no, t_end no, mass ~1/3 of the way. Remaining suspects are missing physics
  (or a real-member selection thinner than the true stream).
- Session 2026-09-12 (v4 = v3 at 1e4 particles + NGC3201 progenitor re-centred; 333 test set):
  `conf/simulator/stream_agama_rnbody_ibata_m200c_v4.yaml` (inherits v3): `n_particles: 10000`
  (1e3 under-sampled the stream and UNDER-estimated sim widths), NGC3201 `m_progenitor U[1.0e5,
  3.5e5]` (floor below the B&H18 present-day 1.6e5, accepted knowingly; ceiling halved because the
  B&H18 M_Ini includes stellar-evolution mass loss the N-body lacks), NGC3201 `t_end U[2,5]`
  (survival); Pal5/M68 keep U[2,10]. Dataset dir `data_jarvis/data_agama_rnbody_ibata_m200c_v4_
  hydrabflow/`. **Test set** `test_multistream_333.npz` (seed 7, 64 workers, 8 min, 720 MB): 0 NaN;
  survival Pal5/NGC3201/M68 20/70/44 %. **Coverage** (`outputs/v4_prior_coverage/`): all cells inside
  the central 99 %; 3 NGC3201 cells outside 95 % (std_phi2 bin 1 at 1.5 pct; bin-0 med/std vlos at
  99). Resolution effect confirmed for M68: its real dispersions move from the 77-98th pct (v3, 1e3)
  to the 65-80th (all four quantities; noise-convolved P(sim<real) 0.67-0.82), so half of the "M68
  too cold" verdict was 1e3-particle under-sampling. Pal5 unchanged and centred (P 0.14-0.55).
  NGC3201 std_phi2 is NOT fixed by the mass change at the prior level: real at the 2-17th pct per
  bin (v3: 4-19), noise-convolved sim median 1.93 deg vs real 0.87, P=0.02 — the probe's 1.69 deg was
  at m pinned to 1e5, the prior median ~2.2e5 gives back most of it. Decision: proceed anyway (no
  progenitor knob closes it; it is a local-nuisance mismatch, |rho| with globals <= 0.11). **10k
  training set launched** via `logs/run_full_10k.sh` (headroom watchdog, seed 2026, chunk 1000,
  ~6 h at 64 workers) — **then STOPPED by the user** (killed at row ~50, chunks removed) for two
  changes, and the first v4 test set / coverage run archived under the dataset's
  `superseded_zhou_huang_grid_fullstore/` (+ `outputs/v4_prior_coverage_zhou_huang_grid_fullstore/`).
- Session 2026-09-12 (v4 final: Ou+2024 rotation-curve grid, in-window storage cap, 1e5 rows):
  - **`obs_r_grid: custom`** (new in `stream_agama`): the vcirc observable is evaluated on an
    explicit config table `obs_r_kpc`/`obs_vc_kms`/`obs_sigma_vc` (all three required, equal length,
    increasing radii — `_custom_rotation_curve` validates). v4 uses the **Ou et al. 2024** (MNRAS
    528, 693; Gaia DR3 + APOGEE) curve supplied by the user: 19 radii 6.6-25.5 kpc, copy in
    `assets/rotation_curve_custom_v4.csv`; `vcirc_kms` is `(n, 19, 1)`. `fill_stream_grid_from_
    simulator` now triggers for `custom` as well as `extended`, so `mask_vcirc_radii` (r_min 5.5
    keeps all 19), `add_noise_to_vcirc` sigma and `attach_observed_vcirc` follow the simulator with
    no preset edits. Models trained on v2/v3 data are NOT evaluable on v4 data (different observable).
  - **`store_window_subsample`** (new in `stream_agama.simulate`, helper `window_subsample`): the
    simulation runs at `n_particles` (1e4) but `sim_data_projected` keeps only the stars inside the
    row's stream RA/Dec window (table duplicated in the simulator yaml, test-enforced == the
    augmentation's `observational_window`), at most `max_particles` (2000; uniform random subset
    beyond), float32, padded with the finite sentinel `pad_value=-999` (outside every window, so the
    training `observational_window` step masks it; `convert_distance_to_parallax` runs first and
    maps it to -0.001, harmless). `sim_data_carthesian` is not stored. Failed rows stay all-NaN so
    `drop_nan` still drops them. Verified: the full `stream_global_ibata_grid_v2` chain on capped
    sims attends exactly min(stored, real count) stars, no sentinel attended. Motivation: 1e5 x 1e4
    rows = ~72 GB (720 KB/row) — un-assemblable and un-trainable under this box's ~10 GB commit
    headroom (BayesFlow's offline fit holds the whole set in RAM), while the observation model keeps
    ~200 stars/row anyway. Stored size ~48 KB/row (~5 GB per 1e5). **Anything reading
    `sim_data_projected` raw must apply the window / drop the -999 sentinel first** (the PPC scripts
    go through `augment_sim`, which does). `sample_compositional` reshapes with the stored count.
  - Tests: `test_custom_rotation_curve_grid_is_the_config_table`, `test_window_subsample_keeps_in_
    window_stars_and_pads`, `test_v4_store_window_matches_augmentation_window` (tests/test_streams.py).
  - **Launched** `logs/run_v4_full.sh` (two-stage headroom watchdog): 333-group test set (seed 7)
    then **`training_data_100000.npz`** (seed 2026, chunk 1000). New test set: 38 min at 35 workers,
    0 NaN, **48 MB** (720 MB with full storage), survival 20/70/44 %, stored in-window stars capped
    (2000) in 61/9/51 % of Pal5/NGC3201/M68 rows, NGC3201 median 535. **Coverage on it**
    (`outputs/v4_prior_coverage/`) reproduces the full-storage set to the percent: all cells inside
    99 %, NGC3201 std_phi2 bin 1 at 1.5 pct + bin-0 std mu_phi1/vlos at 97-98 pct; noise-convolved
    cold table Pal5 P 0.09-0.54, NGC3201 phi2 0.01 (sim 1.92 vs 0.87 deg), M68 0.69-0.80 — so the
    storage cap changes nothing the network sees. **Worker sizing**: the watchdog first got 35
    workers (headroom 10 GB) → ETA ~56 h; the user's idle vscode-server held ~80 GiB of commit
    (four helper processes 33/18/13/11 GiB) — killed after the user closed VS Code, headroom → 20
    GiB; relaunched (resume from chunk 1) at 64, then 85 (formula), then **forced 100 workers**
    (`MIN_WORKERS=MAX_WORKERS=100`, ~26 GiB reserved, ~9 GiB margin) per user. Each relaunch costs
    only the chunk in flight. Other users hold 388 + 210 GiB of the 508 GiB limit
    (`vm.overcommit_ratio=50` on a 1 TB box — an admin could raise it). **Completed 2026-09-14 10:56**:
  `training_data_100000.npz` (4.6 GB; `sim_data_projected` (1e5,2000,6) float32, `vcirc_kms` (1e5,19,1)),
  **2 NaN rows** (0.002 %), streams 33469/33181/33348, stored in-window stars median 2000/480/1288
  (capped in 67/8/45 %), survival 21/73/37 %. Wall ~22 h of which ~12 h was waiting on commit headroom:
  crashed twice (12:06 and 21:56 on 09-13, both `_ArrayMemoryError` in a worker/parent when OTHER users'
  processes filled the limit — 09-13 evening a third user's fresh vscode-server took ~100 GiB within 15
  min); the watchdog resumed from the saved chunks each time (63 → 91 → 100). Ran at 35/64/85/100/40/65/
  70/100 workers across restarts; ~14 min per 1000-row chunk at 100, ~19 at 65.
  **Coverage on the training set** (3000-group triplet subsample of the flat file, one row per stream;
  `outputs/v4_prior_coverage_trainingset/`): summary-track cells (PPC estimator) all inside the central
  99 %, NGC3201 std_phi2 bin 1 at 1.6 pct + bin-0 vlos at 98-99 — identical to the 333 test set.
  **NEW binned check** `scripts/ppc_summary_grid_coverage.py` (`.../binned/`): rebuilds the exact
  14-channel `sim_summary` grid the network ingests (training chain up to `stream_summary_grid` for
  sims, the real preset's grid step for Gaia; MAD scale, min_count 3, cells with real occupancy <3
  skipped, sim cells <3 treated as missing like the masked backbone). All ten STATISTIC channels are
  inside (Pal5 97 % / NGC3201 89 % / M68 93 % of cells inside 95 %; every exception is an occupancy
  cell). **The OCCUPANCY channels are out of distribution**: the real per-bin counts are flat by
  construction (equal-count quantile edges: 13/19/30 stars per bin) while the sims put 1-8 stars in
  the central bins and pile up at the ends — real n_track sits at the 97-100th pct in bins 1-8 for
  NGC3201 and M68, 91-99 for Pal5; n_vlos likewise. Only ~96/92/84 of the ~129/195/297 attended sim
  stars land inside the real members' phi1 span (26/53/72 % out of range): the simulated streams are
  far longer along phi1 than the real member footprint (the 2026-07-29 edge/centre finding, now in
  the network's input). Consequence: a `masked_time_series_transformer` model trained on v4 reads the
  real occupancy vector as an extreme input — either drop occupancy as a FEATURE (keep it for masking
  only), or accept the extrapolation knowingly. Statistic-channel coverage is conditional on sim bins
  with >=3 stars. **Repeated on a 30 000-group subsample (90 000 rows, `outputs/v4_prior_coverage_trainingset_30k/`): every number reproduces the 3000-group run to within a percentile** (track: NGC3201 std_phi2 bin 1 at 1.2 pct; binned: 97/89/93 % inside 95 %, same occupancy cells out), so the verdict is not sample-noise. Not acted on. Next: coverage check on the new test set, joint MMD vs
    the training set, GPU train (`model=stream_fusion_ibata_grid_masked adapter=... augmentation=
    stream_global_ibata_grid_v2 preprocessing=stream_global_log10_ibata_sumstats composition=global`),
    real eval.

- Session 2026-09-13 (modality masking ported from protoplan; the compositional rotation-curve
  over-counting): the referee's point is a **real bug, confirmed in code**.
  `flatten_members` (`pipeline/compositional.py:285`) copies every *group-level* observable into all
  `m` members, so each compositional item is `(stream_j, vcirc, ...)` and BayesFlow's identity
  `p(θ|Y) ∝ p(θ)^(1-n) Π_j p(θ|y_j)` (`diffusion_model._compositional_score_direct`) raises the
  whole potential-derived likelihood to the power `m` — **not only vcirc**; `vterm_kms` and
  `sigma_z` are copied the same way. Conjugate-Gaussian illustration: precision goes from
  `1 + m/s² + 1/c²` to `1 + m/s² + m/c²`, i.e. 15–42% too narrow as the curve gets more informative,
  bounded by `√m` (73% at m=3), zero where the curve is uninformative (q_halo). **Already visible in
  every generation's own numbers**: compositional calibration error is consistently 2.5–3× the base
  (0.029/0.048, 0.020/0.041, 0.022/0.053, 0.016/0.042) while compositional RMSE is *better* — more
  accurate + less calibrated = too narrow. That table is the cheapest referee figure.
  - **Fix = modality masking**, ported from `protoplan_sbi`'s `GroupedFlowMatching` with the base
    class swapped: `networks/grouped_diffusion.py::GroupedDiffusionModel` (compositional sampling
    lives only on `DiffusionModel`). Group-coherent condition dropout during training
    (`missing_modality_prob`, `random_mask(keep_one=True)`), plus **per-compositional-item masks**,
    which protoplan did not need. Evaluate `m+1` items — m stream-only + 1 curve-only — so each
    likelihood enters once. Exact, not approximate: streams and curve are conditionally independent
    given θ. `score` is the single sampling-time hook (`velocity` and the compositional score both
    delegate to it); `compute_metrics` does the training dropout.
  - **bayesflow upgraded in place** to the git rev protoplan uses (`c4c0a59`, 2.0.13) — PyPI 2.0.12
    has no `utils/masks.py`, no `DiffusionTransformer`, no `observed_condition_mask` plumbing.
    `requires-python` narrowed to `>=3.12,<3.14` (bayesflow's own floor). `_bf_patches.py` is STILL
    needed: the compositional summary-outputs reshape bug is unfixed upstream at that rev. Full
    suite green (170) after the upgrade.
  - **Two bayesflow gotchas found and handled**: (1) `_repeat_mask_kwargs_over_items` broadcasts one
    per-*row* mask to every item, so a per-item mask needs the rank-3 `(batch, items, width)` layout
    and its own flattening — overridden in `GroupedDiffusionModel`. (2) `_inverse_compositional`
    defaults `mini_batch_size = max(0.1*n, 2)`, i.e. **2 of 3 items** in every run to date; it
    subsamples items with `take_along_axis` on the conditions alone, so a plan's mask rows would
    stop matching their items. `compositional_score` forces it off while a plan is active.
  - **Config quartet (2 modalities, ancillary dropped per user)**: `adapter/stream_2modal`
    (`summary_variables=[sim_summary, vcirc_kms]`, `inference_conditions: [none]`),
    `model/summary_network/stream_fusion_2modal` (`head: null` — REQUIRED; a fusion head mixes the
    branches so a slice is not a modality), `model/stream_fusion_2modal`
    (`inference_network=grouped_diffusion`, `subnet: diffusion_transformer`,
    `missing_modality_prob: 0.25`), `eval/stream_compositional_masked` (`member_groups`,
    `extra_items`). Group widths are read off the fusion backbones' `summary_dim`
    (`build_inference_network(cfg, model_cfg)` + a `needs_model_cfg` flag), so a tuning trial that
    searches `summary_dim` stays correct with nothing restated.
  - **`j` is not a modality** (user decision): it is already channel −2 of the grid `sim_summary`,
    so it must not be a condition group. That needed a new `inference_conditions: [none]` sentinel
    (`adapter.NO_CONDITIONS`) — an empty list means "derive from the simulator", and there was no
    way to say "genuinely none" — plus `j` in `adapter.drop` so the stream-frame augmentations can
    still index by it before the adapter prunes it.
  - **Also wired**: `eval.observed_groups` masks ordinary (non-compositional) `sample` calls, which
    is the "one stream only" / "curve only" regime — applied in `_load_model`, so every eval path
    picks it up.
  - Verified end-to-end on CPU (64 flat rows + 8 groups, m200c simulator): train → masked
    compositional evaluate (log confirms "4 items = 3 members observing [sim_summary] + 1 item
    observing [vcirc_kms]") → curve-only evaluate. 22 new tests in `tests/test_grouped_diffusion.py`
    incl. the conjugate-Gaussian √m illustration; full suite 170 green, ruff clean.
  - **v4 does NOT reject on the rotation curve** (user, same session): the `vcirc_rejection` prior
    is off there, so the curve enters the posterior through exactly one likelihood factor and
    nothing else. The de-duplicated factorisation is then clean with no "the curve is also in the
    prior" asterisk — the opposite of the rejection-prior generations (2026-07-07 onward), where
    the truncation was an additional, separate use of the same data. Two knock-ons: `vcirc_rejection`
    is config-driven (absent = no-op), so no code change; and the 2026-07-10 finding that the jax
    full-covariance KDE prior score is unusable was diagnosed as a consequence of the *rejection-
    truncated* prior's near-degenerate covariance (condition number ~280, eigenvalues 0.004-0.006,
    "the rejection prior pinning r_Disk/z_Disk/a") — that diagnosis does not automatically carry
    over to an untruncated v4 prior, so re-check before reusing the diagonal-only conclusion.
  - **agama broke and was repaired**: `uv sync` rebuilt it against a GSL that lived in uv's temp
    build dir (`ImportError: libgsl.so.27`), and its own GSL download then failed. Rebuilt against
    the conda GSL 2.8 with an rpath; recipe recorded in `pyproject.toml` under `[tool.uv]`. Note
    `pytest.importorskip("agama")` turns this into skipped tests, not failures — a green suite does
    not prove agama works.
  - **Not done**: no v4 training run (dataset still in the making); `tuning/` has no
    `grouped_diffusion` search space yet; `misspecification.py` summaries are still computed with
    every modality observed.

- Session 2026-09-15 (v4 trained with the 2-modality masked model; MLP ablation prepared, not run):
  first training on the v4 dataset (`data_jarvis/data_agama_rnbody_ibata_m200c_v4_hydrabflow`,
  10^5 flat rows + `test_multistream_333.npz`, Ou+2024 19-radius `vcirc_kms`), using the gridded
  per-stream summary statistics (`stream_global_ibata_grid_v2`) and last session's
  `grouped_diffusion` modality masking. Runner `scripts/train_v4_2modal.sh` (train -> sim eval ->
  real eval, autocvd, `eval.batch_size=8` pinned). Run: `outputs/v4_2modal/default/`.
  - **Occupancy channels are a validity signal, not an observable** (user decision, mid-run): the
    first launch was stopped because the counts reached the network as features. `MaskedTimeSeries
    Transformer` now zeroes `n_track`/`n_vlos` right after they build the validity mask (the prep is
    factored into `_prepare`), so the network learns WHICH phi1 bins are real but never trains on a
    member count — which reflects the progenitor draw, not anything the Gaia catalogue measures.
    Regression test `test_masked_tst_never_sees_count_values` (counts 8/5 vs 400/91 -> identical
    summaries). Alternative rejected: `summary_include_occupancy: false` reinstates the 2026-07-28
    bias (an empty bin's substituted 0 IS the on-track phi2 value).
  - **Training**: 1000 epochs, batch 1024, ~1h35m on one GPU, `missing_modality_prob=0.3`.
    convergence.json clean (no NaN, final/best val_loss 1.07, best weights restored).
  - **Sim eval (333 groups)**: the mask plan is in the log — `4 items = 3 members observing
    [sim_summary] + 1 item observing [vcirc_kms]`. base RMSE **0.837** / calib **0.015**;
    compositional RMSE **0.810** / calib **0.048**. **The de-duplication did NOT fix the pooled
    calibration**: it is still ~3x the base, and it is concentrated — gamma 0.106, alpha 0.106,
    q 0.085 vs <=0.039 for every other parameter, i.e. the halo-SHAPE directions only. So the
    sqrt(m) over-counting was not the whole story; open question for the next session.
  - **Real Gaia**: q_halo **1.02** [0.76, 1.26] — between the raw-particle models' prolate ~1.3 and
    the summary-statistic models' oblate ~0.78; gamma 0.79, log10 M200 11.78, Sigma_Disk 1.24e9,
    r/z_Disk 3.12/0.301 kpc, rho_Bulge 9.9e10. `ln_cvprime` sits exactly on its prior mean 2.56 =
    prior-dominated. MMD 3.15 vs null 2.32, p_strat 0 — still flagged, but per-member Mahalanobis
    percentiles are **86 / 87 / 89** (Pal5/NGC3201/M68), far milder and far more EVEN than the
    98-100th of every earlier generation; M68 is no longer the outlier.
  - **GPU-memory gotcha**: the compositional stage OOM'd at the default `eval.batch_size=30` — 4
    items x 30 groups x 1000 samples asked for a single 24.18 GiB allocation in the adaptive
    stochastic integrator (`utils/oom.py` backoff is wired into train/tune, NOT evaluate). 8 works.
  - **MLP ablation (built + tested, never ran — no free GPU)**: `mlp` backbone now flattens rank-3
    input (no-op on rank-2), so it is a drop-in for `time_series_transformer` on both observables;
    new `masked_mlp` (subclass of the masked TST, shares `_prepare`) keeps the identical occupancy
    handling so the arm differs ONLY in architecture. Configs
    `model[/summary_network]/stream_fusion_2modal_mlp` (3x256, same summary_dim 32/62 so the
    compositional group widths are unchanged). Run with
    `MODEL=stream_fusion_2modal_mlp N_EPOCHS=300 RUNS_DIR=outputs/v4_2modal/mlp bash
    scripts/train_v4_2modal.sh`. When comparing, remember the transformer arm had 1000 epochs.
  - Figures published as an artifact gallery (all 12 plots + the per-parameter table).
- Session 2026-09-15 (particle-spray twin of the v4 rnbody dataset — generating): per user, the v4
  simulations rerun with the PARTICLE-SPRAY forward model to compare against the restricted-N-body
  set. `conf/simulator/stream_agama_spray_massloss_ibata_m200c_v4.yaml` inherits
  `stream_agama_rnbody_ibata_m200c_v4` wholesale (same potential/priors/Ou+2024 grid/1e4 particles/
  in-window storage cap/seeds) and changes: `name: stream_agama` (Chen+2024 spray, inherited
  `spray_method: chen`), `mass_loss: linear` down to the B&H18 present-day masses (identity
  `m_progenitor_final` 1.34e4/1.93e5/1.28e5 — the v4 `m_progenitor` priors are INITIAL masses, so a
  fixed-mass spray would be inconsistent; draws below the final mass degenerate to fixed mass), and —
  user decision — **`alpha_TwoPowerTriaxial_halo` and `rho_Bulge` pinned** (identity 1.0 / 9.93e10,
  their pre-v2 constants), so the inferred global set is 7 + the 6 locals (rnbody v4 inferred 9
  globals). NOTE this means the two datasets differ in the prior as well as the forward model; for a
  pure forward-model A/B un-pin them (`type: uniform`/`normal` as in v4). Pilot (24 rows): 0 NaN,
  ~100 s/row/worker (the time-dependent Plummer `scale` modifier makes mass-loss spray slower than
  the 2026-07-29 fixed-mass Cautun spray), in-window storage capped at 2000 in nearly every row
  (spray keeps far more stars in-window than rnbody did: 2000/480/1288 medians there). Launched
  `data_jarvis/data_agama_spray_massloss_ibata_m200c_v4_hydrabflow/logs/run_spray_v4_full.sh`
  (333 test set seed 7, then 1e5 training seed 2026; same headroom watchdog; log `spray_v4_full.log`).
  **Environment gotcha (cost ~30 min)**: a bare `uv run` recreated `.venv` on Python 3.12 (the old
  venv was 3.11, below `requires-python`) and agama failed to rebuild because uv's cached sdist kept a
  stale 3.11 `Makefile.local`; fixed with `uv cache clean agama && uv sync --frozen`. The venv is now
  3.12 with bayesflow 2.0.13 (the committed lock — the "2.0.12 pin" note in Stack is stale). Launch
  stages with `.venv/bin/python -m hydrabflow.pipeline.<stage>`, not `uv run`.

- Session 2026-09-16 (information-maximising single-modality summary nets, `imm_summarynet`): two
  CouplingFlow posteriors on the Palau spray v4 set (`data_jarvis/data_agama_spray_massloss_ibata_
  m200c_v4_palau_hydrabflow`), each conditioned on ONE modality with no modality dropout, so the
  summary net is forced to extract all it can — to be frozen later inside the 2-modality score model.
  New registry entry `coupling_flow` (`networks/factory.py`, `bf.networks.CouplingFlow`, config
  `inference_network/coupling_flow.yaml`). Arms: `model=imm_vcirc` (`time_series_transformer`,
  adapter `stream_vcirc_only`, aug `stream_global_vcirc_only` = add_noise_to_vcirc + log10_vcirc)
  and `model=imm_streams` (`masked_time_series_transformer`, adapter `stream_streams_only`, aug
  `stream_global_sumstats_only` = grid_v2 minus the potential-observable steps); both summary_dim 32.
  Runner `scripts/train_imm_summarynet.sh` (ARM=vcirc|streams; train → sim eval), outputs
  `outputs/imm_summarynet/{vcirc,streams}/`. Infra: `build_workflow` falls back to `BasicWorkflow`
  when the inference net is not a `DiffusionModel` (bf's CompositionalWorkflow rejects anything else;
  training is identical and composition=global still derives the global inference variables);
  `evaluate` at composition=global skips the compositional stage when the workflow has no
  `compositional_sample` (base_* metrics only); `train` saves `summary_network_weights.npz`
  (summary nets are keras Layers — `get_weights`/`set_weights`, no `save_weights`). Gotchas: a
  single-key adapter with `attention_mask_key` routes the STAR-level mask (n_stars) into the gridded
  net (K bins) → shape error, so the streams adapter drops `attention_mask` instead; the compositional
  evaluate reads m off the raw `j` array, so every single-modality adapter must keep `j` in `drop`.
  Both arms smoke-tested (1 epoch on the 1e5 set + eval); ~100-110 s/epoch at batch 512 on a shared
  card during the smoke (compile-inclusive first epoch — recheck on the real run). Per user, launched
  via `outputs/imm_summarynet/launch_when_free.sh` (autocvd waits for a FREE GPU, arms staggered by
  240 s); logs `outputs/imm_summarynet/{vcirc,streams}.log`. Not committed.

- Session 2026-09-16 (legacy 1e6 agama set re-gridded onto the Ou+2024 curve + v4_2modal_legacy
  MLP training): the reference-project spray set `data/streams/data_agama/training_data_local_
  1000000.npz` (1e6 rows, 1000 particles, base `stream_agama` priors, NO `vcirc_kms` — the old
  project kept the curve as a separate array) copied to `data/data_jarvis/data_agama_spray_legacy_
  ou24_hydrabflow/` with `vcirc_kms (1e6,19,1)` RECOMPUTED with AGAMA on the Ou et al. 2024 grid
  (`_host_potential(pot_cfg=None)` IS the legacy potential: the stored bulge columns equal
  `BULGE_PARAMS`). New `scripts/recompute_vcirc_grid.py` (reuses `extend_vcirc_huang.recompute_vcirc`,
  radii read from a simulator yaml) + `conf/simulator/stream_agama_ou24.yaml` (base priors +
  `obs_r_grid: custom` v4 table, so `fill_stream_grid_from_simulator` propagates 19 radii/sigma/obs).
  1e6 rows: 32 nice'd workers, ~13 min + 24 GB write, 0 NaN curves; test set = the merged
  `data_agama_hydrabflow/simulation_multistream_333.npz` re-gridded 34->19 (`test_multistream_333.npz`;
  interpolated old curve agrees to 0.07 %). Dataset-median curve is ~11 % off the observed one
  (broad legacy prior, expected). No ancillary observables stored (user). Blocker for the v4 stack:
  `stream_global_log10_ibata_sumstats` pins `drop_nan.keys` to `vterm_kms` -> new preset
  `stream_global_log10_sumstats_2modal` (sim_data_projected + vcirc_kms only).
  `train_v4_2modal.sh` gained `PY`/`AUTOCVD` overrides (use `.venv/bin/python`, not `uv run`).
  **Launched** `outputs/v4_2modal_legacy/` (launch.sh -> run.log): `model=stream_fusion_2modal_mlp`
  (MLP summary nets, per user), `simulator=stream_agama_ou24`, 300 epochs, batch 1024, GPU 6. Box
  note: `vm.overcommit_memory` is now 0 (heuristic), so the 24 GB set loads without the 2026-09
  commit-headroom dance.
  **NaN at epoch 1 (batch 113) — root cause + fix**: `log10_vcirc` was a bare `jnp.log10`; the
  legacy halo prior reaches v_c ~ 40 km/s at 25 kpc where the Ou sigma is 17 km/s, so
  `add_noise_to_vcirc` drives ~11 rows per 1e6 negative -> NaN loss -> TerminateOnNaN (the runner
  then wasted a GPU evaluating the dead model; kill by PID list — `pkill -f <pattern>` matched the
  calling shell). Fix in `augmentation/streams.py`: `log10(max(v, vcirc_floor_kms=1.0))`; test
  `test_log10_vcirc_floors_negative_noisy_bins`. Verified the whole train chain on a 20k slice
  (forced near-zero curves) produces only finite outputs; `drop_nan` removes 5.7 % of the legacy
  rows (NaN particles). Failed attempt archived under `outputs/v4_2modal_legacy/failed_nan_attempt/`.
  **Results (relaunch, GPU 6, 300 epochs ~3.5 h, ~40 s/epoch; `outputs/v4_2modal_legacy/{train,
  eval_sim_333,eval_real}`)**: convergence clean (final/best val_loss 1.07, best ~1.30). Sim eval
  (333 groups, 7 legacy globals): base RMSE 0.711 / calib 0.013; compositional (3 streams + 1 curve
  item) RMSE 0.705 / calib **0.063** — pooled calibration again ~5x the base and again concentrated
  in the halo (rho 0.111, a 0.085, q 0.072, gamma 0.066) while the disk stays calibrated (r/z_Disk
  0.016/0.018): same pattern as the v4 rnbody run, so it is not specific to that dataset or to the
  transformer nets. Per-stream recovery: gamma/a/Sigma_Disk r~0.8, rho/r_Disk ~0.5, q 0.30 (near
  prior), z_Disk ~0. Real Gaia global: q_halo **1.02 [0.74, 1.27]**, gamma 0.34 [-0.74, 1.17],
  a 7.5 kpc, rho 5.2e7, Sigma_Disk 1.14e9, r_Disk 3.32, z_Disk 0.298. MMD 2.83 (null 95 % 2.75),
  p_plain 0.04 / p_strat 0.025 — the mildest flag of any generation; Mahalanobis pct Pal5 73 /
  NGC3201 95 / M68 73 (NGC3201, not M68, is the outlier here).
  **Why q_halo is unconstrained (diagnosed on these posteriors)**: per stream the q posterior std
  equals the prior std (0.288/0.289/0.272 vs 0.289 for Pal5/NGC3201/M68), the pooled one is 0.232
  and its median sits at ~1.0 whatever the truth (bias +0.29 / -0.29 for oblate / prolate truths) —
  shrinkage to the prior mean, i.e. an uninformative summary, NOT a degeneracy: within-posterior
  |corr(q, other)| < 0.1 and 230/299 test groups are halo-dominated inside 20 kpc. Consistent with
  the 2026-09-12 sensitivity numbers (q 0.8->1.0 moves the phi2 track 0.015/0.18/0.89 deg for
  Pal5/NGC3201/M68 vs ~0.1 deg realization scatter): only M68 carries a detectable q signal. The
  pooled halo-only miscalibration follows: three near-prior factors divided by the prior squared.
- Session 2026-09-16 (particle-SetTransformer twin of v4_2modal_legacy — running): same data/epochs
  with the raw star cloud as the stream modality. New presets: `adapter/stream_2modal_particles`
  (`[sim_data_projected, vcirc_kms]`, `j` dropped — it is channel 14 of the particles), `model[/
  summary_network]/stream_fusion_2modal_particles` (`masked_set_transformer` with the model5_maskedvlos
  hyperparameters + the 2modal vcirc TST, summary_dims 32/62 kept, `mask_backbone: sim_data_projected`,
  `head: null`), `eval/stream_compositional_masked_particles` (`member_groups: [sim_data_projected]`);
  augmentation = plain `stream_global`/`stream_real_global` with `vlos_impute=zero`; real preproc
  `stream_real_global_log10`. `train_v4_2modal.sh` gained `EXTRA` (extra Hydra overrides for all 3
  stages). CPU smoke (600 rows/6 groups) passes with the plan "3 members observing
  [sim_data_projected] + 1 item observing [vcirc_kms]". **Batch 1024 OOMs** on a 40 GB card (one
  15.6 GiB attention buffer) and `run_with_oom_backoff` did NOT rescue it — every retry down to 16
  died within a minute with the same error (the failed attempt's device memory stays pinned inside
  the same process; the backoff needs a fresh approximator/`jax` state to be useful). Measured on a
  20k slice: batch 512 peaks at 13.3 GB, ~0.1 s/step. Launched at 512:
  `outputs/v4_2modal_legacy_particles/` (launch.sh -> run.log; failed attempt in `failed_oom_b1024/`),
  ~150 s/epoch -> ~12.5 h for 300 epochs. Epochs 1-2 val_loss 1.85 -> 1.58 (MLP arm: 1.89 -> 1.63).
  **Results (`outputs/v4_2modal_legacy_particles/{train,eval_sim_333,eval_real}`, 300 epochs 12.5 h,
  best val_loss 1.03, convergence clean, final/best 1.04)**: sim base RMSE **0.480** / calib **0.008**
  (MLP-sumstats arm 0.711 / 0.013); compositional RMSE **0.453** / calib 0.057. **The stars DO carry
  q_halo**: per-stream q RMSE 0.138 (sumstats 0.888), pooled 0.084 with posterior std 0.026 and
  corr(median, truth) 0.96 — so the 2026-09-16 "q signal below the noise floor" ranking was wrong:
  the information is in the star cloud and the binned summary statistics discard it (the 10-bin
  medians/dispersions in a frame fitted to the REAL members). Pooled calibration is again ~7x base
  and again halo-shape-only (rho 0.092, gamma 0.099, a 0.093; q itself 0.028, disk 0.02) — same
  pattern in a third architecture. **Real Gaia**: q_halo **1.47 [1.43, 1.49]** RAILS at the prior
  edge 1.5 (per stream Pal5 1.45, NGC3201 1.49, M68 1.32 [1.01, 1.46]) — the raw-particle prolate
  result of every earlier generation, now sharper; gamma 0.61, per-stream gamma DISAGREE (Pal5 1.03,
  NGC3201 -0.22, M68 1.52); a 6.4 kpc, rho 6.3e7, Sigma_Disk 8.9e8, r/z_Disk 3.36/0.30. MMD 2.90
  (null95 2.70), p 0.015, all three members at the 100th Mahalanobis pct (sumstats arm: 73/95/73).
  Representation summary on the SAME data: sumstats-MLP q 1.02 ± 0.26 (uninformative), particles
  1.47 railing + all members OOD ⇒ the particle model is more informative in-sim but extrapolates on
  the real streams (the recurring finding, now with a matched-dataset control).
- Session 2026-09-17 (why trial_1 of `stream_ibata_grid_m200c_median_study` recovers q_halo, and the
  2modal ablation): the rejection prior is NOT the reason — in the cut 3e5 m200c set the q marginal
  is flat (mean 1.007, std 0.288 vs 1.0/0.289 uniform), |corr(q, other globals)| <= 0.08, and a
  gradient-boosted regressor predicting q from the potential-only observables (log vcirc, vterm,
  sigma_z) reaches the same normalized RMSE with and without the cut (0.85 cut / 0.88 uncut v4;
  vcirc alone 0.97/0.99). trial_1 gets q 0.388 base / 0.240 pooled (normalized by prior std; per
  stream 0.46/0.35/0.48 Pal5/NGC3201/M68, corr(median,truth) 0.88/0.94/0.88 — NGC3201 is the best
  carrier). **Ablation** `outputs/ablation_2modal_m200c_3e5/test/` (`model=stream_fusion_2modal`,
  grid_v2 masked TST + curve TST, no v_term/sigma_z, grouped diffusion with dropout 0.3, SAME 3e5
  data / 400 epochs / batch 1024): q collapses to **0.89 base / 0.79 pooled** (Pal5/NGC3201 corr
  0.23, M68 0.65); every other parameter within ~0.1 of trial_1. Real Gaia q 0.93 [0.68,1.23] =
  prior-like (trial_1: 0.56 [0.54,0.60], but all members at the 99.7-100th MMD pct there, so that
  tightness is extrapolation). Conclusion: the q information is lost in the INPUT STACK, not the
  dataset — one or more of (a) dropping the ancillary observables, (b) the grid_v2 estimator
  (out-of-range stars excluded, MAD, occupancy zeroed), (c) grouped-diffusion modality dropout. The
  old grid (`stream_global_ibata_grid`, 12/7 ch) + `stream_fusion_ibata_grid` + ancillary keeps q.
  Not split further yet: next run = `model=stream_fusion_ibata_grid adapter=stream_ibata_sumstats`
  with `augmentation=stream_global_ibata_grid_v2` on the same set (isolates the grid estimator).
- Session 2026-09-17 (TimeSeriesTransformer twins of the two MLP 2-modal runs): `model=stream_fusion_2modal`
  (masked TST over the gridded summary statistics + TST over the Ou+2024 curve), everything else identical to
  the MLP references. `outputs/v4_2modal_palau_tst/` (palau spray 1e5, 1000 ep; real MMD 3.24 p=0, members at
  the 92/95/96th pct) and `outputs/v4_2modal_legacy_tst/` (legacy spray 1e6, 300 ep, ~29 s/epoch). **Legacy
  three-arm comparison on the SAME data** (sumstats-MLP / sumstats-TST / particles-SetTransformer): sim base
  RMSE 0.711 / 0.711 / 0.480, calib 0.013 / 0.013 / 0.008; compositional RMSE 0.705 / 0.701 / 0.453, calib
  0.063 / 0.060 / 0.057. Per-parameter the two summary-statistic arms agree to <0.01 RMSE everywhere and both
  leave q at prior width (q RMSE 0.89 vs 0.14 for particles) — **the architecture over the binned summaries
  is irrelevant; the information loss is in the summary statistics themselves.** Real Gaia: MLP q 1.02
  [0.74,1.27], TST 0.98 [0.72,1.27] (per stream 0.96/0.99/1.03), particles 1.47 railing; every other global
  agrees between the two sumstats arms to ~1 %. TST MMD 2.78 p_strat 0.01, members 72/92/78 (MLP 73/95/73;
  particles 100/100/100). Halo-only pooled miscalibration (rho/gamma/a ~0.1, disk 0.02) reproduced in all
  three, as before.
- Session 2026-09-18 (compositional_bridge_d1 sweep on the oldgrid model + v4 palau oldgrid run):
  `scripts/sweep_bridge_d1.py` (Hydra app via `make_cli`; compositional stage ONLY, model loaded
  once, `D1_VALUES` env var, `d1_<v>/compositional_*` + `sweep_summary.json`, resumable). Swept
  d1 in {0.05,0.1,0.15,1/6,0.2,0.25,0.333} on `outputs/v4_2modal_legacy_oldgrid` (table +
  verdict in its `sweep_d1/README.md`). **No d1 calibrates the pooled posterior**: the bridge is
  one scalar damping while the miscalibration is direction-specific — d1=0.05 makes the halo
  rho/gamma/a near-perfect (0.009/0.016/0.011) but the disk r/z under-confident (0.19/0.21);
  raising d1 does the reverse. Mean optimum flat at 0.15–1/6 (0.0627/0.0625), so the default 1/6
  stays and the existing `eval_real` IS the best-calibrated real run (not re-run). `mini_batch_size`
  removed from `conf/eval/stream_compositional_masked.yaml`: bayesflow 2.0.13 forces it to
  num_items on JAX ("jax does not support mini-batching") and the mask plan disabled it anyway —
  a pure no-op. **Runner bug fixed** (`train_v4_2modal.sh`): `eval "$(autocvd -n 1)"` sets an
  UNexported shell variable, so `utils/backend.py`'s own autocvd call waited "indefinitely" for a
  free GPU whenever none was — added `export CUDA_VISIBLE_DEVICES` (or pass `-e`). Note
  `autocvd -l` (least-used) to share a card. **v4 palau + oldgrid** (`outputs/v4_2modal_palau_oldgrid`,
  = palau_tst but `augmentation=stream_global_ibata_grid`, `model=stream_fusion_2modal_oldgrid`,
  1000 ep, 1.48 h): sim base RMSE 0.593/calib 0.017, compositional **0.568/0.029** (palau_tst
  grid_v2: 0.738/0.012 and 0.725/0.043) — the old grid is more informative on every parameter AND
  the pooled calibration is the best of any 2-modal run (3x below the legacy oldgrid's 0.0625, so
  the halo-only pooled miscalibration is dataset/prior-dependent, worst on the broad legacy prior).
  Real Gaia: q_halo **1.28 [1.19,1.38]** (per stream 1.35/1.26/1.32, coherent; grid_v2 gave a
  prior-like 1.04 [0.75,1.29]), gamma 0.98, log10 M200 11.7, ln c' 2.47, Sigma/r/z_Disk 9.14/0.49/
  -0.52 (log10). MMD 2.72, p_strat 0.015, members 99.4/98.4/100th pct (grid_v2: 92/95/96).
  Uncommitted.
- Session 2026-09-18 (Optuna study on the 2-modal oldgrid model, palau v4 set — RUNNING): per user,
  `conf/tuning/stream_2modal_oldgrid.yaml` (study `stream_2modal_oldgrid_study`; both TST backbones:
  summary_dim/num_blocks/num_heads/embed_dim_per_head/dropout; DiffusionTransformer subnet:
  dt_width{64..256}/dt_num_layers/dt_num_heads{2,4,8}/dt_time_embedding_dim/dt_dropout), 1000
  epochs/trial, 50 trials, **missing_modality_prob pinned 0.5**, objectives = base RMSE + calib on
  the val split (`eval.batch_size=256 num_samples=500`: at 32 the 12k-row val scoring alone took ~70
  min/trial, now ~20). One worker on GPU 6: `scripts/tune_2modal_oldgrid.sh` (~1.5 h train + 20 min
  score per trial ⇒ ~4 days). Companion **`scripts/tune_post_eval.sh`** polls the trials dir and,
  for every finished trial, runs the full `evaluate` twice — pooled compositional on the 333 test set
  (`trial_XXXX/eval_sim_333/`) and real Gaia (`trial_XXXX/eval_real/`: corner, MMD, pairs) —
  re-expressing the sampled hyperparameters from the new per-trial `params.json` (dumped by
  `tune.py`) as Hydra overrides; the shared preprocessing state is symlinked into each trial dir.
  Study + trials under `data_jarvis/data_agama_spray_massloss_ibata_m200c_v4_palau_hydrabflow/tuning/
  stream_2modal_oldgrid_study/`; logs `outputs/tuning_2modal_oldgrid/{worker,post_eval}.log`.
  **Bug fixed (affected every earlier tuning study)**: the inline `tuning.search_space` in
  `conf/config.yaml` dict-MERGED into any selected `tuning=` preset, so the Ibata studies silently
  also searched `training.learning_rate` + unused top-level `summary_network.*`/`mlp_*` fields.
  Moved it to `conf/tuning/default.yaml` (`- tuning: default`): a group file is replaced wholesale.
  `params.json` of the smoke trial is what exposed it. Also: `scripts/corner_modalities.py` (real
  Gaia: per-stream with curve masked via `eval.observed_groups=[sim_summary]`, curve-only, and the
  compositional pooling; palau oldgrid: q from the streams 1.22-1.36, curve-only q ≈ prior, pooled
  1.28; Sigma_Disk is the modality tension — streams at the 1.5e9 ceiling vs curve 1.25e9).
  Shell gotcha (bit three times): `pkill -f`/`pgrep -f <pattern>` match the calling `bash -c`
  itself → exit 144; anchor the pattern (`^\.venv/bin/python -m ...`) or kill by PID.
  Follow-ups same session: the watcher's overrides must be `++path=value` (struct mode rejects the
  undeclared `dt_time_embedding_dim`/`dt_dropout`; `tune.py` adds them with `force_add`) — first
  2 h of watcher polls failed on that, fixed and restarted. `scripts/tuning_pareto.py <study_dir>
  --out outputs/tuning_2modal_oldgrid` collates the Optuna journal + per-trial eval_sim_333 /
  eval_real results into `trials.csv` and plots `pareto_val.png` (Optuna objectives, front
  highlighted), `pareto_compositional.png`, `real_q_vs_calib.png`; re-run any time. Trials 0/1:
  val RMSE 0.616/0.604, calib 0.006, ~2 h each.

- Session 2026-09-20 (gala particle-spray twin in a MilkyWayPotential2022-style Galaxy —
  `stream_gala`, 10^4 x 10^4-star dataset + prior-predictive coverage in three representations):
  a controlled twin of the v4 spray set on a different forward-model stack, asked for as "q in the
  potential, prior flat in the derived density q". **Physics** (derivation in-session): flattening
  the potential Phi(sqrt(R^2+z^2/c^2)) maps to a density axis ratio by the l=2 quadrupole match
  `(1-q_rho)/(1-q_Phi) = (f''+2f'/r-6f/r^2)/(f''-2f/r^2)`, f=v_c^2 (=(5-gamma)/(3-gamma) for a
  power law, the classic 3 at gamma=2); for NFW it runs 2.0 (r<<r_s) -> 2.3 (r_s) -> 4 (10 r_s),
  so the factor-3 rule over-converts by 20-40 % at 5-30 kpc. The potential-flattened NFW has
  NEGATIVE density on the pole for c_Phi < 0.888 (q_rho < 0.78), but only at z > 36 kpc (c=0.75)
  — accepted by the user, recorded per row (`halo_rho_neg_r_kpc_derived`).
  - **gala 1.11.0 added as a dependency** (`uv add gala`; agama unaffected; coexists with JAX in
    one process in both import orders, still imported worker-only). `NFWPotential.from_M200_c`
    uses astropy's default Planck18 rho_crit (127.05 Msun/kpc^3; MW2022 halo == M200 9.6e11,
    c200 13.3) and returns a spherical halo, so `stream_gala` redoes its algebra in numpy and
    builds `NFWPotential(m, r_s, c=c_phi)`.
  - **`src/hydrabflow/simulators/stream_gala.py`** (`@register_simulator("stream_gala")`,
    subclass of `AgamaStreamSimulator`, overrides ONLY `_row_jobs`): MW2022 family = fixed
    Hernquist bulge/nucleus + free MN3 exponential disk (`m_disk`, `h_R_disk`, `h_z_disk`) + NFW
    from (`log10_M200_halo`, `c200_halo`); **`q_rho_halo` ~ U[0.5,1.5] is inferred and c_Phi is
    derived per row** by brentq on the analytic flattened-NFW Laplacian at `q_ref_r_kpc=15`
    (stored `c_phi_halo_derived`, `m_nfw_halo_derived`, `r_s_halo_derived`); Chen+2024 spray via
    gala `ChenStreamDF` + `MockStreamGenerator` with a Plummer progenitor (30 pc) and linear mass
    loss through gala's time-varying `prog_mass` (index 0 = past, length n_steps+1); astropy default
    Galactocentric frame both ways (solar frame NOT varied, unlike v4). Generic-diagnostics
    refactor: the worker's slot 3 is now a `{name_derived: value}` dict (`_m200c_derived` for the
    agama m200_c halo) that `simulate` stores `(n,1)` and `sample_compositional` regroups per
    potential for every `*_derived` key.
  - **gala gotchas (cost ~1 h)**: (1) gala releases `2*(n_steps+1)*n_particles_per_release`
    stars -> `n_steps: 4999` for 1e4 (class checks the identity). (2) gala's default dop853
    steps ALL particles in ONE shared adaptive call per time step (mockstream.pyx), so one star
    through the Plummer core collapses the step -> `Integration failed with code -4` in ~15-40 %
    of 1e4-star NGC3201/M68 rows, the stock MilkyWayPotential2022 included, independent of
    tolerances/seed/progenitor potential. **Default `integrator: leapfrog`** (dt = t_end/n_steps
    ~0.4-2 Myr): 0 failures in 48+24+999+... rows, agrees with dop853 to 6 pc median / 40 pc 90th
    pct per particle after 4 Gyr (<0.06 deg on the in-window track), and is faster (~15 s/row vs
    ~30). (3) the Chen DF release radius N(1.6,0.35) r_j has no floor -> ~2e-6 NaN stars; the
    worker re-draws up to 0.1 % of them from the finite ones.
  - Configs: `conf/simulator/stream_gala_spray_mw22.yaml` (standalone; 6 inferred globals in
    prior order, identity bulge/nucleus, v4 spray_massloss locals verbatim, v4 window cap/Ou+2024
    grid); `conf/preprocessing/stream_global_log10_gala_2modal.yaml` +
    `stream_real_global_log10_gala.yaml` (log10 on the three disk params). Adapters/augmentations/
    models are the 2-modality presets unchanged. Tests `tests/test_stream_gala.py` (11) + the
    registry test; full suite green; ruff clean on the new files.
  - **Dataset** `data_jarvis/data_gala_spray_mw22_hydrabflow/`: pilot (24 rows: 0 NaN, 21 s wall
    at 24 workers, 473 MB parent RSS, in-window medians 2000/1988/2000 capped 100/50/83 %),
    `test_multistream_333.npz` (seed 7, 48 workers, ~4 min, 48 MB), `training_data_10000.npz`
    (seed 2026, 48 workers, 52 min, 484 MB: 0 NaN rows, streams 3308/3272/3420, in-window medians
    2000 capped 100/60/70 %, c_phi in [0.739, 1.22], 3267 rows with a negative outer-halo density) via `scripts/create_gala_mw22_dataset.sh` (pilot ->
    test -> train -> PPCs -> prints the two `train_v4_2modal.sh` commands; PILOT_ONLY=1 stops
    after the pilot). Half the rows carry the negative outer-halo density (min radius 36 kpc).
  - **New `scripts/ppc_particle_coverage.py`** = the raw-PARTICLE-representation twin of
    `ppc_summary_grid_coverage.py`: sims through the training observation model, real members
    through their preset, both in the real-fitted stream frame; per-observable quantile envelopes,
    2-D overlays, and an RBF-MMD coverage rank (real-vs-sim MMD^2 vs the sim-vs-sim null; 100 =
    real farther from every sim than sims from each other) for all / sky / pm / vlos / parallax.
  - **Prior-predictive coverage on the 333 set (`<dataset>/ppc/`)** — the "are we still
    misspecified?" answer, prior-level, before any training:
    * Summary GRID (network input, 14 ch): Pal5 97 % of cells inside the central 95 %, NGC3201
      84 %, M68 78 % (v4 rnbody: 97/89/93). Same signatures as v4 — NGC3201 occupancy channels at
      98-100th pct in bins 1-8 (real footprint shorter than the sims), M68 every dispersion
      channel (std_phi2/mu_phi2/vlos) at 97-100 (sims too cold: noise-convolved std_phi2 0.67 vs
      1.38 deg real, P(sim<real)=1.00) — PLUS a new one: **NGC3201 med_mu_phi1 at the 92-100th
      pct in 9/10 bins** (the whole simulated mu_phi1 track runs below the real one by ~3-5 mas/yr
      at d=4.9 kpc, ~100 km/s — a potential/orbit effect, not the solar frame). Track PPC
      (estimator) 98/89/89 % inside 95 %.
    * PARTICLES (MMD rank all/sky/pm/vlos): Pal5 **52/30/41/70** — indistinguishable from the sims
      (as the agama v4 rnbody 51/39/42/70 and spray 56/41/43/76 baselines run with the SAME
      script). NGC3201 **100/69/100/100** (rnbody baseline 100/86/96/93, spray 100/93/100/98):
      OOD in every family; in gala the mismatch is pm+vlos (the mu_phi1 offset above) while the
      sky is the least discrepant. M68 **100/100/100/32** vs rnbody **78/65/76/37** and spray
      99/97/99/28: the rnbody v4 set is the ONLY family where M68's star cloud is not OOD (its
      restricted-N-body heating gives the width; both spray sets are too cold on sky+pm).
    * Verdict: the gala/MW2022 potential family does not remove the misspecification; it
      reproduces the two known ones (NGC3201 footprint/occupancy, M68 too cold — spray-intrinsic,
      cf. 2026-07-29) and adds an NGC3201 proper-motion track offset. Pal 5 is well covered in
      every representation and every family. Training the two arms (commands printed by the
      script; `RUNS_DIR=outputs/gala_mw22_2modal/{sumstats,particles}`) is the next step, not run.

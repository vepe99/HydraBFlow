# HydraBFlow: SBI Pipeline Template with BayesFlow 

## Goal

A reusable, cookiecutter-style repository for setting up Simulation-Based Inference (SBI)
pipelines using BayesFlow and Hydra. The template handles all infrastructure (training,
inference, dataset generation, experiment tracking) so that a new user only needs to:

1. Write their simulator (forward model)
2. Choose and configure their SBI components (summary network, inference network, etc.)

Everything else — config management, output tracing, reproducibility — is fixed infrastructure.

## Core Design Principles

- **Full traceability**: every run (training, inference, dataset generation) must save its
  Hydra config to the output directory. A run is only valid if it can be fully reconstructed
  from its output folder.
- **Hydra-native**: all entry points are Hydra apps. No argparse. Config composition via
  config groups covers all axes of variation (model, simulator, training, data).
- **Modularity via structured configs + registries** (NOT `_target_`): every config group has a
  typed dataclass schema registered in Hydra's `ConfigStore`; YAML files fill in values. Factory
  functions read those dataclasses and resolve names through registries (`networks.factory`,
  `simulators.registry`, `preprocessing.registry`, `augmentation.registry`, `pipeline.adapter`).
  Components self-register by name (`@register_simulator`, `@register_step`,
  `@register_augmentation`, `@register_summary_network`, `@register_inference_network`), and each
  package auto-imports its modules (`utils.discovery`), so adding a component = dropping a file +
  a config entry, no infrastructure edits (not even `__init__.py`).
- **The simulator is the single source of truth for variable names**: empty
  `adapter.inference_variables` / `summary_variables` are derived from the simulator's
  `parameter_names` / `observable_keys` at CLI entry (`pipeline.adapter.fill_adapter_from_simulator`).
  Explicit adapter config overrides (required for bring-your-own-data, where no class exists).
- **Separation of concerns**: infrastructure code (training loop, logging, checkpointing)
  is never modified by the end user. User-facing code lives in clearly marked locations
  (`src/hydrabflow/simulators/`, plus optional custom `networks`/`preprocessing`/`augmentation`).
- **Single-level inference by default, compositional as an opt-in level** (stream_project
  branch): `composition=none` keeps the original `bf.BasicWorkflow` path. `composition=global` /
  `composition=local` switch to `bf.CompositionalWorkflow` and train/evaluate one level of a
  hierarchical simulator (global parameters shared by exchangeable group members vs per-member
  local parameters). The simulator declares the split (`global_parameter_names`,
  `local_parameter_names`, `context_keys`, `sample_compositional`); the adapter derivation
  follows `composition.level`. Evaluation: global = `compositional_sample` with the simulator's
  prior score; local = per-member sampling on simulated data and `ancestral_sample` on real data
  (globals drawn from a saved global posterior, `composition.global_run_dir`).
- **Preprocessing vs augmentation are distinct stages**: preprocessing is deterministic,
  whole-dataset, applied once and fitted on the train split (`src/hydrabflow/preprocessing/`);
  augmentation is stochastic and per-batch, applied inside `fit_offline`
  (`src/hydrabflow/augmentation/`).

## Tech Stack

- **SBI framework**: BayesFlow 2.x (Keras 3)
- **Compute backend**: JAX. `KERAS_BACKEND=jax` is pinned by `hydrabflow.utils.backend` (imported
  first via `hydrabflow/__init__.py`) before any keras/bayesflow import. Override via env var.
- **Packaging / env**: `uv` (`pyproject.toml`, src-layout, console scripts `hydrabflow-*`).
- **Config management**: Hydra with structured dataclass configs (`ConfigStore`) + config groups.
- **Neural architectures**: SetTransformer / DeepSet / TimeSeriesTransformer (summary network),
  FlowMatching / DiffusionModel (inference network) — user-swappable via config. Summary defaults
  to a single observable; multi-observable FusionNetwork is a documented seam in
  `pipeline.adapter` + `networks.factory`.
- **Hyperparameter tuning**: Optuna (multi-objective: RMSE + calibration error).
- **Notebooks**: Marimo (`notebooks/explore.py`).

## Folder Structure (finalized)

HydraBFlow/
├── pyproject.toml               # uv-managed; deps + console scripts (hydrabflow-*)
├── conf/                        # Hydra config groups (YAML values; schemas live in code)
│   ├── config.yaml              # Root: defaults list, seed, model_dir, hydra.run.dir
│   ├── simulator/               # skeleton.yaml (+ your simulators)
│   ├── model/                   # default.yaml -> summary_network/ + inference_network/
│   ├── training/  data/  preprocessing/  augmentation/
│   ├── adapter/   inference/    eval/   tuning/
├── src/hydrabflow/
│   ├── config/schema.py         # ALL dataclass schemas + register_configs()
│   ├── simulators/              # USER MODIFIES: base.py, registry.py, skeleton.py
│   ├── networks/factory.py      # build_summary_network / build_inference_network
│   ├── preprocessing/           # base, standardize, steps, registry (deterministic, once)
│   ├── augmentation/            # base/registry + examples (stochastic, per-batch)
│   ├── pipeline/                # INFRASTRUCTURE: adapter, workflow, io, checkpoint,
│   │                            #   simulate, train, evaluate, evaluate_real, tune, _app
│   └── utils/                   # backend (JAX pin), seed, logging, paths
├── scripts/                     # thin Hydra entry points -> pipeline.<stage>.cli
│   ├── simulate.py  train.py  evaluate.py  evaluate_real.py  tune.py
├── tests/                       # config-compose, registries, preprocessing, workflow smoke tests
├── notebooks/explore.py         # Marimo
├── outputs/                     # Hydra run dirs (gitignored)
└── CLAUDE.md

### Run stages (6 entry points)
- `simulate`  — sample prior + run forward model in chunks -> aggregated `.npz`.
- `simulate_multistream` — compositional datasets: one shared global draw per row, one
                observation per group member (`sample_compositional`) -> grouped `.npz`
                (globals `(n,1)`, member arrays `(n,m,...)`); used by compositional evaluation.
- `train`     — load `.npz` -> preprocessing (fit on train, save state) -> `fit_offline` with
                augmentations -> save approximator + loss curve.
- `evaluate`  — load model + preprocessing state from `model_dir`, sample posterior on a
                simulated test set, write truth-aware diagnostics (RMSE/calibration, recovery,
                calibration ECDF, z-score contraction).
- `evaluate_real` — same, but on a user-provided real-data `.npz` (no truth, no resimulation).
- `tune`      — Optuna multi-objective study over a config-driven search space.

## What the User Modifies

- `conf/simulator/<name>.yaml` + `src/hydrabflow/simulators/<name>.py`: the forward model
  (a `@register_simulator`-decorated `BaseSimulator` subclass; auto-imported, self-registers).
- `conf/adapter/*`: normally untouched — variables derive from the simulator. Explicit config
  only for bring-your-own-data or to override the derivation (subset inference, fusion).
- `conf/model/...`: choose/configure summary + inference networks.
- Optionally: custom preprocessing steps, augmentations, or network architectures (drop a module
  in the package; each self-registers; no infra edits).
- Nothing else should need to change for a new problem.

## What Is Fixed Infrastructure (do not modify)

- Entry point scripts (`scripts/`) and the `pipeline.*.cli` wrappers (`pipeline/_app.py`).
- The five run stages, adapter/workflow builders, IO, checkpointing (`src/hydrabflow/pipeline/`).
- Config schema + registration (`src/hydrabflow/config/schema.py`).
- Hydra output directory setup and config saving; JAX backend pin (`utils/backend.py`).

## Output Directory Convention

Hydra's `hydra.run.dir` is set to:
`outputs/${simulator.name}/${model.name}/${now:%Y-%m-%d_%H-%M-%S}`

Every run saves:
- `.hydra/` folder with full config (Hydra does this automatically)
- `simulate`: dataset `.npz` in `data.data_dir`, plus a `<dataset_stem>.hydra/` config snapshot
  next to it (copied from Hydra's `.hydra/` via `utils.paths.save_config_snapshot`) so each
  dataset is traceable to the config that generated it. Keyed by the dataset filename so
  training and test sets in the same `data_dir` don't overwrite each other's snapshot.
- `train`: `approximator.keras`, `preprocessing_state.npz`, `loss.png`
- `evaluate`: `posterior.npz`, `metrics.json`, diagnostic plots
- `evaluate_real`: `posterior.npz`, posterior pair plots. At `composition=global` it additionally
  saves `single_stream_posterior.npz` (per-member posteriors) and
  `real_global_vs_streams_corner.png` — an overlay corner plot of the pooled global posterior plus
  each single-stream posterior over the shared global parameters (mirrors the reference
  `main_eval_gaiastreams.py` `global_cornerplot`; members named from the simulator's
  `target_streams`). Best-effort hook (never aborts the run).
- `tune`: `best_trials.json` (Optuna study in `tuning.storage_dir`)

`evaluate` / `evaluate_real` load the trained model + fitted preprocessing from `model_dir`
(set it to a completed `train` run dir).

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

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

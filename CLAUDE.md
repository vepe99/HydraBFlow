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

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

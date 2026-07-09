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
- **TODO — native missing-vlos handling in the SetTransformer (designed 2026-07-08, not
  implemented)**: replace the `mask_vlos` mean imputation with a missingness-aware summary network.
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

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

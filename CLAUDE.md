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

BayesFlow is pinned to git rev `c4c0a59` (2.0.13; needed for `utils/masks.py`, `DiffusionTransformer`,
`observed_condition_mask`). `pipeline/_bf_patches.py` still fixes the compositional summary-outputs
reshape bug (unfixed upstream at that rev) — re-check it on any bayesflow bump. Python 3.12 venv;
launch stages with `.venv/bin/python -m hydrabflow.pipeline.<stage>` rather than a bare `uv run`.

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
`streams.md` (the compositional stream project: levels, stages, presets, resources),
`decisions_log.md` (full session history), `priors_literature.md` (prior review).

## graphify

Knowledge graph at `graphify-out/`.

- For codebase questions, run `graphify query "<question>"` first (also `graphify path "<A>" "<B>"`,
  `graphify explain "<concept>"`) — a scoped subgraph, far smaller than `GRAPH_REPORT.md` or grep.
- `graphify-out/wiki/index.md`, if present, beats raw source browsing for navigation.
- Read `GRAPH_REPORT.md` only for broad architecture review.
- After changing code, run `graphify update .` (AST-only, no API cost).

## Decisions Log

*(Update this section after each Claude Code session. Keep it short: current state + durable
lessons + the latest sessions. The complete session-by-session history up to 2026-10-06 is in
`docs/decisions_log.md` — search it before re-deriving an old result.)*

### Current state (as of 2026-10-06)

- **Prior family**: `stream_agama_spray_massloss_ibata_m200c_v4_prog2026_rc38*` (Chen+2024 spray with
  linear mass loss, 10^3 particles, halo by (log10 M200, ln c'), Moster-conditional stellar mass with
  thin+thick disks, no rotation-curve rejection). Active variant: **rc38 x Eilers19 sys3**
  (`..._rc38_eilers19_sys3.yaml`, Eilers+2019 curve, sigma = stat ⊕ 3 % of v_c). sys1 (1 %) is the
  calibrated alternative (observed curve at ~30th pct), configured but not launched.
  Dataset: `data/data_jarvis/data_agama_spray_massloss_ibata_m200c_v4_p1e3_prog2026_rc38_eilers19_hydrabflow/`
  (`training_data_300000.npz` + `test_multistream_333.npz`, both now carrying `pot_scalars`).
- **Global model**: 2-modality grouped diffusion (`stream_fusion_2modal_oldgrid`, B-spline stream
  summary `core080` + curve TST, modality dropout 0.5); masked compositional eval = m stream items +
  1 curve item (the curve is counted once). Best sys3 global = **trial 24** of the core080 sys3 study
  (MMD p_strat 0.195, least misspecified so far; comp RMSE 0.679 / calib 0.013; real q 0.87 [0.78,1.01]).
- **Local model** (per-stream locals | true globals, coupling flow, ancestral on real data):
  particle locals are better in sim (carry t_end) but extrapolate on real data in every trial
  (NGC3201 t_end at the 2 Gyr floor, M68 mass below its prior floor; stream-slice MMD p≈0); the
  own-support-spline local (`train_local_ownsupport_rc38_eilers19_sys3.sh`) is worse in sim but NOT
  MMD-rejected and never rails. Best real-data pair so far: global 24 x particle local 65.
  Test-set nRMSE rank does not predict real-data PPC quality.
- **Persistent residuals** (survive every forward model, prior and estimator tried): M68 phi2 bow
  (~+2 deg at phi1 30-70), NGC3201 mu_phi1 offset, Pal5 parallax offset (+0.04-0.05 mas, data-side),
  sims run 5-15 km/s high beyond ~19 kpc vs the observed curve. Real-data q_halo depends on the
  summary representation (particles prolate ~1.3-1.5, binned summaries oblate ~0.8, prior-like in
  between) and on how the summary net is trained (frozen IMM vs end-to-end differ by ~3 sigma).
- **Open / next**: GPU training of the `pot_scalars` model (`scripts/train_potscalars_rc38_eilers19_sys3.sh`,
  CPU smoke passed); the sys1 studies; pooled-compositional miscalibration is halo-shape-only and not
  fixed by `compositional_bridge_d1` (no d1 calibrates it).

### Durable lessons (details in `docs/decisions_log.md`)

- **Physics/config conventions**: agama Disk `scaleHeight > 0` = EXPONENTIAL, `< 0` = sech² — every
  config before rc38 had it inverted (kept for old data; fix = `agama_vertical_sign: correct`). agama
  `orientation` Euler angles are radians (tilt bug fixed 2026-09-12). Progenitor `m_progenitor` is the
  INITIAL mass. Our M200 is halo-only inside its own r200; q, p are density axis ratios.
  `local_parameter_names` reads the FIRST stream only — any freed local must be freed for all streams
  (`frozen_locals.check_frozen_locals` guards the surrogates). `params.marginalize` = drawn but not inferred.
- **Forward models**: restricted N-body (rnbody) is the only one that makes M68 wide enough; spray is
  colder/thinner. Short t_end (1.5 Gyr) gives no observable NGC3201/M68 stream under rnbody (spray
  fabricates it). Surrogates (trihedron remap, orbit+DeltaTheta spline) freeze along-track structure —
  exact near the fiducial, fail M68 and any mass change. In-window phi1 extent is window-saturated for
  NGC3201/M68: use the edge/centre density ratio instead.
- **Observation model**: magnitudes must come from the real catalogue being matched
  (`magnitude_source: real_streams`); `sample_obs_error_empirical` + `override_obs_error_with_real`
  are a pair. M68 "too cold" is largely a member-selection (envelope) effect. Published STREAMFINDER
  frames (`scripts/streamfinder_frame.py`) vs catalogue-fitted frames differ — `stream_summary.py`
  still fits its own frame.
- **Network/pipeline gotchas**: BayesFlow standardizes `summary_variables` BEFORE the summary net —
  masked backbones need `summary_occupancy: valid` (all masked runs before 2026-09-22 invalid).
  Occupancy counts carry no q and are OOD on real data (equal-count real bins). Group-level
  observables are copied into every member by `flatten_members` → mask the compositional plan.
  `j` must be first in `inference_conditions` when it is a condition. `drop_nan.keys` must list
  npz-resident keys (batch-only keys like `sim_summary` KeyError). `log10_vcirc` floors at 1 km/s.
  `evaluate` has no OOM backoff (use `eval.batch_size=8` for compositional). keras needs
  `build_workflow(cfg)` in-process before loading `approximator.keras`. Posteriors in `posterior.npz`
  are in preprocessed (log10) space. `sky_projection` returns all-NaN if one input row is NaN.
  Diagonal KDE prior score works, jax full-cov does not (on truncated priors).
- **PPCs**: B-spline PPCs use the model's own training spline (`--fit aug`); average >=10 observation
  seeds before comparing widths; judge stream models distributionally, never per star.
- **Shared box**: check `Committed_AS` vs CommitLimit before big joblib runs; never `pkill -f`
  (matches your own shell) — kill by PID; `export CUDA_VISIBLE_DEVICES` after `eval "$(autocvd ...)"`;
  never pin a GPU someone else uses; don't edit a bash script while it runs; jarvis has no GPUs;
  `JAX_PLATFORMS=cuda,cpu` (not `cuda` alone); a `uv sync` can break agama's GSL link
  (rebuild recipe in `pyproject.toml`); galpy and JAX can't share a process.

### Recent sessions

- Session 2026-10-03 (rc38 x Eilers19 dataset): `conf/simulator/stream_agama_spray_massloss_ibata_m200c_v4_prog2026_rc38_eilers19.yaml`
  (38 Eilers+2019 radii 5.27-24.82 kpc, sigma = mean of the asymmetric errors; asset `assets/rotation_curve_eilers19.csv`)
  + dataset `data/data_jarvis/data_agama_spray_massloss_ibata_m200c_v4_p1e3_prog2026_rc38_eilers19_hydrabflow/`
  (training_data_300000 + test_multistream_333 = the rc38 x Jiao26 sets with vcirc_kms recomputed by AGAMA, 0 NaN).
  Tuning: `conf/tuning/tuningtest_2modal_bspline_core080_rc38_eilers19.yaml`, `scripts/tune_bspline_core080_rc38_eilers19.sh`,
  `scripts/tune_local_particles_rc38_eilers19.sh`, `PRESET=rc38_eilers19` in tune_post_eval_local_particles.sh.
  `rotation_curve_ood.py` now follows `defaults:` inheritance of simulator yamls. PPC (`<dataset>/ppc/rotation_curve/`):
  with the QUOTED Eilers errors (0.4-3 km/s inside 21 kpc) the observed curve is OUT of distribution (observed->nearest
  chi2 87 vs noisy-sim median 35, 99.6th pct; best row 87/37 dof) - residual wiggles of +-2-3 sigma at 11.7-13.7 kpc and a
  coherent -2.8 sigma run at 19.7-20.8 kpc that smooth axisymmetric potentials cannot follow. A systematic floor added in
  quadrature fixes it: 1 km/s -> 96th pct, 1.5 -> 74, 2 -> 38, 3 -> 4 (too generous); 1 % of v_c -> 33. Stored vcirc is
  noise-free (noise added at training from obs_sigma_vc), so a floor needs NO dataset regeneration. Floor not yet applied.
  3 % systematic variant: `conf/simulator/..._rc38_eilers19_sys3.yaml` (obs_sigma_vc = sqrt(sigma_stat^2 + (0.03 v_c)^2),
  ~7 km/s inside 21 kpc; same dataset). Typicality (`scripts/ppc_rotation_curve_sysfloor.py`, figure
  `<dataset>/ppc/rotation_curve_sys3/sysfloor_comparison.png`): quoted 99.4th pct (OOD), +1 % 29.7, +2 km/s 38.5,
  +3 % 0.0 (observed chi2 8.5 vs noisy-sim 1st pct 18.9: closer than any sim -> no OOD risk, but over-generous; the
  curve's constraining power is diluted). Runners `scripts/tune_bspline_core080_rc38_eilers19_sys3.sh` +
  `scripts/tune_local_particles_rc38_eilers19_sys3.sh` (studies *_rc38_eilers19_sys3_study; post-eval
  PRESET=rc38_eilers19_sys3). Verified the composed tuning config feeds the 3 % sigma to add_noise_to_vcirc. Not launched.
  NOTE `rotation_curve_ood.py` tests only the upper tail (labels a 0th-pct curve "IN distribution").
  1 % systematic variant (calibrated choice, observed curve at the ~30th pct): `conf/simulator/..._rc38_eilers19_sys1.yaml`
  (obs_sigma_vc = sqrt(sigma_stat^2 + (0.01 v_c)^2), ~2.6-3 km/s inside 21 kpc), `conf/tuning/tuningtest_2modal_bspline_core080_rc38_eilers19_sys1.yaml`,
  runners `scripts/tune_bspline_core080_rc38_eilers19_sys1.sh` + `scripts/tune_local_particles_rc38_eilers19_sys1.sh`
  (studies *_rc38_eilers19_sys1_study under <dataset>/tuning/, worker logs under outputs/Bsline/spray_p1e3_v4_prog2026_rc38_eilers19_sys1_*),
  post-eval PRESET=rc38_eilers19_sys1. Same dataset as sys3; separate studies/folders. Not launched.
- Session 2026-10-03 (rc38 x Eilers19 sys3: global trial 2 rotation-curve PPC; local particle study test-set eval +
  ancestral + joint PPC): `ppc_rotation_curve_ibata.py` (halo-only decomposition loop now try/except like the full
  curves; one pooled draw is rejected by agama) on global trial 2 `eval_real`: <=1.2 % median |frac dev|, chi2 10-16/38
  with the 3 %-inflated sigma, but every group 5-15 km/s HIGH vs Eilers beyond ~19 kpc. Local study
  `stream_2modal_particles_local_rc38_eilers19_sys3_study`: only trials 7/14/15 finished (22 weights only); 333 test
  set mean nRMSE/calib 0.757/0.013, 0.762/0.016, 0.783/0.013 -> trial 7. `outputs/Bsline/rc38_eilers19_sys3_ppc_global2/`
  (run.sh, LOCAL_TRIAL=trial_0007): local posteriors again rail/extrapolate on real data (NGC3201 t_end 1.97 < 2 floor,
  mass 1.42e5 < 1.49e5 floor, d 4.53 = -4.7 sigma; M68 mass 8.5e4 << 1.28e5 floor, d 10.1 = -3 sigma). Joint PPC: Pal5
  parallax z -2.3 (known), NGC3201 mu_phi1 inside 0.45 z -1.8, M68 phi2 bow inside 0.15 z +4.1 (worst yet).
- Session 2026-10-04 (rc38 x Eilers19 sys3: best non-misspecified global + all finished local trials + joint PPC):
  `outputs/Bsline/rc38_eilers19_sys3_ppc_best/` (README, parametrised `run.sh` GLOBAL_TRIAL/LOCAL_TRIAL). Global trial 2
  is the only one of 16 not MMD-flagged (p_strat 0.08, members 91/85/92 pct; 15/4/5 borderline). Local: 7 more trials
  evaluated on the 333 set; best = 41 (nRMSE 0.744 vs trial 7 0.757). Joint PPC global 2 x local 41: Pal5 phi2 inside
  0.00 z -3.9 (Pal5 pm/vr pulled -1.2/-2.5 sd, arms rotated), NGC3201 fine (t_end at floor), M68 bow z +2.7 and M68 mass
  below its prior floor. With local 7 Pal5 was fully covered -> test-set nRMSE rank does not predict real-data PPC.
  Follow-up with the borderline globals 5 and 15 (x locals 41 and 7, ancestral on CPU — another user held GPUs 0/1/2/4):
  the LOCAL trial decides the PPC, the global hardly matters (local 41: Pal5 phi2 inside 0.00 for all globals; local 7:
  1.00, but NGC3201 d -4.5 sd and M68 mass below its prior floor). M68 phi2 bow (inside 0.15-0.25) in every pair.
  Global 5 x PRIOR-drawn locals (placeholder local posterior.npz -> ppc_joint_posterior.py falls back to the prior):
  every track better than with any trained local model (Pal5 phi2 1.00/z 0.0, M68 phi2 0.85/z +1.3 vs 0.15-0.25,
  NGC3201 >= 0.90 except mu_phi1 0.65 / vlos 0.50). The local particle models degrade the real-data PPC; all 10 trained
  locals extrapolate (NGC3201 t_end < 2 Gyr floor in 9/10, M68 mass < floor in 6/10; least bad: 48, 42).
- Session 2026-10-05 (rc38 x Eilers19 sys3 check-in; README in `outputs/Bsline/rc38_eilers19_sys3_ppc_best/`): new best
  global = **trial 24** (MMD p_strat 0.195, the least misspecified so far; comp 0.679/0.013; real q 0.87 [0.78,1.01]).
  New best local on the 333 set = trial 79 (0.716), but its real-data joint PPC is poor (Pal5 v_los inside 0.10).
  Best real-data pair = global 24 x local 65 (mean inside 0.79). All locals still extrapolate past their priors
  (NGC3201 t_end at the floor, M68 mass below the floor). BUG fixed in `ppc_joint_posterior.py`: global posteriors leak
  to gamma >= 2 (outside the U[0,1.5] prior, 0.6-6 % of draws), which makes the m200_c halo scale radius non-positive
  and crashes agama. The script now pairs only draws with gamma < 2.
- Session 2026-10-05 (local MMD of the sys3 particle study; own-support spline LOCAL model): `mmd_local_per_stream.py`
  over all 17 evaluated local particle trials (`rc38_eilers19_sys3_ppc_best/run_mmd_local.sh` -> `mmd_local/`) and the
  prog2026 `stream_2modal_particles_local_study` trial 6: the real stream particles are OOD in EVERY trial (stream-slice
  p ~ 0, 100th pct; Pal5 passes MMD only in 44/75/79/7), curve slice in distribution. New runner
  `scripts/train_local_ownsupport_rc38_eilers19_sys3.sh` (local model on `stream_spline_ownsupport` + curve, 2 TSTs with
  global trial 24's backbone sizes, coupling flow, 300 ep; train -> eval sim -> ancestral real from global trial 24 ->
  MMD -> joint PPC) -> `outputs/Bsline/spray_p1e3_v4_prog2026_rc38_eilers19_sys3_ownsupport_local/` (README). Test
  nRMSE 0.634/0.898/0.901 (worse than particles 0.716 mean) but real MMD NOT rejected for any stream (p 0.32-0.53; NGC3201/
  M68 Mahalanobis 98-99th pct) and no local rails at a prior edge. PPC: Pal5 phi2 0.95, NGC3201 mu_phi1 0.10 (z +2.4),
  M68 phi2 bow 0.00 (z +3.0) -- the two familiar residuals survive. `ppc_joint_posterior.py --fit aug` now handles the
  own-support step (picks it from the training steps; scores only grid points covered by >= half the draws).
  Gotcha: `JAX_PLATFORMS=cuda` alone breaks jax.debug.callback (needs `cuda,cpu`).

- Session 2026-10-06 (pot_scalars = [K_z, mu_l, M200] appended to the curve branch; prior-predictive checks):
  `scripts/add_pot_scalars.py` appends `pot_scalars` (n,3) = [K_z(R0_Sun, z=1.1 kpc) km^2 s^-2 pc^-1, mu_l(Sgr A*) =
  -(v_c(R0)+V_Sun)/R0 mas/yr, TOTAL M200/1e12 (all components, 200 rho_crit, H0 70.4)] IN PLACE to an npz (zipfile
  append, nothing else rewritten; idempotent). Done for the rc38 x Eilers19 training (300k, 6 min at 32 workers, 0 NaN)
  and 333 test sets. Checks: mu_l == the simulator's vc_R0_derived exactly; McMillan17 gives K_z 2.00, mu_l -6.30,
  M200 1.37e12. Network: `MaskedFusionNetwork` gained `params.post: {vcirc_kms: {inputs: [pot_scalars], widths}}` =
  Dense(summary_dim)(MLP(concat(TST(vcirc), pot_scalars))) -- same width, so it stays ONE modality for grouped_diffusion
  and the masked compositional eval (curve item carries the scalars). Training noise = observed errors (0.16, 0.026,
  0.21) via `add_noise_to_sigma_z` (sigma may now be a list); real data via `attach_observed_sigma_z` (value may be a
  list: 2.00, -6.379, 1.17). Presets `*_potscalars` (adapter, aug, preproc train/real, model). Runner
  `scripts/train_potscalars_rc38_eilers19_sys3.sh` (step 0 = add_pot_scalars, skips if present; trial 31 of the sys3
  core080 study = Optuna Pareto front with the best real MMD p on the front; front = 1/3/14/31). CPU smoke (999 rows,
  1 epoch) passed train, sim eval (mask plan 3 x sim_summary + 1 x vcirc_kms) and real eval. Tests in
  test_fusion_frozen.py. GPU training NOT run (user launches).
  **Prior-predictive** (`scripts/ppc_pot_scalars_prior.py`, `<dataset>/ppc/pot_scalars_sys3/`): observed inside the
  prior for all three; K_z P(sim<obs) 0.80 (9.5 % of rows within 1 sigma), mu_l 0.15 (1.0 %), M200 0.54 (20 %).
  Likelihood-weighted ESS: curve 309, scalars 989, joint 12 of 300k (independence would give ~1 => curve and scalars
  agree; the curve-weighted prior predicts K_z 2.32, mu_l -6.22 [-6.30,-6.13], M200 0.91 -- within ~1-1.8 sigma). The
  joint occupies ~1/25000 of the prior (mu_l is the bottleneck, ESS 5420 alone) and wants log10 M* 10.78 (+0.9 sd),
  gamma 1.15 (toward the 1.5 edge), V_Sun 15.8 (marginalized nuisance, prior 12.2), sigma_z ~80 (obs 71+-6, unused).
  Streams (`ppc/streams_bspline_core080/`, = rc38 run): all tracks inside the sim 90 % band except Pal5 parallax (0.45),
  |z|max <= 1.9; NN pull log10 M200 +0.75-0.82 sd and log10 M* +0.75-0.99 sd for all three streams (curve+scalars put
  M200 slightly LOW, 11.92 -- the one mild disagreement). Rotation curve sys3: see 2026-10-03.

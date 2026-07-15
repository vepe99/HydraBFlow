# Graph Report - HydraBFlow  (2026-07-14)

## Corpus Check
- 118 files · ~396,241 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1273 nodes · 1822 edges · 156 communities (109 shown, 47 thin omitted)
- Extraction: 85% EXTRACTED · 15% INFERRED · 0% AMBIGUOUS · INFERRED: 275 edges (avg confidence: 0.75)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `e09ea17b`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Preprocessing Pipeline & Steps
- Eval / Checkpoint Stages
- Design Principles & Configs
- Augmentation Registry & Tests
- Simulate Stage & Registries
- Example Simulators (Skeleton/TwoMoons)
- Config Schemas
- Network Factory & Adapter
- Graphify Tooling
- Config Composition Tests
- Community 11
- Dataset IO
- Hydra App Boilerplate
- JAX Backend Pin
- Logging Helper
- Augmentation Package Init
- Package Root Init
- Marimo Notebook
- Pipeline Package Init
- Preprocessing Package Init
- Simulators Package Init
- Package/Init cluster 24
- Community 31
- Community 32
- Community 33
- Community 34
- Community 35
- Community 36
- Community 37
- streams.py
- Community 39
- Community 40
- Community 41
- Community 42
- Community 43
- Community 44
- Part B — Support a file format other than `.npz`
- MaskedFusionNetwork
- compositional.py
- Community 56
- Community 57
- run_tuning
- build_workflow
- MaskedFusionNetwork
- Run stages (5 entry points)
- hooks
- PreToolUse
- permissions
- allow
- AST Structural Extraction
- EXTRACTED/INFERRED/AMBIGUOUS Audit Trail
- Community Detection
- Detect Files Step
- Existing-Graph Fast Path
- Gemini Extraction Backend
- God Nodes
- graph.json Output
- GRAPH_REPORT.md Output
- Python Interpreter Detection
- Knowledge Graph
- Obsidian Vault Export
- Semantic Extraction Cache
- Semantic LLM Extraction
- Parallel Subagent Dispatch
- hydrabflow
- BaseSimulator
- test_streams.py
- reporting.py
- Stream project (compositional score modeling)
- PerStreamParameterStandardize
- load_approximator
- Running a full pipeline with the Two Moons simulator
- _load_clean
- ppc_prior_predictive.py
- corner_parameters.py
- StreamObservationStats
- test_streams.py
- 2. The four commands (full run)
- compose_cfg
- _vcirc_worker
- create_rnbody_huang_dataset.sh
- training_eval_missing_vlos_ablation.sh
- training_eval_rnbody_huand_dataset.sh
- ndarray
- stream_agama.py
- stream_common.py
- assets/gaia — portable static inputs for the stream project
- StreamObservationStats
- extended_rotation_curve
- compose
- PerStreamParameterStandardize
- prior_score_from_spec
- test_config.py
- get_run_dir
- prior_score_from_kde
- eval_rnbody_huand_dataset_kde_prior.sh
- training_eval_agama_1e6.sh
- training_eval_rnbody_huand_dataset copy.sh
- PerStreamParameterStandardize
- load_approximator
- prior_score_from_spec
- MaskVcircRadii
- compose_cfg
- ppc_ancillary_observables.py
- _spray_stream
- test_config.py
- test_workflow.py
- inferred_names
- create_ibata_dataset.sh
- train_ibata_sumstats.sh
- tune_ibata_sumstats.sh
- main
- 4. Adding a SummaryNetwork that isn't shipped
- 5. Adding an InferenceNetwork that isn't shipped
- MaskedFusionNetwork
- streams.py
- _spray_stream
- create_ibata_onedisk_beta3_dataset.sh
- train_ibata_m200c.sh
- tune_ibata_m200c.sh
- AttachObservedSigmaZ
- AttachObservedVterm
- create_ibata_m200c_dataset.sh
- register_summary_network

## God Nodes (most connected - your core abstractions)
1. `AgamaStreamSimulator` - 39 edges
2. `_jax()` - 31 edges
3. `PreprocessStep` - 25 edges
4. `_evaluate_compositional_global()` - 21 edges
5. `compose()` - 20 edges
6. `register_configs()` - 18 edges
7. `_evaluate_local()` - 17 edges
8. `_evaluate_real_compositional()` - 17 edges
9. `build_workflow()` - 17 edges
10. `BaseSimulator` - 17 edges

## Surprising Connections (you probably didn't know these)
- `test_build_workflow()` --calls--> `build_workflow()`  [INFERRED]
  tests/test_workflow.py → src/hydrabflow/pipeline/workflow.py
- `test_two_moons_shapes_and_reproducibility()` --calls--> `get_simulator()`  [INFERRED]
  tests/test_augmentation.py → src/hydrabflow/simulators/registry.py
- `main()` --calls--> `get_simulator()`  [INFERRED]
  scripts/probe_vcirc_acceptance.py → src/hydrabflow/simulators/registry.py
- `main()` --calls--> `sample_stream_prior()`  [INFERRED]
  scripts/probe_vcirc_acceptance.py → src/hydrabflow/simulators/stream_common.py
- `main()` --calls--> `compose()`  [INFERRED]
  scripts/probe_vcirc_acceptance.py → tests/conftest.py

## Import Cycles
- None detected.

## Communities (156 total, 47 thin omitted)

### Community 0 - "Preprocessing Pipeline & Steps"
Cohesion: 0.06
Nodes (33): PreprocessPipeline, PreprocessStep, ABC, Dataset, ndarray, Preprocessing step protocol and the pipeline that orchestrates them.  A :class:`, Element-wise (dataset-in, dataset-out) transform with optional fitted state., Estimate any state from ``data`` (train split). Stateless steps leave this empty (+25 more)

### Community 1 - "Eval / Checkpoint Stages"
Cohesion: 0.33
Nodes (6): Model Default Config, Diffusion Inference Network Config, Flow Matching Inference Network Config, DeepSet Summary Network Config, SetTransformer Summary Network Config, TimeSeriesTransformer Summary Network Config

### Community 2 - "Design Principles & Configs"
Cohesion: 0.29
Nodes (9): _n(), Stage 2: training.  Load dataset -> preprocessing pipeline (fit on train, save f, Load the best-val-loss weights BayesFlow checkpointed during training back into, Persist ``history.json`` + ``convergence.json``; best-effort, never fails a run., Train the approximator and return (workflow, history)., _restore_best_weights(), run_training(), _save_history_and_convergence() (+1 more)

### Community 3 - "Augmentation Registry & Tests"
Cohesion: 0.08
Nodes (35): _band_pass_worker(), main(), ndarray, Measure the vcirc-rejection acceptance rate of a stream simulator's prior (calib, (n_rows, n_bands) bool: does each row's model curve pass each band? (one vc eval, feature_dropout(), gaussian_noise(), multiplicative_noise() (+27 more)

### Community 4 - "Simulate Stage & Registries"
Cohesion: 0.07
Nodes (35): available_augmentations(), Name -> augmentation-factory registry and builder.  An augmentation factory rece, available_steps(), Name -> preprocessing-step registry and pipeline builder., Register a step factory (usually the step class itself) under ``name``., register_step(), available_simulators(), Name -> simulator-class registry.  New simulators self-register with the ``@regi (+27 more)

### Community 5 - "Example Simulators (Skeleton/TwoMoons)"
Cohesion: 0.10
Nodes (19): BaseException, RuntimeError, Dataset, ndarray, Per-feature z-score standardization step.  Generalizes the reference project's `, Standardizer, is_oom_error(), Retry a GPU computation with a progressively smaller batch size when it runs out (+11 more)

### Community 6 - "Config Schemas"
Cohesion: 0.05
Nodes (21): BaseSimulator, BaseSimulator, ABC, Any, ndarray, Base interface every forward model implements.  A simulator is the ONLY piece a, Abstract forward model. Subclass + register via ``@register_simulator``., Ordered names of the inferred parameters (become ``inference_variables``). (+13 more)

### Community 7 - "Network Factory & Adapter"
Cohesion: 0.07
Nodes (28): 0. Prerequisites & install, 1. The five stages at a glance, 2. Changing the simulator, 2a. Write the simulator class, 2b. Registration is automatic, 2c. Add the simulator config, 2d. The adapter wires itself, 2e. Shape contract cheat-sheet (+20 more)

### Community 10 - "Config Composition Tests"
Cohesion: 0.08
Nodes (27): Logger, adapter_keys(), _as_list(), build_adapter(), fill_adapter_from_simulator(), Any, Build the BayesFlow ``Adapter`` from ``AdapterConfig``.  The adapter is the stru, All dataset keys the adapter (``cfg.adapter``) consumes, in a stable order. (+19 more)

### Community 11 - "Community 11"
Cohesion: 0.07
Nodes (12): AgamaStreamSimulator, Stellar streams in a parametrized Milky Way potential, simulated with AGAMA., Split radius for the extended (Zhou u Huang) rotation-curve grid., Host-potential configuration threaded to the joblib workers. Legacy default (all, Requested Ibata ancillary observables (``params.ancillary_observables``); empty, Spec passed to the joblib worker: requested names + their fixed grids (or None)., Dataset keys the requested ancillary observables are stored under (``vterm``->``, Prior spec of the inferred global parameters (used for compositional prior score (+4 more)

### Community 12 - "Dataset IO"
Cohesion: 0.14
Nodes (8): _masked_set_transformer(), MaskedSetTransformer, Layer, SummaryNetwork, Tensor, Missingness-aware SetTransformer for stars without a measured line-of-sight velo, Build a :class:`MaskedSetTransformer` from a ``SummaryNetworkConfig`` (see modul, Zero masked feature channels, add a learned missing-value embedding, run a SetTr

### Community 13 - "Hydra App Boilerplate"
Cohesion: 0.33
Nodes (5): conf_path(), make_cli(), Shared Hydra-app boilerplate for the five run stages., Absolute path to the repo-root ``conf/`` directory., Wrap a ``run_fn(cfg)`` into a Hydra console entry point.      Registers the stru

### Community 14 - "JAX Backend Pin"
Cohesion: 0.13
Nodes (23): _bf_mmd(), member_summaries(), mmd_test(), _null_mmd(), per_member_scores(), ndarray, Summary-space model misspecification test (observed group vs simulated reference, Mahalanobis OOD score of each observed member vs its own stream's reference clou (+15 more)

### Community 15 - "Logging Helper"
Cohesion: 0.17
Nodes (12): compose(), Expose the composer so tests can build configs with custom overrides., Config composition + schema validation smoke tests., test_adapter_derived_from_simulator(), test_adapter_explicit_config_wins(), test_group_override(), Stream-project components: config composition, hierarchy derivation, per-stream, test_adapter_derivation_follows_composition_level() (+4 more)

### Community 31 - "Community 31"
Cohesion: 0.08
Nodes (23): For /graphify add and --watch, For /graphify query, For the commit hook and native CLAUDE.md integration, For --update and --cluster-only, /graphify, Honesty Rules, Interpreter guard for subcommands, Part A - Structural extraction for code files (+15 more)

### Community 32 - "Community 32"
Cohesion: 0.20
Nodes (15): _deep_set(), _diffusion(), _embed_dim(), _feature_transformer(), _flow_matching(), _mlp(), Any, Build BayesFlow networks from structured dataclass configs (no ``_target_``).  B (+7 more)

### Community 33 - "Community 33"
Cohesion: 0.20
Nodes (10): 1. Prerequisites, 2. Run a study, 3. What gets saved, 4. Run many processes at once (parallel tuning), 5. Reading the results, 6. Changing what is tuned (the search space), 7. Key config reference (`tuning` group), 8. Command recap (+2 more)

### Community 34 - "Community 34"
Cohesion: 0.17
Nodes (11): Core Design Principles, Decisions Log, Folder Structure (finalized), Goal, graphify, HydraBFlow: SBI Pipeline Template with BayesFlow, Output Directory Convention, Run stages (6 entry points) (+3 more)

### Community 35 - "Community 35"
Cohesion: 0.12
Nodes (25): _np_fit_frame(), _np_phi1(), _np_unit_vec(), Hand-crafted per-stream summary statistics in a data-driven stream-aligned frame, Per-stream summary statistics laid out as a **φ1 time series** for a single, Rows of R are [x, y, z]: z = normal to best-fit plane (great-circle pole), x = m, Per-stream great-circle rotation ``R_j`` and φ1 bin edges, fitted once from the, _stream_frames() (+17 more)

### Community 37 - "Community 37"
Cohesion: 0.07
Nodes (30): 1. How the config system works, 2. The root master config — `config.yaml`, 3.10 `tuning/`, 3.1 `simulator/`, 3.2 `model/`, 3.3 `data/`, 3.4 `training/`, 3.5 `preprocessing/` (+22 more)

### Community 38 - "streams.py"
Cohesion: 0.09
Nodes (45): _add_noise_to_rho_z(), _add_noise_to_sigma_z(), _add_noise_to_vcirc(), _add_noise_to_vterm(), _apply_obs_error(), _compact_to_attended(), _concatenate_magnitudes(), _concatenate_sigma_errors() (+37 more)

### Community 39 - "Community 39"
Cohesion: 0.36
Nodes (7): PreprocessPipeline, build_pipeline(), Build a :class:`PreprocessPipeline` from ``cfg.preprocessing`` (a ``Preprocessin, Preprocessing pipeline: fit/transform/split + state save/load round-trip., test_pipeline_fit_transform_and_split(), test_state_roundtrip(), _toy_data()

### Community 40 - "Community 40"
Cohesion: 0.26
Nodes (16): _evaluate_compositional_global(), _evaluate_local(), _load_test_data(), Stage 3: evaluation on a simulated test set (with known ground truth).  Loads th, Global-level evaluation on a grouped (multistream) test set, both ways:      * *, Local-level evaluation: per-member sampling conditioned on the true globals., Best-effort ``report.md`` from the metrics/figures just written; never aborts a, Load the held-out test set and replay the *fitted* preprocessing (no re-fit, no (+8 more)

### Community 46 - "Part B — Support a file format other than `.npz`"
Cohesion: 0.29
Nodes (7): B.1 The single seam, B.2 Option 1 — Quick swap (one format, replace the body), B.3 Option 2 — A format registry (support several formats by extension), B.4 (Optional) validate keys/shapes on load, Bring your own data (no simulator) & supporting other file formats, Checklist, Part B — Support a file format other than `.npz`

### Community 47 - "MaskedFusionNetwork"
Cohesion: 0.32
Nodes (7): fix_keras_model(), load_approximator(), Any, Model save/load helpers, including the BayesFlow ``.keras`` deserialization work, Return a path to a load-safe copy of ``model_path`` (patching the ArrayImpl tag), Load a saved approximator, applying the ArrayImpl fix first., save_approximator()

### Community 54 - "compositional.py"
Cohesion: 0.16
Nodes (14): apply_augmentations_once(), build_prior_score(), composition_level(), flatten_members(), group_members(), log10_keys_from_pipeline(), ndarray, Helpers for compositional (grouped) evaluation.  A compositional dataset stores (+6 more)

### Community 56 - "Community 56"
Cohesion: 0.25
Nodes (13): condition_keys(), Raw batch keys that act as sampling conditions (everything the adapter consumes, _evaluate_real_compositional(), _max_particles(), _prepare_real_members(), Stage 5: application to real (observed) data.  Like :mod:`evaluate`, but the inp, Load the observed group and normalize it to flat member rows.      Expected layo, Overlay corner plot of the pooled *global* posterior and each *single-stream* po (+5 more)

### Community 57 - "Community 57"
Cohesion: 0.50
Nodes (3): import_submodules(), Auto-import the modules of a package so ``@register_*`` decorators run.  The reg, Import every non-underscore module directly inside a package.      Call from a p

### Community 58 - "run_tuning"
Cohesion: 0.29
Nodes (10): _objective(), Stage 4: hyperparameter tuning with Optuna.  Runs a (by default multi-objective), Per-trial artifact directory, keyed by the study-global Optuna trial number., Save the fit-once preprocessing state, shared by every trial/model.      Written, _report(), run_tuning(), _save_shared_preprocessing(), _suggest() (+2 more)

### Community 59 - "build_workflow"
Cohesion: 0.20
Nodes (9): apply_bayesflow_patches(), _patch_compositional_condition_reshape(), Targeted runtime fixes for known BayesFlow bugs (version-checked, applied once)., Idempotently install the fixes. Called when a compositional workflow is built., build_workflow(), Any, Assemble the BayesFlow workflow from config.  Single-level inference (the defaul, Build a ``bf.BasicWorkflow`` from the root ``cfg``. (+1 more)

### Community 60 - "MaskedFusionNetwork"
Cohesion: 0.18
Nodes (10): 1. The canonical set: 6D stream tracks in stream-aligned coordinates, 2. Width, dispersion and length (second-moment tracks), 3. Action–angle / frequency-space summaries (most directly potential-sensitive), 4. Orbital-pole / great-circle summaries, 5. Density-structure / power-spectrum summaries (mostly for substructure — lower priority for you), 6. Progenitor / global scalars, Key references, Recommended concrete feature block to concatenate with the SetTransformer embedding (+2 more)

### Community 96 - "test_streams.py"
Cohesion: 0.09
Nodes (21): build_augmentations(), Augmentation, Build the ordered augmentation list from ``cfg.augmentation`` (an ``Augmentation, Stage 1b: compositional (grouped) dataset generation.  Like :mod:`simulate`, but, Generate the compositional dataset described by ``cfg`` and return its path., run_multistream_simulation(), Stage 1: dataset generation.  Samples the prior and runs the forward model in ch, Generate the dataset described by ``cfg`` and return its path. (+13 more)

### Community 97 - "reporting.py"
Cohesion: 0.20
Nodes (13): _finite(), _has_nonfinite(), inspect_history(), _load_json(), _metrics_table(), Any, _rate(), Training-convergence inspection and Markdown report generation.  Two pure-Python (+5 more)

### Community 98 - "Stream project (compositional score modeling)"
Cohesion: 0.29
Nodes (7): 1. What more flexible potential family, 2. More realistic stream simulation with agama — straight from their bundled examples, Adding your own simulator, Design at a glance, Future work (TODO), HydraBFlow, Quickstart

### Community 99 - "PerStreamParameterStandardize"
Cohesion: 0.29
Nodes (7): 0. What you're running, 1. Prerequisites, 3. Fast smoke run (≈1 minute), 4. Optional: train with observational noise (augmentations), 5. Tuning the prior / observation knobs (optional), 6. Command recap, Running a full pipeline with the Two Moons simulator

### Community 100 - "load_approximator"
Cohesion: 0.15
Nodes (17): Exception, _assert_reached(), _OrbitCapExceeded, _plummer_sample(), ndarray, Restricted N-body stellar-stream forward model on AGAMA (CPU, joblib).  Same pri, Restricted N-body stream: forward-integrate Plummer particles from the rewound o, joblib worker: one restricted-N-body stream + the rotation curve of its potentia (+9 more)

### Community 101 - "Running a full pipeline with the Two Moons simulator"
Cohesion: 0.13
Nodes (17): convert_concentration(), Convert an NFW halo concentration between spherical-overdensity definitions., _m200c_params(), _priors_local_ident(), Tests for the (M200, c_v') halo reparameterization (McMillan 2017; stream_agama., m200_c: simulate() emits rho/a_..._derived (n,1) equal to the per-row _halo_para, Legacy rho_a halo: simulate() does NOT emit the *_derived diagnostic keys., M200=1.3e12, c_v'=15.4, gamma=1 -> r_h=19.6 kpc, rho_0=8.54e6 (McMillan Table 3) (+9 more)

### Community 102 - "_load_clean"
Cohesion: 0.43
Nodes (6): _load_clean(), main(), ndarray, Cross-model posterior tension report (offline analysis helper — not a Hydra run, Load a posterior .npz as {param: (n_datasets, n_samples)} float arrays., _resolve()

### Community 104 - "ppc_prior_predictive.py"
Cohesion: 0.20
Nodes (15): main(), ndarray, Overlay two particle-spray recipes (Fardal+2015 vs Chen+2024) against the real G, (sky RA*cos(dec), Dec) and (pm_ra_cosdec, pm_dec) for the finite particles of st, _stream_xy_pm(), load_dataset(), main(), ndarray (+7 more)

### Community 105 - "corner_parameters.py"
Cohesion: 0.22
Nodes (12): NpzFile, autoscale(), detect_param_keys(), expand_inputs(), load_columns(), main(), ndarray, Corner plot of parameter draws from a simulate dataset (offline helper — not a H (+4 more)

### Community 106 - "StreamObservationStats"
Cohesion: 0.33
Nodes (6): A.1 The data contract, A.2 Convert your existing arrays into the dataset file, A.3 Tell the pipeline about it (config only), A.4 Run train + evaluate, A.5 What reads what, Part A — Use a pre-existing dataset (no simulator)

### Community 107 - "test_streams.py"
Cohesion: 0.16
Nodes (21): _build(), Missing-v_los handling: fill modes (mask_vlos / impute_vlos) and the missingness, The real stream_global params (resources from the git-tracked assets/gaia copy),, Default (mean) mode: unmeasured v_los carries the mean of the measured stars, si, Batch shaped like the real path: vlos_mask given, unmeasured v_los pre-filled wi, _real_like_batch(), _star_batch(), _stream_params() (+13 more)

### Community 108 - "2. The four commands (full run)"
Cohesion: 0.29
Nodes (7): quiet_worker(), Silence noisy C-extension output (AGAMA) during simulation.  AGAMA writes diagno, True unless ``HYDRABFLOW_SIM_QUIET`` is set to a falsy value., Redirect fd 1 & 2 to ``/dev/null`` for the duration of the block (C-level output, Decorator: run a (joblib) worker with its C-level stdout/stderr redirected to /d, sim_quiet_enabled(), suppress_c_stdio()

### Community 110 - "_vcirc_worker"
Cohesion: 0.27
Nodes (11): _agama(), extended_rotation_curve(), _host_potential(), _load_posterior(), main(), Posterior-predictive check on the *model rotation curve* only.  Takes a complete, Model circular velocity [km/s] at obs_r; NaN where v^2 < 0., Zhou up to split, Huang beyond; sorted by radius. (r, vc, sigma). (+3 more)

### Community 114 - "ndarray"
Cohesion: 0.18
Nodes (7): ndarray, Galactic longitudes [deg] the terminal-velocity curve is evaluated on (single so, Heights z [kpc] the vertical stellar-density profile is evaluated on., Resolve ``vcirc_rejection`` into a list of band specs for the accept worker., Screen each draw's global potential against the observed rotation curve., Draw with ``sample_fn`` until ``n`` rows pass the rotation-curve cut., One shared global draw per dataset, one stream realization per target stream.

### Community 115 - "stream_agama.py"
Cohesion: 0.17
Nodes (21): _agama(), _ancillary_observables(), _halo_params(), _halo_params_m200c(), _host_potential(), Stellar-stream forward model built on AGAMA (CPU, parallelized with joblib).  Po, Halo ``Spheroid`` params from (virial mass, concentration) instead of (densityNo, Assemble the Milky Way host potential from one parameter row.      Legacy model (+13 more)

### Community 116 - "stream_common.py"
Cohesion: 0.09
Nodes (30): ndarray, Shared helpers for the stellar-stream simulators (agama, gala, ...).  Ports the, Circular velocity vc(R) [km/s] in the midplane; vc^2 = R dPhi/dR = -R F_R. NaN w, HI terminal velocity v_term(l) (Ibata 2023, Eq. 13), tangent-point method., Local surface density Sigma(z) [Msun/pc^2] from the vertical force (Kuijken & Gi, Vertical stellar-density profile rho(z) [Msun/kpc^3] at R (Ibata et al. 2017b)., Draw ``(n, 1)`` samples from one prior spec (uniform / normal / identity)., Single-stream draw: global parameters, a random stream index ``j``, and that str (+22 more)

### Community 117 - "assets/gaia — portable static inputs for the stream project"
Cohesion: 0.50
Nodes (3): assets/gaia — portable static inputs for the stream project, Contents, Using these on a new cluster

### Community 118 - "StreamObservationStats"
Cohesion: 0.14
Nodes (11): PerStreamParameterStandardize, Dataset, ndarray, Fit per-stream observation stats + log10(vcirc) per-bin stats on the (clean) tra, Integer stream ids broadcastable against ``like``.      ``j``'s leading axes alw, z-score each stream's local parameters with that stream's prior mean/std., _stream_index(), StreamObservationStats (+3 more)

### Community 119 - "extended_rotation_curve"
Cohesion: 0.22
Nodes (7): fill_stream_grid_from_simulator(), Align the training-time rotation-curve grid with the simulator's ``vcirc_kms`` g, Radii the model rotation curve is evaluated on (also the ``vcirc_kms`` grid)., Per-bin observed 1-sigma on the rotation curve, aligned with ``obs_r_kpc``., Observed Milky Way circular velocity aligned with ``obs_r_kpc`` — the fixed curv, extended_rotation_curve(), Union rotation-curve grid: Zhou (2023) up to ``split_kpc``, Huang (2016) beyond

### Community 120 - "compose"
Cohesion: 0.20
Nodes (17): _agama(), _band_accept(), extended_rotation_curve(), _host_potential(), _load_post(), main(), _menc(), _menc_stack() (+9 more)

### Community 121 - "PerStreamParameterStandardize"
Cohesion: 0.42
Nodes (8): binned_median(), fit_frame(), main(), project(), Overlay simulated per-stream summary tracks on the real Gaia streams and save to, render(), unit_vec(), window_subsample()

### Community 122 - "prior_score_from_spec"
Cohesion: 0.67
Nodes (3): GPU_IDS, run_arm(), training_eval_summary_stats.sh script

### Community 123 - "test_config.py"
Cohesion: 0.17
Nodes (17): concatenate_chunks(), load_chunk(), load_dataset(), _n_rows(), Dataset, Dataset IO. Datasets are ``.npz`` archives where each key maps to an array whose, Concatenate a list of dataset dicts along the leading (simulation) axis., Write a chunk ``.npz`` atomically (temp file + rename), so a crash mid-write can (+9 more)

### Community 124 - "get_run_dir"
Cohesion: 0.43
Nodes (7): main(), plot_curve(), ndarray, Extend a stream dataset's rotation-curve observable onto larger radii (offline h, joblib worker: model rotation curve on ``obs_r`` for a chunk of parameter rows., recompute_vcirc(), _vcirc_worker()

### Community 125 - "prior_score_from_kde"
Cohesion: 0.17
Nodes (13): Container, prior_score_from_kde(), prior_score_from_kde_jax(), KDE compositional prior score via ``jax.scipy.stats.gaussian_kde`` + ``jax.grad`, KDE compositional prior score, fit in the network's native (un-standardized, log, The KDE prior score must be fit in the network's native space (log10 for ``log10, With no ``log10_keys`` every parameter stays in physical/linear space; the KDE m, The jax.scipy.stats.gaussian_kde implementation must (a) fit in the network's na (+5 more)

### Community 129 - "PerStreamParameterStandardize"
Cohesion: 0.12
Nodes (15): Galaxy potential model — components to add (following Ibata et al. 2023, Sec. 5), Goal, HI terminal velocity, How to augment the dataset, Notes / gotchas, Observational data (observed values + reported uncertainties), (Optional) explicit Gaussian likelihood terms, Priors on the new components (+7 more)

### Community 130 - "load_approximator"
Cohesion: 0.33
Nodes (9): _artifact_dir(), _best_for_cutoff(), _cutoffs(), _load_completed(), main(), _overrides(), Milestone best-trial selector for the Ibata gridded-summary tuning study (offlin, Ranking key: (rmse, calibration_error); missing user-attrs fall back to objectiv (+1 more)

### Community 131 - "prior_score_from_spec"
Cohesion: 0.29
Nodes (7): prior_score_from_spec(), Score of the log prior for compositional sampling, from a prior-spec mapping., The callable's signature names a ``time`` parameter, so BayesFlow's     ``prior_, d/dy log p_Y(y) for y=log10(x) must include the +ln(10) Jacobian term, not just, test_prior_score_applies_time_decay(), test_prior_score_from_spec(), test_prior_score_log10_jacobian_correction()

### Community 132 - "MaskVcircRadii"
Cohesion: 0.29
Nodes (6): ADAPTER, MODEL, RUNS_DIR, tune_ibata_settransformer.sh script, STUDY, TUNING

### Community 133 - "compose_cfg"
Cohesion: 0.50
Nodes (4): cfg(), compose_cfg(), Shared test fixtures., Compose the root config with the structured schemas registered.      ``fill=True

### Community 134 - "ppc_ancillary_observables.py"
Cohesion: 0.32
Nodes (7): _band(), _load_2d(), main(), ndarray, Prior-predictive checks for the Ibata (2023) ancillary potential observables (of, Return a stored observable as (n_rows, n_bins), or None if absent., Median + 68/95% percentile band of `rows` (n, len(x)) vs x.

### Community 135 - "_spray_stream"
Cohesion: 0.83
Nodes (3): export_eval_gpu(), run_worker(), tune_ibata_onedisk_grid.sh script

### Community 137 - "test_workflow.py"
Cohesion: 0.50
Nodes (3): DATA_DIR, train_ibata_wdisk_beta3.sh script, SIM

### Community 142 - "main"
Cohesion: 0.29
Nodes (7): Configuration used explicitly, Data generation, Evaluation (plots + metrics), GPU, Hyperparameter tuning, Stream project (compositional score modeling), Training

### Community 143 - "4. Adding a SummaryNetwork that isn't shipped"
Cohesion: 0.16
Nodes (14): Hyperparameters for a single BayesFlow summary network.      ``type`` is resolve, SummaryNetworkConfig, build_inference_network(), build_summary_network(), Return the summary network selected by ``cfg.type`` (a ``SummaryNetworkConfig``), Return the inference (posterior) network selected by ``cfg.type`` (an ``Inferenc, _fusion(), Multi-observable fusion summary network (attention-mask aware).  Consumes the di (+6 more)

### Community 144 - "5. Adding an InferenceNetwork that isn't shipped"
Cohesion: 0.40
Nodes (5): 2.1 Generate the training set, 2.2 Generate a held-out test set, 2.3 Train, 2.4 Evaluate, 2. The four commands (full run)

### Community 145 - "MaskedFusionNetwork"
Cohesion: 0.22
Nodes (6): Shape, MaskedFusionNetwork, Layer, SummaryNetwork, Tensor, Fuse one summary backbone per named input; route the attention mask to one of th

### Community 146 - "streams.py"
Cohesion: 0.18
Nodes (6): AttachObservedVcirc, MaskVcircRadii, Stream-specific preprocessing: per-stream normalization and rotation-curve trimm, Attach the *observed* Milky Way rotation curve to a real dataset that lacks one., Keep only rotation-curve bins with ``r >= r_min`` on the observed radii grid., test_mask_vcirc_radii_trims_grid()

### Community 147 - "_spray_stream"
Cohesion: 0.25
Nodes (8): _ic_chen_spray(), _ic_particle_spray(), Jacobi radius, velocity offset, and host->satellite rotation matrices along the, Fardal+2015 initial conditions for particles escaping through the Lagrange point, Chen+2024 initial conditions: one trailing + one leading particle per orbit seed, Particle-spray stream including the progenitor's own (moving Plummer) potential., _rj_vj_R(), _spray_stream()

### Community 148 - "create_ibata_onedisk_beta3_dataset.sh"
Cohesion: 0.50
Nodes (3): HYDRABFLOW_NUM_GPUS, HYDRABFLOW_SIM_QUIET, create_ibata_onedisk_beta3_dataset.sh script

### Community 150 - "train_ibata_m200c.sh"
Cohesion: 0.40
Nodes (4): DATA_DIR, RUNS_DIR, train_ibata_m200c.sh script, SIM

### Community 151 - "tune_ibata_m200c.sh"
Cohesion: 0.40
Nodes (4): DATA_DIR, RUNS_DIR, tune_ibata_m200c.sh script, SIM

### Community 152 - "AttachObservedSigmaZ"
Cohesion: 0.40
Nodes (3): AttachObservedSigmaZ, Attach the *observed* local surface density Sigma(1.1 kpc) to a real dataset tha, test_attach_observed_sigma_z_tiles_scalar()

### Community 153 - "AttachObservedVterm"
Cohesion: 0.40
Nodes (3): AttachObservedVterm, Attach the *observed* HI terminal-velocity curve to a real dataset that lacks on, test_attach_observed_vterm_tiles_observed_curve()

### Community 154 - "create_ibata_m200c_dataset.sh"
Cohesion: 0.50
Nodes (3): HYDRABFLOW_NUM_GPUS, HYDRABFLOW_SIM_QUIET, create_ibata_m200c_dataset.sh script

### Community 155 - "register_summary_network"
Cohesion: 0.67
Nodes (3): Decorator registering a summary-network builder under ``name`` (the config ``typ, register_summary_network(), test_custom_network_builder_registers()

## Knowledge Gaps
- **204 isolated node(s):** `hydrabflow`, `create_ibata_dataset.sh script`, `create_ibata_m200c_dataset.sh script`, `HYDRABFLOW_NUM_GPUS`, `HYDRABFLOW_SIM_QUIET` (+199 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **47 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `AgamaStreamSimulator` connect `Community 11` to `load_approximator`, `Running a full pipeline with the Two Moons simulator`, `Config Schemas`, `ndarray`, `stream_agama.py`, `stream_common.py`, `extended_rotation_curve`?**
  _High betweenness centrality (0.127) - this node is a cross-community bridge._
- **Why does `get_simulator()` connect `test_streams.py` to `Augmentation Registry & Tests`, `Simulate Stage & Registries`, `Config Schemas`, `Community 40`, `Config Composition Tests`, `JAX Backend Pin`, `Logging Helper`, `extended_rotation_curve`, `Community 56`?**
  _High betweenness centrality (0.103) - this node is a cross-community bridge._
- **Why does `PreprocessStep` connect `Preprocessing Pipeline & Steps` to `Example Simulators (Skeleton/TwoMoons)`, `Community 39`, `streams.py`, `StreamObservationStats`, `AttachObservedSigmaZ`, `AttachObservedVterm`?**
  _High betweenness centrality (0.072) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `AgamaStreamSimulator` (e.g. with `_OrbitCapExceeded` and `RestrictedNbodyStreamSimulator`) actually correct?**
  _`AgamaStreamSimulator` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `_jax()` (e.g. with `_stream_summary_grid()` and `_stream_summary_statistics()`) actually correct?**
  _`_jax()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 13 inferred relationships involving `PreprocessStep` (e.g. with `PreprocessPipeline` and `Standardizer`) actually correct?**
  _`PreprocessStep` has 13 INFERRED edges - model-reasoned connections that need verification._
- **Are the 12 inferred relationships involving `_evaluate_compositional_global()` (e.g. with `load_approximator()` and `apply_augmentations_once()`) actually correct?**
  _`_evaluate_compositional_global()` has 12 INFERRED edges - model-reasoned connections that need verification._
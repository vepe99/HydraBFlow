# Graph Report - HydraBFlow  (2026-07-08)

## Corpus Check
- 80 files · ~351,876 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 908 nodes · 1307 edges · 113 communities (78 shown, 35 thin omitted)
- Extraction: 84% EXTRACTED · 16% INFERRED · 0% AMBIGUOUS · INFERRED: 214 edges (avg confidence: 0.74)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `f9857b15`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Preprocessing Pipeline & Steps|Preprocessing Pipeline & Steps]]
- [[_COMMUNITY_Eval  Checkpoint Stages|Eval / Checkpoint Stages]]
- [[_COMMUNITY_Design Principles & Configs|Design Principles & Configs]]
- [[_COMMUNITY_Augmentation Registry & Tests|Augmentation Registry & Tests]]
- [[_COMMUNITY_Simulate Stage & Registries|Simulate Stage & Registries]]
- [[_COMMUNITY_Example Simulators (SkeletonTwoMoons)|Example Simulators (Skeleton/TwoMoons)]]
- [[_COMMUNITY_Config Schemas|Config Schemas]]
- [[_COMMUNITY_Network Factory & Adapter|Network Factory & Adapter]]
- [[_COMMUNITY_Graphify Tooling|Graphify Tooling]]
- [[_COMMUNITY_Config Composition Tests|Config Composition Tests]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Dataset IO|Dataset IO]]
- [[_COMMUNITY_Hydra App Boilerplate|Hydra App Boilerplate]]
- [[_COMMUNITY_JAX Backend Pin|JAX Backend Pin]]
- [[_COMMUNITY_Logging Helper|Logging Helper]]
- [[_COMMUNITY_Augmentation Package Init|Augmentation Package Init]]
- [[_COMMUNITY_Package Root Init|Package Root Init]]
- [[_COMMUNITY_Marimo Notebook|Marimo Notebook]]
- [[_COMMUNITY_Pipeline Package Init|Pipeline Package Init]]
- [[_COMMUNITY_Preprocessing Package Init|Preprocessing Package Init]]
- [[_COMMUNITY_Simulators Package Init|Simulators Package Init]]
- [[_COMMUNITY_PackageInit cluster 24|Package/Init cluster 24]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_streams.py|streams.py]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_stream_common.py|stream_common.py]]
- [[_COMMUNITY_MaskedFusionNetwork|MaskedFusionNetwork]]
- [[_COMMUNITY_compositional.py|compositional.py]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 57|Community 57]]
- [[_COMMUNITY_run_tuning|run_tuning]]
- [[_COMMUNITY_build_workflow|build_workflow]]
- [[_COMMUNITY_MaskedFusionNetwork|MaskedFusionNetwork]]
- [[_COMMUNITY_Run stages (5 entry points)|Run stages (5 entry points)]]
- [[_COMMUNITY_hooks|hooks]]
- [[_COMMUNITY_PreToolUse|PreToolUse]]
- [[_COMMUNITY_permissions|permissions]]
- [[_COMMUNITY_allow|allow]]
- [[_COMMUNITY_AST Structural Extraction|AST Structural Extraction]]
- [[_COMMUNITY_EXTRACTEDINFERREDAMBIGUOUS Audit Trail|EXTRACTED/INFERRED/AMBIGUOUS Audit Trail]]
- [[_COMMUNITY_Community Detection|Community Detection]]
- [[_COMMUNITY_Detect Files Step|Detect Files Step]]
- [[_COMMUNITY_Existing-Graph Fast Path|Existing-Graph Fast Path]]
- [[_COMMUNITY_Gemini Extraction Backend|Gemini Extraction Backend]]
- [[_COMMUNITY_God Nodes|God Nodes]]
- [[_COMMUNITY_graph.json Output|graph.json Output]]
- [[_COMMUNITY_GRAPH_REPORT.md Output|GRAPH_REPORT.md Output]]
- [[_COMMUNITY_Python Interpreter Detection|Python Interpreter Detection]]
- [[_COMMUNITY_Knowledge Graph|Knowledge Graph]]
- [[_COMMUNITY_Obsidian Vault Export|Obsidian Vault Export]]
- [[_COMMUNITY_Semantic Extraction Cache|Semantic Extraction Cache]]
- [[_COMMUNITY_Semantic LLM Extraction|Semantic LLM Extraction]]
- [[_COMMUNITY_Parallel Subagent Dispatch|Parallel Subagent Dispatch]]
- [[_COMMUNITY_hydrabflow|hydrabflow]]
- [[_COMMUNITY_BaseSimulator|BaseSimulator]]
- [[_COMMUNITY_test_streams.py|test_streams.py]]
- [[_COMMUNITY_reporting.py|reporting.py]]
- [[_COMMUNITY_Stream project (compositional score modeling)|Stream project (compositional score modeling)]]
- [[_COMMUNITY_PerStreamParameterStandardize|PerStreamParameterStandardize]]
- [[_COMMUNITY_load_approximator|load_approximator]]
- [[_COMMUNITY_Running a full pipeline with the Two Moons simulator|Running a full pipeline with the Two Moons simulator]]
- [[_COMMUNITY__load_clean|_load_clean]]
- [[_COMMUNITY_prior_score_from_spec|prior_score_from_spec]]
- [[_COMMUNITY_ppc_prior_predictive.py|ppc_prior_predictive.py]]
- [[_COMMUNITY_corner_parameters.py|corner_parameters.py]]
- [[_COMMUNITY_StreamObservationStats|StreamObservationStats]]
- [[_COMMUNITY_test_streams.py|test_streams.py]]
- [[_COMMUNITY_load_approximator|load_approximator]]
- [[_COMMUNITY_compose_cfg|compose_cfg]]
- [[_COMMUNITY__vcirc_worker|_vcirc_worker]]
- [[_COMMUNITY_inferred_names|inferred_names]]
- [[_COMMUNITY_assetsgaia — portable static inputs for the stream project|assets/gaia — portable static inputs for the stream project]]

## God Nodes (most connected - your core abstractions)
1. `AgamaStreamSimulator` - 27 edges
2. `_jax()` - 24 edges
3. `PreprocessStep` - 23 edges
4. `_evaluate_compositional_global()` - 21 edges
5. `register_configs()` - 18 edges
6. `_evaluate_local()` - 17 edges
7. `_evaluate_real_compositional()` - 17 edges
8. `BaseSimulator` - 17 edges
9. `get_simulator()` - 17 edges
10. `build_workflow()` - 16 edges

## Surprising Connections (you probably didn't know these)
- `test_build_workflow()` --calls--> `build_workflow()`  [INFERRED]
  tests/test_workflow.py → src/hydrabflow/pipeline/workflow.py
- `test_unknown_simulator_errors()` --calls--> `get_simulator()`  [INFERRED]
  tests/test_registries.py → src/hydrabflow/simulators/registry.py
- `_vcirc_worker()` --calls--> `_agama()`  [INFERRED]
  scripts/extend_vcirc_huang.py → src/hydrabflow/simulators/stream_agama.py
- `_vcirc_worker()` --calls--> `_host_potential()`  [INFERRED]
  scripts/extend_vcirc_huang.py → src/hydrabflow/simulators/stream_agama.py
- `_vcirc_worker()` --calls--> `_vcirc()`  [INFERRED]
  scripts/extend_vcirc_huang.py → src/hydrabflow/simulators/stream_agama.py

## Import Cycles
- None detected.

## Communities (113 total, 35 thin omitted)

### Community 0 - "Preprocessing Pipeline & Steps"
Cohesion: 0.06
Nodes (34): ABC, PreprocessPipeline, PreprocessPipeline, PreprocessStep, Dataset, ndarray, Preprocessing step protocol and the pipeline that orchestrates them.  A :class:`, Element-wise (dataset-in, dataset-out) transform with optional fitted state. (+26 more)

### Community 1 - "Eval / Checkpoint Stages"
Cohesion: 0.33
Nodes (6): Model Default Config, Diffusion Inference Network Config, Flow Matching Inference Network Config, DeepSet Summary Network Config, SetTransformer Summary Network Config, TimeSeriesTransformer Summary Network Config

### Community 2 - "Design Principles & Configs"
Cohesion: 0.21
Nodes (10): _n(), Stage 2: training.  Load dataset -> preprocessing pipeline (fit on train, save f, Train the approximator and return (workflow, history)., Persist ``history.json`` + ``convergence.json``; best-effort, never fails a run., run_training(), _save_history_and_convergence(), _save_loss_plot(), Seeding helpers for reproducible runs. (+2 more)

### Community 3 - "Augmentation Registry & Tests"
Cohesion: 0.09
Nodes (32): feature_dropout(), gaussian_noise(), multiplicative_noise(), Augmentation, Example augmentations. Use as templates for problem-specific ones.  Augmentation, Add zero-mean Gaussian noise to one observable key (additive observational noise, Scale an observable by ``(1 + N(0, mult_scale))`` — multiplicative / gain jitter, Randomly zero out entries of an observable with probability ``dropout_prob`` (Be (+24 more)

### Community 4 - "Simulate Stage & Registries"
Cohesion: 0.06
Nodes (38): available_augmentations(), build_augmentations(), Augmentation, Name -> augmentation-factory registry and builder.  An augmentation factory rece, Build the ordered augmentation list from ``cfg.augmentation`` (an ``Augmentation, available_steps(), available_simulators(), Name -> simulator-class registry.  New simulators self-register with the ``@regi (+30 more)

### Community 5 - "Example Simulators (Skeleton/TwoMoons)"
Cohesion: 0.24
Nodes (4): Dataset, ndarray, Per-feature z-score standardization step.  Generalizes the reference project's `, Standardizer

### Community 6 - "Config Schemas"
Cohesion: 0.05
Nodes (20): BaseSimulator, BaseSimulator, Any, ndarray, Base interface every forward model implements.  A simulator is the ONLY piece a, Abstract forward model. Subclass + register via ``@register_simulator``., Ordered names of the inferred parameters (become ``inference_variables``)., Keys of the observable arrays. One key = single observable; >1 enables fusion. (+12 more)

### Community 7 - "Network Factory & Adapter"
Cohesion: 0.07
Nodes (28): 0. Prerequisites & install, 1. The five stages at a glance, 2. Changing the simulator, 2a. Write the simulator class, 2b. Registration is automatic, 2c. Add the simulator config, 2d. The adapter wires itself, 2e. Shape contract cheat-sheet (+20 more)

### Community 10 - "Config Composition Tests"
Cohesion: 0.09
Nodes (24): Logger, adapter_keys(), _as_list(), build_adapter(), fill_adapter_from_simulator(), Any, Build the BayesFlow ``Adapter`` from ``AdapterConfig``.  The adapter is the stru, All dataset keys the adapter (``cfg.adapter``) consumes, in a stable order. (+16 more)

### Community 11 - "Community 11"
Cohesion: 0.13
Nodes (6): AgamaStreamSimulator, Stellar streams in a parametrized Milky Way potential, simulated with AGAMA., Split radius for the extended (Zhou u Huang) rotation-curve grid., Prior spec of the inferred global parameters (used for compositional prior score, Per-stream prior spec of the inferred local parameters (drives their normalizati, Optional rotation-curve rejection prior (``params.vcirc_rejection``), else None.

### Community 12 - "Dataset IO"
Cohesion: 0.29
Nodes (12): concatenate_chunks(), load_chunk(), load_dataset(), _n_rows(), Dataset, Dataset IO. Datasets are ``.npz`` archives where each key maps to an array whose, Concatenate a list of dataset dicts along the leading (simulation) axis., Write a chunk ``.npz`` atomically (temp file + rename), so a crash mid-write can (+4 more)

### Community 13 - "Hydra App Boilerplate"
Cohesion: 0.33
Nodes (5): conf_path(), make_cli(), Shared Hydra-app boilerplate for the five run stages., Absolute path to the repo-root ``conf/`` directory., Wrap a ``run_fn(cfg)`` into a Hydra console entry point.      Registers the stru

### Community 14 - "JAX Backend Pin"
Cohesion: 0.13
Nodes (23): _bf_mmd(), member_summaries(), mmd_test(), _null_mmd(), per_member_scores(), ndarray, Summary-space model misspecification test (observed group vs simulated reference, Mahalanobis OOD score of each observed member vs its own stream's reference clou (+15 more)

### Community 15 - "Logging Helper"
Cohesion: 0.15
Nodes (20): build_inference_network(), build_summary_network(), _deep_set(), _diffusion(), _embed_dim(), _flow_matching(), Any, Build BayesFlow networks from structured dataclass configs (no ``_target_``).  B (+12 more)

### Community 31 - "Community 31"
Cohesion: 0.08
Nodes (23): For /graphify add and --watch, For /graphify query, For the commit hook and native CLAUDE.md integration, For --update and --cluster-only, /graphify, Honesty Rules, Interpreter guard for subcommands, Part A - Structural extraction for code files (+15 more)

### Community 32 - "Community 32"
Cohesion: 0.16
Nodes (19): _host_potential(), _ic_chen_spray(), _ic_particle_spray(), ndarray, Stellar-stream forward model built on AGAMA (CPU, parallelized with joblib).  Po, Jacobi radius, velocity offset, and host->satellite rotation matrices along the, Fardal+2015 initial conditions for particles escaping through the Lagrange point, Chen+2024 initial conditions: one trailing + one leading particle per orbit seed (+11 more)

### Community 33 - "Community 33"
Cohesion: 0.06
Nodes (35): A.1 The data contract, A.2 Convert your existing arrays into the dataset file, A.3 Tell the pipeline about it (config only), A.4 Run train + evaluate, A.5 What reads what, B.1 The single seam, B.2 Option 1 — Quick swap (one format, replace the body), B.3 Option 2 — A format registry (support several formats by extension) (+27 more)

### Community 34 - "Community 34"
Cohesion: 0.17
Nodes (11): Core Design Principles, Decisions Log, Folder Structure (finalized), Goal, graphify, HydraBFlow: SBI Pipeline Template with BayesFlow, Output Directory Convention, Run stages (6 entry points) (+3 more)

### Community 35 - "Community 35"
Cohesion: 0.18
Nodes (6): AttachObservedVcirc, MaskVcircRadii, Stream-specific preprocessing: per-stream normalization and rotation-curve trimm, Attach the *observed* Milky Way rotation curve to a real dataset that lacks one., Keep only rotation-curve bins with ``r >= r_min`` on the observed radii grid., test_mask_vcirc_radii_trims_grid()

### Community 37 - "Community 37"
Cohesion: 0.07
Nodes (30): 1. How the config system works, 2. The root master config — `config.yaml`, 3.10 `tuning/`, 3.1 `simulator/`, 3.2 `model/`, 3.3 `data/`, 3.4 `training/`, 3.5 `preprocessing/` (+22 more)

### Community 38 - "streams.py"
Cohesion: 0.13
Nodes (33): _add_noise_to_vcirc(), _apply_obs_error(), _compact_to_attended(), _concatenate_magnitudes(), _concatenate_sigma_errors(), _concatenate_stream_index(), _concatenate_vlos_mask(), _convert_distance_to_parallax() (+25 more)

### Community 39 - "Community 39"
Cohesion: 0.24
Nodes (9): build_pipeline(), Name -> preprocessing-step registry and pipeline builder., Register a step factory (usually the step class itself) under ``name``., Build a :class:`PreprocessPipeline` from ``cfg.preprocessing`` (a ``Preprocessin, register_step(), Preprocessing pipeline: fit/transform/split + state save/load round-trip., test_pipeline_fit_transform_and_split(), test_state_roundtrip() (+1 more)

### Community 40 - "Community 40"
Cohesion: 0.26
Nodes (16): _evaluate_compositional_global(), _evaluate_local(), _load_test_data(), Stage 3: evaluation on a simulated test set (with known ground truth).  Loads th, Global-level evaluation on a grouped (multistream) test set, both ways:      * *, Local-level evaluation: per-member sampling conditioned on the true globals., Best-effort ``report.md`` from the metrics/figures just written; never aborts a, Load the held-out test set and replay the *fitted* preprocessing (no re-fit, no (+8 more)

### Community 46 - "stream_common.py"
Cohesion: 0.29
Nodes (7): _band_pass_worker(), main(), ndarray, Measure the vcirc-rejection acceptance rate of a stream simulator's prior (calib, (n_rows, n_bands) bool: does each row's model curve pass each band? (one vc eval, _agama(), Import agama with the (kpc, km/s, Msun) unit system set. Safe to call repeatedly

### Community 47 - "MaskedFusionNetwork"
Cohesion: 0.50
Nodes (4): apply_bayesflow_patches(), _patch_compositional_condition_reshape(), Targeted runtime fixes for known BayesFlow bugs (version-checked, applied once)., Idempotently install the fixes. Called when a compositional workflow is built.

### Community 54 - "compositional.py"
Cohesion: 0.19
Nodes (12): apply_augmentations_once(), composition_level(), flatten_members(), group_members(), log10_keys_from_pipeline(), ndarray, Helpers for compositional (grouped) evaluation.  A compositional dataset stores, Inverse of :func:`flatten_members` for arrays of ``n*m`` rows -> ``(n, m, ...)`` (+4 more)

### Community 56 - "Community 56"
Cohesion: 0.15
Nodes (20): fix_keras_model(), load_approximator(), Any, Model save/load helpers, including the BayesFlow ``.keras`` deserialization work, Return a path to a load-safe copy of ``model_path`` (patching the ArrayImpl tag), Load a saved approximator, applying the ArrayImpl fix first., save_approximator(), condition_keys() (+12 more)

### Community 57 - "Community 57"
Cohesion: 0.50
Nodes (3): import_submodules(), Auto-import the modules of a package so ``@register_*`` decorators run.  The reg, Import every non-underscore module directly inside a package.      Call from a p

### Community 58 - "run_tuning"
Cohesion: 0.29
Nodes (10): _objective(), Stage 4: hyperparameter tuning with Optuna.  Runs a (by default multi-objective), Per-trial artifact directory, keyed by the study-global Optuna trial number., Save the fit-once preprocessing state, shared by every trial/model.      Written, _report(), run_tuning(), _save_shared_preprocessing(), _suggest() (+2 more)

### Community 59 - "build_workflow"
Cohesion: 0.33
Nodes (5): build_workflow(), Any, Assemble the BayesFlow workflow from config.  Single-level inference (the defaul, Build a ``bf.BasicWorkflow`` from the root ``cfg``., Build a ``bf.BasicWorkflow`` (or ``bf.CompositionalWorkflow``) from the root ``c

### Community 60 - "MaskedFusionNetwork"
Cohesion: 0.16
Nodes (9): Layer, Shape, _fusion(), MaskedFusionNetwork, Multi-observable fusion summary network (attention-mask aware).  Consumes the di, Build a :class:`MaskedFusionNetwork` from ``cfg.params`` (see module docstring)., Fuse one summary backbone per named input; route the attention mask to one of th, SummaryNetwork (+1 more)

### Community 96 - "test_streams.py"
Cohesion: 0.14
Nodes (12): Stage 1b: compositional (grouped) dataset generation.  Like :mod:`simulate`, but, Generate the compositional dataset described by ``cfg`` and return its path., run_multistream_simulation(), Stage 1: dataset generation.  Samples the prior and runs the forward model in ch, Generate the dataset described by ``cfg`` and return its path., run_simulation(), get_simulator(), Instantiate the simulator selected by ``cfg.simulator`` (a ``SimulatorConfig``). (+4 more)

### Community 97 - "reporting.py"
Cohesion: 0.20
Nodes (13): _finite(), _has_nonfinite(), inspect_history(), _load_json(), _metrics_table(), Any, _rate(), Training-convergence inspection and Markdown report generation.  Two pure-Python (+5 more)

### Community 98 - "Stream project (compositional score modeling)"
Cohesion: 0.14
Nodes (14): 1. What more flexible potential family, 2. More realistic stream simulation with agama — straight from their bundled examples, Adding your own simulator, Configuration used explicitly, Data generation, Design at a glance, Evaluation (plots + metrics), Future work (TODO) (+6 more)

### Community 99 - "PerStreamParameterStandardize"
Cohesion: 0.25
Nodes (5): PerStreamParameterStandardize, ndarray, z-score each stream's local parameters with that stream's prior mean/std., test_per_stream_parameter_standardize_rejects_non_normal(), test_per_stream_parameter_standardize_roundtrip()

### Community 100 - "load_approximator"
Cohesion: 0.10
Nodes (23): _plummer_sample(), ndarray, Restricted N-body stellar-stream forward model on AGAMA (CPU, joblib).  Same pri, joblib worker: one restricted-N-body stream + the rotation curve of its potentia, Stellar streams via restricted N-body (agama example_tidal_stream method)., Positions/velocities (relative to the center) of an isotropic Plummer sphere., Restricted N-body stream: forward-integrate Plummer particles from the rewound o, RestrictedNbodyStreamSimulator (+15 more)

### Community 101 - "Running a full pipeline with the Two Moons simulator"
Cohesion: 0.29
Nodes (4): Config composition + schema validation smoke tests., test_adapter_derived_from_simulator(), test_adapter_explicit_config_wins(), test_group_override()

### Community 102 - "_load_clean"
Cohesion: 0.43
Nodes (6): _load_clean(), main(), ndarray, Cross-model posterior tension report (offline analysis helper — not a Hydra run, Load a posterior .npz as {param: (n_datasets, n_samples)} float arrays., _resolve()

### Community 103 - "prior_score_from_spec"
Cohesion: 0.25
Nodes (8): Container, prior_score_from_spec(), Score of the log prior for compositional sampling, from a prior-spec mapping., The callable's signature names a ``time`` parameter, so BayesFlow's     ``prior_, d/dy log p_Y(y) for y=log10(x) must include the +ln(10) Jacobian term, not just, test_prior_score_applies_time_decay(), test_prior_score_from_spec(), test_prior_score_log10_jacobian_correction()

### Community 104 - "ppc_prior_predictive.py"
Cohesion: 0.20
Nodes (15): main(), ndarray, Overlay two particle-spray recipes (Fardal+2015 vs Chen+2024) against the real G, (sky RA*cos(dec), Dec) and (pm_ra_cosdec, pm_dec) for the finite particles of st, _stream_xy_pm(), load_dataset(), main(), ndarray (+7 more)

### Community 105 - "corner_parameters.py"
Cohesion: 0.22
Nodes (12): NpzFile, autoscale(), detect_param_keys(), expand_inputs(), load_columns(), main(), ndarray, Corner plot of parameter draws from a simulate dataset (offline helper — not a H (+4 more)

### Community 106 - "StreamObservationStats"
Cohesion: 0.24
Nodes (6): Dataset, Fit per-stream observation stats + log10(vcirc) per-bin stats on the (clean) tra, Integer stream ids broadcastable against ``like``.      ``j``'s leading axes alw, _stream_index(), StreamObservationStats, test_stream_observation_stats_fit_and_state_roundtrip()

### Community 107 - "test_streams.py"
Cohesion: 0.33
Nodes (8): compose(), Expose the composer so tests can build configs with custom overrides., Stream-project components: config composition, hierarchy derivation, per-stream, test_adapter_derivation_follows_composition_level(), test_simulator_declares_hierarchy(), test_stream_config_composes(), test_stream_global_log10_and_nolos_presets_compose(), test_stream_noerr_and_nolos_variants_compose()

### Community 108 - "load_approximator"
Cohesion: 0.29
Nodes (3): Resolve ``vcirc_rejection`` into a list of band specs for the accept worker., Screen each draw's global potential against the observed rotation curve., Draw with ``sample_fn`` until ``n`` rows pass the rotation-curve cut.

### Community 109 - "compose_cfg"
Cohesion: 0.50
Nodes (4): cfg(), compose_cfg(), Shared test fixtures., Compose the root config with the structured schemas registered.      ``fill=True

### Community 110 - "_vcirc_worker"
Cohesion: 0.43
Nodes (7): main(), plot_curve(), ndarray, Extend a stream dataset's rotation-curve observable onto larger radii (offline h, joblib worker: model rotation curve on ``obs_r`` for a chunk of parameter rows., recompute_vcirc(), _vcirc_worker()

### Community 111 - "inferred_names"
Cohesion: 0.29
Nodes (6): fill_stream_grid_from_simulator(), Align the training-time rotation-curve grid with the simulator's ``vcirc_kms`` g, Radii the model rotation curve is evaluated on (also the ``vcirc_kms`` grid)., Per-bin observed 1-sigma on the rotation curve, aligned with ``obs_r_kpc``., extended_rotation_curve(), Union rotation-curve grid: Zhou (2023) up to ``split_kpc``, Huang (2016) beyond

### Community 117 - "assets/gaia — portable static inputs for the stream project"
Cohesion: 0.50
Nodes (3): assets/gaia — portable static inputs for the stream project, Contents, Using these on a new cluster

## Knowledge Gaps
- **146 isolated node(s):** `hydrabflow`, `graphify`, `Usage`, `What graphify is for`, `Step 0 - GitHub repos and multi-path merge (only if a URL or several paths)` (+141 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **35 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `get_simulator()` connect `test_streams.py` to `Simulate Stage & Registries`, `Config Schemas`, `Community 40`, `Config Composition Tests`, `test_streams.py`, `stream_common.py`, `inferred_names`, `JAX Backend Pin`, `Community 56`?**
  _High betweenness centrality (0.132) - this node is a cross-community bridge._
- **Why does `BaseSimulator` connect `Config Schemas` to `Preprocessing Pipeline & Steps`, `Community 11`, `test_streams.py`?**
  _High betweenness centrality (0.108) - this node is a cross-community bridge._
- **Why does `PreprocessStep` connect `Preprocessing Pipeline & Steps` to `PerStreamParameterStandardize`, `StreamObservationStats`, `Community 35`, `Example Simulators (Skeleton/TwoMoons)`?**
  _High betweenness centrality (0.088) - this node is a cross-community bridge._
- **Are the 11 inferred relationships involving `PreprocessStep` (e.g. with `PreprocessPipeline` and `Standardizer`) actually correct?**
  _`PreprocessStep` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 12 inferred relationships involving `_evaluate_compositional_global()` (e.g. with `load_approximator()` and `apply_augmentations_once()`) actually correct?**
  _`_evaluate_compositional_global()` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Are the 16 inferred relationships involving `register_configs()` (e.g. with `main()` and `AdapterConfig`) actually correct?**
  _`register_configs()` has 16 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Marimo notebook: inspect a training run's posterior samples and diagnostics.  Ru`, `hydrabflow`, `Overlay two particle-spray recipes (Fardal+2015 vs Chen+2024) against the real G` to the rest of the system?**
  _368 weakly-connected nodes found - possible documentation gaps or missing edges._
# Graph Report - HydraBFlow  (2026-07-09)

## Corpus Check
- 87 files · ~357,898 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 979 nodes · 1414 edges · 115 communities (76 shown, 39 thin omitted)
- Extraction: 84% EXTRACTED · 16% INFERRED · 0% AMBIGUOUS · INFERRED: 222 edges (avg confidence: 0.74)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `6cd97c92`
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
- [[_COMMUNITY_Part B — Support a file format other than `.npz`|Part B — Support a file format other than `.npz`]]
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
- [[_COMMUNITY_ppc_prior_predictive.py|ppc_prior_predictive.py]]
- [[_COMMUNITY_corner_parameters.py|corner_parameters.py]]
- [[_COMMUNITY_StreamObservationStats|StreamObservationStats]]
- [[_COMMUNITY_test_streams.py|test_streams.py]]
- [[_COMMUNITY_2. The four commands (full run)|2. The four commands (full run)]]
- [[_COMMUNITY_compose_cfg|compose_cfg]]
- [[_COMMUNITY__vcirc_worker|_vcirc_worker]]
- [[_COMMUNITY_create_rnbody_huang_dataset.sh|create_rnbody_huang_dataset.sh]]
- [[_COMMUNITY_training_eval_missing_vlos_ablation.sh|training_eval_missing_vlos_ablation.sh]]
- [[_COMMUNITY_training_eval_rnbody_huand_dataset.sh|training_eval_rnbody_huand_dataset.sh]]
- [[_COMMUNITY_assetsgaia — portable static inputs for the stream project|assets/gaia — portable static inputs for the stream project]]

## God Nodes (most connected - your core abstractions)
1. `AgamaStreamSimulator` - 29 edges
2. `_jax()` - 25 edges
3. `PreprocessStep` - 23 edges
4. `_evaluate_compositional_global()` - 21 edges
5. `register_configs()` - 18 edges
6. `compose()` - 18 edges
7. `_evaluate_local()` - 17 edges
8. `_evaluate_real_compositional()` - 17 edges
9. `BaseSimulator` - 17 edges
10. `get_simulator()` - 17 edges

## Surprising Connections (you probably didn't know these)
- `test_build_workflow()` --calls--> `build_workflow()`  [INFERRED]
  tests/test_workflow.py → src/hydrabflow/pipeline/workflow.py
- `test_two_moons_shapes_and_reproducibility()` --calls--> `get_simulator()`  [INFERRED]
  tests/test_augmentation.py → src/hydrabflow/simulators/registry.py
- `test_unknown_simulator_errors()` --calls--> `get_simulator()`  [INFERRED]
  tests/test_registries.py → src/hydrabflow/simulators/registry.py
- `main()` --calls--> `register_configs()`  [INFERRED]
  scripts/probe_vcirc_acceptance.py → src/hydrabflow/config/schema.py
- `main()` --calls--> `get_simulator()`  [INFERRED]
  scripts/probe_vcirc_acceptance.py → src/hydrabflow/simulators/registry.py

## Import Cycles
- None detected.

## Communities (115 total, 39 thin omitted)

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
Nodes (39): available_augmentations(), build_augmentations(), Augmentation, Name -> augmentation-factory registry and builder.  An augmentation factory rece, Build the ordered augmentation list from ``cfg.augmentation`` (an ``Augmentation, available_steps(), Name -> preprocessing-step registry and pipeline builder., Register a step factory (usually the step class itself) under ``name``. (+31 more)

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
Cohesion: 0.05
Nodes (39): fill_stream_grid_from_simulator(), Align the training-time rotation-curve grid with the simulator's ``vcirc_kms`` g, _agama(), AgamaStreamSimulator, _host_potential(), _ic_chen_spray(), _ic_particle_spray(), ndarray (+31 more)

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
Cohesion: 0.12
Nodes (23): build_inference_network(), build_summary_network(), _deep_set(), _diffusion(), _embed_dim(), _flow_matching(), Any, Build BayesFlow networks from structured dataclass configs (no ``_target_``).  B (+15 more)

### Community 31 - "Community 31"
Cohesion: 0.08
Nodes (23): For /graphify add and --watch, For /graphify query, For the commit hook and native CLAUDE.md integration, For --update and --cluster-only, /graphify, Honesty Rules, Interpreter guard for subcommands, Part A - Structural extraction for code files (+15 more)

### Community 32 - "Community 32"
Cohesion: 0.33
Nodes (5): fix_keras_model(), Any, Model save/load helpers, including the BayesFlow ``.keras`` deserialization work, Return a path to a load-safe copy of ``model_path`` (patching the ArrayImpl tag), save_approximator()

### Community 33 - "Community 33"
Cohesion: 0.20
Nodes (10): 1. Prerequisites, 2. Run a study, 3. What gets saved, 4. Run many processes at once (parallel tuning), 5. Reading the results, 6. Changing what is tuned (the search space), 7. Key config reference (`tuning` group), 8. Command recap (+2 more)

### Community 34 - "Community 34"
Cohesion: 0.17
Nodes (11): Core Design Principles, Decisions Log, Folder Structure (finalized), Goal, graphify, HydraBFlow: SBI Pipeline Template with BayesFlow, Output Directory Convention, Run stages (6 entry points) (+3 more)

### Community 35 - "Community 35"
Cohesion: 0.09
Nodes (17): AttachObservedVcirc, MaskVcircRadii, PerStreamParameterStandardize, Dataset, ndarray, Stream-specific preprocessing: per-stream normalization and rotation-curve trimm, Fit per-stream observation stats + log10(vcirc) per-bin stats on the (clean) tra, Attach the *observed* Milky Way rotation curve to a real dataset that lacks one. (+9 more)

### Community 37 - "Community 37"
Cohesion: 0.07
Nodes (30): 1. How the config system works, 2. The root master config — `config.yaml`, 3.10 `tuning/`, 3.1 `simulator/`, 3.2 `model/`, 3.3 `data/`, 3.4 `training/`, 3.5 `preprocessing/` (+22 more)

### Community 38 - "streams.py"
Cohesion: 0.11
Nodes (39): _add_noise_to_vcirc(), _apply_obs_error(), _compact_to_attended(), _concatenate_magnitudes(), _concatenate_sigma_errors(), _concatenate_stream_index(), _concatenate_vlos_mask(), _convert_distance_to_parallax() (+31 more)

### Community 39 - "Community 39"
Cohesion: 0.43
Nodes (6): build_pipeline(), Build a :class:`PreprocessPipeline` from ``cfg.preprocessing`` (a ``Preprocessin, Preprocessing pipeline: fit/transform/split + state save/load round-trip., test_pipeline_fit_transform_and_split(), test_state_roundtrip(), _toy_data()

### Community 40 - "Community 40"
Cohesion: 0.30
Nodes (14): _evaluate_compositional_global(), _evaluate_local(), _load_test_data(), Stage 3: evaluation on a simulated test set (with known ground truth).  Loads th, Global-level evaluation on a grouped (multistream) test set, both ways:      * *, Local-level evaluation: per-member sampling conditioned on the true globals., Best-effort ``report.md`` from the metrics/figures just written; never aborts a, Load the held-out test set and replay the *fitted* preprocessing (no re-fit, no (+6 more)

### Community 46 - "Part B — Support a file format other than `.npz`"
Cohesion: 0.29
Nodes (7): B.1 The single seam, B.2 Option 1 — Quick swap (one format, replace the body), B.3 Option 2 — A format registry (support several formats by extension), B.4 (Optional) validate keys/shapes on load, Bring your own data (no simulator) & supporting other file formats, Checklist, Part B — Support a file format other than `.npz`

### Community 47 - "MaskedFusionNetwork"
Cohesion: 0.50
Nodes (4): apply_bayesflow_patches(), _patch_compositional_condition_reshape(), Targeted runtime fixes for known BayesFlow bugs (version-checked, applied once)., Idempotently install the fixes. Called when a compositional workflow is built.

### Community 54 - "compositional.py"
Cohesion: 0.19
Nodes (12): apply_augmentations_once(), composition_level(), flatten_members(), group_members(), log10_keys_from_pipeline(), ndarray, Helpers for compositional (grouped) evaluation.  A compositional dataset stores, Inverse of :func:`flatten_members` for arrays of ``n*m`` rows -> ``(n, m, ...)`` (+4 more)

### Community 56 - "Community 56"
Cohesion: 0.19
Nodes (17): load_approximator(), Load a saved approximator, applying the ArrayImpl fix first., condition_keys(), Raw batch keys that act as sampling conditions (everything the adapter consumes, _evaluate_real_compositional(), _max_particles(), _prepare_real_members(), Stage 5: application to real (observed) data.  Like :mod:`evaluate`, but the inp (+9 more)

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
Cohesion: 0.08
Nodes (16): Shape, _fusion(), MaskedFusionNetwork, Layer, Tensor, Multi-observable fusion summary network (attention-mask aware).  Consumes the di, Build a :class:`MaskedFusionNetwork` from ``cfg.params`` (see module docstring)., Fuse one summary backbone per named input; route the attention mask to one of th (+8 more)

### Community 96 - "test_streams.py"
Cohesion: 0.15
Nodes (11): Stage 1b: compositional (grouped) dataset generation.  Like :mod:`simulate`, but, Generate the compositional dataset described by ``cfg`` and return its path., run_multistream_simulation(), Stage 1: dataset generation.  Samples the prior and runs the forward model in ch, Generate the dataset described by ``cfg`` and return its path., run_simulation(), get_run_dir(), Run-directory helpers and shared artifact filenames. (+3 more)

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
Cohesion: 0.10
Nodes (26): Exception, _assert_reached(), _OrbitCapExceeded, _plummer_sample(), ndarray, Restricted N-body stellar-stream forward model on AGAMA (CPU, joblib).  Same pri, Restricted N-body stream: forward-integrate Plummer particles from the rewound o, joblib worker: one restricted-N-body stream + the rotation curve of its potentia (+18 more)

### Community 101 - "Running a full pipeline with the Two Moons simulator"
Cohesion: 0.29
Nodes (7): Configuration used explicitly, Data generation, Evaluation (plots + metrics), GPU, Hyperparameter tuning, Stream project (compositional score modeling), Training

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
Cohesion: 0.06
Nodes (45): Container, prior_score_from_spec(), Score of the log prior for compositional sampling, from a prior-spec mapping., cfg(), compose(), compose_cfg(), Shared test fixtures., Compose the root config with the structured schemas registered.      ``fill=True (+37 more)

### Community 108 - "2. The four commands (full run)"
Cohesion: 0.40
Nodes (5): 2.1 Generate the training set, 2.2 Generate a held-out test set, 2.3 Train, 2.4 Evaluate, 2. The four commands (full run)

### Community 110 - "_vcirc_worker"
Cohesion: 0.13
Nodes (23): main(), plot_curve(), ndarray, Extend a stream dataset's rotation-curve observable onto larger radii (offline h, joblib worker: model rotation curve on ``obs_r`` for a chunk of parameter rows., recompute_vcirc(), _vcirc_worker(), _agama() (+15 more)

### Community 117 - "assets/gaia — portable static inputs for the stream project"
Cohesion: 0.50
Nodes (3): assets/gaia — portable static inputs for the stream project, Contents, Using these on a new cluster

## Knowledge Gaps
- **151 isolated node(s):** `hydrabflow`, `create_rnbody_huang_dataset.sh script`, `train_eval_base_cpu.sh script`, `JAX_PLATFORMS`, `training_eval_missing_vlos_ablation.sh script` (+146 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **39 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `get_simulator()` connect `Community 56` to `test_streams.py`, `Simulate Stage & Registries`, `Config Schemas`, `Community 40`, `Config Composition Tests`, `Community 11`, `test_streams.py`, `_vcirc_worker`, `JAX Backend Pin`, `Logging Helper`?**
  _High betweenness centrality (0.151) - this node is a cross-community bridge._
- **Why does `BaseSimulator` connect `Config Schemas` to `Preprocessing Pipeline & Steps`, `Community 56`, `Community 11`?**
  _High betweenness centrality (0.113) - this node is a cross-community bridge._
- **Why does `main()` connect `_vcirc_worker` to `Community 56`, `test_streams.py`, `Augmentation Registry & Tests`, `load_approximator`?**
  _High betweenness centrality (0.103) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `AgamaStreamSimulator` (e.g. with `_OrbitCapExceeded` and `RestrictedNbodyStreamSimulator`) actually correct?**
  _`AgamaStreamSimulator` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 11 inferred relationships involving `PreprocessStep` (e.g. with `PreprocessPipeline` and `Standardizer`) actually correct?**
  _`PreprocessStep` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 12 inferred relationships involving `_evaluate_compositional_global()` (e.g. with `load_approximator()` and `apply_augmentations_once()`) actually correct?**
  _`_evaluate_compositional_global()` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Are the 16 inferred relationships involving `register_configs()` (e.g. with `main()` and `AdapterConfig`) actually correct?**
  _`register_configs()` has 16 INFERRED edges - model-reasoned connections that need verification._
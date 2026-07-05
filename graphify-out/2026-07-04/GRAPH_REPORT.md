# Graph Report - HydraBFlow  (2026-07-03)

## Corpus Check
- 69 files · ~39,364 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 735 nodes · 983 edges · 96 communities (61 shown, 35 thin omitted)
- Extraction: 84% EXTRACTED · 16% INFERRED · 0% AMBIGUOUS · INFERRED: 162 edges (avg confidence: 0.74)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `f1cd7cd9`
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
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_MaskedFusionNetwork|MaskedFusionNetwork]]
- [[_COMMUNITY_compositional.py|compositional.py]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 57|Community 57]]
- [[_COMMUNITY_run_tuning|run_tuning]]
- [[_COMMUNITY_build_workflow|build_workflow]]
- [[_COMMUNITY_apply_bayesflow_patches|apply_bayesflow_patches]]
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

## God Nodes (most connected - your core abstractions)
1. `PreprocessStep` - 22 edges
2. `AgamaStreamSimulator` - 20 edges
3. `_evaluate_compositional_global()` - 18 edges
4. `register_configs()` - 17 edges
5. `BaseSimulator` - 17 edges
6. `_evaluate_local()` - 16 edges
7. `build_workflow()` - 16 edges
8. `_sim_key()` - 13 edges
9. `MaskedFusionNetwork` - 13 edges
10. `run_evaluation()` - 13 edges

## Surprising Connections (you probably didn't know these)
- `test_build_workflow()` --calls--> `build_workflow()`  [INFERRED]
  tests/test_workflow.py → src/hydrabflow/pipeline/workflow.py
- `test_augmentation_registry_builds()` --calls--> `build_augmentations()`  [INFERRED]
  tests/test_registries.py → src/hydrabflow/augmentation/registry.py
- `compose_cfg()` --calls--> `register_configs()`  [INFERRED]
  tests/conftest.py → src/hydrabflow/config/schema.py
- `test_build_adapter()` --calls--> `build_adapter()`  [INFERRED]
  tests/test_workflow.py → src/hydrabflow/pipeline/adapter.py
- `test_prior_score_from_spec()` --calls--> `prior_score_from_spec()`  [INFERRED]
  tests/test_streams.py → src/hydrabflow/pipeline/compositional.py

## Import Cycles
- None detected.

## Communities (96 total, 35 thin omitted)

### Community 0 - "Preprocessing Pipeline & Steps"
Cohesion: 0.07
Nodes (31): ABC, PreprocessPipeline, PreprocessPipeline, PreprocessStep, Dataset, ndarray, Preprocessing step protocol and the pipeline that orchestrates them.  A :class:`, Element-wise (dataset-in, dataset-out) transform with optional fitted state. (+23 more)

### Community 1 - "Eval / Checkpoint Stages"
Cohesion: 0.33
Nodes (6): Model Default Config, Diffusion Inference Network Config, Flow Matching Inference Network Config, DeepSet Summary Network Config, SetTransformer Summary Network Config, TimeSeriesTransformer Summary Network Config

### Community 2 - "Design Principles & Configs"
Cohesion: 0.20
Nodes (10): fix_keras_model(), Any, Model save/load helpers, including the BayesFlow ``.keras`` deserialization work, Return a path to a load-safe copy of ``model_path`` (patching the ArrayImpl tag), save_approximator(), _n(), Stage 2: training.  Load dataset -> preprocessing pipeline (fit on train, save f, Train the approximator and return (workflow, history). (+2 more)

### Community 3 - "Augmentation Registry & Tests"
Cohesion: 0.09
Nodes (32): feature_dropout(), gaussian_noise(), multiplicative_noise(), Augmentation, Example augmentations. Use as templates for problem-specific ones.  Augmentation, Add zero-mean Gaussian noise to one observable key (additive observational noise, Scale an observable by ``(1 + N(0, mult_scale))`` — multiplicative / gain jitter, Randomly zero out entries of an observable with probability ``dropout_prob`` (Be (+24 more)

### Community 4 - "Simulate Stage & Registries"
Cohesion: 0.05
Nodes (45): available_augmentations(), build_augmentations(), Augmentation, Name -> augmentation-factory registry and builder.  An augmentation factory rece, Build the ordered augmentation list from ``cfg.augmentation`` (an ``Augmentation, available_steps(), Name -> preprocessing-step registry and pipeline builder., Register a step factory (usually the step class itself) under ``name``. (+37 more)

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
Nodes (25): adapter_keys(), _as_list(), build_adapter(), fill_adapter_from_simulator(), Any, Build the BayesFlow ``Adapter`` from ``AdapterConfig``.  The adapter is the stru, Construct ``bf.adapters.Adapter`` from ``cfg`` (an ``AdapterConfig``)., Fill empty adapter variable lists from the simulator's own declaration (in place (+17 more)

### Community 11 - "Community 11"
Cohesion: 0.15
Nodes (11): Stage 1b: compositional (grouped) dataset generation.  Like :mod:`simulate`, but, Generate the compositional dataset described by ``cfg`` and return its path., run_multistream_simulation(), Stage 1: dataset generation.  Samples the prior and runs the forward model in ch, Generate the dataset described by ``cfg`` and return its path., run_simulation(), get_run_dir(), Run-directory helpers and shared artifact filenames. (+3 more)

### Community 12 - "Dataset IO"
Cohesion: 0.38
Nodes (6): concatenate_chunks(), load_dataset(), Dataset, Dataset IO. Datasets are ``.npz`` archives where each key maps to an array whose, Concatenate a list of dataset dicts along the leading (simulation) axis., save_dataset()

### Community 13 - "Hydra App Boilerplate"
Cohesion: 0.33
Nodes (5): conf_path(), make_cli(), Shared Hydra-app boilerplate for the five run stages., Absolute path to the repo-root ``conf/`` directory., Wrap a ``run_fn(cfg)`` into a Hydra console entry point.      Registers the stru

### Community 14 - "JAX Backend Pin"
Cohesion: 0.20
Nodes (9): Logger, limit_gpus(), Pin compute settings *before* keras/bayesflow/JAX are imported anywhere.  Two th, Pin ``CUDA_VISIBLE_DEVICES`` to the least-used GPU(s) before JAX/CUDA initialize, Set ``KERAS_BACKEND`` unless the user already chose one. Returns the active back, set_backend(), get_logger(), Minimal logging helper so all pipeline stages log consistently. (+1 more)

### Community 15 - "Logging Helper"
Cohesion: 0.06
Nodes (34): _agama(), AgamaStreamSimulator, _host_potential(), _ic_particle_spray(), ndarray, Stellar-stream forward model built on AGAMA (CPU, parallelized with joblib).  Po, Fardal+2015 initial conditions for particles escaping through the Lagrange point, Particle-spray stream including the progenitor's own (moving Plummer) potential. (+26 more)

### Community 31 - "Community 31"
Cohesion: 0.08
Nodes (23): For /graphify add and --watch, For /graphify query, For the commit hook and native CLAUDE.md integration, For --update and --cluster-only, /graphify, Honesty Rules, Interpreter guard for subcommands, Part A - Structural extraction for code files (+15 more)

### Community 32 - "Community 32"
Cohesion: 0.15
Nodes (13): A.1 The data contract, A.2 Convert your existing arrays into the dataset file, A.3 Tell the pipeline about it (config only), A.4 Run train + evaluate, A.5 What reads what, B.1 The single seam, B.2 Option 1 — Quick swap (one format, replace the body), B.3 Option 2 — A format registry (support several formats by extension) (+5 more)

### Community 33 - "Community 33"
Cohesion: 0.08
Nodes (26): 1. Prerequisites, 2. Run a study, 3. What gets saved, 4. Run many processes at once (parallel tuning), 5. Reading the results, 6. Changing what is tuned (the search space), 7. Key config reference (`tuning` group), 8. Command recap (+18 more)

### Community 34 - "Community 34"
Cohesion: 0.17
Nodes (11): Core Design Principles, Decisions Log, Folder Structure (finalized), Goal, graphify, HydraBFlow: SBI Pipeline Template with BayesFlow, Output Directory Convention, Run stages (6 entry points) (+3 more)

### Community 35 - "Community 35"
Cohesion: 0.08
Nodes (20): AttachObservedVcirc, MaskVcircRadii, PerStreamParameterStandardize, Dataset, ndarray, Stream-specific preprocessing: per-stream normalization and rotation-curve trimm, Fit per-stream observation stats + log10(vcirc) per-bin stats on the (clean) tra, Attach the *observed* Milky Way rotation curve to a real dataset that lacks one. (+12 more)

### Community 37 - "Community 37"
Cohesion: 0.07
Nodes (30): 1. How the config system works, 2. The root master config — `config.yaml`, 3.10 `tuning/`, 3.1 `simulator/`, 3.2 `model/`, 3.3 `data/`, 3.4 `training/`, 3.5 `preprocessing/` (+22 more)

### Community 38 - "streams.py"
Cohesion: 0.11
Nodes (27): _add_noise_to_vcirc(), _apply_obs_error(), _compact_to_attended(), _concatenate_magnitudes(), _concatenate_sigma_errors(), _concatenate_stream_index(), _concatenate_vlos_mask(), _convert_distance_to_parallax() (+19 more)

### Community 39 - "Community 39"
Cohesion: 0.43
Nodes (6): build_pipeline(), Build a :class:`PreprocessPipeline` from ``cfg.preprocessing`` (a ``Preprocessin, Preprocessing pipeline: fit/transform/split + state save/load round-trip., test_pipeline_fit_transform_and_split(), test_state_roundtrip(), _toy_data()

### Community 40 - "Community 40"
Cohesion: 0.21
Nodes (17): condition_keys(), Raw batch keys that act as sampling conditions (everything the adapter consumes, _evaluate_compositional_global(), _evaluate_local(), _load_test_data(), Stage 3: evaluation on a simulated test set (with known ground truth).  Loads th, Global-level compositional evaluation on a grouped (multistream) test set., Local-level evaluation: per-member sampling conditioned on the true globals. (+9 more)

### Community 46 - "Community 46"
Cohesion: 0.15
Nodes (20): build_inference_network(), build_summary_network(), _deep_set(), _diffusion(), _embed_dim(), _flow_matching(), Any, Build BayesFlow networks from structured dataclass configs (no ``_target_``).  B (+12 more)

### Community 47 - "MaskedFusionNetwork"
Cohesion: 0.16
Nodes (9): Layer, Shape, _fusion(), MaskedFusionNetwork, Multi-observable fusion summary network (attention-mask aware).  Consumes the di, Build a :class:`MaskedFusionNetwork` from ``cfg.params`` (see module docstring)., Fuse one summary backbone per named input; route the attention mask to one of th, SummaryNetwork (+1 more)

### Community 54 - "compositional.py"
Cohesion: 0.18
Nodes (13): apply_augmentations_once(), composition_level(), flatten_members(), group_members(), prior_score_from_spec(), ndarray, Helpers for compositional (grouped) evaluation.  A compositional dataset stores, Score of the log prior for compositional sampling, from a prior-spec mapping. (+5 more)

### Community 56 - "Community 56"
Cohesion: 0.29
Nodes (11): load_approximator(), Load a saved approximator, applying the ArrayImpl fix first., _evaluate_real_compositional(), _max_particles(), _prepare_real_members(), Stage 5: application to real (observed) data.  Like :mod:`evaluate`, but the inp, Save a posterior pair plot per observation (real data has no ground truth)., Load the observed group and normalize it to flat member rows.      Expected layo (+3 more)

### Community 57 - "Community 57"
Cohesion: 0.50
Nodes (3): import_submodules(), Auto-import the modules of a package so ``@register_*`` decorators run.  The reg, Import every non-underscore module directly inside a package.      Call from a p

### Community 58 - "run_tuning"
Cohesion: 0.29
Nodes (10): _objective(), Stage 4: hyperparameter tuning with Optuna.  Runs a (by default multi-objective), Per-trial artifact directory, keyed by the study-global Optuna trial number., Save the fit-once preprocessing state, shared by every trial/model.      Written, _report(), run_tuning(), _save_shared_preprocessing(), _suggest() (+2 more)

### Community 59 - "build_workflow"
Cohesion: 0.33
Nodes (5): build_workflow(), Any, Assemble a single-level BayesFlow workflow from config.  Uses ``bf.BasicWorkflow, Build a ``bf.BasicWorkflow`` from the root ``cfg``., Build a ``bf.BasicWorkflow`` (or ``bf.CompositionalWorkflow``) from the root ``c

### Community 60 - "apply_bayesflow_patches"
Cohesion: 0.50
Nodes (4): apply_bayesflow_patches(), _patch_compositional_condition_reshape(), Targeted runtime fixes for known BayesFlow bugs (version-checked, applied once)., Idempotently install the fixes. Called when a compositional workflow is built.

## Knowledge Gaps
- **136 isolated node(s):** `hydrabflow`, `graphify`, `Usage`, `What graphify is for`, `Step 0 - GitHub repos and multi-path merge (only if a URL or several paths)` (+131 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **35 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `BaseSimulator` connect `Config Schemas` to `Preprocessing Pipeline & Steps`, `Simulate Stage & Registries`, `Logging Helper`?**
  _High betweenness centrality (0.123) - this node is a cross-community bridge._
- **Why does `get_simulator()` connect `Simulate Stage & Registries` to `Config Schemas`, `Community 40`, `Config Composition Tests`, `Community 11`, `Community 56`?**
  _High betweenness centrality (0.120) - this node is a cross-community bridge._
- **Why does `PreprocessStep` connect `Preprocessing Pipeline & Steps` to `Community 35`, `Example Simulators (Skeleton/TwoMoons)`?**
  _High betweenness centrality (0.101) - this node is a cross-community bridge._
- **Are the 10 inferred relationships involving `PreprocessStep` (e.g. with `PreprocessPipeline` and `Standardizer`) actually correct?**
  _`PreprocessStep` has 10 INFERRED edges - model-reasoned connections that need verification._
- **Are the 10 inferred relationships involving `_evaluate_compositional_global()` (e.g. with `load_approximator()` and `apply_augmentations_once()`) actually correct?**
  _`_evaluate_compositional_global()` has 10 INFERRED edges - model-reasoned connections that need verification._
- **Are the 15 inferred relationships involving `register_configs()` (e.g. with `AdapterConfig` and `AugmentationConfig`) actually correct?**
  _`register_configs()` has 15 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Marimo notebook: inspect a training run's posterior samples and diagnostics.  Ru`, `hydrabflow`, `HydraBFlow: a reusable BayesFlow + Hydra SBI pipeline template.  Importing the p` to the rest of the system?**
  _301 weakly-connected nodes found - possible documentation gaps or missing edges._
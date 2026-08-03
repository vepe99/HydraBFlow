# Graph Report - HydraBFlow  (2026-08-03)

## Corpus Check
- 62 files · ~29,542 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 592 nodes · 665 edges · 108 communities (60 shown, 48 thin omitted)
- Extraction: 86% EXTRACTED · 14% INFERRED · 0% AMBIGUOUS · INFERRED: 91 edges (avg confidence: 0.75)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `9e812c5e`
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
- [[_COMMUNITY_Base Simulator Interface|Base Simulator Interface]]
- [[_COMMUNITY_Config Composition Tests|Config Composition Tests]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Dataset IO|Dataset IO]]
- [[_COMMUNITY_Hydra App Boilerplate|Hydra App Boilerplate]]
- [[_COMMUNITY_JAX Backend Pin|JAX Backend Pin]]
- [[_COMMUNITY_Logging Helper|Logging Helper]]
- [[_COMMUNITY_Claude Settings Hooks|Claude Settings Hooks]]
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
- [[_COMMUNITY_permissions|permissions]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_AST Structural Extraction|AST Structural Extraction]]
- [[_COMMUNITY_EXTRACTEDINFERREDAMBIGUOUS Audit Trail|EXTRACTED/INFERRED/AMBIGUOUS Audit Trail]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 57|Community 57]]
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
- [[_COMMUNITY_Ordered list of preprocessing steps. Each entry is ``{name registry key, ...p|Ordered list of preprocessing steps. Each entry is ``{name: <registry key>, ...p]]
- [[_COMMUNITY_Maps raw dataset keys to BayesFlow roles.      ``inference_variables`` are conca|Maps raw dataset keys to BayesFlow roles.      ``inference_variables`` are conca]]
- [[_COMMUNITY_Register all schemas in Hydra's ConfigStore.      Must be called before ``@hydra|Register all schemas in Hydra's ConfigStore.      Must be called before ``@hydra]]
- [[_COMMUNITY_Hyperparameters for the posterior (inference) network.      ``type`` is resolved|Hyperparameters for the posterior (inference) network.      ``type`` is resolved]]
- [[_COMMUNITY_Return the summary network selected by ``cfg.type`` (a ``SummaryNetworkConfig``)|Return the summary network selected by ``cfg.type`` (a ``SummaryNetworkConfig``)]]
- [[_COMMUNITY_Return the inference (posterior) network selected by ``cfg.type`` (an ``Inferenc|Return the inference (posterior) network selected by ``cfg.type`` (an ``Inferenc]]
- [[_COMMUNITY_BaseSimulator|BaseSimulator]]
- [[_COMMUNITY_Same seed + same step list - identical end-to-end result through build_augmenta|Same seed + same step list -> identical end-to-end result through build_augmenta]]
- [[_COMMUNITY_A step's random stream is its own spawn child, so it doesn't depend on trailing|A step's random stream is its own spawn child, so it doesn't depend on trailing]]
- [[_COMMUNITY_Build a single augmentation through the public registry with a seeded generator.|Build a single augmentation through the public registry with a seeded generator.]]
- [[_COMMUNITY_At non-trivial strength, each augmentation changes the batch.|At non-trivial strength, each augmentation changes the batch.]]
- [[_COMMUNITY_Same seed - bit-identical augmented output.|Same seed -> bit-identical augmented output.]]
- [[_COMMUNITY_Different seed - different draws (the randomness is genuinely seed-controlled).|Different seed -> different draws (the randomness is genuinely seed-controlled).]]
- [[_COMMUNITY_Randomness comes only from the injected generator, not global np.random.|Randomness comes only from the injected generator, not global np.random.]]

## God Nodes (most connected - your core abstractions)
1. `PreprocessStep` - 16 edges
2. `build_workflow()` - 13 edges
3. `run_training()` - 12 edges
4. `TwoMoonsSimulator` - 12 edges
5. `_objective()` - 11 edges
6. `SplitStep` - 11 edges
7. `build_pipeline()` - 11 edges
8. `BaseSimulator` - 11 edges
9. `What You Must Do When Invoked` - 11 edges
10. `3. Config groups, one by one` - 11 edges

## Surprising Connections (you probably didn't know these)
- `test_build_adapter()` --calls--> `build_adapter()`  [INFERRED]
  tests/test_workflow.py → src/hydrabflow/pipeline/adapter.py
- `test_unknown_preprocess_step_errors()` --calls--> `build_pipeline()`  [INFERRED]
  tests/test_registries.py → src/hydrabflow/preprocessing/registry.py
- `test_augmentation_registry_builds()` --calls--> `build_augmentations()`  [INFERRED]
  tests/test_registries.py → src/hydrabflow/augmentation/registry.py
- `compose_cfg()` --calls--> `register_configs()`  [INFERRED]
  tests/conftest.py → src/hydrabflow/config/schema.py
- `test_embed_dim_is_per_head()` --calls--> `_embed_dim()`  [INFERRED]
  tests/test_networks_embed_dim.py → src/hydrabflow/networks/factory.py

## Import Cycles
- None detected.

## Communities (108 total, 48 thin omitted)

### Community 0 - "Preprocessing Pipeline & Steps"
Cohesion: 0.08
Nodes (27): ABC, PreprocessPipeline, PreprocessStep, Dataset, ndarray, Preprocessing step protocol and the pipeline that orchestrates them.  A :class:`, Element-wise (dataset-in, dataset-out) transform with optional fitted state., Estimate any state from ``data`` (train split). Stateless steps leave this empty (+19 more)

### Community 1 - "Eval / Checkpoint Stages"
Cohesion: 0.33
Nodes (6): Model Default Config, Diffusion Inference Network Config, Flow Matching Inference Network Config, DeepSet Summary Network Config, SetTransformer Summary Network Config, TimeSeriesTransformer Summary Network Config

### Community 2 - "Design Principles & Configs"
Cohesion: 0.06
Nodes (41): fix_keras_model(), load_approximator(), Any, Model save/load helpers, including the BayesFlow ``.keras`` deserialization work, Return a path to a load-safe copy of ``model_path`` (patching the ArrayImpl tag), Load a saved approximator, applying the ArrayImpl fix first., save_approximator(), Stage 3: evaluation on a simulated test set (with known ground truth).  Loads th (+33 more)

### Community 3 - "Augmentation Registry & Tests"
Cohesion: 0.07
Nodes (28): AdapterConfig, AugmentationConfig, DataConfig, EvalConfig, InferenceConfig, InferenceNetworkConfig, ModelConfig, PreprocessingConfig (+20 more)

### Community 4 - "Simulate Stage & Registries"
Cohesion: 0.22
Nodes (6): build_augmentations(), Augmentation, Name -> augmentation-factory registry and builder.  An augmentation factory rece, Build the ordered augmentation list from ``cfg.augmentation`` (an ``Augmentation, Build the ordered augmentation list from ``cfg.augmentation``., test_augmentation_registry_builds()

### Community 5 - "Example Simulators (Skeleton/TwoMoons)"
Cohesion: 0.10
Nodes (19): BaseException, RuntimeError, Dataset, ndarray, Per-feature z-score standardization step.  Generalizes the reference project's `, Standardizer, is_oom_error(), Retry a GPU computation with a progressively smaller batch size when it runs out (+11 more)

### Community 6 - "Config Schemas"
Cohesion: 0.19
Nodes (11): _agg_pyplot(), Artifacts a stage writes into a run directory: loss curve, posterior, diagnostic, Truth-free diagnostic: one posterior pair plot per observation (used for real da, Import matplotlib with the headless backend selected., Persist the raw Keras loss history as JSON, plus a loss curve., Load the best-val-loss weights BayesFlow checkpointed during training.      Make, Write the truth-aware diagnostics listed in ``cfg.eval.diagnostics`` into ``run_, restore_best_weights() (+3 more)

### Community 7 - "Network Factory & Adapter"
Cohesion: 0.07
Nodes (28): 0. Prerequisites & install, 1. The five stages at a glance, 2. Changing the simulator, 2a. Write the simulator class, 2b. Registration is automatic, 2c. Add the simulator config, 2d. The adapter wires itself, 2e. Shape contract cheat-sheet (+20 more)

### Community 9 - "Base Simulator Interface"
Cohesion: 0.08
Nodes (13): BaseSimulator, BaseSimulator, Any, ndarray, Base interface every forward model implements.  A simulator is the ONLY piece a, Abstract forward model. Subclass + register via ``@register_simulator``., Ordered names of the inferred parameters (become ``inference_variables``)., Keys of the observable arrays. One key = single observable; >1 enables fusion. (+5 more)

### Community 10 - "Config Composition Tests"
Cohesion: 0.09
Nodes (26): adapter_keys(), _as_list(), build_adapter(), fill_adapter_from_simulator(), _lists(), Any, Build the BayesFlow ``Adapter`` from ``AdapterConfig``.  The adapter is the stru, The four adapter key lists as plain Python lists (resolving interpolations). (+18 more)

### Community 11 - "Community 11"
Cohesion: 0.10
Nodes (24): Stage 1: dataset generation.  Samples the prior and runs the forward model in ch, Generate the dataset described by ``cfg`` and return its path., run_simulation(), get_simulator(), Name -> simulator-class registry.  New simulators self-register with the ``@regi, Class decorator registering a :class:`BaseSimulator` subclass under ``name``., Instantiate the simulator selected by ``cfg.simulator`` (a ``SimulatorConfig``)., register_simulator() (+16 more)

### Community 12 - "Dataset IO"
Cohesion: 0.29
Nodes (10): concatenate_chunks(), load_dataset(), n_rows(), Dataset, Dataset IO. Datasets are ``.npz`` archives where each key maps to an array whose, Number of simulations in a dataset dict (length of its leading axis)., Concatenate a list of dataset dicts along the leading (simulation) axis., Generate ``n_total`` rows in chunks of ``chunk`` and save them to ``out_path``. (+2 more)

### Community 13 - "Hydra App Boilerplate"
Cohesion: 0.40
Nodes (4): gaussian_noise(), Augmentation, Observational-noise augmentation — and the template for your own.  An augmentati, Add zero-mean Gaussian noise to one observable key.      Params: ``noise_key`` (

### Community 14 - "JAX Backend Pin"
Cohesion: 0.20
Nodes (9): Logger, limit_gpus(), Pin compute settings *before* keras/bayesflow/JAX are imported anywhere.  Two th, Pin ``CUDA_VISIBLE_DEVICES`` to the least-used GPU(s) before JAX/CUDA initialize, Set ``KERAS_BACKEND`` unless the user already chose one. Returns the active back, set_backend(), get_logger(), Minimal logging helper so all pipeline stages log consistently. (+1 more)

### Community 31 - "Community 31"
Cohesion: 0.08
Nodes (23): For /graphify add and --watch, For /graphify query, For the commit hook and native CLAUDE.md integration, For --update and --cluster-only, /graphify, Honesty Rules, Interpreter guard for subcommands, Part A - Structural extraction for code files (+15 more)

### Community 32 - "Community 32"
Cohesion: 0.15
Nodes (13): A.1 The data contract, A.2 Convert your existing arrays into the dataset file, A.3 Tell the pipeline about it (config only), A.4 Run train + evaluate, A.5 What reads what, B.1 The single seam, B.2 Option 1 — Quick swap (one format, replace the body), B.3 Option 2 — A format registry (support several formats by extension) (+5 more)

### Community 33 - "Community 33"
Cohesion: 0.17
Nodes (12): 0. What you're running, 1. Prerequisites, 2.1 Generate the training set, 2.2 Generate a held-out test set, 2.3 Train, 2.4 Evaluate, 2. The four commands (full run), 3. Fast smoke run (≈1 minute) (+4 more)

### Community 34 - "Community 34"
Cohesion: 0.17
Nodes (11): Core Design Principles, Decisions Log, Folder Structure (finalized), Goal, graphify, HydraBFlow: SBI Pipeline Template with BayesFlow, Output Directory Convention, Run stages (5 entry points) (+3 more)

### Community 35 - "Community 35"
Cohesion: 0.15
Nodes (14): 1. Prerequisites, 2. Run a study, 3. What gets saved, 4. Run many processes at once (parallel tuning), 5. Reading the results, 6. Changing what is tuned (the search space), 7. Key config reference (`tuning` group), 8. Command recap (+6 more)

### Community 37 - "Community 37"
Cohesion: 0.07
Nodes (30): 1. How the config system works, 2. The root master config — `config.yaml`, 3.10 `tuning/`, 3.1 `simulator/`, 3.2 `model/`, 3.3 `data/`, 3.4 `training/`, 3.5 `preprocessing/` (+22 more)

### Community 39 - "Community 39"
Cohesion: 0.19
Nodes (10): PreprocessPipeline, build_pipeline(), Name -> preprocessing-step registry and pipeline builder., Register a step factory (usually the step class itself) under ``name``., Build a :class:`PreprocessPipeline` from ``cfg.preprocessing`` (a ``Preprocessin, register_step(), Preprocessing pipeline: fit/transform/split + state save/load round-trip., test_pipeline_fit_transform_and_split() (+2 more)

### Community 46 - "Community 46"
Cohesion: 0.06
Nodes (37): build_inference_network(), build_summary_network(), _deep_set(), _diffusion(), _embed_dim(), _flow_matching(), Any, Build BayesFlow networks from structured dataclass configs (no ``_target_``).  B (+29 more)

### Community 57 - "Community 57"
Cohesion: 0.50
Nodes (3): import_submodules(), Auto-import the modules of a package so ``@register_*`` decorators run.  The reg, Import every non-underscore module directly inside a package.      Call from a p

## Knowledge Gaps
- **140 isolated node(s):** `hydrabflow`, `ModelConfig`, `DataConfig`, `TrainingConfig`, `TuningConfig` (+135 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **48 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `get_simulator()` connect `Community 11` to `Base Simulator Interface`, `Config Composition Tests`?**
  _High betweenness centrality (0.093) - this node is a cross-community bridge._
- **Why does `BaseSimulator` connect `Base Simulator Interface` to `Preprocessing Pipeline & Steps`, `Community 11`?**
  _High betweenness centrality (0.073) - this node is a cross-community bridge._
- **Why does `build_pipeline()` connect `Community 39` to `Preprocessing Pipeline & Steps`, `Design Principles & Configs`, `Community 46`?**
  _High betweenness centrality (0.069) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `PreprocessStep` (e.g. with `PreprocessPipeline` and `Standardizer`) actually correct?**
  _`PreprocessStep` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `build_workflow()` (e.g. with `run_real_evaluation()` and `run_evaluation()`) actually correct?**
  _`build_workflow()` has 9 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `run_training()` (e.g. with `build_augmentations()` and `select_adapter_keys()`) actually correct?**
  _`run_training()` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `_objective()` (e.g. with `save_approximator()` and `_run_diagnostics()`) actually correct?**
  _`_objective()` has 7 INFERRED edges - model-reasoned connections that need verification._
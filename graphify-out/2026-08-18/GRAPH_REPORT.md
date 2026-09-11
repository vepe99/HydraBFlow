# Graph Report - HydraBFlow  (2026-08-03)

## Corpus Check
- 54 files · ~28,431 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 538 nodes · 602 edges · 88 communities (45 shown, 43 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 75 edges (avg confidence: 0.76)
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
- [[_COMMUNITY_PackageInit cluster 23|Package/Init cluster 23]]
- [[_COMMUNITY_PackageInit cluster 24|Package/Init cluster 24]]
- [[_COMMUNITY_Evaluate Entry Script|Evaluate Entry Script]]
- [[_COMMUNITY_Evaluate-Real Entry Script|Evaluate-Real Entry Script]]
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
1. `PreprocessStep` - 14 edges
2. `run_training()` - 12 edges
3. `build_workflow()` - 12 edges
4. `What You Must Do When Invoked` - 11 edges
5. `3. Config groups, one by one` - 11 edges
6. `_objective()` - 10 edges
7. `build_pipeline()` - 10 edges
8. `run_with_oom_backoff()` - 10 edges
9. `/graphify` - 10 edges
10. `HydraBFlow: SBI Pipeline Template with BayesFlow` - 10 edges

## Surprising Connections (you probably didn't know these)
- `test_build_adapter()` --calls--> `build_adapter()`  [INFERRED]
  tests/test_workflow.py → src/hydrabflow/pipeline/adapter.py
- `test_unknown_preprocess_step_errors()` --calls--> `build_pipeline()`  [INFERRED]
  tests/test_registries.py → src/hydrabflow/preprocessing/registry.py
- `test_augmentation_registry_builds()` --calls--> `build_augmentations()`  [INFERRED]
  tests/test_registries.py → src/hydrabflow/augmentation/registry.py
- `compose_cfg()` --calls--> `register_configs()`  [INFERRED]
  tests/conftest.py → src/hydrabflow/config.py
- `test_custom_network_builder_registers()` --calls--> `build_summary_network()`  [INFERRED]
  tests/test_registries.py → src/hydrabflow/networks/factory.py

## Import Cycles
- None detected.

## Communities (88 total, 43 thin omitted)

### Community 0 - "Preprocessing Pipeline & Steps"
Cohesion: 0.16
Nodes (11): CastDtype, DropNaNSimulations, _num_rows(), Dataset, Built-in stateless preprocessing steps (besides standardization).  Add your own, Drop rows (simulations) that contain any NaN/Inf in the listed keys., Random hold-out split. Steps listed after this one are fit on the train split on, Cast the listed keys (or all keys) to a target dtype, e.g. float32 for training. (+3 more)

### Community 1 - "Eval / Checkpoint Stages"
Cohesion: 0.33
Nodes (6): Model Default Config, Diffusion Inference Network Config, Flow Matching Inference Network Config, DeepSet Summary Network Config, SetTransformer Summary Network Config, TimeSeriesTransformer Summary Network Config

### Community 2 - "Design Principles & Configs"
Cohesion: 0.05
Nodes (45): build_augmentations(), Augmentation, Name -> augmentation factory, plus the builder.  A factory is ``(params, rng, co, Build the ordered augmentation list from ``cfg.augmentation``., Build the ordered augmentation list from ``cfg.augmentation`` (an ``Augmentation, fix_keras_model(), load_approximator(), Any (+37 more)

### Community 3 - "Augmentation Registry & Tests"
Cohesion: 0.07
Nodes (27): AdapterConfig, AugmentationConfig, DataConfig, EvalConfig, InferenceNetworkConfig, ModelConfig, PreprocessingConfig, Typed config schema. ``conf/config.yaml`` fills these in; the factories read the (+19 more)

### Community 4 - "Simulate Stage & Registries"
Cohesion: 0.12
Nodes (13): fill_adapter_from_simulator(), Fill empty adapter variable lists from the simulator's own declaration (in place, get_simulator(), Name -> simulator class. New simulators self-register with ``@register_simulator, Instantiate the simulator selected by ``cfg.simulator`` (a ``SimulatorConfig``)., test_two_moons_shapes_and_reproducibility(), Registry resolution: unknown names fail loudly, custom builders plug in., Every extension point is a Registry filled by import side effects (utils/registr (+5 more)

### Community 5 - "Example Simulators (Skeleton/TwoMoons)"
Cohesion: 0.17
Nodes (15): BaseException, RuntimeError, is_oom_error(), T, Retry a GPU computation with a smaller batch size when it runs out of memory.  B, True if ``exc`` looks like a GPU out-of-memory error (matches on the message)., Call ``fn(batch_size)``, halving the batch size on OOM until ``min_batch``., run_with_oom_backoff() (+7 more)

### Community 6 - "Config Schemas"
Cohesion: 0.18
Nodes (9): Artifacts a stage writes into a run directory: loss curve, posterior, diagnostic, Truth-free diagnostic: one posterior pair plot per observation (used for real da, Persist the raw Keras loss history as JSON, plus a loss curve., Load the best-val-loss weights BayesFlow checkpointed during training.      Make, Write the truth-aware diagnostics listed in ``cfg.eval.diagnostics`` into ``run_, restore_best_weights(), run_diagnostics(), save_history() (+1 more)

### Community 7 - "Network Factory & Adapter"
Cohesion: 0.07
Nodes (28): 0. Prerequisites & install, 1. The five stages at a glance, 2. Changing the simulator, 2a. Write the simulator class, 2b. Registration is automatic, 2c. Add the simulator config, 2d. The adapter wires itself, 2e. Shape contract cheat-sheet (+20 more)

### Community 9 - "Base Simulator Interface"
Cohesion: 0.13
Nodes (12): ABC, BaseSimulator, BaseSimulator, Any, ndarray, Base interface every forward model implements.  A simulator is the ONLY piece a, Abstract forward model. Subclass + register via ``@register_simulator``., Draw ``n`` prior samples. Returns ``{param_name: (n, 1)}``. (+4 more)

### Community 10 - "Config Composition Tests"
Cohesion: 0.17
Nodes (12): cfg(), compose(), compose_cfg(), Shared test fixtures., Compose the root config with the structured schemas registered.      ``fill=True, Expose the composer so tests can build configs with custom overrides., Config composition + schema validation smoke tests., The typed schema is the only validation layer now that group YAMLs have no base (+4 more)

### Community 11 - "Community 11"
Cohesion: 0.25
Nodes (14): _batch(), _build_one(), Two Moons simulator + the augmentation reproducibility/stochasticity contract., Build the shipped augmentation through the registry with a seeded generator., The shipped config trains without augmentation: zero scale must leave the batch, Consecutive calls on the *same* built augmentation differ (re-drawn every batch), Same seed + same step list -> identical result through the public builder., test_actually_perturbs() (+6 more)

### Community 12 - "Dataset IO"
Cohesion: 0.27
Nodes (10): concatenate_chunks(), load_dataset(), n_rows(), Dataset, Dataset IO. Datasets are ``.npz`` archives where each key maps to an array whose, Number of simulations in a dataset dict (length of its leading axis)., Concatenate a list of dataset dicts along the leading (simulation) axis., Generate ``n_total`` rows in chunks of ``chunk`` and save them to ``out_path``. (+2 more)

### Community 13 - "Hydra App Boilerplate"
Cohesion: 0.40
Nodes (4): gaussian_noise(), Augmentation, Observational-noise augmentation — and the template for your own.  An augmentati, Add zero-mean Gaussian noise to one observable key.      Params: ``noise_key`` (

### Community 14 - "JAX Backend Pin"
Cohesion: 0.33
Nodes (5): limit_gpus(), Pin compute settings *before* keras/bayesflow/JAX are imported anywhere.  Both l, Pin ``CUDA_VISIBLE_DEVICES`` to the least-used GPU(s) before JAX/CUDA initialize, Set ``KERAS_BACKEND`` unless the user already chose one. Returns the active back, set_backend()

### Community 23 - "Package/Init cluster 23"
Cohesion: 0.21
Nodes (12): adapter_keys(), _as_list(), build_adapter(), _lists(), Any, Build the BayesFlow ``Adapter`` from ``AdapterConfig``.  The adapter is the stru, The four adapter key lists as plain Python lists (resolving interpolations)., Every dataset key the adapter consumes, in a stable order.      ``drop`` is incl (+4 more)

### Community 25 - "Evaluate Entry Script"
Cohesion: 0.22
Nodes (4): Dataset, ndarray, Per-feature z-score standardization step.  Mean/std are fit on the train split o, Standardizer

### Community 26 - "Evaluate-Real Entry Script"
Cohesion: 0.22
Nodes (6): discover(), T, One name -> component registry, shared by every extension point.  Simulators, pr, A named collection filled by ``@registry.add("name")`` decorators., Import every non-underscore module in a package, so its decorators run., Registry

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
Nodes (11): Core Design Principles, Decisions Log, Folder Structure (finalized), Goal, graphify, HydraBFlow: SBI Pipeline Template with BayesFlow, Output Directory Convention, Run stages (4 entry points: `hydrabflow-<stage>`, or `python -m hydrabflow.pipeline.<stage>`) (+3 more)

### Community 35 - "Community 35"
Cohesion: 0.15
Nodes (14): 1. Prerequisites, 2. Run a study, 3. What gets saved, 4. Run many processes at once (parallel tuning), 5. Reading the results, 6. Changing what is tuned (the search space), 7. Key config reference (`tuning` group), 8. Command recap (+6 more)

### Community 37 - "Community 37"
Cohesion: 0.07
Nodes (30): 1. How the config system works, 2. The root master config — `config.yaml`, 3.10 `tuning/`, 3.1 `simulator/`, 3.2 `model/`, 3.3 `data/`, 3.4 `training/`, 3.5 `preprocessing/` (+22 more)

### Community 39 - "Community 39"
Cohesion: 0.09
Nodes (20): PreprocessPipeline, PreprocessPipeline, PreprocessStep, Dataset, ndarray, Preprocessing step protocol and the pipeline that orchestrates them.  A :class:`, Dataset-in, dataset-out transform with optional fitted state., Estimate any state from ``data`` (train split). Stateless steps leave this empty (+12 more)

### Community 46 - "Community 46"
Cohesion: 0.09
Nodes (29): build_inference_network(), build_summary_network(), _deep_set(), _diffusion(), _embed_dim(), _flow_matching(), Any, Build BayesFlow networks from the typed config, resolved by ``cfg.type``.  A cus (+21 more)

## Knowledge Gaps
- **139 isolated node(s):** `hydrabflow`, `ModelConfig`, `DataConfig`, `TrainingConfig`, `TuningConfig` (+134 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **43 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `get_simulator()` connect `Simulate Stage & Registries` to `Base Simulator Interface`, `Design Principles & Configs`?**
  _High betweenness centrality (0.084) - this node is a cross-community bridge._
- **Why does `PreprocessStep` connect `Community 39` to `Preprocessing Pipeline & Steps`, `Base Simulator Interface`, `Evaluate Entry Script`?**
  _High betweenness centrality (0.069) - this node is a cross-community bridge._
- **Why does `compose_cfg()` connect `Config Composition Tests` to `Augmentation Registry & Tests`, `Simulate Stage & Registries`?**
  _High betweenness centrality (0.064) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `PreprocessStep` (e.g. with `PreprocessPipeline` and `Standardizer`) actually correct?**
  _`PreprocessStep` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `run_training()` (e.g. with `build_augmentations()` and `select_adapter_keys()`) actually correct?**
  _`run_training()` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `build_workflow()` (e.g. with `run_evaluation()` and `run_training()`) actually correct?**
  _`build_workflow()` has 8 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Marimo notebook: inspect a training run's posterior samples and diagnostics.  Ru`, `hydrabflow`, `HydraBFlow: a reusable BayesFlow + Hydra SBI pipeline template.  Importing the p` to the rest of the system?**
  _265 weakly-connected nodes found - possible documentation gaps or missing edges._
# Graph Report - HydraFlow  (2026-06-07)

## Corpus Check
- 55 files · ~22,678 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 489 nodes · 626 edges · 37 communities (29 shown, 8 thin omitted)
- Extraction: 86% EXTRACTED · 14% INFERRED · 0% AMBIGUOUS · INFERRED: 87 edges (avg confidence: 0.72)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `79bc195c`
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
- [[_COMMUNITY_Example Augmentations|Example Augmentations]]
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
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]

## God Nodes (most connected - your core abstractions)
1. `PreprocessStep` - 19 edges
2. `Dataset` - 14 edges
3. `build_workflow()` - 11 edges
4. `TwoMoonsSimulator` - 11 edges
5. `What You Must Do When Invoked` - 11 edges
6. `HydraFlow` - 11 edges
7. `Root Hydra Config (config.yaml)` - 11 edges
8. `run_training()` - 10 edges
9. `SplitStep` - 10 edges
10. `/graphify` - 10 edges

## Surprising Connections (you probably didn't know these)
- `Dataset Format Registry by Extension` --semantically_similar_to--> `Self-Registration by Name Pattern`  [INFERRED] [semantically similar]
  docs/bring_your_own_data.md → CLAUDE.md
- `test_augmentation_registry_builds()` --calls--> `build_augmentations()`  [INFERRED]
  tests/test_registries.py → src/hydraflow/augmentation/registry.py
- `test_build_adapter()` --calls--> `build_adapter()`  [INFERRED]
  tests/test_workflow.py → src/hydraflow/pipeline/adapter.py
- `test_build_workflow()` --calls--> `build_workflow()`  [INFERRED]
  tests/test_workflow.py → src/hydraflow/pipeline/workflow.py
- `test_two_moons_shapes_and_reproducibility()` --calls--> `get_simulator()`  [INFERRED]
  tests/test_augmentation.py → src/hydraflow/simulators/registry.py

## Import Cycles
- None detected.

## Communities (37 total, 8 thin omitted)

### Community 0 - "Preprocessing Pipeline & Steps"
Cohesion: 0.06
Nodes (32): ABC, PreprocessPipeline, PreprocessStep, Preprocessing step protocol and the pipeline that orchestrates them.  A :class:`, Element-wise (dataset-in, dataset-out) transform with optional fitted state., Estimate any state from ``data`` (train split). Stateless steps leave this empty, Return a transformed copy/view of ``data``., Arrays to persist so the fitted transform can be reloaded. Default: nothing. (+24 more)

### Community 1 - "Eval / Checkpoint Stages"
Cohesion: 0.07
Nodes (45): Adapter Default Config, Adapter Two Moons Config, Augmentation Default Config, Dataset Data Contract (shared leading axis), Bring Your Own Data Guide, Dataset Format Registry by Extension, Single IO Seam (load_dataset/save_dataset), Full Traceability Principle (+37 more)

### Community 2 - "Design Principles & Configs"
Cohesion: 0.06
Nodes (43): fix_keras_model(), load_approximator(), Model save/load helpers, including the BayesFlow ``.keras`` deserialization work, Return a path to a load-safe copy of ``model_path`` (patching the ArrayImpl tag), Load a saved approximator, applying the ArrayImpl fix first., save_approximator(), Stage 3: evaluation on a simulated test set (with known ground truth).  Loads th, Stage 5: application to real (observed) data.  Like :mod:`evaluate`, but the inp (+35 more)

### Community 3 - "Augmentation Registry & Tests"
Cohesion: 0.08
Nodes (28): feature_dropout(), gaussian_noise(), multiplicative_noise(), Example augmentations. Use as templates for problem-specific ones.  Augmentation, Add zero-mean Gaussian noise to one observable key (additive observational noise, Scale an observable by ``(1 + N(0, mult_scale))`` — multiplicative / gain jitter, Randomly zero out entries of an observable with probability ``dropout_prob`` (Be, AdapterConfig (+20 more)

### Community 4 - "Simulate Stage & Registries"
Cohesion: 0.06
Nodes (41): available_augmentations(), build_augmentations(), Name -> augmentation-factory registry and builder.  An augmentation factory rece, Build the ordered augmentation list from ``cfg.augmentation`` (an ``Augmentation, available_steps(), available_simulators(), get_simulator(), Name -> simulator-class registry.  New simulators self-register with the ``@regi (+33 more)

### Community 5 - "Example Simulators (Skeleton/TwoMoons)"
Cohesion: 0.11
Nodes (19): build_inference_network(), build_summary_network(), Build BayesFlow networks from structured dataclass configs (no ``_target_``).  T, Return a single BayesFlow summary network for ``cfg`` (a ``SummaryNetworkConfig`, Return a BayesFlow inference (posterior) network for ``cfg`` (an ``InferenceNetw, _as_list(), build_adapter(), Build the BayesFlow ``Adapter`` from ``AdapterConfig``.  The adapter is the stru (+11 more)

### Community 6 - "Config Schemas"
Cohesion: 0.11
Nodes (8): BaseSimulator, Skeleton Simulator Config, Skeleton simulator: the intentional stub shipped with the template.  It declares, SkeletonSimulator, Two Moons: the classic bimodal SBI benchmark, as a worked example simulator.  Th, TwoMoonsSimulator, ndarray, ndarray

### Community 7 - "Network Factory & Adapter"
Cohesion: 0.07
Nodes (28): 0. Prerequisites & install, 1. The five stages at a glance, 2. Changing the simulator, 2a. Write the simulator class, 2b. Make it self-register, 2c. Add the simulator config, 2d. Wire the adapter to your parameter / observable names, 2e. Shape contract cheat-sheet (+20 more)

### Community 8 - "Graphify Tooling"
Cohesion: 0.12
Nodes (16): graphify Skill Trigger, AST Structural Extraction, EXTRACTED/INFERRED/AMBIGUOUS Audit Trail, Community Detection, Detect Files Step, Existing-Graph Fast Path, Gemini Extraction Backend, God Nodes (+8 more)

### Community 9 - "Base Simulator Interface"
Cohesion: 0.16
Nodes (9): BaseSimulator, Base interface every forward model implements.  A simulator is the ONLY piece a, Abstract forward model. Subclass + register via ``@register_simulator``., Ordered names of the inferred parameters (become ``inference_variables``)., Keys of the observable arrays. One key = single observable; >1 enables fusion., Draw ``n`` prior samples. Returns ``{param_name: (n, 1)}``., Run the forward model on a batch of parameters. Returns ``{observable_key: (n, ., Any (+1 more)

### Community 10 - "Config Composition Tests"
Cohesion: 0.15
Nodes (10): Register all schemas in Hydra's ConfigStore.      Must be called before ``@hydra, register_configs(), cfg(), compose(), compose_cfg(), Shared test fixtures., Compose the root config with the structured schemas registered.      Provides th, Expose the composer so tests can build configs with custom overrides. (+2 more)

### Community 11 - "Example Augmentations"
Cohesion: 0.24
Nodes (9): build_pipeline(), Name -> preprocessing-step registry and pipeline builder., Register a step factory (usually the step class itself) under ``name``., Build a :class:`PreprocessPipeline` from ``cfg.preprocessing`` (a ``Preprocessin, register_step(), Preprocessing pipeline: fit/transform/split + state save/load round-trip., test_pipeline_fit_transform_and_split(), test_state_roundtrip() (+1 more)

### Community 12 - "Dataset IO"
Cohesion: 0.38
Nodes (6): concatenate_chunks(), load_dataset(), Dataset IO. Datasets are ``.npz`` archives where each key maps to an array whose, Concatenate a list of dataset dicts along the leading (simulation) axis., save_dataset(), Dataset

### Community 13 - "Hydra App Boilerplate"
Cohesion: 0.33
Nodes (5): conf_path(), make_cli(), Shared Hydra-app boilerplate for the five run stages., Absolute path to the repo-root ``conf/`` directory., Wrap a ``run_fn(cfg)`` into a Hydra console entry point.      Registers the stru

### Community 14 - "JAX Backend Pin"
Cohesion: 0.33
Nodes (5): limit_gpus(), Pin compute settings *before* keras/bayesflow/JAX are imported anywhere.  Two th, Pin ``CUDA_VISIBLE_DEVICES`` to the least-used GPU(s) before JAX/CUDA initialize, Set ``KERAS_BACKEND`` unless the user already chose one. Returns the active back, set_backend()

### Community 15 - "Logging Helper"
Cohesion: 0.40
Nodes (4): Logger, get_logger(), Minimal logging helper so all pipeline stages log consistently., Return a configured logger. Hydra also installs its own handlers; this is a safe

### Community 31 - "Community 31"
Cohesion: 0.09
Nodes (23): For /graphify add and --watch, For /graphify query, For the commit hook and native CLAUDE.md integration, For --update and --cluster-only, /graphify, Honesty Rules, Interpreter guard for subcommands, Part A - Structural extraction for code files (+15 more)

### Community 32 - "Community 32"
Cohesion: 0.14
Nodes (13): A.1 The data contract, A.2 Convert your existing arrays into the dataset file, A.3 Tell the pipeline about it (config only), A.4 Run train + evaluate, A.5 What reads what, B.1 The single seam, B.2 Option 1 — Quick swap (one format, replace the body), B.3 Option 2 — A format registry (support several formats by extension) (+5 more)

### Community 33 - "Community 33"
Cohesion: 0.15
Nodes (12): 0. What you're running, 1. Prerequisites, 2.1 Generate the training set, 2.2 Generate a held-out test set, 2.3 Train, 2.4 Evaluate, 2. The four commands (full run), 3. Fast smoke run (≈1 minute) (+4 more)

### Community 34 - "Community 34"
Cohesion: 0.17
Nodes (11): Core Design Principles, Decisions Log, Folder Structure (finalized), Goal, graphify, HydraFlow: SBI Pipeline Template with BayesFlow, Output Directory Convention, Run stages (5 entry points) (+3 more)

### Community 35 - "Community 35"
Cohesion: 0.18
Nodes (10): 1. Prerequisites, 2. Run a study, 3. What gets saved, 4. Run many processes at once (parallel tuning), 5. Reading the results, 6. Changing what is tuned (the search space), 7. Key config reference (`tuning` group), 8. Command recap (+2 more)

## Knowledge Gaps
- **113 isolated node(s):** `PreToolUse`, `ModelConfig`, `DataConfig`, `TrainingConfig`, `InferenceConfig` (+108 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **8 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `build_pipeline()` connect `Example Augmentations` to `Preprocessing Pipeline & Steps`, `Design Principles & Configs`?**
  _High betweenness centrality (0.130) - this node is a cross-community bridge._
- **Why does `run_training()` connect `Design Principles & Configs` to `Example Augmentations`, `Simulate Stage & Registries`, `Example Simulators (Skeleton/TwoMoons)`?**
  _High betweenness centrality (0.123) - this node is a cross-community bridge._
- **Why does `build_augmentations()` connect `Simulate Stage & Registries` to `Design Principles & Configs`?**
  _High betweenness centrality (0.114) - this node is a cross-community bridge._
- **Are the 9 inferred relationships involving `PreprocessStep` (e.g. with `Standardizer` and `CastDtype`) actually correct?**
  _`PreprocessStep` has 9 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `Dataset` (e.g. with `Standardizer` and `CastDtype`) actually correct?**
  _`Dataset` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `build_workflow()` (e.g. with `run_real_evaluation()` and `run_evaluation()`) actually correct?**
  _`build_workflow()` has 8 INFERRED edges - model-reasoned connections that need verification._
- **What connects `PreToolUse`, `Marimo notebook: inspect a training run's posterior samples and diagnostics.  Ru`, `HydraFlow: a reusable BayesFlow + Hydra SBI pipeline template.  Importing the pa` to the rest of the system?**
  _216 weakly-connected nodes found - possible documentation gaps or missing edges._
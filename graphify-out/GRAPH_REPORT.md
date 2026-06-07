# Graph Report - .  (2026-06-07)

## Corpus Check
- 72 files · ~98,812 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 378 nodes · 498 edges · 31 communities (24 shown, 7 thin omitted)
- Extraction: 83% EXTRACTED · 17% INFERRED · 0% AMBIGUOUS · INFERRED: 85 edges (avg confidence: 0.72)
- Token cost: 0 input · 0 output

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

## God Nodes (most connected - your core abstractions)
1. `PreprocessStep` - 18 edges
2. `Dataset` - 14 edges
3. `graphify Skill` - 13 edges
4. `build_workflow()` - 11 edges
5. `TwoMoonsSimulator` - 11 edges
6. `Root Hydra Config (config.yaml)` - 11 edges
7. `run_training()` - 10 edges
8. `SplitStep` - 10 edges
9. `End-to-End Pipeline Guide` - 10 edges
10. `build_pipeline()` - 9 edges

## Surprising Connections (you probably didn't know these)
- `test_build_workflow()` --calls--> `build_workflow()`  [INFERRED]
  tests/test_workflow.py → src/hydraflow/pipeline/workflow.py
- `test_augmentation_registry_builds()` --calls--> `build_augmentations()`  [INFERRED]
  tests/test_registries.py → src/hydraflow/augmentation/registry.py
- `test_build_adapter()` --calls--> `build_adapter()`  [INFERRED]
  tests/test_workflow.py → src/hydraflow/pipeline/adapter.py
- `test_two_moons_shapes_and_reproducibility()` --calls--> `get_simulator()`  [INFERRED]
  tests/test_augmentation.py → src/hydraflow/simulators/registry.py
- `test_skeleton_simulator_raises()` --calls--> `get_simulator()`  [INFERRED]
  tests/test_registries.py → src/hydraflow/simulators/registry.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **graphify Build Pipeline Stages** — graphify_skill_detect_step, graphify_skill_ast_extraction, graphify_skill_semantic_extraction, graphify_skill_community_detection, graphify_skill_graph_report [EXTRACTED 0.85]
- **Root Config Group Composition** — config_config_yaml, simulator_two_moons, model_default, adapter_two_moons, preprocessing_default [EXTRACTED 0.85]
- **Default Model = Summary + Inference Network** — model_default, summary_network_set_transformer, inference_network_flow_matching [EXTRACTED 0.95]
- **Five Run Stages Pipeline Flow** — end_to_end_guide_five_stages, simulator_two_moons, training_default, eval_default, tuning_default [EXTRACTED 0.85]

## Communities (31 total, 7 thin omitted)

### Community 0 - "Preprocessing Pipeline & Steps"
Cohesion: 0.06
Nodes (32): ABC, PreprocessPipeline, PreprocessStep, Preprocessing step protocol and the pipeline that orchestrates them.  A :class:`, Element-wise (dataset-in, dataset-out) transform with optional fitted state., Estimate any state from ``data`` (train split). Stateless steps leave this empty, Return a transformed copy/view of ``data``., Arrays to persist so the fitted transform can be reloaded. Default: nothing. (+24 more)

### Community 1 - "Eval / Checkpoint Stages"
Cohesion: 0.06
Nodes (46): fix_keras_model(), load_approximator(), Model save/load helpers, including the BayesFlow ``.keras`` deserialization work, Return a path to a load-safe copy of ``model_path`` (patching the ArrayImpl tag), Load a saved approximator, applying the ArrayImpl fix first., save_approximator(), Stage 3: evaluation on a simulated test set (with known ground truth).  Loads th, Stage 5: application to real (observed) data.  Like :mod:`evaluate`, but the inp (+38 more)

### Community 2 - "Design Principles & Configs"
Cohesion: 0.08
Nodes (44): Adapter Default Config, Adapter Two Moons Config, Augmentation Default Config, Dataset Data Contract (shared leading axis), Bring Your Own Data Guide, Dataset Format Registry by Extension, Single IO Seam (load_dataset/save_dataset), Full Traceability Principle (+36 more)

### Community 3 - "Augmentation Registry & Tests"
Cohesion: 0.10
Nodes (26): available_augmentations(), build_augmentations(), Name -> augmentation-factory registry and builder.  An augmentation factory rece, Build the ordered augmentation list from ``cfg.augmentation`` (an ``Augmentation, Augmentation, _batch(), _build_one(), _compose_aug() (+18 more)

### Community 4 - "Simulate Stage & Registries"
Cohesion: 0.09
Nodes (21): Stage 1: dataset generation.  Samples the prior and runs the forward model in ch, Generate the dataset described by ``cfg`` and return its path., run_simulation(), available_steps(), Name -> preprocessing-step registry and pipeline builder., Register a step factory (usually the step class itself) under ``name``., register_step(), available_simulators() (+13 more)

### Community 5 - "Example Simulators (Skeleton/TwoMoons)"
Cohesion: 0.10
Nodes (7): BaseSimulator, Skeleton simulator: the intentional stub shipped with the template.  It declares, SkeletonSimulator, Two Moons: the classic bimodal SBI benchmark, as a worked example simulator.  Th, TwoMoonsSimulator, ndarray, ndarray

### Community 6 - "Config Schemas"
Cohesion: 0.10
Nodes (20): AdapterConfig, AugmentationConfig, DataConfig, EvalConfig, InferenceConfig, InferenceNetworkConfig, ModelConfig, PreprocessingConfig (+12 more)

### Community 7 - "Network Factory & Adapter"
Cohesion: 0.13
Nodes (15): build_inference_network(), build_summary_network(), Build BayesFlow networks from structured dataclass configs (no ``_target_``).  T, Return a single BayesFlow summary network for ``cfg`` (a ``SummaryNetworkConfig`, Return a BayesFlow inference (posterior) network for ``cfg`` (an ``InferenceNetw, _as_list(), build_adapter(), Build the BayesFlow ``Adapter`` from ``AdapterConfig``.  The adapter is the stru (+7 more)

### Community 8 - "Graphify Tooling"
Cohesion: 0.12
Nodes (17): graphify Skill Trigger, graphify Skill, AST Structural Extraction, EXTRACTED/INFERRED/AMBIGUOUS Audit Trail, Community Detection, Detect Files Step, Existing-Graph Fast Path, Gemini Extraction Backend (+9 more)

### Community 9 - "Base Simulator Interface"
Cohesion: 0.16
Nodes (9): BaseSimulator, Base interface every forward model implements.  A simulator is the ONLY piece a, Abstract forward model. Subclass + register via ``@register_simulator``., Ordered names of the inferred parameters (become ``inference_variables``)., Keys of the observable arrays. One key = single observable; >1 enables fusion., Draw ``n`` prior samples. Returns ``{param_name: (n, 1)}``., Run the forward model on a batch of parameters. Returns ``{observable_key: (n, ., Any (+1 more)

### Community 10 - "Config Composition Tests"
Cohesion: 0.15
Nodes (10): Register all schemas in Hydra's ConfigStore.      Must be called before ``@hydra, register_configs(), cfg(), compose(), compose_cfg(), Shared test fixtures., Compose the root config with the structured schemas registered.      Provides th, Expose the composer so tests can build configs with custom overrides. (+2 more)

### Community 11 - "Example Augmentations"
Cohesion: 0.28
Nodes (8): feature_dropout(), gaussian_noise(), multiplicative_noise(), Example augmentations. Use as templates for problem-specific ones.  Augmentation, Add zero-mean Gaussian noise to one observable key (additive observational noise, Scale an observable by ``(1 + N(0, mult_scale))`` — multiplicative / gain jitter, Randomly zero out entries of an observable with probability ``dropout_prob`` (Be, Augmentation

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

## Knowledge Gaps
- **32 isolated node(s):** `PreToolUse`, `ModelConfig`, `DataConfig`, `TrainingConfig`, `InferenceConfig` (+27 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `build_pipeline()` connect `Eval / Checkpoint Stages` to `Preprocessing Pipeline & Steps`, `Simulate Stage & Registries`?**
  _High betweenness centrality (0.214) - this node is a cross-community bridge._
- **Why does `run_training()` connect `Eval / Checkpoint Stages` to `Augmentation Registry & Tests`?**
  _High betweenness centrality (0.201) - this node is a cross-community bridge._
- **Why does `build_augmentations()` connect `Augmentation Registry & Tests` to `Eval / Checkpoint Stages`?**
  _High betweenness centrality (0.185) - this node is a cross-community bridge._
- **Are the 9 inferred relationships involving `PreprocessStep` (e.g. with `Standardizer` and `CastDtype`) actually correct?**
  _`PreprocessStep` has 9 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `Dataset` (e.g. with `Standardizer` and `CastDtype`) actually correct?**
  _`Dataset` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `build_workflow()` (e.g. with `run_real_evaluation()` and `run_evaluation()`) actually correct?**
  _`build_workflow()` has 8 INFERRED edges - model-reasoned connections that need verification._
- **What connects `PreToolUse`, `Marimo notebook: inspect a training run's posterior samples and diagnostics.  Ru`, `HydraFlow: a reusable BayesFlow + Hydra SBI pipeline template.  Importing the pa` to the rest of the system?**
  _134 weakly-connected nodes found - possible documentation gaps or missing edges._
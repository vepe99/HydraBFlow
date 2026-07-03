# Graph Report - HydraBFlow  (2026-07-03)

## Corpus Check
- 58 files · ~28,072 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 599 nodes · 808 edges · 58 communities (45 shown, 13 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 88 edges (avg confidence: 0.72)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `952f55c8`
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
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 57|Community 57]]

## God Nodes (most connected - your core abstractions)
1. `PreprocessStep` - 20 edges
2. `Dataset` - 14 edges
3. `build_workflow()` - 12 edges
4. `TwoMoonsSimulator` - 12 edges
5. `run_training()` - 11 edges
6. `SplitStep` - 11 edges
7. `compose()` - 11 edges
8. `/graphify` - 11 edges
9. `What You Must Do When Invoked` - 11 edges
10. `3. Config groups, one by one` - 11 edges

## Surprising Connections (you probably didn't know these)
- `compose_cfg()` --calls--> `register_configs()`  [INFERRED]
  tests/conftest.py → src/hydrabflow/config/schema.py
- `test_build_networks()` --calls--> `build_summary_network()`  [INFERRED]
  tests/test_workflow.py → src/hydrabflow/networks/factory.py
- `test_build_networks()` --calls--> `build_inference_network()`  [INFERRED]
  tests/test_workflow.py → src/hydrabflow/networks/factory.py
- `compose_cfg()` --calls--> `fill_adapter_from_simulator()`  [INFERRED]
  tests/conftest.py → src/hydrabflow/pipeline/adapter.py
- `test_build_workflow()` --calls--> `build_workflow()`  [INFERRED]
  tests/test_workflow.py → src/hydrabflow/pipeline/workflow.py

## Import Cycles
- None detected.

## Communities (58 total, 13 thin omitted)

### Community 0 - "Preprocessing Pipeline & Steps"
Cohesion: 0.07
Nodes (31): PreprocessPipeline, PreprocessStep, Preprocessing step protocol and the pipeline that orchestrates them.  A :class:`, Element-wise (dataset-in, dataset-out) transform with optional fitted state., Estimate any state from ``data`` (train split). Stateless steps leave this empty, Return a transformed copy/view of ``data``., Arrays to persist so the fitted transform can be reloaded. Default: nothing., Restore arrays produced by :meth:`state`. (+23 more)

### Community 1 - "Eval / Checkpoint Stages"
Cohesion: 0.33
Nodes (6): Diffusion Inference Network Config, Flow Matching Inference Network Config, Model Default Config, DeepSet Summary Network Config, SetTransformer Summary Network Config, TimeSeriesTransformer Summary Network Config

### Community 2 - "Design Principles & Configs"
Cohesion: 0.06
Nodes (42): fix_keras_model(), load_approximator(), Model save/load helpers, including the BayesFlow ``.keras`` deserialization work, Return a path to a load-safe copy of ``model_path`` (patching the ArrayImpl tag), Load a saved approximator, applying the ArrayImpl fix first., save_approximator(), Stage 3: evaluation on a simulated test set (with known ground truth).  Loads th, Stage 5: application to real (observed) data.  Like :mod:`evaluate`, but the inp (+34 more)

### Community 3 - "Augmentation Registry & Tests"
Cohesion: 0.12
Nodes (24): AdapterConfig, DataConfig, EvalConfig, InferenceConfig, InferenceNetworkConfig, ModelConfig, PreprocessingConfig, Structured (dataclass) config schemas for every Hydra config group.  The whole p (+16 more)

### Community 4 - "Simulate Stage & Registries"
Cohesion: 0.12
Nodes (29): available_augmentations(), build_augmentations(), Name -> augmentation-factory registry and builder.  An augmentation factory rece, Build the ordered augmentation list from ``cfg.augmentation`` (an ``Augmentation, register_augmentation(), Augmentation, _batch(), _build_one() (+21 more)

### Community 5 - "Example Simulators (Skeleton/TwoMoons)"
Cohesion: 0.20
Nodes (12): _as_list(), build_adapter(), fill_adapter_from_simulator(), Build the BayesFlow ``Adapter`` from ``AdapterConfig``.  The adapter is the stru, Fill empty adapter variable lists from the simulator's own declaration (in place, Construct ``bf.adapters.Adapter`` from ``cfg`` (an ``AdapterConfig``)., Any, Adapter / network / workflow construction. Skipped if bayesflow isn't installed. (+4 more)

### Community 6 - "Config Schemas"
Cohesion: 0.10
Nodes (7): BaseSimulator, Skeleton simulator: the intentional stub shipped with the template.  It declares, SkeletonSimulator, Two Moons: the classic bimodal SBI benchmark, as a worked example simulator.  Th, TwoMoonsSimulator, ndarray, ndarray

### Community 7 - "Network Factory & Adapter"
Cohesion: 0.05
Nodes (35): 0. Prerequisites & install, 1. The five stages at a glance, 2. Changing the simulator, 2a. Write the simulator class, 2b. Make it self-register, 2b. Registration is automatic, 2c. Add the simulator config, 2d. The adapter wires itself (+27 more)

### Community 8 - "Graphify Tooling"
Cohesion: 0.12
Nodes (16): graphify Skill Trigger, AST Structural Extraction, EXTRACTED/INFERRED/AMBIGUOUS Audit Trail, Community Detection, Detect Files Step, Existing-Graph Fast Path, Gemini Extraction Backend, God Nodes (+8 more)

### Community 9 - "Base Simulator Interface"
Cohesion: 0.16
Nodes (10): ABC, BaseSimulator, Base interface every forward model implements.  A simulator is the ONLY piece a, Abstract forward model. Subclass + register via ``@register_simulator``., Ordered names of the inferred parameters (become ``inference_variables``)., Keys of the observable arrays. One key = single observable; >1 enables fusion., Draw ``n`` prior samples. Returns ``{param_name: (n, 1)}``., Run the forward model on a batch of parameters. Returns ``{observable_key: (n, . (+2 more)

### Community 10 - "Config Composition Tests"
Cohesion: 0.18
Nodes (14): cfg(), compose(), compose_cfg(), Shared test fixtures., Compose the root config with the structured schemas registered.      ``fill=True, Expose the composer so tests can build configs with custom overrides., Expose the composer so tests can build configs with custom overrides., Config composition + schema validation smoke tests. (+6 more)

### Community 11 - "Community 11"
Cohesion: 0.14
Nodes (15): Stage 1: dataset generation.  Samples the prior and runs the forward model in ch, Generate the dataset described by ``cfg`` and return its path., run_simulation(), available_simulators(), get_simulator(), Name -> simulator-class registry.  New simulators self-register with the ``@regi, Class decorator registering a :class:`BaseSimulator` subclass under ``name``., Instantiate the simulator selected by ``cfg.simulator`` (a ``SimulatorConfig``). (+7 more)

### Community 12 - "Dataset IO"
Cohesion: 0.43
Nodes (6): concatenate_chunks(), load_dataset(), Dataset IO. Datasets are ``.npz`` archives where each key maps to an array whose, Concatenate a list of dataset dicts along the leading (simulation) axis., save_dataset(), Dataset

### Community 13 - "Hydra App Boilerplate"
Cohesion: 0.38
Nodes (5): conf_path(), make_cli(), Shared Hydra-app boilerplate for the five run stages., Absolute path to the repo-root ``conf/`` directory., Wrap a ``run_fn(cfg)`` into a Hydra console entry point.      Registers the stru

### Community 14 - "JAX Backend Pin"
Cohesion: 0.38
Nodes (5): limit_gpus(), Pin compute settings *before* keras/bayesflow/JAX are imported anywhere.  Two th, Pin ``CUDA_VISIBLE_DEVICES`` to the least-used GPU(s) before JAX/CUDA initialize, Set ``KERAS_BACKEND`` unless the user already chose one. Returns the active back, set_backend()

### Community 15 - "Logging Helper"
Cohesion: 0.40
Nodes (4): Logger, get_logger(), Minimal logging helper so all pipeline stages log consistently., Return a configured logger. Hydra also installs its own handlers; this is a safe

### Community 31 - "Community 31"
Cohesion: 0.08
Nodes (23): For /graphify add and --watch, For /graphify query, For the commit hook and native CLAUDE.md integration, For --update and --cluster-only, /graphify, Honesty Rules, Interpreter guard for subcommands, Part A - Structural extraction for code files (+15 more)

### Community 32 - "Community 32"
Cohesion: 0.13
Nodes (13): A.1 The data contract, A.2 Convert your existing arrays into the dataset file, A.3 Tell the pipeline about it (config only), A.4 Run train + evaluate, A.5 What reads what, B.1 The single seam, B.2 Option 1 — Quick swap (one format, replace the body), B.3 Option 2 — A format registry (support several formats by extension) (+5 more)

### Community 33 - "Community 33"
Cohesion: 0.14
Nodes (12): 0. What you're running, 1. Prerequisites, 2.1 Generate the training set, 2.2 Generate a held-out test set, 2.3 Train, 2.4 Evaluate, 2. The four commands (full run), 3. Fast smoke run (≈1 minute) (+4 more)

### Community 34 - "Community 34"
Cohesion: 0.17
Nodes (11): Core Design Principles, Decisions Log, Folder Structure (finalized), Goal, graphify, HydraBFlow: SBI Pipeline Template with BayesFlow, Output Directory Convention, Run stages (5 entry points) (+3 more)

### Community 35 - "Community 35"
Cohesion: 0.17
Nodes (10): 1. Prerequisites, 2. Run a study, 3. What gets saved, 4. Run many processes at once (parallel tuning), 5. Reading the results, 6. Changing what is tuned (the search space), 7. Key config reference (`tuning` group), 8. Command recap (+2 more)

### Community 37 - "Community 37"
Cohesion: 0.06
Nodes (30): 1. How the config system works, 2. The root master config — `config.yaml`, 3.10 `tuning/`, 3.1 `simulator/`, 3.2 `model/`, 3.3 `data/`, 3.4 `training/`, 3.5 `preprocessing/` (+22 more)

### Community 38 - "Community 38"
Cohesion: 0.17
Nodes (11): Core Design Principles, Decisions Log, Folder Structure (finalized), Goal, graphify, HydraBFlow: SBI Pipeline Template with BayesFlow, Output Directory Convention, Run stages (5 entry points) (+3 more)

### Community 39 - "Community 39"
Cohesion: 0.22
Nodes (11): available_steps(), build_pipeline(), Name -> preprocessing-step registry and pipeline builder., Register a step factory (usually the step class itself) under ``name``., Build a :class:`PreprocessPipeline` from ``cfg.preprocessing`` (a ``Preprocessin, register_step(), Preprocessing pipeline: fit/transform/split + state save/load round-trip., test_pipeline_fit_transform_and_split() (+3 more)

### Community 40 - "Community 40"
Cohesion: 0.40
Nodes (4): Adding your own simulator, Design at a glance, HydraBFlow, Quickstart

### Community 41 - "Community 41"
Cohesion: 0.67
Nodes (3): Adapter Default Config, Preprocessing Default Config, Training Default Config

### Community 46 - "Community 46"
Cohesion: 0.14
Nodes (19): build_inference_network(), build_summary_network(), _deep_set(), _diffusion(), _flow_matching(), Build BayesFlow networks from structured dataclass configs (no ``_target_``).  B, Return a single BayesFlow summary network for ``cfg`` (a ``SummaryNetworkConfig`, Decorator registering a summary-network builder under ``name`` (the config ``typ (+11 more)

### Community 47 - "Community 47"
Cohesion: 0.40
Nodes (4): Adding your own simulator, Design at a glance, HydraBFlow, Quickstart

### Community 56 - "Community 56"
Cohesion: 0.24
Nodes (11): feature_dropout(), gaussian_noise(), multiplicative_noise(), Example augmentations. Use as templates for problem-specific ones.  Augmentation, Add zero-mean Gaussian noise to one observable key (additive observational noise, Scale an observable by ``(1 + N(0, mult_scale))`` — multiplicative / gain jitter, Randomly zero out entries of an observable with probability ``dropout_prob`` (Be, AugmentationConfig (+3 more)

### Community 57 - "Community 57"
Cohesion: 0.50
Nodes (3): import_submodules(), Auto-import the modules of a package so ``@register_*`` decorators run.  The reg, Import every non-underscore module directly inside a package.      Call from a p

## Knowledge Gaps
- **151 isolated node(s):** `PreToolUse`, `allow`, `Any`, `Any`, `Any` (+146 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **13 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `build_pipeline()` connect `Community 39` to `Preprocessing Pipeline & Steps`, `Design Principles & Configs`?**
  _High betweenness centrality (0.114) - this node is a cross-community bridge._
- **Why does `PreprocessPipeline` connect `Preprocessing Pipeline & Steps` to `Community 39`?**
  _High betweenness centrality (0.098) - this node is a cross-community bridge._
- **Why does `PreprocessStep` connect `Preprocessing Pipeline & Steps` to `Base Simulator Interface`?**
  _High betweenness centrality (0.093) - this node is a cross-community bridge._
- **Are the 9 inferred relationships involving `PreprocessStep` (e.g. with `Standardizer` and `CastDtype`) actually correct?**
  _`PreprocessStep` has 9 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `Dataset` (e.g. with `Standardizer` and `CastDtype`) actually correct?**
  _`Dataset` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `build_workflow()` (e.g. with `run_real_evaluation()` and `run_evaluation()`) actually correct?**
  _`build_workflow()` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `run_training()` (e.g. with `build_augmentations()` and `save_approximator()`) actually correct?**
  _`run_training()` has 6 INFERRED edges - model-reasoned connections that need verification._
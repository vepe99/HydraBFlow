# Graph Report - HydraBFlow  (2026-08-07)

## Corpus Check
- 51 files · ~20,963 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 481 nodes · 570 edges · 81 communities (39 shown, 42 thin omitted)
- Extraction: 86% EXTRACTED · 14% INFERRED · 0% AMBIGUOUS · INFERRED: 78 edges (avg confidence: 0.77)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `f245ae80`
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
- [[_COMMUNITY_StationarySIR|StationarySIR]]
- [[_COMMUNITY_PackageInit cluster 24|Package/Init cluster 24]]
- [[_COMMUNITY_Evaluate Entry Script|Evaluate Entry Script]]
- [[_COMMUNITY_test_simulate_unbatched.py|test_simulate_unbatched.py]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
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
- [[_COMMUNITY_Same seed + same step list - identical end-to-end result through build_augmenta|Same seed + same step list -> identical end-to-end result through build_augmenta]]
- [[_COMMUNITY_A step's random stream is its own spawn child, so it doesn't depend on trailing|A step's random stream is its own spawn child, so it doesn't depend on trailing]]
- [[_COMMUNITY_Build a single augmentation through the public registry with a seeded generator.|Build a single augmentation through the public registry with a seeded generator.]]
- [[_COMMUNITY_At non-trivial strength, each augmentation changes the batch.|At non-trivial strength, each augmentation changes the batch.]]
- [[_COMMUNITY_Same seed - bit-identical augmented output.|Same seed -> bit-identical augmented output.]]
- [[_COMMUNITY_Different seed - different draws (the randomness is genuinely seed-controlled).|Different seed -> different draws (the randomness is genuinely seed-controlled).]]
- [[_COMMUNITY_Randomness comes only from the injected generator, not global np.random.|Randomness comes only from the injected generator, not global np.random.]]

## God Nodes (most connected - your core abstractions)
1. `PreprocessStep` - 13 edges
2. `build_workflow()` - 12 edges
3. `compose()` - 12 edges
4. `run_training()` - 11 edges
5. `BaseSimulator` - 11 edges
6. `What You Must Do When Invoked` - 11 edges
7. `build_pipeline()` - 10 edges
8. `build_summary_network()` - 10 edges
9. `run_with_oom_backoff()` - 10 edges
10. `/graphify` - 10 edges

## Surprising Connections (you probably didn't know these)
- `test_unknown_simulator_errors()` --calls--> `get_simulator()`  [INFERRED]
  tests/test_registries.py → src/hydrabflow/registry.py
- `compose_cfg()` --calls--> `register_configs()`  [INFERRED]
  tests/conftest.py → src/hydrabflow/config.py
- `test_embed_dim_is_per_head()` --calls--> `_embed_dim()`  [INFERRED]
  tests/test_networks_embed_dim.py → src/hydrabflow/networks/factory.py
- `test_build_adapter()` --calls--> `build_adapter()`  [INFERRED]
  tests/test_workflow.py → src/hydrabflow/pipeline/adapter.py
- `test_build_workflow()` --calls--> `build_workflow()`  [INFERRED]
  tests/test_workflow.py → src/hydrabflow/pipeline/workflow.py

## Import Cycles
- None detected.

## Communities (81 total, 42 thin omitted)

### Community 0 - "Preprocessing Pipeline & Steps"
Cohesion: 0.16
Nodes (11): CastDtype, DropNaNSimulations, _num_rows(), Dataset, Built-in stateless preprocessing steps (besides standardization).  Add your own, Drop rows (simulations) that contain any NaN/Inf in the listed keys., Random hold-out split. Steps listed after this one are fit on the train split on, Cast the listed keys (or all keys) to a target dtype, e.g. float32 for training. (+3 more)

### Community 1 - "Eval / Checkpoint Stages"
Cohesion: 0.33
Nodes (6): Model Default Config, Diffusion Inference Network Config, Flow Matching Inference Network Config, DeepSet Summary Network Config, SetTransformer Summary Network Config, TimeSeriesTransformer Summary Network Config

### Community 2 - "Design Principles & Configs"
Cohesion: 0.27
Nodes (6): discover(), T, A named collection filled by ``@registry.add("name")`` decorators.      ``packag, Import ``self.package``'s modules so their decorators have run. Idempotent., Import every non-underscore module in a package, so its decorators run., Registry

### Community 3 - "Augmentation Registry & Tests"
Cohesion: 0.07
Nodes (27): AdapterConfig, AugmentationConfig, DataConfig, EvalConfig, InferenceNetworkConfig, ModelConfig, PreprocessingConfig, Typed config schema. ``conf/config.yaml`` fills these in; the factories read the (+19 more)

### Community 4 - "Simulate Stage & Registries"
Cohesion: 0.08
Nodes (31): Stage 3: posterior inference on held-out data, simulated or real.  Loads the app, _require_model_dir(), _run_diagnostics(), run_evaluation(), build_workflow(), Any, Assemble the ``bf.BasicWorkflow`` (adapter + summary network + inference network, Build a ``bf.BasicWorkflow`` from the root ``cfg``.      ``run_dir`` (passed by (+23 more)

### Community 5 - "Example Simulators (Skeleton/TwoMoons)"
Cohesion: 0.17
Nodes (15): BaseException, RuntimeError, is_oom_error(), T, Retry a GPU computation at a smaller batch size when it runs out of memory.  JAX, True if ``exc`` looks like a GPU out-of-memory error (matches on the message)., Call ``fn(batch_size)``, halving the batch size on OOM until ``min_batch``., run_with_oom_backoff() (+7 more)

### Community 6 - "Config Schemas"
Cohesion: 0.13
Nodes (15): fix_keras_model(), load_approximator(), Any, What a stage writes into its run directory: model, loss curve, posterior, diagno, Write the truth-aware diagnostics listed in ``cfg.eval.diagnostics`` into ``run_, Truth-free diagnostic: one posterior pair plot per observation (used for real da, Return a path to a load-safe copy of ``model_path`` (patching the ArrayImpl tag), Load a saved approximator, applying the ArrayImpl fix first. (+7 more)

### Community 7 - "Network Factory & Adapter"
Cohesion: 0.08
Nodes (31): Stage 1: dataset generation.  Samples the prior and runs the forward model in ch, Generate the dataset described by ``cfg`` and return its path., run_simulation(), _n(), Stage 2: training.  Load dataset -> preprocessing (fit on train, save the state, Train the approximator and return (workflow, history)., run_training(), _save_loss_plot() (+23 more)

### Community 9 - "Base Simulator Interface"
Cohesion: 0.09
Nodes (16): ABC, BaseSimulator, BaseSimulator, Any, ndarray, Base interface every forward model implements.  A simulator is the only piece a, Abstract forward model. Subclass + register via ``@register_simulator``., Draw ``n`` prior samples. Returns ``{param_name: (n, 1)}``. (+8 more)

### Community 10 - "Config Composition Tests"
Cohesion: 0.06
Nodes (38): adapter_keys(), _as_list(), build_adapter(), fill_adapter_from_simulator(), _lists(), Any, Build the BayesFlow ``Adapter``: dataset keys -> the roles BayesFlow expects.  `, The four adapter key lists as plain lists (resolving interpolations). (+30 more)

### Community 11 - "Community 11"
Cohesion: 0.25
Nodes (14): _batch(), _build_one(), Two Moons simulator + the augmentation reproducibility/stochasticity contract., Build the shipped augmentation through the registry with a seeded generator., The shipped config trains without augmentation: zero scale must leave the batch, Consecutive calls on the *same* built augmentation differ (re-drawn every batch), Same seed + same step list -> identical result through the public builder., test_actually_perturbs() (+6 more)

### Community 12 - "Dataset IO"
Cohesion: 0.27
Nodes (10): concatenate_chunks(), load_dataset(), n_rows(), Dataset, Dataset IO. Datasets are ``.npz`` archives where each key maps to an array whose, Number of simulations in a dataset dict (length of its leading axis)., Concatenate a list of dataset dicts along the leading (simulation) axis., Generate ``n_total`` rows in chunks of ``chunk`` and save them to ``out_path``. (+2 more)

### Community 13 - "Hydra App Boilerplate"
Cohesion: 0.33
Nodes (4): gaussian_noise(), Augmentation, Observational-noise augmentations — and the template for your own.  An augmentat, Add zero-mean Gaussian noise to one observable key.      Params: ``noise_key`` (

### Community 14 - "JAX Backend Pin"
Cohesion: 0.33
Nodes (5): limit_gpus(), Pin GPU selection and the Keras backend *before* keras/bayesflow/JAX import anyw, Pin ``CUDA_VISIBLE_DEVICES`` to the least-used GPU(s) before JAX/CUDA initialize, Set ``KERAS_BACKEND`` unless the user already chose one. Returns the active back, set_backend()

### Community 23 - "StationarySIR"
Cohesion: 0.24
Nodes (6): convert_params(), Any, ndarray, Helper function to convert mean/dispersion parameterization of a negative binomi, One draw: ``theta`` values are shape ``(1,)``, returns ``cases`` of shape ``(T,, StationarySIR

### Community 25 - "Evaluate Entry Script"
Cohesion: 0.22
Nodes (4): Dataset, ndarray, Per-feature z-score standardization step.  Mean/std are fit on the train split o, Standardizer

### Community 26 - "test_simulate_unbatched.py"
Cohesion: 0.24
Nodes (5): _Batched, _PerDraw, Per-draw simulators (``is_batched = False``) and the shape/name guard in ``BaseS, A toy forward model. Deterministic, so the batched and per-draw paths must agree, test_per_draw_matches_batched()

### Community 31 - "Community 31"
Cohesion: 0.08
Nodes (23): For /graphify add and --watch, For /graphify query, For the commit hook and native CLAUDE.md integration, For --update and --cluster-only, /graphify, Honesty Rules, Interpreter guard for subcommands, Part A - Structural extraction for code files (+15 more)

### Community 34 - "Community 34"
Cohesion: 0.20
Nodes (9): Design principles, docs/, graphify, HydraBFlow: SBI pipeline template (BayesFlow + Hydra), Layout, Notes worth keeping, Stack, Stages (+1 more)

### Community 35 - "Community 35"
Cohesion: 0.07
Nodes (25): Blocks, Configuration, Network groups, No per-block `output_dir`, Notes that actually bite, `run_name` vs `model_dir`, Augmentation (stochastic, per batch), Extending (+17 more)

### Community 39 - "Community 39"
Cohesion: 0.13
Nodes (12): PreprocessPipeline, PreprocessStep, Dataset, ndarray, Preprocessing step protocol and the pipeline that runs them.  A step transforms, Dataset-in, dataset-out transform with optional fitted state., Estimate any state from ``data`` (train split). Stateless steps leave this empty, Return a transformed copy/view of ``data``. (+4 more)

### Community 46 - "Community 46"
Cohesion: 0.19
Nodes (13): _deep_set(), _diffusion(), _embed_dim(), _flow_matching(), Any, The shipped network builders. A builder maps a network config to a BayesFlow net, Attention width, expressed per head so ``embed_dim % num_heads == 0`` always hol, _set_transformer() (+5 more)

## Knowledge Gaps
- **79 isolated node(s):** `hydrabflow`, `ModelConfig`, `DataConfig`, `TrainingConfig`, `TuningConfig` (+74 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **42 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `get_simulator()` connect `Config Composition Tests` to `Simulate Stage & Registries`, `Network Factory & Adapter`?**
  _High betweenness centrality (0.118) - this node is a cross-community bridge._
- **Why does `compose_cfg()` connect `Config Composition Tests` to `Augmentation Registry & Tests`?**
  _High betweenness centrality (0.076) - this node is a cross-community bridge._
- **Why does `register_configs()` connect `Augmentation Registry & Tests` to `Config Composition Tests`?**
  _High betweenness centrality (0.069) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `PreprocessStep` (e.g. with `Standardizer` and `CastDtype`) actually correct?**
  _`PreprocessStep` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `build_workflow()` (e.g. with `run_evaluation()` and `run_training()`) actually correct?**
  _`build_workflow()` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `compose()` (e.g. with `test_adapter_derived_from_simulator()` and `test_adapter_explicit_config_wins()`) actually correct?**
  _`compose()` has 9 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `run_training()` (e.g. with `select_adapter_keys()` and `build_workflow()`) actually correct?**
  _`run_training()` has 7 INFERRED edges - model-reasoned connections that need verification._
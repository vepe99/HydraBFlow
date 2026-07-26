# Graph Report - HydraBFlow  (2026-07-26)

## Corpus Check
- 78 files · ~54,089 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 832 nodes · 1098 edges · 104 communities (70 shown, 34 thin omitted)
- Extraction: 85% EXTRACTED · 15% INFERRED · 0% AMBIGUOUS · INFERRED: 166 edges (avg confidence: 0.75)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `d67f50dd`
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
- [[_COMMUNITY_reference_posterior|reference_posterior]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_test_lv_jax.py|test_lv_jax.py]]
- [[_COMMUNITY_reporting.py|reporting.py]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 57|Community 57]]
- [[_COMMUNITY_Simulator-gradient guidance in diffusion sampling|Simulator-gradient guidance in diffusion sampling]]
- [[_COMMUNITY_get_run_dir|get_run_dir]]
- [[_COMMUNITY_run_training|run_training]]
- [[_COMMUNITY_test_registries.py|test_registries.py]]
- [[_COMMUNITY_run_evaluation|run_evaluation]]
- [[_COMMUNITY_seed_everything|seed_everything]]
- [[_COMMUNITY_LogTransform|LogTransform]]
- [[_COMMUNITY_get_simulator|get_simulator]]
- [[_COMMUNITY_load_approximator|load_approximator]]
- [[_COMMUNITY_quiet.py|quiet.py]]
- [[_COMMUNITY_compose_cfg|compose_cfg]]
- [[_COMMUNITY_PreToolUse|PreToolUse]]
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
1. `PreprocessStep` - 19 edges
2. `compose()` - 19 edges
3. `run_guided_evaluation()` - 18 edges
4. `register_configs()` - 16 edges
5. `build_workflow()` - 16 edges
6. `LVConfig` - 16 edges
7. `BaseSimulator` - 15 edges
8. `guided_class()` - 15 edges
9. `run_diagnose_guidance()` - 14 edges
10. `run_training()` - 14 edges

## Surprising Connections (you probably didn't know these)
- `test_augmentation_registry_builds()` --calls--> `build_augmentations()`  [INFERRED]
  tests/test_registries.py → src/hydrabflow/augmentation/registry.py
- `compose_cfg()` --calls--> `register_configs()`  [INFERRED]
  tests/conftest.py → src/hydrabflow/config/schema.py
- `guided_class()` --calls--> `guided_diffusion_class()`  [INFERRED]
  tests/test_guidance.py → src/hydrabflow/networks/guided_diffusion.py
- `compose_cfg()` --calls--> `fill_adapter_from_simulator()`  [INFERRED]
  tests/conftest.py → src/hydrabflow/pipeline/adapter.py
- `test_build_adapter()` --calls--> `build_adapter()`  [INFERRED]
  tests/test_workflow.py → src/hydrabflow/pipeline/adapter.py

## Import Cycles
- None detected.

## Communities (104 total, 34 thin omitted)

### Community 0 - "Preprocessing Pipeline & Steps"
Cohesion: 0.07
Nodes (30): PreprocessPipeline, PreprocessPipeline, PreprocessStep, Dataset, ndarray, Preprocessing step protocol and the pipeline that orchestrates them.  A :class:`, Element-wise (dataset-in, dataset-out) transform with optional fitted state., Estimate any state from ``data`` (train split). Stateless steps leave this empty (+22 more)

### Community 1 - "Eval / Checkpoint Stages"
Cohesion: 0.33
Nodes (6): Model Default Config, Diffusion Inference Network Config, Flow Matching Inference Network Config, DeepSet Summary Network Config, SetTransformer Summary Network Config, TimeSeriesTransformer Summary Network Config

### Community 2 - "Design Principles & Configs"
Cohesion: 0.29
Nodes (10): _objective(), Stage 4: hyperparameter tuning with Optuna.  Runs a (by default multi-objective), Per-trial artifact directory, keyed by the study-global Optuna trial number., Save the fit-once preprocessing state, shared by every trial/model.      Written, _report(), run_tuning(), _save_shared_preprocessing(), _suggest() (+2 more)

### Community 3 - "Augmentation Registry & Tests"
Cohesion: 0.10
Nodes (30): feature_dropout(), gaussian_noise(), multiplicative_noise(), Augmentation, Example augmentations. Use as templates for problem-specific ones.  Augmentation, Add zero-mean Gaussian noise to one observable key (additive observational noise, Scale an observable by ``(1 + N(0, mult_scale))`` — multiplicative / gain jitter, Randomly zero out entries of an observable with probability ``dropout_prob`` (Be (+22 more)

### Community 4 - "Simulate Stage & Registries"
Cohesion: 0.10
Nodes (26): available_augmentations(), build_augmentations(), Augmentation, Name -> augmentation-factory registry and builder.  An augmentation factory rece, Build the ordered augmentation list from ``cfg.augmentation`` (an ``Augmentation, _batch(), _build_one(), _compose_aug() (+18 more)

### Community 5 - "Example Simulators (Skeleton/TwoMoons)"
Cohesion: 0.10
Nodes (19): BaseException, RuntimeError, Dataset, ndarray, Per-feature z-score standardization step.  Generalizes the reference project's `, Standardizer, is_oom_error(), Retry a GPU computation with a progressively smaller batch size when it runs out (+11 more)

### Community 6 - "Config Schemas"
Cohesion: 0.06
Nodes (19): ABC, BaseSimulator, BaseSimulator, Any, ndarray, Base interface every forward model implements.  A simulator is the ONLY piece a, Abstract forward model. Subclass + register via ``@register_simulator``., Ordered names of the inferred parameters (become ``inference_variables``). (+11 more)

### Community 7 - "Network Factory & Adapter"
Cohesion: 0.07
Nodes (28): 0. Prerequisites & install, 1. The five stages at a glance, 2. Changing the simulator, 2a. Write the simulator class, 2b. Registration is automatic, 2c. Add the simulator config, 2d. The adapter wires itself, 2e. Shape contract cheat-sheet (+20 more)

### Community 10 - "Config Composition Tests"
Cohesion: 0.16
Nodes (11): compose(), Expose the composer so tests can build configs with custom overrides., Config composition + schema validation smoke tests., test_adapter_derived_from_simulator(), test_adapter_explicit_config_wins(), test_group_override(), `adapter.drop` must survive the derive-from-simulator default.      Arm A is con, test_lv_arm_configs_compose() (+3 more)

### Community 11 - "Community 11"
Cohesion: 0.29
Nodes (6): available_simulators(), Name -> simulator-class registry.  New simulators self-register with the ``@regi, Class decorator registering a :class:`BaseSimulator` subclass under ``name``., register_simulator(), test_two_moons_registered(), test_simulator_registry_has_shipped_simulators()

### Community 12 - "Dataset IO"
Cohesion: 0.15
Nodes (19): concatenate_chunks(), load_chunk(), load_dataset(), _n_rows(), Dataset, Dataset IO. Datasets are ``.npz`` archives where each key maps to an array whose, Concatenate a list of dataset dicts along the leading (simulation) axis., Write a chunk ``.npz`` atomically (temp file + rename), so a crash mid-write can (+11 more)

### Community 13 - "Hydra App Boilerplate"
Cohesion: 0.33
Nodes (5): conf_path(), make_cli(), Shared Hydra-app boilerplate for the five run stages., Absolute path to the repo-root ``conf/`` directory., Wrap a ``run_fn(cfg)`` into a Hydra console entry point.      Registers the stru

### Community 14 - "JAX Backend Pin"
Cohesion: 0.07
Nodes (31): Logger, adapter_keys(), _as_list(), build_adapter(), fill_adapter_from_simulator(), Any, Build the BayesFlow ``Adapter`` from ``AdapterConfig``.  The adapter is the stru, Fill empty adapter variable lists from the simulator's own declaration (in place (+23 more)

### Community 15 - "Logging Helper"
Cohesion: 0.07
Nodes (47): _cos(), _norm(), _particle_diagnostics(), _plot_trace(), Any, ndarray, Diagnostic stage: how does the guidance term compare with the diffusion model's, Integrate ``dz = (f - 0.5 g^2 score) dt`` from ``t=1`` to ``t=0`` with explicit (+39 more)

### Community 31 - "Community 31"
Cohesion: 0.08
Nodes (23): For /graphify add and --watch, For /graphify query, For the commit hook and native CLAUDE.md integration, For --update and --cluster-only, /graphify, Honesty Rules, Interpreter guard for subcommands, Part A - Structural extraction for code files (+15 more)

### Community 32 - "Community 32"
Cohesion: 0.05
Nodes (39): A.1 The data contract, A.2 Convert your existing arrays into the dataset file, A.3 Tell the pipeline about it (config only), A.4 Run train + evaluate, A.5 What reads what, B.1 The single seam, B.2 Option 1 — Quick swap (one format, replace the body), B.3 Option 2 — A format registry (support several formats by extension) (+31 more)

### Community 33 - "Community 33"
Cohesion: 0.11
Nodes (29): Array, NamedTuple, add_observation_noise(), key_from_rng(), log_likelihood(), log_posterior(), log_prior(), lv_states() (+21 more)

### Community 34 - "Community 34"
Cohesion: 0.17
Nodes (11): Core Design Principles, Decisions Log, Folder Structure (finalized), Goal, graphify, HydraBFlow: SBI Pipeline Template with BayesFlow, Output Directory Convention, Run stages (5 entry points) (+3 more)

### Community 35 - "Community 35"
Cohesion: 0.13
Nodes (24): GuidanceTarget, Runtime (non-serialized) guidance target: what to compute the likelihood gradien, guided_class(), Tests for the LV simulator contract, the guided diffusion network, and the confi, No target attached => the hook must return the score untouched (a plain Diffusio, A likelihood that produces huge / non-finite gradients must not poison the score, The gradient must be taken w.r.t. the diffusion state, through the untransform., K=1 must reproduce the plain Tweedie gradient EXACTLY, not a one-sample cloud. (+16 more)

### Community 37 - "Community 37"
Cohesion: 0.07
Nodes (30): 1. How the config system works, 2. The root master config — `config.yaml`, 3.10 `tuning/`, 3.1 `simulator/`, 3.2 `model/`, 3.3 `data/`, 3.4 `training/`, 3.5 `preprocessing/` (+22 more)

### Community 38 - "reference_posterior"
Cohesion: 0.23
Nodes (16): _ess(), laplace_preconditioner(), _log_reference(), mala_sample(), Any, ndarray, _r_hat(), Exact reference posterior by MALA, for judging guided samples against ground tru (+8 more)

### Community 39 - "Community 39"
Cohesion: 0.43
Nodes (6): build_pipeline(), Build a :class:`PreprocessPipeline` from ``cfg.preprocessing`` (a ``Preprocessin, Preprocessing pipeline: fit/transform/split + state save/load round-trip., test_pipeline_fit_transform_and_split(), test_state_roundtrip(), _toy_data()

### Community 40 - "Community 40"
Cohesion: 0.14
Nodes (7): LotkaVolterraSimulator, ndarray, Lotka-Volterra simulator: the differentiable testbed for simulator-gradient guid, The validated forward-model settings (also used by the guidance and reference st, ``(theta_batch) -> (batch,)`` log-likelihood of a single observation ``x_obs``., Smooth tanh squash of the log-rates into ``prior_mean +- n_std * prior_std``., ``(theta_batch) -> (batch,)`` log-prior — used by the MALA reference sampler.

### Community 46 - "Community 46"
Cohesion: 0.10
Nodes (27): build_inference_network(), build_summary_network(), _deep_set(), _diffusion(), _embed_dim(), _flow_matching(), Any, Build BayesFlow networks from structured dataclass configs (no ``_target_``).  B (+19 more)

### Community 47 - "test_lv_jax.py"
Cohesion: 0.13
Nodes (7): Tests for the differentiable Lotka-Volterra core (no BayesFlow / Keras needed)., The hand-rolled RK4 must agree with a high-accuracy reference solver.      The t, With x_obs == states(theta), the residual is identically 0, so the gradient is e, The state clamp must keep out-of-prior evaluations finite (the 'bad gradients' p, test_gradient_vanishes_at_noise_free_truth(), test_rk4_matches_scipy(), test_wild_parameters_give_finite_gradients()

### Community 54 - "reporting.py"
Cohesion: 0.20
Nodes (13): _finite(), _has_nonfinite(), inspect_history(), _load_json(), _metrics_table(), Any, _rate(), Training-convergence inspection and Markdown report generation.  Two pure-Python (+5 more)

### Community 56 - "Community 56"
Cohesion: 0.17
Nodes (12): build_theta_untransform(), _guided_diffusion(), guided_diffusion_class(), _make_guided_class(), _no_summary_network(), Any, Simulator-gradient guidance for BayesFlow's diffusion sampler.  Idea ---- During, Return the map from diffusion-state space to physical parameter space.      Baye (+4 more)

### Community 57 - "Community 57"
Cohesion: 0.50
Nodes (3): import_submodules(), Auto-import the modules of a package so ``@register_*`` decorators run.  The reg, Import every non-underscore module directly inside a package.      Call from a p

### Community 58 - "Simulator-gradient guidance in diffusion sampling"
Cohesion: 0.10
Nodes (20): 1. Where the guidance term goes, 2. The two arms, 3. Measured results, 3a. Guidance vs the model's own prediction (`diagnose_guidance`, Arm B), 3b. Does guidance carry real information? (Arm A, 8 observations, 200 draws), 3c. The blow-up is not a solver artifact, 4. Pros, cons, pitfalls, 5. Reference posterior — status (+12 more)

### Community 59 - "get_run_dir"
Cohesion: 0.22
Nodes (8): Stage 1: dataset generation.  Samples the prior and runs the forward model in ch, Generate the dataset described by ``cfg`` and return its path., run_simulation(), get_run_dir(), Run-directory helpers and shared artifact filenames., Return the current Hydra run output dir (works regardless of the ``job.chdir`` s, Copy Hydra's auto-generated ``.hydra/`` config folder next to a generated artifa, save_config_snapshot()

### Community 60 - "run_training"
Cohesion: 0.29
Nodes (9): _n(), Stage 2: training.  Load dataset -> preprocessing pipeline (fit on train, save f, Load the best-val-loss weights BayesFlow checkpointed during training back into, Persist ``history.json`` + ``convergence.json``; best-effort, never fails a run., Train the approximator and return (workflow, history)., _restore_best_weights(), run_training(), _save_history_and_convergence() (+1 more)

### Community 61 - "test_registries.py"
Cohesion: 0.33
Nodes (5): available_steps(), Name -> preprocessing-step registry and pipeline builder., Register a step factory (usually the step class itself) under ``name``., register_step(), test_preprocess_registry()

### Community 62 - "run_evaluation"
Cohesion: 0.33
Nodes (8): Stage 3: evaluation on a simulated test set (with known ground truth).  Loads th, Resolve ``<node>.sample_kwargs`` into a plain dict of extra ``workflow.sample``, Best-effort ``report.md`` from the metrics/figures just written; never aborts a, _require_model_dir(), _run_diagnostics(), run_evaluation(), _sample_kwargs(), _write_report()

### Community 63 - "seed_everything"
Cohesion: 0.25
Nodes (7): Stage 5: application to real (observed) data.  Like :mod:`evaluate`, but the inp, Save a posterior pair plot per observation (real data has no ground truth)., run_real_evaluation(), _save_posterior_plot(), Seeding helpers for reproducible runs., Seed Python, NumPy, and (best effort) the active Keras backend.      Returns a N, seed_everything()

### Community 64 - "LogTransform"
Cohesion: 0.25
Nodes (5): LogTransform, Dataset, Natural-log transform for strictly positive observables.  Motivating case: the L, Stateless ``x -> log(max(x, floor))``, inverted by ``exp``., test_log_transform_roundtrip()

### Community 65 - "get_simulator"
Cohesion: 0.20
Nodes (10): get_simulator(), Instantiate the simulator selected by ``cfg.simulator`` (a ``SimulatorConfig``)., test_two_moons_shapes_and_reproducibility(), Same seed => bit-identical dataset, including the JAX-side observation noise., A simulator that does not opt in must raise a clear error, not fail obscurely., test_base_simulator_guidance_seam_is_optional(), test_lv_simulator_is_reproducible(), Registry resolution + skeleton-simulator behavior. (+2 more)

### Community 66 - "load_approximator"
Cohesion: 0.32
Nodes (7): fix_keras_model(), load_approximator(), Any, Model save/load helpers, including the BayesFlow ``.keras`` deserialization work, Return a path to a load-safe copy of ``model_path`` (patching the ArrayImpl tag), Load a saved approximator, applying the ArrayImpl fix first., save_approximator()

### Community 67 - "quiet.py"
Cohesion: 0.29
Nodes (7): quiet_worker(), Silence noisy C-extension output during simulation.  A simulator backed by a C/C, True unless ``HYDRABFLOW_SIM_QUIET`` is set to a falsy value., Redirect fd 1 & 2 to ``/dev/null`` for the duration of the block (C-level output, Decorator: run a (joblib) worker with its C-level stdout/stderr redirected to /d, sim_quiet_enabled(), suppress_c_stdio()

### Community 68 - "compose_cfg"
Cohesion: 0.50
Nodes (4): cfg(), compose_cfg(), Shared test fixtures., Compose the root config with the structured schemas registered.      ``fill=True

## Knowledge Gaps
- **151 isolated node(s):** `hydrabflow`, `graphify`, `Usage`, `What graphify is for`, `Step 0 - GitHub repos and multi-path merge (only if a URL or several paths)` (+146 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **34 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `get_simulator()` connect `get_simulator` to `Config Schemas`, `Config Composition Tests`, `Community 11`, `JAX Backend Pin`, `Logging Helper`, `get_run_dir`?**
  _High betweenness centrality (0.152) - this node is a cross-community bridge._
- **Why does `BaseSimulator` connect `Config Schemas` to `Community 40`, `get_simulator`?**
  _High betweenness centrality (0.136) - this node is a cross-community bridge._
- **Why does `run_guided_evaluation()` connect `Logging Helper` to `get_simulator`, `load_approximator`, `Community 39`, `JAX Backend Pin`, `get_run_dir`, `run_evaluation`, `seed_everything`?**
  _High betweenness centrality (0.096) - this node is a cross-community bridge._
- **Are the 7 inferred relationships involving `PreprocessStep` (e.g. with `PreprocessPipeline` and `LogTransform`) actually correct?**
  _`PreprocessStep` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 16 inferred relationships involving `compose()` (e.g. with `_compose_aug()` and `test_adapter_derived_from_simulator()`) actually correct?**
  _`compose()` has 16 INFERRED edges - model-reasoned connections that need verification._
- **Are the 11 inferred relationships involving `run_guided_evaluation()` (e.g. with `load_approximator()` and `_sample_kwargs()`) actually correct?**
  _`run_guided_evaluation()` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 14 inferred relationships involving `register_configs()` (e.g. with `AdapterConfig` and `AugmentationConfig`) actually correct?**
  _`register_configs()` has 14 INFERRED edges - model-reasoned connections that need verification._
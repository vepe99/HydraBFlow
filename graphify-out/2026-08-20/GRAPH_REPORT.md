# Graph Report - HydraBFlow  (2026-08-20)

## Corpus Check
- 63 files · ~58,287 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 831 nodes · 1107 edges · 110 communities (65 shown, 45 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 114 edges (avg confidence: 0.77)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `d168ec3f`
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
- [[_COMMUNITY_test_prior_predictive_checks.py|test_prior_predictive_checks.py]]
- [[_COMMUNITY_PackageInit cluster 24|Package/Init cluster 24]]
- [[_COMMUNITY_Evaluate Entry Script|Evaluate Entry Script]]
- [[_COMMUNITY__realdisk.py|_realdisk.py]]
- [[_COMMUNITY_test_protoplan.py|test_protoplan.py]]
- [[_COMMUNITY_The protoplanetary-disk project|The protoplanetary-disk project]]
- [[_COMMUNITY_protoplan.py|protoplan.py]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_build_real_batch|build_real_batch]]
- [[_COMMUNITY_GroupedFlowMatching|GroupedFlowMatching]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_ProtoplanetaryDiskSimulator|ProtoplanetaryDiskSimulator]]
- [[_COMMUNITY_permissions|permissions]]
- [[_COMMUNITY_compose|compose]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_AST Structural Extraction|AST Structural Extraction]]
- [[_COMMUNITY_ConditionedFusionNetwork|ConditionedFusionNetwork]]
- [[_COMMUNITY_registry.py|registry.py]]
- [[_COMMUNITY_ConditionedConvolutionalNetwork|ConditionedConvolutionalNetwork]]
- [[_COMMUNITY__objective|_objective]]
- [[_COMMUNITY_build_workflow|build_workflow]]
- [[_COMMUNITY_test_registries.py|test_registries.py]]
- [[_COMMUNITY_EXTRACTEDINFERREDAMBIGUOUS Audit Trail|EXTRACTED/INFERRED/AMBIGUOUS Audit Trail]]
- [[_COMMUNITY_disk_config|disk_config]]
- [[_COMMUNITY_run_evaluation|run_evaluation]]
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
- [[_COMMUNITY_get_run_dir|get_run_dir]]
- [[_COMMUNITY_test_workflow.py|test_workflow.py]]
- [[_COMMUNITY_build_pipeline|build_pipeline]]
- [[_COMMUNITY__ext_hiav|_ext_hiav]]
- [[_COMMUNITY_Return the summary network selected by ``cfg.type`` (a ``SummaryNetworkConfig``)|Return the summary network selected by ``cfg.type`` (a ``SummaryNetworkConfig``)]]
- [[_COMMUNITY_Return the inference (posterior) network selected by ``cfg.type`` (an ``Inferenc|Return the inference (posterior) network selected by ``cfg.type`` (an ``Inferenc]]
- [[_COMMUNITY__vendor_real_images.py|_vendor_real_images.py]]
- [[_COMMUNITY_Same seed + same step list - identical end-to-end result through build_augmenta|Same seed + same step list -> identical end-to-end result through build_augmenta]]
- [[_COMMUNITY_A step's random stream is its own spawn child, so it doesn't depend on trailing|A step's random stream is its own spawn child, so it doesn't depend on trailing]]
- [[_COMMUNITY_Build a single augmentation through the public registry with a seeded generator.|Build a single augmentation through the public registry with a seeded generator.]]
- [[_COMMUNITY_At non-trivial strength, each augmentation changes the batch.|At non-trivial strength, each augmentation changes the batch.]]
- [[_COMMUNITY_Same seed - bit-identical augmented output.|Same seed -> bit-identical augmented output.]]
- [[_COMMUNITY_Different seed - different draws (the randomness is genuinely seed-controlled).|Different seed -> different draws (the randomness is genuinely seed-controlled).]]
- [[_COMMUNITY_Randomness comes only from the injected generator, not global np.random.|Randomness comes only from the injected generator, not global np.random.]]
- [[_COMMUNITY_Accum|Accum]]
- [[_COMMUNITY_compose_cfg|compose_cfg]]
- [[_COMMUNITY_.__init__|.__init__]]
- [[_COMMUNITY_make_real_npz.py|make_real_npz.py]]
- [[_COMMUNITY_mahalanobis|mahalanobis]]
- [[_COMMUNITY_missing_alma_channels|missing_alma_channels]]
- [[_COMMUNITY_normal_score_block|normal_score_block]]
- [[_COMMUNITY__radius_at_fraction|_radius_at_fraction]]
- [[_COMMUNITY_regrid_to_model|regrid_to_model]]
- [[_COMMUNITY_check_sed_range.py|check_sed_range.py]]

## God Nodes (most connected - your core abstractions)
1. `AugmentationsClass` - 39 edges
2. `compose()` - 18 edges
3. `build_workflow()` - 16 edges
4. `ConditionedFusionNetwork` - 14 edges
5. `PreprocessStep` - 14 edges
6. `GroupedFlowMatching` - 12 edges
7. `build_real_batch()` - 11 edges
8. `run_training()` - 11 edges
9. `build_summary_network()` - 11 edges
10. `BaseSimulator` - 11 edges

## Surprising Connections (you probably didn't know these)
- `test_build_workflow()` --calls--> `build_workflow()`  [INFERRED]
  tests/test_workflow.py → src/hydrabflow/pipeline/workflow.py
- `test_unknown_preprocess_step_errors()` --calls--> `build_pipeline()`  [INFERRED]
  tests/test_registries.py → src/hydrabflow/registry.py
- `test_build_adapter()` --calls--> `build_adapter()`  [INFERRED]
  tests/test_workflow.py → src/hydrabflow/simulators/_protoplan_spec.py
- `load_cfg()` --calls--> `register_configs()`  [INFERRED]
  notebooks/prior_predictive_checks/_realdisk.py → src/hydrabflow/config.py
- `load_cfg()` --calls--> `compose()`  [INFERRED]
  notebooks/prior_predictive_checks/_realdisk.py → tests/conftest.py

## Import Cycles
- None detected.

## Communities (110 total, 45 thin omitted)

### Community 0 - "Preprocessing Pipeline & Steps"
Cohesion: 0.07
Nodes (25): PreprocessPipeline, PreprocessStep, Dataset, ndarray, Preprocessing step protocol and the pipeline that runs them.  A step transforms, Dataset-in, dataset-out transform with optional fitted state., Estimate any state from ``data`` (train split). Stateless steps leave this empty, Return a transformed copy/view of ``data``. (+17 more)

### Community 1 - "Eval / Checkpoint Stages"
Cohesion: 0.33
Nodes (6): Model Default Config, Diffusion Inference Network Config, Flow Matching Inference Network Config, DeepSet Summary Network Config, SetTransformer Summary Network Config, TimeSeriesTransformer Summary Network Config

### Community 2 - "Design Principles & Configs"
Cohesion: 0.24
Nodes (8): _n(), Stage 2: training.  Load dataset -> preprocessing (fit on train, save the state, Train the approximator and return (workflow, history)., run_training(), _save_loss_plot(), Seeding helpers for reproducible runs., Seed Python, NumPy, and (best effort) Keras.      Returns a ``Generator`` to thr, seed_everything()

### Community 3 - "Augmentation Registry & Tests"
Cohesion: 0.07
Nodes (27): AdapterConfig, AugmentationConfig, DataConfig, EvalConfig, InferenceNetworkConfig, ModelConfig, PreprocessingConfig, Typed config schema. ``conf/config.yaml`` fills these in; the factories read the (+19 more)

### Community 4 - "Simulate Stage & Registries"
Cohesion: 0.28
Nodes (9): build_inference_network(), build_summary_network(), Any, Return the summary network selected by ``cfg.type``.      One key (the default):, Return the inference (posterior) network selected by ``cfg.type``., Adding an experimental architecture = one decorated function, no infrastructure, test_custom_network_builder_registers(), test_network_registries_list_available_on_unknown_type() (+1 more)

### Community 5 - "Example Simulators (Skeleton/TwoMoons)"
Cohesion: 0.17
Nodes (14): BaseException, is_oom_error(), T, Retry a GPU computation at a smaller batch size when it runs out of memory.  JAX, True if ``exc`` looks like a GPU out-of-memory error (matches on the message)., Call ``fn(batch_size)``, halving the batch size on OOM until ``min_batch``., run_with_oom_backoff(), _FakeOOM (+6 more)

### Community 6 - "Config Schemas"
Cohesion: 0.08
Nodes (27): epoch_log_callback(), fix_keras_model(), load_approximator(), load_prior_bounds(), _mask_rows(), _per_branch_figures(), Any, What a stage writes into its run directory: model, loss curve, posterior, diagno (+19 more)

### Community 7 - "Network Factory & Adapter"
Cohesion: 0.05
Nodes (40): RuntimeError, AugmentationsClass, compute_sed_bin_statistics(), protoplan_instrument(), ndarray, protoplan_instrument.py ======================= Vectorised post-processing pipel, JIT-compiled correlated ALMA noise for one channel.          Convolves i.i.d. wh, Three unit-conversion steps, all fully vectorised:          1. Extinction correc (+32 more)

### Community 9 - "Base Simulator Interface"
Cohesion: 0.07
Nodes (21): ABC, BaseSimulator, BaseSimulator, Any, ndarray, Base interface every forward model implements.  A simulator is the only piece a, Abstract forward model. Subclass + register via ``@register_simulator``., Draw ``n`` prior samples. Returns ``{param_name: (n, 1)}``. (+13 more)

### Community 10 - "Config Composition Tests"
Cohesion: 0.16
Nodes (15): load_cfg(), Compose the root Hydra config, exactly as the CLI stages do.      The notebooks, adapter_keys(), _as_list(), build_adapter(), fill_adapter_from_simulator(), _lists(), Any (+7 more)

### Community 11 - "Community 11"
Cohesion: 0.25
Nodes (14): _batch(), _build_one(), Two Moons simulator + the augmentation reproducibility/stochasticity contract., Build the shipped augmentation through the registry with a seeded generator., The shipped config trains without augmentation: zero scale must leave the batch, Consecutive calls on the *same* built augmentation differ (re-drawn every batch), Same seed + same step list -> identical result through the public builder., test_actually_perturbs() (+6 more)

### Community 12 - "Dataset IO"
Cohesion: 0.12
Nodes (20): _(), _(), Marimo notebook: is a real disk's *amplitude* inside the training population?  T, Marimo notebook: is a real disk's *morphology* reachable by the simulator?  The, concatenate_chunks(), load_config_dataset(), load_dataset(), n_rows() (+12 more)

### Community 13 - "Hydra App Boilerplate"
Cohesion: 0.33
Nodes (5): gaussian_noise(), jax_noise(), Augmentation, Observational-noise augmentations — and the template for your own.  An augmentat, Add zero-mean Gaussian noise to one observable key.      Params: ``noise_key`` (

### Community 14 - "JAX Backend Pin"
Cohesion: 0.33
Nodes (5): limit_gpus(), Pin GPU selection and the Keras backend *before* keras/bayesflow/JAX import anyw, Pin ``CUDA_VISIBLE_DEVICES`` to the least-used GPU(s) before JAX/CUDA initialize, Set ``KERAS_BACKEND`` unless the user already chose one. Returns the active back, set_backend()

### Community 23 - "test_prior_predictive_checks.py"
Cohesion: 0.11
Nodes (17): feats(), gaussian(), Analytic checks for the prior-predictive notebooks' shared numerics.  `_realdisk, A registry that has drifted from the measurements is a silent wrong-beam bug., A scalar A_V gives one correction for the population; a range gives one row each, Why sampling A_V cannot meaningfully move an ALMA finding.      Extinction enter, The A_V claim, pinned.      Extinction enters as one scalar per channel, so with, ring() (+9 more)

### Community 25 - "Evaluate Entry Script"
Cohesion: 0.22
Nodes (4): Dataset, ndarray, Per-feature z-score standardization step.  Mean/std are fit on the train split o, Standardizer

### Community 26 - "_realdisk.py"
Cohesion: 0.08
Nodes (23): av_ref(), av_suffix(), jwst_footprint_gap(), neighbour_panel(), parse_av(), plot_dir(), randomized_alma_obs_px(), Shared machinery for the prior-predictive check notebooks.  The notebooks beside (+15 more)

### Community 27 - "test_protoplan.py"
Cohesion: 0.12
Nodes (13): data_path(), Checks for the protoplanetary-disk arm: labels, adapter/network agreement, one t, Each band's CNN sees its own beam/noise plus the shared geometry, and nothing el, A `split_combined/`-shaped directory of five `.npy` caches., The one config error that would otherwise be silent: a differently-sampled image, `av: [lo, hi]` draws one extinction per disk; `av: x` is the old batch-wide cons, The corner plot survives a prior box, a median and a truth row., 17 targets `(N,1)`, two conditions flat `(N,)` and exactly 0/1, no sentinel left (+5 more)

### Community 28 - "The protoplanetary-disk project"
Cohesion: 0.11
Nodes (18): 10. Troubleshooting, 1. What the pipeline is, 2. Prerequisites, 3. The knobs: beam and noise, 4. Three coupled choices in the encoding, 5. Four image branches, one per band, 6. Reading the output, 7. Checks (+10 more)

### Community 29 - "protoplan.py"
Cohesion: 0.18
Nodes (15): _hp(), _protoplan_flow_matching(), _protoplan_fusion(), The two conditioned summary networks and the modality-dropout flow, plus their b, The architecture dict out of `cfg.params.hp`, defaulted key by key.      Default, `hp[f"{name}_{tag}"]`, falling back to the shared `_alma` value for an ALMA band, Per-backbone output width, keyed by input key.  Also the `group_sizes` source., One conditioned CNN per image band (JWST + ALMA B9/B7/B6) + a SED transformer, l (+7 more)

### Community 31 - "Community 31"
Cohesion: 0.08
Nodes (23): For /graphify add and --watch, For /graphify query, For the commit hook and native CLAUDE.md integration, For --update and --cluster-only, /graphify, Honesty Rules, Interpreter guard for subcommands, Part A - Structural extraction for code files (+15 more)

### Community 32 - "build_real_batch"
Cohesion: 0.15
Nodes (15): alma_obs_px(), build_real_batch(), check_conditions_in_prior(), extinction_correction(), interp_real_sed(), load_real_images(), obs_axes(), ndarray (+7 more)

### Community 33 - "GroupedFlowMatching"
Cohesion: 0.18
Nodes (9): GroupedFlowMatching, `FlowMatching` with modality-coherent condition dropout, in place of stock     `, `(batch_size, sum(group_sizes))`, 0 on every column of a dropped group., Zero the columns of `conditions` that `mask` marks unobserved.          Upstream, The sampling-time counterpart of the masking in `compute_metrics`.          `_in, With p=1 the mask is group-coherent, keep_one leaves one droppable group, and th, The mask must reach the network, under either subnet.      Regression test for a, test_missing_modality_mask_actually_changes_the_output() (+1 more)

### Community 34 - "Community 34"
Cohesion: 0.20
Nodes (13): alma_band_condition_keys(), _asinh_images(), build_adapter(), Parameter spec, condition layout and adapter for the protoplanetary-disk RT data, The five batch keys describing ALMA channel `j`'s beam and noise, in column orde, `{summary_group_key: [scalar batch keys, in concatenation order]}`., Every member of `summary_variables`: the five modalities plus the condition grou, Compress each image key as `asinh(x / sigma)`, per band.      Without it the ima (+5 more)

### Community 35 - "Community 35"
Cohesion: 0.05
Nodes (41): Design principles, docs/, Four image branches, one per band, graphify, HydraBFlow: SBI pipeline template (BayesFlow + Hydra), Layout, Notes worth keeping, Prior-predictive checks on a real disk (+33 more)

### Community 37 - "ProtoplanetaryDiskSimulator"
Cohesion: 0.17
Nodes (9): ProtoplanetaryDiskSimulator, ndarray, The protoplanetary-disk RT dataset, as a simulator that owns its data.  The forw, Reader for the cached RT dataset.  See `conf/simulator/protoplan.yaml` for the k, The cached RT rows as one dataset dict, row-aligned and filtered.          The j, The bespoke adapter (asinh images, per-instrument routed conditions, SED fusion), filter_and_jitter_params(), Read `params_combined.npy` and apply the discrete-as-conditions encoding.      T (+1 more)

### Community 39 - "compose"
Cohesion: 0.19
Nodes (10): compose(), Expose the composer so tests can build configs with custom overrides., Config composition + schema validation smoke tests., The typed schema is the only validation layer now that group YAMLs have no base, test_adapter_derived_from_simulator(), test_adapter_explicit_config_wins(), test_group_override(), test_unknown_key_is_rejected() (+2 more)

### Community 46 - "Community 46"
Cohesion: 0.19
Nodes (13): _deep_set(), _diffusion(), _embed_dim(), _flow_matching(), Any, The shipped network builders. A builder maps a network config to a BayesFlow net, Attention width, expressed per head so ``embed_dim % num_heads == 0`` always hol, _set_transformer() (+5 more)

### Community 48 - "ConditionedFusionNetwork"
Cohesion: 0.27
Nodes (4): ConditionedFusionNetwork, `FusionNetwork` plus a routing table saying which entries of `summary_variables`, `(B, sum_of_widths)` for a routed backbone, `None` for an unrouted one., Mirrors `FusionNetwork.compute_metrics`, including collecting an optional

### Community 49 - "registry.py"
Cohesion: 0.16
Nodes (12): load_training_data(), The row-filtered training cache, via the configured simulator's own `load_datase, Stage 1: dataset generation.  Samples the prior and runs the forward model in ch, Generate the dataset described by ``cfg`` and return its path., run_simulation(), get_simulator(), Everything you can register, in one file.  Five extension points share one mecha, Instantiate the simulator selected by ``cfg.simulator``. (+4 more)

### Community 50 - "ConditionedConvolutionalNetwork"
Cohesion: 0.20
Nodes (3): ConditionedConvolutionalNetwork, A `ConvolutionalNetwork` whose pooled features are concatenated with a condition, SummaryNetwork

### Community 51 - "_objective"
Cohesion: 0.33
Nodes (9): _objective(), Stage 4: hyperparameter tuning with Optuna.  A multi-objective study (RMSE + cal, Save the fit-once preprocessing state, shared by every trial.      Written atomi, Save the fit-once preprocessing state, shared by every trial/model.      Written, _report(), run_tuning(), _save_shared_preprocessing(), _suggest() (+1 more)

### Community 52 - "build_workflow"
Cohesion: 0.13
Nodes (17): build_workflow(), Any, Assemble the ``bf.BasicWorkflow`` (adapter + summary network + inference network, Build a ``bf.BasicWorkflow`` from the root ``cfg``.      ``run_dir`` (passed by, Build a ``bf.BasicWorkflow`` from the root ``cfg``., _augmentor(), The augmentation's key names are the adapter's inputs; a rename drift is silent, Adapter -> four CNNs + transformer -> flow, and a gradient that actually moves w (+9 more)

### Community 53 - "test_registries.py"
Cohesion: 0.22
Nodes (8): build_augmentations(), Augmentation, Build the ordered augmentation list from ``cfg.augmentation``.      Each step ge, Registry resolution: unknown names fail loudly, custom builders plug in., Every extension point is a Registry filled on discovery (registry.py)., test_augmentation_registry_builds(), test_shipped_components_are_registered(), test_unknown_preprocess_step_errors()

### Community 56 - "disk_config"
Cohesion: 0.32
Nodes (7): configured_augmentation(), disk_config(), fixed_setup_augmentation(), load_obs_setup(), `(setup, rot_deg)` for one disk, from `assets/protoplan/obs_setup_measurements.j, The augmentation exactly as configured for training -- randomized observing prio, `AugmentationsClass` fixed to this disk's own measured beams, noise, distance an

### Community 57 - "run_evaluation"
Cohesion: 0.36
Nodes (7): observed_condition_mask(), ndarray, Stage 3: posterior inference on held-out data, simulated or real.  Loads the app, ``(n_rows, sum(group_sizes))``, 0 on every column of a group named in ``drop``., _require_model_dir(), _run_diagnostics(), run_evaluation()

### Community 72 - "get_run_dir"
Cohesion: 0.29
Nodes (7): get_run_dir(), Run-directory helpers., The current Hydra run output dir (works regardless of the ``job.chdir`` setting), Return the current Hydra run output dir (works regardless of the ``job.chdir`` s, Copy Hydra's auto-generated ``.hydra/`` config folder next to a generated artifa, Copy Hydra's ``.hydra/`` config folder next to a generated dataset, keyed by its, save_config_snapshot()

### Community 73 - "test_workflow.py"
Cohesion: 0.25
Nodes (7): Adapter / network / workflow construction. Skipped if bayesflow isn't installed., Passing run_dir turns on BayesFlow best-weights checkpointing; omitting it leave, Two observable keys -> one backbone per key behind a FusionNetwork, types from p, test_build_adapter(), test_build_workflow(), test_build_workflow_checkpointing(), test_fusion_when_several_summary_variables()

### Community 74 - "build_pipeline"
Cohesion: 0.43
Nodes (6): build_pipeline(), Build the preprocessing pipeline from ``cfg.preprocessing``.      Each entry in, Preprocessing pipeline: fit/transform/split + state save/load round-trip., test_pipeline_fit_transform_and_split(), test_state_roundtrip(), _toy_data()

### Community 75 - "_ext_hiav"
Cohesion: 0.33
Nodes (6): _ext_hiav(), extinction_ratio(), implied_av(), `A_lam / A_K` from the McClure09 table `AugmentationsClass` itself reads., `corr(av)/corr(av_ref)` at `lam_um`, from the same formula the augmentation uses, The A_V at which the model's flux at `lam_um` would move by `shift_dex` decades.

### Community 78 - "_vendor_real_images.py"
Cohesion: 0.40
Nodes (5): _crop(), Vendor the real-disk images into `assets/protoplan/` -- run once, then the noteb, `(img, ra_1d, dec_1d)` cut to +/-HALF_ARCSEC.  Accepts 1-D or 2-D coordinate arr, vendor(), Path

### Community 98 - "compose_cfg"
Cohesion: 0.27
Nodes (6): discover(), T, A named collection filled by ``@registry.add("name")`` decorators.      ``packag, Import ``self.package``'s modules so their decorators have run. Idempotent., Import every non-underscore module in a package, so its decorators run., Registry

### Community 101 - "make_real_npz.py"
Cohesion: 0.50
Nodes (3): build(), Write one real disk's observation as an ``evaluate``-ready ``.npz``.  ``hydrabfl, `(dataset, mask_groups)` -- the rows to save and the branches to mask when sampl

### Community 102 - "mahalanobis"
Cohesion: 0.50
Nodes (4): mahalanobis(), percentile_rank(), Fraction of training rows below `value`, in percent., `(d2 of every training row, d2 of the real disk, percentile of the latter)`.

### Community 103 - "missing_alma_channels"
Cohesion: 0.50
Nodes (4): missing_alma_channels(), present_image_keys(), The ALMA channels this disk was never observed in -- dropped everywhere downstre, `IMAGE_KEYS` minus the bands this disk has no measurement for.

### Community 105 - "normal_score_block"
Cohesion: 0.50
Nodes (4): normal_score_block(), normal_scores(), Rank-transform a training column to standard normal scores through its own empir, Column-wise `normal_scores` over a whole feature matrix -> ((N,p), (p,)).

### Community 107 - "_radius_at_fraction"
Cohesion: 0.50
Nodes (4): _radius_at_fraction(), Radius enclosing `frac` of the flux, linearly interpolated in the cumulative pro, Flux-normalised morphology of a batch of observed-grid images.      Parameters, shape_features()

### Community 108 - "regrid_to_model"
Cohesion: 0.50
Nodes (4): Flip `img` along `axis` if `coord` descends, so both end up ascending., Bilinearly resample a real image onto the square model observed grid.      Conve, regrid_to_model(), _to_ascending()

## Knowledge Gaps
- **97 isolated node(s):** `hydrabflow`, `ModelConfig`, `DataConfig`, `TrainingConfig`, `TuningConfig` (+92 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **45 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `AugmentationsClass` connect `Network Factory & Adapter` to `Accum`, `disk_config`, `test_protoplan.py`, `build_workflow`?**
  _High betweenness centrality (0.116) - this node is a cross-community bridge._
- **Why does `get_simulator()` connect `registry.py` to `Config Composition Tests`, `Simulate Stage & Registries`?**
  _High betweenness centrality (0.105) - this node is a cross-community bridge._
- **Why does `compose()` connect `compose` to `Community 34`, `test_workflow.py`, `Config Composition Tests`, `Dataset IO`, `Community 46`, `build_workflow`, `protoplan.py`?**
  _High betweenness centrality (0.071) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `AugmentationsClass` (e.g. with `Accum` and `_augmentor()`) actually correct?**
  _`AugmentationsClass` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 15 inferred relationships involving `compose()` (e.g. with `load_cfg()` and `test_adapter_derived_from_simulator()`) actually correct?**
  _`compose()` has 15 INFERRED edges - model-reasoned connections that need verification._
- **Are the 12 inferred relationships involving `build_workflow()` (e.g. with `run_evaluation()` and `run_training()`) actually correct?**
  _`build_workflow()` has 12 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Marimo notebook: inspect a training run's posterior samples and diagnostics.  Ru`, `Shared machinery for the prior-predictive check notebooks.  The notebooks beside`, `Compose the root Hydra config, exactly as the CLI stages do.      The notebooks` to the rest of the system?**
  _360 weakly-connected nodes found - possible documentation gaps or missing edges._
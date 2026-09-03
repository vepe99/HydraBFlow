# Graph Report - HydraBFlow  (2026-08-28)

## Corpus Check
- 67 files · ~64,515 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 860 nodes · 1150 edges · 113 communities (67 shown, 46 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 122 edges (avg confidence: 0.78)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `2945c8ed`
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
- [[_COMMUNITY_adapter.py|adapter.py]]
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
- [[_COMMUNITY__Batched|_Batched]]
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
- [[_COMMUNITY_HydraBFlow SBI pipeline template (BayesFlow + Hydra)|HydraBFlow: SBI pipeline template (BayesFlow + Hydra)]]
- [[_COMMUNITY_build_workflow|build_workflow]]
- [[_COMMUNITY_test_registries.py|test_registries.py]]
- [[_COMMUNITY_EXTRACTEDINFERREDAMBIGUOUS Audit Trail|EXTRACTED/INFERRED/AMBIGUOUS Audit Trail]]
- [[_COMMUNITY_disk_config|disk_config]]
- [[_COMMUNITY_Configuration|Configuration]]
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
- [[_COMMUNITY_Extending|Extending]]
- [[_COMMUNITY_The protoplanetary-disk arm (`protoplan_sbi` branch)|The protoplanetary-disk arm (`protoplan_sbi` branch)]]
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
- [[_COMMUNITY_check_summary_pca.py|check_summary_pca.py]]
- [[_COMMUNITY_4. Three coupled choices in the encoding|4. Three coupled choices in the encoding]]
- [[_COMMUNITY_HydraBFlow|HydraBFlow]]

## God Nodes (most connected - your core abstractions)
1. `AugmentationsClass` - 41 edges
2. `compose()` - 20 edges
3. `build_workflow()` - 17 edges
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
- `load_cfg()` --calls--> `register_configs()`  [INFERRED]
  notebooks/prior_predictive_checks/_realdisk.py → src/hydrabflow/config.py
- `load_cfg()` --calls--> `compose()`  [INFERRED]
  notebooks/prior_predictive_checks/_realdisk.py → tests/conftest.py
- `Accum` --uses--> `AugmentationsClass`  [INFERRED]
  notebooks/prior_predictive_checks/_realdisk.py → src/hydrabflow/augmentation/protoplan_instrument.py
- `summary_fn()` --calls--> `build_workflow()`  [INFERRED]
  notebooks/prior_predictive_checks/check_summary_range.py → src/hydrabflow/pipeline/workflow.py

## Import Cycles
- None detected.

## Communities (113 total, 46 thin omitted)

### Community 0 - "Preprocessing Pipeline & Steps"
Cohesion: 0.13
Nodes (13): CastDtype, DropNaNSimulations, _num_rows(), Dataset, Built-in stateless preprocessing steps (besides standardization).  Add your own, Drop rows (simulations) that contain any NaN/Inf in the listed keys., Random hold-out split. Steps listed after this one are fit on the train split on, Cast the listed keys (or all keys) to a target dtype, e.g. float32 for training. (+5 more)

### Community 1 - "Eval / Checkpoint Stages"
Cohesion: 0.33
Nodes (6): Model Default Config, Diffusion Inference Network Config, Flow Matching Inference Network Config, DeepSet Summary Network Config, SetTransformer Summary Network Config, TimeSeriesTransformer Summary Network Config

### Community 2 - "Design Principles & Configs"
Cohesion: 0.17
Nodes (9): ProtoplanetaryDiskSimulator, ndarray, The protoplanetary-disk RT dataset, as a simulator that owns its data.  The forw, Reader for the cached RT dataset.  See `conf/simulator/protoplan.yaml` for the k, The cached RT rows as one dataset dict, row-aligned and filtered.          The j, The bespoke adapter (asinh images, per-instrument routed conditions, SED fusion), filter_and_jitter_params(), Read `params_combined.npy` and apply the discrete-as-conditions encoding.      T (+1 more)

### Community 3 - "Augmentation Registry & Tests"
Cohesion: 0.07
Nodes (27): AdapterConfig, AugmentationConfig, DataConfig, EvalConfig, InferenceNetworkConfig, ModelConfig, PreprocessingConfig, Typed config schema. ``conf/config.yaml`` fills these in; the factories read the (+19 more)

### Community 4 - "Simulate Stage & Registries"
Cohesion: 0.22
Nodes (9): build_inference_network(), Return the inference (posterior) network selected by ``cfg.type``., Six groups -- the discrete indicators, then one per band/modality -- indicators, `discrete_condition_groups` splits the indicators into width-1 *droppable* group, `experiment=protoplan_mask_discrete_av`: A_V is a maskable condition, not a nuis, test_av_is_exported_per_row_and_becomes_a_third_droppable_group(), test_grouped_flow_matching_group_sizes(), test_per_indicator_condition_groups_are_droppable_and_the_old_layout_is_unchanged() (+1 more)

### Community 5 - "Example Simulators (Skeleton/TwoMoons)"
Cohesion: 0.05
Nodes (49): BaseException, observed_condition_mask(), ndarray, Stage 3: posterior inference on held-out data, simulated or real.  Loads the app, ``(n_rows, sum(group_sizes))``, 0 on every column of a group named in ``drop``., _require_model_dir(), _run_diagnostics(), run_evaluation() (+41 more)

### Community 6 - "Config Schemas"
Cohesion: 0.08
Nodes (27): epoch_log_callback(), fix_keras_model(), load_approximator(), load_prior_bounds(), _mask_rows(), _per_branch_figures(), Any, What a stage writes into its run directory: model, loss curve, posterior, diagno (+19 more)

### Community 7 - "Network Factory & Adapter"
Cohesion: 0.05
Nodes (39): RuntimeError, AugmentationsClass, compute_sed_bin_statistics(), protoplan_instrument(), ndarray, protoplan_instrument.py ======================= Vectorised post-processing pipel, JIT-compiled correlated ALMA noise for one channel.          Convolves i.i.d. wh, Three unit-conversion steps, all fully vectorised:          1. Extinction correc (+31 more)

### Community 9 - "Base Simulator Interface"
Cohesion: 0.06
Nodes (28): ABC, BaseSimulator, PreprocessPipeline, PreprocessStep, Dataset, ndarray, Preprocessing step protocol and the pipeline that runs them.  A step transforms, Dataset-in, dataset-out transform with optional fitted state. (+20 more)

### Community 10 - "Config Composition Tests"
Cohesion: 0.50
Nodes (3): check_conditions_in_prior(), Print each measured condition beside the prior the networks were trained on., Solid angle [sr] of an elliptical Gaussian beam given its FWHMs [arcsec].

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

### Community 25 - "adapter.py"
Cohesion: 0.16
Nodes (15): load_cfg(), Compose the root Hydra config, exactly as the CLI stages do.      The notebooks, adapter_keys(), _as_list(), build_adapter(), fill_adapter_from_simulator(), _lists(), Any (+7 more)

### Community 26 - "_realdisk.py"
Cohesion: 0.08
Nodes (23): av_ref(), av_suffix(), jwst_footprint_gap(), neighbour_panel(), parse_av(), plot_dir(), randomized_alma_obs_px(), Shared machinery for the prior-predictive check notebooks.  The notebooks beside (+15 more)

### Community 27 - "test_protoplan.py"
Cohesion: 0.11
Nodes (15): data_path(), Checks for the protoplanetary-disk arm: labels, adapter/network agreement, one t, Each band's CNN sees its own beam/noise plus the shared geometry, and nothing el, A `split_combined/`-shaped directory of five `.npy` caches., The one config error that would otherwise be silent: a differently-sampled image, `av: [lo, hi]` draws one extinction per disk; `av: x` is the old batch-wide cons, The corner plot survives a prior box, a median and a truth row., `random_flip=False` must still emit `sky_flip`, and it must be +1 for every row. (+7 more)

### Community 28 - "The protoplanetary-disk project"
Cohesion: 0.12
Nodes (16): 10. Troubleshooting, 1. What the pipeline is, 2. Prerequisites, 3. The knobs: beam and noise, 5. Four image branches, one per band, 6. Reading the output, 7. Checks, 8. Prior-predictive checks on a real disk (+8 more)

### Community 29 - "protoplan.py"
Cohesion: 0.18
Nodes (15): _hp(), _protoplan_flow_matching(), _protoplan_fusion(), The two conditioned summary networks and the modality-dropout flow, plus their b, The architecture dict out of `cfg.params.hp`, defaulted key by key.      Default, `hp[f"{name}_{tag}"]`, falling back to the shared `_alma` value for an ALMA band, Per-backbone output width, keyed by input key.  Also the `group_sizes` source., One conditioned CNN per image band (JWST + ALMA B9/B7/B6) + a SED transformer, l (+7 more)

### Community 31 - "Community 31"
Cohesion: 0.08
Nodes (23): For /graphify add and --watch, For /graphify query, For the commit hook and native CLAUDE.md integration, For --update and --cluster-only, /graphify, Honesty Rules, Interpreter guard for subcommands, Part A - Structural extraction for code files (+15 more)

### Community 32 - "build_real_batch"
Cohesion: 0.18
Nodes (13): alma_obs_px(), build_real_batch(), extinction_correction(), interp_real_sed(), load_real_images(), obs_axes(), ndarray, The `(N, L)` (or `(1, L)`) multiplicative extinction correction `preprocess` app (+5 more)

### Community 33 - "GroupedFlowMatching"
Cohesion: 0.18
Nodes (9): GroupedFlowMatching, `FlowMatching` with modality-coherent condition dropout, in place of stock     `, `(batch_size, sum(group_sizes))`, 0 on every column of a dropped group., Zero the columns of `conditions` that `mask` marks unobserved.          Upstream, The sampling-time counterpart of the masking in `compute_metrics`.          `_in, With p=1 the mask is group-coherent, keep_one leaves one droppable group, and th, The mask must reach the network, under either subnet.      Regression test for a, test_missing_modality_mask_actually_changes_the_output() (+1 more)

### Community 34 - "Community 34"
Cohesion: 0.16
Nodes (16): alma_band_condition_keys(), _asinh_images(), build_adapter(), Parameter spec, condition layout and adapter for the protoplanetary-disk RT data, The five batch keys describing ALMA channel `j`'s beam and noise, in column orde, `{summary_group_key: [scalar batch keys, in concatenation order]}`., Every member of `summary_variables`: the five modalities plus the condition grou, Compress each image key as `asinh(x / sigma)`, per band.      Without it the ima (+8 more)

### Community 35 - "Community 35"
Cohesion: 0.18
Nodes (11): A dataset with no simulator, GPU / CPU, Install, One config file per experiment, Running, The four stages, The protoplanetary-disk arm, Tuning (+3 more)

### Community 37 - "_Batched"
Cohesion: 0.24
Nodes (5): _Batched, _PerDraw, Per-draw simulators (``is_batched = False``) and the shape/name guard in ``BaseS, A toy forward model. Deterministic, so the batched and per-draw paths must agree, test_per_draw_matches_batched()

### Community 39 - "compose"
Cohesion: 0.24
Nodes (8): compose(), Expose the composer so tests can build configs with custom overrides., Config composition + schema validation smoke tests., The typed schema is the only validation layer now that group YAMLs have no base, test_adapter_derived_from_simulator(), test_adapter_explicit_config_wins(), test_group_override(), test_unknown_key_is_rejected()

### Community 46 - "Community 46"
Cohesion: 0.19
Nodes (13): _deep_set(), _diffusion(), _embed_dim(), _flow_matching(), Any, The shipped network builders. A builder maps a network config to a BayesFlow net, Attention width, expressed per head so ``embed_dim % num_heads == 0`` always hol, _set_transformer() (+5 more)

### Community 48 - "ConditionedFusionNetwork"
Cohesion: 0.27
Nodes (4): ConditionedFusionNetwork, `FusionNetwork` plus a routing table saying which entries of `summary_variables`, `(B, sum_of_widths)` for a routed backbone, `None` for an unrouted one., Mirrors `FusionNetwork.compute_metrics`, including collecting an optional

### Community 49 - "registry.py"
Cohesion: 0.22
Nodes (4): Dataset, ndarray, Per-feature z-score standardization step.  Mean/std are fit on the train split o, Standardizer

### Community 50 - "ConditionedConvolutionalNetwork"
Cohesion: 0.20
Nodes (3): ConditionedConvolutionalNetwork, A `ConvolutionalNetwork` whose pooled features are concatenated with a condition, SummaryNetwork

### Community 51 - "HydraBFlow: SBI pipeline template (BayesFlow + Hydra)"
Cohesion: 0.20
Nodes (9): Design principles, docs/, graphify, HydraBFlow: SBI pipeline template (BayesFlow + Hydra), Layout, Notes worth keeping, Stack, Stages (+1 more)

### Community 52 - "build_workflow"
Cohesion: 0.13
Nodes (18): build_workflow(), Any, Build a ``bf.BasicWorkflow`` from the root ``cfg``.      ``run_dir`` (passed by, Build a ``bf.BasicWorkflow`` from the root ``cfg``., _augmentor(), The augmentation's key names are the adapter's inputs; a rename drift is silent, Adapter -> four CNNs + transformer -> flow, and a gradient that actually moves w, The `group_sizes` width must equal the resolved condition width, or `ops.repeat` (+10 more)

### Community 53 - "test_registries.py"
Cohesion: 0.06
Nodes (37): load_training_data(), The row-filtered training cache, via the configured simulator's own `load_datase, Stage 1: dataset generation.  Samples the prior and runs the forward model in ch, Generate the dataset described by ``cfg`` and return its path., run_simulation(), Assemble the ``bf.BasicWorkflow`` (adapter + summary network + inference network, build_pipeline(), build_summary_network() (+29 more)

### Community 56 - "disk_config"
Cohesion: 0.32
Nodes (7): configured_augmentation(), disk_config(), fixed_setup_augmentation(), load_obs_setup(), `(setup, rot_deg)` for one disk, from `assets/protoplan/obs_setup_measurements.j, The augmentation exactly as configured for training -- randomized observing prio, `AugmentationsClass` fixed to this disk's own measured beams, noise, distance an

### Community 57 - "Configuration"
Cohesion: 0.33
Nodes (6): Blocks, Configuration, Network groups, No per-block `output_dir`, Notes that actually bite, `run_name` vs `model_dir`

### Community 72 - "Extending"
Cohesion: 0.33
Nodes (6): A forward model you cannot run in-process, Augmentation (stochastic, per batch), Extending, Preprocessing step (deterministic, once, fit on train), Your network, Your simulator

### Community 73 - "The protoplanetary-disk arm (`protoplan_sbi` branch)"
Cohesion: 0.40
Nodes (5): Four image branches, one per band, Prior-predictive checks on a real disk, The encoding: three coupled choices, The instrument model is an augmentation, The protoplanetary-disk arm (`protoplan_sbi` branch)

### Community 75 - "_ext_hiav"
Cohesion: 0.33
Nodes (6): _ext_hiav(), extinction_ratio(), implied_av(), `A_lam / A_K` from the McClure09 table `AugmentationsClass` itself reads., `corr(av)/corr(av_ref)` at `lam_um`, from the same formula the augmentation uses, The A_V at which the model's flux at `lam_um` would move by `shift_dex` decades.

### Community 78 - "_vendor_real_images.py"
Cohesion: 0.40
Nodes (5): _crop(), Vendor the real-disk images into `assets/protoplan/` -- run once, then the noteb, `(img, ra_1d, dec_1d)` cut to +/-HALF_ARCSEC.  Accepts 1-D or 2-D coordinate arr, vendor(), Path

### Community 98 - "compose_cfg"
Cohesion: 0.67
Nodes (3): chains(), main(), One corner plot per model: the training prior shaded behind, every condition reg

### Community 101 - "make_real_npz.py"
Cohesion: 0.15
Nodes (14): blocks(), main(), Is a real disk inside the training population *in the network's own summary spac, Zero-fill the target and indicator columns the adapter concatenates but we never, `(batch_dict) -> (B, summary_dim)`, standardized exactly as the flow's condition, Equal-width per-branch slices; every branch is built with the same `summary_dim`, summary_fn(), with_placeholders() (+6 more)

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

### Community 111 - "4. Three coupled choices in the encoding"
Cohesion: 0.50
Nodes (4): 4. Three coupled choices in the encoding, Discrete parameters are conditions, not targets, Images are compressed with asinh, The measurement conditions go into the summary networks

### Community 112 - "HydraBFlow"
Cohesion: 0.50
Nodes (4): Adding your own simulator, Design at a glance, HydraBFlow, Quickstart

## Knowledge Gaps
- **99 isolated node(s):** `hydrabflow`, `ModelConfig`, `DataConfig`, `TrainingConfig`, `TuningConfig` (+94 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **46 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `AugmentationsClass` connect `Network Factory & Adapter` to `Accum`, `Simulate Stage & Registries`, `Config Composition Tests`, `build_workflow`, `disk_config`, `test_protoplan.py`?**
  _High betweenness centrality (0.116) - this node is a cross-community bridge._
- **Why does `get_simulator()` connect `test_registries.py` to `adapter.py`?**
  _High betweenness centrality (0.098) - this node is a cross-community bridge._
- **Why does `compose()` connect `compose` to `Community 34`, `Simulate Stage & Registries`, `Dataset IO`, `Community 46`, `build_workflow`, `test_registries.py`, `adapter.py`, `protoplan.py`?**
  _High betweenness centrality (0.072) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `AugmentationsClass` (e.g. with `Accum` and `_augmentor()`) actually correct?**
  _`AugmentationsClass` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 17 inferred relationships involving `compose()` (e.g. with `load_cfg()` and `test_adapter_derived_from_simulator()`) actually correct?**
  _`compose()` has 17 INFERRED edges - model-reasoned connections that need verification._
- **Are the 13 inferred relationships involving `build_workflow()` (e.g. with `summary_fn()` and `run_evaluation()`) actually correct?**
  _`build_workflow()` has 13 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Marimo notebook: inspect a training run's posterior samples and diagnostics.  Ru`, `One corner plot per model: the training prior shaded behind, every condition reg`, `Shared machinery for the prior-predictive check notebooks.  The notebooks beside` to the rest of the system?**
  _372 weakly-connected nodes found - possible documentation gaps or missing edges._
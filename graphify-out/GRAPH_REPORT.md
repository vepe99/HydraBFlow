# Graph Report - HydraBFlow  (2026-09-15)

## Corpus Check
- 136 files · ~451,035 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1559 nodes · 2445 edges · 147 communities (97 shown, 50 thin omitted)
- Extraction: 83% EXTRACTED · 17% INFERRED · 0% AMBIGUOUS · INFERRED: 407 edges (avg confidence: 0.77)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `3db7835d`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Preprocessing Pipeline & Steps
- Eval / Checkpoint Stages
- Design Principles & Configs
- Augmentation Registry & Tests
- Simulate Stage & Registries
- Example Simulators (Skeleton/TwoMoons)
- Config Schemas
- Network Factory & Adapter
- Graphify Tooling
- Config Composition Tests
- Community 11
- Dataset IO
- Hydra App Boilerplate
- JAX Backend Pin
- Logging Helper
- Augmentation Package Init
- Package Root Init
- Marimo Notebook
- Pipeline Package Init
- Preprocessing Package Init
- Simulators Package Init
- Package/Init cluster 23
- Package/Init cluster 24
- Evaluate Entry Script
- Evaluate-Real Entry Script
- Simulate Entry Script
- Train Entry Script
- Tune Entry Script
- Community 31
- Community 32
- Community 33
- Community 34
- Community 35
- Community 36
- Community 37
- streams.py
- Community 39
- Community 40
- Community 41
- Community 42
- Community 43
- Community 44
- Part B — Support a file format other than `.npz`
- MaskedFusionNetwork
- Community 48
- Community 49
- Community 50
- Community 51
- Community 52
- Community 53
- compositional.py
- Community 56
- Community 57
- run_tuning
- build_workflow
- MaskedFusionNetwork
- Run stages (5 entry points)
- hooks
- PreToolUse
- permissions
- allow
- AST Structural Extraction
- EXTRACTED/INFERRED/AMBIGUOUS Audit Trail
- Community Detection
- Detect Files Step
- Existing-Graph Fast Path
- Gemini Extraction Backend
- God Nodes
- graph.json Output
- GRAPH_REPORT.md Output
- Python Interpreter Detection
- Knowledge Graph
- Obsidian Vault Export
- Semantic Extraction Cache
- Semantic LLM Extraction
- Parallel Subagent Dispatch
- hydrabflow
- simulate_multistream.py
- test_streams.py
- reporting.py
- Stream project (compositional score modeling)
- load_approximator
- Running a full pipeline with the Two Moons simulator
- _load_clean
- ppc_prior_predictive.py
- corner_parameters.py
- test_streams.py
- 2. The four commands (full run)
- compose_cfg
- _vcirc_worker
- create_rnbody_huang_dataset.sh
- training_eval_missing_vlos_ablation.sh
- training_eval_rnbody_huand_dataset.sh
- ndarray
- stream_agama.py
- stream_common.py
- assets/gaia — portable static inputs for the stream project
- StreamObservationStats
- compose
- PerStreamParameterStandardize
- prior_score_from_spec
- test_config.py
- get_run_dir
- prior_score_from_kde
- eval_rnbody_huand_dataset_kde_prior.sh
- training_eval_agama_1e6.sh
- training_eval_rnbody_huand_dataset copy.sh
- PerStreamParameterStandardize
- load_approximator
- prior_score_from_spec
- MaskVcircRadii
- compose_cfg
- ppc_ancillary_observables.py
- _spray_stream
- test_config.py
- test_workflow.py
- inferred_names
- create_ibata_dataset.sh
- train_ibata_sumstats.sh
- tune_ibata_sumstats.sh
- 4. Adding a SummaryNetwork that isn't shipped
- MaskedFusionNetwork
- PerStreamParameterStandardize
- create_ibata_onedisk_beta3_dataset.sh
- train_ibata_m200c.sh
- tune_ibata_m200c.sh
- AttachObservedSigmaZ
- apply_bayesflow_patches
- create_ibata_m200c_dataset.sh
- AttachObservedVterm
- create_ibata_rnbody_m200c_10kstars_test.sh
- PerStreamParameterStandardize
- AttachObservedVterm
- _objective
- create_ibata_rnbody_m200c_nfw_dataset.sh
- main
- compose_cfg

## God Nodes (most connected - your core abstractions)
1. `AgamaStreamSimulator` - 44 edges
2. `_jax()` - 32 edges
3. `compose()` - 32 edges
4. `get_simulator()` - 26 edges
5. `main()` - 24 edges
6. `PreprocessStep` - 24 edges
7. `main()` - 23 edges
8. `GroupedDiffusionModel` - 21 edges
9. `_sim_key()` - 17 edges
10. `_evaluate_compositional_global()` - 17 edges

## Surprising Connections (you probably didn't know these)
- `test_per_member_scores_flags_the_outlying_member()` --calls--> `per_member_scores()`  [INFERRED]
  tests/test_misspecification.py → src/hydrabflow/pipeline/misspecification.py
- `test_build_workflow()` --calls--> `build_workflow()`  [INFERRED]
  tests/test_workflow.py → src/hydrabflow/pipeline/workflow.py
- `test_two_moons_shapes_and_reproducibility()` --calls--> `get_simulator()`  [INFERRED]
  tests/test_augmentation.py → src/hydrabflow/registry.py
- `test_unknown_simulator_errors()` --calls--> `get_simulator()`  [INFERRED]
  tests/test_registries.py → src/hydrabflow/registry.py
- `test_unknown_preprocess_step_errors()` --calls--> `build_pipeline()`  [INFERRED]
  tests/test_registries.py → src/hydrabflow/registry.py

## Import Cycles
- None detected.

## Communities (147 total, 50 thin omitted)

### Community 0 - "Preprocessing Pipeline & Steps"
Cohesion: 0.06
Nodes (33): PreprocessPipeline, PreprocessStep, ABC, Dataset, ndarray, Preprocessing step protocol and the pipeline that orchestrates them.  A :class:`, Element-wise (dataset-in, dataset-out) transform with optional fitted state., Estimate any state from ``data`` (train split). Stateless steps leave this empty (+25 more)

### Community 1 - "Eval / Checkpoint Stages"
Cohesion: 0.33
Nodes (6): Model Default Config, Diffusion Inference Network Config, Flow Matching Inference Network Config, DeepSet Summary Network Config, SetTransformer Summary Network Config, TimeSeriesTransformer Summary Network Config

### Community 2 - "Design Principles & Configs"
Cohesion: 0.14
Nodes (18): _band(), main(), POSTERIOR-predictive checks for the Ibata (2023) ancillary potential observables, _build_pot_cfg(), _identity_constants(), _load_posterior(), _log10_keys(), main() (+10 more)

### Community 3 - "Augmentation Registry & Tests"
Cohesion: 0.09
Nodes (34): arm_cache_name(), arm_xy(), density_contours(), fiducial_row(), in_window(), load_or_run(), load_simulator(), main() (+26 more)

### Community 4 - "Simulate Stage & Registries"
Cohesion: 0.23
Nodes (15): _batch(), _build_one(), Two Moons simulator + the augmentation reproducibility/stochasticity contract., Build the shipped augmentation through the registry with a seeded generator., The shipped config trains without augmentation: zero scale must leave the batch, Consecutive calls on the *same* built augmentation differ (re-drawn every batch), Same seed + same step list -> identical result through the public builder., test_actually_perturbs() (+7 more)

### Community 5 - "Example Simulators (Skeleton/TwoMoons)"
Cohesion: 0.10
Nodes (19): BaseException, RuntimeError, Dataset, ndarray, Per-feature z-score standardization step.  Generalizes the reference project's `, Standardizer, is_oom_error(), T (+11 more)

### Community 6 - "Config Schemas"
Cohesion: 0.05
Nodes (25): BaseSimulator, BaseSimulator, ABC, Any, ndarray, Base interface every forward model implements.  A simulator is the only piece a, Draw ``n`` grouped datasets: one shared global draw + one local draw per member., Abstract forward model. Subclass + register via ``@register_simulator``. (+17 more)

### Community 7 - "Network Factory & Adapter"
Cohesion: 0.15
Nodes (23): _capture_score_calls(), _net(), Modality masking + the compositional over-counting it fixes.  The last test is t, A rank-3 plan mask must flatten to the same row order as `conditions`.      `_co, The fix, at the level of what the mask plan actually asserts.      m member item, Record what the stock ``DiffusionModel.score`` sees, and return zeros of the rig, The plan must reach the score call, per item, in the flattened row order.      W, Without an explicit prior score, upstream appends a zero-condition item to get t (+15 more)

### Community 10 - "Config Composition Tests"
Cohesion: 0.24
Nodes (10): adapter_keys(), composition_level(), fill_adapter_from_simulator(), _lists(), Build the BayesFlow ``Adapter``: dataset keys -> the roles BayesFlow expects.  `, Every dataset key the adapter consumes.      ``drop`` is included so dropped key, The four adapter key lists as plain lists (resolving interpolations)., ``cfg.composition.level`` as a plain string (``none`` when the block is absent). (+2 more)

### Community 11 - "Community 11"
Cohesion: 0.06
Nodes (15): AgamaStreamSimulator, One joblib ``delayed`` call per row. The forward-model seam: subclasses swap the, Stellar streams in a parametrized Milky Way potential, simulated with AGAMA., ``params.store_window_subsample`` -> ``{max_particles, pad_value, windows}`` wit, Split radius for the extended (Zhou u Huang) rotation-curve grid., Host-potential configuration threaded to the joblib workers. Legacy default (all, Requested Ibata ancillary observables (``params.ancillary_observables``); empty, Spec passed to the joblib worker: requested names + their fixed grids (or None). (+7 more)

### Community 12 - "Dataset IO"
Cohesion: 0.31
Nodes (8): _flat2d(), _group_level_flags(), main(), _per_member_safe(), ndarray, Per-channel model-misspecification MMD test (localize WHERE the misspecification, per_member_scores, but robust to scalar channels (F==1, e.g. sigma_z), where np., Which raw test-set channels are group-level (one value per potential, shared by

### Community 13 - "Hydra App Boilerplate"
Cohesion: 0.11
Nodes (11): GroupedDiffusionModel, ndarray, Width of the always-observed leading group, inferred from the condition vector., ``(1, width)`` row that is 1 on the columns of every group named in *observed*., One list of observed group names per compositional item (or ``None`` to disable), Condition groups observed by ordinary ``sample()`` calls (``None`` = all of them, ``(batch_size, width)``, 0 on every column of a dropped group.          The draw, Inject the per-item mask, then defer to the stock compositional score. (+3 more)

### Community 14 - "JAX Backend Pin"
Cohesion: 0.16
Nodes (18): main(), Stream-channel (sim_summary) misspecification MMD vs a large training-set refere, _bf_mmd(), member_summaries(), mmd_test(), _null_mmd(), per_member_scores(), ndarray (+10 more)

### Community 15 - "Logging Helper"
Cohesion: 0.19
Nodes (12): compose(), Expose the composer so tests can build configs with custom overrides., Stream-project components: config composition, hierarchy derivation, per-stream, `obs_r_grid: custom` makes the config table the vcirc grid (v4 config), the fill, The simulator-side storage window must be the augmentation's observation window., test_adapter_derivation_follows_composition_level(), test_custom_rotation_curve_grid_is_the_config_table(), test_simulator_declares_hierarchy() (+4 more)

### Community 23 - "Package/Init cluster 23"
Cohesion: 0.16
Nodes (20): auto_window(), build_template(), _hermite(), nearest_time(), orbit_window(), ndarray, Frenet-Serret stream remapping (Palau & Miralda-Escude 2023, MNRAS 524, 2124, Ap, Cubic Hermite interpolation of the orbit at arbitrary times ``t``.      Position (+12 more)

### Community 25 - "Evaluate Entry Script"
Cohesion: 0.20
Nodes (15): draw_halo_rows(), load_samples(), main(), panel_label(), panel_metrics(), Path, Keep ``n`` random stars of a template, as a new Template sharing the same knot g, ``n`` uniform draws of the halo prior, shared by all three streams. (+7 more)

### Community 26 - "Evaluate-Real Entry Script"
Cohesion: 0.25
Nodes (14): _build(), _params(), The final feature is the stream index j (so the MLP can distinguish streams)., A stream with zero measured v_los still yields finite track features (only its v, summary_include_std=false drops the per-bin std channels: (n, K, 14) -> (n, K, 9, The two occupancy channels hold the actual per-bin member counts, and dropping t, The flat variant exposes the same knobs: occupancy channels, and a MAD dispersio, test_summary_grid_median_only_layout() (+6 more)

### Community 27 - "Simulate Entry Script"
Cohesion: 0.29
Nodes (9): corner_matrix(), make_corner_figures(), The 6 observables as one rectangular array, restricted to stars with a measured, One observable-space corner per swept value: the three arms as contours, real da, main(), phi2_grid_figure(), project_sample(), One grid figure, one axis per sample, phi2 vs phi1 with the real members in grey (+1 more)

### Community 28 - "Train Entry Script"
Cohesion: 0.27
Nodes (6): discover(), T, A named collection filled by ``@registry.add("name")`` decorators.      ``packag, Import ``self.package``'s modules so their decorators have run. Idempotent., Import every non-underscore module in a package, so its decorators run., Registry

### Community 29 - "Tune Entry Script"
Cohesion: 0.25
Nodes (6): gaussian_noise(), jax_noise(), Augmentation, Observational-noise augmentations — and the template for your own.  An augmentat, Add zero-mean Gaussian noise to one observable key.      Params: ``noise_key`` (, Zero the columns of *conditions* that *mask* marks unobserved.          Only ``D

### Community 31 - "Community 31"
Cohesion: 0.08
Nodes (23): For /graphify add and --watch, For /graphify query, For the commit hook and native CLAUDE.md integration, For --update and --cluster-only, /graphify, Honesty Rules, Interpreter guard for subcommands, Part A - Structural extraction for code files (+15 more)

### Community 32 - "Community 32"
Cohesion: 0.16
Nodes (9): Shape, _fusion(), MaskedFusionNetwork, Layer, SummaryNetwork, Tensor, Multi-observable fusion summary network (attention-mask aware).  Consumes the di, Build a :class:`MaskedFusionNetwork` from ``cfg.params`` (see module docstring). (+1 more)

### Community 33 - "Community 33"
Cohesion: 0.25
Nodes (8): _bin_assignment(), _binned_stats(), Per-star bin index plus an ``in_range`` flag.      ``searchsorted`` returns 0 fo, Per-bin ``(median, dispersion, count)`` over the masked members.      ``vals`` i, The estimator fix. With ddof=0 the population std is biased low by sqrt((N-1)/N), Stars beyond the outermost φ1 edge used to be clipped into the end bins, inflati, test_binned_stats_is_unbiased_and_flags_sparse_bins(), test_out_of_range_stars_are_excluded_not_clipped()

### Community 34 - "Community 34"
Cohesion: 0.18
Nodes (10): Decisions Log, Design principles, docs/, graphify, HydraBFlow: SBI pipeline template (BayesFlow + Hydra), Layout, Notes worth keeping, Stack (+2 more)

### Community 35 - "Community 35"
Cohesion: 0.17
Nodes (17): _estimator_params(), _np_fit_frame(), _np_phi1(), _np_unit_vec(), Hand-crafted per-stream summary statistics in a data-driven stream-aligned frame, ``(scale, min_count, include_occupancy)`` from the augmentation params., Per-stream summary statistics laid out as a **φ1 time series** for a single, Rows of R are [x, y, z]: z = normal to best-fit plane (great-circle pole), x = m (+9 more)

### Community 37 - "Community 37"
Cohesion: 0.25
Nodes (8): ``type`` resolves through the summary-network registry (``networks/factory.py``), SummaryNetworkConfig, _masked_mlp(), The MLP ablation arm must hide exactly what the transformer arm hides: the stati, The Transformer alternative to `mlp` on the flat summary vector: reshapes (n, F), test_feature_transformer_backbone_forward(), test_masked_mlp_shares_the_occupancy_prep(), test_mlp_backbone_forward()

### Community 38 - "streams.py"
Cohesion: 0.09
Nodes (47): _add_noise_to_rho_z(), _add_noise_to_sigma_z(), _add_noise_to_vcirc(), _add_noise_to_vterm(), _apply_obs_error(), _compact_to_attended(), _concatenate_magnitudes(), _concatenate_sigma_errors() (+39 more)

### Community 39 - "Community 39"
Cohesion: 0.32
Nodes (7): _fusion_group_sizes(), _grouped_diffusion(), Modality-coherent condition masking for the diffusion inference network.  Ported, ``(subnet name, subnet kwargs)`` from the inference-network config.      ``param, Droppable group widths + names from the fusion summary network's backbones., ``DiffusionModel`` with per-modality condition dropout; groups read off the fusi, _subnet_kwargs()

### Community 40 - "Community 40"
Cohesion: 0.16
Nodes (20): apply_observed_groups(), condition_keys(), Raw batch keys that act as sampling conditions (everything the adapter consumes, Restrict ordinary ``sample()`` calls to the condition groups in ``eval.observed_, _evaluate_compositional_global(), _evaluate_local(), _load_model(), _load_test_data() (+12 more)

### Community 46 - "Part B — Support a file format other than `.npz`"
Cohesion: 0.36
Nodes (7): _mean_dist(), Unit tests for the summary-space misspecification helpers (no bayesflow import n, Cheap stand-in for MMD: distance between sample means., test_mmd_test_without_strata_has_no_stratified_keys(), test_per_member_scores_flags_the_outlying_member(), test_stratified_null_differs_from_plain(), _three_strata()

### Community 47 - "MaskedFusionNetwork"
Cohesion: 0.15
Nodes (17): Exception, _assert_reached(), _bound_mass(), _OrbitCapExceeded, _plummer_sample(), ndarray, Restricted N-body stellar-stream forward model on AGAMA (CPU, joblib).  Same pri, Present-day bound mass of the progenitor remnant, by the agama example's binding (+9 more)

### Community 48 - "Community 48"
Cohesion: 0.40
Nodes (6): _contam_batch(), _contam_params(), Real augmentation params (the contamination step needs the full StreamResources, The step must be safe to leave in a chain: contamination_max_frac=0 changes noth, test_contamination_replaces_only_attended_members_and_inflates_dispersion(), test_contamination_zero_fraction_is_a_no_op()

### Community 49 - "Community 49"
Cohesion: 0.50
Nodes (4): apply_bayesflow_patches(), _patch_compositional_condition_reshape(), Targeted runtime fixes for known BayesFlow bugs (version-checked, applied once)., Idempotently install the fixes. Called when a compositional workflow is built.

### Community 50 - "Community 50"
Cohesion: 0.50
Nodes (3): Stellar streams via restricted N-body (agama example_tidal_stream method)., Swap the forward-model worker for the restricted-N-body one; the base class's, RestrictedNbodyStreamSimulator

### Community 51 - "Community 51"
Cohesion: 0.50
Nodes (4): arm_metrics(), edge_centre_ratio(), Number density in the outer ``frac`` of the phi1 range (both ends) over the cent, Distributional summary of one arm, in the stream's great-circle frame.

### Community 52 - "Community 52"
Cohesion: 0.50
Nodes (4): _gaussian_widths(), Posterior sd of theta~N(0,1) given m streams (noise s) and one curve (noise c)., test_duplicated_curve_narrows_the_compositional_posterior(), test_overcounting_bias_is_bounded_by_sqrt_m_and_vanishes_without_the_curve()

### Community 54 - "compositional.py"
Cohesion: 0.12
Nodes (18): main(), Per-statistic sim-vs-real comparison of the hand-crafted stream summary statisti, _stat_channels(), apply_augmentations_once(), apply_mask_plan(), build_prior_score(), flatten_members(), group_members() (+10 more)

### Community 56 - "Community 56"
Cohesion: 0.19
Nodes (13): _load_posterior(), main(), log10_keys_from_pipeline(), Keys the ``log10_transform`` preprocessing step reparametrized, if present (else, _evaluate_real_compositional(), _max_particles(), _prepare_real_members(), Real-data mode of the evaluate stage (``data.real_data_path`` set).  Dispatched (+5 more)

### Community 58 - "run_tuning"
Cohesion: 0.11
Nodes (15): _masked_mlp(), _masked_time_series_transformer(), MaskedMLP, MaskedTimeSeriesTransformer, Layer, ndarray, SummaryNetwork, Tensor (+7 more)

### Community 59 - "build_workflow"
Cohesion: 0.22
Nodes (6): Config composition + schema validation smoke tests., The typed schema is the only validation layer now that group YAMLs have no base, test_adapter_derived_from_simulator(), test_adapter_explicit_config_wins(), test_group_override(), test_unknown_key_is_rejected()

### Community 60 - "MaskedFusionNetwork"
Cohesion: 0.18
Nodes (10): 1. The canonical set: 6D stream tracks in stream-aligned coordinates, 2. Width, dispersion and length (second-moment tracks), 3. Action–angle / frequency-space summaries (most directly potential-sensitive), 4. Orbital-pole / great-circle summaries, 5. Density-structure / power-spectrum summaries (mostly for substructure — lower priority for you), 6. Progenitor / global scalars, Key references, Recommended concrete feature block to concatenate with the SetTransformer embedding (+2 more)

### Community 96 - "test_streams.py"
Cohesion: 0.15
Nodes (10): Stage 1b: compositional (grouped) dataset generation.  Like :mod:`simulate`, but, Generate the compositional dataset described by ``cfg`` and return its path., run_multistream_simulation(), Stage 1: dataset generation.  Samples the prior and runs the forward model in ch, Generate the dataset described by ``cfg`` and return its path., run_simulation(), ensure_dir(), Run-directory helpers and the artifact filenames shared between stages. (+2 more)

### Community 97 - "reporting.py"
Cohesion: 0.07
Nodes (30): fix_keras_model(), load_approximator(), Any, What a stage writes into its run directory: model, loss curve, posterior, diagno, Load the best-val-loss weights BayesFlow checkpointed during training.      Make, Write the truth-aware diagnostics listed in ``cfg.eval.diagnostics`` into ``run_, Truth-free diagnostic: one posterior pair plot per observation (used for real da, Best-effort ``report.md`` from the metrics/figures just written; never aborts a (+22 more)

### Community 98 - "Stream project (compositional score modeling)"
Cohesion: 0.06
Nodes (33): Blocks, Configuration, Network groups, No per-block `output_dir`, Notes that actually bite, `run_name` vs `model_dir`, Augmentation (stochastic, per batch), Extending (+25 more)

### Community 100 - "load_approximator"
Cohesion: 0.16
Nodes (18): load_chunk(), load_dataset(), n_rows(), Dataset, Dataset IO. Datasets are ``.npz`` archives where each key maps to an array whose, Number of simulations in a dataset dict (length of its leading axis)., Write a chunk atomically (temp file + rename) so a crash mid-write can never lea, Generate ``n_total`` rows in chunks of ``chunk``, checkpointing each chunk to di (+10 more)

### Community 101 - "Running a full pipeline with the Two Moons simulator"
Cohesion: 0.08
Nodes (33): convert_concentration(), Convert an NFW halo concentration between spherical-overdensity definitions., _m200c_params(), _priors_local_ident(), Tests for the (M200, c_v') halo reparameterization (McMillan 2017; stream_agama., The NFW config fixes gamma/alpha/beta (so they are not inferred) and keeps q/p/t, m200_c dispatch builds a potential whose halo == the rho_a halo with the derived, m200_c: simulate() emits rho/a_..._derived (n,1) equal to the per-row _halo_para (+25 more)

### Community 102 - "_load_clean"
Cohesion: 0.43
Nodes (6): _load_clean(), main(), ndarray, Cross-model posterior tension report (offline analysis helper — not a Hydra run, Load a posterior .npz as {param: (n_datasets, n_samples)} float arrays., _resolve()

### Community 104 - "ppc_prior_predictive.py"
Cohesion: 0.20
Nodes (15): main(), ndarray, Overlay two particle-spray recipes (Fardal+2015 vs Chen+2024) against the real G, (sky RA*cos(dec), Dec) and (pm_ra_cosdec, pm_dec) for the finite particles of st, _stream_xy_pm(), load_dataset(), main(), ndarray (+7 more)

### Community 105 - "corner_parameters.py"
Cohesion: 0.22
Nodes (12): NpzFile, autoscale(), detect_param_keys(), expand_inputs(), load_columns(), main(), ndarray, Corner plot of parameter draws from a simulate dataset (offline helper — not a H (+4 more)

### Community 107 - "test_streams.py"
Cohesion: 0.16
Nodes (21): _build(), Missing-v_los handling: fill modes (mask_vlos / impute_vlos) and the missingness, The real stream_global params (resources from the git-tracked assets/gaia copy),, Default (mean) mode: unmeasured v_los carries the mean of the measured stars, si, Batch shaped like the real path: vlos_mask given, unmeasured v_los pre-filled wi, _real_like_batch(), _star_batch(), _stream_params() (+13 more)

### Community 108 - "2. The four commands (full run)"
Cohesion: 0.29
Nodes (7): quiet_worker(), Silence noisy C-extension output (AGAMA) during simulation.  AGAMA writes diagno, True unless ``HYDRABFLOW_SIM_QUIET`` is set to a falsy value., Redirect fd 1 & 2 to ``/dev/null`` for the duration of the block (C-level output, Decorator: run a (joblib) worker with its C-level stdout/stderr redirected to /d, sim_quiet_enabled(), suppress_c_stdio()

### Community 110 - "_vcirc_worker"
Cohesion: 0.27
Nodes (11): _agama(), extended_rotation_curve(), _host_potential(), _load_posterior(), main(), Posterior-predictive check on the *model rotation curve* only.  Takes a complete, Model circular velocity [km/s] at obs_r; NaN where v^2 < 0., Zhou up to split, Huang beyond; sorted by radius. (r, vc, sigma). (+3 more)

### Community 114 - "ndarray"
Cohesion: 0.10
Nodes (17): fill_stream_grid_from_simulator(), Align the training-time rotation-curve grid with the simulator's ``vcirc_kms`` g, ndarray, Screen each draw's global potential against the observed rotation curve., Draw with ``sample_fn`` until ``n`` rows pass the rotation-curve cut., One shared global draw per dataset, one stream realization per target stream., Keep only the stars inside each row's stream observation window, at most ``max_p, ``obs_r_grid: custom`` -> the (radii, observed v_c, 1-sigma) table given VERBATI (+9 more)

### Community 115 - "stream_agama.py"
Cohesion: 0.13
Nodes (29): halo_m200(), (M200, r200, c200) of this row's halo, so a ``rho`` sweep can be read as a mass, _agama(), _ancillary_observables(), _halo_params(), _halo_params_m200c(), _halo_shape_extras(), _host_potential() (+21 more)

### Community 116 - "stream_common.py"
Cohesion: 0.09
Nodes (34): ndarray, Shared helpers for the stellar-stream simulators (agama, gala, ...).  Ports the, Circular velocity vc(R) [km/s] in the midplane; vc^2 = R dPhi/dR = -R F_R. NaN w, HI terminal velocity v_term(l) (Ibata 2023, Eq. 13), tangent-point method., Local surface density Sigma(z) [Msun/pc^2] from the vertical force (Kuijken & Gi, Vertical stellar-density profile rho(z) [Msun/kpc^3] at R (Ibata et al. 2017b)., Draw ``(n, 1)`` samples from one prior spec (uniform / normal / identity)., Single-stream draw: global parameters, a random stream index ``j``, and that str (+26 more)

### Community 117 - "assets/gaia — portable static inputs for the stream project"
Cohesion: 0.50
Nodes (3): assets/gaia — portable static inputs for the stream project, Contents, Using these on a new cluster

### Community 118 - "StreamObservationStats"
Cohesion: 0.07
Nodes (23): AttachObservedSigmaZ, AttachObservedVcirc, AttachObservedVterm, MaskVcircRadii, PerStreamParameterStandardize, Dataset, ndarray, Stream-specific preprocessing: per-stream normalization and rotation-curve trimm (+15 more)

### Community 120 - "compose"
Cohesion: 0.20
Nodes (17): _agama(), _band_accept(), extended_rotation_curve(), _host_potential(), _load_post(), main(), _menc(), _menc_stack() (+9 more)

### Community 121 - "PerStreamParameterStandardize"
Cohesion: 0.15
Nodes (23): Great-circle frame + projected real tracks for one stream's real members., stream_frame(), main(), cell_stats(), main(), percentile_rank(), Per-bin median and std for the four quantities -> dict[(qty, stat)] = array over, augment_sim() (+15 more)

### Community 122 - "prior_score_from_spec"
Cohesion: 0.67
Nodes (3): GPU_IDS, run_arm(), training_eval_summary_stats.sh script

### Community 123 - "test_config.py"
Cohesion: 0.07
Nodes (27): _deep_set(), _diffusion(), _embed_dim(), _feature_transformer(), _flow_matching(), _mlp(), Any, The shipped network builders. A builder maps a network config to a BayesFlow net (+19 more)

### Community 124 - "get_run_dir"
Cohesion: 0.43
Nodes (7): main(), plot_curve(), ndarray, Extend a stream dataset's rotation-curve observable onto larger radii (offline h, joblib worker: model rotation curve on ``obs_r`` for a chunk of parameter rows., recompute_vcirc(), _vcirc_worker()

### Community 125 - "prior_score_from_kde"
Cohesion: 0.13
Nodes (21): Container, Sampling-time counterpart of the masking in ``compute_metrics``.          ``velo, prior_score_from_kde(), prior_score_from_kde_jax(), prior_score_from_spec(), KDE compositional prior score via ``jax.scipy.stats.gaussian_kde`` + ``jax.grad`, Score of the log prior for compositional sampling, from a prior-spec mapping., KDE compositional prior score, fit in the network's native (un-standardized, log (+13 more)

### Community 129 - "PerStreamParameterStandardize"
Cohesion: 0.12
Nodes (15): Galaxy potential model — components to add (following Ibata et al. 2023, Sec. 5), Goal, HI terminal velocity, How to augment the dataset, Notes / gotchas, Observational data (observed values + reported uncertainties), (Optional) explicit Gaussian likelihood terms, Priors on the new components (+7 more)

### Community 130 - "load_approximator"
Cohesion: 0.33
Nodes (9): _artifact_dir(), _best_for_cutoff(), _cutoffs(), _load_completed(), main(), _overrides(), Milestone best-trial selector for the Ibata gridded-summary tuning study (offlin, Ranking key: (rmse, calibration_error); missing user-attrs fall back to objectiv (+1 more)

### Community 131 - "prior_score_from_spec"
Cohesion: 0.20
Nodes (9): AUG, DATA_DIR, N_TRAIN, N_TRIALS_TOTAL, REAL_AUG, RUNS_DIR, tune_ibata_m200c_median.sh script, SIM (+1 more)

### Community 132 - "MaskVcircRadii"
Cohesion: 0.29
Nodes (6): ADAPTER, MODEL, RUNS_DIR, tune_ibata_settransformer.sh script, STUDY, TUNING

### Community 133 - "compose_cfg"
Cohesion: 0.50
Nodes (3): HYDRABFLOW_NUM_GPUS, HYDRABFLOW_SIM_QUIET, create_ibata_rnbody_m200c_dataset.sh script

### Community 134 - "ppc_ancillary_observables.py"
Cohesion: 0.32
Nodes (7): _band(), _load_2d(), main(), ndarray, Prior-predictive checks for the Ibata (2023) ancillary potential observables (of, Return a stored observable as (n_rows, n_bins), or None if absent., Median + 68/95% percentile band of `rows` (n, len(x)) vs x.

### Community 135 - "_spray_stream"
Cohesion: 0.83
Nodes (3): export_eval_gpu(), run_worker(), tune_ibata_onedisk_grid.sh script

### Community 137 - "test_workflow.py"
Cohesion: 0.50
Nodes (3): DATA_DIR, train_ibata_wdisk_beta3.sh script, SIM

### Community 143 - "4. Adding a SummaryNetwork that isn't shipped"
Cohesion: 0.33
Nodes (9): analyze(), main(), _mean_sim(), median_gamma(), observed_vcirc(), Per-channel misspecification for the POTENTIAL observables (vcirc, vterm, sigma_, Mean RBF kernel similarity of each query row to the reference cloud: (n_q,)., Median-heuristic bandwidth from a subsample of the reference (pooled pairwise sq (+1 more)

### Community 145 - "MaskedFusionNetwork"
Cohesion: 0.16
Nodes (19): _grid_batch(), _masked_tst(), Stream-frame summary statistics: great-circle frame fit, stream-frame projection, A (n, K, 14) stream_summary_grid-shaped batch with every bin well populated., Whatever sits in the statistic channels of an under-populated bin must not reach, The occupancy channels are a validity signal, not an observable: once they have, The property BayesFlow's own nets do NOT have: padding a K-bin grid out with emp, A row whose every bin is under-populated must not produce 0/0. (+11 more)

### Community 147 - "PerStreamParameterStandardize"
Cohesion: 0.12
Nodes (21): _ic_chen_spray(), _ic_particle_spray(), _mass_track(), Jacobi radius, velocity offset, and host->satellite rotation matrices along the, Fardal+2015 initial conditions for particles escaping through the Lagrange point, Chen+2024 initial conditions: one trailing + one leading particle per orbit seed, Progenitor bound mass at each release time, from ``mass_initial`` at ``t = -time, Particle-spray stream including the progenitor's own (moving Plummer) potential. (+13 more)

### Community 148 - "create_ibata_onedisk_beta3_dataset.sh"
Cohesion: 0.50
Nodes (3): HYDRABFLOW_NUM_GPUS, HYDRABFLOW_SIM_QUIET, create_ibata_onedisk_beta3_dataset.sh script

### Community 150 - "train_ibata_m200c.sh"
Cohesion: 0.40
Nodes (4): DATA_DIR, RUNS_DIR, train_ibata_m200c.sh script, SIM

### Community 151 - "tune_ibata_m200c.sh"
Cohesion: 0.40
Nodes (4): DATA_DIR, RUNS_DIR, tune_ibata_m200c.sh script, SIM

### Community 152 - "AttachObservedSigmaZ"
Cohesion: 0.50
Nodes (4): HYDRABFLOW_NUM_GPUS, HYDRABFLOW_SIM_QUIET, report_survival(), create_ibata_rnbody_m200c_v2_dataset.sh script

### Community 153 - "apply_bayesflow_patches"
Cohesion: 0.10
Nodes (21): build_workflow(), Any, Assemble the BayesFlow workflow (adapter + summary network + inference network), Build a ``bf.BasicWorkflow`` from the root ``cfg``., Build the workflow from the root ``cfg``.      ``run_dir`` (passed by train and, build_inference_network(), build_summary_network(), Any (+13 more)

### Community 154 - "create_ibata_m200c_dataset.sh"
Cohesion: 0.50
Nodes (3): HYDRABFLOW_NUM_GPUS, HYDRABFLOW_SIM_QUIET, create_ibata_m200c_dataset.sh script

### Community 156 - "create_ibata_rnbody_m200c_10kstars_test.sh"
Cohesion: 0.50
Nodes (3): HYDRABFLOW_NUM_GPUS, HYDRABFLOW_SIM_QUIET, create_ibata_rnbody_m200c_10kstars_test.sh script

### Community 157 - "PerStreamParameterStandardize"
Cohesion: 0.18
Nodes (12): build_adapter(), Any, Construct ``bf.adapters.Adapter`` from an ``AdapterConfig``., Adapter / network / workflow construction. Skipped if bayesflow isn't installed., Passing run_dir turns on BayesFlow best-weights checkpointing; omitting it leave, Two observable keys -> one backbone per key behind a FusionNetwork, types from p, test_build_adapter(), test_build_adapter_unconfigured_raises() (+4 more)

### Community 159 - "AttachObservedVterm"
Cohesion: 0.33
Nodes (5): limit_gpus(), Pin GPU selection and the Keras backend *before* keras/bayesflow/JAX import anyw, Pin ``CUDA_VISIBLE_DEVICES`` to the least-used GPU(s) before JAX/CUDA initialize, Set ``KERAS_BACKEND`` unless the user already chose one. Returns the active back, set_backend()

### Community 161 - "_objective"
Cohesion: 0.11
Nodes (23): Keep only the dataset keys the adapter consumes (simulators write extra arrays)., select_adapter_keys(), Stage 2: training.  Load dataset -> preprocessing (fit on train, save the state, Train the approximator and return (workflow, history)., run_training(), _objective(), Stage 4: hyperparameter tuning with Optuna.  A multi-objective study (RMSE + cal, Save the fit-once preprocessing state, shared by every trial.      Written atomi (+15 more)

### Community 163 - "create_ibata_rnbody_m200c_nfw_dataset.sh"
Cohesion: 0.33
Nodes (5): CORNER_PARAMS_LIST, DATA_DIR, LABEL, create_ibata_rnbody_m200c_nfw_dataset.sh script, SIM

### Community 166 - "main"
Cohesion: 0.06
Nodes (33): compose_aug(), main(), rank(), run_chain(), _band_pass_worker(), main(), ndarray, Measure the vcirc-rejection acceptance rate of a stream simulator's prior (calib (+25 more)

### Community 172 - "compose_cfg"
Cohesion: 0.50
Nodes (4): cfg(), compose_cfg(), Shared test fixtures., Compose the root config with the structured schemas registered.      ``fill=True

## Knowledge Gaps
- **173 isolated node(s):** `hydrabflow`, `create_ibata_dataset.sh script`, `create_ibata_m200c_dataset.sh script`, `HYDRABFLOW_NUM_GPUS`, `HYDRABFLOW_SIM_QUIET` (+168 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **50 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `get_simulator()` connect `Community 56` to `test_streams.py`, `_objective`, `Augmentation Registry & Tests`, `Simulate Stage & Registries`, `Running a full pipeline with the Two Moons simulator`, `main`, `Community 40`, `Config Composition Tests`, `Dataset IO`, `JAX Backend Pin`, `Logging Helper`, `MaskedFusionNetwork`, `ndarray`, `compositional.py`, `apply_bayesflow_patches`?**
  _High betweenness centrality (0.099) - this node is a cross-community bridge._
- **Why does `AgamaStreamSimulator` connect `Community 11` to `Running a full pipeline with the Two Moons simulator`, `Config Schemas`, `JAX Backend Pin`, `MaskedFusionNetwork`, `Logging Helper`, `ndarray`, `stream_agama.py`, `Community 50`, `stream_common.py`?**
  _High betweenness centrality (0.091) - this node is a cross-community bridge._
- **Why does `compose()` connect `Logging Helper` to `apply_bayesflow_patches`, `Augmentation Registry & Tests`, `Running a full pipeline with the Two Moons simulator`, `main`, `Config Composition Tests`, `test_streams.py`, `compose_cfg`, `test_config.py`, `MaskedFusionNetwork`, `PerStreamParameterStandardize`, `build_workflow`, `PerStreamParameterStandardize`?**
  _High betweenness centrality (0.066) - this node is a cross-community bridge._
- **Are the 7 inferred relationships involving `AgamaStreamSimulator` (e.g. with `_OrbitCapExceeded` and `RestrictedNbodyStreamSimulator`) actually correct?**
  _`AgamaStreamSimulator` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `_jax()` (e.g. with `_stream_summary_grid()` and `_stream_summary_statistics()`) actually correct?**
  _`_jax()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 29 inferred relationships involving `compose()` (e.g. with `load_simulator()` and `compose_aug()`) actually correct?**
  _`compose()` has 29 INFERRED edges - model-reasoned connections that need verification._
- **Are the 23 inferred relationships involving `get_simulator()` (e.g. with `load_simulator()` and `main()`) actually correct?**
  _`get_simulator()` has 23 INFERRED edges - model-reasoned connections that need verification._
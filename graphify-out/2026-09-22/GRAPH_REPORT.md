# Graph Report - HydraBFlow  (2026-09-22)

## Corpus Check
- 180 files · ~510,696 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2034 nodes · 3296 edges · 177 communities (122 shown, 55 thin omitted)
- Extraction: 84% EXTRACTED · 16% INFERRED · 0% AMBIGUOUS · INFERRED: 540 edges (avg confidence: 0.78)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `bd4e03c0`
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
- test_compose_v2_presets
- tune_post_eval.sh
- main
- build_augmentations
- compose_cfg
- main
- train_v5_2modal_oldgrid.sh
- train_v5_2modal_nodisp.sh
- train_v5_2modal_oldgrid_nodisp.sh
- test_streams.py
- reporting.py
- Stream project (compositional score modeling)
- main
- load_approximator
- Running a full pipeline with the Two Moons simulator
- _load_clean
- PreprocessStep
- ppc_prior_predictive.py
- corner_parameters.py
- streamfinder_frame.py
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
- main
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
- _key_cell
- 4. Adding a SummaryNetwork that isn't shipped
- stream_frame
- test_config.py
- StreamResources
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
- RealVlosModel
- AttachObservedVterm
- test_misspecification.py
- _objective
- main
- create_ibata_rnbody_m200c_nfw_dataset.sh
- create_trihedron_dataset.sh
- main
- main
- RestrictedNbodyStreamSimulator
- fiducial_row
- _impute_vlos
- trihedron.py
- _add_noise_to_rho_z
- _measured_vlos_stats
- arm_metrics
- main
- test_compose_v2_presets

## God Nodes (most connected - your core abstractions)
1. `compose()` - 50 edges
2. `AgamaStreamSimulator` - 49 edges
3. `get_simulator()` - 37 edges
4. `_jax()` - 36 edges
5. `PreprocessStep` - 24 edges
6. `main()` - 23 edges
7. `main()` - 22 edges
8. `GroupedDiffusionModel` - 21 edges
9. `augment_sim()` - 19 edges
10. `_sim_key()` - 18 edges

## Surprising Connections (you probably didn't know these)
- `test_build_workflow()` --calls--> `build_workflow()`  [INFERRED]
  tests/test_workflow.py → src/hydrabflow/pipeline/workflow.py
- `test_two_moons_shapes_and_reproducibility()` --calls--> `get_simulator()`  [INFERRED]
  tests/test_augmentation.py → src/hydrabflow/registry.py
- `test_unknown_simulator_errors()` --calls--> `get_simulator()`  [INFERRED]
  tests/test_registries.py → src/hydrabflow/registry.py
- `test_unknown_preprocess_step_errors()` --calls--> `build_pipeline()`  [INFERRED]
  tests/test_registries.py → src/hydrabflow/registry.py
- `test_flattened_density_matches_finite_difference_laplacian()` --calls--> `flattened_nfw_density()`  [INFERRED]
  tests/test_stream_gala.py → src/hydrabflow/simulators/stream_gala.py

## Import Cycles
- None detected.

## Communities (177 total, 55 thin omitted)

### Community 0 - "Preprocessing Pipeline & Steps"
Cohesion: 0.12
Nodes (14): CastDtype, DropNaNSimulations, Log10Transform, _num_rows(), Dataset, Built-in stateless preprocessing steps (besides standardization).  These mirror, Random hold-out split. Steps listed after this one are fit on the train split on, Drop rows (simulations) that contain any NaN/Inf in the listed keys. (+6 more)

### Community 1 - "Eval / Checkpoint Stages"
Cohesion: 0.33
Nodes (6): Model Default Config, Diffusion Inference Network Config, Flow Matching Inference Network Config, DeepSet Summary Network Config, SetTransformer Summary Network Config, TimeSeriesTransformer Summary Network Config

### Community 2 - "Design Principles & Configs"
Cohesion: 0.14
Nodes (18): _band(), main(), POSTERIOR-predictive checks for the Ibata (2023) ancillary potential observables, _build_pot_cfg(), _identity_constants(), _load_posterior(), _log10_keys(), main() (+10 more)

### Community 3 - "Augmentation Registry & Tests"
Cohesion: 0.09
Nodes (38): arm_cache_name(), arm_xy(), corner_matrix(), density_contours(), fiducial_row(), in_window(), load_or_run(), load_simulator() (+30 more)

### Community 4 - "Simulate Stage & Registries"
Cohesion: 0.23
Nodes (15): _batch(), _build_one(), Two Moons simulator + the augmentation reproducibility/stochasticity contract., Build the shipped augmentation through the registry with a seeded generator., The shipped config trains without augmentation: zero scale must leave the batch, Consecutive calls on the *same* built augmentation differ (re-drawn every batch), Same seed + same step list -> identical result through the public builder., test_actually_perturbs() (+7 more)

### Community 5 - "Example Simulators (Skeleton/TwoMoons)"
Cohesion: 0.07
Nodes (32): BaseException, RuntimeError, _cli_overrides(), _evaluate_subprocess(), _objective(), Stage 4: hyperparameter tuning with Optuna.  A multi-objective study (RMSE + cal, This process's Hydra task overrides, minus the output dir (each evaluate gets it, Objectives from the `evaluate` stage on the test set (+ optional real-data evalu (+24 more)

### Community 6 - "Config Schemas"
Cohesion: 0.05
Nodes (25): BaseSimulator, BaseSimulator, ABC, Any, ndarray, Base interface every forward model implements.  A simulator is the only piece a, Draw ``n`` grouped datasets: one shared global draw + one local draw per member., Abstract forward model. Subclass + register via ``@register_simulator``. (+17 more)

### Community 7 - "Network Factory & Adapter"
Cohesion: 0.15
Nodes (23): _capture_score_calls(), _net(), Modality masking + the compositional over-counting it fixes.  The last test is t, A rank-3 plan mask must flatten to the same row order as `conditions`.      `_co, The fix, at the level of what the mask plan actually asserts.      m member item, Record what the stock ``DiffusionModel.score`` sees, and return zeros of the rig, The plan must reach the score call, per item, in the flattened row order.      W, Without an explicit prior score, upstream appends a zero-condition item to get t (+15 more)

### Community 10 - "Config Composition Tests"
Cohesion: 0.07
Nodes (43): field_at(), main(), pct(), One-at-a-time sensitivity of the gala MW22 streams to each global potential para, Acceleration |a| at 3-D positions X (N,3) and v_c at the stars' cylindrical R, f, run_one(), variants(), agama_potential() (+35 more)

### Community 11 - "Community 11"
Cohesion: 0.07
Nodes (14): AgamaStreamSimulator, Stellar streams in a parametrized Milky Way potential, simulated with AGAMA., ``params.store_window_subsample`` -> ``{max_particles, pad_value, windows}`` wit, Split radius for the extended (Zhou u Huang) rotation-curve grid., Host-potential configuration threaded to the joblib workers. Legacy default (all, Requested Ibata ancillary observables (``params.ancillary_observables``); empty, Spec passed to the joblib worker: requested names + their fixed grids (or None)., Dataset keys the requested ancillary observables are stored under (``vterm``->`` (+6 more)

### Community 12 - "Dataset IO"
Cohesion: 0.10
Nodes (34): _estimator_params(), _np_fit_frame(), _np_phi1(), _np_unit_vec(), occupancy_encoding(), Hand-crafted per-stream summary statistics in a data-driven stream-aligned frame, ``(scale, min_count, include_occupancy)`` from the augmentation params., ``summary_occupancy``: how the per-bin occupancy channels are written.      * `` (+26 more)

### Community 13 - "Hydra App Boilerplate"
Cohesion: 0.11
Nodes (11): GroupedDiffusionModel, ndarray, Width of the always-observed leading group, inferred from the condition vector., ``(1, width)`` row that is 1 on the columns of every group named in *observed*., One list of observed group names per compositional item (or ``None`` to disable), Condition groups observed by ordinary ``sample()`` calls (``None`` = all of them, ``(batch_size, width)``, 0 on every column of a dropped group.          The draw, Inject the per-item mask, then defer to the stock compositional score. (+3 more)

### Community 14 - "JAX Backend Pin"
Cohesion: 0.13
Nodes (23): _bf_mmd(), member_summaries(), mmd_test(), _null_mmd(), per_member_scores(), ndarray, Summary-space model misspecification test (observed group vs simulated reference, Mahalanobis OOD score of each observed member vs its own stream's reference clou (+15 more)

### Community 15 - "Logging Helper"
Cohesion: 0.15
Nodes (9): Tests for the Frenet-Serret remap simulator (stream_trihedron).  Covers the exac, A surrogate that ignored its potential would silently produce one stream forever, The template's t_knots are in the (kpc, km/s, Msun) time unit the workers use; w, The remap cannot respond to m/a/t_end, so they must not reach local_parameter_na, The template's one exactness anchor, and the check that its labels belong to the, test_remap_is_exact_at_the_fiducial_potential(), test_remap_moves_the_stream_when_the_potential_changes(), test_stripping_history_is_frozen_not_inferred() (+1 more)

### Community 22 - "Simulators Package Init"
Cohesion: 0.08
Nodes (41): _binstat(), main(), make_figure(), Published (Ibata+2024 Table 3) stream frames, keyed by stream index., Rebuild the stream in its OWN potential and compare to the simulation it came fr, self_test(), stream_frames(), halo_in_m200c() (+33 more)

### Community 23 - "Package/Init cluster 23"
Cohesion: 0.14
Nodes (11): PerStreamParameterStandardize, Dataset, ndarray, Fit per-stream observation stats + log10(vcirc) per-bin stats on the (clean) tra, Integer stream ids broadcastable against ``like``.      ``j``'s leading axes alw, z-score each stream's local parameters with that stream's prior mean/std., _stream_index(), StreamObservationStats (+3 more)

### Community 25 - "Evaluate Entry Script"
Cohesion: 0.20
Nodes (15): draw_halo_rows(), load_samples(), main(), panel_label(), panel_metrics(), Path, Keep ``n`` random stars of a template, as a new Template sharing the same knot g, ``n`` uniform draws of the halo prior, shared by all three streams. (+7 more)

### Community 26 - "Evaluate-Real Entry Script"
Cohesion: 0.18
Nodes (23): _build(), _contam_batch(), _contam_params(), _params(), Stream-frame summary statistics: great-circle frame fit, stream-frame projection, The final feature is the stream index j (so the MLP can distinguish streams)., A stream with zero measured v_los still yields finite track features (only its v, summary_include_std=false drops the per-bin std channels: (n, K, 14) -> (n, K, 9 (+15 more)

### Community 27 - "Simulate Entry Script"
Cohesion: 0.09
Nodes (15): _random_frame(), Tests for the orbit + DeltaTheta(phi1) spline simulator (stream_orbit_offset)., The worker emits observables and returns Galactocentric coordinates, which ``sim, The guard against a surrogate that quietly ignores the parameter it is meant to, The template's T0/knot_dt are in the (kpc, km/s, Msun) time unit the workers use, Ibata fits "a fourth order polynomial"; we fit a degree-4 spline. With no interi, Regression: quantile knots on a clumped stream made the least-squares system nea, The simulator's frame math and the diagnostics' frame math must be the same esti (+7 more)

### Community 28 - "Train Entry Script"
Cohesion: 0.31
Nodes (8): _flat2d(), _group_level_flags(), main(), _per_member_safe(), ndarray, Per-channel model-misspecification MMD test (localize WHERE the misspecification, per_member_scores, but robust to scalar channels (F==1, e.g. sigma_z), where np., Which raw test-set channels are group-level (one value per potential, shared by

### Community 29 - "Tune Entry Script"
Cohesion: 0.22
Nodes (7): gaussian_noise(), jax_noise(), Augmentation, Observational-noise augmentations — and the template for your own.  An augmentat, Add zero-mean Gaussian noise to one observable key.      Params: ``noise_key`` (, Zero the columns of *conditions* that *mask* marks unobserved.          Only ``D, Sampling-time counterpart of the masking in ``compute_metrics``.          ``velo

### Community 31 - "Community 31"
Cohesion: 0.08
Nodes (23): For /graphify add and --watch, For /graphify query, For the commit hook and native CLAUDE.md integration, For --update and --cluster-only, /graphify, Honesty Rules, Interpreter guard for subcommands, Part A - Structural extraction for code files (+15 more)

### Community 32 - "Community 32"
Cohesion: 0.16
Nodes (9): Shape, _fusion(), MaskedFusionNetwork, Layer, SummaryNetwork, Tensor, Multi-observable fusion summary network (attention-mask aware).  Consumes the di, Build a :class:`MaskedFusionNetwork` from ``cfg.params`` (see module docstring). (+1 more)

### Community 33 - "Community 33"
Cohesion: 0.47
Nodes (5): in_window_count(), main(), npz_memmap(), Visual curation check of a FLAT stream training set: what does the network get f, Memmap one member of an UNCOMPRESSED npz (zip 'Stored' entries).

### Community 34 - "Community 34"
Cohesion: 0.18
Nodes (10): Decisions Log, Design principles, docs/, graphify, HydraBFlow: SBI pipeline template (BayesFlow + Hydra), Layout, Notes worth keeping, Stack (+2 more)

### Community 35 - "Community 35"
Cohesion: 0.10
Nodes (31): fill_stream_grid_from_simulator(), Align the training-time rotation-curve grid with the simulator's ``vcirc_kms`` g, get_simulator(), Instantiate the simulator selected by ``cfg.simulator``., compose(), Expose the composer so tests can build configs with custom overrides., The NFW config fixes gamma/alpha/beta (so they are not inferred) and keeps q/p/t, `params.marginalize` keeps the Solar parameters out of the inferred set (and the (+23 more)

### Community 37 - "Community 37"
Cohesion: 0.17
Nodes (20): _binned(), build_offset_template(), dm_from_distance(), _mad_std(), _monotone_segment(), observables_to_cartesian(), orbit_track(), orbit_track_covering() (+12 more)

### Community 38 - "streams.py"
Cohesion: 0.07
Nodes (57): _add_noise_to_rho_z(), _add_noise_to_sigma_z(), _add_noise_to_vcirc(), _add_noise_to_vterm(), _apply_obs_error(), _compact_to_attended(), _concatenate_magnitudes(), _concatenate_sigma_errors() (+49 more)

### Community 39 - "Community 39"
Cohesion: 0.32
Nodes (7): _fusion_group_sizes(), _grouped_diffusion(), Modality-coherent condition masking for the diffusion inference network.  Ported, ``(subnet name, subnet kwargs)`` from the inference-network config.      ``param, Droppable group widths + names from the fusion summary network's backbones., ``DiffusionModel`` with per-modality condition dropout; groups read off the fusi, _subnet_kwargs()

### Community 40 - "Community 40"
Cohesion: 0.10
Nodes (34): Sweep ``compositional_bridge_d1`` on the compositional (pooled) posterior of one, run(), apply_mask_plan(), build_prior_score(), condition_keys(), flatten_members(), group_members(), log10_keys_from_pipeline() (+26 more)

### Community 46 - "Part B — Support a file format other than `.npz`"
Cohesion: 0.20
Nodes (9): HYDRABFLOW_NUM_GPUS, HYDRABFLOW_SIM_QUIET, JAX_PLATFORMS, MKL_NUM_THREADS, NUMEXPR_NUM_THREADS, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, run_stage() (+1 more)

### Community 47 - "MaskedFusionNetwork"
Cohesion: 0.13
Nodes (20): Exception, _assert_reached(), _bound_mass(), _OrbitCapExceeded, _plummer_sample(), ndarray, Restricted N-body stellar-stream forward model on AGAMA (CPU, joblib).  Same pri, Present-day bound mass of the progenitor remnant, by the agama example's binding (+12 more)

### Community 48 - "Community 48"
Cohesion: 0.15
Nodes (17): ``type`` resolves through the summary-network registry (``networks/factory.py``), SummaryNetworkConfig, build_inference_network(), build_summary_network(), Any, Return the summary network selected by ``cfg.type``.      One key (the default):, Return the inference (posterior) network selected by ``cfg.type``.      A builde, test_network_registries_list_available_on_unknown_type() (+9 more)

### Community 49 - "Community 49"
Cohesion: 0.25
Nodes (8): _bin_assignment(), _binned_stats(), Per-star bin index plus an ``in_range`` flag.      ``searchsorted`` returns 0 fo, Per-bin ``(median, dispersion, count)`` over the masked members.      ``vals`` i, The estimator fix. With ddof=0 the population std is biased low by sqrt((N-1)/N), Stars beyond the outermost φ1 edge used to be clipped into the end bins, inflati, test_binned_stats_is_unbiased_and_flags_sparse_bins(), test_out_of_range_stars_are_excluded_not_clipped()

### Community 50 - "Community 50"
Cohesion: 0.21
Nodes (18): contiguous_runs(), fig_icrs_overlay(), fig_orbit_overlay(), fig_orbits(), fig_sky(), fig_stream_frame(), frame_of(), in_window() (+10 more)

### Community 51 - "Community 51"
Cohesion: 0.06
Nodes (50): binned(), fig_marginals(), fig_mmd(), fig_tracks(), mad_std(), main(), _nanmedian(), _nanpercentile() (+42 more)

### Community 52 - "Community 52"
Cohesion: 0.50
Nodes (4): _gaussian_widths(), Posterior sd of theta~N(0,1) given m streams (noise s) and one curve (noise c)., test_duplicated_curve_narrows_the_compositional_posterior(), test_overcounting_bias_is_bounded_by_sqrt_m_and_vanishes_without_the_curve()

### Community 56 - "Community 56"
Cohesion: 0.11
Nodes (23): main(), Stream-channel (sim_summary) misspecification MMD vs a large training-set refere, main(), Per-statistic sim-vs-real comparison of the hand-crafted stream summary statisti, _stat_channels(), apply_augmentations_once(), apply_observed_groups(), Replay the configured augmentation chain once (fixed draw) on flattened rows. (+15 more)

### Community 58 - "run_tuning"
Cohesion: 0.11
Nodes (15): _masked_mlp(), _masked_time_series_transformer(), MaskedMLP, MaskedTimeSeriesTransformer, Layer, ndarray, SummaryNetwork, Tensor (+7 more)

### Community 59 - "build_workflow"
Cohesion: 0.39
Nodes (7): DataFrame, collect(), _load(), main(), _pareto_mask(), Collate an Optuna study's trials into one table and plot its Pareto fronts., _scatter()

### Community 60 - "MaskedFusionNetwork"
Cohesion: 0.18
Nodes (10): 1. The canonical set: 6D stream tracks in stream-aligned coordinates, 2. Width, dispersion and length (second-moment tracks), 3. Action–angle / frequency-space summaries (most directly potential-sensitive), 4. Orbital-pole / great-circle summaries, 5. Density-structure / power-spectrum summaries (mostly for substructure — lower priority for you), 6. Progenitor / global scalars, Key references, Recommended concrete feature block to concatenate with the SetTransformer embedding (+2 more)

### Community 87 - "main"
Cohesion: 0.20
Nodes (9): AUG, DATA_DIR, EXTRA, OCC, REAL, REAL_AUG, RUNS_DIR, train_v5_2modal.sh script (+1 more)

### Community 88 - "build_augmentations"
Cohesion: 0.20
Nodes (9): Keep only the dataset keys the adapter consumes (simulators write extra arrays)., select_adapter_keys(), Stage 2: training.  Load dataset -> preprocessing (fit on train, save the state, Train the approximator and return (workflow, history)., run_training(), build_augmentations(), Augmentation, Build the ordered augmentation list from ``cfg.augmentation``.      Each step ge (+1 more)

### Community 90 - "compose_cfg"
Cohesion: 0.27
Nodes (8): cfg(), compose_cfg(), Shared test fixtures., Compose the root config with the structured schemas registered.      ``fill=True, The single-modality coupling-flow presets compose and build., As the only summary net, BayesFlow calls compute_metrics(stage=...) -- a bare Se, test_imm_presets_compose_and_build(), test_mlp_standalone_is_a_summary_network()

### Community 91 - "main"
Cohesion: 0.16
Nodes (14): Local surface density Sigma(z) [Msun/pc^2] from the vertical force (Kuijken & Gi, Vertical stellar-density profile rho(z) [Msun/kpc^3] at R (Ibata et al. 2017b)., surface_density(), vertical_density_profile(), _base_params(), _mcmillan_potential(), _priors_local(), Sanity checks for the Ibata (2023) ancillary potential observables (new_constrai (+6 more)

### Community 92 - "train_v5_2modal_oldgrid.sh"
Cohesion: 0.29
Nodes (6): EXTRA, MODEL, N_TRAIN, OCC, RUNS_DIR, train_v5_2modal_oldgrid.sh script

### Community 94 - "train_v5_2modal_nodisp.sh"
Cohesion: 0.33
Nodes (5): AUG, MODEL, REAL_AUG, RUNS_DIR, train_v5_2modal_nodisp.sh script

### Community 95 - "train_v5_2modal_oldgrid_nodisp.sh"
Cohesion: 0.33
Nodes (5): AUG, N_TRAIN, REAL_AUG, RUNS_DIR, train_v5_2modal_oldgrid_nodisp.sh script

### Community 96 - "test_streams.py"
Cohesion: 0.15
Nodes (10): Stage 1b: compositional (grouped) dataset generation.  Like :mod:`simulate`, but, Generate the compositional dataset described by ``cfg`` and return its path., run_multistream_simulation(), Stage 1: dataset generation.  Samples the prior and runs the forward model in ch, Generate the dataset described by ``cfg`` and return its path., run_simulation(), ensure_dir(), Run-directory helpers and the artifact filenames shared between stages. (+2 more)

### Community 97 - "reporting.py"
Cohesion: 0.07
Nodes (30): fix_keras_model(), load_approximator(), Any, What a stage writes into its run directory: model, loss curve, posterior, diagno, Load the best-val-loss weights BayesFlow checkpointed during training.      Make, Write the truth-aware diagnostics listed in ``cfg.eval.diagnostics`` into ``run_, Truth-free diagnostic: one posterior pair plot per observation (used for real da, Best-effort ``report.md`` from the metrics/figures just written; never aborts a (+22 more)

### Community 98 - "Stream project (compositional score modeling)"
Cohesion: 0.06
Nodes (34): Blocks, Configuration, Network groups, No per-block `output_dir`, Notes that actually bite, `run_name` vs `model_dir`, Augmentation (stochastic, per batch), Extending (+26 more)

### Community 99 - "main"
Cohesion: 0.18
Nodes (10): Dataset, Estimate any state from ``data`` (train split). Stateless steps leave this empty, Return a transformed copy/view of ``data``., Undo :meth:`transform` where meaningful (e.g. map normalized posterior samples b, Marker base for the train/validation split (handled specially by the pipeline)., Return ``(train, val)``., Fit on the train split and transform train (+ val if a split is present)., Inference path: apply fitted element-wise steps, skipping the split. (+2 more)

### Community 100 - "load_approximator"
Cohesion: 0.16
Nodes (18): load_chunk(), load_dataset(), n_rows(), Dataset, Dataset IO. Datasets are ``.npz`` archives where each key maps to an array whose, Number of simulations in a dataset dict (length of its leading axis)., Write a chunk atomically (temp file + rename) so a crash mid-write can never lea, Generate ``n_total`` rows in chunks of ``chunk``, checkpointing each chunk to di (+10 more)

### Community 101 - "Running a full pipeline with the Two Moons simulator"
Cohesion: 0.08
Nodes (35): _halo_params_m200c(), _halo_shape_extras(), Optional halo shape/sharpness parameters, each defaulting to the historical fixe, Halo ``Spheroid`` params from (virial mass, concentration) instead of (densityNo, convert_concentration(), Convert an NFW halo concentration between spherical-overdensity definitions., _m200c_params(), _priors_local_ident() (+27 more)

### Community 102 - "_load_clean"
Cohesion: 0.43
Nodes (6): _load_clean(), main(), ndarray, Cross-model posterior tension report (offline analysis helper — not a Hydra run, Load a posterior .npz as {param: (n_datasets, n_samples)} float arrays., _resolve()

### Community 103 - "PreprocessStep"
Cohesion: 0.18
Nodes (9): PreprocessPipeline, PreprocessStep, ABC, ndarray, Preprocessing step protocol and the pipeline that orchestrates them.  A :class:`, Element-wise (dataset-in, dataset-out) transform with optional fitted state., Arrays to persist so the fitted transform can be reloaded. Default: nothing., Restore arrays produced by :meth:`state`. (+1 more)

### Community 104 - "ppc_prior_predictive.py"
Cohesion: 0.18
Nodes (17): main(), ndarray, Overlay two particle-spray recipes (Fardal+2015 vs Chen+2024) against the real G, (sky RA*cos(dec), Dec) and (pm_ra_cosdec, pm_dec) for the finite particles of st, _stream_xy_pm(), custom_grid(), load_dataset(), main() (+9 more)

### Community 105 - "corner_parameters.py"
Cohesion: 0.22
Nodes (12): NpzFile, autoscale(), detect_param_keys(), expand_inputs(), load_columns(), main(), ndarray, Corner plot of parameter draws from a simulate dataset (offline helper — not a H (+4 more)

### Community 106 - "streamfinder_frame.py"
Cohesion: 0.20
Nodes (15): centre(), clouds(), edge_centre(), figure(), grid_params(), load_sim(), mad(), main() (+7 more)

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
Cohesion: 0.08
Nodes (24): _ic_chen_spray(), _ic_particle_spray(), ndarray, Screen each draw's global potential against the observed rotation curve., Draw with ``sample_fn`` until ``n`` rows pass the rotation-curve cut., One joblib ``delayed`` call per row. The forward-model seam: subclasses swap the, One shared global draw per dataset, one stream realization per target stream., Jacobi radius, velocity offset, and host->satellite rotation matrices along the (+16 more)

### Community 115 - "stream_agama.py"
Cohesion: 0.09
Nodes (41): halo_m200(), (M200, r200, c200) of this row's halo, so a ``rho`` sweep can be read as a mass, Everything you can register, in one file.  Five extension points share one mecha, check_frozen_locals(), The guard both surrogate simulators need: locals baked into an artifact must sta, Raise unless every frozen local is ``identity``-pinned at its ``fiducial`` value, Forward models. Drop a module here and ``@register_simulator`` makes it selectab, _agama() (+33 more)

### Community 116 - "stream_common.py"
Cohesion: 0.16
Nodes (20): ndarray, Shared helpers for the stellar-stream simulators (agama, gala, ...).  Ports the, Circular velocity vc(R) [km/s] in the midplane; vc^2 = R dPhi/dR = -R F_R. NaN w, HI terminal velocity v_term(l) (Ibata 2023, Eq. 13), tangent-point method., Draw ``(n, 1)`` samples from one prior spec (uniform / normal / identity)., Single-stream draw: global parameters, a random stream index ``j``, and that str, Compositional draw: one global draw shared by *all* streams of each dataset., Project Galactocentric phase-space coordinates to observed ICRS quantities. (+12 more)

### Community 117 - "assets/gaia — portable static inputs for the stream project"
Cohesion: 0.50
Nodes (3): assets/gaia — portable static inputs for the stream project, Contents, Using these on a new cluster

### Community 118 - "StreamObservationStats"
Cohesion: 0.10
Nodes (12): AttachObservedSigmaZ, AttachObservedVcirc, AttachObservedVterm, MaskVcircRadii, Stream-specific preprocessing: per-stream normalization and rotation-curve trimm, Attach the *observed* Milky Way rotation curve to a real dataset that lacks one., Attach the *observed* HI terminal-velocity curve to a real dataset that lacks on, Attach the *observed* local surface density Sigma(1.1 kpc) to a real dataset tha (+4 more)

### Community 119 - "main"
Cohesion: 0.23
Nodes (12): main(), Plot the BINNED network input (``sim_summary`` from ``stream_summary_grid``) for, compose_aug(), main(), rank(), run_chain(), build_real_grid(), build_sim_grid() (+4 more)

### Community 120 - "compose"
Cohesion: 0.20
Nodes (17): _agama(), _band_accept(), extended_rotation_curve(), _host_potential(), _load_post(), main(), _menc(), _menc_stack() (+9 more)

### Community 121 - "PerStreamParameterStandardize"
Cohesion: 0.14
Nodes (27): main(), _load_posterior(), main(), cell_stats(), main(), percentile_rank(), Per-bin median and std for the four quantities -> dict[(qty, stat)] = array over, augment_sim() (+19 more)

### Community 122 - "prior_score_from_spec"
Cohesion: 0.67
Nodes (3): GPU_IDS, run_arm(), training_eval_summary_stats.sh script

### Community 123 - "test_config.py"
Cohesion: 0.07
Nodes (31): _as_summary_network(), _coupling_flow(), _deep_set(), _diffusion(), _embed_dim(), _feature_transformer(), _flow_matching(), _mlp() (+23 more)

### Community 124 - "get_run_dir"
Cohesion: 0.29
Nodes (9): main(), plot_curve(), ndarray, Extend a stream dataset's rotation-curve observable onto larger radii (offline h, joblib worker: model rotation curve on ``obs_r`` for a chunk of parameter rows., recompute_vcirc(), _vcirc_worker(), main() (+1 more)

### Community 125 - "prior_score_from_kde"
Cohesion: 0.08
Nodes (27): Container, prior_score_from_kde(), prior_score_from_kde_jax(), prior_score_from_spec(), KDE compositional prior score via ``jax.scipy.stats.gaussian_kde`` + ``jax.grad`, Score of the log prior for compositional sampling, from a prior-spec mapping., KDE compositional prior score, fit in the network's native (un-standardized, log, Stream-project components: config composition, hierarchy derivation, per-stream (+19 more)

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

### Community 142 - "_key_cell"
Cohesion: 0.18
Nodes (15): _grid_batch(), _masked_tst(), A (n, K, 14) stream_summary_grid-shaped batch with every bin well populated., Whatever sits in the statistic channels of an under-populated bin must not reach, The occupancy channels are a validity signal, not an observable: once they have, The 2026-09-22 bug: BayesFlow standardizes ``summary_variables`` BEFORE the summ, The property BayesFlow's own nets do NOT have: padding a K-bin grid out with emp, A row whose every bin is under-populated must not produce 0/0. (+7 more)

### Community 143 - "4. Adding a SummaryNetwork that isn't shipped"
Cohesion: 0.33
Nodes (9): analyze(), main(), _mean_sim(), median_gamma(), observed_vcirc(), Per-channel misspecification for the POTENTIAL observables (vcirc, vterm, sigma_, Mean RBF kernel similarity of each query row to the reference cloud: (n_q,)., Median-heuristic bandwidth from a subsample of the reference (pooled pairwise sq (+1 more)

### Community 144 - "stream_frame"
Cohesion: 0.22
Nodes (12): main(), Observation-space corner plots: training-set stars (through the training observa, main(), phi2_grid_figure(), project_sample(), Great-circle frame + projected real tracks for one stream's real members., One grid figure, one axis per sample, phi2 vs phi1 with the real members in grey, robust_lim() (+4 more)

### Community 145 - "test_config.py"
Cohesion: 0.22
Nodes (6): Config composition + schema validation smoke tests., The typed schema is the only validation layer now that group YAMLs have no base, test_adapter_derived_from_simulator(), test_adapter_explicit_config_wins(), test_group_override(), test_unknown_key_is_rejected()

### Community 146 - "StreamResources"
Cohesion: 0.26
Nodes (12): deproject(), project(), ndarray, Great-circle stream frames: (ra, dec, pm) <-> (phi1, phi2, mu_phi1, mu_phi2).  A, Unit vectors ``(..., 3)`` from spherical angles in degrees., The orthonormal tangent pair ``(e_a, e_d)`` at angles given in RADIANS., ICRS -> stream frame. Returns ``(phi1, phi2, mu_phi1, mu_phi2)``, degrees and ma, Stream frame -> ICRS, the exact inverse of :func:`project`.      ``ra`` is wrapp (+4 more)

### Community 147 - "PerStreamParameterStandardize"
Cohesion: 0.21
Nodes (13): _mass_track(), Progenitor bound mass at each release time, from ``mass_initial`` at ``t = -time, _pot(), Progenitor mass loss in the particle-spray forward model., ``mass_final=None`` must leave the original fixed-mass spray untouched., Shedding mass shrinks the Jacobi radius and the escape speed, so the tails are t, _stream(), test_mass_loss_law_changes_the_realization() (+5 more)

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
Cohesion: 0.27
Nodes (6): discover(), T, A named collection filled by ``@registry.add("name")`` decorators.      ``packag, Import ``self.package``'s modules so their decorators have run. Idempotent., Import every non-underscore module in a package, so its decorators run., Registry

### Community 154 - "create_ibata_m200c_dataset.sh"
Cohesion: 0.50
Nodes (3): HYDRABFLOW_NUM_GPUS, HYDRABFLOW_SIM_QUIET, create_ibata_m200c_dataset.sh script

### Community 156 - "create_ibata_rnbody_m200c_10kstars_test.sh"
Cohesion: 0.50
Nodes (3): HYDRABFLOW_NUM_GPUS, HYDRABFLOW_SIM_QUIET, create_ibata_rnbody_m200c_10kstars_test.sh script

### Community 157 - "PerStreamParameterStandardize"
Cohesion: 0.18
Nodes (12): build_adapter(), Any, Construct ``bf.adapters.Adapter`` from an ``AdapterConfig``., Adapter / network / workflow construction. Skipped if bayesflow isn't installed., Passing run_dir turns on BayesFlow best-weights checkpointing; omitting it leave, Two observable keys -> one backbone per key behind a FusionNetwork, types from p, test_build_adapter(), test_build_adapter_unconfigured_raises() (+4 more)

### Community 158 - "RealVlosModel"
Cohesion: 0.18
Nodes (8): Streams remapped from a stored trihedron template (see the module docstring)., The template's star count, NOT ``params.n_particles``: the remap replays exactly, Every frozen local must be pinned at the value the template was built with., Swap the forward-model worker for the remap; the base class's ``simulate`` keeps, The frozen locals (``m_progenitor``/``a_progenitor``/``t_end``) the template was, template_fiducial_locals(), TrihedronStreamSimulator, test_registry_resolves_stream_trihedron()

### Community 159 - "AttachObservedVterm"
Cohesion: 0.33
Nodes (5): limit_gpus(), Pin GPU selection and the Keras backend *before* keras/bayesflow/JAX import anyw, Pin ``CUDA_VISIBLE_DEVICES`` to the least-used GPU(s) before JAX/CUDA initialize, Set ``KERAS_BACKEND`` unless the user already chose one. Returns the active back, set_backend()

### Community 160 - "test_misspecification.py"
Cohesion: 0.20
Nodes (9): apply_bayesflow_patches(), _patch_compositional_condition_reshape(), Targeted runtime fixes for known BayesFlow bugs (version-checked, applied once)., Idempotently install the fixes. Called when a compositional workflow is built., build_workflow(), Any, Assemble the BayesFlow workflow (adapter + summary network + inference network), Build a ``bf.BasicWorkflow`` from the root ``cfg``. (+1 more)

### Community 161 - "_objective"
Cohesion: 0.16
Nodes (14): adapter_keys(), check_masked_backbone_occupancy(), composition_level(), fill_adapter_from_simulator(), _lists(), Build the BayesFlow ``Adapter``: dataset keys -> the roles BayesFlow expects.  `, Every dataset key the adapter consumes.      ``drop`` is included so dropped key, The four adapter key lists as plain lists (resolving interpolations). (+6 more)

### Community 162 - "main"
Cohesion: 0.20
Nodes (9): distance_from_dm(), OffsetTemplate, OrbitTrack, The progenitor orbit in observation space, restricted to one monotone branch of, ``(n, 6)`` orbit observables at the requested phi1; NaN outside the branch., Everything needed to paint one stream onto an arbitrary progenitor orbit., ``(n, 6)`` ICRS observables (ra, dec, distance, mu_ra_cosdec, mu_dec, v_los)., Heliocentric distance in kpc from distance modulus. Positive by construction, wh (+1 more)

### Community 163 - "create_ibata_rnbody_m200c_nfw_dataset.sh"
Cohesion: 0.33
Nodes (5): CORNER_PARAMS_LIST, DATA_DIR, LABEL, create_ibata_rnbody_m200c_nfw_dataset.sh script, SIM

### Community 164 - "create_trihedron_dataset.sh"
Cohesion: 0.33
Nodes (5): HYDRABFLOW_NUM_GPUS, JAX_PLATFORMS, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, create_trihedron_dataset.sh script

### Community 165 - "main"
Cohesion: 0.40
Nodes (5): _band_pass_worker(), main(), ndarray, Measure the vcirc-rejection acceptance rate of a stream simulator's prior (calib, (n_rows, n_bands) bool: does each row's model curve pass each band? (one vc eval

### Community 166 - "main"
Cohesion: 0.10
Nodes (19): AdapterConfig, AugmentationConfig, CompositionConfig, DataConfig, EvalConfig, InferenceNetworkConfig, ModelConfig, PreprocessingConfig (+11 more)

### Community 167 - "RestrictedNbodyStreamSimulator"
Cohesion: 0.20
Nodes (8): main(), Prior-predictive check of the ROTATION-CURVE observable against the observed cur, Register the root schema so ``conf/config.yaml`` validates against it., register_configs(), RootConfig, make_cli(), Turn a ``run(cfg)`` function into the stage's Hydra console entry point., sim()

### Community 168 - "fiducial_row"
Cohesion: 0.25
Nodes (6): OrbitOffsetStreamSimulator, Streams painted onto the progenitor orbit from stored spline corrections., Swap in the orbit+offset worker; the base class keeps the dispatch and the whole, The stripping history each stream's template was built at., template_fiducial_locals(), test_registry_resolves_stream_orbit_offset()

### Community 169 - "_impute_vlos"
Cohesion: 0.25
Nodes (7): Registry resolution: unknown names fail loudly, custom builders plug in., Every extension point is a Registry filled on discovery (registry.py)., Adding an experimental architecture = one decorated function, no infrastructure, test_custom_network_builder_registers(), test_shipped_components_are_registered(), test_unknown_preprocess_step_errors(), test_unknown_simulator_errors()

### Community 171 - "_add_noise_to_rho_z"
Cohesion: 0.29
Nodes (7): BSpline, eval_spline(), fit_offset_spline(), _knot_intervals_occupied(), Least-squares degree-4 spline of ``y`` on ``x``, with knots spanning ``[x_lo, x_, Schoenberg-Whitney in the form that matters here: no empty interior knot span., Evaluate a fitted spline, clipping ``x`` into its knot span.      Degree-4 extra

### Community 172 - "_measured_vlos_stats"
Cohesion: 0.33
Nodes (5): HYDRABFLOW_NUM_GPUS, JAX_PLATFORMS, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, create_orbit_offset_dataset.sh script

### Community 174 - "arm_metrics"
Cohesion: 0.50
Nodes (4): arm_metrics(), edge_centre_ratio(), Number density in the outer ``frac`` of the phi1 range (both ends) over the cent, Distributional summary of one arm, in the stream's great-circle frame.

## Knowledge Gaps
- **221 isolated node(s):** `hydrabflow`, `HYDRABFLOW_NUM_GPUS`, `HYDRABFLOW_SIM_QUIET`, `JAX_PLATFORMS`, `OMP_NUM_THREADS` (+216 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **55 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `AgamaStreamSimulator` connect `Community 11` to `Community 35`, `Running a full pipeline with the Two Moons simulator`, `Config Schemas`, `fiducial_row`, `Config Composition Tests`, `MaskedFusionNetwork`, `ndarray`, `stream_agama.py`, `Community 56`, `main`, `RealVlosModel`?**
  _High betweenness centrality (0.099) - this node is a cross-community bridge._
- **Why does `get_simulator()` connect `Community 35` to `test_streams.py`, `_objective`, `Augmentation Registry & Tests`, `Simulate Stage & Registries`, `main`, `RestrictedNbodyStreamSimulator`, `Community 40`, `_impute_vlos`, `streamfinder_frame.py`, `JAX Backend Pin`, `main`, `Community 48`, `stream_agama.py`, `Community 56`, `PerStreamParameterStandardize`, `Train Entry Script`?**
  _High betweenness centrality (0.078) - this node is a cross-community bridge._
- **Why does `compose()` connect `Community 35` to `Augmentation Registry & Tests`, `Config Composition Tests`, `Dataset IO`, `test_config.py`, `PerStreamParameterStandardize`, `_objective`, `main`, `RestrictedNbodyStreamSimulator`, `Community 48`, `test_compose_v2_presets`, `Community 50`, `Community 51`, `compose_cfg`, `streamfinder_frame.py`, `test_streams.py`, `main`, `PerStreamParameterStandardize`, `test_config.py`, `prior_score_from_kde`?**
  _High betweenness centrality (0.058) - this node is a cross-community bridge._
- **Are the 47 inferred relationships involving `compose()` (e.g. with `load_simulator()` and `main()`) actually correct?**
  _`compose()` has 47 INFERRED edges - model-reasoned connections that need verification._
- **Are the 10 inferred relationships involving `AgamaStreamSimulator` (e.g. with `_OrbitCapExceeded` and `RestrictedNbodyStreamSimulator`) actually correct?**
  _`AgamaStreamSimulator` has 10 INFERRED edges - model-reasoned connections that need verification._
- **Are the 34 inferred relationships involving `get_simulator()` (e.g. with `load_simulator()` and `main()`) actually correct?**
  _`get_simulator()` has 34 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `_jax()` (e.g. with `_stream_summary_grid()` and `_stream_summary_statistics()`) actually correct?**
  _`_jax()` has 2 INFERRED edges - model-reasoned connections that need verification._
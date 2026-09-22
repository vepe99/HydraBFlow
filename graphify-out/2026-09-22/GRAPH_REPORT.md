# Graph Report - HydraBFlow  (2026-09-20)

## Corpus Check
- 150 files · ~466,845 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1678 nodes · 2641 edges · 148 communities (94 shown, 54 thin omitted)
- Extraction: 83% EXTRACTED · 17% INFERRED · 0% AMBIGUOUS · INFERRED: 454 edges (avg confidence: 0.78)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `ae4ad487`
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

## God Nodes (most connected - your core abstractions)
1. `AgamaStreamSimulator` - 45 edges
2. `compose()` - 39 edges
3. `_jax()` - 32 edges
4. `get_simulator()` - 32 edges
5. `main()` - 24 edges
6. `PreprocessStep` - 24 edges
7. `main()` - 23 edges
8. `GroupedDiffusionModel` - 21 edges
9. `_sim_key()` - 17 edges
10. `_evaluate_compositional_global()` - 17 edges

## Surprising Connections (you probably didn't know these)
- `test_two_moons_shapes_and_reproducibility()` --calls--> `get_simulator()`  [INFERRED]
  tests/test_augmentation.py → src/hydrabflow/registry.py
- `test_unknown_simulator_errors()` --calls--> `get_simulator()`  [INFERRED]
  tests/test_registries.py → src/hydrabflow/registry.py
- `load_simulator()` --calls--> `register_configs()`  [INFERRED]
  scripts/compare_trihedron_vs_spray.py → src/hydrabflow/config.py
- `progenitor_state()` --calls--> `_solar_frame()`  [INFERRED]
  scripts/compare_trihedron_vs_spray.py → src/hydrabflow/simulators/stream_agama.py
- `halo_m200()` --calls--> `_halo_params_m200c()`  [INFERRED]
  scripts/compare_trihedron_vs_spray.py → src/hydrabflow/simulators/stream_agama.py

## Import Cycles
- None detected.

## Communities (148 total, 54 thin omitted)

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
Nodes (36): arm_cache_name(), arm_xy(), corner_matrix(), density_contours(), fiducial_row(), in_window(), load_or_run(), main() (+28 more)

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
Cohesion: 0.10
Nodes (28): _build_potential(), c_phi_from_q_rho(), flattened_nfw_density(), GalaStreamSimulator, halo_rho_negative_radius(), nfw_m_rs_from_m200_c(), _prog_w0(), ndarray (+20 more)

### Community 11 - "Community 11"
Cohesion: 0.07
Nodes (14): AgamaStreamSimulator, Stellar streams in a parametrized Milky Way potential, simulated with AGAMA., ``params.store_window_subsample`` -> ``{max_particles, pad_value, windows}`` wit, Split radius for the extended (Zhou u Huang) rotation-curve grid., Host-potential configuration threaded to the joblib workers. Legacy default (all, Requested Ibata ancillary observables (``params.ancillary_observables``); empty, Spec passed to the joblib worker: requested names + their fixed grids (or None)., Dataset keys the requested ancillary observables are stored under (``vterm``->`` (+6 more)

### Community 13 - "Hydra App Boilerplate"
Cohesion: 0.11
Nodes (11): GroupedDiffusionModel, ndarray, Width of the always-observed leading group, inferred from the condition vector., ``(1, width)`` row that is 1 on the columns of every group named in *observed*., One list of observed group names per compositional item (or ``None`` to disable), Condition groups observed by ordinary ``sample()`` calls (``None`` = all of them, ``(batch_size, width)``, 0 on every column of a dropped group.          The draw, Inject the per-item mask, then defer to the stock compositional score. (+3 more)

### Community 14 - "JAX Backend Pin"
Cohesion: 0.13
Nodes (23): _bf_mmd(), member_summaries(), mmd_test(), _null_mmd(), per_member_scores(), ndarray, Summary-space model misspecification test (observed group vs simulated reference, Mahalanobis OOD score of each observed member vs its own stream's reference clou (+15 more)

### Community 15 - "Logging Helper"
Cohesion: 0.17
Nodes (9): Stream-project components: config composition, hierarchy derivation, per-stream, The simulator-side storage window must be the augmentation's observation window., A noisy rotation-curve bin that went negative must not become NaN (it killed a 1, test_adapter_derivation_follows_composition_level(), test_log10_vcirc_floors_negative_noisy_bins(), test_stream_config_composes(), test_stream_global_log10_and_nolos_presets_compose(), test_stream_noerr_and_nolos_variants_compose() (+1 more)

### Community 23 - "Package/Init cluster 23"
Cohesion: 0.16
Nodes (20): auto_window(), build_template(), _hermite(), nearest_time(), orbit_window(), ndarray, Frenet-Serret stream remapping (Palau & Miralda-Escude 2023, MNRAS 524, 2124, Ap, Cubic Hermite interpolation of the orbit at arbitrary times ``t``.      Position (+12 more)

### Community 25 - "Evaluate Entry Script"
Cohesion: 0.20
Nodes (15): draw_halo_rows(), load_samples(), main(), panel_label(), panel_metrics(), Path, Keep ``n`` random stars of a template, as a new Template sharing the same knot g, ``n`` uniform draws of the halo prior, shared by all three streams. (+7 more)

### Community 26 - "Evaluate-Real Entry Script"
Cohesion: 0.08
Nodes (47): ``type`` resolves through the summary-network registry (``networks/factory.py``), SummaryNetworkConfig, _build(), _contam_batch(), _contam_params(), _grid_batch(), _masked_mlp(), _masked_tst() (+39 more)

### Community 27 - "Simulate Entry Script"
Cohesion: 0.39
Nodes (7): main(), phi2_grid_figure(), project_sample(), Great-circle frame + projected real tracks for one stream's real members., One grid figure, one axis per sample, phi2 vs phi1 with the real members in grey, robust_lim(), stream_frame()

### Community 28 - "Train Entry Script"
Cohesion: 0.31
Nodes (8): _flat2d(), _group_level_flags(), main(), _per_member_safe(), ndarray, Per-channel model-misspecification MMD test (localize WHERE the misspecification, per_member_scores, but robust to scalar channels (F==1, e.g. sigma_z), where np., Which raw test-set channels are group-level (one value per potential, shared by

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
Cohesion: 0.47
Nodes (5): in_window_count(), main(), npz_memmap(), Visual curation check of a FLAT stream training set: what does the network get f, Memmap one member of an UNCOMPRESSED npz (zip 'Stored' entries).

### Community 34 - "Community 34"
Cohesion: 0.18
Nodes (10): Decisions Log, Design principles, docs/, graphify, HydraBFlow: SBI pipeline template (BayesFlow + Hydra), Layout, Notes worth keeping, Stack (+2 more)

### Community 35 - "Community 35"
Cohesion: 0.13
Nodes (24): load_simulator(), Instantiate the simulator named by a config, purely to read its resolved params., fill_stream_grid_from_simulator(), Align the training-time rotation-curve grid with the simulator's ``vcirc_kms`` g, get_simulator(), Instantiate the simulator selected by ``cfg.simulator``., compose(), Expose the composer so tests can build configs with custom overrides. (+16 more)

### Community 37 - "Community 37"
Cohesion: 0.26
Nodes (11): compose_aug(), features(), main(), mmd2(), Prior-predictive coverage of the real Gaia streams in the RAW-PARTICLE represent, (N, 6) ICRS table -> (N, 6) [phi1, phi2, parallax, mu_phi1, mu_phi2, v_los]., Unbiased RBF MMD^2 estimator., Rows with every selected observable measured (v_los only where the mask says mea (+3 more)

### Community 38 - "streams.py"
Cohesion: 0.05
Nodes (72): _bin_assignment(), _binned_stats(), _estimator_params(), _np_fit_frame(), _np_phi1(), _np_unit_vec(), Hand-crafted per-stream summary statistics in a data-driven stream-aligned frame, Per-star bin index plus an ``in_range`` flag.      ``searchsorted`` returns 0 fo (+64 more)

### Community 39 - "Community 39"
Cohesion: 0.32
Nodes (7): _fusion_group_sizes(), _grouped_diffusion(), Modality-coherent condition masking for the diffusion inference network.  Ported, ``(subnet name, subnet kwargs)`` from the inference-network config.      ``param, Droppable group widths + names from the fusion summary network's backbones., ``DiffusionModel`` with per-modality condition dropout; groups read off the fusi, _subnet_kwargs()

### Community 40 - "Community 40"
Cohesion: 0.10
Nodes (33): run(), apply_mask_plan(), build_prior_score(), condition_keys(), flatten_members(), group_members(), log10_keys_from_pipeline(), ndarray (+25 more)

### Community 46 - "Part B — Support a file format other than `.npz`"
Cohesion: 0.20
Nodes (9): HYDRABFLOW_NUM_GPUS, HYDRABFLOW_SIM_QUIET, JAX_PLATFORMS, MKL_NUM_THREADS, NUMEXPR_NUM_THREADS, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, run_stage() (+1 more)

### Community 47 - "MaskedFusionNetwork"
Cohesion: 0.11
Nodes (22): Exception, _assert_reached(), _bound_mass(), _OrbitCapExceeded, _plummer_sample(), ndarray, Restricted N-body stellar-stream forward model on AGAMA (CPU, joblib).  Same pri, Present-day bound mass of the progenitor remnant, by the agama example's binding (+14 more)

### Community 48 - "Community 48"
Cohesion: 0.08
Nodes (29): Sweep ``compositional_bridge_d1`` on the compositional (pooled) posterior of one, Stage 2: training.  Load dataset -> preprocessing (fit on train, save the state, Train the approximator and return (workflow, history)., run_training(), build_augmentations(), build_inference_network(), build_pipeline(), build_summary_network() (+21 more)

### Community 49 - "Community 49"
Cohesion: 0.50
Nodes (4): apply_bayesflow_patches(), _patch_compositional_condition_reshape(), Targeted runtime fixes for known BayesFlow bugs (version-checked, applied once)., Idempotently install the fixes. Called when a compositional workflow is built.

### Community 50 - "Community 50"
Cohesion: 0.22
Nodes (6): Config composition + schema validation smoke tests., The typed schema is the only validation layer now that group YAMLs have no base, test_adapter_derived_from_simulator(), test_adapter_explicit_config_wins(), test_group_override(), test_unknown_key_is_rejected()

### Community 51 - "Community 51"
Cohesion: 0.50
Nodes (4): arm_metrics(), edge_centre_ratio(), Number density in the outer ``frac`` of the phi1 range (both ends) over the cent, Distributional summary of one arm, in the stream's great-circle frame.

### Community 52 - "Community 52"
Cohesion: 0.50
Nodes (4): _gaussian_widths(), Posterior sd of theta~N(0,1) given m streams (noise s) and one curve (noise c)., test_duplicated_curve_narrows_the_compositional_posterior(), test_overcounting_bias_is_bounded_by_sqrt_m_and_vanishes_without_the_curve()

### Community 56 - "Community 56"
Cohesion: 0.14
Nodes (17): main(), Stream-channel (sim_summary) misspecification MMD vs a large training-set refere, main(), Per-statistic sim-vs-real comparison of the hand-crafted stream summary statisti, _stat_channels(), apply_augmentations_once(), apply_observed_groups(), Replay the configured augmentation chain once (fixed draw) on flattened rows. (+9 more)

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
Cohesion: 0.40
Nodes (5): _band_pass_worker(), main(), ndarray, Measure the vcirc-rejection acceptance rate of a stream simulator's prior (calib, (n_rows, n_bands) bool: does each row's model curve pass each band? (one vc eval

### Community 96 - "test_streams.py"
Cohesion: 0.15
Nodes (10): Stage 1b: compositional (grouped) dataset generation.  Like :mod:`simulate`, but, Generate the compositional dataset described by ``cfg`` and return its path., run_multistream_simulation(), Stage 1: dataset generation.  Samples the prior and runs the forward model in ch, Generate the dataset described by ``cfg`` and return its path., run_simulation(), ensure_dir(), Run-directory helpers and the artifact filenames shared between stages. (+2 more)

### Community 97 - "reporting.py"
Cohesion: 0.07
Nodes (30): fix_keras_model(), load_approximator(), Any, What a stage writes into its run directory: model, loss curve, posterior, diagno, Load the best-val-loss weights BayesFlow checkpointed during training.      Make, Write the truth-aware diagnostics listed in ``cfg.eval.diagnostics`` into ``run_, Truth-free diagnostic: one posterior pair plot per observation (used for real da, Best-effort ``report.md`` from the metrics/figures just written; never aborts a (+22 more)

### Community 98 - "Stream project (compositional score modeling)"
Cohesion: 0.06
Nodes (34): Blocks, Configuration, Network groups, No per-block `output_dir`, Notes that actually bite, `run_name` vs `model_dir`, Augmentation (stochastic, per batch), Extending (+26 more)

### Community 100 - "load_approximator"
Cohesion: 0.16
Nodes (18): load_chunk(), load_dataset(), n_rows(), Dataset, Dataset IO. Datasets are ``.npz`` archives where each key maps to an array whose, Number of simulations in a dataset dict (length of its leading axis)., Write a chunk atomically (temp file + rename) so a crash mid-write can never lea, Generate ``n_total`` rows in chunks of ``chunk``, checkpointing each chunk to di (+10 more)

### Community 101 - "Running a full pipeline with the Two Moons simulator"
Cohesion: 0.09
Nodes (31): _halo_params_m200c(), Halo ``Spheroid`` params from (virial mass, concentration) instead of (densityNo, convert_concentration(), Convert an NFW halo concentration between spherical-overdensity definitions., _m200c_params(), _priors_local_ident(), Tests for the (M200, c_v') halo reparameterization (McMillan 2017; stream_agama., m200_c dispatch builds a potential whose halo == the rho_a halo with the derived (+23 more)

### Community 102 - "_load_clean"
Cohesion: 0.43
Nodes (6): _load_clean(), main(), ndarray, Cross-model posterior tension report (offline analysis helper — not a Hydra run, Load a posterior .npz as {param: (n_datasets, n_samples)} float arrays., _resolve()

### Community 104 - "ppc_prior_predictive.py"
Cohesion: 0.18
Nodes (17): main(), ndarray, Overlay two particle-spray recipes (Fardal+2015 vs Chen+2024) against the real G, (sky RA*cos(dec), Dec) and (pm_ra_cosdec, pm_dec) for the finite particles of st, _stream_xy_pm(), custom_grid(), load_dataset(), main() (+9 more)

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
Cohesion: 0.09
Nodes (20): ndarray, Screen each draw's global potential against the observed rotation curve., Draw with ``sample_fn`` until ``n`` rows pass the rotation-curve cut., One joblib ``delayed`` call per row. The forward-model seam: subclasses swap the, One shared global draw per dataset, one stream realization per target stream., Keep only the stars inside each row's stream observation window, at most ``max_p, Model circular velocity [km/s] at the observed radii; NaN where v^2 < 0., joblib worker: does each parameter row's model rotation curve pass the rejection (+12 more)

### Community 115 - "stream_agama.py"
Cohesion: 0.10
Nodes (31): halo_m200(), (M200, r200, c200) of this row's halo, so a ``rho`` sweep can be read as a mass, _agama(), _ancillary_observables(), _halo_params(), _halo_shape_extras(), _host_potential(), _ic_chen_spray() (+23 more)

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
Nodes (23): main(), _load_posterior(), main(), cell_stats(), main(), percentile_rank(), Per-bin median and std for the four quantities -> dict[(qty, stat)] = array over, augment_sim() (+15 more)

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
Cohesion: 0.13
Nodes (17): build_adapter(), Any, Construct ``bf.adapters.Adapter`` from an ``AdapterConfig``., build_workflow(), Any, Assemble the BayesFlow workflow (adapter + summary network + inference network), Build a ``bf.BasicWorkflow`` from the root ``cfg``., Build the workflow from the root ``cfg``.      ``run_dir`` (passed by train and (+9 more)

### Community 159 - "AttachObservedVterm"
Cohesion: 0.33
Nodes (5): limit_gpus(), Pin GPU selection and the Keras backend *before* keras/bayesflow/JAX import anyw, Pin ``CUDA_VISIBLE_DEVICES`` to the least-used GPU(s) before JAX/CUDA initialize, Set ``KERAS_BACKEND`` unless the user already chose one. Returns the active back, set_backend()

### Community 161 - "_objective"
Cohesion: 0.21
Nodes (12): adapter_keys(), composition_level(), fill_adapter_from_simulator(), _lists(), Build the BayesFlow ``Adapter``: dataset keys -> the roles BayesFlow expects.  `, Every dataset key the adapter consumes.      ``drop`` is included so dropped key, Keep only the dataset keys the adapter consumes (simulators write extra arrays)., The four adapter key lists as plain lists (resolving interpolations). (+4 more)

### Community 163 - "create_ibata_rnbody_m200c_nfw_dataset.sh"
Cohesion: 0.33
Nodes (5): CORNER_PARAMS_LIST, DATA_DIR, LABEL, create_ibata_rnbody_m200c_nfw_dataset.sh script, SIM

### Community 166 - "main"
Cohesion: 0.05
Nodes (38): main(), Plot the BINNED network input (``sim_summary`` from ``stream_summary_grid``) for, compose_aug(), main(), rank(), run_chain(), AdapterConfig, AugmentationConfig (+30 more)

## Knowledge Gaps
- **186 isolated node(s):** `hydrabflow`, `HYDRABFLOW_NUM_GPUS`, `HYDRABFLOW_SIM_QUIET`, `JAX_PLATFORMS`, `OMP_NUM_THREADS` (+181 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **54 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `get_simulator()` connect `Community 35` to `test_streams.py`, `_objective`, `Simulate Stage & Registries`, `Community 40`, `Dataset IO`, `JAX Backend Pin`, `Logging Helper`, `Community 48`, `main`, `Community 56`, `PerStreamParameterStandardize`, `Train Entry Script`?**
  _High betweenness centrality (0.133) - this node is a cross-community bridge._
- **Why does `compose()` connect `Community 35` to `_objective`, `Community 37`, `main`, `test_streams.py`, `Logging Helper`, `Community 50`, `main`, `PerStreamParameterStandardize`, `Evaluate-Real Entry Script`, `test_config.py`, `PerStreamParameterStandardize`?**
  _High betweenness centrality (0.077) - this node is a cross-community bridge._
- **Why does `AgamaStreamSimulator` connect `Community 11` to `Community 35`, `Running a full pipeline with the Two Moons simulator`, `Config Schemas`, `Config Composition Tests`, `MaskedFusionNetwork`, `ndarray`, `stream_agama.py`, `stream_common.py`, `Community 56`?**
  _High betweenness centrality (0.065) - this node is a cross-community bridge._
- **Are the 8 inferred relationships involving `AgamaStreamSimulator` (e.g. with `_OrbitCapExceeded` and `RestrictedNbodyStreamSimulator`) actually correct?**
  _`AgamaStreamSimulator` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Are the 36 inferred relationships involving `compose()` (e.g. with `load_simulator()` and `compose_aug()`) actually correct?**
  _`compose()` has 36 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `_jax()` (e.g. with `_stream_summary_grid()` and `_stream_summary_statistics()`) actually correct?**
  _`_jax()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 29 inferred relationships involving `get_simulator()` (e.g. with `load_simulator()` and `main()`) actually correct?**
  _`get_simulator()` has 29 INFERRED edges - model-reasoned connections that need verification._
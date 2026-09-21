# Stellar streams (compositional inference)

The stream project lives on top of the plain pipeline as **registered components + config
presets**. Nothing in `pipeline/` is stream-specific except the compositional branches it
switches on: `composition.level`, the `attention_mask_key` adapter role, and the real-data helpers
in `pipeline/evaluate_real.py`. With everything left at its default the pipeline is the single-level
template described in [running.md](running.md).

## What is added

| Piece | Where | Selected by |
| --- | --- | --- |
| Simulators `stream_agama` (particle spray), `stream_agama_rnbody` (restricted N-body) | `simulators/stream_agama*.py`, `stream_common.py` | `simulator=stream_agama_*` (many presets) |
| Gaia observation model, stream-frame summary statistics | `augmentation/streams.py`, `stream_summary.py` | `augmentation=stream_*` |
| Per-stream parameter standardization, rotation-curve mask, real-data attachments | `preprocessing/streams.py` | `preprocessing=stream_*` |
| `fusion` (per-observable backbones + attention-mask routing), `masked_set_transformer`, `masked_time_series_transformer`, `mlp`, `feature_transformer` | `networks/` | `model=stream_fusion_*` |
| Compositional prior scores, member flatten/group helpers | `pipeline/compositional.py` | `composition=global` |
| Summary-space misspecification (MMD) test | `pipeline/misspecification.py` | `eval.misspecification_reference` |
| BayesFlow 2.0.12 compositional-conditions reshape fix | `pipeline/_bf_patches.py` | applied whenever `composition!=none` |

## Hierarchy and composition levels

A hierarchical simulator declares `global_parameter_names` (shared by all members of a group, e.g.
the Milky Way potential), `local_parameter_names` (per stream), `context_keys` (the member index
`j`) and implements `sample_compositional`. `composition.level` picks what is inferred:

| level | inference_variables | inference_conditions | workflow | evaluation |
| --- | --- | --- | --- | --- |
| `none` | `parameter_names` | — | `BasicWorkflow` | per row |
| `global` | globals | context (`j`) | `CompositionalWorkflow` | per member (`base_*`) **and** pooled `compositional_sample` (`compositional_*`) with the prior score (`eval.prior_score`: `spec` analytic, `kde`) |
| `local` | locals | globals + context | `CompositionalWorkflow` | simulated: per member conditioned on the true globals; real: `ancestral_sample` with globals from `composition.global_run_dir` |

The adapter derivation follows the level automatically (`pipeline.adapter.fill_adapter_from_simulator`).

## Stages

`simulate-multistream` (or `python -m hydrabflow.pipeline.simulate_multistream`) writes the grouped
test sets: globals `(n, 1)`, member arrays `(n, m, ...)`. Training uses the flat `simulate` output.
`evaluate` handles both levels and both data kinds:

```bash
# train the global model
uv run hydrabflow-train simulator=stream_agama_rnbody_huang model=stream_fusion_model5 \
    composition=global adapter=stream preprocessing=stream_global_log10 augmentation=stream_global \
    data.data_dir=data/streams/... data.n_simulations=60000

# grouped test set + simulated evaluation (base + compositional)
uv run hydrabflow-simulate-multistream simulator=stream_agama_rnbody_huang \
    data.dataset_name=test_multistream_333.npz data.n_simulations=333
uv run hydrabflow-evaluate ...same groups... eval=stream_compositional model_dir=<train run>

# real Gaia streams: preset the *real* preprocessing/augmentation chains
uv run hydrabflow-evaluate ...same model/composition... \
    preprocessing=stream_real_global_log10 augmentation=stream_real_global \
    data.real_data_path=assets/gaia/gaia_observed_streams_6Dwitherrors.npz model_dir=<train run> \
    eval.misspecification_reference=<simulated evaluate run dir>
```

Real-data runs at `composition=global` additionally write `single_stream_posterior.npz` and
`real_global_vs_streams_corner.png` (pooled vs per-stream posteriors) and, when a reference run is
given, `misspecification.json` + `mmd_hypothesis_test.png`.

The shell scripts under `scripts/` are complete worked pipelines (dataset creation, training,
tuning, PPC) for each dataset generation; the analysis scripts (`ppc_*.py`, `sumstat_sim_vs_real.py`,
`misspecification_per_channel.py`, the trihedron remap study) read finished run dirs.

## Presets

Because the stream experiments vary every block, those groups keep swappable files while their
*defaults* stay inline in `conf/config.yaml` (that file lists `- <group>: null` so `adapter=stream`
works without a `+`). Presets compose by inheritance where that avoids repetition
(`conf/simulator/stream_agama_ibata.yaml` inherits `stream_agama_spray_huang`, ...). Whole-model
presets under `conf/model/` select their networks with `override summary_network: ...`.

Pairings that must match: a `stream_global*` augmentation with the corresponding
`stream_real_global*` one for real data; the fusion `params.backbones` keys with
`adapter.summary_variables`; `j` first in `inference_conditions` (the member count is read from it).

## gala forward model (`stream_gala`)

`simulator=stream_gala_spray_mw22` is a particle-spray twin built on **gala** instead of AGAMA
(`src/hydrabflow/simulators/stream_gala.py`, subclassing `AgamaStreamSimulator` and swapping only
the row worker). Potential = the `MilkyWayPotential2022` family: fixed Hernquist bulge + nucleus,
free `MN3ExponentialDiskPotential` (`m_disk`, `h_R_disk`, `h_z_disk`) and an NFW halo built from
(`log10_M200_halo`, `c200_halo`) with gala's `from_M200_c` algebra (Planck18 ρ_crit) and
**flattening in the potential** (`NFWPotential(c=c_phi)`). The prior is on the **density** axis
ratio `q_rho_halo` at `q_ref_r_kpc` (15 kpc); `c_phi` is derived per row by inverting the
analytic Poisson map (`c_phi_halo_derived`; the potential-flattened NFW has negative density on
the pole for `c_phi < ~0.89`, only at z > ~36 kpc — `halo_rho_neg_r_kpc_derived` records where).
Stream = gala `ChenStreamDF` + `MockStreamGenerator` with a Plummer progenitor and linear mass
loss via a time-varying `prog_mass`; gala releases `2 (n_steps + 1)` stars, so `n_steps = 4999`
for 10^4 particles (checked). Integrator defaults to **leapfrog**: gala steps all particles in one
shared adaptive dop853 call, and a single star through the 30 pc Plummer core collapses the step
("Integration failed with code -4") in ~15-40 % of 1e4-star rows — leapfrog never fails and agrees
with dop853 to 6 pc median per particle after 4 Gyr. Frame = astropy's default `Galactocentric`
(the solar frame is not varied). gala is imported only inside the loky worker. Cost ~15 s/row.

Presets: `preprocessing=stream_global_log10_gala_2modal` / `stream_real_global_log10_gala`
(log10 on the disk parameters); adapters/augmentations/models are the 2-modality ones unchanged.
`scripts/create_gala_mw22_dataset.sh` = pilot + 333-group test set + 10^4 flat set + the
prior-predictive coverage checks in the track, grid and **raw-particle**
(`scripts/ppc_particle_coverage.py`: quantile envelopes, 2-D overlays, kernel-MMD rank per
observable subset) representations, then prints the two training commands. Dataset dir
`data_jarvis/data_gala_spray_mw22_hydrabflow/`.

## Resources

Gaia member/error tables and the observed stream `.npz` live in `assets/gaia/` (see its README).
The augmentations default to `data/` (a symlink to shared storage); on a fresh clone pass
`++augmentation.params.resources_dir=assets/gaia`. Datasets are large and gitignored
(`data/`, `data_local/`); each `create_*_dataset.sh` regenerates one.

## Training notes

- `training.standardize` includes `summary_variables` (the stream observables are standardized
  per batch after augmentation, not by a preprocessing step); `training=stream_local` swaps that
  for `inference_conditions`.
- Attention widths are `num_heads * embed_dim_per_head` everywhere, including inside
  `params.backbones` and the tuning search spaces.
- BayesFlow is pinned to the PyPI 2.0.12 release because `_bf_patches.py` targets it; `main`
  tracks the git HEAD (2.0.13). Re-check the patch before upgrading.

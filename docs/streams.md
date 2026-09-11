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

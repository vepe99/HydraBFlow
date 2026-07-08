# assets/gaia — portable static inputs for the stream project

Small, static input files that the stream scripts read, gathered here so the repo is
**self-contained across clusters**. These are *inputs* (a few MB total), distinct from the
generated datasets under the `data/` symlink (which is machine-local shared storage and does
**not** travel with the repo).

> **Dataset creation needs none of these files.** The `simulate` / `simulate_multistream`
> stages read no external data — the Zhou (2023) and Huang (2016) rotation curves are hardcoded
> arrays in `src/hydrabflow/simulators/stream_common.py` (`OBS_*` / `HUANG_*`,
> `extended_rotation_curve`), and the stream progenitors are constants. They travel with the
> source. So a GPU-less cluster whose only job is generating datasets can ignore this folder.
> It matters for the PPC scripts and for any training / real-data evaluation you run.

## Contents

| file | bytes | needed by |
|------|-------|-----------|
| `apjad382dt1_mrt.txt` | 2.4 M | training/real-eval augmentation (`augmentation/streams.py`): Ibata+23 stream member table (member G magnitudes → per-stream KDE) |
| `gaia_stream_id.csv` | 6 K | same: maps stream name → Ibata source id |
| `gaia_DR3_erorr_6D.txt` | 612 B | same: Gaia DR3 6D measurement error vs magnitude |
| `gaia_observed_streams_6Dwitherrors_cutNGC3201.npz` | 242 K | **current** real observations — PPC `--real` default + `data.real_data_path` for `evaluate_real` (Pal5/NGC3201/M68, NGC3201 window cut) |
| `gaia_observed_streams_6Dwitherrors.npz` | 242 K | prior variant (pre-NGC3201-cut); kept for provenance |
| `gaia_observed_streams.npz` | 169 K | older variant; kept for provenance |
| `Pal5_track.npz`, `NGC3201_track.npz`, `M68_track.npz` | ~490 K | reference stream sky tracks; **not read by current repo code** — kept for plotting/provenance |

Provenance: Gaia tables copied from the `data/` shared-storage symlink target
(`/export/data/vgiusepp/multistream/`); the `gaia_observed_streams*` and `*_track` npz from the
reference project `…/diffusion-experiments/case_study5/project_stream/data/`.

## Using these on a new cluster

- **PPC scripts** (`ppc_prior_predictive.py`, `compare_spray_methods.py`): `DEFAULT_REAL`
  auto-prefers `assets/gaia/gaia_observed_streams_6Dwitherrors_cutNGC3201.npz` when present, so
  no flag is needed. Override with `--real <path>` if you want a different observation set.
- **Training / real-data eval augmentation** reads its Gaia tables from `resources_dir`, which
  defaults to `data` (the shared-storage symlink on this cluster). On a cluster without that
  symlink, point it here:
  `+augmentation.params.resources_dir=assets/gaia` (and `data.real_data_path=assets/gaia/gaia_observed_streams_6Dwitherrors_cutNGC3201.npz`).
- **Rotation curves (Zhou + Huang):** nothing to copy — they live in `stream_common.py`.

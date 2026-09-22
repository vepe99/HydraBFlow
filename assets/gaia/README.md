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
| `gaia_observed_streams_6Dwitherrors_cutNGC3201_desi.npz` | 236 K | 2026-09-21: same STREAMFINDER members, v_los augmented with DESI DR1 MWS (`desi_dr1.mws` via NOIRLab Data Lab TAP, 2" match, `rvs_warn=0`, lowest error wins): Pal5 69->77, NGC3201 37->48, M68 29->47 |
| `gaia_observed_streams_6Dwitherrors_cutNGC3201_desi_m68palau.npz` | 236 K | as above, but the M68 arm = Palau & Miralda-Escude 2025 Gaia DR3 selection (Zenodo 17020518 `stream.csv`, 287 of 291 stars inside the M68 window: 195 main component + 92 envelope; 32 v_los incl. 30 DESI) |
| `gaia_observed_streams_6Dwitherrors_cutNGC3201_desi_m68palau_main.npz` | 236 K | **recommended (2026-09-21)**: as above with the M68 ENVELOPE removed (main component only, 195 stars, 16 v_los). Prior-predictive checks: the spray forward model reproduces this M68 arm's per-bin dispersions but not the envelope's |
| `gaia_observed_streams.npz` | 169 K | older variant; kept for provenance |
| `Pal5.txt`, `Pal5_radvel.txt`, `NGC3201.txt`, `M68.txt` | ~40 K | Palau & Miralda-Escude (2023) appendix member tables (GDR2), transcribed from the paper: Table B1 (126 Pal 5 stars) + B2 (15 with Ibata+2017 radial velocities), C1 (170 NGC 3201), E1 (115 M68, no radial velocities). Read by `scripts/build_palau23_members.py` |
| `gaia_observed_streams_palau23.npz` | 236 K | 2026-09-22: the Palau23 members in the standard real-observation layout, cut to the observation windows (126 / 71 / 96 stars; M68's box narrowed to 190<alpha<260, dec>=-8). Built by `scripts/build_palau23_members.py`; provenance in the sidecar `.json` |
| `gaia_observed_streams_palau23_full.npz` | 236 K | same, `--window keep`: the tables verbatim (126 / 170 / 115), including the 99 NGC 3201 stars beyond RA 140 that no simulation in this project can produce |
| `gaia_observed_streams_palau23_sfrv.npz` | 236 K | as `..._palau23.npz` but with radial velocities cross-matched in from the Ibata+2024 STREAMFINDER atlas (`apjad382dt1_mrt.txt`): 17 / 18 / 8 v_los for Pal5 / NGC3201 / M68 (Palau23 alone gives 15 / 0 / 0). `build_palau23_members.py --streamfinder-rv`, on by default |
| `gaia_observed_streams_palau23_dr3.npz` | 236 K | the same Palau23 membership with **Gaia DR3** astrometry (resolved through `gaiadr3.dr2_neighbourhood` over the ESA TAP; 411/411 ids resolved, 1 two-parameter solution dropped) and the catalogue's own per-star errors in an extra `obs_error` (1,3,P,6) array. `build_palau23_members.py --astrometry dr3`. DR2 and DR3 values are never mixed in one file |
| `palau23_gaiadr3.csv` | 90 K | cached TAP response behind the file above, so rebuilds are offline |
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

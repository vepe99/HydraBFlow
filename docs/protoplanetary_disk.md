# The protoplanetary-disk project

Multimodal simulation-based inference for **protoplanetary disks**. A radiative-transfer (RT) code
produced ~49 000 model disks; each row is 18 physical parameters plus a clean 4-channel image and a
clean 19-bin SED. We train a neural posterior estimator to recover the parameters from *mock
observations* of those rows — JWST NIRSpec at 3.9 µm plus ALMA in three bands — and the thing that
turns a clean RT row into a mock observation is applied **on the fly, per batch**, so one dataset
serves any observing setup.

```bash
uv run hydrabflow-train    experiment=protoplan
uv run hydrabflow-evaluate experiment=protoplan model_dir=outputs/protoplan/protoplan_npe/<timestamp>
```

That is the whole interface. Everything below is what those two commands do and which knobs are
worth turning.

---

## 1. What the pipeline is

```
  params_combined.npy            ProtoplanetaryDiskSimulator.load_dataset()
  im_jy_*.npy          ────────► 17 targets + 2 discrete conditions + im_jy, seds, ...
  sed_flx_jy_*.npy                        │
  sed_lams_*.npy                          │  tail_split: hold out the last n_test rows
  im_lams_*.npy                           ▼
                          protoplan_instrument  (per batch, on device)
                            extinction · distance · Jy/px → MJy/sr
                            JWST PSF · per-band ALMA beams
                            rotation · parity flip
                            resample to each instrument's own pixel grid
                            instrument noise · SED noise
                                          │
                                          ▼
                          the adapter  (_protoplan_spec.build_adapter)
                            asinh(x/σ) per band
                            → jwst_input, b9_input, b7_input, b6_input, sed_input
                            → geom_cond, jwst_cond, b9_cond, b7_cond, b6_cond
                            → inference_conditions = [sil_id, has_cavity]
                                          │
                                          ▼
                          protoplan_fusion   5 CNN/transformer branches, late-fused
                          protoplan_flow_matching   the posterior over 17 parameters
```

There is **no `simulate` stage**. The forward model is an external code, so the simulator class
implements two optional `BaseSimulator` hooks — `load_dataset()` and `build_adapter()` — and
`simulator.params.data_path` points at the directory of `.npy` caches the RT runs produced. See
[extending.md](extending.md) for that pattern in general.

### Files

| what | where |
|---|---|
| dataset reader, parameter spec, the adapter | `src/hydrabflow/simulators/protoplan.py`, `_protoplan_spec.py` |
| the instrument model | `src/hydrabflow/augmentation/protoplan_instrument.py` |
| conditioned CNNs, modality dropout, builders | `src/hydrabflow/networks/protoplan.py` |
| config | `conf/experiment/protoplan.yaml`, `conf/simulator/protoplan.yaml`, `conf/model/*/protoplan_*.yaml` |
| extinction law + real disks' SED error bars | `assets/protoplan/` |
| checks | `tests/test_protoplan.py` (CPU, synthetic data, no GPU) |

---

## 2. Prerequisites

Beyond `uv sync`:

* **`$STPSF_PATH`** — `stpsf` needs its reference-data tree for the JWST NIRSpec IFU PSF. Without
  it, building the instrument model fails immediately.
* **the caches** — under `simulator.params.data_path`: `params_combined.npy`, one `im_jy_*` variant,
  `im_lams_combined.npy`, `sed_flx_jy_combined.npy`, `sed_lams_combined.npy`.

The image cache comes in several samplings of the same fixed 3″ field:

| file | grid | size | `px_arcsec_mod` |
|---|---|---|---|
| `im_jy_combined_NHWC.npy` | 300×300×4 | 72 GB | `3.0 / 299` |
| `im_jy_combined_resized_128x128.npy` | 128×128×4 | 13 GB | `3.0 / 127` ← the default |
| `im_jy_combined_resized_55x55.npy` | 55×55×4 | 2.4 GB | `3.0 / 54` |

The row filter (`np.load(..., mmap_mode="r")[keep]`) materialises the kept rows in RAM, so the
300×300 cache needs ~70 GB of it. Use the 55×55 one for smoke runs.

> **`px_arcsec_mod` must match the grid you pick.** The PSF and beam kernels are sized from it in
> the instrument model's `__init__`, before any batch exists, while the Jy/px → MJy/sr conversion
> derives its solid angle from the array's own height. So a mismatch is not a shape error and not a
> unit error — it is the *right* image convolved with the *wrong* beam. `preprocess` therefore
> checks the two against each other on the first batch and raises.

---

## 3. The knobs: beam and noise

Every constructor argument of `AugmentationsClass` is available under `augmentation.params`, so
there is no per-knob plumbing to keep in step. These are the ones that matter:

```yaml
augmentation:
  steps: [protoplan_instrument]
  params:
    randomize_alma_setup: false
    alma_beams: [[0.20, 0.10, 0.0], [0.20, 0.10, 0.0], [0.20, 0.10, 0.0]]  # (maj, min, PA)
    alma_noises_jy_beam: [8.0e-5, 8.0e-5, 8.0e-5]
    jwst_noise_mjy_sr: [0.5, 1.0]
```

| knob | meaning |
|---|---|
| `randomize_alma_setup` | `false` = one fixed measured setup, so the estimator is `p(θ \| data)` for *that* instrument. `true` = draw per image from the ranges below, and the estimator marginalises over them |
| `alma_beams` | `[[FWHM_maj, FWHM_min, PA], …]`, one entry per band (450 / 880 / 1300 µm), arcsec and degrees |
| `alma_noises_jy_beam` | per-band RMS, Jy/beam |
| `jwst_noise_mjy_sr` | `[lo, hi]` MJy/sr, drawn uniform per image; `[x, x]` pins it |
| `alma_beam_fwhm_maj_range` | per-band `[lo, hi]` major FWHM for the randomized prior, arcsec. Ignored when `alma_beam_theta_range` is set |
| `alma_beam_theta_range` | per-band `[lo, hi]` on θ = `sqrt(maj·min)`, arcsec — the *other* parametrisation: `maj = θ/sqrt(q)`, `min = θ·sqrt(q)`. The measured beams are tight in it (an array configuration sets the resolution θ, the uv coverage's shape sets `q`), so a band's box can be narrowed without also constraining its elongation. `null` keeps the major-axis one, which is what every run before `experiment=protoplan_newbeam` used |
| `alma_beam_axis_ratio_range` | minor = `ratio × major`, so the beam cannot come out inverted — **there is no separate minor-axis range**. One `[lo, hi]` is broadcast to every band; a list of three sets it per band |
| `alma_noise_jy_beam_range` | per-band `[lo, hi]`, log-uniform |
| `alma_obs_px_arcsec` | ALMA output pixel scale; `null` → `min(beam major)/7`, which under the randomized prior is a function of *that prior* and so moves between runs |
| `dist_pc` | the source's distance; the RT models are at 140 pc |
| `av` | foreground extinction, mag |
| `sed_sigma_jy` | fixed per-bin SED σ. Default `null` = a wavelength-binned log-normal calibrated from five real disks' error bars (`assets/protoplan/seddata_*.txt`); supply this and those files are not read at all |
| `sed_frac_cal_err` | fractional calibration error added in quadrature to that drawn floor: `σ = sqrt((frac·flux)² + floor²)`. Default `0.0` = the floor alone, which is an *absolute* error bar in Jy with no flux dependence — across a population spanning fourteen decades that gives a median training SED of S/N 67, cleaner than any real photometry. `0.04` puts the median at S/N 25 and the p10–p90 at 3.4–43, bracketing hvtauc (2.3% → 43) and oph163131 (4.7% → 21) |
| `jwst_psf_gaussian_fwhm_arcsec` | replace the stpsf NIRSpec PSF with a circular Gaussian of this FWHM, arcsec. Default `null` keeps stpsf. Also removes the `$STPSF_PATH` dependency. See §3.1 for why you might want to |

To point a run at a real disk's measured setup: put its beam in `alma_beams`, its per-band RMS in
`alma_noises_jy_beam`, its JWST σ in `jwst_noise_mjy_sr`, its distance in `dist_pc`, and set
`randomize_alma_setup: false`.

### 3.1 The JWST PSF, and why a Gaussian is a real option

The stpsf NIRSpec IFU kernel carries the hexagonal-aperture diffraction structure — the four-lobed
cross visible all over `plot_population.py`'s JWST gallery — and that structure is fixed on the
**detector**, while `apply_rotation` spins the source through 0–360°. So in training the spike
pattern always sits at the same angle relative to the array while the disk rotates against it: an
absolute orientation reference that exists only in the training set, because a real JWST
observation has its own V3 position angle. It is the same class of error as convolving before
rotating (fixed in `__call__`) — the physics is right, and a training-only landmark is left in the
kernel.

`jwst_psf_gaussian_fwhm_arcsec` removes it. The trade is the halo, and it is not small:

| | stpsf kernel | Gaussian, FWHM 0.16″ |
|---|---|---|
| core FWHM (half-max) | 0.1605″ | 0.16″ |
| second-moment FWHM | **0.527″** | 0.16″ |
| r(EE=50 / 80 / 90%) | 0.109 / 0.234 / 0.429″ | 0.068 / 0.123 / 0.157″ |

On the 0.1″ observed grid over a 3.1″ field, 20% of the stpsf flux sits beyond r = 0.23″ and 10%
beyond r = 0.43″ — a third of the field. Dropping it raises the peak of a bright compact source by
**1.3–1.6×**. Use **0.16″**, the measured core width, not the 0.151″ textbook diffraction limit
(1.22 λ/D at 3.9 µm on 6.5 m): matching what the old runs actually convolved with is what makes the
comparison controlled. The other way out — randomizing the PSF orientation along with the source —
needs one kernel per rotation group, i.e. the ALMA beam-group machinery again, and is not
implemented.

`experiment=protoplan_gausspsf` sets this and `sed_frac_cal_err` together, on top of
`protoplan_newbeam`.

#### Measured, 2026-09-11: the swap made both disks *more* out-of-distribution

`gausspsf_all17` (2000 epochs) was scored against `newbeam_all17` with `check_summary_range` on the
full 49811-row population, both disks, all branches. Read the NN-distance ratio (the real disk's
distance to its nearest training row, over the population's own typical nearest-neighbour spacing)
rather than the percentile, which saturates near 100 and compresses the range that matters:

| | branch | gausspsf | newbeam | |
|---|---|---|---|---|
| hvtauc | all bands | ×2.89 (p99.99) | ×2.42 (p99.93) | worse |
| | jwst_input | ×4.36 | ×4.30 | ~same |
| oph163131 | all bands | ×2.71 (p99.98) | ×2.21 (p99.86) | worse |
| | **jwst_input** | **×3.58 (p99.91)** | **×1.67 (p84.62)** | **much worse** |

oph163131's JWST branch was comfortably *inside* the population under the stpsf PSF (p84.6) and is
badly outside it with the Gaussian (p99.9). That branch is the one the knob changes, so this is close
to a direct measurement: **the halo carries real structure.** A real JWST observation has those wings
whether the training set does or not, so removing them makes the training population less like the
data, not more. The spike-orientation argument above is sound in principle and simply does not
dominate — the ×1.3–1.6 peak change was the bigger effect all along, and should have been read as
evidence against the swap rather than as an acceptable cost.

**Do not use `jwst_psf_gaussian_fwhm_arcsec` to chase an OOD verdict.** It stays in the code (default
`null`, so nothing changes unless asked) for the cases it was really useful for: running without
`$STPSF_PATH`, and any test that wants a PSF with no orientation structure.

One thing this does *not* separate: `protoplan_gausspsf` turns both changes on at once, so strictly
the result indicts the *pair*. `sed_input` barely moved (hvtauc ×1.84 vs ×1.42, oph163131 ×2.01 vs
×2.05) while the JWST branch moved a lot, which points at the PSF — but a `sed_frac_cal_err`-only run
is what would settle it, and has not been done.

**Changing the noise means changing the asinh knees too.** `simulator.params.asinh_sigma_jwst` and
`asinh_sigma_alma` are where the adapter's image compression turns over from linear to logarithmic
(§4). They should stay in the same ballpark as the noise you configured — the same order of
magnitude is enough, but far off and the transform degenerates into either a near-identity (knee ≫
noise) or a pure log (knee ≪ noise), which defeats the point of having it.

---

## 4. Three coupled choices in the encoding

These are not flags to explore; they are what this arm settled on, and each has a reason worth
knowing before changing it.

### Discrete parameters are conditions, not targets

`sil_id` (silicate composition) and a derived `has_cavity` are `inference_conditions`, so the
estimator is `p(θ₁₇ | data, sil_id, has_cavity)`. A flow cannot represent a delta atom, and the
alternative — smearing both indicators into narrow continuous clusters — is what this replaces.

`log_r_cav` stays a target, and on the no-cavity rows it is redrawn from the *real* cavity branch
(`CAVITY_LOG_R_CAV_RANGE`) rather than held at the sentinel. Conditional on `has_cavity = 0` the
cavity radius is unidentified, so a posterior that returns the prior there is the honest answer;
drawing from the real branch is what makes that a true statement instead of an artefact of an
off-support range.

Consequence for reading results: what comes out is **four conditional posteriors**, one per
`(sil_id, has_cavity)`. Nothing in NPE estimates `p(sil_id, has_cavity | data)`, so they cannot be
mixed into one marginal without separately estimated weights.

The redraw is seeded (`simulator.params.param_jitter_seed`). Leave it fixed: unseeded, two runs of
the same config differ in their *training targets*, and any A/B between them is confounded.

### The measurement conditions go into the summary networks

Geometry (`rot_sin`, `rot_cos`, `sky_flip`) and each band's beam/noise are spliced into the CNN
trunks — after the conv stages and the pooling head, before the projection to `summary_dim` — rather
than concatenated onto the summary vector and handed to the flow. Whether a faint ring is real or a
noise fluctuation, and how much a beam smeared it, is a question about the *image*, and the CNN is
the only network that sees the image. By the time the noise level reaches the flow the image has
already been summarised without it.

Routing is per instrument and defined once, in `_protoplan_spec.summary_condition_routing()`, from
the same source as the adapter's groups, so the two cannot disagree. Geometry gets its own key
because `Concatenate` *pops* what it consumes, so one key cannot feed two groups — the reuse happens
inside the network. The SED transformer is deliberately unrouted: its per-bin noise already arrives
as an input channel of `sed_input`.

### Images are compressed with asinh

`asinh(x / σ_c)` per band, before the standardizer. Without it the only preprocessing is one global
mean/sd per channel, which cannot serve a channel whose pixels span five decades: the sd is set by
the inner rim, so the p10–p90 span of *background* pixels collapses to 0.004 sd on JWST and 0.16 on
ALMA 1300 µm, against a rim reaching 533 sd. Under asinh the same spans are 2.04 and 1.67.

asinh rather than log or log1p because ~40–48% of post-noise pixels are negative: asinh is odd and
finite below zero, so a −2σ excursion is treated exactly as +2σ. Clipping at zero first scores
measurably worse on every channel, precisely because it throws away half the noise floor.

---

## 5. Four image branches, one per band

JWST plus ALMA **B9** (450 µm), **B7** (880 µm) and **B6** (1300 µm) each get their own CNN, rather
than being stacked as four channels of one. They are observed with *different beams*: a shared
convolutional trunk would have to serve three different point-spread functions at once, and each
band's beam/noise conditions could only be handed to it pooled. Per band, each CNN is told the setup
that produced its own map and nothing else.

Architecture lives in `model.summary_network.params.hp`. One `*_alma` key configures all three
bands; add a `*_b9` / `*_b7` / `*_b6` key to let one band differ:

```yaml
model:
  summary_network:
    params:
      hp:
        cnn_num_stages_alma: 2      # all three bands
        cnn_num_stages_b6: 3        # …except B6
```

Any of those keys can be searched by its dotted path in `tuning.search_space`, e.g.
`model.summary_network.params.hp.cnn_base_width_alma: {type: int, low: 8, high: 32, step: 8}`.

### Missing-modality robustness

```yaml
model:
  summary_network:   { params: { fuse_head: false } }
  inference_network: { params: { missing_modality_prob: 0.3 } }
```

This swaps `FlowMatching` for `GroupedFlowMatching`, which drops each of {B6, B7, B9, JWST, SED}
from the condition vector as a **whole group** — so the network learns to work when a band is
genuinely absent, not merely noisy. Upstream's `missing_conditions_prob` drops each *element*
independently, so a modality's slice of the summary vector essentially never goes together
(probability `p ** summary_dim`); that trains resilience to feature dropout, which is a different
thing.

Two constraints, both enforced rather than documented-and-hoped:

* `fuse_head: false` is required. An MLP head mixes the branch embeddings, so there is no modality
  boundary left in the fused vector to drop. Get this wrong and the run raises, because the widths
  no longer line up (`group_sizes sums to …`).
* The two discrete indicators are never dropped (`always_observed_groups=1`) *by default*. They are
  conditions on the *density*, not descriptions of a measurement, so dropping them asks the network
  to marginalise over a parameter it is meant to be conditioned on -- which is sometimes exactly
  what you want; see below.

### Maskable discrete indicators (`experiment=protoplan_mask_discrete`)

```yaml
adapter: { inference_conditions: [sil_id, has_cavity] }
model:
  inference_network:
    params:
      subnet: diffusion_transformer          # required, see below
      discrete_condition_groups: [sil_id, has_cavity]
```

`discrete_condition_groups` replaces `discrete_condition_dim`: instead of one always-observed group
of width 2, each indicator becomes its **own width-1 droppable group**, dropped independently at
`missing_modality_prob` like a modality (and `always_observed_groups` becomes 0 -- splitting them is
only useful if they can drop). The names must be `adapter.inference_conditions` in the same order;
a count mismatch raises at the first training step via the `group_sizes sums to ...` check.

One model then serves four queries, selected at evaluation time:

| `eval.mask_condition_groups` | estimator |
|---|---|
| `[]` | `p(θ \| data, sil_id, has_cavity)` |
| `[sil_id]` | `p(θ \| data, has_cavity)` |
| `[has_cavity]` | `p(θ \| data, sil_id)` |
| `[sil_id, has_cavity]` | `p(θ \| data)` |

Three things to know before using it:

* **`subnet: diffusion_transformer` is not optional here.** There each condition scalar is its own
  token: a masked one gets the learnable *missing* state embedding and is excluded from attention.
  Under an MLP subnet "missing" degrades to zeroing the column (`GroupedFlowMatching._apply`), and
  `0` is a legal value of both indicators -- "unknown" would be indistinguishable from
  "`sil_id=0`, no cavity".
* **The marginal is over the grid's prior**, which is 50/50 on each indicator *by construction*
  (12.5k rows in each of the four cells), not a measured occurrence rate. Reweighting means
  retraining; the mixture route (`Σ p(k|data) p(θ|data,k)`, weights from a separate classifier on
  the summary embeddings) is the reweightable alternative, and it is also the only route that
  estimates `p(has_cavity | data)` at all. Masking marginalises the indicator; it never infers it.
* **`keep_one` can now spend its guarantee on an indicator**, so `p ** n_modalities` of draws
  (0.3⁵ = 0.24%) have no modality observed at all and train the conditional prior. Deliberate and
  pinned by
  `tests/test_protoplan.py::test_per_indicator_condition_groups_are_droppable_and_the_old_layout_is_unchanged`,
  which checks both layouts side by side.

---

## 6. Reading the output

Each launch writes to `outputs/protoplan/${run_name}/<timestamp>/`: `approximator.keras`,
`preprocessing_state.npz`, `history.json`, `loss.png`, `.hydra/` (the resolved config), and after
`evaluate`, `posterior.npz`, `metrics.json` and the diagnostic figures.

`eval.diagnostics` includes `per_discrete_branch`, so `recovery`, `calibration_ecdf` and
`z_score_contraction` are written once pooled **and** once per branch
(`recovery_sil_id0_has_cavity1.png`, …). Branches with fewer than 50 held-out rows are skipped, with
a log line saying so.

**Read the per-branch figures.** The estimator is a separate conditional posterior per branch, so
the pooled `recovery` panel superimposes four of them: `log_r_cav` looks broken there because on the
`has_cavity = 0` rows it is unidentified by construction, and looks correct in none of them.

**Read contraction, not just calibration.** A posterior equal to the prior is perfectly calibrated,
so a mean contraction near 0 means the run learned nothing however good `calibration_ecdf` looks.
Some parameters are genuinely uninformed by the data (`X_sil` is the known case) and their wide
marginals are not a bug.

---

## 7. Checks

```bash
CUDA_VISIBLE_DEVICES="" HYDRABFLOW_NUM_GPUS=0 uv run pytest tests/test_protoplan.py -q
```

CPU only, synthetic caches in `tmp_path`, ~1 minute. What it pins is the wiring, not the physics:
the label round-trip and the `(N,1)` vs `(N,)` shape-by-role split, that the reader's shapes are what
the adapter expects, that the instrument model's output keys are what the adapter renames, that each
band's routed conditions are its own, that `standardize: [all]` survives as the string `"all"`, that
the modality mask drops whole groups and never the discrete pair, and that a gradient reaches the
weights in one real `fit_offline` step.

A smoke run on real data, still on CPU:

```bash
CUDA_VISIBLE_DEVICES="" HYDRABFLOW_NUM_GPUS=0 uv run hydrabflow-train experiment=protoplan \
  simulator.params.image_file=im_jy_combined_resized_55x55.npy \
  augmentation.params.px_arcsec_mod=0.05555555555555555 \
  training.n_epochs=2 preprocessing.steps.0.n_test=200 preprocessing.steps.0.n_train_cap=600 \
  run_name=smoke
```

Note the paired `image_file` / `px_arcsec_mod`: 3.0 / (55 − 1).

## 8. Prior-predictive checks on a real disk

Before trusting a posterior for a real observation, ask whether that disk is inside the population
the networks were trained on. Three marimo notebooks under `notebooks/prior_predictive_checks/`,
one per space, each writing to `plots/<disk>/`:

| notebook | question | needs |
|---|---|---|
| `check_sed_range.py` | is the SED reachable, band by band? | nothing but `assets/protoplan/` — CPU, no pickle |
| `check_sed_corner.py` | is it reachable *jointly*, i.e. are the **colours** right? | GPU, the real-disk pickle |
| `check_obs_range.py` | is the *amplitude* in range? | GPU, `$STPSF_PATH`, the real-disk pickle |
| `check_shape_range.py` | is the *morphology* reachable? | GPU, `$STPSF_PATH`, the real-disk pickle |

```bash
uv run marimo edit notebooks/prior_predictive_checks/check_sed_range.py
# or headless, on a short pass:
OBS_N_ROWS=4096 uv run python notebooks/prior_predictive_checks/check_obs_range.py
# the joint version, and the plain gallery of what the population looks like:
uv run python notebooks/prior_predictive_checks/check_sed_corner.py 4000 experiment=protoplan_gausspsf
CHECK_DISK=hvtauc uv run python notebooks/prior_predictive_checks/plot_population.py 200
```

**Per-band and joint are different questions.** A disk can sit at the 50th percentile in every
band and still be nowhere near the population, because the bands are strongly correlated and it is
the *colours* that carry the shape of the SED. `check_sed_corner.py` is the joint version — a
corner plot over nine representative bands plus the Mahalanobis distance in log-flux space,
referenced against the empirical distribution of the training rows' own distances. Both targets
pass it: hvtauc `d² = 6.3` (59th percentile), oph163131 `d² = 10.7` (81st), against a training
median of 5.3. oph163131's only tension is the mid/far-IR — 24 µm at the 8.5th percentile, 70 µm at
the 5.4th — it is colder than the typical training disk, consistent with being more edge-on. **The
SED is not the channel these disks fall out of.**

`_realdisk.py` beside them holds everything shared: the per-disk `DISKS` registry, `build_real_batch`
(the only place that reads a real observation), and the scoring toolkit (rank-normal scores,
Mahalanobis against the training rows' *own* distance distribution, and the eight flux-normalised
morphology features). `tests/test_prior_predictive_checks.py` pins the numerics analytically — no
GPU, no caches, no pickles.

Three things about these checks that are easy to get wrong:

* **The real images come from `assets/protoplan/`, not the repo's git history.** `load_real_images`
  reads `realimg_<disk>.npz` -- the four arrays it actually uses (`im_cent` / `im_cent_MJyster`,
  `rRA`, `rDEC`), cropped to +/-3", twice the model field, a few hundred kB per disk. Regenerate them
  from the upstream `data/obs_data/imdata_<disk>.pkl` (~150-300 MB each, `dill`, class definition
  inline, so they unpickle without importing that package) with
  `uv run python notebooks/prior_predictive_checks/_vendor_real_images.py`; `$PROTOPLAN_OBS_DIR` says
  where those live. With neither present the notebooks say so and stop. `assets/` is gitignored, so a
  fresh clone starts from the pickles.
* **The regrid target must match the forward model's grid.** `build_real_batch(disk, lams)` uses the
  disk's own measured beams, which is what `fixed_setup_augmentation` produces (122 px for
  oph163131). The *randomized* training prior samples ALMA finer (162 px), so `check_obs_range` must
  pass `alma_px=randomized_alma_obs_px(cfg)`. Comparing pixel statistics across two samplings of the
  same field is not a like-for-like comparison; both notebooks assert the shapes agree rather than
  trusting it.
* **Some per-disk constants are assumptions, not measurements** — the distance, `A_V`, and the JWST
  noise floor. They are declared in one place (`DISKS`) and echoed at the top of every notebook, and
  `check_shape_range`'s A_V scan exists precisely to test the `A_V = 4.0` one. Upstream deliberately
  overrode the measured JWST sigma for the saucer (0.03 rather than 0.98, the JSON value being
  estimated over a radius where the surface brightness is still falling, so it tracked the outer halo
  rather than the noise floor); that argument has not been redone for the other disks.

A disk missing a band is a **missing modality**, not an error: oph163131 has no B7 (880 µm)
measurement, so it gets no b7 figure, row, or score. The trained model already handles this
(`missing_modality_prob` + `GroupedFlowMatching` drop whole modalities). The forward model still
needs *some* beam for all three ALMA channels, so the absent one is given the widest measured beam
and every output of it is discarded — and `alma_obs_px_arcsec` is passed explicitly from the measured
beams alone, since it is derived from `min(maj)` across channels and a placeholder could otherwise
silently re-grid every band.

### Sampling A_V instead of assuming it

`A_V` is the assumption most worth testing, and testing it needs **no change to the augmentation**:
`AugmentationsClass` already reads an `(lo, hi)` `av` as "draw one A_V per disk and marginalise over
it", exactly as it does for the ALMA beam when `randomize_alma_setup` is on. All three notebooks take
an `A_V` override (blank = the disk's assumption, a scalar, or `lo,hi`; `$CHECK_AV` seeds it
headlessly) and write to `plots/<disk>_av<lo>-<hi>/` so a non-default run cannot overwrite the
as-trained figures.

What it can and cannot reach follows from extinction being one multiplicative scalar per channel.
Over `A_V` in [1, 5] the swing is:

| band | 0.5 µm | 0.76 µm | 1.65 µm | 3.9 µm | 24 µm | 70 µm | 450/880/1300 µm |
|---|---|---|---|---|---|---|---|
| dex | 1.72 | 1.14 | 0.33 | 0.12 | 0.08 | 0.014 | < 0.001 |

So it moves the optical/near-IR SED decisively, the JWST channel slightly, and the ALMA channels not
at all. An ALMA finding — a pixel-scale flag or a shape feature — is therefore **not** an extinction
problem, whatever the posterior says. `tests/test_prior_predictive_checks.py` pins both ends of that
table so the argument cannot rot.

Measured on oph163131 (`A_V = 4` → `A_V ~ U(1,5)`): the 0.5 µm SED rank falls from p96.8 to p89.2
and the best free-rescale χ²/N from 548 to 169, so the optical excess *is* recovered; the 70 µm
deficit (p5.4) and the ALMA shape features do not move.

Sampling `A_V` in *training* is a one-line config change (`augmentation.params.av: [1.0, 5.0]`), but
it is a different model: `A_V` becomes a marginalised nuisance the networks cannot infer, so the
posteriors widen, and the JWST asinh knee was tuned against the fixed-`A_V` amplitude distribution
(see §4) and is worth rechecking.

### The same question in the network's own summary space

`check_summary_range.py` is the fourth check and the only one that asks the question where it
matters — the space the inference network actually conditions on, rather than one we chose. It needs
a finished run, which is why it came last.

```bash
uv run python notebooks/prior_predictive_checks/check_summary_range.py <model_dir> [n_rows] \
    experiment=protoplan_mask_discrete 'adapter.inference_variables=[]' \
    'augmentation.params.av=[1.0,5.0]'      # the overrides the run was TRAINED with
```

It streams the training set through the augmentation as configured for training, then through the
adapter and the summary network with the approximator's own standardization applied, and puts the
real disk beside it. With `fuse_head: false` the summary vector is the per-branch embeddings
concatenated in `sorted(backbones)` order, so it scores **per instrument as well as globally**,
which is what localizes a verdict to a band. Scoring reuses `_realdisk`'s toolkit (rank-transform
each dimension through the training column's own ECDF, then Mahalanobis against the *empirical*
distribution of the training rows' own distances) so the numbers stay comparable to the other three.

Three things make it readable:

* **The absent band is a built-in positive control.** oph163131 has no B7, so that branch's
  embedding comes from a zero image and *must* come out extreme. It does — nearest-neighbour
  distance 4–5.5x the typical training spacing, against ~2x for the bands the disk really has. If
  b7 is not the most extreme block, the check is not working.
* **Judge in the 99%-variance subspace, not the raw one.** Mahalanobis divides by variance, so a
  near-degenerate direction turns a tiny offset into a huge score: two thirds of oph163131's raw
  `d2` comes from the ~40 components holding under 1% of the variance between them, where the
  training covariance is worst estimated. `check_summary_pca.py` decomposes `d2` per component and
  reports both numbers; the truncated one is the defensible one.
* **The embeddings are cached** to `<plot_dir>/summaries.npz` (train, real, branch keys, per-row
  `has_cavity`/`sil_id`). They cost a full augmented pass, so anything else that wants this space
  reads them from there:

```bash
uv run python notebooks/prior_predictive_checks/check_summary_pca.py out.png \
    17-target=<plot_dir_a> 7-target=<plot_dir_b>
```

Measured on oph163131, and consistent across the 7- and 17-target maskable runs: **p99.98 in the
99%-variance subspace** (24 and 26 of 64 components), 21–31% of those components placing the disk
outside their own central 99% against 1% expected, and the displacement driven by the ALMA branches
(`b9` p99.1–99.9, `b6` p94–99.7) with JWST the most in-distribution (p84–87). PC1–PC2 alone put it
*inside* the cloud, which is the reminder that a 2-D projection cannot see this — 61% of the
variance is not the test.

## 9. Posterior inference on a real disk

`hydrabflow-evaluate` has one real-data mode for the whole repo (`data.real_data_path`, see
`docs/running.md`): no truth, no resimulation, posterior corner plots only. Three things are specific
to this arm, and all three are handled by
`notebooks/prior_predictive_checks/make_real_npz.py` -- run it first, then evaluate.

```bash
DISK=oph163131
SRC=outputs/protoplan/protoplan_npe/<timestamp>        # a finished (or in-progress) training run
DST=$SRC/real_$DISK                                    # this evaluation's own directory

# 1. Stage the model. Copying rather than reading `$SRC` in place matters only while that run is
#    still training: it rewrites `approximator_best.weights.h5` on every improvement.
mkdir -p $DST
cp $SRC/approximator_best.weights.h5 $SRC/preprocessing_state.npz $DST/
cp $SRC/prior_bounds.npz $DST/            # the prior box the corner plots shade (optional)
cp -r $SRC/.hydra $DST/source_run_hydra

# 2. The real observation as an evaluate-ready `.npz`. Prints the mask groups step 3 needs.
uv run python notebooks/prior_predictive_checks/make_real_npz.py $DISK $DST/real_$DISK.npz

# 3. Sample. `hydra.run.dir` is what puts the results beside the weights instead of in a fresh
#    timestamped directory.
uv run hydrabflow-evaluate experiment=protoplan \
  model_dir=$DST \
  data.real_data_path=$DST/real_$DISK.npz \
  'eval.mask_condition_groups=[b7_input]' \
  eval.num_samples=1000 eval.batch_size=2 \
  hydra.run.dir=$DST
```

`make_real_npz.py` takes the overrides the model was **trained** with, and emits one row per
combination of whatever `adapter.inference_conditions` holds — two for `[has_cavity]`, four for
`[sil_id, has_cavity]`, in `itertools.product` order. It prints the rows it wrote and the mask
groups step 3 needs.

Out come `posterior.npz` (one `(n_rows, num_samples, 1)` array per target), one
`posterior_corner_obs<i>.png` per row, `evaluate.log` and this launch's `.hydra/`.
`notebooks/overlay_corner.py` puts several of those runs on **one** corner plot — the prior box
shaded behind, every condition regime overlaid, colour for the regime and linestyle for the branch
inside it — which is how a masked posterior is compared against the conditioned ones it should sit
between:

```bash
uv run python notebooks/overlay_corner.py <dir holding mask_none/ mask_sil_id/ mask_has_cavity/> \
    out.png rc h_c log_Mgas ...
``` The corner plots shade the training prior's box (from `prior_bounds.npz`, written
by `train`; absent, the axes just fall back to the posterior's own range) and mark the posterior
median in blue; on a simulated test set the truth is marked in red instead of nothing, and only the
first 5 rows get a figure. Prepend `HYDRABFLOW_NUM_GPUS=0` to both python steps for a CPU run -- one disk at
1000 samples takes about two minutes and needs no GPU.

### What the three steps are for

* **Two rows, not one.** The estimator is `p(theta | data, sil_id, has_cavity)` (§4), and nothing in
  the pipeline estimates the indicators from the data. A real disk therefore gets *both* branches
  sampled, and they are read as two conditional posteriors -- mixing them into one marginal needs
  weights this model does not provide. On oph163131 they agree on every target except `log_r_cav`,
  which is exactly the intended behaviour: unidentified (≈ prior) with no cavity, pinned with one.
  Add `sil_id` to `adapter.inference_conditions` and the same argument makes it four rows.
* **The image grids are the training prior's, not the disk's.** `make_real_npz.py` regrids onto
  `randomized_alma_obs_px(cfg)` (162 px for the shipped config), *not* the disk's own measured beams
  -- the network was trained on that sampling. This is the same trap as `check_obs_range` (§8), from
  the other side.
* **A band the disk lacks is masked, not zero-filled.** oph163131 has no B7 map, so its branch gets a
  zero image plus in-prior placeholder conditions, and `eval.mask_condition_groups=[b7_input]` zeroes
  that branch's whole 16-column slice of the condition vector (`observed_condition_mask` in
  `pipeline/evaluate.py`). Feeding the placeholder unmasked would be inventing an observation. This
  needs a model trained with `missing_modality_prob > 0`; without it `evaluate` raises rather than
  masking a layout it cannot know, because such a model has never seen a masked condition vector.

`make_real_npz.py` also writes zeros for the target parameters. They are not data: the real path
never scores them, but `load_approximator`'s rebuild-from-config fallback probes the adapter, which
concatenates them into `inference_variables`. They must be `(N, 1)`, like the training cache.

Before reading any of this as a measurement, run §8's prior-predictive checks on the disk -- a
posterior conditioned on an out-of-population observation is extrapolation, and it will not say so.
The run above used a mid-training checkpoint deliberately (`inc = 83 (+/-2) deg` for oph163131, a known
near-edge-on disk, was the sanity check); a `metrics.json`-backed statement needs a finished run.

## 10. Troubleshooting

| symptom | cause |
|---|---|
| `silent beam error` on the first batch | `px_arcsec_mod` does not match the image cache's grid (§2) |
| `group_sizes sums to N but the resolved condition vector is M wide` | `missing_modality_prob > 0` with `fuse_head: true` (§5) |
| MemoryError while loading | the 300×300 cache needs ~70 GB after the row filter (§2) |
| `stpsf` fails in `__init__` | `$STPSF_PATH` is unset or its reference data is missing |
| loss is `nan` from step 0 | historically a single non-finite pixel: 48 811 rows and one is enough, and `clipnorm` cannot help since the norm of a NaN vector is NaN |
| a parameter's posterior tracks its prior | may be correct — see §6 on contraction |
| a notebook cannot find `realimg_<disk>.npz` *or* `imdata_<disk>.pkl` | no real images locally: set `$PROTOPLAN_OBS_DIR` and re-run `_vendor_real_images.py` (§8) |
| `expected 2 variables, but received 0` loading the weights for a real run | a target in the `.npz` is `(N,)` rather than `(N, 1)`, so the adapter built a 2N-wide `inference_variables` (§9) |
| `eval.mask_condition_groups needs an inference network exposing group_names` | the run was trained with `missing_modality_prob: 0`; nothing to mask (§9) |
| `real (H,W) vs augmented (H,W)` assertion in a check notebook | the regrid target and the forward model's ALMA grid disagree (§8) |

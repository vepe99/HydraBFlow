"""
protoplan_instrument.py
=======================
Vectorised post-processing pipeline for protoplanetary disk RT simulations: the
instrument model that turns clean RT output into mock JWST + ALMA observations.

Applied on the fly as a BayesFlow augmentation during training, registered as
`protoplan_instrument`.  Every constructor argument is a config knob under
`augmentation.params` -- in particular the ALMA beam (major/minor/PA) and the per-band noise
levels, either as fixed measured values (`randomize_alma_setup: false`) or as a prior to
marginalise over (`randomize_alma_setup: true`).

Merges the per-image workflow from the marimo notebook into a single class
that operates on batches of shape (N, H, W, C) without Python loops over N.

Batch convention
----------------
batch['im_jy']      : (N, H, W, C)  float, Jy / pixel  (raw RT output)
batch['sed_flx_jy'] : (N, Λ)         float, Jy           (optional)

Default channel layout (matches the marimo notebook)
------------------------------------------------------
  0  →  JWST  3.9 µm  (stpsf NIRSpec IFU PSF)
  1  →  ALMA  band A  (elliptical Gaussian beam)
  2  →  ALMA  band B
  3  →  ALMA  band C

Full pipeline (called via __call__)
-------------------------------------
  preprocess()             extinction · distance rescaling · Jy/px → MJy/sr
  apply_rotation()         in-plane rotation of all (N, C) slices at once
  apply_psf_and_convolve() FFT convolution, vectorised over N without any loop
  apply_resampling()       bilinear resample to per-instrument observed grids
  apply_noise()            Gaussian instrument noise
  apply_sed_noise()        per-band SED flux noise
  log10_sed()              log10 of fluxes, sigmas and wavelengths

Vectorisation strategy
----------------------
  * Convolution : numpy rfft2 broadcasts over the N axis — one PSF/beam FFT,
                  multiplied against the whole batch in Fourier space.
  * Rotation    : N angles are sampled independently, then the inverse rotation
                  maps are built analytically for all (N, H, W) pixels via
                  broadcasting — a single map_coordinates call applies every
                  per-image angle without any Python loop.
  * Resampling  : scipy.ndimage.map_coordinates on the full (N, H, W) array —
                  integer coordinates in the N dimension prevent cross-batch
                  blending while floating-point coordinates interpolate in H, W.
  * Noise       : numpy Generator draws the full (N, H_obs, W_obs) array at once.

Random state
------------
A single numpy.random.Generator (seeded at construction) drives all stochastic
operations, giving fully reproducible results.
"""


from __future__ import annotations

import functools
import os
import pathlib

# Must be set BEFORE scipy is imported: it is read at scipy import time, so setting it
# afterwards (as this module used to) left the opt-in inert.
os.environ["SCIPY_ARRAY_API"] = "1"   # opt-in to JAX array support in scipy

import numpy as np
from scipy.interpolate import interpn
import stpsf
import jax
import jax.numpy as jnp
from hydrabflow.registry import register_augmentation
from hydrabflow.simulators._protoplan_spec import DIST_FID_PC

#: Small reference tables shipped with the repo: the McClure (2009) extinction law and the real
#: disks' SED error bars the wavelength-binned SED noise is calibrated from.  Resolved from
#: `__file__`, not the cwd.
ASSETS_DIR = pathlib.Path(__file__).resolve().parents[3] / "assets" / "protoplan"
DEFAULT_EXTINCTION_LAW_PATH = ASSETS_DIR / "extinction_law_mcclure09.txt"
DEFAULT_SED_DATA_PREFIX = str(ASSETS_DIR / "seddata_")
#: The five real disks whose error bars calibrate the SED noise statistics.  Unused when
#: `sed_sigma_jy` is given.
DEFAULT_SED_DISKS = ("saucer", "oph163131", "esoha574", "hvtauc", "lkha263c")

# FWHM = 2 * sqrt(2 * ln 2) * sigma = 2.354820...
# Named because the old code used sigma = FWHM / 2, which made every ALMA beam 17.7%
# wider than requested while the beam solid angle used for the Jy/beam -> MJy/sr
# conversion still assumed the nominal FWHM.
FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))

ARCSEC_PER_RAD = 206_265.0



def compute_sed_bin_statistics(disks, sed_path, bin_edges):
    """
    Compute per-bin log10(flux_err) statistics across all disks.

    Returns
    -------
    sed_bin_means : (K,)   mean   of log10(flux_err) per populated bin
    sed_bin_stds  : (K,)   std    of log10(flux_err) per populated bin
    used_centers  : (K,)   bin centers for populated bins
    """
    all_wav, all_err = [], []
    for d in disks:
        data = np.loadtxt(sed_path + d + '.txt')
        all_wav.append(data[:, 0])
        all_err.append(data[:, 2])
    total_wavelength = np.concatenate(all_wav)
    total_fluxerror  = np.concatenate(all_err)

    n_bins = len(bin_edges) - 1
    bin_centers     = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    bin_centers[0]  = bin_edges[0]
    bin_centers[-1] = bin_edges[-1]

    bin_idx = np.digitize(total_wavelength, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    sed_bin_means, sed_bin_stds, used_centers = [], [], []
    for i in range(n_bins):
        vals = total_fluxerror[bin_idx == i]
        if len(vals) == 0:
            continue
        log_vals = np.log10(vals[vals > 0])
        if len(log_vals) == 0:
            continue
        sed_bin_means.append(np.mean(log_vals))
        sed_bin_stds.append(np.std(log_vals, ddof=1) if len(log_vals) > 1 else 0.0)
        used_centers.append(bin_centers[i])

    return np.array(sed_bin_means), np.array(sed_bin_stds), np.array(used_centers)

class AugmentationsClass:
    """
    Parameters
    ----------
    px_arcsec_mod : float
        Pixel scale of the RT model images [arcsec/px].  Fixed by the
        simulation code (default 0.01).
    fov_arcsec_mod : float
        Field-of-view of the RT model images [arcsec].  Fixed by the
        simulation code (default 3.0).
    dist_fid_pc : float
        Fiducial distance at which the RT models were run [pc] (default 140).
    dist_pc : float
        Target source distance [pc] used to rescale fluxes and pixel scales.
    rot_range_deg : (float, float)
        Closed interval [low, high] from which one rotation angle (deg, CCW)
        is sampled independently per image.  Use (0.0, 360.0) for a uniform
        prior over all orientations (default).  Pass (θ, θ) to fix a single
        angle for the whole batch.  The sampled angles are stored in the
        returned batch under the key 'rot_deg' as a (N,) array.
    jwst_channel : int
        Index of the JWST channel in the C axis of batch['im_jy'] (default 0).
    jwst_lam_um : float
        Central wavelength for the stpsf PSF calculation [µm] (default 3.9).
    jwst_obs_px_arcsec : float
        Observed pixel scale for the JWST resampled image [arcsec/px]
        (default 0.1, the NIRSpec IFU detector sampling).
    jwst_noise_mjy_sr : [lo, hi]
        Uniform prior on the per-image JWST 1-σ noise [MJy/sr].  Default
        [0.15, 3.80] brackets all four disks measured by
        `measure_obs_setups.py` (0.26 / 0.54 / 0.98 / 2.67); the previous
        [0.5, 1.0] excluded three of them.
    alma_channels : sequence of int
        Indices of the ALMA channels in the C axis (default (1, 2, 3), i.e.
        450 / 880 / 1300 µm).
    alma_obs_px_arcsec : float or None
        FIXED observed pixel scale for every ALMA channel [arcsec/px], shared across the
        whole randomized beam prior so array shapes stay static under jax.jit.  None
        (default) auto-derives it as (smallest allowed beam major FWHM) / 7, matching
        model_to_obs.ipynb's `obs_px_arcsec = alma_bm[0] / 7` convention applied to the
        worst case in the prior, so every possible beam draw is sampled at least as
        finely as the notebook's reference beam.  With the default
        `alma_beam_fwhm_maj_range` this works out to 0.09/7 ~ 0.0129" (~234x234 grid on
        [-1.5, +1.5], vs. 106x106 for the notebook's fixed 0.2" beam at the same
        convention) — pass an explicit float to trade fidelity for a smaller/cheaper grid.
    alma_beam_fwhm_maj_range : sequence of (lo, hi), one per ALMA channel
        Uniform prior on the beam major-axis FWHM [arcsec].  Ignored when
        `alma_beam_theta_range` is given.
    alma_beam_theta_range : sequence of (lo, hi), one per ALMA channel, or None
        Uniform prior on the geometric-mean FWHM θ = sqrt(maj·min) [arcsec].  Given, it
        replaces `alma_beam_fwhm_maj_range`: maj = θ/sqrt(q), min = θ·sqrt(q).  This is
        the parametrisation the measured beams are tight in — an array configuration sets
        the resolution θ, while the axis ratio q is the uv-coverage's shape — so a band's
        box can be narrowed without also constraining its elongation.  None (default)
        keeps the major-axis parametrisation, so archived run configs reload unchanged.
    alma_beam_axis_ratio_range : (lo, hi), or one (lo, hi) per ALMA channel
        Uniform prior on minor/major, so the minor axis is always ≤ major.  A single pair
        is broadcast to every band.
    alma_beam_pa_deg_range : (lo, hi)
        Uniform prior on the beam position angle [deg], measured CCW from
        the image row axis.
    alma_noise_jy_beam_range : sequence of (lo, hi), one per ALMA channel
        LOG-uniform prior on the noise [Jy/beam] — it spans decades.
    n_beam_groups : int
        Independent beam/noise draws per batch (default 8).  Per-image draws
        would need one convolution kernel per image; a single per-batch draw
        would give the network no within-batch contrast on the conditions.
    randomize_alma_setup : bool
        When False, use the fixed `alma_beams` / `alma_noises_jy_beam`
        instead of sampling — needed to re-simulate one specific source
        (a posterior-predictive check, say) with its measured beams.
    alma_beams : sequence of (maj, min, pa)
        Fixed beam per ALMA channel: major [arcsec], minor [arcsec], PA [deg].
        Only used when `randomize_alma_setup=False`.
    alma_noises_jy_beam : sequence of float
        Fixed 1-σ noise per ALMA channel [Jy/beam].  Only used when
        `randomize_alma_setup=False`.
    has_jwst : bool
        Set False when the batch's channel axis holds no JWST channel at all (e.g. an
        ALMA-only per-modality emulator's clean output) — skips the JWST PSF entirely
        (including the stpsf calculation in `__init__`) instead of misapplying it to
        `jwst_channel`'s index in a channel layout that no longer has one.
    antialias_resampling : bool
        Box-average the model image over each output pixel before sampling
        it (default True), so decimation behaves like a pixel-integrating
        detector rather than point-sampling a 10× finer grid.
    random_flip : bool
        Apply a random mirror flip alongside the rotation (default True), so
        the training set covers the sky parity a real observation
        introduces via RA_FLIP.  Stored in the batch as 'sky_flip' (±1).
    obs_half_extent : float
        Half-width of the observed sky window [arcsec] (default 1.5).
    seed : int
        Seed for the JAX PRNG key (default 42).

    Keys added to the batch
    ----------------------
    Beyond the resampled images and SED, the pipeline exports the *observing
    setup it drew*, so the adapter can hand it to the network as inference
    conditions: 'rot_deg', 'sky_flip', 'sigma_jwst', and per ALMA channel j
    'beam_maj_j', 'beam_min_j', 'beam_pa_j', 'sigma_alma_j'.
    """

    # ── The observing-setup priors, as named constants ─────────────────────────
    # These are the distributions the trained networks actually saw, so inference code has
    # to read them to check whether a measured real value is inside the training support,
    # and to reproduce the ALMA obs-grid the network was trained on.
    #
    # They are class attributes rather than only `__init__` defaults because the inference
    # scripts used to recover them with
    # `inspect.signature(AugmentationsClass.__init__).parameters[...].default` — which works,
    # but makes a signature default part of the public API, so reordering or renaming a
    # keyword silently changes what another module believes the prior is.  `__init__` defaults
    # to these names, so the two cannot disagree.
    #
    # Ranges bracket the values measured from `data/obs_data/imdata_*.pkl` (see
    # measured from the real maps' headers) with generous padding: only 2-4 disks constrain
    # each band, so these are order-of-magnitude bounds, not hard limits.
    DEFAULT_ALMA_BEAM_FWHM_MAJ_RANGE = (
        (0.09, 0.30),    # 450 um  (measured 0.174, 0.181)
        (0.11, 0.75),    # 880 um  (measured 0.210, 0.467, 0.497)
        (0.09, 0.75),    # 1300 um (measured 0.177, 0.210, 0.220, 0.536)
    )
    # Minor axis is drawn as (axis_ratio x major) so that min <= maj always holds.
    # Measured ratios span 0.59-0.96 across all disks and bands.
    DEFAULT_ALMA_BEAM_AXIS_RATIO_RANGE = (0.50, 1.00)
    # Per-channel noise prior [Jy/beam], sampled LOG-uniformly (it spans decades).
    DEFAULT_ALMA_NOISE_JY_BEAM_RANGE = (
        (1.5e-4, 2.8e-3),   # 450 um  (measured 5.9e-4, 7.1e-4)
        (2.1e-5, 1.7e-3),   # 880 um  (measured 8.5e-5, 2.2e-4, 4.3e-4)
        (8.1e-6, 3.4e-4),   # 1300 um (measured 3.2e-5, 3.6e-5, 4.2e-5, 8.5e-5)
    )
    # Uniform prior on the per-image JWST 1-sigma noise [MJy/sr].  Brackets all four disks
    # measured by `measure_obs_setups.py` (0.26 / 0.54 / 0.98 / 2.67); an earlier
    # [0.5, 1.0] excluded three of them.
    DEFAULT_JWST_NOISE_MJY_SR = (0.15, 3.80)

    def __init__(
        self,
        # ── RT model grid (fixed by simulation code) ──────────────────────
        px_arcsec_mod: float = 0.01,
        fov_arcsec_mod: float = 3.0,
        dist_fid_pc: float = DIST_FID_PC,
        # ── Target distance ───────────────────────────────────────────────
        # Defaults to the fiducial, i.e. no rescaling.  A real source is almost never at
        # 140 pc; set `dist_pc` to the real source's.
        dist_pc: float = DIST_FID_PC,
        # ── Disk orientation ──────────────────────────────────────────────
        rot_range_deg: tuple[float, float] = (0.0, 360.0),
        # ── Extinction (one multiplicative scalar per channel; None → skip)
        # Either a fixed magnitude (float) or a [min, max] range, in which case one A_V is
        # drawn uniformly *per disk* in the batch.
        av: float | tuple[float, float] = 4.0,
        # ── JWST ──────────────────────────────────────────────────────────
        jwst_channel: int = 0,
        jwst_lam_um: float = 3.9,
        jwst_obs_px_arcsec: float = 0.1,
        jwst_noise_mjy_sr: tuple[float, float] = DEFAULT_JWST_NOISE_MJY_SR,
        # ── ALMA ──────────────────────────────────────────────────────────
        alma_channels: tuple[int, ...] = (1, 2, 3),
        # Observed pixel scale is a FIXED constant shared by every ALMA channel, since the
        # beam is randomized per batch-group and array shapes must stay static for
        # jax.jit.  None -> auto-derive as (smallest allowed beam major FWHM) / 7,
        # matching model_to_obs.ipynb's obs_px = alma_bm[0] / 7 convention for the
        # worst case in the prior (see __init__ below).
        alma_obs_px_arcsec: float | None = None,
        # Per-channel beam major-axis FWHM prior [arcsec], ordered as alma_channels
        # (450 / 880 / 1300 um).  See the class constant for provenance.
        alma_beam_fwhm_maj_range: tuple[tuple[float, float], ...] =
            DEFAULT_ALMA_BEAM_FWHM_MAJ_RANGE,
        # Per-channel prior on the *geometric-mean* FWHM theta = sqrt(maj * min) [arcsec].
        # When given it replaces `alma_beam_fwhm_maj_range`: maj = theta / sqrt(q),
        # min = theta * sqrt(q).  None keeps the major-axis parametrisation.
        alma_beam_theta_range: tuple[tuple[float, float], ...] | None = None,
        # Either one (lo, hi) shared by every band, or one pair per band.
        alma_beam_axis_ratio_range = DEFAULT_ALMA_BEAM_AXIS_RATIO_RANGE,
        # Beam position angle is unconstrained on the sky.
        alma_beam_pa_deg_range: tuple[float, float] = (0.0, 180.0),
        alma_noise_jy_beam_range: tuple[tuple[float, float], ...] =
            DEFAULT_ALMA_NOISE_JY_BEAM_RANGE,
        # How many independent beam/noise draws per batch.  One draw per *image* would
        # need one convolution kernel per image (~0.6 GB at batch 1024); one draw per
        # *batch* would give the network no within-batch contrast on the conditions.
        # Groups are the compromise: G kernels per channel, G-fold contrast.
        n_beam_groups: int = 8,
        # Set False to use a single fixed setup instead of randomizing — needed to
        # re-simulate a specific source with its measured beams.
        randomize_alma_setup: bool = True,
        alma_beams: tuple[tuple[float, float, float], ...] = (
            (0.2, 0.1, 0.0),
            (0.2, 0.1, 0.0),
            (0.2, 0.1, 0.0),
        ),
        alma_noises_jy_beam: tuple[float, ...] = (8e-5, 8e-5, 8e-5),
        # Fixed per-bin SED 1-sigma [Jy] on the full model grid, or None for the
        # wavelength-binned log-normal drawn from real disks' error bars.  See
        # `apply_sed_noise`; give it a real source's per-bin error bars to reproduce that source.
        sed_sigma_jy=None,
        # Set False to build a JWST-less instance over an ALMA-only (or otherwise
        # JWST-free) channel set — needed for the per-modality emulator MCMC path,
        # where the `alma` emulator's target has no channel 0 to run the (expensive,
        # unrelated) JWST PSF through.  Default True preserves every existing caller,
        # which always has channel 0 = JWST.
        has_jwst: bool = True,
        # ── Anti-aliasing ─────────────────────────────────────────────────
        # Average the model image over each output pixel before sampling it, so the
        # decimation matches a pixel-integrating detector instead of point-sampling an
        # (aliased) 10x-finer grid.
        antialias_resampling: bool = True,
        # ── Sky-parity augmentation ───────────────────────────────────────
        # Random mirror flip.  Rotation alone cannot generate a parity flip, but
        # a real observation is put into sky convention by its own flip before inference.
        random_flip: bool = True,
        # ── Observed field extent (symmetric, arcsec) ─────────────────────
        obs_half_extent: float = 1.5,
        # ── Reproducible random state ─────────────────────────────────────
        seed: int = 42,
    ) -> None:

        # ── Single random generator (all noise draws flow through here) ───
        self.key = jax.random.PRNGKey(seed)

        # ── Derived grid / distance quantities ────────────────────────────
        self.dist_scale    = dist_fid_pc / dist_pc      # linear scale factor
        self.dist_scale_sq = self.dist_scale ** 2       # flux rescaling (∝ 1/d²)
        self.px_arcsec_mod = px_arcsec_mod
        self.fov_arcsec_mod = fov_arcsec_mod
        self.px_arcsec     = px_arcsec_mod * self.dist_scale   # model px at target dist
        self.fov_arcsec    = fov_arcsec_mod * self.dist_scale  # model FOV at target dist
        self.obs_half_extent = obs_half_extent

        # ── Misc observation settings ─────────────────────────────────────
        self.rot_range_deg = (float(rot_range_deg[0]), float(rot_range_deg[1]))
        self.av = (tuple(map(float, av)) if isinstance(av, (list, tuple)) else float(av))

        # ── JWST settings ─────────────────────────────────────────────────
        self.has_jwst           = bool(has_jwst)
        self.jwst_channel       = jwst_channel
        self.jwst_obs_px_arcsec = jwst_obs_px_arcsec
        self.jwst_noise_mjy_sr  = jwst_noise_mjy_sr

        # float32 array, not a list: it is broadcast against the (N, L) SED inside a jitted
        # function, and a Python list would be re-traced as a new constant each call.
        self.sed_sigma_jy = (None if sed_sigma_jy is None
                             else np.asarray(sed_sigma_jy, dtype=np.float32).reshape(-1))

        # ── ALMA settings ─────────────────────────────────────────────────
        self.alma_channels       = list(alma_channels)
        self.alma_beams          = list(alma_beams)         # [(maj, min, pa), …]
        self.alma_noises_jy_beam = list(alma_noises_jy_beam)
        self.alma_beam_fwhm_maj_range   = [tuple(map(float, r))
                                           for r in alma_beam_fwhm_maj_range]
        self.alma_beam_theta_range = (
            None if alma_beam_theta_range is None
            else [tuple(map(float, r)) for r in alma_beam_theta_range])
        # One (lo, hi) is shorthand for "the same range in every band"; a list of pairs sets
        # it per band, which the measured elongations need (they differ band to band).
        self.alma_beam_axis_ratio_range = (
            [tuple(map(float, r)) for r in alma_beam_axis_ratio_range]
            if np.ndim(alma_beam_axis_ratio_range) == 2
            else [tuple(map(float, alma_beam_axis_ratio_range))] * len(self.alma_channels))
        # The major-axis span the prior can actually reach, whichever parametrisation is in
        # use.  Both the convolution kernel sizes below and the observed pixel scale derive
        # from it, and both fail *silently* when it is wrong (a truncated beam / the wrong
        # grid), so it is computed once here rather than read off the raw knob.
        self._maj_lo_hi = self.effective_maj_range(
            self.alma_beam_fwhm_maj_range, self.alma_beam_theta_range,
            self.alma_beam_axis_ratio_range)
        if alma_obs_px_arcsec is None:
            alma_obs_px_arcsec = self.default_alma_obs_px_arcsec(self._maj_lo_hi)
        self.alma_obs_px_arcsec  = float(alma_obs_px_arcsec)
        self.alma_beam_pa_deg_range     = tuple(map(float, alma_beam_pa_deg_range))
        self.alma_noise_jy_beam_range   = [tuple(map(float, r))
                                           for r in alma_noise_jy_beam_range]
        self.n_beam_groups        = int(n_beam_groups)
        self.randomize_alma_setup = bool(randomize_alma_setup)
        self.antialias_resampling = bool(antialias_resampling)
        self.random_flip          = bool(random_flip)

        n_ch = len(self.alma_channels)
        if len(self.alma_beam_fwhm_maj_range) != n_ch:
            raise ValueError(
                f"alma_beam_fwhm_maj_range has {len(self.alma_beam_fwhm_maj_range)} "
                f"entries but there are {n_ch} ALMA channels"
            )
        if len(self.alma_noise_jy_beam_range) != n_ch:
            raise ValueError(
                f"alma_noise_jy_beam_range has {len(self.alma_noise_jy_beam_range)} "
                f"entries but there are {n_ch} ALMA channels"
            )
        if len(self.alma_beam_axis_ratio_range) != n_ch:
            raise ValueError(
                f"alma_beam_axis_ratio_range has {len(self.alma_beam_axis_ratio_range)} "
                f"entries but there are {n_ch} ALMA channels"
            )
        if (self.alma_beam_theta_range is not None
                and len(self.alma_beam_theta_range) != n_ch):
            raise ValueError(
                f"alma_beam_theta_range has {len(self.alma_beam_theta_range)} "
                f"entries but there are {n_ch} ALMA channels"
            )

        # ── PSF cache (keyed by n_pix so the class handles any image size)
        self._jwst_psf_cache: dict[int, np.ndarray] = {}

        # ── Precompute the raw stpsf PSF (done once) ──────────────────────
        # Skip the (expensive) stpsf calculation entirely for a JWST-less instance —
        # it would never be read, since apply_psf_and_convolve/apply_resampling only
        # touch it when has_jwst is True.
        if self.has_jwst:
            self._init_jwst_psf(jwst_lam_um)

        # Absolute, via paths.py: this used to be a bare relative path, so the class
        # silently required the repo root as the working directory.
        ext_lam, ext_loav, ext_hiav = np.loadtxt(
            DEFAULT_EXTINCTION_LAW_PATH, skiprows=21).T # um, alam/ak for low ak, alam/ak for high ak
        # Per-magnitude-of-A_V exponent: A_lam = (av / 7.75) * ext_hiav and the correction is
        # 10**(-A_lam/2.5), so factor `av` out and raise to it later.  That is what lets A_V vary
        # per disk without re-interpolating the 1532-point table on every batch.
        ext_exp = -(ext_hiav / 7.75) / 2.5
        # jnp.interp (linear, 1D) instead of scipy's CubicSpline: verified to agree to
        # <=0.9% relative error against the 1532-point table (tests/test_forward_model.py),
        # and this is host-side numpy at init, never inside a jitted per-batch path.
        # left/right=nan reproduces CubicSpline(extrapolate=False)'s out-of-domain NaN.
        _ext_lam = jnp.asarray(ext_lam, dtype=jnp.float32)
        _ext_exp = jnp.asarray(ext_exp, dtype=jnp.float32)
        #: log10 correction *per magnitude of A_V*, interpolated onto arbitrary wavelengths.
        self.i_ext_exp = lambda xq: jnp.interp(
            jnp.asarray(xq), _ext_lam, _ext_exp, left=jnp.nan, right=jnp.nan
        )


        self.disks    = list(DEFAULT_SED_DISKS)
        self.sed_path = DEFAULT_SED_DATA_PREFIX
        self.bin_edges = np.array([
            5.0400001e-01, 6.3300002e-01, 9.9500000e-01,
            1.4390000e+00, 1.9100001e+00, 2.8850000e+00,
            3.7500000e+00, 4.1999998e+00, 5.1500001e+00,
            6.9000001e+00, 9.8000002e+00, 1.7799999e+01,
            4.7000000e+01, 8.5000000e+01, 1.3000000e+02,
            2.5500000e+02, 4.0000000e+02, 6.6500000e+02,
            1.0900000e+03, 1.3000000e+03,
        ])
        # Only the binned log-normal branch needs the real disks' error bars; a run that
        # supplies a fixed per-bin sigma needs none of those text files.
        if sed_sigma_jy is None:
            self.sed_bin_means, self.sed_bin_stds, self.used_centers = \
                compute_sed_bin_statistics(self.disks, self.sed_path, self.bin_edges)
        else:
            self.sed_bin_means = self.sed_bin_stds = self.used_centers = None

        # ── Pre-compute observation grid axes (constant across all batches) ───
        # np.linspace, not np.arange: the old `np.arange(-E, E + 0.1, obs_px)` overshot
        # to +1.5857" whenever obs_px did not divide 0.1, which put 3 rows/cols outside
        # the model grid (silently extrapolated by the interpolator) and left the phase
        # centre 1.5 px off the array centre.  linspace pins both endpoints exactly.
        E = self.obs_half_extent
        _jwst_ax = jnp.asarray(self._obs_axis(E, self.jwst_obs_px_arcsec))
        self._jwst_obs_axes: tuple[jnp.ndarray, jnp.ndarray] = (_jwst_ax, _jwst_ax)

        # All ALMA channels share one fixed observed pixel scale.
        _alma_ax = jnp.asarray(self._obs_axis(E, self.alma_obs_px_arcsec))
        self._alma_obs_axis: tuple[jnp.ndarray, jnp.ndarray] = (_alma_ax, _alma_ax)
        self.n_alma_obs_px = int(_alma_ax.shape[0])

        # ── Convolution kernel sizes, from the widest beam in the prior ───────
        # Sizes must be static for jax.jit and ODD, so that fftconvolve(mode="same")
        # crops at exactly (K-1)/2 and introduces no sub-pixel shift.  (The old
        # even-sized 300x300 kernels shifted the image by half a model pixel.)
        # default=0.0 -> ksize floors at _odd_kernel_size's minimum of 3: an ALMA-less
        # instance (alma_channels=(), e.g. the per-modality JWST-only MCMC forward
        # model) never reads these, but __init__ still needs them to not crash.
        max_fwhm_maj = max((r[1] for r in self._maj_lo_hi), default=0.0)
        self._alma_beam_ksize = self._odd_kernel_size(
            8.0 * max_fwhm_maj / FWHM_PER_SIGMA / self.px_arcsec_mod
        )
        self._alma_noise_ksize = self._odd_kernel_size(
            8.0 * max_fwhm_maj / FWHM_PER_SIGMA / self.alma_obs_px_arcsec
        )

        # Per-group beam/noise draws for the current batch.  Written by
        # apply_psf_and_convolve and consumed by apply_noise, which must therefore run
        # after it (as __call__ does).  None until the first draw.
        self._alma_setup: list[dict] | None = None

        # Lazy-precomputed constants (populated on first augmentation call)
        self._ext_exp_im  = None  # log10 extinction correction per mag, per image channel
        self._ext_exp_sed = None  # log10 extinction correction per mag, per SED wavelength
        self._log_sig_mu            = None  # SED noise: mean log10(sigma) per wavelength bin
        self._log_sig_std           = None  # SED noise: std  log10(sigma) per wavelength bin

        # Anti-aliasing box filters, keyed by (n_model_px, obs_px) — tiny and reusable.
        self._boxcar_cache: dict[tuple, jnp.ndarray] = {}

    # ══════════════════════════════════════════════════════════════════════
    # Private helpers
    # ══════════════════════════════════════════════════════════════════════

    def _split_key(self):
        """Split the internal PRNG key, update self.key, return a subkey."""
        self.key, subkey = jax.random.split(self.key)
        return subkey

    def _model_coords(self, n_pix: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        """1-D arcsec coordinate axes for the (n_pix × n_pix) model image."""
        half = self.fov_arcsec / 2.0
        x = jnp.linspace(-half, half, n_pix, endpoint=True)
        return x, x.copy()

    @staticmethod
    def effective_maj_range(alma_beam_fwhm_maj_range, alma_beam_theta_range,
                            alma_beam_axis_ratio_range) -> list[tuple[float, float]]:
        """
        Per band, the (lo, hi) major-axis FWHM the beam prior can actually draw [arcsec].

        With the theta parametrisation the major axis is a *derived* quantity — maj =
        theta / sqrt(q) — so its extremes pair the extreme theta with the opposite extreme
        of the axis ratio.  `alma_beam_axis_ratio_range` must already be per band here.
        """
        if alma_beam_theta_range is None:
            return [tuple(map(float, r)) for r in alma_beam_fwhm_maj_range]
        return [(t_lo / np.sqrt(q_hi), t_hi / np.sqrt(q_lo))
                for (t_lo, t_hi), (q_lo, q_hi)
                in zip(alma_beam_theta_range, alma_beam_axis_ratio_range)]

    @staticmethod
    def default_alma_obs_px_arcsec(alma_beam_fwhm_maj_range) -> float:
        """
        Derive the shared ALMA observed pixel scale from a beam-major-FWHM prior.

        model_to_obs.ipynb uses obs_px_arcsec = alma_bm[0] / 7 for its one fixed beam.
        Applied here to the smallest major-axis FWHM the prior can draw, across all
        channels, so every possible randomized beam ends up sampled at least as finely as
        the notebook's convention.  A staticmethod (not instance state) so callers that
        need the value without paying for a full AugmentationsClass construction — e.g.
        the grid a real observation must be regridded onto — cannot drift from it.
        """
        return min(lo for lo, _ in alma_beam_fwhm_maj_range) / 7.0

    @staticmethod
    def _obs_axis(half_extent: float, px_arcsec: float) -> np.ndarray:
        """
        Symmetric observed-grid axis spanning exactly [-half_extent, +half_extent].

        The number of pixels is chosen so the spacing is as close to `px_arcsec` as an
        exact endpoint-to-endpoint fit allows.  Using linspace (rather than arange with a
        fudged upper bound) guarantees three things the old code got wrong: the axis never
        leaves the model grid, it is exactly centred on 0, and the spacing is exact.
        """
        n = int(round(2.0 * half_extent / px_arcsec)) + 1
        return np.linspace(-half_extent, half_extent, n, endpoint=True)

    @staticmethod
    def _odd_kernel_size(width_px: float) -> int:
        """
        Smallest ODD kernel size >= width_px.

        Odd matters: jax.scipy.signal.fftconvolve(mode="same") crops starting at
        (K - 1) // 2, so an odd kernel whose peak sits at (K - 1) / 2 introduces zero
        shift, while an even kernel shifts the result by half a pixel.
        """
        k = int(np.ceil(width_px))
        if k % 2 == 0:
            k += 1
        return max(k, 3)

    def _boxcar_2d(self, n_model_px: int, obs_px_arcsec: float) -> jnp.ndarray:
        """
        Separable box filter that averages the model image over one *output* pixel.

        Resampling by point-sampling a 10x-finer grid aliases: the JWST PSF at 3.9 um is
        only ~1.3 output pixels across, so single-pixel intensities in the sims carry
        high-frequency structure a real pixel-integrating detector would have averaged
        away.  Pre-convolving with this box makes the decimation an area average.

        The box width is generally fractional (0.1 / 0.010033 = 9.967 model px for JWST),
        so edge taps get partial weight — this keeps the filter exactly the requested
        width *and* exactly centred, which an integer-width box could not do.
        """
        key = (n_model_px, round(obs_px_arcsec, 9))
        if key in self._boxcar_cache:
            return self._boxcar_cache[key]

        src_x, _ = self._model_coords(n_model_px)
        model_px = float(np.asarray(src_x)[1] - np.asarray(src_x)[0])
        width = obs_px_arcsec / model_px          # box width in model pixels

        n = self._odd_kernel_size(width + 1.0)
        centres = np.arange(n, dtype=np.float64) - (n - 1) / 2.0
        half = width / 2.0
        # Overlap of pixel [c - 0.5, c + 0.5] with the box [-half, +half].
        lo = np.maximum(centres - 0.5, -half)
        hi = np.minimum(centres + 0.5, half)
        w = np.maximum(hi - lo, 0.0)
        w /= w.sum()

        kern = jnp.asarray(np.outer(w, w).astype(np.float32))
        self._boxcar_cache[key] = kern
        return kern

    # ── JWST PSF ──────────────────────────────────────────────────────────

    def _init_jwst_psf(self, lam_um: float) -> None:
        """
        Compute the NIRSpec IFU PSF via stpsf and store the raw array + its
        arcsec grid.  The PSF is regridded to the model pixel scale lazily
        (and cached) the first time a given n_pix is encountered.
        """
        nrs           = stpsf.NIRSpec()
        nrs.mode      = "IFU"
        nrs.disperser = "G395H"
        nrs.filter    = "F290LP"
        cube = nrs.calc_datacube(
            [lam_um * 1e-6], fov_arcsec=self.fov_arcsec, oversample=11
        )

        hdr = cube[2].header
        self._psf_raw    = np.squeeze(cube[2].data).astype(np.float64)
        fov_s            = hdr["FOV"]
        self._psf_grid_x = np.linspace(-fov_s / 2, fov_s / 2, hdr["NAXIS1"], endpoint=True)
        self._psf_grid_y = np.linspace(-fov_s / 2, fov_s / 2, hdr["NAXIS2"], endpoint=True)

    def _get_jwst_kernel(self, n_pix: int) -> np.ndarray:
        """
        Regrid the stpsf PSF onto an ODD grid at the model pixel scale, normalised.

        Two changes from the original, both needed for photometric correctness:

        1. The kernel is built on its own odd-sized axis centred exactly on 0, rather
           than on the even 300-px model grid.  A convolution kernel only needs the right
           *pixel scale*, not the same extent — and an even grid put the PSF centroid at
           index 149.5, which fftconvolve(mode="same") turned into a half-pixel image
           shift.
        2. The kernel is normalised to unit sum.  Previously it summed to 0.856 (0.953
           encircled-energy truncation at the 3" FOV x 0.894 oversampled-to-model pixel
           area ratio), so the JWST channel silently lost 14.4% of its flux while
           `sigma_jwst` was left unscaled — a 14% error in the effective S/N.

        Cached by n_pix (which fixes the model pixel scale).
        """
        if n_pix in self._jwst_psf_cache:
            return self._jwst_psf_cache[n_pix]

        src_x, _ = self._model_coords(n_pix)
        model_px = float(np.asarray(src_x)[1] - np.asarray(src_x)[0])

        # Kernel extent: as much of the PSF as the stpsf FOV actually provides, on an odd
        # grid so the peak lands on a pixel centre.
        psf_half = min(abs(self._psf_grid_x[0]), abs(self._psf_grid_x[-1]))
        k = self._odd_kernel_size(2.0 * psf_half / model_px)
        ax = (np.arange(k, dtype=np.float64) - (k - 1) / 2.0) * model_px
        mx, my = np.meshgrid(ax, ax, indexing="ij")

        kernel = interpn((self._psf_grid_x, self._psf_grid_y), self._psf_raw.T,
                         (mx, my), method="linear",
                         bounds_error=False, fill_value=None)

        kernel = np.maximum(kernel, 0.0)     # interpolation can undershoot below 0
        total  = kernel.sum()
        if total <= 0:
            raise RuntimeError("JWST PSF kernel has non-positive total flux")
        kernel /= total                      # flux-conserving normalisation

        self._jwst_psf_cache[n_pix] = kernel
        return kernel

    # ── ALMA beam ─────────────────────────────────────────────────────────

    @staticmethod
    def beam_area_sr(fwhm_maj_arcsec, fwhm_min_arcsec):
        """Solid angle [sr] of an elliptical Gaussian beam given its FWHMs [arcsec]."""
        return (np.pi * fwhm_maj_arcsec * fwhm_min_arcsec
                / (4.0 * np.log(2.0)) / ARCSEC_PER_RAD ** 2)

    @staticmethod
    @functools.partial(jax.jit, static_argnums=(3,))
    def _gaussian_beam_kernel(
        sigma_maj_px: jnp.ndarray,
        sigma_min_px: jnp.ndarray,
        pa_rad: jnp.ndarray,
        n: int,
    ) -> jnp.ndarray:
        """
        Unit-sum elliptical Gaussian kernel, built analytically from traced parameters.

        `astropy.convolution.Gaussian2DKernel` cannot accept JAX tracers, so randomizing
        the beam requires building it here.  Vectorise over a batch of beams with
        jax.vmap; `n` is static so the output shape is known at trace time.

        Convention: `pa_rad` is the position angle of the major axis measured
        counter-clockwise from the +row (axis 0) direction of the image array.  Inference
        code must convert the observed BPA into this frame.
        """
        ax = jnp.arange(n, dtype=jnp.float32) - (n - 1) / 2.0
        di, dj = jnp.meshgrid(ax, ax, indexing="ij")
        ct, st = jnp.cos(pa_rad), jnp.sin(pa_rad)
        u =  ct * di + st * dj        # along the major axis
        v = -st * di + ct * dj        # along the minor axis
        k = jnp.exp(-0.5 * ((u / sigma_maj_px) ** 2 + (v / sigma_min_px) ** 2))
        return k / jnp.sum(k)

    def _draw_alma_setup(self, n_images: int) -> list[dict]:
        """
        Draw one beam + noise level per (batch group, ALMA channel).

        Returns a list of `n_groups` dicts, each holding per-channel arrays:
            fwhm_maj, fwhm_min [arcsec], pa_deg [deg], rms_jy_beam [Jy/beam],
            sigma_mjy_sr [MJy/sr], and the (start, stop) image slice for the group.

        One draw per *image* would need one convolution kernel per image (~0.6 GB of
        kernels at batch 1024); one draw per *batch* would leave the network no
        within-batch contrast to learn the conditioning from.  Groups give both.
        """
        n_ch = len(self.alma_channels)

        if not self.randomize_alma_setup:
            # Single fixed setup replicated across the whole batch.
            maj = np.array([b[0] for b in self.alma_beams], dtype=np.float64)
            mn  = np.array([b[1] for b in self.alma_beams], dtype=np.float64)
            pa  = np.array([b[2] for b in self.alma_beams], dtype=np.float64)
            rms = np.array(self.alma_noises_jy_beam, dtype=np.float64)
            return [{
                "slice": (0, n_images),
                "fwhm_maj": maj, "fwhm_min": mn, "pa_deg": pa,
                "rms_jy_beam": rms,
                "sigma_mjy_sr": rms / self.beam_area_sr(maj, mn) / 1.0e6,
            }]

        # Use the largest group count that divides the batch evenly, so every group has
        # the same shape and jax.jit compiles the convolution once.
        n_groups = 1
        for g in range(min(self.n_beam_groups, n_images), 0, -1):
            if n_images % g == 0:
                n_groups = g
                break
        per = n_images // n_groups

        u = np.asarray(jax.random.uniform(
            self._split_key(), (n_groups, n_ch, 4), dtype=jnp.float32
        ), dtype=np.float64)

        groups = []
        for g in range(n_groups):
            maj, ratio = np.empty(n_ch), np.empty(n_ch)
            pa, rms = np.empty(n_ch), np.empty(n_ch)
            for c in range(n_ch):
                r_lo, r_hi = self.alma_beam_axis_ratio_range[c]
                ratio[c] = r_lo + u[g, c, 1] * (r_hi - r_lo)
                if self.alma_beam_theta_range is not None:
                    # Resolution and elongation are drawn independently: theta =
                    # sqrt(maj * min) is what the array configuration sets, q = min/maj
                    # is the uv-coverage's shape.  Same u draw, so the PRNG stream is
                    # identical in shape to the major-axis parametrisation.
                    t_lo, t_hi = self.alma_beam_theta_range[c]
                    theta = t_lo + u[g, c, 0] * (t_hi - t_lo)
                    maj[c] = theta / np.sqrt(ratio[c])
                else:
                    lo, hi = self.alma_beam_fwhm_maj_range[c]
                    maj[c] = lo + u[g, c, 0] * (hi - lo)
                p_lo, p_hi = self.alma_beam_pa_deg_range
                pa[c] = p_lo + u[g, c, 2] * (p_hi - p_lo)
                # Noise spans decades, so sample log-uniformly.
                n_lo, n_hi = self.alma_noise_jy_beam_range[c]
                rms[c] = 10.0 ** (np.log10(n_lo)
                                  + u[g, c, 3] * (np.log10(n_hi) - np.log10(n_lo)))
            mn = ratio * maj
            groups.append({
                "slice": (g * per, (g + 1) * per),
                "fwhm_maj": maj, "fwhm_min": mn, "pa_deg": pa,
                "rms_jy_beam": rms,
                # Jy/beam -> MJy/sr using the solid angle of the beam we actually drew,
                # so the smoothing width and the unit conversion always agree.
                "sigma_mjy_sr": rms / self.beam_area_sr(maj, mn) / 1.0e6,
            })
        return groups

    # ── Vectorised resampling ──────────────────────────────────────────────

    @staticmethod
    def _resample_batch(
        image: np.ndarray,
        src_x: np.ndarray,
        src_y: np.ndarray,
        tgt_x: np.ndarray,
        tgt_y: np.ndarray,
        fill_value: float | None,
    ) -> np.ndarray:
        """
        Resample a single (H, W, C) image to the (tgt_x, tgt_y) grid.

        Designed to be called under jax.vmap over the N axis, so `image` is
        one sample.  RegularGridInterpolator expects query points as an
        (M, 2) array, so we meshgrid the target axes, flatten to (M, 2),
        interpolate, then reshape back to (H_obs, W_obs, C).

        `fill_value` controls out-of-grid behaviour, matching model_to_obs.ipynb:
        `None` linearly extrapolates (jax's RegularGridInterpolator contract — passing
        `None` explicitly, as opposed to leaving the nan default, disables the
        out-of-bounds override) for JWST; a float (0.0) hard-fills for ALMA, since the
        notebook assumes no flux outside the simulated model FOV rather than
        extrapolating it.
        """
        H, W, C = image.shape
        H_obs, W_obs = len(tgt_x), len(tgt_y)

        # Build the (H_obs * W_obs, 2) query array expected by RegularGridInterpolator
        obs_mx, obs_my = jnp.meshgrid(tgt_x, tgt_y, indexing="ij")  # (H_obs, W_obs) each
        query_pts = jnp.stack(
            [obs_mx.ravel(), obs_my.ravel()], axis=-1
        )  # (H_obs * W_obs, 2)

        def interp_channel(im_hw):
            # im_hw: (H, W)
            # points must be strictly increasing — same contract as interpax
            interp = jax.scipy.interpolate.RegularGridInterpolator(
                points=(src_x, src_y),
                values=im_hw,
                method="linear",
                fill_value=fill_value,
            )
            flat = interp(query_pts)           # (H_obs * W_obs,)
            return flat.reshape(H_obs, W_obs)  # (H_obs, W_obs)

        # vmap over the C axis (H, W, C → C, H, W → back to H_obs, W_obs, C)
        result = jax.vmap(interp_channel)(image.transpose(2, 0, 1))  # (C, H_obs, W_obs)
        return result.transpose(1, 2, 0)                              # (H_obs, W_obs, C)


    @staticmethod
    @jax.jit
    def _batch_fftconvolve(images: jnp.ndarray, kernel: jnp.ndarray) -> jnp.ndarray:
        """JIT-compiled batched FFT convolution."""
        return jax.vmap(lambda img: jax.scipy.signal.fftconvolve(img, kernel, mode="same"))(images)

    @staticmethod
    @functools.partial(jax.jit, static_argnames=("fill_value",))
    def _batch_resample_jit(
        images: jnp.ndarray,
        src_x: jnp.ndarray,
        src_y: jnp.ndarray,
        tgt_x: jnp.ndarray,
        tgt_y: jnp.ndarray,
        fill_value: float | None = None,
    ) -> jnp.ndarray:
        """JIT-compiled batched resampling. `fill_value` is static (None or a float)."""
        return jax.vmap(
            functools.partial(AugmentationsClass._resample_batch, fill_value=fill_value),
            in_axes=(0, None, None, None, None),
        )(images, src_x, src_y, tgt_x, tgt_y)

    @staticmethod
    @jax.jit
    def _rotate_batch_jit(
        images: jnp.ndarray,
        angles_rad: jnp.ndarray,
        flip: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """
        JIT-compiled batched rotation (and optional mirror) via JAX-native
        map_coordinates.

        Builds inverse-rotation coordinate maps analytically for all (N, H, W)
        pixels at once via broadcasting — no Python loop over N or C.
        The same (N, H, W) coordinate arrays are shared across all channels
        via vmap over the C axis.

        `flip` is a (N,) array of +1 / -1.  Negating the column offset before rotating
        mirrors the image, which together with the rotation covers the full O(2) sky
        symmetry group.  Rotation alone covers only SO(2), yet a real observation
        applies an RA flip to put the real data in sky convention — a parity the training
        set could never produce.
        """
        N, H, W, C = images.shape
        cx = (H - 1) / 2.0
        cy = (W - 1) / 2.0
        cos_a = jnp.cos(angles_rad)                              # (N,)
        sin_a = jnp.sin(angles_rad)                              # (N,)
        di = jnp.arange(H, dtype=jnp.float32) - cx              # (H,)
        dj = jnp.arange(W, dtype=jnp.float32) - cy              # (W,)
        di, dj = jnp.meshgrid(di, dj, indexing="ij")            # (H, W)
        di = jnp.broadcast_to(di[None], (N, H, W))
        if flip is None:
            dj = jnp.broadcast_to(dj[None], (N, H, W))
        else:
            dj = dj[None] * flip[:, None, None].astype(dj.dtype)
        # Inverse rotation maps: (N, H, W)
        src_i = cx + cos_a[:, None, None] * di + sin_a[:, None, None] * dj
        src_j = cy - sin_a[:, None, None] * di + cos_a[:, None, None] * dj
        n_idx = jnp.broadcast_to(
            jnp.arange(N, dtype=jnp.float32)[:, None, None], (N, H, W)
        )
        coords = [n_idx, src_i, src_j]

        def _rotate_channel(channel_nhw: jnp.ndarray) -> jnp.ndarray:
            """Rotate one (N, H, W) channel slice using shared coord maps."""
            return jax.scipy.ndimage.map_coordinates(
                channel_nhw, coords, order=1, mode="constant", cval=0.0
            )

        # Transpose → (C, N, H, W), vmap over C, transpose back
        rotated = jax.vmap(_rotate_channel)(images.transpose(3, 0, 1, 2))  # (C, N, H, W)
        return rotated.transpose(1, 2, 3, 0)                               # (N, H, W, C)

    @staticmethod
    @jax.jit
    def _sed_noise_jit(
        seds_2d: jnp.ndarray,
        log_sig_mu: jnp.ndarray,
        log_sig_std: jnp.ndarray,
        key1: jnp.ndarray,
        key2: jnp.ndarray,
    ):
        """JIT-compiled log-normal SED noise (all ops fused in one XLA call)."""
        log_sigma   = log_sig_mu + jax.random.normal(key1, seds_2d.shape) * log_sig_std
        sigma       = 10.0 ** log_sigma
        sigma_sq_ln = jnp.log(1.0 + sigma ** 2 / seds_2d ** 2)
        mu_ln       = jnp.log(seds_2d) - 0.5 * sigma_sq_ln
        seds_noisy  = jnp.exp(mu_ln + jax.random.normal(key2, seds_2d.shape) * jnp.sqrt(sigma_sq_ln))
        return seds_noisy, sigma

    @staticmethod
    @jax.jit
    def _sed_noise_fixed_sigma_jit(
        seds_2d: jnp.ndarray,
        sigma_1d: jnp.ndarray,
        key: jnp.ndarray,
    ):
        """
        Log-normal SED noise at a **fixed, per-bin** sigma [Jy] instead of a drawn one.

        Same parameterization as `_sed_noise_jit` -- mean = flux, variance = sigma^2, so the
        draw cannot go negative -- but sigma is a constant of the observing setup rather than
        a per-row sample from the population of real disks' error bars.  One PRNG key, not
        two: there is no sigma to draw.

        Note the log-normal becomes strongly right-skewed where `sigma >> flux`
        (`sigma_sq_ln` grows without bound), which is the intended behaviour for a real
        absolute noise floor: a disk far below it is not measured, only bounded.
        """
        sigma       = jnp.broadcast_to(sigma_1d[jnp.newaxis, :], seds_2d.shape)
        sigma_sq_ln = jnp.log(1.0 + sigma ** 2 / seds_2d ** 2)
        mu_ln       = jnp.log(seds_2d) - 0.5 * sigma_sq_ln
        seds_noisy  = jnp.exp(
            mu_ln + jax.random.normal(key, seds_2d.shape) * jnp.sqrt(sigma_sq_ln))
        return seds_noisy, sigma

    @staticmethod
    @jax.jit
    def _log10_arrays(
        seds: jnp.ndarray,
        sed_lams: jnp.ndarray,
        sigma_sed_flux: jnp.ndarray,
    ):
        """JIT-compiled log10 compression for SED arrays (fused into one XLA call)."""
        return jnp.log10(seds), jnp.log10(sed_lams), jnp.log10(sigma_sed_flux)

    @staticmethod
    @jax.jit
    def _multi_alma_convolve(alma_imgs: jnp.ndarray, beams: jnp.ndarray) -> jnp.ndarray:
        """
        JIT-compiled batched FFT convolution for all ALMA channels in one XLA call.

        Parameters
        ----------
        alma_imgs : (N, H, W, n_alma)  — all ALMA channel slices stacked on last axis
        beams     : (n_alma, H, W)     — one beam kernel per channel

        Returns
        -------
        (N, H, W, n_alma) convolved images
        """
        def _one_channel(imgs_nhw: jnp.ndarray, beam_hw: jnp.ndarray) -> jnp.ndarray:
            return jax.vmap(
                lambda img: jax.scipy.signal.fftconvolve(img, beam_hw, mode="same")
            )(imgs_nhw)

        # vmap outer axis: n_alma (axis 3 of alma_imgs, axis 0 of beams)
        return jax.vmap(_one_channel, in_axes=(3, 0), out_axes=3)(alma_imgs, beams)

    @staticmethod
    @jax.jit
    def _apply_correlated_alma_noise(
        white_noise: jnp.ndarray,  # (B, H, W)
        beam_kernel: jnp.ndarray,  # (KH, KW)  — pre-computed, constant
        noise_std: float,
    ) -> jnp.ndarray:              # (B, H, W)
        """
        JIT-compiled correlated ALMA noise for one channel.

        Convolves i.i.d. white noise with the beam kernel so the spatial
        correlation of the noise matches the beam, then normalises to the
        target noise_std.  Defined as a @staticmethod so that the *same*
        Python function object is presented to jax.jit on every call,
        enabling proper compilation caching (unlike a closure defined inside
        a method body, which produces a new object each call and forces a
        full retrace).
        """
        def _single(wn: jnp.ndarray) -> jnp.ndarray:
            cn = jax.scipy.signal.fftconvolve(wn, beam_kernel, mode="same")
            cn = cn / jnp.std(cn)
            return cn * noise_std
        return jax.vmap(_single)(white_noise)


    # ══════════════════════════════════════════════════════════════════════
    # Public pipeline steps
    # ══════════════════════════════════════════════════════════════════════

    def preprocess(self, batch: dict) -> dict:
        """
        Three unit-conversion steps, all fully vectorised:

        1. Extinction correction — multiply each channel by its correction
           factor (broadcast over N, H, W).
        2. Distance rescaling — flux ∝ 1/d²  →  multiply by (d_fid/d)².
        3. Unit conversion — Jy/pixel  →  MJy/sr using the solid angle of
           one model pixel: Ω_px = (px_arcsec_mod)² / 206265² sr.

        The SED array (if present) is rescaled for distance and extinction too.

        Parameters
        ----------
        batch['im_jy']      : (N, H, W, C)  Jy/pixel
        batch['sed_flx_jy'] : (N, Λ)         Jy  [optional]

        Returns
        -------
        batch['im_jy']      : (N, H, W, C)  MJy/sr
        batch['sed_flx_jy'] : (N, Λ)         Jy   [optional, distance-rescaled]
        """
        images = jnp.asarray(batch["im_jy"], dtype=jnp.float32)  # (N, H, W, C) — on device
        sed    = jnp.asarray(batch["seds"],  dtype=jnp.float32)  # (N, Λ, 1)

        # The cached image variants all cover the same fixed FOV and differ only in sampling, so
        # `px_arcsec_mod` is a function of the grid -- but the PSF and beam kernels were sized from
        # it back in `__init__`, before any batch existed. Feeding a differently-sampled cache
        # without updating it is a *silent beam error*: the images still come out in the right
        # units (the solid angle below is derived from the array's own H), just convolved with a
        # kernel of the wrong physical width. Checked here, once, on the first batch.
        implied_px = self.fov_arcsec / (images.shape[1] - 1)
        if abs(implied_px / self.px_arcsec - 1.0) > 0.01:
            raise ValueError(
                f"px_arcsec_mod={self.px_arcsec_mod:.6g}\" implies a "
                f"{round(self.fov_arcsec_mod / self.px_arcsec_mod) + 1}-pixel grid, but the batch's "
                f"images are {images.shape[1]}x{images.shape[2]} over the same "
                f"{self.fov_arcsec_mod}\" field of view (px={implied_px / self.dist_scale:.6g}\"). "
                "The PSF/beam kernels are sized from px_arcsec_mod, so this is a silent beam error."
            )

        # Lazily precompute the wavelength half of the extinction law (constant across batches;
        # the A_V scaling below is what may differ per disk).
        if self._ext_exp_im is None:
            self._ext_exp_im  = jnp.asarray(
                self.i_ext_exp(batch['im_lams'][0]), dtype=jnp.float32
            )
            self._ext_exp_sed = jnp.asarray(
                self.i_ext_exp(batch['sed_lams'][0, :, 0]), dtype=jnp.float32
            )

        # 1. Extinction correction (stays on device).  A scalar `av` is one correction for the
        #    whole batch; an (lo, hi) `av` draws one A_V per disk, so the correction gains a
        #    leading batch axis.  The drawn value is exported as `batch['av']` below: a run that
        #    lists it in `adapter.inference_conditions` conditions on it, one that does not
        #    marginalises over it, like the ALMA beam when `randomize_alma_setup` is on.
        if isinstance(self.av, tuple):
            av = jax.random.uniform(
                self._split_key(), (images.shape[0], 1),
                minval=self.av[0], maxval=self.av[1], dtype=jnp.float32
            )
            av_per_row = av[:, 0]
            images = images * (10.0 ** (self._ext_exp_im[jnp.newaxis, :] * av))[
                :, jnp.newaxis, jnp.newaxis, :]
            sed    = sed    * (10.0 ** (self._ext_exp_sed[jnp.newaxis, :] * av))[
                :, :, jnp.newaxis]
        else:
            images = images * (10.0 ** (self._ext_exp_im * self.av))[
                jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
            sed    = sed    * (10.0 ** (self._ext_exp_sed * self.av))[
                jnp.newaxis, :, jnp.newaxis]
            av_per_row = jnp.full((images.shape[0],), self.av, dtype=jnp.float32)

        # 2. Distance rescaling — flux ∝ 1/d².  A no-op at the default dist_pc =
        #    dist_fid_pc, but previously `dist_scale_sq` was computed and never applied
        #    while `px_arcsec` / `fov_arcsec` *were* rescaled — so changing dist_pc would
        #    silently have decoupled the geometry from the photometry.
        #
        #    What the two steps do together is worth stating, because it is the opposite of
        #    what "rescale the fluxes" suggests.  Step 3 divides by Omega_px, which is
        #    derived from `fov_arcsec` and therefore also grew by dist_scale**2 — so for the
        #    *images* the two factors cancel exactly and the surface brightness in MJy/sr is
        #    distance-invariant, as it must be.  Moving a source closer makes it bigger on
        #    the sky (fov_arcsec grows), not brighter per solid angle.  The SED is in Jy,
        #    never divided by a solid angle, so it is the one output that really does scale
        #    by dist_scale**2.  Applying a flux ratio to the images *on top of* dist_pc
        #    would double-count.
        if self.dist_scale_sq != 1.0:
            images = images * self.dist_scale_sq
            sed    = sed    * self.dist_scale_sq

        # 3. Jy/pixel → MJy/sr.  Ω_px uses the *actual* model grid spacing, which is
        #    fov/(n-1) = 0.0100334", not the nominal px_arcsec_mod = 0.01" (a 0.67%
        #    error in area).
        src_x, _    = self._model_coords(images.shape[1])
        px_actual   = float(np.asarray(src_x)[1] - np.asarray(src_x)[0])
        omega_px_sr = px_actual ** 2 / ARCSEC_PER_RAD ** 2
        images      = images / omega_px_sr / 1.0e6

        new_batch          = dict(batch)
        new_batch["im_jy"] = images
        new_batch["seds"]  = sed
        # The A_V that was actually applied, one per row, so a run can condition on it.  Written
        # unconditionally: the adapter drops the key when it is not listed, and a fixed `av` is a
        # constant column rather than a missing one.
        new_batch["av"]    = av_per_row
        return new_batch

    def apply_psf_and_convolve(self, batch: dict) -> dict:
        """
        Convolve each channel with its instrument PSF — vectorised over N.

        Channel assignment
        ------------------
        JWST  (channel `jwst_channel`): stpsf NIRSpec IFU PSF, regridded to the
            model pixel scale and applied via FFT multiplication that broadcasts
            over the entire batch dimension.
        ALMA  (channels in `alma_channels`): elliptical Gaussian beam (one per
            channel), also applied via batched FFT.

        No Python loop over images is performed; the cost per channel is
        O(N · H · W · log(H · W)).
        """
        images = jnp.asarray(batch["im_jy"])           # (N, H, W, C)
        N, H, W, C = images.shape

        # JWST — single JIT call.  Kernel is odd-sized and unit-sum (see _get_jwst_kernel).
        if self.has_jwst:
            psf_jwst = jnp.asarray(self._get_jwst_kernel(H))
            images = images.at[..., self.jwst_channel].set(
                self._batch_fftconvolve(images[..., self.jwst_channel], psf_jwst)
            )

        # ── ALMA ──────────────────────────────────────────────────────────────
        valid_idxs = [i for i, ch in enumerate(self.alma_channels) if ch < C]
        valid_chs  = [self.alma_channels[i] for i in valid_idxs]

        # Draw the beam + noise setup for this batch.  Stored on `self` because
        # apply_noise needs the *same* draws to build matching noise kernels; __call__
        # guarantees the ordering.
        self._alma_setup = self._draw_alma_setup(N)

        if valid_idxs:
            ksize = self._alma_beam_ksize
            for grp in self._alma_setup:
                lo, hi = grp["slice"]
                sig_maj = jnp.asarray(
                    grp["fwhm_maj"][valid_idxs] / self.px_arcsec / FWHM_PER_SIGMA,
                    dtype=jnp.float32,
                )
                sig_min = jnp.asarray(
                    grp["fwhm_min"][valid_idxs] / self.px_arcsec / FWHM_PER_SIGMA,
                    dtype=jnp.float32,
                )
                pa_rad = jnp.asarray(
                    np.deg2rad(grp["pa_deg"][valid_idxs]), dtype=jnp.float32
                )
                beams = jax.vmap(
                    lambda a, b, c: self._gaussian_beam_kernel(a, b, c, ksize)
                )(sig_maj, sig_min, pa_rad)                 # (n_valid, k, k)

                sub = self._multi_alma_convolve(images[lo:hi][..., valid_chs], beams)
                for j, ch in enumerate(valid_chs):
                    images = images.at[lo:hi, :, :, ch].set(sub[..., j])

        # Expose the drawn setup per image so the adapter can pass it to the network as
        # inference conditions.  Without this the network cannot know what beam / noise
        # level produced the image it is looking at.
        for j, i_ch in enumerate(valid_idxs):
            maj = np.empty(N, dtype=np.float32)
            mn  = np.empty(N, dtype=np.float32)
            pa  = np.empty(N, dtype=np.float32)
            sig = np.empty(N, dtype=np.float32)
            for grp in self._alma_setup:
                lo, hi = grp["slice"]
                maj[lo:hi] = grp["fwhm_maj"][i_ch]
                mn[lo:hi]  = grp["fwhm_min"][i_ch]
                pa[lo:hi]  = grp["pa_deg"][i_ch]
                sig[lo:hi] = grp["sigma_mjy_sr"][i_ch]
            batch[f"beam_maj_{j}"]   = jnp.asarray(maj)
            batch[f"beam_min_{j}"]   = jnp.asarray(mn)
            batch[f"beam_pa_{j}"]    = jnp.asarray(pa)
            batch[f"sigma_alma_{j}"] = jnp.asarray(sig)
            # Beam PA is degenerate mod 180 (an ellipse at PA 179 and PA -1 are the
            # same), so condition on sin/cos of *twice* the angle: that is continuous
            # across the wrap and respects the degeneracy.  Raw `beam_pa_j` is kept
            # for diagnostics only.
            pa2 = np.deg2rad(2.0 * pa)
            batch[f"beam_pa_sin_{j}"] = jnp.asarray(np.sin(pa2, dtype=np.float32))
            batch[f"beam_pa_cos_{j}"] = jnp.asarray(np.cos(pa2, dtype=np.float32))

        batch["im_jy"] = images
        return batch

    def apply_rotation(self, batch: dict) -> dict:
        """
        Sample one rotation angle per image and apply them all in a single
        map_coordinates call — no Python loop over N.

        Angle sampling
        --------------
        N angles are drawn i.i.d. from Uniform(rot_range_deg[0], rot_range_deg[1]).
        If both bounds are equal the same fixed angle is used for every image.
        The sampled angles are stored in batch['rot_deg'] as a (N,) float array
        so downstream code (e.g. the posterior) can use them as labels.

        Vectorisation
        -------------
        For image n with angle θ_n, the inverse rotation (output → source) is:

            src_i = cx + cos(θ_n)·(i − cx) + sin(θ_n)·(j − cy)
            src_j = cy − sin(θ_n)·(i − cx) + cos(θ_n)·(j − cy)

        Broadcasting over (N, H, W) gives the full coordinate arrays without
        any loop.  A single map_coordinates call then interpolates all N images
        simultaneously.  The same coordinate arrays are reused for every channel
        C, so the cost is O(N · H · W) coordinate arithmetic + one
        map_coordinates call per channel.
        """
        images = jnp.asarray(batch["im_jy"])           # (N, H, W, C)
        N, H, W, C = images.shape

        # ── Sample one angle per image ────────────────────────────────────
        lo, hi = self.rot_range_deg
        if lo == hi:
            angles_deg = jnp.full(N, lo)
        else:
            # _split_key() both advances self.key and returns a fresh subkey
            angles_deg = jax.random.uniform(
                self._split_key(), (N,), minval=lo, maxval=hi
            )
        angles_rad = jnp.deg2rad(angles_deg)

        # ── Optional mirror flip (sky parity) ─────────────────────────────
        if self.random_flip:
            flip = jnp.where(
                jax.random.bernoulli(self._split_key(), 0.5, (N,)), -1.0, 1.0
            ).astype(jnp.float32)
        else:
            flip = None
            # No flip still *is* a parity, and `sky_flip` is a geometry condition the adapter
            # reads unconditionally (`_protoplan_spec.GEOMETRY_CONDITION_KEYS`). Recording +1
            # keeps a `random_flip=False` batch adapter-readable; leaving the key out made a
            # fixed-parity reference population impossible to embed.
            batch["sky_flip"] = jnp.ones(N, dtype=jnp.float32)

        # ── Rotate all N images and all C channels in one JIT-compiled call
        images = self._rotate_batch_jit(images, angles_rad, flip)

        batch["im_jy"]   = images
        batch["rot_deg"] = angles_deg
        # Circular encoding for use as an inference condition.  A raw angle in [0, 360)
        # has a discontinuity at the wrap — 359 deg and 1 deg are two degrees apart on the
        # sky but 358 apart numerically — which the network would have to waste capacity
        # learning around.  `rot_deg` is kept for diagnostics; only these are conditioned.
        batch["rot_sin"] = jnp.sin(angles_rad)
        batch["rot_cos"] = jnp.cos(angles_rad)
        if flip is not None:
            batch["sky_flip"] = flip
        return batch

    def apply_resampling(self, batch: dict) -> dict:
        """
        Resample each channel to its instrument's observed pixel grid.

        All interpolation is done via a single map_coordinates call per channel
        (vectorised over N — see _resample_batch for details).

        Because JWST and ALMA may use different pixel scales the outputs are
        stored under separate keys:

            batch['im_jy_jwst']    : (N, H_jw, W_jw)  MJy/sr
            batch['im_jy_alma_0']  : (N, H_al, W_al)  MJy/sr   ← 1st ALMA ch
            batch['im_jy_alma_1']  : ...
            ...

        The original 'im_jy' key is retained for reference.
        """
        images = jnp.asarray(batch["im_jy"])           # (N, H, W, C)
        N, H, W, C = images.shape
        src_x, src_y = self._model_coords(H)

        # JWST — box-average over the output pixel first, then sample.  Without this the
        # 10x decimation point-samples a marginally Nyquist-sampled PSF and aliases.
        if self.has_jwst:
            tx_jwst, ty_jwst = self._jwst_obs_axes
            jwst_img = images[..., self.jwst_channel, None]
            if self.antialias_resampling:
                box = self._boxcar_2d(H, self.jwst_obs_px_arcsec)
                jwst_img = self._batch_fftconvolve(jwst_img[..., 0], box)[..., None]
            # fill_value=None: linear extrapolation outside the model grid, matching
            # model_to_obs.ipynb's `interpn(..., fill_value=None)` for the JWST channel.
            batch["im_jy_jwst"] = self._batch_resample_jit(
                jwst_img, src_x, src_y, tx_jwst, ty_jwst, fill_value=None
            )

        # ALMA — one JIT call for all channels (they share one fixed obs pixel scale)
        if self.alma_channels:
            tx, ty = self._alma_obs_axis
            alma_imgs = images[..., self.alma_channels]      # (N, H, W, n_alma)
            if self.antialias_resampling:
                box = self._boxcar_2d(H, self.alma_obs_px_arcsec)
                alma_imgs = self._multi_alma_convolve(
                    alma_imgs, jnp.broadcast_to(box, (alma_imgs.shape[-1],) + box.shape)
                )
            # fill_value=0.0: hard zero outside the model grid, matching
            # model_to_obs.ipynb's `interpn(..., fill_value=0)` for the ALMA channels —
            # unlike JWST, the notebook does not extrapolate flux beyond the simulated FOV.
            alma_resampled = self._batch_resample_jit(
                alma_imgs, src_x, src_y, tx, ty, fill_value=0.0
            )
            for i in range(len(self.alma_channels)):
                batch[f"im_jy_alma_{i}"] = alma_resampled[..., i:i+1]

        batch.pop("im_jy", None)
        return batch

    def apply_noise(self, batch: dict) -> dict:
        """
        Add Gaussian instrument noise to every resampled image array.

        Noise levels
        ------------
        JWST : `jwst_noise_mjy_sr` [MJy/sr] — direct standard deviation in
               the image units; added to 'im_jy_jwst'.
        ALMA : `alma_noises_jy_beam[i]` [Jy/beam] converted to MJy/sr via
               Ω_beam = π · θ_maj · θ_min / (4 ln 2)  [sr];
               added to 'im_jy_alma_i'.

        All noise arrays are drawn from the class-level Generator so the
        random state advances deterministically regardless of call order.
        """

        # ── JWST ──────────────────────────────────────────────────────────────
        # Guarded on `has_jwst`: an ALMA-only instance (`has_jwst=False`, as
        # an ALMA-only instance is) never has `apply_resampling` write
        # `im_jy_jwst`, so an unguarded read is a KeyError.  That never surfaced while the
        # only callers were `__call__` (always four channels) and the MCMC (which skips
        # noise entirely), and it is what a per-instrument posterior predictive check hits
        # first, which is what a posterior-predictive check does.
        if self.has_jwst:
            img_jwst = batch['im_jy_jwst']
            N = img_jwst.shape[0]
            # _split_key() advances self.key and returns a fresh subkey each time
            noise_jwst_sampled = jax.random.uniform(
                self._split_key(), shape=(N,),
                minval=self.jwst_noise_mjy_sr[0], maxval=self.jwst_noise_mjy_sr[1],
            )
            batch['sigma_jwst'] = noise_jwst_sampled
            img_jwst = (
                img_jwst
                + jax.random.normal(self._split_key(), img_jwst.shape)
                * noise_jwst_sampled[:, None, None, None]
            )
            batch['im_jy_jwst'] = jnp.asarray(img_jwst)

        # ── ALMA ──────────────────────────────────────────────────────────────
        # Noise is correlated on the scale of the beam that was drawn for this group.
        #
        # The critical detail: the noise kernel sigma must be expressed in *observed*
        # pixels, because the noise lives on the observed grid.  The original code reused
        # a sigma computed in *model* pixels (0.01"/px) on the observed grid (0.0286"/px),
        # which stretched the noise correlation length to 0.59" instead of 0.2" — nearly
        # 3x the beam, and ~9x fewer independent noise elements per map than intended.
        if self._alma_setup is None:
            raise RuntimeError(
                "apply_noise called before apply_psf_and_convolve: the ALMA beam/noise "
                "setup has not been drawn for this batch"
            )

        nksize = self._alma_noise_ksize
        for i in range(len(self.alma_channels)):
            img_alma = batch[f'im_jy_alma_{i}']
            xn, yn = img_alma.shape[1], img_alma.shape[2]
            out = img_alma

            for grp in self._alma_setup:
                lo, hi = grp["slice"]
                sig_maj = float(grp["fwhm_maj"][i]
                                / self.alma_obs_px_arcsec / FWHM_PER_SIGMA)
                sig_min = float(grp["fwhm_min"][i]
                                / self.alma_obs_px_arcsec / FWHM_PER_SIGMA)
                kern = self._gaussian_beam_kernel(
                    jnp.float32(sig_maj), jnp.float32(sig_min),
                    jnp.float32(np.deg2rad(grp["pa_deg"][i])), nksize,
                )
                white = jax.random.normal(self._split_key(), (hi - lo, xn, yn))
                corr = self._apply_correlated_alma_noise(
                    white, kern, float(grp["sigma_mjy_sr"][i])
                )
                if img_alma.ndim == 4:
                    corr = jnp.expand_dims(corr, axis=-1)
                out = out.at[lo:hi].add(corr)

            batch[f'im_jy_alma_{i}'] = jnp.asarray(out)

        return batch

    def apply_sed_noise(self, batch: dict) -> dict:
        """
        Add wavelength-dependent noise to SED fluxes.

        Uses a Log-Normal distribution parameterized to have the exact
        mean = flux and variance = sigma^2. This strictly prevents negative
        fluxes without resorting to hard clipping.

        log_sig_mu / log_sig_std are precomputed lazily on the first call
        (sed_lams is constant across all batches), so the digitize/lookup
        overhead is paid only once per run.

        `sed_sigma_jy` (constructor) replaces that whole population lookup with one fixed
        per-bin sigma in Jy -- the measured error bars of the *specific* source being
        re-simulated.  It is the SED's analogue of `randomize_alma_setup=False`: the
        wavelength-binned log-normal describes the spread of error bars across real disks,
        which is the right prior when the run must serve any observation, and the wrong one
        when the run is specialized to a single SED whose uncertainties are known.
        """
        seds    = batch['seds']                          # (N, L, 1)
        seds_2d = jnp.asarray(seds)[..., 0]            # (N, L)

        if self.sed_sigma_jy is not None:
            if self.sed_sigma_jy.shape[0] != seds_2d.shape[1]:
                raise ValueError(
                    f"sed_sigma_jy has {self.sed_sigma_jy.shape[0]} bins but the batch's "
                    f"SED has {seds_2d.shape[1]}; it must be given on the full model grid, "
                    "before any adapter-level bin subset")
            seds_noisy, sigma = self._sed_noise_fixed_sigma_jit(
                seds_2d, self.sed_sigma_jy, self._split_key())
            batch['seds']           = seds_noisy[..., jnp.newaxis]
            batch['sigma_sed_flux'] = sigma
            return batch

        # Lazy precompute: bin mapping depends only on sed_lams, which is fixed
        if self._log_sig_mu is None:
            lams_1d = jnp.asarray(batch['sed_lams'][0, :, 0])  # (L,)
            bin_idx = jnp.digitize(lams_1d, self.bin_edges) - 1
            bin_idx = jnp.clip(bin_idx, 0, len(self.sed_bin_means) - 1)
            self._log_sig_mu  = jnp.take(jnp.array(self.sed_bin_means), bin_idx)
            self._log_sig_std = jnp.take(jnp.array(self.sed_bin_stds),  bin_idx)

        k1 = self._split_key()
        k2 = self._split_key()
        seds_noisy, sigma = self._sed_noise_jit(
            seds_2d, self._log_sig_mu, self._log_sig_std, k1, k2
        )

        batch['seds']           = seds_noisy[..., jnp.newaxis]  # (N, L, 1)
        batch['sigma_sed_flux'] = sigma                          # (N, L)
        return batch

    def log10_sed(self, batch: dict) -> dict:
        """Apply log10 compression to SED fluxes, wavelengths, and noise sigma."""
        seds_log, lams_log, sigma_log = self._log10_arrays(
            batch['seds'], batch['sed_lams'], batch['sigma_sed_flux']
        )
        batch['seds']           = seds_log
        batch['sed_lams']       = lams_log
        batch['sigma_sed_flux'] = sigma_log
        return batch


    def __call__(self, batch: dict) -> dict:
        """
        Run the full augmentation pipeline in one Python call.

        Pass `[augmentation_class]` to BayesFlow's `augmentations` argument
        instead of a 7-element list so that BayesFlow invokes one Python
        function per batch rather than seven, reducing dispatch overhead.
        """
        batch = self.preprocess(batch)
        # Rotation FIRST: the instrument response is fixed on the sky, so rotating the
        # source must happen before the beam / PSF is applied.  Convolving first made the
        # anisotropic ALMA beam and the JWST diffraction pattern rotate along with the
        # disk, which no real observation does.
        batch = self.apply_rotation(batch)
        batch = self.apply_psf_and_convolve(batch)
        batch = self.apply_resampling(batch)
        batch = self.apply_noise(batch)
        # No image compression here: the images leave in linear MJy/sr, because compression is
        # the adapter's job -- there it is recorded per run and stays invertible.
        batch = self.apply_sed_noise(batch)
        batch = self.log10_sed(batch)
        return batch


# ══════════════════════════════════════════════════════════════════════════════════════
# Registration
# ══════════════════════════════════════════════════════════════════════════════════════

#: Constructor arguments whose values are nested sequences.  OmegaConf hands them over as lists,
#: and the class indexes/iterates them either way, but tuples keep them hashable for the jitted
#: static methods' closure-free precomputation and match the declared defaults.
_TUPLE_ARGS = (
    "rot_range_deg", "jwst_noise_mjy_sr", "alma_channels", "alma_beam_fwhm_maj_range",
    "alma_beam_theta_range", "alma_beam_axis_ratio_range", "alma_beam_pa_deg_range",
    "alma_noise_jy_beam_range",
    "alma_beams", "alma_noises_jy_beam",
)


def _tuplify(value):
    if isinstance(value, (list, tuple)):
        return tuple(_tuplify(v) for v in value)
    return value


@register_augmentation("protoplan_instrument")
def protoplan_instrument(params, rng, context):
    """`AugmentationsClass` over `augmentation.params`.

    Every constructor argument is passed straight through, so there is no per-knob plumbing to
    keep in step -- the beam and noise knobs in particular (see the class' `__init__`).  The seed
    is drawn from the injected generator, so the whole augmentation stream still reproduces from
    `cfg.seed`.

    Returned as the instance itself, relying on `__call__`: BayesFlow then makes **one** Python
    call per batch instead of one per pipeline step.
    """
    kwargs = {k: (_tuplify(v) if k in _TUPLE_ARGS else v) for k, v in dict(params).items()}
    kwargs.setdefault("seed", int(rng.integers(2 ** 31)))
    return AugmentationsClass(**kwargs)

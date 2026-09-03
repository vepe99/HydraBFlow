"""Shared machinery for the prior-predictive check notebooks.

The notebooks beside this file ask one question in three spaces: **is a real disk inside the
population the protoplan networks were trained on?**  In amplitude (`check_obs_range.py`), in SED
colour (`check_sed_range.py`) and in morphology (`check_shape_range.py`).

Answering it needs two things HydraBFlow does not otherwise have:

1. **The real observation, on the model's own observed grid.**  The images live in
   `<upstream>/data/obs_data/imdata_<disk>.pkl`, written with `dill` and carrying their class
   definition inline -- so they unpickle without importing the upstream package, but `dill` is
   needed to read them.  `build_real_batch` regrids them onto exactly the grid the augmentation
   produces and assembles the condition encodings the adapter expects.
2. **The out-of-distribution scoring toolkit** -- rank-normal scores, Mahalanobis against the
   training rows' *own* distance distribution, and the flux-normalised morphology features.  Ported
   from `<upstream>/scripts/saucer/check_shape_range.py` unchanged, so the numbers stay comparable.

Everything else is reused rather than reimplemented: `AugmentationsClass` supplies the observed-grid
axis, the ALMA pixel scale a real observation must be regridded onto, the beam solid angle and the
extinction curve; `ProtoplanetaryDiskSimulator.load_dataset` supplies the row-filtered training
cache.

Set ``HYDRABFLOW_NUM_GPUS=0`` before importing this module for a CPU-only session (the SED check
needs no GPU); otherwise importing `hydrabflow` picks a free GPU via autocvd.
"""

from __future__ import annotations

import json
import os
import pathlib

import numpy as np

import hydrabflow  # noqa: F401  -- pins KERAS_BACKEND=jax and selects a GPU before jax loads
from hydrabflow.augmentation.protoplan_instrument import ASSETS_DIR, AugmentationsClass
from hydrabflow.simulators import _protoplan_spec as spec

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PLOTS_ROOT = REPO_ROOT / "notebooks" / "prior_predictive_checks" / "plots"
OBS_SETUP_JSON = ASSETS_DIR / "obs_setup_measurements.json"

#: Where the upstream real-disk image pickles live -- only a fallback now: `load_real_images` reads
#: `assets/protoplan/realimg_<disk>.npz` (a few hundred kB, in the repo) when it is there.
#: `PROTOPLAN_OBS_DIR` overrides the location.
OBS_DATA_DIR = pathlib.Path(os.environ.get(
    "PROTOPLAN_OBS_DIR",
    "/export/data/vgiusepp/ProtoplanetaryDisk_SBI/data/obs_data"))

#: `stpsf` needs its reference-data tree for the JWST PSF (see docs/protoplanetary_disk.md).  An
#: already-exported `$STPSF_PATH` always wins; this fallback only fills it in when the variable is
#: unset *and* the directory happens to exist, so it cannot mask a real misconfiguration elsewhere.
_STPSF_FALLBACK = pathlib.Path.home() / "data" / "stpsf-data"
if "STPSF_PATH" not in os.environ and _STPSF_FALLBACK.is_dir():
    os.environ["STPSF_PATH"] = str(_STPSF_FALLBACK)

TINY = 1e-30

#: Half-width of the observed field, arcsec.  `AugmentationsClass`' own default.
OBS_HALF_EXTENT = 1.5
JWST_OBS_PX = 0.1
JWST_LAM_UM = 3.9

#: Mask levels the shape features are computed at.  See `shape_features`.
N_SIGMA = 3.0
REL_FRAC = 0.02

#: ALMA channel index -> (band name, wavelength).  Channel order is ascending wavelength, matching
#: the image cache's channels 1/2/3.
ALMA_CHANNELS = {0: ("B9", 450.0), 1: ("B7", 880.0), 2: ("B6", 1300.0)}

IMAGE_KEYS = ["im_jy_jwst", "im_jy_alma_0", "im_jy_alma_1", "im_jy_alma_2"]
BAND_LABEL = {"im_jy_jwst": "JWST 3.9um", "im_jy_alma_0": "ALMA 450um",
              "im_jy_alma_1": "ALMA 880um", "im_jy_alma_2": "ALMA 1300um"}

FEATURES = ["log_r50", "conc", "axis_ratio", "dip", "a1", "a2", "cen_off", "log_peak_frac"]
FEATURE_DOC = {
    "log_r50":       "log10 half-flux radius [arcsec]",
    "conc":          "log10(r90/r50), concentration",
    "axis_ratio":    "sqrt(lam_min/lam_max) of the 2nd-moment tensor (inclination proxy)",
    "dip":           "SB(0.25 r50)/SB_max -- ring (0) vs filled (1)",
    "a1":            "|m=1| Fourier amplitude over the 0.5-1.5 r50 annulus",
    "a2":            "|m=2| Fourier amplitude over the 0.5-1.5 r50 annulus",
    "cen_off":       "centroid offset / r50",
    "log_peak_frac": "log10(peak pixel / masked total flux)",
}


# ──────────────────────────────────────────────────────────────────────────
# The repo's own config
# ──────────────────────────────────────────────────────────────────────────
def load_cfg(overrides=("experiment=protoplan",)):
    """
    Compose the root Hydra config, exactly as the CLI stages do.

    The notebooks read the data path, the asinh knees and the augmentation knobs from here rather
    than restating them, so a check cannot silently drift from what the trained runs actually used.
    Mirrors `tests/conftest.py::compose_cfg`.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    from hydrabflow.config import register_configs
    from hydrabflow.pipeline.adapter import fill_adapter_from_simulator

    register_configs()
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "conf")):
        cfg = compose(config_name="config", overrides=list(overrides))
    fill_adapter_from_simulator(cfg)
    return cfg


def load_training_data(cfg):
    """
    The row-filtered training cache, via the configured simulator's own `load_dataset` hook.

    Images are memory-mapped, so this is cheap even for the SED-only notebook.  The filter (drop
    non-positive SED rows) and the `log_r_cav` jitter are applied inside the hook, which is why this
    goes through the simulator rather than `np.load`.
    """
    from hydrabflow.registry import get_simulator
    return get_simulator(cfg.simulator).load_dataset()


# ──────────────────────────────────────────────────────────────────────────
# The disk registry
# ──────────────────────────────────────────────────────────────────────────
#: Per-disk constants the measurements JSON does not carry.
#:
#: `sigma_jwst`, `av` and `dist_pc` are *assumptions*, not measurements, and the notebooks say so:
#:   * `sigma_jwst` is the JSON's `jwst_per_disk.bg_std_mjy_sr`.  Upstream deliberately overrode this
#:     for the saucer (0.03 rather than the measured 0.98), having found the JSON value was estimated
#:     over r>2" where the surface brightness is still falling, so it tracked the outer halo rather
#:     than the noise floor.  The same argument presumably applies to the other disks; until it is
#:     redone per disk, the measured value is used and flagged.
#:   * `av = 4.0` is the extinction every trained run assumed.  `check_shape_range`'s A_V scan is the
#:     diagnostic for whether that assumption is doing damage.
#:   * `dist_pc = DIST_FID_PC` means "no distance rescale" -- correct only in the sense that it is
#:     the neutral choice.  A real distance belongs here as soon as one is available.
#: `rot_band` is the band whose measured `theta_deg` is taken as the disk position angle.
#: `alma_channels` lists the channels actually measured; the rest are treated as missing modalities.
DISKS = {
    "oph163131": dict(
        pickle="imdata_oph163131.pkl",
        sed="seddata_oph163131.txt",
        jwst_attr="g395",
        dist_pc=spec.DIST_FID_PC,
        av=4.0,
        sigma_jwst=0.8,                   # assumed floor, not measured -- see the caveat above
        rot_band="B6",
        alma_channels=(0, 1, 2),          # B7 arrived with assets/protoplan/extracted_fits/
    ),
    "hvtauc": dict(
        pickle="imdata_hvtauc.pkl",
        sed="seddata_hvtauc.txt",
        jwst_attr="g395",
        dist_pc=spec.DIST_FID_PC,
        av=4.0,
        sigma_jwst=0.8,                   # assumed floor; the annulus reads 2.2 off the envelope
        rot_band="B6",
        alma_channels=(0, 1, 2),
    ),
    "saucer": dict(
        pickle="imdata_saucer.pkl",
        sed="seddata_saucer.txt",
        jwst_attr="g395",
        dist_pc=120.0,
        av=4.0,
        sigma_jwst=0.03,                  # upstream's hand-derived value, not the JSON's 0.98
        rot_band="B6",
        alma_channels=(0, 1, 2),
    ),
}


def disk_config(disk: str) -> dict:
    if disk not in DISKS:
        raise KeyError(f"unknown disk {disk!r}; known: {sorted(DISKS)}")
    return DISKS[disk]


def plot_dir(disk: str, tag: str = "") -> str:
    """
    `plots/<disk><tag>/`, created on demand.

    `tag` keeps a run under a non-default assumption (a sampled A_V, say) out of the directory
    holding the as-trained figures.  Two runs answer different questions and a shared name would
    silently overwrite the first.
    """
    path = PLOTS_ROOT / f"{disk}{tag}"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def parse_av(text, default) -> float | tuple[float, float]:
    """
    `"4"` -> `4.0`; `"1,5"` -> `(1.0, 5.0)`; empty -> `default`.

    A tuple is what `AugmentationsClass` reads as "draw one A_V per disk and marginalise over it",
    exactly as it already does for the ALMA beam when `randomize_alma_setup` is on -- no change to
    the augmentation is needed to sample extinction.
    """
    text = str(text).strip()
    if not text:
        return default
    parts = [float(v) for v in text.replace(" ", "").split(",") if v != ""]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        lo, hi = sorted(parts)
        if lo == hi:
            return lo
        return (lo, hi)
    raise ValueError(f"av must be a scalar or 'lo,hi', got {text!r}")


def av_suffix(av, default) -> str:
    """`""` when `av` is the disk's own assumption, else a filesystem-safe tag like `_av1-5`."""
    if av is None or av == default:
        return ""
    if isinstance(av, (tuple, list)):
        return f"_av{av[0]:g}-{av[1]:g}"
    return f"_av{av:g}"


def av_ref(av) -> float:
    """
    A single A_V to reference a rescale against, given either a scalar or a sampled range.

    `extinction_ratio` and `implied_av` both need one number.  For a sampled range the midpoint is
    the honest choice: it is the population's central assumption, and the figures that use it say so.
    """
    return float(np.mean(av)) if isinstance(av, (tuple, list)) else float(av)


def extinction_correction(aug, lams, av, rng=None, n_rows: int = 1) -> np.ndarray:
    """
    The `(N, L)` (or `(1, L)`) multiplicative extinction correction `preprocess` applies to an SED.

    Mirrors `AugmentationsClass.preprocess`' own two branches: a scalar `av` is one correction for
    the whole population, an `(lo, hi)` draws one A_V per row.  The augmentation stores the log10
    correction *per magnitude* of A_V (`i_ext_exp`), which is what lets A_V vary per row without
    re-interpolating the extinction table.
    """
    exp = np.asarray(aug.i_ext_exp(np.asarray(lams, dtype=np.float64)), dtype=np.float64)
    if isinstance(av, (tuple, list)):
        rng = rng or np.random.default_rng(0)
        draw = rng.uniform(av[0], av[1], size=(n_rows, 1))
        return 10.0 ** (exp[None, :] * draw)
    return 10.0 ** (exp * float(av))[None, :]


def missing_alma_channels(disk: str) -> tuple[int, ...]:
    """The ALMA channels this disk was never observed in -- dropped everywhere downstream."""
    present = set(disk_config(disk)["alma_channels"])
    return tuple(j for j in range(spec.N_ALMA) if j not in present)


def present_image_keys(disk: str) -> list[str]:
    """`IMAGE_KEYS` minus the bands this disk has no measurement for."""
    gone = {f"im_jy_alma_{j}" for j in missing_alma_channels(disk)}
    return [k for k in IMAGE_KEYS if k not in gone]


# ──────────────────────────────────────────────────────────────────────────
# The measured observing setup
# ──────────────────────────────────────────────────────────────────────────
def _obs_setup_from_npz(disk: str, cfg: dict):
    """
    `(setup, rot_deg)` out of `realimg_<disk>.npz`, when it was vendored from the FITS.

    The FITS route measures the beam, the noise and the position angle itself and stores them
    beside the images (`_vendor_fits_images.py`), so the disk carries its own setup and there is
    nothing to keep in step with `obs_setup_measurements.json`.  Returns None for a disk vendored
    the old way, which falls through to the JSON.
    """
    path = ASSETS_DIR / f"realimg_{disk}.npz"
    if not path.exists():
        return None
    z = np.load(path)
    if not any(k.endswith("_beam") for k in z.files):
        return None

    setup, theta_deg = {}, {}
    for j in cfg["alma_channels"]:
        if f"alma_{j}_beam" not in z.files:
            raise RuntimeError(
                f"{disk}: the registry declares ALMA channel {j} but {path.name} has no beam for "
                f"it -- one of the two is stale")
        maj, mn, bpa = (float(v) for v in z[f"alma_{j}_beam"])
        band = ALMA_CHANNELS[j][0]
        setup[j] = dict(band=band, fwhm_maj=maj, fwhm_min=mn, bpa_deg=bpa,
                        sigma_mjy_sr=float(z[f"alma_{j}_bkg"][1]))
        theta_deg[band] = float(z[f"alma_{j}_theta_deg"])
    band = cfg["rot_band"] if cfg["rot_band"] in theta_deg else setup[max(setup)]["band"]
    return setup, theta_deg[band]


def load_obs_setup(disk: str, path=OBS_SETUP_JSON) -> tuple[dict, float]:
    """
    `(setup, rot_deg)` for one disk, from `assets/protoplan/obs_setup_measurements.json`.

    `setup` is keyed by ALMA channel index (0 = 450 um / B9, 1 = 880 / B7, 2 = 1300 / B6) holding
    `band`, `fwhm_maj`/`fwhm_min` [arcsec], `bpa_deg` and `sigma_mjy_sr`.  Only the channels the
    disk actually has are present -- unlike upstream's `config.load_obs_setup`, which raises when
    any of the three is absent.  That raise is exactly what blocks oph163131, which has no B7; here
    a missing channel is a missing modality, which the trained networks already handle.

    `beam_pa` sign convention: the header BPA is measured North through East, the array handed to
    the network has +row = North and +col = East, and `_gaussian_beam_kernel` measures its PA
    counter-clockwise from +row toward +col -- the same rotation sense -- so the header BPA is used
    unchanged.
    """
    cfg = disk_config(disk)
    from_npz = _obs_setup_from_npz(disk, cfg)
    if from_npz is not None:
        return from_npz
    with open(path) as fh:
        meas = json.load(fh)

    key_to_j = {f"im_jy_alma_{j}": j for j in range(spec.N_ALMA)}
    setup: dict[int, dict] = {}
    theta_deg: dict[str, float] = {}
    for row in meas["alma_per_disk"]:
        if row["disk"] != disk:
            continue
        j = key_to_j[row["alma_key"]]
        setup[j] = dict(
            band=row["band"],
            fwhm_maj=float(row["bmaj_arcsec"]),
            fwhm_min=float(row["bmin_arcsec"]),
            bpa_deg=float(row["bpa_deg"]),
            sigma_mjy_sr=float(row["rms_mjy_sr"]),
        )
        theta_deg[row["band"]] = float(row["theta_deg"])

    expected = set(cfg["alma_channels"])
    if set(setup) != expected:
        raise RuntimeError(
            f"{disk}: the registry declares ALMA channels {sorted(expected)} but "
            f"{path} has {sorted(setup)} -- one of the two is stale"
        )
    if not theta_deg:
        raise RuntimeError(f"{disk}: no ALMA rows in {path} to take rot_deg from")
    # The registry's `rot_band` when the disk has it, else the longest wavelength measured -- a disk
    # missing B6 still has a position angle, just measured through a different beam.
    band = cfg["rot_band"] if cfg["rot_band"] in theta_deg else setup[max(setup)]["band"]
    return setup, theta_deg[band]


def jwst_footprint_gap(disk: str, path=OBS_SETUP_JSON) -> float:
    """
    Fraction of the real JWST cutout that falls outside the IFU footprint.

    Those pixels are exactly 0 in the regridded real image (the regrid fills out-of-footprint with
    0) while the sims carry noise there, so any *unmasked* field mean of the real JWST channel is
    deflated by roughly this fraction -- about 0.06 dex at 13%.  Reported rather than corrected: a
    threshold that removed them would bias each band by its own SNR instead, which is worse.
    """
    vendored = ASSETS_DIR / f"realimg_{disk}.npz"
    if vendored.exists():
        z = np.load(vendored)
        if "jwst_observed" in z.files:
            # The FITS route knows this exactly, per pixel, instead of re-deriving it from a
            # zero-fill: `observed` is False only where the IFU footprint never reached.
            return float(np.mean(~z["jwst_observed"]))
    with open(path) as fh:
        rows = json.load(fh)["jwst_per_disk"]
    for row in rows:
        if row["disk"] == disk:
            return float(row["frac_outside_footprint"])
    return float("nan")


def alma_obs_px(setup: dict) -> float:
    """
    The ALMA pixel scale to sample every band at, from the *measured* beams only.

    Delegates to `AugmentationsClass.default_alma_obs_px_arcsec`, which is documented as the single
    source of truth for "the grid a real observation must be regridded onto".  It takes
    `min(maj)` **across channels**, so it must be fed the measured beams alone: a placeholder beam
    for a missing band, if it happened to be the smallest, would silently re-grid every other band.
    """
    return AugmentationsClass.default_alma_obs_px_arcsec(
        [(s["fwhm_maj"], s["fwhm_maj"]) for s in setup.values()])


def obs_axes(setup: dict, alma_px: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    `(jwst_axis, alma_axis)` -- the arcsec coordinates of the two observed grids.

    `alma_px` defaults to this disk's own measured beams, which is the grid
    `fixed_setup_augmentation` produces.  Pass it explicitly to regrid onto some *other* forward
    model's grid -- notably the randomized training prior's, which is finer.  Comparing a real image
    against sims sampled on a different grid is not a like-for-like comparison, so the caller must
    say which grid it wants rather than getting one by accident.
    """
    return (AugmentationsClass._obs_axis(OBS_HALF_EXTENT, JWST_OBS_PX),
            AugmentationsClass._obs_axis(OBS_HALF_EXTENT,
                                         alma_obs_px(setup) if alma_px is None else alma_px))


def randomized_alma_obs_px(cfg) -> float:
    """
    The ALMA pixel scale the *randomized* training prior samples at, from a composed config.

    `conf/experiment/protoplan.yaml` leaves `alma_obs_px_arcsec: null`, so the augmentation derives
    it from the beam-major prior.  Read it back the same way rather than restating the number.
    """
    params = cfg.augmentation.params
    if params.get("alma_obs_px_arcsec") is not None:
        return float(params.alma_obs_px_arcsec)
    theta = params.get("alma_beam_theta_range")
    ratio = params.alma_beam_axis_ratio_range
    n_ch = len(params.alma_beam_fwhm_maj_range)
    return AugmentationsClass.default_alma_obs_px_arcsec(
        AugmentationsClass.effective_maj_range(
            [tuple(r) for r in params.alma_beam_fwhm_maj_range],
            None if theta is None else [tuple(r) for r in theta],
            [tuple(r) for r in ratio] if np.ndim(ratio) == 2
            else [tuple(ratio)] * n_ch))


def check_conditions_in_prior(setup: dict, sigma_jwst: float) -> bool:
    """
    Print each measured condition beside the prior the networks were trained on.

    Conditioning on a value the training distribution never produced makes the posterior
    extrapolation rather than inference, so it is reported loudly rather than discovered later.
    The ranges come from `AugmentationsClass`' own named constants, so this cannot drift from what
    the networks actually saw.
    """
    maj_r = AugmentationsClass.DEFAULT_ALMA_BEAM_FWHM_MAJ_RANGE
    rat_r = AugmentationsClass.DEFAULT_ALMA_BEAM_AXIS_RATIO_RANGE
    rms_r = AugmentationsClass.DEFAULT_ALMA_NOISE_JY_BEAM_RANGE
    jw_r = AugmentationsClass.DEFAULT_JWST_NOISE_MJY_SR
    ok = True

    def chk(name, val, lo, hi):
        nonlocal ok
        inside = lo <= val <= hi
        ok &= inside
        print(f"  {'ok ' if inside else 'OUT'} {name:24s} {val:12.5g}  prior [{lo:.4g}, {hi:.4g}]")

    print("Measured conditions vs the training priors:")
    chk("sigma_jwst", sigma_jwst, *jw_r)
    for j, s in sorted(setup.items()):
        omega = AugmentationsClass.beam_area_sr(s["fwhm_maj"], s["fwhm_min"])
        chk(f"beam_maj_{j} ({s['band']})", s["fwhm_maj"], *maj_r[j])
        chk(f"axis_ratio_{j}", s["fwhm_min"] / s["fwhm_maj"], *rat_r)
        chk(f"rms_jy_beam_{j}", s["sigma_mjy_sr"] * 1e6 * omega, *rms_r[j])
    if not ok:
        print("  !! at least one condition is outside the training prior -- the posterior for "
              "this source would be extrapolation, not inference")
    return ok


# ──────────────────────────────────────────────────────────────────────────
# The real observation
# ──────────────────────────────────────────────────────────────────────────
def _to_ascending(coord, img, axis):
    """Flip `img` along `axis` if `coord` descends, so both end up ascending."""
    coord = np.asarray(coord, dtype=np.float64)
    if coord.size > 1 and coord[0] > coord[-1]:
        return coord[::-1], np.flip(img, axis=axis)
    return coord, img


def regrid_to_model(img, ra_1d, dec_1d, tgt):
    """
    Bilinearly resample a real image onto the square model observed grid.

    Convention `img[axis0 = DEC, axis1 = RA]` in, the same out, on `tgt` for both axes.  Points
    outside the source footprint become 0 -- which is a real difference from the sims, where the
    same pixels carry noise; `check_obs_range`'s negative-fraction flag exists to catch it.
    """
    img = np.nan_to_num(np.asarray(img, dtype=np.float64), nan=0.0)
    dec_1d, img = _to_ascending(dec_1d, img, 0)
    ra_1d, img = _to_ascending(ra_1d, img, 1)
    from scipy.interpolate import interpn
    DD, RR = np.meshgrid(tgt, tgt, indexing="ij")
    out = interpn((dec_1d, ra_1d), img, np.stack([DD.ravel(), RR.ravel()], axis=-1),
                  method="linear", bounds_error=False, fill_value=0.0)
    return out.reshape(len(tgt), len(tgt)).astype(np.float32)


def load_real_images(disk: str) -> dict:
    """
    Ungridded real images per band: `assets/protoplan/realimg_<disk>.npz` if vendored, else the
    upstream `dill` pickle.

    Returns `{"jwst": {...}, "alma_0": {...}, ...}` with `img` [MJy/sr], `rRA`, `rDEC` [arcsec] and
    a `label`, for the bands this disk has.  JWST comes from `im_cent` and ALMA from
    `im_cent_MJyster`.  The vendored `.npz` holds exactly those arrays, cropped to +/-3" -- twice the
    model field, so `regrid_to_model` never reaches the edge (`_vendor_real_images.py`).
    """
    cfg = disk_config(disk)
    vendored = ASSETS_DIR / f"realimg_{disk}.npz"
    if vendored.exists():
        z = np.load(vendored)
        bands = {"jwst": dict(img=z["jwst_img"], rRA=z["jwst_rRA"], rDEC=z["jwst_rDEC"],
                              label=BAND_LABEL["im_jy_jwst"])}
        for j in cfg["alma_channels"]:
            bands[f"alma_{j}"] = dict(img=z[f"alma_{j}_img"], rRA=z[f"alma_{j}_rRA"],
                                      rDEC=z[f"alma_{j}_rDEC"],
                                      label=BAND_LABEL[f"im_jy_alma_{j}"])
        return bands

    import dill

    path = OBS_DATA_DIR / cfg["pickle"]
    if not path.exists():
        raise FileNotFoundError(
            f"neither {vendored} nor {path} exists.  Vendor the images with "
            f"`_vendor_real_images.py`, or point PROTOPLAN_OBS_DIR at the upstream pickles.")
    with open(path, "rb") as fh:
        obs = dill.load(fh)

    jw = getattr(obs, cfg["jwst_attr"])
    bands = {"jwst": dict(img=np.asarray(jw.im_cent),
                          rRA=np.asarray(jw.rRA), rDEC=np.asarray(jw.rDEC),
                          label=BAND_LABEL["im_jy_jwst"])}
    for j in cfg["alma_channels"]:
        band_name, _lam = ALMA_CHANNELS[j]
        b = getattr(obs, band_name)
        bands[f"alma_{j}"] = dict(img=np.asarray(b.im_cent_MJyster),
                                  rRA=np.asarray(b.rRA), rDEC=np.asarray(b.rDEC),
                                  label=BAND_LABEL[f"im_jy_alma_{j}"])
    return bands


def interp_real_sed(disk: str, train_lams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    `(flux, err, raw)` -- the real SED on the training wavelength grid, in linear Jy.

    Interpolated in log-log with `extrapolate`, matching how the training SEDs were prepared.
    `raw` is the `(n, 3)` table as measured, for plotting the actual data points.
    """
    raw = np.loadtxt(ASSETS_DIR / disk_config(disk)["sed"])
    lam_obs, flux_obs, err_obs = raw[:, 0], raw[:, 1], raw[:, 2]
    floor = 1e-10
    from scipy.interpolate import interp1d
    log_lam = np.log10(lam_obs)
    f_flux = interp1d(log_lam, np.log10(np.clip(flux_obs, floor, None)),
                      bounds_error=False, fill_value="extrapolate")
    f_err = interp1d(log_lam, np.log10(np.clip(err_obs, floor, None)),
                     bounds_error=False, fill_value="extrapolate")
    lg = np.log10(np.asarray(train_lams, dtype=np.float64))
    return 10.0 ** f_flux(lg), 10.0 ** f_err(lg), raw


def build_real_batch(disk: str, train_lams, alma_px: float | None = None,
                     verbose: bool = True) -> tuple[dict, dict]:
    """
    `(batch, aux)` -- the real observation as the adapter's own keys, N=1.

    The images are regridded onto the model observed grids, the SED interpolated onto the training
    grid and log10-compressed to match `AugmentationsClass.log10_sed`, and the observing setup
    written out in the encodings the adapter expects: `rot_sin`/`rot_cos` rather than the raw angle
    (which is discontinuous at the 0/360 wrap), and beam PA as sin/cos of **twice** the angle (an
    ellipse at theta and theta+180 is the same beam).

    Bands this disk lacks emit no keys at all.  `sigma_alma_j`, `beam_maj_j` and `beam_min_j` are in
    linear units (MJy/sr, arcsec) because the adapter log-transforms them itself.
    """
    cfg = disk_config(disk)
    setup, rot_deg = load_obs_setup(disk)
    jwst_axis, alma_axis = obs_axes(setup, alma_px)
    px = alma_obs_px(setup) if alma_px is None else alma_px
    if verbose:
        print(f"{disk}: JWST grid {len(jwst_axis)} pts, ALMA grid {len(alma_axis)} pts "
              f"(px={px:.6g}\"{'' if alma_px is None else ', caller-supplied'})")
        print(f"rot_deg = {rot_deg:.4f} deg (from {cfg['rot_band']})")
        missing = missing_alma_channels(disk)
        if missing:
            print(f"missing ALMA channels {list(missing)} "
                  f"({', '.join(ALMA_CHANNELS[j][0] for j in missing)}) -- treated as absent "
                  f"modalities, no figures or scores")
        check_conditions_in_prior(setup, cfg["sigma_jwst"])

    bands = load_real_images(disk)
    regridded = {}
    for key, b in bands.items():
        tgt = jwst_axis if key == "jwst" else alma_axis
        ra_1d = b["rRA"][0, :] if b["rRA"].ndim == 2 else b["rRA"]
        dec_1d = b["rDEC"][:, 0] if b["rDEC"].ndim == 2 else b["rDEC"]
        regridded[key] = regrid_to_model(b["img"], ra_1d, dec_1d, tgt)
        if verbose:
            rg = regridded[key]
            print(f"  {key:7s} {b['img'].shape} -> {rg.shape}  "
                  f"min={rg.min():.4g} max={rg.max():.4g}")

    flux_i, err_i, sed_raw = interp_real_sed(disk, train_lams)

    f32 = np.float32
    floor = 1e-10
    batch = {
        "im_jy_jwst": regridded["jwst"][None, :, :, None].astype(f32),
        "seds": np.log10(np.clip(flux_i, floor, None))[None, :, None].astype(f32),
        "sigma_sed_flux": np.log10(np.clip(err_i, floor, None))[None, :].astype(f32),
        "sed_lams": np.log10(np.asarray(train_lams, float))[None, :, None].astype(f32),
    }
    for j in cfg["alma_channels"]:
        batch[f"im_jy_alma_{j}"] = regridded[f"alma_{j}"][None, :, :, None].astype(f32)

    rot_rad = np.deg2rad(rot_deg)
    batch["rot_deg"] = np.array([rot_deg], dtype=f32)
    batch["rot_sin"] = np.array([np.sin(rot_rad)], dtype=f32)
    batch["rot_cos"] = np.array([np.cos(rot_rad)], dtype=f32)
    # Mirror parity is a no-op for these sims (major axis along the column axis, and an
    # axisymmetric disk is mirror-symmetric about it), so the real disk takes the +1 branch.
    batch["sky_flip"] = np.array([1.0], dtype=f32)
    batch["sigma_jwst"] = np.array([cfg["sigma_jwst"]], dtype=f32)
    for j, s in sorted(setup.items()):
        pa2 = np.deg2rad(2.0 * s["bpa_deg"])
        batch[f"beam_maj_{j}"] = np.array([s["fwhm_maj"]], dtype=f32)
        batch[f"beam_min_{j}"] = np.array([s["fwhm_min"]], dtype=f32)
        batch[f"beam_pa_{j}"] = np.array([s["bpa_deg"]], dtype=f32)
        batch[f"beam_pa_sin_{j}"] = np.array([np.sin(pa2)], dtype=f32)
        batch[f"beam_pa_cos_{j}"] = np.array([np.cos(pa2)], dtype=f32)
        batch[f"sigma_alma_{j}"] = np.array([s["sigma_mjy_sr"]], dtype=f32)

    aux = dict(setup=setup, rot_deg=rot_deg, bands=bands, regridded=regridded, alma_px=px,
               jwst_axis=jwst_axis, alma_axis=alma_axis,
               sed_raw=sed_raw, sed_flux_interp=flux_i, sed_err_interp=err_i)
    return batch, aux


# ──────────────────────────────────────────────────────────────────────────
# The forward model, pinned to one disk's setup
# ──────────────────────────────────────────────────────────────────────────
def configured_augmentation(cfg, disk: str | None = None, **overrides) -> AugmentationsClass:
    """
    The augmentation exactly as configured for training -- randomized observing prior and all.

    This is the right forward model for the *amplitude* check, whose question is "what population
    did the network actually see", not "what would this disk look like through its own beam".  Only
    the distance is pinned to the real disk when one is named: comparing a source at its true
    distance against a population placed at the fiducial 140 pc would report an angular-size and
    SED-flux mismatch that is an artefact of the distance rather than of the physics.
    """
    from omegaconf import OmegaConf
    params = OmegaConf.to_container(cfg.augmentation.params, resolve=True)
    if disk is not None:
        params["dist_pc"] = disk_config(disk)["dist_pc"]
    params.setdefault("seed", 0)
    params.update(overrides)
    return AugmentationsClass(**params)


def measured_beams_noises(setup: dict) -> tuple[list[tuple[float, float, float]], list[float]]:
    """
    `(alma_beams, alma_noises_jy_beam)` for `AugmentationsClass` from one disk's measured setup.

    A channel this disk lacks still needs *some* beam for the forward model to produce all
    `spec.N_ALMA` channels, so it is given the widest measured one and every output of it is
    dropped downstream.  Split out of `fixed_setup_augmentation` because `summary_neighbours.py`
    needs the same beams on a *different* pixel grid -- the one the network was trained on -- so it
    cannot reuse the whole augmentation.
    """
    widest = max(setup.values(), key=lambda s: s["fwhm_maj"])
    beams, noises = [], []
    for j in range(spec.N_ALMA):
        s = setup.get(j, widest)
        beams.append((s["fwhm_maj"], s["fwhm_min"], s["bpa_deg"]))
        noises.append(s["sigma_mjy_sr"] * 1e6
                      * AugmentationsClass.beam_area_sr(s["fwhm_maj"], s["fwhm_min"]))
    return beams, noises


def fixed_setup_augmentation(disk: str, n_model_px: int, random_rotation: bool = False,
                             av: float | tuple[float, float] | None = None,
                             **overrides) -> AugmentationsClass:
    """
    `AugmentationsClass` fixed to this disk's own measured beams, noise, distance and PA.

    Two things this gets right that are easy to get wrong:

    * **Fixed position angle needs no new code.**  `apply_rotation` already collapses to a single
      angle for every image when `rot_range_deg=(theta, theta)`.  That is the default here rather
      than a fresh draw per sim, because the ALMA beam is elongated and sky-fixed: sampling PA
      injects disk-vs-beam-angle scatter into the reference population that the real disk, at one
      known PA, does not have -- which widens the null and makes the test *less* able to see a
      mismatch.  `random_rotation=True` restores the training-time marginal.
    * **`alma_obs_px_arcsec` is passed explicitly**, from the measured beams only (see
      `alma_obs_px`).  A channel this disk lacks still needs *some* beam for the forward model to
      produce all three ALMA channels, so it is given the widest measured one -- the choice cannot
      affect the grid, and every output of that channel is dropped downstream anyway.
    """
    cfg = disk_config(disk)
    setup, rot_deg = load_obs_setup(disk)
    px = alma_obs_px(setup)

    beams, noises = measured_beams_noises(setup)

    kwargs = dict(
        px_arcsec_mod=spec.FOV_ARCSEC_MOD / (n_model_px - 1),
        fov_arcsec_mod=spec.FOV_ARCSEC_MOD,
        dist_pc=cfg["dist_pc"],
        av=cfg["av"] if av is None else av,
        randomize_alma_setup=False,
        alma_beams=beams,
        alma_noises_jy_beam=noises,
        alma_obs_px_arcsec=px,
        jwst_noise_mjy_sr=(cfg["sigma_jwst"], cfg["sigma_jwst"]),
        jwst_obs_px_arcsec=JWST_OBS_PX,
        obs_half_extent=OBS_HALF_EXTENT,
        rot_range_deg=(0.0, 360.0) if random_rotation else (rot_deg, rot_deg),
        random_flip=random_rotation,
        seed=0,
    )
    kwargs.update(overrides)
    return AugmentationsClass(**kwargs)


# ──────────────────────────────────────────────────────────────────────────
# Streaming the training set
# ──────────────────────────────────────────────────────────────────────────
class Accum:
    """
    Streaming accumulator: exact moments over the full set plus a bounded uniform subsample.

    The augmented training set is ~20 GB of pixels, so nothing here keeps them.  `count/sum/sumsq/
    min/max/neg/nan` are exact in float64; the subsample exists only for percentiles and histograms.
    """

    def __init__(self, per_batch_sample, rng=None):
        self.count = 0
        self.sum = 0.0
        self.sumsq = 0.0
        self.min = np.inf
        self.max = -np.inf
        self.neg = 0
        self.nan = 0
        self.per_batch_sample = per_batch_sample
        self._rng = rng or np.random.default_rng(0)
        self._samples = []

    def update(self, x):
        x = np.asarray(x, dtype=np.float64).ravel()
        n_nan = int(np.isnan(x).sum())
        self.nan += n_nan
        if n_nan:
            x = x[~np.isnan(x)]
        if x.size == 0:
            return
        self.count += x.size
        self.sum += float(x.sum())
        self.sumsq += float(np.dot(x, x))
        self.min = min(self.min, float(x.min()))
        self.max = max(self.max, float(x.max()))
        self.neg += int((x < 0).sum())
        k = min(self.per_batch_sample, x.size)
        if k > 0:
            idx = self._rng.choice(x.size, size=k, replace=False)
            self._samples.append(x[idx].astype(np.float32))

    @property
    def samples(self):
        return (np.concatenate(self._samples) if self._samples
                else np.array([np.nan], dtype=np.float32))

    def finalize(self):
        mean = self.sum / max(self.count, 1)
        var = max(self.sumsq / max(self.count, 1) - mean * mean, 0.0)
        p1, p50, p99 = np.percentile(self.samples, [1, 50, 99])
        return dict(n=self.count, min=self.min, max=self.max, mean=mean,
                    std=float(np.sqrt(var)), p1=p1, p50=p50, p99=p99,
                    frac_neg=self.neg / max(self.count, 1), n_nan=self.nan)


def stats(x):
    """`Accum.finalize`'s dict computed directly, for an array small enough to hold."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n_nan = int(np.isnan(x).sum())
    good = x[~np.isnan(x)]
    p1, p50, p99 = (np.percentile(good, [1, 50, 99]) if good.size
                    else (np.nan, np.nan, np.nan))
    return dict(n=good.size, min=float(good.min()) if good.size else np.nan,
                max=float(good.max()) if good.size else np.nan,
                mean=float(good.mean()) if good.size else np.nan,
                std=float(good.std()) if good.size else np.nan,
                p1=p1, p50=p50, p99=p99,
                frac_neg=float((good < 0).mean()) if good.size else np.nan, n_nan=n_nan)


def training_batches(data: dict, batch_size: int, n_rows: int = 0):
    """
    Yield row slices of a `load_dataset()` dict, ready to hand to the augmentation.

    The image cache is memory-mapped, so slicing is what pulls each batch into RAM; `n_rows=0`
    streams the whole set.  Only the keys the forward model needs are materialised.
    """
    keys = ["im_jy", "im_lams", "seds", "sed_lams"]
    total = len(data["im_jy"])
    n = total if n_rows in (0, None) else min(int(n_rows), total)
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        yield i, j, {k: np.asarray(data[k][i:j]) for k in keys}


# ──────────────────────────────────────────────────────────────────────────
# Morphology features
# ──────────────────────────────────────────────────────────────────────────
def _radius_at_fraction(cum, edges, frac):
    """Radius enclosing `frac` of the flux, linearly interpolated in the cumulative profile."""
    n, nb = cum.shape
    rows = np.arange(n)
    j = np.clip((cum < frac).sum(axis=1), 0, nb - 1)
    c_hi = cum[rows, j]
    c_lo = np.where(j > 0, cum[rows, np.maximum(j - 1, 0)], 0.0)
    t = np.clip((frac - c_lo) / np.maximum(c_hi - c_lo, TINY), 0.0, 1.0)
    return edges[j] + t * (edges[j + 1] - edges[j])


def shape_features(img, axis, sigma, rel_frac=None, n_bins=None):
    """
    Flux-normalised morphology of a batch of observed-grid images.

    Parameters
    ----------
    img : (n, H, W) or (n, H, W, 1) linear surface brightness (MJy/sr), H == W.
    axis : (H,) arcsec coordinate of the grid, ascending and symmetric about 0.
    sigma : the band's measured RMS in the same units -- the absolute mask level.
    rel_frac : if given, mask at `rel_frac * peak` instead of `N_SIGMA * sigma`.  That mask is
        invariant to any per-channel rescale (e.g. a wrong A_V); the absolute one is not, which is
        exactly the difference the two arms exist to expose.

    Returns
    -------
    (n, len(FEATURES)) float64, NaN on rows where the mask kept nothing.
    """
    img = np.asarray(img, dtype=np.float32)
    if img.ndim == 4:
        img = img[..., 0]
    n, H, W = img.shape
    assert H == W, f"expected a square grid, got {(H, W)}"
    axis = np.asarray(axis)
    assert axis.shape == (H,), f"axis {axis.shape} does not match image {(H, W)}"
    nb = n_bins or max(8, H // 3)
    rows = np.arange(n)

    peak = img.max(axis=(1, 2))
    thr = (rel_frac * np.maximum(peak, 0.0)) if rel_frac is not None \
        else np.full(n, N_SIGMA * float(sigma), dtype=np.float32)
    w = np.where(img > thr[:, None, None], img, np.float32(0.0))
    F = w.sum(axis=(1, 2))
    ok = F > 0
    Fs = np.where(ok, F, 1.0)

    X = axis.astype(np.float32)[None, :]      # columns = RA
    Y = axis.astype(np.float32)[:, None]      # rows    = DEC
    xc = (w * X).sum(axis=(1, 2)) / Fs
    yc = (w * Y).sum(axis=(1, 2)) / Fs
    dx = X[None] - xc[:, None, None]
    dy = Y[None] - yc[:, None, None]

    Mxx = (w * dx * dx).sum(axis=(1, 2)) / Fs
    Myy = (w * dy * dy).sum(axis=(1, 2)) / Fs
    Mxy = (w * dx * dy).sum(axis=(1, 2)) / Fs
    tr, det = Mxx + Myy, Mxx * Myy - Mxy ** 2
    disc = np.sqrt(np.maximum(tr ** 2 / 4.0 - det, 0.0))
    lam_hi = np.maximum(tr / 2.0 + disc, 0.0)
    lam_lo = np.maximum(tr / 2.0 - disc, 0.0)
    axis_ratio = np.sqrt(lam_lo / np.maximum(lam_hi, TINY))

    # Radial profile via one bincount over (row, annulus) -- no Python loop over images.
    r = np.sqrt(dx * dx + dy * dy)
    edges = np.linspace(0.0, float(axis.max()) * np.sqrt(2.0), nb + 1)
    ib = np.clip(np.digitize(r, edges) - 1, 0, nb - 1)
    flat = (ib + nb * rows[:, None, None]).ravel()
    prof = np.bincount(flat, weights=w.ravel().astype(np.float64),
                       minlength=nb * n).reshape(n, nb)
    cnt = np.bincount(flat, minlength=nb * n).reshape(n, nb)
    sb = prof / np.maximum(cnt, 1)
    cum = np.cumsum(prof, axis=1)
    cum = cum / np.maximum(cum[:, -1:], TINY)

    r50 = _radius_at_fraction(cum, edges, 0.5)
    r90 = _radius_at_fraction(cum, edges, 0.9)

    jb = np.clip(np.searchsorted(edges, 0.25 * r50, side="right") - 1, 0, nb - 1)
    dip = sb[rows, jb] / np.maximum(sb.max(axis=1), TINY)

    # m=1/m=2 amplitudes over the 0.5-1.5 r50 annulus.  Amplitude only, so this is invariant to
    # the in-plane rotation `apply_rotation` applies.
    ann = (r >= 0.5 * r50[:, None, None]) & (r <= 1.5 * r50[:, None, None])
    wa = np.where(ann, w, np.float32(0.0))
    Fa = np.maximum(wa.sum(axis=(1, 2)), TINY)
    phi = np.arctan2(dy, dx)
    amp = []
    for m in (1, 2):
        c = (wa * np.cos(m * phi)).sum(axis=(1, 2)) / Fa
        s = (wa * np.sin(m * phi)).sum(axis=(1, 2)) / Fa
        amp.append(np.hypot(c, s))

    feats = np.stack([
        np.log10(np.maximum(r50, TINY)),
        np.log10(np.maximum(r90, TINY) / np.maximum(r50, TINY)),
        axis_ratio,
        dip,
        amp[0],
        amp[1],
        np.hypot(xc, yc) / np.maximum(r50, TINY),
        np.log10(np.maximum(peak, TINY) / Fs),
    ], axis=1).astype(np.float64)

    feats[~ok] = np.nan
    feats[~np.isfinite(feats).all(axis=1)] = np.nan
    return feats


# ──────────────────────────────────────────────────────────────────────────
# Out-of-distribution scoring
# ──────────────────────────────────────────────────────────────────────────
def percentile_rank(train, value):
    """Fraction of training rows below `value`, in percent."""
    train = np.asarray(train)
    train = train[np.isfinite(train)]
    return 100.0 * float(np.searchsorted(np.sort(train), value)) / max(train.size, 1)


def normal_scores(train, value):
    """
    Rank-transform a training column to standard normal scores through its own empirical CDF, and
    map `value` through the same CDF.

    Rank-transforming first is what makes a plain Mahalanobis distance meaningful here: these
    features are bounded and skewed (`axis_ratio` and `dip` both live in [0,1]), so a covariance on
    the raw values would be dominated by the tails of the wrong shape.
    """
    from scipy.stats import norm
    train = np.asarray(train, dtype=np.float64)
    n = train.size
    order = np.argsort(train)
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n)
    z_tr = norm.ppf((ranks + 0.5) / n)
    u = np.clip((np.searchsorted(train[order], value) + 0.5) / n, 0.5 / n, 1 - 0.5 / n)
    return z_tr, float(norm.ppf(u))


def normal_score_block(F_tr, f_sa):
    """Column-wise `normal_scores` over a whole feature matrix -> ((N,p), (p,))."""
    Z = np.empty_like(F_tr)
    z = np.empty(F_tr.shape[1])
    for k in range(F_tr.shape[1]):
        Z[:, k], z[k] = normal_scores(F_tr[:, k], f_sa[k])
    return Z, z


def mahalanobis(Z_tr, z_sa):
    """
    `(d2 of every training row, d2 of the real disk, percentile of the latter)`.

    The reference is the *empirical* distribution of the training rows' own distances, not a
    chi-square -- these features are skewed enough that the analytic null would be wrong.
    """
    cov = np.cov(Z_tr, rowvar=False)
    cov = np.atleast_2d(cov) + 1e-6 * np.eye(Z_tr.shape[1])
    inv = np.linalg.inv(cov)
    d2_tr = np.einsum("ij,jk,ik->i", Z_tr, inv, Z_tr)
    d2_sa = float(z_sa @ inv @ z_sa)
    return d2_tr, d2_sa, percentile_rank(d2_tr, d2_sa)


def whiten(Z_tr, z_sa):
    """Whiten the normal-score space so Euclidean distance there is Mahalanobis distance."""
    cov = np.cov(Z_tr, rowvar=False) + 1e-6 * np.eye(Z_tr.shape[1])
    vals, vecs = np.linalg.eigh(cov)
    W = vecs @ np.diag(1.0 / np.sqrt(np.maximum(vals, 1e-12))) @ vecs.T
    return Z_tr @ W, z_sa @ W


# ──────────────────────────────────────────────────────────────────────────
# Extinction
# ──────────────────────────────────────────────────────────────────────────
def _ext_hiav(lam_um):
    """`A_lam / A_K` from the McClure09 table `AugmentationsClass` itself reads."""
    from hydrabflow.augmentation.protoplan_instrument import DEFAULT_EXTINCTION_LAW_PATH
    ext_lam, _lo, ext_hiav = np.loadtxt(DEFAULT_EXTINCTION_LAW_PATH, skiprows=21).T
    return np.interp(lam_um, ext_lam, ext_hiav)


def extinction_ratio(lam_um, av, av_ref=4.0):
    """`corr(av)/corr(av_ref)` at `lam_um`, from the same formula the augmentation uses."""
    return float(10 ** (-(av - av_ref) / 7.75 * _ext_hiav(lam_um) / 2.5))


def implied_av(shift_dex, lam_um, av_model):
    """
    The A_V at which the model's flux at `lam_um` would move by `shift_dex` decades.

    `log10 corr(av) = -(av/7.75) * (A_lam/A_K) / 2.5`, so a required shift inverts to
    `av' = av_model - shift_dex * 2.5 * 7.75 / (A_lam/A_K)`.  This is what turns "the real disk's
    JWST/ALMA ratio sits `shift_dex` below the training median" into "the sims would need this A_V
    to match" -- and if that number is not a physically plausible extinction, extinction is not the
    cause.
    """
    return av_model - shift_dex * 2.5 * 7.75 / float(_ext_hiav(lam_um))


# ──────────────────────────────────────────────────────────────────────────
# Shared plot helpers
# ──────────────────────────────────────────────────────────────────────────
def mask_label(rel: bool) -> str:
    return f"relative {REL_FRAC:g}x peak mask" if rel else f"absolute {N_SIGMA:g}-sigma mask"


def feature_hist(ax, train, value, name, rank, disk="real"):
    train = np.asarray(train)
    train = train[np.isfinite(train)]
    lo, hi = np.percentile(train, [0.1, 99.9])
    lo, hi = min(lo, value), max(hi, value)
    if not hi > lo:                       # constant column (possible under a hard mask)
        lo, hi = lo - 1e-6, hi + 1e-6
    ax.hist(train, bins=np.linspace(lo, hi, 60), color="tab:blue", alpha=0.6, density=True)
    ax.axvline(value, color="tab:red", lw=2)
    ax.set_title(f"{name}  ({disk} p{rank:.1f})", fontsize=9)
    ax.tick_params(labelsize=7)


def neighbour_panel(sa_img, nb_imgs, nb_labels, band, axis, out_dir, disk, suffix=""):
    """
    The real disk beside its nearest training sims, **each panel stretched to its own peak**.

    A shared normalisation would just render the fainter panels black; amplitude is
    `check_obs_range`'s question, not this one's.  Every panel is at the real disk's own measured
    PA, so the comparison is like-for-like.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    ims = [sa_img] + list(nb_imgs)
    ncol = min(5, len(ims))
    nrow = int(np.ceil(len(ims) / ncol))
    ext = [axis.max(), axis.min(), axis.min(), axis.max()]   # RA increases left
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.7 * ncol, 2.9 * nrow))
    for k, ax in enumerate(np.atleast_1d(axes).ravel()):
        if k >= len(ims):
            ax.axis("off")
            continue
        vmax = float(np.nanmax(ims[k]))
        norm = mpl.colors.PowerNorm(gamma=0.6, vmin=0, vmax=vmax) if vmax > 0 else None
        ax.imshow(np.asarray(ims[k]).squeeze(), origin="lower", norm=norm, cmap="inferno",
                  extent=ext)
        ax.set_title(disk.upper() if k == 0 else nb_labels[k - 1], fontsize=8,
                     color="tab:red" if k == 0 else "black")
        ax.tick_params(labelsize=6)
    fig.suptitle(f"{BAND_LABEL[band]}: {disk} vs its {len(nb_imgs)} nearest training sims "
                 f"in shape space (each panel scaled to its own peak)")
    fig.tight_layout()
    path = os.path.join(out_dir, f"shape_neighbours_{band}{suffix}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def save(fig, out_dir, name, dpi=120):
    """Write a figure into `out_dir` and return its path."""
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path

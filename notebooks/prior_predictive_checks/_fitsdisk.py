"""
Read the real disks straight out of `assets/protoplan/extracted_fits/`.

The rest of this directory consumes `realimg_<disk>.npz` (vendored from upstream `dill` pickles by
`_vendor_real_images.py`).  This module is the same contract from the *FITS* instead -- the ALMA
`*.pbcor` maps and the NIRSpec G395 cubes -- so a disk needs nothing but `assets/`:

    {"jwst": {img, rRA, rDEC, label, ...}, "alma_0": {...}, "alma_1": {...}, "alma_2": {...}}

`img` is MJy/sr, `rRA`/`rDEC` are arcsec offsets from the field centre, and -- as everywhere in this
directory -- **rRA decreases with column** (East to the left) while rDEC increases with row.  The
direction comes from each file's full CD matrix, never from `CDELT` alone: the NIRSpec cubes carry
`PC1_1 = -1` against a *positive* `CDELT1`, so reading `CDELT` raw mirrors the JWST image against the
ALMA ones and the disk comes out inclined the other way.

Two things this reader does not paper over:

* **Band label != wavelength.** `hvtauc_B6` is 278.3 GHz = 1077 um and `oph163131_B6` is 1385 um,
  while the pipeline's B6 slot is 1300 um; both B7 files are 910 um against a 880 um slot.  The
  files are assigned to channels by their *name* (`BAND_CHANNEL`) and the true wavelength is
  reported in every band dict as `lam_um` so the offset stays visible.
* **`CRPIX` is not the source.**  Each map's reference pixel is its own phase centre, and those
  disagree between bands -- by 1.46" for oph163131's B6, which puts the disk off the corner of a
  +/-1.5" cut.  In *world* coordinates the bands agree to ~0.01", so every cut is taken about one
  per-disk source position (`SOURCE_RADEC`) through each file's own WCS, and `centroid` is returned
  per band so a residual mis-registration is visible rather than assumed away.
"""

from __future__ import annotations

import pathlib
import sys
import warnings
import zlib

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _realdisk import ASSETS_DIR, BAND_LABEL  # noqa: E402

from hydrabflow.augmentation.protoplan_instrument import AugmentationsClass  # noqa: E402

FITS_DIR = ASSETS_DIR / "extracted_fits"
#: File-name band -> the pipeline's ALMA channel index (0 = 450 um, 1 = 880, 2 = 1300).
BAND_CHANNEL = {"B9": 0, "B7": 1, "B6": 2}
#: Half-width of the cut, arcsec.  1.5" is the model field itself (`_realdisk.OBS_HALF_EXTENT`);
#: vendoring for `regrid_to_model` wants 3.0", twice that, so the regrid never reaches an edge.
HALF_ARCSEC = 1.5
JWST_LAM_UM = 3.9

C_KM_S = 299792.458

#: Source position per disk, ICRS degrees -- the origin every band's cut is taken about.  Measured
#: once by `measure_source_radec` -- the mean of the three ALMA bands' iterated centroids.
#: Reproduce with `uv run python notebooks/prior_predictive_checks/_fitsdisk.py --measure`.
SOURCE_RADEC = {
    "hvtauc":    (69.6479660, 26.1780606),
    "oph163131": (247.8801754, -24.4412694),
}


def disks() -> list[str]:
    """The disks `extracted_fits/` holds, from the file names."""
    return sorted({p.name.split("_")[0] for p in FITS_DIR.glob("*.fits")})


def _celestial_wcs(hdr):
    with warnings.catch_warnings():            # OBSGEO-[XYZ] -> OBSGEO-L/B/H, nothing we depend on
        warnings.simplefilter("ignore", FITSFixedWarning)
        return WCS(hdr).celestial


def _pixel_deltas(wcs):
    """
    Signed deg/pixel along each axis, from the full CD matrix.

    **Not** `CDELT` alone: the NIRSpec cubes carry `PC1_1 = -1`, so their RA runs the same way as
    ALMA's while `CDELT1` is positive.  Reading `CDELT` raw and flipping on its sign mirrored the
    JWST image against the ALMA ones -- the disk came out inclined the other way.
    """
    m = wcs.pixel_scale_matrix
    if abs(m[0, 1]) > 1e-3 * abs(m[0, 0]) or abs(m[1, 0]) > 1e-3 * abs(m[1, 1]):
        raise ValueError("the field is rotated on the sky; 1-D rRA/rDEC axes cannot express it "
                         "-- it needs a resampling, not an axis")
    return float(m[0, 0]), float(m[1, 1])


def _offset_axes(hdr, source_radec=None):
    """
    `(rRA, rDEC)` arcsec offsets, one entry per column / row.

    Relative to `source_radec` (through this file's own WCS) when given, else to the reference
    pixel.  The WCS route is what makes the bands share an origin -- CRPIX does not.
    """
    wcs = _celestial_wcs(hdr)
    d_ra, d_dec = _pixel_deltas(wcs)
    if source_radec is None:
        x0, y0 = hdr["CRPIX1"] - 1.0, hdr["CRPIX2"] - 1.0
    else:
        x0, y0 = wcs.world_to_pixel_values(*source_radec)
    ra = (np.arange(hdr["NAXIS1"]) - float(x0)) * d_ra * 3600.0
    dec = (np.arange(hdr["NAXIS2"]) - float(y0)) * d_dec * 3600.0
    return ra, dec


def _pixel_to_radec(wcs, x, y) -> tuple[float, float]:
    """`(ra, dec)` degrees with RA wrapped into [0, 360).

    `pixel_to_world_values` can hand back a negative RA near the 0h branch of a projection -- the
    NIRSpec cubes do, returning -112.12 for oph163131's 247.88.  Feeding that straight back into
    `world_to_pixel_values` is harmless, so the cuts were always right, but it is stored in the
    vendored `.npz` as `<band>_source_radec`, where a consumer differencing it against anything
    gets 1.2e6 arcsec.
    """
    ra, dec = wcs.pixel_to_world_values(x, y)
    return float(ra) % 360.0, float(dec)


def _source(disk, source_radec):
    """`"disk"` -> the registry position, None -> CRPIX, or an explicit (ra, dec)."""
    return SOURCE_RADEC[disk] if source_radec == "disk" else source_radec


def _cut(img, ra, dec, half):
    """Central `2*half` arcsec of `img`, with East kept to the left (rRA decreasing)."""
    if ra[1] > ra[0]:                      # +RA with column: flip so every band shares one parity
        img, ra = img[:, ::-1], ra[::-1]
    if dec[1] < dec[0]:
        img, dec = img[::-1, :], dec[::-1]
    ir, idec = np.abs(ra) <= half, np.abs(dec) <= half
    return (np.asarray(img, np.float32)[np.ix_(idec, ir)],
            ra[ir].astype(np.float32), dec[idec].astype(np.float32))


#: Aperture the recentring centroid is measured in, arcsec -- the model field, *not* the crop being
#: taken.  Same reason `position_angle_deg` fixes its own: vendoring crops at 3.0" so `regrid_to_model`
#: never reaches an edge, and hvtauc's B9 has bright outer arcs between 1.5" and 3".  Letting the
#: aperture follow the crop recentres on those instead of the disk, and the band then lands 0.17" off
#: in the +/-1.5" field that is the only part the network ever sees.
CENTROID_APERTURE_ARCSEC = HALF_ARCSEC

#: Centroid only pixels above this fraction of the cut's peak.  Weighting *every* positive pixel
#: instead lets the faint, asymmetric outer halo (and one-sided positive noise) pull the answer by
#: ~0.15" -- a tenth of the field.  0.2 keeps the bright ridge, which is what sits symmetrically
#: about the star.
CENTROID_FLOOR = 0.2


def _centroid(img, ra, dec, floor: float = CENTROID_FLOOR):
    """Flux-weighted centre of the bright emission, arcsec — the registration sanity check."""
    finite = np.isfinite(img)
    if not finite.any():
        return float("nan"), float("nan")
    w = np.where(finite & (img > floor * np.nanmax(img)), img, 0.0)
    if w.sum() <= 0:
        return float("nan"), float("nan")
    return (float((w.sum(axis=0) * ra).sum() / w.sum()),
            float((w.sum(axis=1) * dec).sum() / w.sum()))


#: Off-source annulus the sky is measured in, arcsec.  Outside any of these disks (~1" across) and
#: inside the part of the primary beam where pbcor has not yet amplified the noise: at 666 GHz the
#: 12 m primary beam is ~8.7" FWHM, so by 6" -- the annulus `obs_setup_measurements.json` used on
#: the older, coarser maps -- the correction has inflated sigma by ~4x.  JWST gets its own, since
#: the IFU field is only ~2.6" in radius.
NOISE_ANNULUS_ARCSEC = (1.5, 2.5)
JWST_NOISE_ANNULUS_ARCSEC = (1.0, 1.6)

#: Cap on the JWST cut's half-width, arcsec.  The ALMA maps are vendored at +/-3", twice the model
#: field, so `regrid_to_model` never reaches an edge -- but the IFU footprint is a rotated square,
#: so a +/-3" JWST cut is 52% unobserved and vendoring it would store more fabricated pixels than
#: real ones.  1.8" still clears the +/-1.5" model field by three JWST pixels, which is all the
#: interpolation needs, and holds the fill to a few percent.
JWST_MAX_HALF_ARCSEC = 1.8

#: Per-disk JWST noise floor, MJy/sr -- an **assumption**, overriding what `background_stats`
#: measures, in the same spirit as upstream's saucer override (0.03 for a measured 0.98).  The
#: annulus that is off-source is still on hvtauc's extended envelope, so the 2.2 MJy/sr it reads
#: there is the halo's scatter, not the detector's, and it falls outside the [0.5, 1.0] prior the
#: models are trained under.  Only the *width* is overridden; the level stays the measured local
#: median, so the filled corners join continuously onto the observed pixels.
JWST_NOISE_MJY_SR = {"hvtauc": 0.8, "oph163131": 0.8}


def background_stats(plane, px_arcsec: float, annulus=NOISE_ANNULUS_ARCSEC,
                     centre_px=None) -> tuple[float, float]:
    """
    `(median, sigma)` of the sky in an off-source annulus, arcsec-defined, MAD-based.

    Arcsec and not a fraction of the array: the six maps span 0.009-0.031"/px, so a fractional
    radius lands in a different physical place in every one of them -- for the finest ALMA maps,
    far out in the pbcor-amplified wing where the measured sigma is 100x the real noise floor.
    Widens outward, then falls back to every finite pixel, if the annulus is too empty (the B9 maps
    are ~78% NaN).
    """
    cy, cx = ((plane.shape[0] - 1) / 2.0, (plane.shape[1] - 1) / 2.0) if centre_px is None \
        else centre_px
    y, x = np.mgrid[:plane.shape[0], :plane.shape[1]]
    r = np.hypot(y - cy, x - cx) * px_arcsec
    finite = np.isfinite(plane)
    for lo, hi in [annulus, (annulus[0], 2.0 * annulus[1]), (annulus[0], np.inf)]:
        sky = plane[finite & (r >= lo) & (r < hi)]
        if sky.size >= 200:
            break
    else:
        sky = plane[finite]
    med = float(np.median(sky))
    return med, float(1.4826 * np.median(np.abs(sky - med)))


def _fill_unobserved(img, observed, bkg, seed_key) -> np.ndarray:
    """
    Replace unobserved pixels with draws from the image's own background.

    The alternative conventions are both worse.  Leaving NaN propagates through every statistic
    downstream; zero-filling (what the pickle route did) asserts "0 MJy/sr was measured here",
    which is a *stronger* claim than noise and biases any field mean low by the footprint fraction.
    The training images carry noise in every pixel, so noise is what makes the unobserved corners
    look like what the network was trained on.  Seeded on the disk name, so a rerun is identical.
    """
    if observed.all():
        return np.asarray(img, np.float32)
    mean, sigma = bkg
    rng = np.random.default_rng(zlib.crc32("/".join(seed_key).encode()))
    out = np.array(img, np.float32, copy=True)
    out[~observed] = rng.normal(mean, sigma, size=int((~observed).sum())).astype(np.float32)
    return out


def position_angle_deg(img, ra, dec, aperture: float = 1.5,
                       floor: float = CENTROID_FLOOR) -> float:
    """
    Orientation of the emission, degrees, in `obs_setup_measurements.json`'s `theta_deg` convention.

    Second-moment angle, then `180 - x`: the JSON measures the other way round from this frame's
    East-left axis.  Reproduces its three overlapping entries to 0.06 deg (oph163131 B9 139.39 and
    B6 139.51, hvtauc B6 18.31) -- the calibration that fixes the sign, since `rot_deg` feeds the
    augmentation's rotation and a flipped one would silently mis-orient every model image.

    Measured inside a fixed `aperture`, so it does not drift with the crop it is handed.
    """
    x, y = np.meshgrid(np.asarray(ra, float), np.asarray(dec, float))
    keep = (np.abs(x) <= aperture) & (np.abs(y) <= aperture) & np.isfinite(img)
    w = np.where(keep & (img > floor * np.nanmax(np.where(keep, img, np.nan))), img, 0.0)
    x, y = x - (w * x).sum() / w.sum(), y - (w * y).sum() / w.sum()
    t = 0.5 * np.arctan2(2.0 * (w * x * y).sum(),
                         (w * x * x).sum() - (w * y * y).sum())
    return float((180.0 - np.rad2deg(t)) % 180.0)


def measure_band_radec(disk: str, band: str, aperture: float = CENTROID_APERTURE_ARCSEC,
                       n_iter: int = 4) -> tuple[float, float]:
    """
    The emission centroid of one ALMA band as an ICRS position.

    Centroids the cut, re-cuts about that centroid, and repeats: the iteration is what stops a disk
    sitting near the edge of the first cut (oph163131's B6, whose reference pixel is 1.4" off the
    source) from dragging the answer.  Converges in two passes on both disks.

    `aperture` is deliberately not the caller's crop -- see `CENTROID_APERTURE_ARCSEC`.
    """
    with fits.open(FITS_DIR / f"{disk}_{band}.fits") as hdul:
        hdr = hdul[0].header
    wcs = _celestial_wcs(hdr)
    d_ra, d_dec = _pixel_deltas(wcs)
    radec = (float(hdr["CRVAL1"]) % 360.0, float(hdr["CRVAL2"]))
    for _ in range(n_iter):
        cx, cy = read_alma(disk, band, half_arcsec=aperture, source_radec=radec)["centroid"]
        x0, y0 = wcs.world_to_pixel_values(*radec)
        # The offsets were built as (pixel - x0) * d_ra, so inverting is the same division.
        radec = _pixel_to_radec(wcs, x0 + cx / (d_ra * 3600.0), y0 + cy / (d_dec * 3600.0))
    return radec


def measure_source_radec(disk: str,
                         aperture: float = CENTROID_APERTURE_ARCSEC) -> tuple[float, float]:
    """
    The disk's position as the mean of its ALMA bands' centroids — the *initial* cut origin.

    One position for all four channels was the original convention, on the argument that the star is
    at a single place on the sky.  It does not survive measurement, and `read_disk(recenter=True)`
    is now the default; this position remains the starting guess each band iterates from, and
    `recenter=False` restores the shared-origin cut that shows the residual.

    **Why the shared origin fails.**  Cut about one origin, the real bands land 0.065-0.146" off
    centre, and the augmented training population does not contain that.  Its own centroid scatter
    is *noise wander*, not disk asymmetry: on the clean RT images before any instrument model, the
    three mm channels have |r| median 0.001-0.003" and never exceed 0.155", so the augmented p90 of
    ~0.13" is manufactured by the noise, and it shrinks as SNR rises.  Matched on peak SNR to each
    real band (512 augmented rows, `experiment=protoplan_newbeam`):

        band     sim |r| med   p90     p99   | oph163131 shared-origin
        alma_0        0.021  0.049   0.088  | 0.065" = pct 97
        alma_1        0.009  0.021   0.041  | 0.081" = pct 100
        alma_2        0.006  0.017   0.028  | 0.146" = pct 100

    The real disks sit in the top decile of SNR, where the sims are centred to a few thousandths of
    an arcsec; re-cutting each band on its own centroid lands *at* the SNR-matched sim median, not
    below it.  (Against the *whole* population -- median SNR 11-39 -- the same offsets read as pct
    63/84/92, which is how they were once mistaken for normal.)

    The offsets are also not the brightness asymmetry a shared origin would be right to preserve.
    For oph163131 they lie at PA 193/196/15 deg while the disk's major axis is at 139.6: B9 and B7
    point one way and B6 exactly opposite (178 deg apart), i.e. a ~0.21" *relative* registration
    offset between B6 and the other two, which the mean origin then splits three ways.  hvtauc's
    are 0.004-0.035" in scattered directions -- noise, and recentring it is a no-op.

    The mm continuum is still the right vote to average for the starting guess: it is close to
    axisymmetric about the star, while the JWST scattered light is not (hvtauc's envelope is visibly
    lopsided), so the JWST slice is registered from this position, never used to set it.  JWST has
    recentred by default all along, for the separate reason that NIRSpec absolute astrometry is only
    good to 0.1-0.3".
    """
    pos = np.array([measure_band_radec(disk, band, aperture) for band in BAND_CHANNEL
                    if (FITS_DIR / f"{disk}_{band}.fits").exists()], float)
    return float(pos[:, 0].mean()), float(pos[:, 1].mean())


def read_alma(disk: str, band: str, half_arcsec: float = HALF_ARCSEC,
              source_radec="disk", recenter: bool = False,
              fill_unobserved: bool = True) -> dict:
    """
    One ALMA band, cut to the central field and converted Jy/beam -> MJy/sr.

    `recenter` cuts about this band's own iterated centroid (`measure_band_radec`) rather than the
    disk's shared `SOURCE_RADEC` -- see that function's docstring for why the shared origin leaves a
    residual the training population does not contain.  Off by default so `measure_band_radec`,
    which calls this, does not recurse; `read_disk` turns it on.

    The pbcor maps are NaN outside the primary beam.  That is far outside the model field, so the
    fill is normally a no-op here -- it exists so every band carries the same contract.
    """
    with fits.open(FITS_DIR / f"{disk}_{band}.fits") as hdul:
        hdr, data = hdul[0].header, np.squeeze(hdul[0].data)
    if hdr.get("BUNIT", "").strip() != "Jy/beam":
        raise ValueError(f"{disk} {band}: expected Jy/beam, got {hdr.get('BUNIT')!r}")

    maj, mn = hdr["BMAJ"] * 3600.0, hdr["BMIN"] * 3600.0
    omega = AugmentationsClass.beam_area_sr(maj, mn)     # one source of truth with the forward model
    plane = data / omega / 1.0e6
    origin = measure_band_radec(disk, band) if recenter else _source(disk, source_radec)
    ra, dec = _offset_axes(hdr, origin)
    img, ra, dec = _cut(plane, ra, dec, half_arcsec)
    observed = np.isfinite(img)
    bkg = background_stats(plane, abs(hdr['CDELT2']) * 3600.0)
    if fill_unobserved:
        img = _fill_unobserved(img, observed, bkg, seed_key=(disk, band))
    j = BAND_CHANNEL[band]
    return dict(img=img, rRA=ra, rDEC=dec, observed=observed, bkg=bkg,
                label=BAND_LABEL[f"im_jy_alma_{j}"], channel=j,
                band=band, lam_um=C_KM_S / (hdr["RESTFRQ"] / 1e9),   # c[km/s] / nu[GHz] = lambda in um
                fwhm_maj=maj, fwhm_min=mn, bpa_deg=float(hdr["BPA"]),
                theta=float(np.sqrt(maj * mn)), q=float(mn / maj),
                theta_deg=position_angle_deg(img, ra, dec),
                omega_beam_sr=float(omega), px_arcsec=abs(hdr["CDELT2"]) * 3600.0,
                source_radec=origin, centroid=_centroid(img, ra, dec))


def read_jwst(disk: str, lam_um: float = JWST_LAM_UM, half_arcsec: float = HALF_ARCSEC,
              source_radec="disk", recenter: bool = True,
              fill_unobserved: bool = True) -> dict:
    """
    The NIRSpec G395 cube's `lam_um` slice, cut to the central field.  Already MJy/sr.

    `recenter` cuts about the slice's own centroid instead of the ALMA-derived star position;
    `fill_unobserved` replaces the out-of-footprint corners with background noise.  The band dict
    carries `observed`, the boolean mask of pixels that are real measurements, either way.
    """
    with fits.open(FITS_DIR / f"{disk}_g395_cube.fits") as hdul:
        hdr, cube = hdul[0].header, hdul[0].data
        lams = np.asarray(hdul["WAVELENGTH"].data.field(0), float)
    if hdr.get("BUNIT", "").strip() != "MJy/sr":
        raise ValueError(f"{disk} JWST: expected MJy/sr, got {hdr.get('BUNIT')!r}")
    if not lams.min() <= lam_um <= lams.max():
        raise ValueError(f"{disk}: {lam_um} um outside the cube's {lams.min()}-{lams.max()} um")

    # Linear in wavelength between the two bracketing channels: the grid is 6.65e-4 um, four orders
    # finer than any structure the 3.9 um continuum has, so nothing more elaborate buys anything.
    k = int(np.searchsorted(lams, lam_um))
    w = (lam_um - lams[k - 1]) / (lams[k] - lams[k - 1])
    plane = (1.0 - w) * cube[k - 1] + w * cube[k]

    origin = _source(disk, source_radec)
    if recenter:
        # NIRSpec IFU absolute astrometry is uncertain at the 0.1-0.3" level -- the same size as the
        # offset between this slice's centroid and the ALMA-derived star position -- so the offset
        # cannot be attributed to physical asymmetry alone.  Re-cutting about the slice's own
        # centroid buys a centred image at the cost of the JWST-to-ALMA relative registration.
        wcs = _celestial_wcs(hdr)
        d_ra, d_dec = _pixel_deltas(wcs)
        for _ in range(3):
            a, d = _offset_axes(hdr, origin)
            cx, cy = _centroid(*_cut(plane, a, d, CENTROID_APERTURE_ARCSEC))
            x0, y0 = wcs.world_to_pixel_values(*origin)
            origin = _pixel_to_radec(wcs, x0 + cx / (d_ra * 3600.0),
                                     y0 + cy / (d_dec * 3600.0))

    ra, dec = _offset_axes(hdr, origin)
    img, ra, dec = _cut(plane, ra, dec, min(half_arcsec, JWST_MAX_HALF_ARCSEC))
    observed = np.isfinite(img)
    bkg = background_stats(plane, abs(hdr['CDELT2']) * 3600.0,
                           annulus=JWST_NOISE_ANNULUS_ARCSEC)
    measured_sigma = bkg[1]
    if disk in JWST_NOISE_MJY_SR:
        bkg = (bkg[0], float(JWST_NOISE_MJY_SR[disk]))
    if fill_unobserved:
        img = _fill_unobserved(img, observed, bkg, seed_key=(disk, "jwst"))
    return dict(img=img, rRA=ra, rDEC=dec, label=BAND_LABEL["im_jy_jwst"], channel=None,
                band="g395", lam_um=float(lam_um), px_arcsec=abs(hdr["CDELT2"]) * 3600.0,
                observed=observed, bkg=bkg, measured_sigma=measured_sigma,
                source_radec=origin, centroid=_centroid(img, ra, dec))


def read_disk(disk: str, half_arcsec: float = HALF_ARCSEC, source_radec="disk",
              recenter: bool = True, **jwst_kw) -> dict:
    """
    `{"jwst": ..., "alma_0": ..., "alma_1": ..., "alma_2": ...}` for every band on disk.

    `recenter` cuts every band about its own centroid.  On by default: see `measure_source_radec`
    for the measurement that says the shared-origin residual is outside the training population.
    `recenter=False` restores the single-origin cut, which is what shows the residual.
    """
    out = {}
    if (FITS_DIR / f"{disk}_g395_cube.fits").exists():
        out["jwst"] = read_jwst(disk, half_arcsec=half_arcsec, source_radec=source_radec,
                                **{"recenter": recenter, **jwst_kw})
    for band, j in BAND_CHANNEL.items():
        if (FITS_DIR / f"{disk}_{band}.fits").exists():
            out[f"alma_{j}"] = read_alma(disk, band, half_arcsec=half_arcsec,
                                         source_radec=source_radec, recenter=recenter)
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--measure" in sys.argv:
        for name in args or disks():
            ra, dec = measure_source_radec(name)
            print(f'    "{name}": ({ra:.7f}, {dec:.7f}),')
        sys.exit(0)

    for name in args or disks():
        for key, b in read_disk(name).items():
            cx, cy = b["centroid"]
            beam = (f"beam {b['fwhm_maj']:.4f}x{b['fwhm_min']:.4f}\" @ {b['bpa_deg']:+.1f}deg"
                    if "fwhm_maj" in b else "no beam")
            print(f"{name:<10} {key:<7} {b['label']:<11} lam={b['lam_um']:7.1f}um  "
                  f"{str(b['img'].shape):>10} @ {b['px_arcsec']:.5f}\"  {beam}  "
                  f"peak={np.nanmax(b['img']):.4g} MJy/sr  centroid=({cx:+.3f}, {cy:+.3f})\"")

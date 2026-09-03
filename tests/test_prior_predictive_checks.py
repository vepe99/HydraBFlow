"""Analytic checks for the prior-predictive notebooks' shared numerics.

`_realdisk.shape_features` is vectorised morphology on masked, flux-normalised images -- the one
piece of these notebooks whose output nobody can eyeball.  Every assertion here has a closed-form
answer, so none of it needs a GPU, the training cache, or the real-disk pickles.

Ported from ``<upstream>/scripts/saucer/check_shape_range.py::selftest``.
"""

import os
import pathlib
import sys

import numpy as np
import pytest

os.environ.setdefault("HYDRABFLOW_NUM_GPUS", "0")   # CPU: nothing here touches a device
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]
                       / "notebooks" / "prior_predictive_checks"))

rd = pytest.importorskip("_realdisk")

N = 64
AXIS = np.linspace(-1.5, 1.5, N)
PX = AXIS[1] - AXIS[0]
X, Y = np.meshgrid(AXIS, AXIS)
IDX = {name: k for k, name in enumerate(rd.FEATURES)}
SIGMA = 0.3


def feats(img, sigma=0.0, **kw):
    return rd.shape_features(np.asarray(img, np.float32)[None], AXIS, sigma, n_bins=48, **kw)[0]


def gaussian(sx=SIGMA, sy=SIGMA, x0=0.0):
    return np.exp(-(((X - x0) / sx) ** 2 + (Y / sy) ** 2) / 2.0)


def ring(radius=0.7, width=0.12):
    return np.exp(-((np.hypot(X, Y) - radius) ** 2) / (2 * width ** 2))


def test_round_gaussian_is_filled_symmetric_and_has_r50_1177_sigma():
    g = feats(gaussian())
    r50 = 10 ** g[IDX["log_r50"]]
    assert abs(r50 - 1.177 * SIGMA) < 2 * PX, f"gaussian r50 {r50:.4f} vs {1.177 * SIGMA:.4f}"
    assert g[IDX["axis_ratio"]] > 0.97
    assert g[IDX["dip"]] > 0.90, f"a filled source must not read as a ring: {g[IDX['dip']]:.3f}"
    assert g[IDX["a1"]] < 0.05 and g[IDX["a2"]] < 0.05
    assert g[IDX["cen_off"]] < 0.05


def test_inclined_gaussian_recovers_the_injected_axis_ratio():
    q_true = 0.4
    # Squashed along Y, not stretched: a sigma of 0.3/0.4 = 0.75" runs into the 1.5" field edge and
    # the truncation biases the second moments (it reads 0.45).
    e = feats(gaussian(sy=SIGMA * q_true))
    assert abs(e[IDX["axis_ratio"]] - q_true) < 0.05, e[IDX["axis_ratio"]]


def test_ring_reads_as_a_hole_and_is_less_centrally_extended():
    g, ri = feats(gaussian()), feats(ring())
    assert ri[IDX["dip"]] < 0.2, f"a ring must have a hole: {ri[IDX['dip']]:.3f}"
    assert ri[IDX["a1"]] < 0.05 and ri[IDX["a2"]] < 0.05
    assert ri[IDX["conc"]] < g[IDX["conc"]]


def test_lopsided_ring_turns_on_m1():
    r = ring()
    # m=2 rises too: the moments are referenced to the *shifted* centroid, so a lopsided ring also
    # reads as an off-centre one.  a1 staying dominant is the actual claim.
    lop = r * (1.0 + 0.6 * X / np.maximum(np.hypot(X, Y), 1e-6))
    lo, ri = feats(lop), feats(r)
    assert lo[IDX["a1"]] > 0.15
    assert lo[IDX["a1"]] > lo[IDX["a2"]] > ri[IDX["a2"]]
    assert lo[IDX["a1"]] > 3 * ri[IDX["a1"]]


def test_offset_source_recovers_its_centroid_offset():
    off = 0.45
    o = feats(gaussian(x0=off))
    assert abs(o[IDX["cen_off"]] * 10 ** o[IDX["log_r50"]] - off) < 2 * PX


def test_relative_mask_is_rescale_invariant_and_the_absolute_one_is_not():
    """The A_V claim, pinned.

    Extinction enters as one scalar per channel, so within a band a wrong A_V is a pure
    multiplicative rescale -- which every feature is invariant to *under the relative mask*.  The
    absolute mask must not be, since noise is injected after `preprocess` at fixed absolute sigma,
    and that asymmetry is the entire reason both arms are computed.
    """
    r = ring()
    base = feats(r, rel_frac=rd.REL_FRAC)
    scaled = feats(r * 7.3, rel_frac=rd.REL_FRAC)
    assert np.allclose(base, scaled, rtol=1e-5, atol=1e-6), base - scaled   # float32 tolerance
    assert not np.allclose(feats(r, sigma=0.05), feats(r * 7.3, sigma=0.05)), \
        "the absolute mask must NOT be rescale-invariant"


def test_fully_masked_rows_are_nan_not_silently_zero():
    out = rd.shape_features(np.zeros((1, N, N), np.float32), AXIS, 1.0)
    assert np.isnan(out).all()


@pytest.mark.parametrize("av_true", [0.5, 2.0, 7.0, 19.0])
def test_implied_av_inverts_extinction_ratio(av_true):
    shift = np.log10(rd.extinction_ratio(3.9, av_true, av_ref=4.0))
    assert abs(rd.implied_av(shift, 3.9, 4.0) - av_true) < 1e-6


def test_extinction_ratio_is_unity_at_the_reference_and_monotone_away_from_it():
    assert abs(rd.extinction_ratio(3.9, 4.0) - 1.0) < 1e-12
    assert rd.extinction_ratio(3.9, 8.0) < 1.0 < rd.extinction_ratio(3.9, 2.0)


def test_whiten_makes_the_covariance_identity():
    rng = np.random.default_rng(0)
    p = 17
    a = rng.normal(size=(p, p))
    pop = rng.multivariate_normal(np.zeros(p), a @ a.T + np.eye(p), size=4000)
    Zw, _ = rd.whiten(pop, np.zeros(p))
    assert np.allclose(np.cov(Zw, rowvar=False), np.eye(p), atol=0.08)


def test_mahalanobis_flags_an_outlier_and_not_a_typical_row():
    rng = np.random.default_rng(1)
    p = 6
    pop = rng.normal(size=(4000, p))
    _, d2_out, pct_out = rd.mahalanobis(pop, np.full(p, 8.0))
    assert pct_out > 99.0 and d2_out > 100
    _, _, pct_mid = rd.mahalanobis(pop, np.median(pop, axis=0))
    assert pct_mid < 20.0


def test_registry_and_measurements_json_agree_on_every_disk():
    """A registry that has drifted from the measurements is a silent wrong-beam bug."""
    for disk in rd.DISKS:
        setup, rot_deg = rd.load_obs_setup(disk)          # raises on mismatch
        assert set(setup) == set(rd.DISKS[disk]["alma_channels"])
        assert np.isfinite(rot_deg)


def test_oph163131_gained_its_b7_and_hvtauc_is_registered():
    """
    oph163131's missing B7 was a standing special case until `extracted_fits/` supplied one.

    The registry and the vendored images have to agree about that, in both directions: a channel
    declared here but absent from the `.npz` is a stale registry, and one present but undeclared is
    a band silently dropped from every check.
    """
    for disk in ["oph163131", "hvtauc"]:
        assert rd.DISKS[disk]["alma_channels"] == (0, 1, 2)
        assert rd.missing_alma_channels(disk) == ()
        assert set(rd.present_image_keys(disk)) >= {f"im_jy_alma_{j}" for j in range(3)}
        setup, rot_deg = rd.load_obs_setup(disk)          # raises on a registry/data mismatch
        assert set(setup) == {0, 1, 2} and np.isfinite(rot_deg)


# ──────────────────────────────────────────────────────────────────────────
# A_V: scalar vs sampled range
# ──────────────────────────────────────────────────────────────────────────
def test_parse_av_accepts_a_scalar_a_range_and_blank():
    assert rd.parse_av("", 4.0) == 4.0
    assert rd.parse_av("2.5", 4.0) == 2.5
    assert rd.parse_av("1,5", 4.0) == (1.0, 5.0)
    assert rd.parse_av("5,1", 4.0) == (1.0, 5.0)        # sorted, so order cannot matter
    assert rd.parse_av("3,3", 4.0) == 3.0               # a degenerate range is a scalar
    with pytest.raises(ValueError):
        rd.parse_av("1,2,3", 4.0)


def test_av_suffix_keeps_a_non_default_run_out_of_the_as_trained_directory():
    assert rd.av_suffix(4.0, 4.0) == ""
    assert rd.av_suffix((1.0, 5.0), 4.0) == "_av1-5"
    assert rd.av_suffix(2.0, 4.0) == "_av2"
    # A tag must never contain a path separator, or plot_dir would escape plots/.
    assert "/" not in rd.av_suffix((1.0, 5.0), 4.0)


def test_av_ref_collapses_a_range_to_its_midpoint():
    assert rd.av_ref(4.0) == 4.0
    assert rd.av_ref((1.0, 5.0)) == 3.0


def test_extinction_correction_matches_the_augmentations_two_branches():
    """A scalar A_V gives one correction for the population; a range gives one row each.

    Regression guard for a whole class of bug: every consumer of `av` has to handle both, and a
    scalar-only assumption fails silently (a broadcast) or loudly (`np.isnan` on a tuple).
    """
    class _FakeAug:
        # log10 correction per magnitude of A_V, the sign convention `i_ext_exp` uses.
        def i_ext_exp(self, lams):
            return -0.1 * np.ones_like(np.asarray(lams, dtype=float))

    aug, lams = _FakeAug(), np.array([0.5, 3.9, 1300.0])

    scalar = rd.extinction_correction(aug, lams, 4.0)
    assert scalar.shape == (1, 3)
    assert np.allclose(scalar, 10.0 ** (-0.4))

    sampled = rd.extinction_correction(aug, lams, (1.0, 5.0),
                                       rng=np.random.default_rng(0), n_rows=64)
    assert sampled.shape == (64, 3)
    # One A_V per row, so rows differ; and every row stays inside the range's bounds.
    assert len(np.unique(sampled[:, 0])) > 1
    assert (sampled >= 10.0 ** (-0.5) - 1e-12).all() and (sampled <= 10.0 ** (-0.1) + 1e-12).all()
    # A_V multiplies a per-wavelength exponent, so a row is constant across equal exponents.
    assert np.allclose(sampled[:, 0], sampled[:, 1])


def test_extinction_is_negligible_at_alma_wavelengths_over_any_av():
    """Why sampling A_V cannot meaningfully move an ALMA finding.

    Extinction enters as one multiplicative scalar per channel, and the McClure09 curve is nearly
    flat at mm wavelengths: over A_V in [1, 5] the 450/880/1300 um channels move by under 0.001 dex
    (0.05% in flux), orders of magnitude below the noise. So an ALMA shape feature or pixel
    statistic that sits outside the training population stays there whatever A_V is sampled.
    """
    for lam in (450.0, 880.0, 1300.0):
        assert abs(np.log10(rd.extinction_ratio(lam, 1.0, av_ref=5.0))) < 1e-3, lam
    # The optical, by contrast, swings by more than a decade over the same range.
    assert np.log10(rd.extinction_ratio(0.504, 1.0, av_ref=5.0)) > 1.0


# ──────────────────────────────────────────────────────────────────────────
# The FITS reader (`_fitsdisk`) -- skipped when `extracted_fits/` is absent
# ──────────────────────────────────────────────────────────────────────────
fd = pytest.importorskip("_fitsdisk")
pytestmark_fits = pytest.mark.skipif(not fd.FITS_DIR.exists(),
                                     reason="assets/protoplan/extracted_fits/ not present")


@pytestmark_fits
@pytest.mark.parametrize("disk", ["hvtauc", "oph163131"])
def test_every_band_is_registered_on_the_same_source(disk):
    """
    The point of `SOURCE_RADEC`: cut about one sky position and all four bands hold the same disk.

    Cutting about each file's own CRPIX instead leaves oph163131's B6 1.4" off -- the disk falls
    into a corner -- so this is the assertion that would catch a reader that silently drops back to
    the reference pixel.
    """
    for key, b in fd.read_disk(disk).items():
        cx, cy = b["centroid"]
        # The mm continuum is symmetric about the star, so it must land on it.  JWST scattered
        # light is one-sided (the near side is brighter), so its centroid legitimately sits a
        # couple of tenths off -- the RT models it is compared against are asymmetric the same way.
        tol = 0.35 if key == "jwst" else 0.15
        assert np.hypot(cx, cy) < tol, f"{disk} {key}: centroid ({cx:+.2f}, {cy:+.2f})\" off centre"
        half = fd.HALF_ARCSEC
        assert b["rRA"][0] > b["rRA"][-1], "rRA must decrease with column (East to the left)"
        assert b["rDEC"][0] < b["rDEC"][-1], "rDEC must increase with row"
        assert abs(b["rRA"]).max() <= half + 1e-6 and abs(b["rDEC"]).max() <= half + 1e-6
        nan_frac = float(np.mean(~np.isfinite(b["img"])))
        if key == "jwst":
            # NaN corners are the IFU footprint, not missing data -- `_realdisk.jwst_footprint_gap`
            # is the existing accounting for them. They must stay a corner, not eat the field.
            assert nan_frac < 0.15, f"{disk}: {nan_frac:.0%} of the JWST cut is outside the IFU"
        else:
            assert nan_frac == 0.0, f"{disk} {key}: the central field must be inside the pb"


@pytestmark_fits
def test_alma_headers_land_inside_the_trained_beam_prior():
    """Every measured (theta, q) is inside `experiment=protoplan_newbeam`'s per-band box."""
    from hydrabflow.augmentation.protoplan_instrument import AugmentationsClass

    # The boxes the runs were launched with, in the augmentation's channel order.
    theta = [(0.088, 0.140), (0.070, 0.205), (0.145, 0.205)]
    ratio = [(0.54, 0.73), (0.58, 0.78), (0.53, 0.82)]
    seen = 0
    for disk in fd.SOURCE_RADEC:
        for band, j in fd.BAND_CHANNEL.items():
            if not (fd.FITS_DIR / f"{disk}_{band}.fits").exists():
                continue
            b = fd.read_alma(disk, band)
            assert theta[j][0] <= b["theta"] <= theta[j][1], f"{disk} {band} theta {b['theta']:.4f}"
            assert ratio[j][0] <= b["q"] <= ratio[j][1], f"{disk} {band} q {b['q']:.3f}"
            # The Jy/beam -> MJy/sr conversion uses the forward model's own solid angle.
            np.testing.assert_allclose(
                b["omega_beam_sr"],
                AugmentationsClass.beam_area_sr(b["fwhm_maj"], b["fwhm_min"]), rtol=0)
            seen += 1
    assert seen == 6, "two disks x three bands"


@pytestmark_fits
@pytest.mark.parametrize("disk", ["hvtauc", "oph163131"])
def test_jwst_and_alma_share_one_sky_parity(disk):
    """
    The NIRSpec cubes carry `PC1_1 = -1` against a positive `CDELT1`.

    Read `CDELT` raw and the JWST image comes out mirrored against the ALMA ones -- the disk visibly
    inclined the other way.  Both must give RA decreasing with column, from the CD matrix.
    """
    bands = fd.read_disk(disk)
    for key, b in bands.items():
        assert b["rRA"][0] > b["rRA"][-1], f"{disk} {key}: RA must decrease with column"

    # Same physical disk, so the position angle of the emission agrees between JWST and ALMA. The
    # second moment's orientation is parity-sensitive, which is exactly what a mirrored axis breaks.
    def pa(b):
        img = np.nan_to_num(b["img"], nan=0.0)
        w = np.where(img > 0.2 * img.max(), img, 0.0)
        x, y = np.meshgrid(b["rRA"], b["rDEC"])
        x = x - (w * x).sum() / w.sum()
        y = y - (w * y).sum() / w.sum()
        return 0.5 * np.arctan2(2 * (w * x * y).sum(), (w * x * x).sum() - (w * y * y).sum())

    def elongation(b):
        img = np.nan_to_num(b["img"], nan=0.0)
        w = np.where(img > 0.2 * img.max(), img, 0.0)
        x, y = np.meshgrid(b["rRA"], b["rDEC"])
        x, y = x - (w * x).sum() / w.sum(), y - (w * y).sum() / w.sum()
        cxx, cyy, cxy = (w * x * x).sum(), (w * y * y).sum(), (w * x * y).sum()
        r = np.hypot(cxx - cyy, 2 * cxy) / (cxx + cyy)
        return float(r)

    ref = pa(bands["alma_0"])
    for key, b in bands.items():
        if key == "alma_0":
            continue
        # A round source has no PA to compare -- hvtauc's JWST core is nearly circular, and its
        # second-moment angle is then noise. Only bands that are actually elongated can vote.
        if elongation(b) < 0.25:
            continue
        d = np.rad2deg(np.arctan2(np.sin(pa(b) - ref), np.cos(pa(b) - ref)))
        assert abs(d) < 25.0, f"{disk} {key}: PA differs from B9 by {d:.0f} deg -- mirrored?"


@pytestmark_fits
def test_the_ra_direction_comes_from_the_cd_matrix_not_cdelt():
    """The JWST cubes' `CDELT1` is positive and their `PC1_1` is -1; only the product is the truth."""
    from astropy.io import fits

    for path in sorted(fd.FITS_DIR.glob("*.fits")):
        hdr = fits.open(path)[0].header
        d_ra, d_dec = fd._pixel_deltas(fd._celestial_wcs(hdr))
        assert d_ra < 0 < d_dec, f"{path.name}: RA must run East-left, DEC North-up"
        if "g395" in path.name:
            assert hdr["CDELT1"] > 0 and hdr["PC1_1"] == -1, \
                "the premise of this test: raw CDELT1 would give the wrong sign here"


@pytestmark_fits
@pytest.mark.parametrize("disk", ["hvtauc", "oph163131"])
def test_unobserved_pixels_are_noise_filled_and_flagged(disk):
    """No NaN reaches a consumer, the fill is background-level, and `observed` says where it is."""
    for key, b in fd.read_disk(disk).items():
        assert np.isfinite(b["img"]).all(), f"{disk} {key}: a NaN survived the fill"
        n_filled = int((~b["observed"]).sum())
        assert n_filled == 0 or key == "jwst", f"{disk} {key}: only JWST should need filling"
        if n_filled:
            filled = b["img"][~b["observed"]]
            # Background-level, not signal: within a few sigma of the measured sky.
            assert np.abs(filled - b["bkg"][0]).max() < 6.0 * b["bkg"][1]
            assert np.abs(filled).max() < 0.05 * b["img"].max()

    # Seeded on the disk name, not on `hash()`, which is salted per process.
    assert np.array_equal(fd.read_jwst(disk)["img"], fd.read_jwst(disk)["img"])
    raw = fd.read_jwst(disk, fill_unobserved=False)
    assert not np.isfinite(raw["img"]).all(), "the fill must be doing something"


@pytestmark_fits
@pytest.mark.parametrize("disk", ["hvtauc", "oph163131"])
def test_measured_alma_noise_is_inside_the_trained_prior(disk):
    """
    The measured per-band rms must sit in `experiment=protoplan_newbeam`'s noise prior.

    Sensitive to the annulus: the B9 maps are pbcor'd against a ~8.7" primary beam, so measuring
    the sky at 3-6" (what the older, coarser maps used) inflates sigma by ~1.6x and puts both disks
    out of prior. `NOISE_ANNULUS_ARCSEC` is off-source but inside the flat part.
    """
    prior = {0: (4.0e-4, 7.0e-4), 1: (5.0e-5, 1.1e-4), 2: (1.0e-5, 1.0e-4)}   # Jy/beam
    for band, j in fd.BAND_CHANNEL.items():
        b = fd.read_alma(disk, band)
        rms = b["bkg"][1] * 1e6 * b["omega_beam_sr"]
        lo, hi = prior[j]
        assert lo <= rms <= hi, f"{disk} {band}: rms {rms:.2e} Jy/beam outside [{lo:.1e}, {hi:.1e}]"


@pytestmark_fits
@pytest.mark.parametrize("disk", ["hvtauc", "oph163131"])
def test_jwst_noise_is_the_assumed_floor_not_the_measured_halo(disk):
    """
    The JWST sigma is a declared assumption (0.8 MJy/sr), not what the annulus reads.

    Even the off-source annulus sits on these disks' extended envelopes, so it measures the halo's
    scatter -- 2.2 MJy/sr for hvtauc, outside the [0.5, 1.0] prior the models train under.  Same
    call upstream made for the saucer (0.03 for a measured 0.98).  The override is width-only: the
    fill level stays the measured local median, so the filled corners join continuously onto the
    observed pixels rather than stepping to some global number.
    """
    b = fd.read_jwst(disk)
    assert b["bkg"][1] == fd.JWST_NOISE_MJY_SR[disk] == 0.8
    assert 0.5 <= b["bkg"][1] <= 1.0, "the assumed floor must be inside the trained prior"
    assert b["measured_sigma"] != b["bkg"][1], "what the annulus read must still be reported"
    # The registry hands the network the same number the fill used -- two places, one value.
    assert rd.DISKS[disk]["sigma_jwst"] == b["bkg"][1]
    assert rd.DISKS[disk]["dist_pc"] == 140.0 and rd.DISKS[disk]["av"] == 4.0

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


def test_oph163131_has_no_b7_anywhere():
    assert rd.missing_alma_channels("oph163131") == (1,)
    assert "im_jy_alma_1" not in rd.present_image_keys("oph163131")


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

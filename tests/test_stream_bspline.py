"""``stream_bspline_grid``: LSQ and penalized-smoothing modes share layout; smoothing reproduces a smooth track."""
import os

import numpy as np
import pytest
from omegaconf import OmegaConf

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from hydrabflow.augmentation.stream_bspline import _gcv_lambda, _np_basis, _second_diff_penalty, _uniform_knots  # noqa: E402
from hydrabflow.registry import AUGMENTATIONS  # noqa: E402

REAL = "assets/gaia/gaia_observed_streams_6Dwitherrors_cutNGC3201.npz"
pytestmark = pytest.mark.skipif(not os.path.exists(REAL), reason="real member npz not present")


def _params(**extra):
    p = {"target_streams": {"Pal5": 0, "NGC3201": 1, "M68": 2}, "real_streams_file": REAL,
         "bspline_interior_knots": 5, "bspline_vlos_interior_knots": 1, "bspline_grid_points": 20}
    p.update(extra)
    return OmegaConf.create(p)


def _real_batch():
    d = np.load(REAL)
    sim, am, vm = d["sim_data_projected"], d["attention_mask"], d["vlos_mask"]
    sim = sim[0] if sim.ndim == 4 else sim
    am = am[:, None, :] if am.ndim == 2 else am
    vm = vm[:, None, :] if vm.ndim == 2 else vm
    return {"sim_data_projected": np.asarray(sim, np.float32), "attention_mask": am, "vlos_mask": vm,
            "j": np.asarray(d["j"]).reshape(-1, 1).astype(np.float32)}


def test_smoothing_mode_same_layout_and_agrees_with_lsq():
    b_lsq = AUGMENTATIONS.get("stream_bspline_grid")(_params(), np.random.default_rng(0))(dict(_real_batch()))
    b_sm = AUGMENTATIONS.get("stream_bspline_grid")(_params(bspline_fit="smoothing"), np.random.default_rng(0))(dict(_real_batch()))
    a, b = np.asarray(b_lsq["sim_summary"]), np.asarray(b_sm["sim_summary"])
    assert a.shape == b.shape == (3, 20, 9)
    assert np.all(np.isfinite(b))
    assert np.array_equal(a[..., 5:], b[..., 5:])                 # valid flags, j, grid identical
    # the two estimators agree on the astrometric tracks well inside the member scatter
    for j in range(3):
        v = a[j, :, 5] > 0
        for k in range(4):
            assert np.sqrt(np.mean((a[j, v, k] - b[j, v, k]) ** 2)) < 0.2


def test_pspline_with_gcv_lambda_matches_scipy_smoothing_spline():
    from scipy.interpolate import make_smoothing_spline

    rng = np.random.default_rng(1)
    x = np.sort(rng.uniform(-10, 10, 400))
    y = np.sin(0.5 * x) + rng.normal(0, 0.1, x.size)
    t = _uniform_knots(np.array([[-10.0, 10.0]]), 20)[0]
    om = _second_diff_penalty(t[None])[0]
    lam = _gcv_lambda(x, y, t)
    B = _np_basis(x, t)
    grid = np.linspace(-9, 9, 30)
    psp = _np_basis(grid, t) @ np.linalg.solve(B.T @ B + lam * om + 1e-3 * np.eye(B.shape[1]), B.T @ y)
    assert np.sqrt(np.mean((psp - make_smoothing_spline(x, y, lam=lam)(grid)) ** 2)) < 0.03
    assert lam > _gcv_lambda(x, np.sin(3 * x) + rng.normal(0, 0.02, x.size), t)   # wigglier data -> smaller λ


def test_pinned_polynomial_model_is_an_exact_polynomial_fit():
    """bspline_models pins M68 v_los to a global line: the grid values ARE the LSQ line of the real members."""
    from hydrabflow.augmentation.stream_bspline import _np_project, _stream_frames
    from hydrabflow.augmentation.stream_summary import _DEFAULT_CHANNELS

    p = _params(bspline_fit="smoothing", bspline_models={"M68": {"vlos": "poly1"}})
    out = np.asarray(AUGMENTATIONS.get("stream_bspline_grid")(p, np.random.default_rng(0))(dict(_real_batch()))["sim_summary"])
    base = np.asarray(AUGMENTATIONS.get("stream_bspline_grid")(_params(bspline_fit="smoothing"), np.random.default_rng(0))(dict(_real_batch()))["sim_summary"])
    assert np.array_equal(out[:2], base[:2]) and np.array_equal(out[2, :, :4], base[2, :, :4])   # only M68 v_los moved
    d = np.load(REAL)
    fr = _stream_frames(p, 6, 2, dict(_DEFAULT_CHANNELS))
    m = (d["attention_mask"].reshape(3, -1)[2] > 0) & (d["vlos_mask"].reshape(3, -1)[2] > 0)
    x, *_, v = _np_project(fr.R[2], np.asarray(d["sim_data_projected"]).reshape(3, -1, 6)[2][m], dict(_DEFAULT_CHANNELS))
    line = np.polyval(np.polyfit(x, v, 1), np.linspace(x.min(), x.max(), 20))   # v_los grid = its own phi1 range
    ok = out[2, :, 6] > 0
    assert np.max(np.abs(out[2, ok, 4] - line[ok])) < 0.05         # km/s (float32 + 1e-3 ridge)


def test_cv_selection_prefers_simple_models_for_sparse_vlos():
    from hydrabflow.augmentation.stream_bspline import _select_model, _uniform_knots, _second_diff_penalty
    rng = np.random.default_rng(3)
    x = np.sort(rng.uniform(0, 80, 29)); t = _uniform_knots(np.array([[0.0, 80.0]]), 6)[0]
    pen = 5.0 * _second_diff_penalty(t[None])[0]
    lin = _select_model(x, -110 + 0.9 * x + rng.normal(0, 15, x.size), t, pen, 1e-3, 10, 5, 0, ["poly1", "poly2", "poly3", "spline"])[0]
    xs = np.sort(rng.uniform(0, 80, 400))
    wig = _select_model(xs, np.sin(xs / 5) + rng.normal(0, 0.05, xs.size), t, pen, 1e-3, 10, 3, 0, ["poly1", "poly2", "poly3", "spline"])[0]
    assert lin == "poly1" and wig == "spline"


def test_core_mode_streamfinder_frame_and_full_range_quadratic():
    """core080 preset params: finite layout; M68 v_los == the LSQ quadratic of ALL its measured stars in the
    published Fjorm frame; tracks with the core differ from the full-range fit only near the ends."""
    from hydrabflow.simulators.stream_frame import frames, project
    p = _params(bspline_fit="smoothing", summary_frame="streamfinder", bspline_core_frac=0.8,
                bspline_smoothing_lambda_scale=10.0, bspline_vlos_full_range=["M68"], bspline_models={"M68": {"vlos": "poly2"}})
    out = np.asarray(AUGMENTATIONS.get("stream_bspline_grid")(p, np.random.default_rng(0))(dict(_real_batch()))["sim_summary"])
    assert out.shape == (3, 20, 9) and np.all(np.isfinite(out))
    d = np.load(REAL)
    st = np.asarray(d["sim_data_projected"]).reshape(3, -1, 6)[2]
    m = (d["attention_mask"].reshape(3, -1)[2] > 0) & (d["vlos_mask"].reshape(3, -1)[2] > 0)
    phi1 = project(frames(("M68",))["M68"], st[m, 0], st[m, 1], st[m, 3], st[m, 4])[0]
    quad = np.polyval(np.polyfit(phi1, st[m, 5], 2), np.linspace(phi1.min(), phi1.max(), 20))
    assert np.max(np.abs(out[2, :, 4] - quad)) < 0.05              # km/s

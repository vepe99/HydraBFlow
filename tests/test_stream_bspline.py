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

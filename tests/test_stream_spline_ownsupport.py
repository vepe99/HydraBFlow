"""``stream_spline_ownsupport``: fits on each row's own core, evaluates on the real phi1 grid."""
import os

import numpy as np
import pytest
from omegaconf import OmegaConf

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from hydrabflow.registry import AUGMENTATIONS  # noqa: E402
from test_stream_bspline import REAL, _real_batch  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.exists(REAL), reason="real member npz not present")


def _aug(**extra):
    p = {"target_streams": {"Pal5": 0, "NGC3201": 1, "M68": 2}, "real_streams_file": REAL, "summary_frame": "streamfinder",
         "bspline_grid_points": 20, "spline_core_frac": 0.85, "spline_s_track": 1e-4}
    p.update(extra)
    return AUGMENTATIONS.get("stream_spline_ownsupport")(OmegaConf.create(p), np.random.default_rng(0))


def test_real_members_valid_on_their_own_grid_and_layout():
    o = np.asarray(_aug()(_real_batch())["sim_summary"])
    assert o.shape == (3, 20, 9) and np.all(np.isfinite(o))
    assert np.all(np.diff(o[..., 8], axis=1) >= 0)                  # grid = sorted real phi1 picks
    assert (o[..., 5] > 0).mean() > 0.9                             # the real stream covers its own core grid


def test_truncated_stream_is_invalid_outside_its_own_core():
    b = _real_batch()
    a, o_full = _aug(), None
    o_full = np.asarray(a(dict(b))["sim_summary"])
    grid = o_full[0, :, 8]
    # keep only Pal5 stars in the lower half of its grid: the upper grid points must turn invalid, values 0
    am = np.asarray(b["attention_mask"]).copy()
    from hydrabflow.augmentation.stream_bspline import _np_project, _real_member_xy  # noqa: F401
    import hydrabflow.simulators.stream_frame as sf
    R = sf.frames()["Pal5"]
    s = b["sim_data_projected"][0].astype(float)
    phi1 = sf.project(R, s[:, 0], s[:, 1], s[:, 3], s[:, 4])[0]
    am[0, 0] *= (phi1 < np.median(grid))
    b["attention_mask"] = am
    o = np.asarray(a(b)["sim_summary"])
    hi = grid > np.median(grid) + 1
    assert np.all(o[0, hi, 5] < 0) and np.all(o[0, hi, :4] == 0)

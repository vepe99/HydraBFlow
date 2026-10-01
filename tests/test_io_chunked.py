import os

import numpy as np

from hydrabflow.pipeline.io import concatenate_npz_streaming, load_dataset, run_chunked


def _sample(n, rng):
    return {"theta": rng.normal(size=(n, 1)), "x": rng.normal(size=(n, 7, 3)).astype(np.float32),
            "j": rng.integers(0, 3, size=n)}


def test_run_chunked_streaming_assembly_matches_concatenate(tmp_path):
    out = str(tmp_path / "d.npz")
    run_chunked(out, 25, 10, _sample, base_seed=3)
    got = load_dataset(out)
    ref = [_sample(n, np.random.default_rng(np.random.SeedSequence([3, s])))
           for s, n in [(0, 10), (10, 10), (20, 5)]]
    for k in ref[0]:
        want = np.concatenate([r[k] for r in ref])
        assert got[k].dtype == want.dtype and got[k].shape == want.shape
        np.testing.assert_array_equal(got[k], want)
    assert not os.path.exists(str(tmp_path / "d.chunks"))


def test_concatenate_npz_streaming_equals_savez(tmp_path):
    rng = np.random.default_rng(0)
    parts = [_sample(n, rng) for n in (4, 1, 6)]
    paths = []
    for i, p in enumerate(parts):
        paths.append(str(tmp_path / f"c{i}.npz"))
        np.savez(paths[-1], **p)
    concatenate_npz_streaming(str(tmp_path / "all.npz"), paths)
    got = np.load(str(tmp_path / "all.npz"))
    assert set(got.files) == set(parts[0])
    for k in parts[0]:
        np.testing.assert_array_equal(got[k], np.concatenate([p[k] for p in parts]))

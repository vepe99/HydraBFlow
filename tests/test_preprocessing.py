"""Preprocessing pipeline: fit/transform/split + state save/load round-trip."""

from __future__ import annotations

import numpy as np

from hydrabflow.registry import build_pipeline


def _toy_data(n=100, with_nan=True):
    rng = np.random.default_rng(0)
    data = {
        "theta1": rng.normal(size=(n, 1)),
        "theta2": rng.normal(size=(n, 1)),
        "x": rng.normal(loc=5.0, scale=2.0, size=(n, 3)),
    }
    if with_nan:
        data["x"][0, 0] = np.nan  # one bad simulation to be dropped
    return data


def test_pipeline_fit_transform_and_split(cfg):
    pipeline = build_pipeline(cfg.preprocessing)
    train, val = pipeline.fit_transform(_toy_data(), np.random.default_rng(1))

    # NaN row dropped (100 -> 99), then 10% held out for validation.
    assert len(train["x"]) + len(val["x"]) == 99
    assert len(val["x"]) == round(99 * cfg.training.validation_fraction)

    # Standardized observable is ~zero mean / unit std on the train split.
    assert np.allclose(train["x"].mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(train["x"].std(axis=0), 1.0, atol=1e-6)


def test_state_roundtrip(cfg, tmp_path):
    rng = np.random.default_rng(2)
    p1 = build_pipeline(cfg.preprocessing)
    train, _ = p1.fit_transform(_toy_data(with_nan=False), rng)
    state_path = str(tmp_path / "state.npz")
    p1.save(state_path)

    # Fresh pipeline loads fitted stats and reproduces the same transform (no re-fit/split).
    p2 = build_pipeline(cfg.preprocessing)
    p2.load(state_path)
    raw = _toy_data(with_nan=False)
    transformed = p2.transform(raw)
    # Compare against manually standardizing raw with p1's fitted standardizer.
    std_step = p1.steps[-1]
    expected = std_step.transform(raw)["x"]
    assert np.allclose(transformed["x"], expected)

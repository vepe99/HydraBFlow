"""Unit tests for the summary-space misspecification helpers (no bayesflow import needed:
the comparison function is injected, and the module only imports numpy at import time)."""

import numpy as np

from hydrabflow.pipeline.misspecification import mmd_test, per_member_scores


def _mean_dist(a, b) -> float:
    """Cheap stand-in for MMD: distance between sample means."""
    return float(np.linalg.norm(np.mean(a, axis=0) - np.mean(b, axis=0)))


def _three_strata(rng):
    reference = np.concatenate(
        [rng.normal(loc, 0.1, size=(50, 4)) for loc in (0.0, 5.0, 10.0)]
    )
    reference_j = np.repeat([0, 1, 2], 50)
    return reference, reference_j


def test_stratified_null_differs_from_plain():
    rng = np.random.default_rng(0)
    reference, reference_j = _three_strata(rng)
    observed = np.stack([np.full(4, 0.0), np.full(4, 5.0), np.full(4, 10.0)])

    out = mmd_test(
        observed,
        reference,
        observed_j=[0, 1, 2],
        reference_j=reference_j,
        num_null=100,
        comparison_fn=_mean_dist,
        rng=np.random.default_rng(1),
    )
    assert out["n_observed"] == 3 and out["n_reference"] == 150
    assert "null_stratified" in out and "p_value_stratified" in out
    # A stratified trio (one member per stratum) sits near the pooled mean; a uniform trio
    # usually has duplicate strata and deviates more -> the plain null is strictly wider.
    assert np.mean(out["null_plain"]) > np.mean(out["null_stratified"])
    # The observed trio matches the strata means exactly -> not atypical under either null.
    assert out["p_value_stratified"] > 0.5


def test_mmd_test_without_strata_has_no_stratified_keys():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=(100, 4))
    observed = rng.normal(size=(3, 4))
    out = mmd_test(observed, reference, num_null=20, comparison_fn=_mean_dist, rng=rng)
    assert "null_stratified" not in out and "p_value_stratified" not in out
    assert 0.0 <= out["p_value_plain"] <= 1.0


def test_per_member_scores_flags_the_outlying_member():
    rng = np.random.default_rng(0)
    reference = np.concatenate([rng.normal(0, 1, (200, 3)), rng.normal(5, 1, (200, 3))])
    reference_j = np.repeat([0, 1], 200)
    observed = np.array([[0.0, 0.0, 0.0], [50.0, 50.0, 50.0]])  # member 1 is far OOD

    scores = per_member_scores(observed, reference, [0, 1], reference_j)
    assert set(scores) == {0, 1}
    assert scores[0]["percentile"] < 50.0
    assert scores[1]["percentile"] == 100.0
    assert scores[1]["mahalanobis"] > scores[1]["reference_median"] * 5

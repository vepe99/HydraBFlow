"""Modality masking + the compositional over-counting it fixes.

The last test is the point of the whole feature: on a conjugate Gaussian where the exact posterior
is known, the stock compositional identity applied to (stream_j, curve) items is provably too
narrow, and the mask plan recovers the right width.
"""

import numpy as np
import pytest

pytest.importorskip("bayesflow")

from hydrabflow.networks.grouped_diffusion import CONDITIONS_GROUP, GroupedDiffusionModel  # noqa: E402


def _net(**kwargs):
    return GroupedDiffusionModel(
        group_sizes=[3, 4], group_names=["sim_summary", "vcirc_kms"], **kwargs
    )


def test_lead_width_inferred_from_condition_width():
    net = _net()
    assert net._resolve_lead(7) == 0
    assert net._all_names(7) == ["sim_summary", "vcirc_kms"]

    net = _net()
    assert net._resolve_lead(9) == 2  # two leading inference_conditions columns
    assert net._all_names(9) == [CONDITIONS_GROUP, "sim_summary", "vcirc_kms"]
    assert net._all_sizes(9) == [2, 3, 4]


def test_lead_width_negative_is_an_error():
    with pytest.raises(ValueError, match="only 5 wide"):
        _net()._resolve_lead(5)


def test_group_mask_covers_exactly_its_columns():
    net = _net()
    np.testing.assert_array_equal(net.group_mask(["sim_summary"], 7), [[1, 1, 1, 0, 0, 0, 0]])
    np.testing.assert_array_equal(net.group_mask(["vcirc_kms"], 7), [[0, 0, 0, 1, 1, 1, 1]])
    np.testing.assert_array_equal(net.group_mask([], 7), [[0] * 7])


def test_group_mask_always_keeps_the_leading_conditions():
    # The leading group is inference_conditions, not a modality: masking it out would ask the
    # network to marginalise over something it is conditioned on.
    net = _net()
    np.testing.assert_array_equal(net.group_mask(["vcirc_kms"], 9), [[1, 1, 0, 0, 0, 1, 1, 1, 1]])


def test_group_mask_rejects_an_unknown_group():
    with pytest.raises(ValueError, match="unknown condition group"):
        _net().group_mask(["nope"], 7)


def test_group_names_must_index_the_same_groups():
    with pytest.raises(ValueError, match="index the same groups"):
        GroupedDiffusionModel(group_sizes=[3, 4], group_names=["only_one"])


def test_training_mask_is_group_coherent_and_never_empty():
    from keras import ops

    net = _net(missing_modality_prob=0.5)
    mask = np.asarray(ops.convert_to_numpy(net.observed_condition_mask(256, 7)))
    assert mask.shape == (256, 7)
    # Each group is all-on or all-off: a modality is dropped as a unit, never feature-by-feature.
    assert set(np.unique(mask[:, :3].sum(axis=1))) <= {0, 3}
    assert set(np.unique(mask[:, 3:].sum(axis=1))) <= {0, 4}
    # keep_one: no row is left with nothing observed (that would be a pure-prior training sample).
    assert mask.sum(axis=1).min() > 0
    # Both drop directions actually occur at p=0.5 over 256 rows.
    assert 0 < mask[:, 0].mean() < 1 and 0 < mask[:, 3].mean() < 1


def test_zero_prob_disables_masking():
    net = _net(missing_modality_prob=0.0)
    x = np.ones((4, 7), dtype="float32")
    assert net._apply(x, None) is x


def test_apply_tiles_a_mask_whose_batch_divides_the_conditions():
    # `sample` repeats the conditions once per posterior draw; the caller should not have to
    # reshape the mask to match.
    from keras import ops

    net = _net()
    conditions = np.ones((6, 7), dtype="float32")
    out = np.asarray(ops.convert_to_numpy(net._apply(conditions, net.group_mask(["sim_summary"], 7))))
    assert out.shape == (6, 7)
    assert np.all(out[:, :3] == 1) and np.all(out[:, 3:] == 0)


def test_item_mask_plan_length_must_match_the_items():
    import keras

    net = _net(missing_modality_prob=0.1)
    net.set_item_mask_plan([["sim_summary"]] * 2)
    with pytest.raises(ValueError, match="2 entries but there are 3"):
        net.compositional_score(
            keras.ops.zeros((2, 5)), 0.5, keras.ops.zeros((2, 3, 7)),
            compute_prior_score=lambda x, t: x,
        )


def test_per_item_mask_flattens_like_the_conditions():
    """A rank-3 plan mask must flatten to the same row order as `conditions`.

    `_compositional_score_direct` reshapes conditions (batch, items, width) -> (batch*items, width),
    so flat row b*items+i belongs to item i. If the mask were repeated per batch row instead (what
    upstream does for a per-row mask) every item would get item 0's mask.
    """
    from keras import ops

    net = _net()
    plan = np.stack([net.group_mask(["sim_summary"], 7)[0]] * 3 + [net.group_mask(["vcirc_kms"], 7)[0]])
    batch = 2
    kwargs = {"observed_condition_mask": ops.convert_to_tensor(np.repeat(plan[None], batch, axis=0))}
    out = net._repeat_mask_kwargs_over_items(kwargs, 4)["observed_condition_mask"]
    flat = np.asarray(ops.convert_to_numpy(out))
    assert flat.shape == (batch * 4, 7)
    for b in range(batch):
        for i in range(4):
            np.testing.assert_array_equal(flat[b * 4 + i], plan[i])


def test_serialization_round_trip_keeps_the_layout():
    net = _net(missing_modality_prob=0.3)
    net._resolve_lead(9)
    clone = GroupedDiffusionModel.from_config(net.get_config())
    assert clone.group_sizes == [3, 4]
    assert clone.group_names == ["sim_summary", "vcirc_kms"]
    assert clone.missing_modality_prob == 0.3
    assert clone.lead_width == 2  # resolved layout survives, so a reload needs no training batch


# --------------------------------------------------------------------------- the actual bug

def _gaussian_widths(m=3, s=1.0, c=0.3):
    """Posterior sd of theta~N(0,1) given m streams (noise s) and one curve (noise c).

    Returns (exact, what the stock compositional identity produces when the curve is copied into
    every member). Precisions add for conjugate Gaussians, and each extra copy of the curve adds
    another 1/c^2.
    """
    return (1 + m / s**2 + 1 / c**2) ** -0.5, (1 + m / s**2 + m / c**2) ** -0.5


@pytest.mark.parametrize("c", [1.0, 0.5, 0.3, 0.1])
def test_duplicated_curve_narrows_the_compositional_posterior(c):
    exact, stock = _gaussian_widths(c=c)
    assert stock < exact  # always too confident, never too wide


def test_overcounting_bias_is_bounded_by_sqrt_m_and_vanishes_without_the_curve():
    m = 3
    # Curve-dominated: the bias saturates at sqrt(m) == 73% too narrow.
    exact, stock = _gaussian_widths(m=m, c=1e-3)
    assert abs(exact / stock - np.sqrt(m)) < 1e-3
    # Uninformative curve: nothing to over-count, so no bias.
    exact, stock = _gaussian_widths(m=m, c=1e6)
    assert abs(exact / stock - 1) < 1e-6


def test_mask_plan_gives_each_likelihood_exactly_one_item():
    """The fix, at the level of what the mask plan actually asserts.

    m member items observing only the stream + one item observing only the curve = each factor
    once, which is the `exact` column of `_gaussian_widths`.
    """
    net = _net()
    m = 3
    plan = [["sim_summary"]] * m + [["vcirc_kms"]]
    net.set_item_mask_plan(plan)
    masks = np.concatenate([net.group_mask(p, 7) for p in net._item_mask_plan], axis=0)
    # Column 0 belongs to sim_summary, column 3 to vcirc_kms.
    assert masks[:, 0].sum() == m, "each stream appears in exactly one item"
    assert masks[:, 3].sum() == 1, "the curve appears in exactly ONE item, not m"


# ------------------------------------------------------- end to end through the score path

def _capture_score_calls(monkeypatch):
    """Record what the stock ``DiffusionModel.score`` sees, and return zeros of the right shape."""
    import bayesflow as bf
    from keras import ops

    seen = []

    def fake_score(self, xz, time=None, log_snr_t=None, conditions=None, training=False, **kw):
        seen.append({"conditions": conditions, **kw})
        return ops.zeros_like(xz)

    monkeypatch.setattr(bf.networks.DiffusionModel, "score", fake_score)
    return seen


@pytest.mark.parametrize(
    "subnet,zeroes_conditions",
    [("time_mlp", True), ("diffusion_transformer", False)],
)
def test_compositional_score_masks_each_item(monkeypatch, subnet, zeroes_conditions):
    """The plan must reach the score call, per item, in the flattened row order.

    With an MLP subnet the mask cannot reach the subnet (`_collect_mask_kwargs` filters it out),
    so "absent" is encoded by zeroing the group's columns. With the transformer the values are
    left alone and the mask excludes those tokens from attention instead -- the two are exclusive.
    """
    from keras import ops

    seen = _capture_score_calls(monkeypatch)
    net = _net(missing_modality_prob=0.25, subnet=subnet)
    net.set_item_mask_plan([["sim_summary"]] * 3 + [["vcirc_kms"]])

    batch, num_items, width = 2, 4, 7
    conditions = ops.convert_to_tensor(np.ones((batch, num_items, width), dtype="float32"))
    net.compositional_score(
        ops.zeros((batch, 5)), 0.5, conditions,
        compute_prior_score=lambda x, t: ops.zeros_like(x),
    )

    assert len(seen) == 1, "one flattened score call over batch*items rows"
    call = seen[0]
    expected = np.concatenate(
        [net.group_mask(["sim_summary"], width)] * 3 + [net.group_mask(["vcirc_kms"], width)]
    )
    if zeroes_conditions:
        got = np.asarray(ops.convert_to_numpy(call["conditions"]))
        assert got.shape == (batch * num_items, width)
        for b in range(batch):
            np.testing.assert_array_equal(got[b * num_items : (b + 1) * num_items], expected)
    else:
        got = np.asarray(ops.convert_to_numpy(call["observed_condition_mask"]))
        assert got.shape == (batch * num_items, width)
        for b in range(batch):
            np.testing.assert_array_equal(got[b * num_items : (b + 1) * num_items], expected)
        # values untouched: the transformer masks tokens, it does not need them zeroed
        assert np.all(np.asarray(ops.convert_to_numpy(call["conditions"])) == 1)


def test_no_plan_leaves_the_compositional_path_untouched(monkeypatch):
    from keras import ops

    seen = _capture_score_calls(monkeypatch)
    net = _net(missing_modality_prob=0.25, subnet="time_mlp")
    conditions = ops.convert_to_tensor(np.ones((2, 3, 7), dtype="float32"))
    net.compositional_score(
        ops.zeros((2, 5)), 0.5, conditions,
        compute_prior_score=lambda x, t: ops.zeros_like(x),
    )
    assert "observed_condition_mask" not in seen[0]
    assert np.all(np.asarray(ops.convert_to_numpy(seen[0]["conditions"])) == 1)


def test_plan_adds_a_zero_mask_row_for_the_network_prior(monkeypatch):
    """Without an explicit prior score, upstream appends a zero-condition item to get the prior
    from the network itself -- nothing is observed on that row, so its mask must be all zeros."""
    from keras import ops

    seen = _capture_score_calls(monkeypatch)
    net = _net(subnet="diffusion_transformer")
    net.set_item_mask_plan([["sim_summary"], ["vcirc_kms"]])
    net.compositional_score(
        ops.zeros((1, 5)), 0.5, ops.convert_to_tensor(np.ones((1, 2, 7), dtype="float32")),
        compute_prior_score=None,
    )
    got = np.asarray(ops.convert_to_numpy(seen[0]["observed_condition_mask"]))
    assert got.shape == (3, 7)  # 2 items + the prior row
    np.testing.assert_array_equal(got[2], np.zeros(7))

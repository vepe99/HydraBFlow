"""Stream-project components: config composition, hierarchy derivation, per-stream
normalization, and compositional helpers. No forward simulation (agama-free)."""

from __future__ import annotations

import numpy as np
import pytest

STREAM_OVERRIDES = [
    "simulator=stream_agama",
    "model=stream_fusion",
    "adapter=stream",
    "preprocessing=stream_local",
    "augmentation=stream_local",
    "training=stream_local",
]

GLOBALS = [
    "rho_TwoPowerTriaxial_halo",
    "gamma_TwoPowerTriaxial_halo",
    "a_TwoPowerTriaxial_halo",
    "q_TwoPowerTriaxial_halo",
    "r_Disk",
    "z_Disk",
    "Sigma_Disk",
]
LOCALS = ["vr", "r", "mu_ra_cosdec", "mu_dec"]


def test_stream_config_composes(compose):
    cfg = compose(STREAM_OVERRIDES)
    assert cfg.simulator.name == "stream_agama"
    assert cfg.model.summary_network.type == "fusion"
    assert cfg.adapter.attention_mask_key == "attention_mask"
    assert cfg.composition.level == "none"  # default group value


def test_simulator_declares_hierarchy(compose):
    from hydrabflow.simulators.registry import get_simulator

    cfg = compose(STREAM_OVERRIDES, fill=False)
    sim = get_simulator(cfg.simulator)
    assert sim.global_parameter_names == GLOBALS
    assert sim.local_parameter_names == LOCALS
    assert sim.context_keys == ["j"]
    assert sim.observable_keys == ["sim_data_projected", "vcirc_kms"]
    assert set(sim.prior_spec_global) == set(GLOBALS)
    assert set(sim.prior_spec_local) == {"Pal5", "NGC3201", "M68"}


def test_adapter_derivation_follows_composition_level(compose):
    cfg = compose(STREAM_OVERRIDES + ["composition=global"])
    assert list(cfg.adapter.inference_variables) == GLOBALS
    assert list(cfg.adapter.inference_conditions) == ["j"]

    cfg = compose(STREAM_OVERRIDES + ["composition=local"])
    assert list(cfg.adapter.inference_variables) == LOCALS
    assert list(cfg.adapter.inference_conditions) == GLOBALS + ["j"]


def test_stream_prior_sampling_shapes():
    from hydrabflow.simulators.stream_common import (
        sample_stream_prior,
        sample_stream_prior_shared_global,
    )

    priors_global = {"a": {"type": "uniform", "prior_parameters": [0, 1]}}
    priors_local = {
        "s0": {"v": {"type": "normal", "prior_parameters": [0.0, 1.0]}},
        "s1": {"v": {"type": "normal", "prior_parameters": [5.0, 2.0]}},
    }
    streams = {"s0": 0, "s1": 1}
    rng = np.random.default_rng(0)

    flat = sample_stream_prior(priors_global, priors_local, streams, 8, rng)
    assert flat["a"].shape == (8, 1) and flat["j"].shape == (8, 1) and flat["v"].shape == (8, 1)

    grouped = sample_stream_prior_shared_global(priors_global, priors_local, streams, 4, rng)
    assert grouped["a"].shape == (4, 1)
    assert grouped["j"].shape == (4, 2, 1) and grouped["v"].shape == (4, 2, 1)
    assert (grouped["j"][:, 0, 0] == 0).all() and (grouped["j"][:, 1, 0] == 1).all()


def test_per_stream_parameter_standardize_roundtrip():
    from hydrabflow.preprocessing.streams import PerStreamParameterStandardize

    priors = {
        "s0": {"v": {"type": "normal", "prior_parameters": [10.0, 2.0]}},
        "s1": {"v": {"type": "normal", "prior_parameters": [-5.0, 0.5]}},
    }
    step = PerStreamParameterStandardize(priors, {"s0": 0, "s1": 1}, keys=["v"])

    data = {"v": np.array([[12.0], [-5.5]]), "j": np.array([[0.0], [1.0]])}
    out = step.transform(dict(data))
    np.testing.assert_allclose(out["v"], [[1.0], [-1.0]])
    back = step.inverse_transform(out)
    np.testing.assert_allclose(back["v"], data["v"])

    # Posterior-samples layout: (rows, num_samples, 1) with j (rows, 1).
    posterior = {"v": np.zeros((2, 5, 1)), "j": data["j"]}
    phys = step.inverse_transform(posterior)
    np.testing.assert_allclose(phys["v"][0], 10.0)
    np.testing.assert_allclose(phys["v"][1], -5.0)


def test_stream_observation_stats_fit_and_state_roundtrip():
    from hydrabflow.preprocessing.streams import StreamObservationStats

    rng = np.random.default_rng(1)
    data = {
        "sim_data_projected": rng.normal(3.0, 2.0, size=(10, 7, 6)),
        "vcirc_kms": 10 ** rng.normal(2.3, 0.05, size=(10, 4, 1)),
        "j": np.repeat([[0.0], [1.0]], 5, axis=0),
    }
    step = StreamObservationStats()
    step.fit(data)
    assert step.obs_mean.shape == (2, 6)
    assert step.vcirc_mean.shape == (4, 1)

    reloaded = StreamObservationStats()
    reloaded.load_state({k: v for k, v in step.state().items()})
    np.testing.assert_allclose(reloaded.obs_std, step.obs_std)


def test_flatten_and_group_members_roundtrip():
    from hydrabflow.pipeline.compositional import flatten_members, group_members

    n, m = 4, 3
    data = {
        "obs": np.arange(n * m * 5 * 2, dtype=float).reshape(n, m, 5, 2),
        "j": np.tile(np.arange(m, dtype=float).reshape(1, m, 1), (n, 1, 1)),
        "global_p": np.arange(n, dtype=float).reshape(n, 1),
        "vcirc": np.ones((n, 7, 1)),
    }
    flat = flatten_members(data, m)
    assert flat["obs"].shape == (n * m, 5, 2)
    assert flat["j"].shape == (n * m, 1)
    assert flat["global_p"].shape == (n * m, 1)  # repeated per member
    assert flat["vcirc"].shape == (n * m, 7, 1)

    grouped = group_members({"obs": flat["obs"]}, n, m)
    np.testing.assert_allclose(grouped["obs"], data["obs"])


def test_prior_score_from_spec():
    from hydrabflow.pipeline.compositional import prior_score_from_spec

    score = prior_score_from_spec(
        {
            "u": {"type": "uniform", "prior_parameters": [0, 1]},
            "g": {"type": "normal", "prior_parameters": [2.0, 0.5]},
        }
    )
    out = score({"u": np.ones((3, 1)), "g": np.full((3, 1), 3.0)})
    np.testing.assert_allclose(np.asarray(out["u"]), 0.0)
    np.testing.assert_allclose(np.asarray(out["g"]), -(3.0 - 2.0) / 0.25)


def test_prior_score_applies_time_decay():
    """The callable's signature names a ``time`` parameter, so BayesFlow's
    ``prior_has_time`` inspection (bayesflow.approximators.helpers.compositional) skips its own
    ``(1 - t)`` weighting and expects this function to apply it -- otherwise the raw prior score
    leaks in undamped at every diffusion step (the bug behind z_Disk's miscalibration)."""
    from hydrabflow.pipeline.compositional import prior_score_from_spec

    score = prior_score_from_spec(
        {"g": {"type": "normal", "prior_parameters": [2.0, 0.5]}}
    )
    raw = -(3.0 - 2.0) / 0.25
    out_t0 = score({"g": np.full((3, 1), 3.0)}, time=np.zeros((3, 1)))
    out_t1 = score({"g": np.full((3, 1), 3.0)}, time=np.ones((3, 1)))
    out_t_half = score({"g": np.full((3, 1), 3.0)}, time=np.full((3, 1), 0.5))
    np.testing.assert_allclose(np.asarray(out_t0["g"]), raw)
    np.testing.assert_allclose(np.asarray(out_t1["g"]), 0.0)
    np.testing.assert_allclose(np.asarray(out_t_half["g"]), 0.5 * raw)


def test_prior_score_from_kde_fits_in_log10_space(tmp_path):
    """The KDE prior score must be fit in the network's native space (log10 for ``log10_keys``),
    NOT physical units -- fitting in physical units was the bug that blew up compositional
    sampling (rho/Sigma worst) while base sampling stayed fine. Verify: (a) grad log p matches a
    finite-difference of the log-density recomputed independently in log10 space, and (b) the
    ``(1 - t)`` decay is applied by the callable itself."""
    from scipy.special import logsumexp

    from hydrabflow.pipeline.compositional import prior_score_from_kde

    rng = np.random.default_rng(0)
    n = 4000
    rho = 10 ** rng.uniform(6.0, 8.0, n)  # log10-key, huge physical scale
    gamma = rng.normal(0.0, 1.0, n)  # non-log key
    npz = tmp_path / "kde.npz"
    np.savez(npz, rho=rho, gamma=gamma)

    order = ["rho", "gamma"]
    score = prior_score_from_kde(
        str(npz), param_order=order, log10_keys={"rho"}, max_points=2000, seed=1
    )

    # Independently reconstruct the same diagonal KDE in log10 space for a finite-diff reference.
    X = np.stack([np.log10(rho), gamma], axis=1)
    X = X[np.random.default_rng(1).choice(X.shape[0], 2000, replace=False)]
    m, d = X.shape
    h = np.maximum((m ** (-1.0 / (d + 4))) ** 2 * X.var(0), 1e-12)

    def logpdf(pt):
        ex = -0.5 * np.sum((pt[None, :] - X) ** 2 / h, axis=1)
        return logsumexp(ex) - np.log(m) - 0.5 * np.sum(np.log(2 * np.pi * h))

    pts = np.array([[7.0, 0.2], [6.5, -1.0]])  # rho given in LOG10 space (as bayesflow passes it)
    eps = 1e-4
    fd = np.zeros_like(pts)
    for b in range(pts.shape[0]):
        for jx in range(d):
            p = pts[b].copy(); p[jx] += eps; hi = logpdf(p)
            p[jx] -= 2 * eps; lo = logpdf(p)
            fd[b, jx] = (hi - lo) / (2 * eps)

    theta = {"rho": pts[:, [0]], "gamma": pts[:, [1]]}
    g = score(theta)
    got = np.concatenate([np.asarray(g["rho"]), np.asarray(g["gamma"])], axis=1)
    np.testing.assert_allclose(got, fd, atol=2e-3)

    # (1 - t) decay owned by the callable.
    g_half = score(theta, time=np.full((2, 1), 0.5))
    np.testing.assert_allclose(np.asarray(g_half["rho"]), 0.5 * np.asarray(g["rho"]), rtol=1e-5)


def test_prior_score_from_kde_linear_space(tmp_path):
    """With no ``log10_keys`` every parameter stays in physical/linear space; the KDE must be fit
    and scored there directly (no log10 anywhere). Guards the non-log path before committing:
    grad log p must match a finite-difference of the linear-space diagonal KDE recomputed
    independently, including for a large-magnitude physical parameter."""
    from scipy.special import logsumexp

    from hydrabflow.pipeline.compositional import prior_score_from_kde

    rng = np.random.default_rng(2)
    n = 4000
    big = rng.uniform(1e6, 1e8, n)  # large-magnitude PHYSICAL param, deliberately NOT a log10-key
    small = rng.normal(0.0, 1.0, n)  # ordinary-scale physical param
    npz = tmp_path / "kde_linear.npz"
    np.savez(npz, big=big, small=small)

    order = ["big", "small"]
    score = prior_score_from_kde(str(npz), param_order=order, log10_keys=(), max_points=2000, seed=1)

    # Independently reconstruct the same diagonal KDE in physical (linear) space.
    X = np.stack([big, small], axis=1)
    X = X[np.random.default_rng(1).choice(X.shape[0], 2000, replace=False)]
    m, d = X.shape
    h = np.maximum((m ** (-1.0 / (d + 4))) ** 2 * X.var(0), 1e-12)

    def logpdf(pt):
        ex = -0.5 * np.sum((pt[None, :] - X) ** 2 / h, axis=1)
        return logsumexp(ex) - np.log(m) - 0.5 * np.sum(np.log(2 * np.pi * h))

    pts = np.array([[5.0e7, 0.2], [2.0e7, -1.0]])  # both params in physical space
    fd = np.zeros_like(pts)
    for b in range(pts.shape[0]):
        for jx in range(d):
            eps = 1e-4 * max(abs(pts[b, jx]), 1.0)  # scale-aware step for the huge-magnitude dim
            p = pts[b].copy(); p[jx] += eps; hi = logpdf(p)
            p[jx] -= 2 * eps; lo = logpdf(p)
            fd[b, jx] = (hi - lo) / (2 * eps)

    theta = {"big": pts[:, [0]], "small": pts[:, [1]]}
    g = score(theta)
    got = np.concatenate([np.asarray(g["big"]), np.asarray(g["small"])], axis=1)
    # Relative tolerance: the "big" gradient is ~1e-8-scale, the "small" one ~O(1);
    # atol tolerates float32 softmax vs float64 finite-diff on the O(1) dimension.
    np.testing.assert_allclose(got, fd, rtol=2e-3, atol=5e-3)


def test_prior_score_from_kde_jax_matches_finite_diff(tmp_path):
    """The jax.scipy.stats.gaussian_kde implementation must (a) fit in the network's native
    (log10) space and (b) return grad log p matching a finite-difference of jax's OWN logpdf,
    and (c) apply the (1 - t) decay itself."""
    import jax.numpy as jnp
    from jax.scipy.stats import gaussian_kde

    from hydrabflow.pipeline.compositional import prior_score_from_kde_jax

    rng = np.random.default_rng(3)
    n = 3000
    rho = 10 ** rng.uniform(6.0, 8.0, n)  # log10-key, huge physical scale
    gamma = rng.normal(0.0, 1.0, n)
    npz = tmp_path / "kde_jax.npz"
    np.savez(npz, rho=rho, gamma=gamma)

    order = ["rho", "gamma"]
    score = prior_score_from_kde_jax(
        str(npz), param_order=order, log10_keys={"rho"}, max_points=1500, seed=1
    )

    # Independently rebuild the SAME jax kde in log10 space for a finite-diff reference.
    X = np.stack([np.log10(rho), gamma], axis=1).astype("float32")
    X = X[np.random.default_rng(1).choice(X.shape[0], 1500, replace=False)]
    kde = gaussian_kde(jnp.asarray(X.T), bw_method="scott")

    pts = np.array([[7.0, 0.2], [6.5, -1.0]], dtype="float32")  # rho in LOG10 space
    eps = 1e-3
    fd = np.zeros_like(pts)
    for b in range(pts.shape[0]):
        for jx in range(2):
            p = pts[b].copy(); p[jx] += eps; hi = float(kde.logpdf(jnp.asarray(p[:, None]))[0])
            p[jx] -= 2 * eps; lo = float(kde.logpdf(jnp.asarray(p[:, None]))[0])
            fd[b, jx] = (hi - lo) / (2 * eps)

    theta = {"rho": pts[:, [0]], "gamma": pts[:, [1]]}
    g = score(theta)
    got = np.concatenate([np.asarray(g["rho"]), np.asarray(g["gamma"])], axis=1)
    np.testing.assert_allclose(got, fd, atol=8e-3)  # FD truncation on the fast log10 dim

    g_half = score(theta, time=np.full((2, 1), 0.5))
    np.testing.assert_allclose(np.asarray(g_half["rho"]), 0.5 * np.asarray(g["rho"]), rtol=1e-4)


def test_kde_jax_and_diagonal_agree_when_uncorrelated(tmp_path):
    """Full-covariance (jax) and diagonal (custom) KDEs reduce to the same estimator when the
    training draws are uncorrelated -- so their scores must agree there. This pins down that the
    two implementations differ ONLY by the off-diagonal covariance terms, not by a bug."""
    from hydrabflow.pipeline.compositional import (
        prior_score_from_kde,
        prior_score_from_kde_jax,
    )

    rng = np.random.default_rng(4)
    n = 6000  # large N so ddof=0 (custom) vs ddof=1 (jax cov) is negligible
    a = rng.normal(0.0, 2.0, n)  # independent columns -> ~diagonal covariance
    b = rng.normal(5.0, 0.5, n)
    npz = tmp_path / "kde_uncorr.npz"
    np.savez(npz, a=a, b=b)

    order = ["a", "b"]
    s_diag = prior_score_from_kde(str(npz), param_order=order, max_points=6000, seed=7)
    s_jax = prior_score_from_kde_jax(str(npz), param_order=order, max_points=6000, seed=7)

    theta = {"a": np.array([[0.5], [-1.0], [1.5]]), "b": np.array([[5.2], [4.6], [5.0]])}
    gd = s_diag(theta)
    gj = s_jax(theta)
    for k in order:
        np.testing.assert_allclose(np.asarray(gj[k]), np.asarray(gd[k]), rtol=5e-2, atol=1e-2)


def test_mask_vcirc_radii_trims_grid():
    from hydrabflow.preprocessing.streams import MaskVcircRadii
    from hydrabflow.simulators.stream_common import OBS_R_KPC

    step = MaskVcircRadii(r_min=5.5)
    data = {"vcirc_kms": np.ones((2, OBS_R_KPC.size, 1))}
    out = step.transform(data)
    assert out["vcirc_kms"].shape == (2, (OBS_R_KPC >= 5.5).sum(), 1)


def test_registries_contain_stream_components():
    import hydrabflow.augmentation  # noqa: F401  (discovery)
    import hydrabflow.preprocessing  # noqa: F401
    from hydrabflow.augmentation.registry import available_augmentations
    from hydrabflow.preprocessing.registry import available_steps
    from hydrabflow.simulators.registry import available_simulators

    assert "stream_agama" in available_simulators()
    for step in (
        "per_stream_parameter_standardize",
        "stream_observation_stats",
        "mask_vcirc_radii",
        "attach_observed_vcirc",
    ):
        assert step in available_steps()
    for aug in (
        "observational_window",
        "sample_magnitudes",
        "mask_vlos",
        "per_stream_standardize",
        "concatenate_stream_index",
    ):
        assert aug in available_augmentations()


def test_per_stream_parameter_standardize_rejects_non_normal():
    from hydrabflow.preprocessing.streams import PerStreamParameterStandardize

    priors = {"s0": {"v": {"type": "uniform", "prior_parameters": [0, 1]}}}
    with pytest.raises(ValueError, match="normal priors"):
        PerStreamParameterStandardize(priors, {"s0": 0}, keys=["v"])


def test_log10_transform_roundtrip():
    from hydrabflow.preprocessing.steps import Log10Transform

    step = Log10Transform(keys=["rho", "untouched_missing"])
    data = {"rho": np.array([[1.0], [100.0], [1000.0]]), "other": np.array([[1.0], [2.0], [3.0]])}
    out = step.transform(dict(data))
    np.testing.assert_allclose(out["rho"], [[0.0], [2.0], [3.0]])
    np.testing.assert_allclose(out["other"], data["other"])  # untouched key passes through

    back = step.inverse_transform(out)
    np.testing.assert_allclose(back["rho"], data["rho"])

    # Posterior-samples layout (rows, num_samples, 1) round-trips too.
    posterior = {"rho": np.zeros((3, 5, 1))}
    phys = step.inverse_transform(posterior)
    np.testing.assert_allclose(phys["rho"], 1.0)


def test_prior_score_log10_jacobian_correction():
    """d/dy log p_Y(y) for y=log10(x) must include the +ln(10) Jacobian term, not just the
    change-of-variables-scaled physical score — this is exactly the correction the user asked
    to double check before compositional-sampling evaluation of a log10-reparametrized prior."""
    import math

    from hydrabflow.pipeline.compositional import prior_score_from_spec

    ln10 = math.log(10.0)
    spec = {
        "u": {"type": "uniform", "prior_parameters": [0, 1]},
        "g": {"type": "normal", "prior_parameters": [2.0, 0.5]},
    }
    score = prior_score_from_spec(spec, log10_keys=["u", "g"])

    y_u = np.array([[0.3]])  # x = 10**0.3
    out_u = np.asarray(score({"u": y_u})["u"])
    np.testing.assert_allclose(out_u, ln10)  # uniform: base score 0 -> only the Jacobian term

    y_g = np.array([[0.5]])
    x_g = 10.0**y_g
    expected = x_g * ln10 * (-(x_g - 2.0) / 0.25) + ln10
    out_g = np.asarray(score({"g": y_g})["g"])
    np.testing.assert_allclose(out_g, expected)

    # Keys not in log10_keys are scored exactly as before (no Jacobian term).
    score_plain = prior_score_from_spec(spec)
    out_plain = np.asarray(score_plain({"u": y_u})["u"])
    np.testing.assert_allclose(out_plain, 0.0)


def test_stream_global_log10_and_nolos_presets_compose(compose):
    cfg = compose(
        [
            "simulator=stream_agama",
            "model=stream_fusion",
            "adapter=stream",
            "composition=global",
            "preprocessing=stream_global_log10",
            "augmentation=stream_global_nolos",
        ]
    )
    step_names = [s["name"] if hasattr(s, "get") else s.name for s in cfg.preprocessing.steps]
    assert "log10_transform" in step_names
    assert "rho_TwoPowerTriaxial_halo" in cfg.preprocessing.steps[-1]["keys"]

    aug_steps = list(cfg.augmentation.steps)
    assert "remove_los_velocity" in aug_steps
    assert "mask_vlos" not in aug_steps
    assert "concatenate_sigma_errors" not in aug_steps
    assert "concatenate_vlos_mask" not in aug_steps
    assert list(cfg.augmentation.params.error_keys) == ["ra", "dec", "parallax", "mu_ra", "mu_dec"]


def test_stream_noerr_and_nolos_variants_compose(compose):
    cfg = compose(
        [
            "simulator=stream_agama",
            "model=stream_fusion",
            "adapter=stream",
            "composition=global",
            "preprocessing=stream_global",
            "augmentation=stream_global_noerr",
        ]
    )
    steps = list(cfg.augmentation.steps)
    assert "concatenate_sigma_errors" not in steps
    assert "mask_vlos" in steps and "concatenate_vlos_mask" in steps  # LOS untouched

    cfg_real_noerr = compose(
        [
            "simulator=stream_agama", "model=stream_fusion", "adapter=stream",
            "composition=global", "preprocessing=stream_real_global",
            "augmentation=stream_real_global_noerr",
        ]
    )
    assert "concatenate_sigma_errors" not in list(cfg_real_noerr.augmentation.steps)

    cfg_real_nolos = compose(
        [
            "simulator=stream_agama", "model=stream_fusion", "adapter=stream",
            "composition=global", "preprocessing=stream_real_global",
            "augmentation=stream_real_global_nolos",
        ]
    )
    real_nolos_steps = list(cfg_real_nolos.augmentation.steps)
    assert "remove_los_velocity" in real_nolos_steps
    assert "concatenate_sigma_errors" not in real_nolos_steps
    assert "concatenate_vlos_mask" not in real_nolos_steps

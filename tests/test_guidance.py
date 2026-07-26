"""Tests for the LV simulator contract, the guided diffusion network, and the config wiring."""

from __future__ import annotations

import numpy as np
import pytest


# --------------------------------------------------------------------------------------------- #
# Simulator contract
# --------------------------------------------------------------------------------------------- #


def test_lv_simulator_contract(compose):
    from hydrabflow.simulators.registry import get_simulator

    cfg = compose(["simulator=lotka_volterra"], fill=False)
    sim = get_simulator(cfg.simulator)
    assert sim.parameter_names == ["log_alpha", "log_beta", "log_gamma", "log_delta"]
    assert sim.observable_keys == ["x"]

    data = sim.sample(8, np.random.default_rng(0))
    assert set(data) == {"log_alpha", "log_beta", "log_gamma", "log_delta", "x"}
    for name in sim.parameter_names:
        assert data[name].shape == (8, 1)
    assert data["x"].shape == (8, sim.lv_config.n_obs, 2)
    assert np.isfinite(data["x"]).all()
    assert (data["x"] > 0).all()  # abundances are strictly positive (lognormal noise)


def test_lv_simulator_is_reproducible(compose):
    """Same seed => bit-identical dataset, including the JAX-side observation noise.

    The pipeline seeds one NumPy generator per chunk (``pipeline.io.run_chunked``); the simulator
    derives its JAX key from that generator, so this property is what keeps chunked / resumed dataset
    generation identical to an uninterrupted run.
    """
    from hydrabflow.simulators.registry import get_simulator

    cfg = compose(["simulator=lotka_volterra"], fill=False)
    sim = get_simulator(cfg.simulator)
    a = sim.sample(6, np.random.default_rng(11))
    b = sim.sample(6, np.random.default_rng(11))
    for key in a:
        assert np.array_equal(a[key], b[key]), key


def test_lv_jax_log_likelihood_seam(compose):
    from hydrabflow.simulators.registry import get_simulator

    jnp = pytest.importorskip("jax.numpy")

    cfg = compose(["simulator=lotka_volterra"], fill=False)
    sim = get_simulator(cfg.simulator)
    data = sim.sample(1, np.random.default_rng(0))
    log_lik = sim.jax_log_likelihood(data["x"][0])

    theta = jnp.asarray(
        np.concatenate([data[k] for k in sim.parameter_names], axis=1), dtype=jnp.float32
    )
    assert log_lik(theta).shape == (1,)
    assert np.isfinite(np.asarray(log_lik(theta))).all()
    # The truth should beat a random prior draw on average.
    other = jnp.asarray(sim.sample_prior(1, np.random.default_rng(5))["log_alpha"] * 0 + 5.0)
    worse = jnp.concatenate([other] * 4, axis=1)
    assert float(log_lik(theta)[0]) > float(log_lik(worse)[0])

    # Wrong observation shape must fail loudly rather than broadcast silently.
    with pytest.raises(ValueError, match="shape"):
        sim.jax_log_likelihood(np.zeros((3, 2)))


def test_base_simulator_guidance_seam_is_optional(compose):
    """A simulator that does not opt in must raise a clear error, not fail obscurely."""
    from hydrabflow.simulators.registry import get_simulator

    cfg = compose(["simulator=skeleton"], fill=False)
    sim = get_simulator(cfg.simulator)
    with pytest.raises(NotImplementedError, match="jax_log_likelihood"):
        sim.jax_log_likelihood(np.zeros((10, 2)))
    assert sim.jax_theta_clip() is None


# --------------------------------------------------------------------------------------------- #
# Config wiring
# --------------------------------------------------------------------------------------------- #


def test_lv_arm_configs_compose(compose):
    from omegaconf import OmegaConf

    cfg = compose(["simulator=lotka_volterra", "model=lv_guided", "training=lv", "preprocessing=lv"])
    assert cfg.adapter.inference_variables == ["log_alpha", "log_beta", "log_gamma", "log_delta"]
    assert cfg.adapter.summary_variables == ["x"]
    assert cfg.model.inference_network.type == "guided_diffusion"
    assert cfg.model.summary_network.type == "time_series_transformer"
    # inference_variables must NOT be standardized: the guidance gradient lives in the simulator's
    # log-parameter space, and an extra learned affine map would have to be undone inside the trace.
    assert list(cfg.training.standardize) == ["summary_variables"]
    steps = [s["name"] for s in OmegaConf.to_container(cfg.preprocessing.steps, resolve=True)]
    assert steps.index("log_transform") < steps.index("standardize")


def test_unconditional_arm_drops_the_observable(compose):
    """`adapter.drop` must survive the derive-from-simulator default.

    Arm A is configured by dropping the observable; if ``fill_adapter_from_simulator`` re-derived it,
    the model would silently become conditional again.
    """
    cfg = compose(
        [
            "simulator=lotka_volterra",
            "model=lv_prior_only",
            "training=lv_prior_only",
            "adapter=lv_unconditional",
        ]
    )
    assert list(cfg.adapter.summary_variables) == []
    assert list(cfg.adapter.drop) == ["x"]
    assert cfg.adapter.inference_variables == ["log_alpha", "log_beta", "log_gamma", "log_delta"]
    assert cfg.model.summary_network.type == "none"
    assert list(cfg.training.standardize) == []


def test_eval_guidance_knobs_exist(cfg):
    for key in ("guidance", "sample_kwargs", "n_guided_obs", "n_reference_obs", "diagnose_steps"):
        assert hasattr(cfg.eval, key), key
    # The reference sampler is not yet validated, so it must be off by default.
    assert cfg.eval.n_reference_obs == 0


# --------------------------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------------------------- #


def test_log_transform_roundtrip():
    from hydrabflow.preprocessing.log_transform import LogTransform

    step = LogTransform(keys=["x"], floor=1e-6)
    data = {"x": np.array([[1.0, 10.0], [0.0, 1e-9]]), "theta": np.ones((2, 1))}
    out = step.transform(data)
    assert np.allclose(out["x"][0], np.log([1.0, 10.0]))
    assert np.isfinite(out["x"]).all()  # zeros are floored, not -inf
    assert np.array_equal(out["theta"], data["theta"])  # untouched keys pass through

    back = step.inverse_transform(out)
    assert np.allclose(back["x"][0], [1.0, 10.0])
    with pytest.raises(ValueError, match="positive"):
        LogTransform(keys=["x"], floor=0.0)


# --------------------------------------------------------------------------------------------- #
# Guided network
# --------------------------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def guided_class():
    pytest.importorskip("bayesflow")
    from hydrabflow.networks.guided_diffusion import guided_diffusion_class

    return guided_diffusion_class()


def test_builder_registers_and_sets_defaults(compose):
    pytest.importorskip("bayesflow")
    from hydrabflow.networks.factory import build_inference_network

    cfg = compose(["simulator=lotka_volterra", "model=lv_guided", "training=lv"])
    net = build_inference_network(cfg.model.inference_network)
    assert net.t_on == 0.10
    assert net.t_full == 0.05
    assert net.scaling == "none"
    assert net.skip_outside_window is True
    # Must override BayesFlow's default stochastic ADAPTIVE sampler: a gated guidance ramp makes an
    # adaptive solver collapse its step size, and the JAX stochastic path preallocates noise arrays.
    assert net.integrate_kwargs["method"] == "rk45"
    assert net.integrate_kwargs["steps"] == 200


def test_none_summary_network_builder(compose):
    from hydrabflow.networks.factory import build_summary_network

    cfg = compose(
        ["simulator=lotka_volterra", "model=lv_prior_only", "training=lv_prior_only",
         "adapter=lv_unconditional"]
    )
    assert build_summary_network(cfg.model.summary_network) is None


def test_invalid_guidance_settings_rejected(guided_class):
    with pytest.raises(ValueError, match="scaling"):
        guided_class(scaling="nope")
    with pytest.raises(ValueError, match="t_full"):
        guided_class(t_on=0.05, t_full=0.5)  # ramp must widen as t decreases


def test_guidance_ramp_gates_on_time(guided_class):
    jnp = pytest.importorskip("jax.numpy")

    net = guided_class(t_on=0.2, t_full=0.1)
    times = jnp.asarray([[1.0], [0.5], [0.2], [0.15], [0.1], [0.0]])
    ramp = np.asarray(net.guidance_ramp(times)).ravel()
    assert ramp[0] == 0.0 and ramp[1] == 0.0  # off well before t_on
    assert ramp[2] == pytest.approx(0.0, abs=1e-6)  # exactly at t_on
    assert 0.0 < ramp[3] < 1.0  # ramping
    assert ramp[4] == pytest.approx(1.0)  # full by t_full
    assert ramp[5] == pytest.approx(1.0)  # stays full
    assert np.all(np.diff(ramp) >= -1e-6)  # monotone as t decreases


def test_guidance_inert_without_target(guided_class):
    """No target attached => the hook must return the score untouched (a plain DiffusionModel).

    This is what makes guided/unguided a controlled comparison.
    """
    jnp = pytest.importorskip("jax.numpy")

    net = guided_class()
    assert net.guidance_active is False
    score = jnp.asarray([[1.0, 2.0, 3.0, 4.0]])
    out = net.guidance_function(
        x_pred=jnp.zeros_like(score), time=jnp.asarray([[0.05]]), score=score
    )
    assert np.array_equal(np.asarray(out), np.asarray(score))


def test_guidance_strength_zero_is_inert(guided_class):
    jnp = pytest.importorskip("jax.numpy")

    from hydrabflow.networks.guided_diffusion import GuidanceTarget

    net = guided_class(guidance_strength=0.0)
    net.set_guidance_target(
        GuidanceTarget(log_likelihood_batch=lambda th: jnp.sum(jnp.square(th), axis=-1))
    )
    assert net.guidance_active is False
    score = jnp.asarray([[1.0, 2.0, 3.0, 4.0]])
    out = net.guidance_function(
        x_pred=jnp.ones_like(score), time=jnp.asarray([[0.01]]), score=score
    )
    assert np.array_equal(np.asarray(out), np.asarray(score))


def test_guidance_gradient_is_clipped_and_finite(guided_class):
    """A likelihood that produces huge / non-finite gradients must not poison the score."""
    jnp = pytest.importorskip("jax.numpy")

    from hydrabflow.networks.guided_diffusion import GuidanceTarget

    net = guided_class(max_grad_norm=10.0)

    # Enormous but finite gradients -> clipped to max_grad_norm.
    net.set_guidance_target(
        GuidanceTarget(log_likelihood_batch=lambda th: 1e12 * jnp.sum(th, axis=-1))
    )
    grad = net.guidance_gradient(jnp.ones((3, 4)))
    norms = np.linalg.norm(np.asarray(grad), axis=-1)
    assert np.allclose(norms, 10.0, rtol=1e-4)

    # Non-finite gradients -> zeroed, not propagated as NaN.
    net.set_guidance_target(
        GuidanceTarget(log_likelihood_batch=lambda th: jnp.sum(jnp.log(th - th), axis=-1))
    )
    grad = net.guidance_gradient(jnp.ones((2, 4)))
    assert np.isfinite(np.asarray(grad)).all()


def test_guidance_untransform_is_chain_ruled(guided_class):
    """The gradient must be taken w.r.t. the diffusion state, through the untransform.

    With ``theta = 3*x + 1`` and ``log p = sum(theta)``, d/dx = 3 — if the untransform were applied
    outside the trace the factor 3 would be missing, and the guidance would be wrongly scaled
    whenever BayesFlow's internal standardizer is active.
    """
    jnp = pytest.importorskip("jax.numpy")

    from hydrabflow.networks.guided_diffusion import GuidanceTarget

    net = guided_class(max_grad_norm=1e9)
    net.set_guidance_target(
        GuidanceTarget(
            log_likelihood_batch=lambda th: jnp.sum(th, axis=-1),
            untransform=lambda x: 3.0 * x + 1.0,
        )
    )
    grad = np.asarray(net.guidance_gradient(jnp.zeros((2, 4))))
    assert np.allclose(grad, 3.0)


def test_get_config_roundtrip(guided_class):
    net = guided_class(
        guidance_strength=2.5, t_on=0.4, t_full=0.2, scaling="snr",
        max_grad_norm=55.0, clip_theta_std=3.0, skip_outside_window=False,
    )
    config = net.get_config()
    for key, expected in (
        ("guidance_strength", 2.5), ("t_on", 0.4), ("t_full", 0.2), ("scaling", "snr"),
        ("max_grad_norm", 55.0), ("clip_theta_std", 3.0), ("skip_outside_window", False),
    ):
        assert config[key] == expected, key
    # Guidance settings must survive a save/load cycle, since evaluate replaces the approximator
    # with the deserialized one.
    revived = guided_class.from_config(config)
    assert revived.scaling == "snr"
    assert revived.t_on == 0.4


# --------------------------------------------------------------------------------------------- #
# Guidance overrides
# --------------------------------------------------------------------------------------------- #


def test_apply_guidance_overrides(guided_class):
    from hydrabflow.pipeline.guidance import apply_guidance_overrides

    net = guided_class()
    settings = apply_guidance_overrides(net, {"guidance_strength": 3.0, "scaling": "norm_matched"})
    assert net.guidance_strength == 3.0
    assert net.scaling == "norm_matched"
    assert settings["guidance_strength"] == 3.0

    with pytest.raises(ValueError, match="unknown guidance override"):
        apply_guidance_overrides(net, {"not_a_knob": 1.0})
    # An override that breaks the ramp invariant must be caught too.
    with pytest.raises(ValueError, match="t_full"):
        apply_guidance_overrides(net, {"t_on": 0.01, "t_full": 0.5})


def test_get_guided_network_rejects_plain_diffusion(compose):
    pytest.importorskip("bayesflow")
    from hydrabflow.pipeline.guidance import get_guided_network
    from hydrabflow.pipeline.workflow import build_workflow

    cfg = compose(["simulator=lotka_volterra", "model/inference_network=diffusion"])
    workflow = build_workflow(cfg)
    with pytest.raises(TypeError, match="does not support guidance"):
        get_guided_network(workflow)

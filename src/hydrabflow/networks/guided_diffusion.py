"""Simulator-gradient guidance for BayesFlow's diffusion sampler.

Idea
----
During reverse diffusion, the network's score at time ``t`` implies a *denoised estimate* of the
parameters via the Tweedie identity. BayesFlow computes it already::

    x_hat_0 = (z_t + sigma_t**2 * score) / alpha_t          # bayesflow ... diffusion_model.py:410
    score   = (alpha_t * x_hat_0 - z_t) / sigma_t**2        # ... :495

and then hands it to a user-overridable hook (``diffusion_model.py:497``)::

    score = self.guidance_function(x_pred=x_hat_0, time=t, score=score, **guidance_kwargs)

This module overrides that hook to add the **gradient of the simulator log-likelihood** evaluated at
the denoised estimate::

    score  <-  score + w(t) * s * scale(t) * grad_{x_hat_0} log p(x_obs | x_hat_0)

which is "diffusion posterior sampling" (DPS) with the ``d x_hat_0 / d z_t`` Jacobian dropped
(DPS-lite): the gradient is taken with respect to the denoised estimate, not the noisy state, so no
backpropagation through the summary/inference network is needed. One vmapped simulator + gradient
call per integration step is the entire cost.

Gating on ``t``
---------------
Sampling integrates **t = 1 -> 0** (``diffusion_model.py:787``), where ``t=1`` is pure noise and
``t=0`` is clean data. Early on, ``x_hat_0`` is essentially prior-free noise: evaluating the
simulator there is at best uninformative and at worst numerically fatal. So guidance is ramped in
only near the end — ``w(t) = 0`` for ``t >= t_on``, rising smoothly to 1 by ``t <= t_full``.

Two notes on this that matter for interpreting results:

* ``t_on = 1.0`` recovers standard DPS (guidance everywhere) and is the natural reference point of a
  ``t_on`` sweep. The default ``t_on = 0.1`` is the "last 10% only" variant under study.
* The ramp is **smooth on purpose**. A hard switch puts a step discontinuity in the ODE vector
  field, which makes an adaptive solver collapse its step size around ``t_on``. The builder also
  pins a *fixed-step deterministic* integrator, overriding BayesFlow's default
  ``{"method": "two_step_adaptive", "steps": "adaptive"}`` (``networks/defaults.py:50``), which is
  both stochastic and adaptive and preallocates ``max_steps`` noise arrays on JAX.

Scaling
-------
``|score| ~ 1/sigma_t`` blows up as ``t -> 0``, so a fixed ``guidance_strength`` becomes
*relatively* negligible exactly where guidance is switched on. ``scaling`` controls this:

* ``none`` — add the raw gradient. This is literal DPS-lite and the theoretically correct choice
  when the diffusion model represents the prior (Arm A); keep ``guidance_strength = 1``.
* ``snr`` — multiply by ``alpha_t**2 / sigma_t**2`` so guidance grows at the same rate as the score,
  keeping its relative influence roughly constant across the gated window.
* ``norm_matched`` — rescale the gradient to ``guidance_strength * |score|`` per row. Not a
  likelihood gradient any more (only its direction survives), but the most numerically robust
  option and useful for isolating "is the *direction* right?" from "is the *magnitude* right?".

Safety
------
``x_hat_0`` can be far out of prior, so before touching the simulator the estimate is squashed into
a prior box (``lv_jax.soft_clip_theta``), the ODE state is clamped inside ``lv_jax``, non-finite
gradients are zeroed, and the per-row gradient norm is clipped to ``max_grad_norm``. Use
``scripts/diagnose_guidance.py`` to choose these numbers rather than guessing.

Inertness
---------
With no guidance target attached (or ``guidance_strength = 0``) the hook delegates to
``DiffusionModel.guidance_function``, i.e. the network behaves *exactly* like a plain
``DiffusionModel``. That is what makes the unguided/guided comparison a controlled experiment, and
it is asserted in the tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from hydrabflow.networks.factory import register_inference_network, register_summary_network

# bayesflow is imported at module scope here because the class must subclass DiffusionModel at
# definition time (and be registered as Keras-serializable before a saved model is loaded). The
# guard preserves factory.py's property that config-only contexts work without the backend.
try:  # pragma: no cover - environment dependent
    import keras

    import bayesflow as bf

    _BACKEND_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    _BACKEND_AVAILABLE = False


#: Recognised values of ``scaling``.
SCALING_MODES = ("none", "snr", "norm_matched")


@dataclass
class GuidanceTarget:
    """Runtime (non-serialized) guidance target: what to compute the likelihood gradient against.

    Attached by the sampling stage *after* the approximator is loaded, via
    :meth:`GuidedDiffusionModel.set_guidance_target`. Deliberately not part of ``get_config``: the
    observation is a property of an inference call, not of the trained network.

    Attributes
    ----------
    log_likelihood_batch : callable
        ``(theta, ) -> (batch,)`` log-likelihood of the fixed observation, vectorized over a batch
        of *physical* parameter vectors. Must be JAX-traceable.
    untransform : callable
        Maps the diffusion state to physical parameters (inverse of any standardization). Must be
        JAX-traceable; see :func:`build_theta_untransform`.
    clip : callable or None
        Optional smooth squash applied to the physical parameters before the likelihood, e.g.
        ``lv_jax.soft_clip_theta``.
    """

    log_likelihood_batch: Callable[[Any], Any]
    untransform: Callable[[Any], Any] = lambda x: x
    clip: Optional[Callable[[Any], Any]] = None


def build_theta_untransform(approximator, key: str = "inference_variables") -> Callable:
    """Return the map from diffusion-state space to physical parameter space.

    BayesFlow's internal standardizer (``training.standardize``) inserts a learned affine map in
    front of the inference variables, so ``x_pred`` inside the guidance hook is
    ``(theta - mean) / std`` rather than ``theta``. The likelihood is defined on ``theta``, so the
    inverse must be applied — *inside* the autodiff trace, so the chain rule back to ``x_pred``
    happens automatically.

    Returns the identity when ``key`` is not standardized, which is the configuration the LV study
    uses (``conf/training/lv.yaml``) precisely so this layer is absent and the gradient composition
    is auditable.
    """
    standardizer = getattr(approximator, "standardizer", None)
    if standardizer is None or key not in getattr(standardizer, "standardize", []):
        return lambda x: x

    layers = getattr(standardizer, "standardize_layers", None) or {}
    layer = layers.get(key)
    if layer is None or layer.moving_mean is None:
        raise RuntimeError(
            f"'{key}' is listed in training.standardize but its standardizer has no fitted "
            "moments, so guidance cannot map the diffusion state back to physical parameters. "
            "Either train the model first or drop it from training.standardize "
            "(see conf/training/lv.yaml)."
        )

    # Snapshot the moments as constants now: they are frozen at inference time, and closing over
    # keras Variables would drag them into every traced sampling call.
    mean = keras.ops.convert_to_numpy(layer.moving_mean[0])
    std = keras.ops.convert_to_numpy(layer.moving_std(0))

    def untransform(x):
        import jax.numpy as jnp

        return x * jnp.asarray(std, dtype=x.dtype) + jnp.asarray(mean, dtype=x.dtype)

    return untransform


def _make_guided_class():
    """Define :class:`GuidedDiffusionModel`. Separate function purely to keep the import guard tidy."""

    @keras.saving.register_keras_serializable(package="hydrabflow")
    class GuidedDiffusionModel(bf.networks.DiffusionModel):
        """``DiffusionModel`` whose reverse-time score can be nudged by a simulator gradient."""

        def __init__(
            self,
            *args,
            guidance_strength: float = 1.0,
            t_on: float = 0.1,
            t_full: float = 0.05,
            scaling: str = "none",
            max_grad_norm: float = 1.0e3,
            clip_theta_std: float = 4.0,
            skip_outside_window: bool = True,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            if scaling not in SCALING_MODES:
                raise ValueError(f"scaling must be one of {SCALING_MODES}, got {scaling!r}")
            if not 0.0 <= t_full <= t_on <= 1.0:
                raise ValueError(
                    f"require 0 <= t_full <= t_on <= 1 (guidance ramps in as t decreases), "
                    f"got t_on={t_on}, t_full={t_full}"
                )
            self.guidance_strength = float(guidance_strength)
            self.t_on = float(t_on)
            self.t_full = float(t_full)
            self.scaling = str(scaling)
            self.max_grad_norm = float(max_grad_norm)
            self.clip_theta_std = float(clip_theta_std)
            self.skip_outside_window = bool(skip_outside_window)
            self._guidance_target: GuidanceTarget | None = None

        # -- configuration round-trip ------------------------------------------------------------ #
        def get_config(self):
            # The guidance *target* (the observation) is intentionally excluded: it belongs to an
            # inference call, not to the trained weights.
            return super().get_config() | {
                "guidance_strength": self.guidance_strength,
                "t_on": self.t_on,
                "t_full": self.t_full,
                "scaling": self.scaling,
                "max_grad_norm": self.max_grad_norm,
                "clip_theta_std": self.clip_theta_std,
                "skip_outside_window": self.skip_outside_window,
            }

        # -- runtime target ---------------------------------------------------------------------- #
        def set_guidance_target(self, target: GuidanceTarget | None) -> None:
            """Attach (or clear with ``None``) the observation to guide towards.

            Must be called *after* ``load_approximator``, since loading replaces the network object
            (``pipeline/evaluate.py`` swaps ``workflow.approximator`` wholesale).
            """
            if target is not None and not isinstance(target, GuidanceTarget):
                raise TypeError(f"expected GuidanceTarget or None, got {type(target)!r}")
            self._guidance_target = target

        @property
        def guidance_active(self) -> bool:
            return self._guidance_target is not None and self.guidance_strength != 0.0

        # -- the hook ---------------------------------------------------------------------------- #
        def guidance_ramp(self, time):
            """``w(t)``: 0 for ``t >= t_on``, smoothly 1 by ``t <= t_full`` (t decreases in sampling).

            Uses the C1-continuous smoothstep ``u**2 * (3 - 2u)`` so the ODE vector field has no
            kink at the switch-on point. A degenerate window (``t_on == t_full``) falls back to a
            hard step.
            """
            import jax.numpy as jnp

            if self.t_on <= self.t_full:
                return jnp.where(time <= self.t_on, 1.0, 0.0).astype(time.dtype)
            u = jnp.clip((self.t_on - time) / (self.t_on - self.t_full), 0.0, 1.0)
            return jnp.square(u) * (3.0 - 2.0 * u)

        def guidance_gradient(self, x_pred):
            """``grad_{x_pred} sum_batch log p(x_obs | theta(x_pred))``, sanitized and norm-clipped.

            Summing over the batch is exact per row: each row's likelihood depends only on that
            row's parameters, so the sum's gradient is the per-row gradient.
            """
            import jax
            import jax.numpy as jnp

            target = self._guidance_target

            def objective(xp):
                theta = target.untransform(xp)
                if target.clip is not None:
                    theta = target.clip(theta)
                return jnp.sum(target.log_likelihood_batch(theta))

            grad = jax.grad(objective)(x_pred)

            # Zero out non-finite rows BEFORE measuring the norm, or the clip factor is NaN too.
            grad = jnp.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            norm = jnp.linalg.norm(grad, axis=-1, keepdims=True)
            factor = jnp.minimum(1.0, self.max_grad_norm / (norm + 1e-12))
            return grad * factor

        def guidance_scale(self, time, score):
            """Multiplier turning the raw likelihood gradient into a score increment."""
            import jax.numpy as jnp

            if self.scaling == "none":
                return jnp.asarray(1.0, dtype=score.dtype)
            if self.scaling == "snr":
                log_snr = self.noise_schedule.get_log_snr(t=time, training=False)
                alpha_t, sigma_t = self.noise_schedule.get_alpha_sigma(log_snr_t=log_snr)
                return jnp.square(alpha_t) / jnp.square(sigma_t)
            # norm_matched: replace the magnitude with a fixed fraction of the score's magnitude.
            return jnp.linalg.norm(score, axis=-1, keepdims=True)

        def guidance_update(self, x_pred, time, score):
            """The additive score correction ``w(t) * s * scale(t) * grad`` (no gating shortcut)."""
            import jax.numpy as jnp

            grad = self.guidance_gradient(x_pred)
            if self.scaling == "norm_matched":
                # Direction only: normalize the gradient, then take the score's magnitude.
                grad = grad / (jnp.linalg.norm(grad, axis=-1, keepdims=True) + 1e-12)
            scale = self.guidance_scale(time, score)
            return self.guidance_strength * self.guidance_ramp(time) * scale * grad

        def guidance_function(self, x_pred, time, score, **kwargs):
            """Override of BayesFlow's hook (``diffusion_model.py:265``, called at ``:497``).

            Inert (delegates to the base implementation, so the built-in constraint machinery still
            works) unless a :class:`GuidanceTarget` is attached and ``guidance_strength != 0``.
            """
            import jax
            import jax.numpy as jnp

            if not self.guidance_active:
                return super().guidance_function(x_pred=x_pred, time=time, score=score, **kwargs)

            if not self.skip_outside_window:
                return score + self.guidance_update(x_pred, time, score)

            # The integrator runs inside jax.lax loops, so `time` is a tracer and the ramp can only
            # mask the result — the simulator gradient would still be evaluated at every step, and
            # with t_on=0.1 that wastes ~90% of the work. `lax.cond` skips the whole branch instead.
            # The predicate uses max(time) so it is conservative if `time` ever varies per row
            # (it does not today: it derives from the scalar integrator time), while the per-row
            # ramp still does the actual masking inside the branch.
            def guided(operands):
                xp, t, sc = operands
                return sc + self.guidance_update(xp, t, sc)

            def unguided(operands):
                return operands[2]

            return jax.lax.cond(
                jnp.max(time) <= self.t_on, guided, unguided, (x_pred, time, score)
            )

    return GuidedDiffusionModel


_GUIDED_CLASS = _make_guided_class() if _BACKEND_AVAILABLE else None


def guided_diffusion_class():
    """The :class:`GuidedDiffusionModel` type (raises a clear error without the backend installed)."""
    if _GUIDED_CLASS is None:  # pragma: no cover - environment dependent
        raise ImportError(
            "guided_diffusion requires bayesflow + keras to be importable "
            "(the class subclasses bayesflow.networks.DiffusionModel)."
        )
    return _GUIDED_CLASS


# --------------------------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------------------------- #


@register_inference_network("guided_diffusion")
def _guided_diffusion(cfg) -> Any:
    """Build a :class:`GuidedDiffusionModel` from ``model.inference_network``.

    Guidance knobs live in the free-form ``params`` block (so no schema change is needed, and they
    are tunable by dotted path). ``params.integrate`` overrides the sampler settings; the default is
    a **fixed-step deterministic** integrator, because the gated guidance ramp is a poor fit for an
    adaptive stochastic solver (see the module docstring).
    """
    params = dict(cfg.params or {})
    integrate_kwargs = dict(params.get("integrate", {"method": "rk45", "steps": 200}))
    widths = [int(cfg.mlp_width)] * int(cfg.mlp_depth)
    return guided_diffusion_class()(
        subnet_kwargs={
            "widths": widths,
            "time_embedding_dim": int(cfg.time_embedding_dim),
        },
        integrate_kwargs=integrate_kwargs,
        guidance_strength=float(params.get("guidance_strength", 1.0)),
        t_on=float(params.get("t_on", 0.1)),
        t_full=float(params.get("t_full", 0.05)),
        scaling=str(params.get("scaling", "none")),
        max_grad_norm=float(params.get("max_grad_norm", 1.0e3)),
        clip_theta_std=float(params.get("clip_theta_std", 4.0)),
        skip_outside_window=bool(params.get("skip_outside_window", True)),
    )


@register_summary_network("none")
def _no_summary_network(cfg) -> None:
    """No summary network: an unconditional approximator (Arm A of the guidance study).

    ``bf.BasicWorkflow`` accepts ``summary_network=None``; with the observable also dropped by the
    adapter (``conf/adapter/lv_unconditional.yaml``) the diffusion model learns the prior only, and
    all data information must arrive through the guidance term.
    """
    return None

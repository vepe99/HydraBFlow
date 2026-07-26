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

Where the gradient is evaluated
-------------------------------
``guidance_point`` selects the point at which the likelihood gradient is taken: ``"tweedie"`` (the
denoised estimate ``x_hat_0``, standard DPS-lite) or ``"state"`` (the raw integration state ``z_t``,
recovered by inverting the Tweedie identity). See ``guidance_eval_point`` for why the second is worth
measuring — it is dimensionally the cleaner thing to add to a score, and it stays bounded early where
``x_hat_0`` is inflated by ``1/alpha_t``.

Particle (median) guidance
--------------------------
``x_hat_0`` is the *mean* of the denoising posterior ``p(x_0 | z_t)``, which has spread
``sigma_t/alpha_t``. For a nonlinear forward model the gradient *at the mean* is not the gradient
*over the spread* — and early in sampling the spread is 1-4 prior standard deviations, so the two
directions have little to do with each other. Measured on Lotka-Volterra, sweeping the spread from
0.1 to 4 prior sd:

* ``cos(g_point, g_averaged)`` falls from 0.997 to **0.000** — the point gradient stops representing
  the cloud entirely;
* the **mean** over particles collapses in alignment (0.367 -> **0.000** toward the truth) because the
  per-particle gradients are heavy-tailed and cancel;
* the **componentwise median** holds up (0.308 -> **0.652**).

So ``guidance_particles`` (K) gradients are evaluated across the denoising spread and reduced by
``guidance_reduce``. ``median`` is the point of the exercise; ``mean`` is available for comparison and
``point`` (or ``K=1``) reproduces the plain Tweedie behaviour bit-for-bit.

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

#: Recognised values of ``guidance_reduce`` — how per-particle gradients are combined.
REDUCE_MODES = ("point", "mean", "median")

#: Where the likelihood gradient is evaluated. See ``GuidedDiffusionModel.guidance_eval_point``.
EVAL_POINTS = ("tweedie", "state")


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
            guidance_particles: int = 1,
            guidance_reduce: str = "median",
            particle_width: float = 1.0,
            particle_data_std: float = 1.0,
            particle_seed: int = 0,
            guidance_point: str = "tweedie",
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            if scaling not in SCALING_MODES:
                raise ValueError(f"scaling must be one of {SCALING_MODES}, got {scaling!r}")
            if guidance_reduce not in REDUCE_MODES:
                raise ValueError(
                    f"guidance_reduce must be one of {REDUCE_MODES}, got {guidance_reduce!r}"
                )
            if int(guidance_particles) < 1:
                raise ValueError(f"guidance_particles must be >= 1, got {guidance_particles}")
            if float(particle_width) < 0.0:
                raise ValueError(f"particle_width must be >= 0, got {particle_width}")
            if float(particle_data_std) <= 0.0:
                raise ValueError(f"particle_data_std must be > 0, got {particle_data_std}")
            if guidance_point not in EVAL_POINTS:
                raise ValueError(
                    f"guidance_point must be one of {EVAL_POINTS}, got {guidance_point!r}"
                )
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
            self.guidance_particles = int(guidance_particles)
            self.guidance_reduce = str(guidance_reduce)
            self.particle_width = float(particle_width)
            self.particle_data_std = float(particle_data_std)
            self.particle_seed = int(particle_seed)
            self.guidance_point = str(guidance_point)
            self._guidance_target: GuidanceTarget | None = None
            self._unit_perturbation_cache: dict = {}

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
                "guidance_particles": self.guidance_particles,
                "guidance_reduce": self.guidance_reduce,
                "particle_width": self.particle_width,
                "particle_data_std": self.particle_data_std,
                "particle_seed": self.particle_seed,
                "guidance_point": self.guidance_point,
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

        def _raw_gradient(self, points):
            """``grad sum_rows log p(x_obs | theta(points))`` for a ``(N, dim)`` batch.

            Summing over rows is exact per row: each row's likelihood depends only on that row's
            parameters, so the sum's gradient is the per-row gradient.
            """
            import jax
            import jax.numpy as jnp

            target = self._guidance_target

            def objective(xp):
                theta = target.untransform(xp)
                if target.clip is not None:
                    theta = target.clip(theta)
                return jnp.sum(target.log_likelihood_batch(theta))

            return jax.grad(objective)(points)

        def _unit_perturbations(self, num_particles: int, dim: int):
            """Fixed ``(K, 1, dim)`` unit normals, cached and reused at **every** integration step.

            Deliberately not resampled per step. Fresh noise each step would make the ODE vector
            field stochastic and fight the integrator; common random numbers keep the field smooth
            and keep sampling reproducible without threading a PRNG key through ``jax.lax`` loops.
            The cost is a fixed bias — the particle cloud is not independent across steps — which is
            why ``guidance_particles`` should be >= 16 so the fixed set is representative.
            """
            import jax.numpy as jnp
            import numpy as np

            key = (num_particles, dim)
            if key not in self._unit_perturbation_cache:
                rng = np.random.default_rng(self.particle_seed)
                # Cache as NUMPY, not jnp. This method is first called from inside the traced
                # `lax.cond` guidance branch; a jnp array created there belongs to that trace, and
                # reusing it on the next sampling call raises UnexpectedTracerError. Converting a
                # numpy constant per call yields a fresh trace-local constant instead, which is free.
                self._unit_perturbation_cache[key] = rng.standard_normal(
                    (num_particles, 1, dim)
                ).astype("float32")
            return jnp.asarray(self._unit_perturbation_cache[key])

        def particle_width_at(self, time):
            """Std of the denoising posterior ``p(x_0 | z_t)``, which the particle cloud samples.

            For a Gaussian prior ``x_0 ~ N(mu, s^2)`` and ``z_t = alpha_t x_0 + sigma_t eps``, the
            exact posterior std is::

                s * sigma_t / sqrt(alpha_t^2 s^2 + sigma_t^2)

            with ``s = particle_data_std``. This equals ``sigma_t/alpha_t`` in the low-noise limit but
            **saturates at s** as the noise grows — the denoising posterior can never be wider than
            the prior it came from.

            The naive ``sigma_t/alpha_t`` is unusable: measured on this model it reaches **80** at
            ``t=1`` against a prior std of 0.5, i.e. a cloud 160 prior-sd wide. Every particle then
            lands in the parameter soft-clip's saturation region, where the gradient underflows to
            exactly zero, and guidance silently becomes a no-op at large ``t``.
            """
            import jax.numpy as jnp

            log_snr = self.noise_schedule.get_log_snr(t=time, training=False)
            alpha_t, sigma_t = self.noise_schedule.get_alpha_sigma(log_snr_t=log_snr)
            data_std = jnp.asarray(self.particle_data_std, dtype=sigma_t.dtype)
            posterior_std = (
                data_std * sigma_t / jnp.sqrt(jnp.square(alpha_t * data_std) + jnp.square(sigma_t))
            )
            return jnp.asarray(self.particle_width, dtype=sigma_t.dtype) * posterior_std

        def particle_gradients(self, x_pred, time):
            """Per-particle likelihood gradients, shape ``(K, batch, dim)``.

            All ``K * batch`` points go through a **single** vmapped gradient call: guidance is
            launch-bound rather than FLOP-bound (measured ~7 s/observation at ``euler/200``), so
            folding the particle axis into the batch makes ``K = 32`` nearly free. Never loop over
            particles.
            """
            import jax.numpy as jnp

            num_particles = self.guidance_particles
            dim = int(jnp.shape(x_pred)[-1])
            batch = int(jnp.shape(x_pred)[0])

            eps = self._unit_perturbations(num_particles, dim)  # (K, 1, dim)
            width = self.particle_width_at(time)  # (batch, 1), broadcast over particles
            cloud = x_pred[None, ...] + width[None, ...] * eps  # (K, batch, dim)

            grads = self._raw_gradient(jnp.reshape(cloud, (num_particles * batch, dim)))
            return jnp.reshape(grads, (num_particles, batch, dim))

        def guidance_eval_point(self, x_pred, time, score):
            """Where to evaluate the likelihood gradient: the Tweedie estimate, or the raw state.

            ``"tweedie"`` (default) uses ``x_hat_0``, the denoised estimate — standard DPS-lite.

            ``"state"`` uses the *current integration state* ``z_t`` itself, recovered exactly by
            inverting the Tweedie identity (``diffusion_model.py:495``)::

                z_t = alpha_t * x_hat_0 - sigma_t**2 * score

            i.e. the crude approximation ``p(x_obs | z_t) ~ p(x_obs | x_0 = z_t)``, with no denoising.
            Two reasons it is worth measuring rather than dismissing:

            * It is dimensionally the *cleaner* thing to add to a score: the score is
              ``grad_z log p(z)``, so ``grad_z log p(x_obs | z)`` lives in the same space, whereas
              DPS-lite adds a gradient taken in ``x_hat_0`` space and silently drops
              ``d x_hat_0 / d z_t``.
            * It stays bounded early on. ``alpha_t = 0.012`` at ``t=1``, so ``x_hat_0 = (z + ...)/alpha``
              is inflated ~80x and saturates the parameter soft-clip, while ``z_t`` remains O(1).

            The two coincide as ``t -> 0`` (``alpha -> 1``, ``sigma -> 0``), so any difference between
            them is an early/mid-trajectory effect.
            """
            import jax.numpy as jnp

            if self.guidance_point == "tweedie":
                return x_pred
            log_snr = self.noise_schedule.get_log_snr(t=time, training=False)
            alpha_t, sigma_t = self.noise_schedule.get_alpha_sigma(log_snr_t=log_snr)
            return alpha_t * x_pred - jnp.square(sigma_t) * score

        def guidance_gradient(self, x_pred, time):
            """The guidance direction: gradient(s) reduced over particles, sanitized, norm-clipped.

            With ``guidance_particles == 1`` (or ``guidance_reduce == "point"``) this is the plain
            Tweedie-point gradient, bit-for-bit identical to the un-particled implementation — the
            unperturbed ``x_pred`` is used, not a one-sample cloud.
            """
            import jax.numpy as jnp

            def sanitize(values):
                return jnp.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

            if self.guidance_particles <= 1 or self.guidance_reduce == "point":
                grad = sanitize(self._raw_gradient(x_pred))
            else:
                # Sanitize per particle BEFORE reducing: a single NaN particle would otherwise
                # propagate through jnp.median and poison the whole row.
                grads = sanitize(self.particle_gradients(x_pred, time))
                if self.guidance_reduce == "mean":
                    grad = jnp.mean(grads, axis=0)
                else:
                    # Componentwise median. Per-particle gradients are heavy-tailed and cancel under
                    # averaging (measured: the mean's alignment collapses to ~0 at a 4-prior-sd
                    # spread while the median holds ~0.65); the median is robust to that.
                    grad = jnp.median(grads, axis=0)

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

            grad = self.guidance_gradient(self.guidance_eval_point(x_pred, time, score), time)
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
        guidance_particles=int(params.get("guidance_particles", 1)),
        guidance_reduce=str(params.get("guidance_reduce", "median")),
        particle_width=float(params.get("particle_width", 1.0)),
        particle_data_std=float(params.get("particle_data_std", 1.0)),
        particle_seed=int(params.get("particle_seed", 0)),
        guidance_point=str(params.get("guidance_point", "tweedie")),
    )


@register_summary_network("none")
def _no_summary_network(cfg) -> None:
    """No summary network: an unconditional approximator (Arm A of the guidance study).

    ``bf.BasicWorkflow`` accepts ``summary_network=None``; with the observable also dropped by the
    adapter (``conf/adapter/lv_unconditional.yaml``) the diffusion model learns the prior only, and
    all data information must arrive through the guidance term.
    """
    return None

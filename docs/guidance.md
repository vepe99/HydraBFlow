# Simulator-gradient guidance in diffusion sampling

A study of whether a **differentiable simulator** can improve amortized posterior sampling by
injecting `∇ log p(x_obs | θ)` into the reverse diffusion process of BayesFlow's `DiffusionModel`.

**Headline result: on the Lotka-Volterra testbed, it does not work as hoped — and the measurements
say exactly why.** Gating guidance to the end of the trajectory (the original hypothesis) provably
cannot move the samples; applying it throughout does move them, but toward an over-dispersed
distribution rather than the posterior. Details and numbers below.

---

## 1. Where the guidance term goes

BayesFlow 2.0.12 already has the seam — no fork is needed. In
`bayesflow/networks/inference/diffusion/diffusion_model.py`:

| line | what |
|---|---|
| 410 | Tweedie: `x_pred = (z + σ_t²·score) / α_t` |
| 495 | `score = (α_t·x_pred − z) / σ_t²` |
| **497** | `score = self.guidance_function(x_pred=x_pred, time=time, score=score, **guidance_kwargs)` |
| 265 | `guidance_function`, documented as user-overridable, with a JAX `jax.grad` branch |
| 787 | sampling integrates **t = 1 → 0** (t=1 noise, t=0 data) |

The hook receives exactly the two things needed: the **denoised estimate** `x̂₀` and the
**integration time** `t`. `hydrabflow.networks.guided_diffusion.GuidedDiffusionModel` overrides it:

```
score  ←  score + w(t) · s · scale(t) · ∇_{x̂₀} log p(x_obs | x̂₀)
```

This is DPS with the `∂x̂₀/∂z_t` Jacobian dropped ("DPS-lite"): the gradient is w.r.t. the denoised
estimate, not the noisy state, so **no backpropagation through the summary/inference network** is
needed — one vmapped simulator+gradient call per integration step.

`w(t)` is a smoothstep ramp: 0 for `t ≥ t_on`, rising to 1 by `t ≤ t_full`. `t_on = 0.1` is the
"last 10% only" variant; `t_on = 1.0` recovers standard DPS.

## 2. The two arms

Guidance is a **sampling-time** modification, so one trained network serves an entire sweep.

| arm | config | what it tests |
|---|---|---|
| **B** conditional + guidance | `model=lv_guided training=lv` | Refinement of an already-good posterior. The network already encodes `p(θ\|x_obs)`, so adding `∇log p(x_obs\|θ)` targets `p(θ\|x_obs)·p(x_obs\|θ)^w` — **over-concentrated by construction**. |
| **A** prior-only + guidance | `model=lv_prior_only training=lv_prior_only adapter=lv_unconditional` | The real test. The network encodes only the prior, so *all* data information must arrive through guidance. Calibration is meaningful here. |

## 3. Measured results

Setup: LV with 4 log-rates, 10 observation times × 2 species, 10% lognormal noise, 20k training
simulations, 60 epochs. Sampler `euler/200`. See §6 for commands.

### 3a. Guidance vs the model's own prediction (`diagnose_guidance`, Arm B)

Per-step trace in the gated window (`t_on=0.1`, `scaling=none`, `max_grad_norm=1e3`):

| quantity | value | reading |
|---|---|---|
| `‖update‖ / ‖score‖` | 0.24 (median) | guidance is a quarter of the score — *sounds* significant |
| `cos(update, score)` | +0.04 | nearly **orthogonal** to the model; not fighting it, just unaligned |
| **`dx0_rel`** | **6.5e-5** | **the shift in denoised-parameter space is 0.006%** |
| `clipped_frac` | 1.00 | the gradient was clipped at *every* step — true ‖∇‖ ≈ 1.7e4 |
| endpoint shift | ~4e-4 | vs a posterior std of ~0.05, i.e. under 1% |

**This is the central mechanism.** The score and the denoised estimate are related by
`Δx̂₀ = σ_t²·Δscore / α_t`. In the last 10% of the trajectory `σ_t² → 0`, so a large *score*
correction is a negligible *parameter* correction. **Late-window guidance has no lever arm.** It is
not a tuning problem: at `t_on=0.1` the endpoint shift was 0.0002–0.0004 across
`guidance_strength ∈ {0.05, 0.2, 0.5}` — a 10× strength change did essentially nothing.

`scaling=snr` (×α²/σ²) is the principled fix, since it cancels the σ² shrinkage exactly.

### 3b. Does guidance carry real information? (Arm A, 8 observations, 200 draws)

Baseline is the unguided prior-only model, whose posterior std correctly reproduces the prior (0.50).

| setting | rmse | post_std | calib |
|---|---|---|---|
| **unguided (= the prior)** | **0.4527** | 0.500 | 0.113 |
| `norm_matched` s=0.2, t_on=1.0 | **0.3665** | 0.504 | 0.118 |
| `norm_matched` s=0.5, t_on=1.0 | 0.5899 | 0.606 | 0.142 |
| `norm_matched` s=1.0, t_on=1.0 | 1.6190 | 1.251 | 0.154 |
| `none` clip=1, t_on=1.0 | 0.5309 | 0.504 | 0.169 |
| `none` clip=10, t_on=1.0 | 5.2990 | 0.942 | 0.318 |
| `none` clip=100, t_on=1.0 | 49.68 | 6.94 | 0.382 |
| `none` clip=1000, t_on=1.0 | 410.3 | 64.9 | 0.341 |
| `norm_matched` s=0.5, **t_on=0.1** | 0.4526 | 0.500 | 0.113 |
| `norm_matched` s=2.0, **t_on=0.1** | 0.4522 | 0.500 | 0.113 |

Two conclusions:

1. **Guidance does carry real likelihood information**: RMSE 0.4527 → 0.3665 (−19%) at
   `norm_matched s=0.2`, with no variance inflation. The gradient direction is informative.
2. **But this is not posterior sampling.** The posterior std stays at the prior's 0.50, while the
   true LV posterior std is **0.007–0.1** (measured by MCMC, §5) — 5–70× tighter. Guidance nudges
   the *location* slightly and never *concentrates*. And `t_on=0.1` changes nothing at all
   (0.4526 vs 0.4527), confirming §3a at the level of the final posterior.

### 3c. The blow-up is not a solver artifact

The obvious hypothesis for the `none`-scaling divergence is an under-resolved explicit ODE. It is
not — refining the integrator changes nothing:

| setting | steps | rmse | post_std |
|---|---|---|---|
| `none` clip=10 | 200 | 5.37 | 1.11 |
| `none` clip=10 | 1000 | 5.20 | 1.05 |
| `none` clip=100 | 1000 | 44.84 | 7.49 |
| `none` clip=100 | 2000 | 43.76 | 7.27 |

5× more steps moves RMSE by 3%. The guided ODE **converges** to a wrong, over-dispersed
distribution. The likely culprit is the dropped `∂x̂₀/∂z_t` Jacobian: for a likelihood this sharp
(σ=0.1 over 20 data points, raw ‖∇‖ ≈ 1.7e4 against an O(1) prior score), the DPS-lite
approximation is not accurate enough. Full DPS is the next thing to try (§7).

## 4. Pros, cons, pitfalls

**Pros.** Sampling-time only — no retraining, one network per sweep. Uses information the network
never saw (exact likelihood and noise model). A principled route to correcting a *misspecified* or
under-trained summary network. Late gating means the simulator is only evaluated where `x̂₀` is
already prior-plausible.

**Pitfalls, ranked by how much they actually bit:**

1. **No lever arm at small t** (§3a) — the one that kills the original hypothesis. `Δx̂₀ ∝ σ_t²`.
2. **Scale mismatch.** Raw ‖∇log p‖ ≈ 1.7e4 vs a score of O(1)–O(4e3). Unclipped it dominates the
   vector field and the samples diverge; clipped small it has no effect. The usable window is narrow
   and problem-specific — this is what `max_grad_norm` and `norm_matched` exist for.
3. **Likelihood double-counting (Arm B).** Guidance throughout at `norm_matched s=0.5, t_on=1.0`
   moved the endpoint by ~80% of a posterior std but made the error *worse* (0.0526 → 0.0674).
   Always report calibration next to RMSE.
4. **DPS-lite approximation error** (§3c) — apparently the binding constraint here.
5. **Traced control flow.** The integrator uses `jax.lax` loops, so `t` is a tracer: no Python
   branching. A `where`-mask alone still *evaluates* the simulator every step; `lax.cond`
   (`skip_outside_window`) actually skips it, worth ~10× at `t_on=0.1`.
6. **Adaptive/stochastic default sampler.** BayesFlow defaults to
   `{method: two_step_adaptive, steps: adaptive}` — stochastic *and* adaptive, and on JAX it
   preallocates `max_steps` noise arrays. A gated ramp is a poor fit; the builder pins a fixed-step
   deterministic solver.
7. **Space bookkeeping.** `x_pred` is in *standardized, concatenated* inference-variable space; the
   likelihood is defined on physical parameters and a *raw* observation. `conf/training/lv.yaml`
   drops `inference_variables` from `training.standardize` so the two coincide, and the
   un-standardization happens *inside* the `jax.grad` trace so the chain rule is automatic
   (`test_guidance_untransform_is_chain_ruled` pins this).
8. **Cost.** Guided sampling is ~7 s/observation vs 0.7 s unguided with `euler/200` — 1200 RK45
   stage evaluations × a 400-step LV solve is launch-bound, so `rk45/200` costs 21 s. Not
   recompilation: repeating the same observation costs the same.
9. **float32 is fine.** RK4 matches `scipy` to 4e-5 relative — 3 orders below the 0.1 observation
   noise — so `jax_enable_x64` (process-wide, would perturb Keras) is deliberately not set.

## 5. Reference posterior — status

`pipeline.reference` implements MALA whitened by a Laplace approximation at the MAP. Whitening is
mandatory: unwhitened, the step size collapses to ~1e-3 and mixing dies (`r_hat` = 74, ESS 35/1500).
Whitened it mixes well (`r_hat` = 1.0006, ESS ≈ 6900/16000).

**It is not yet trustworthy and is off by default** (`eval.n_reference_obs: 0`). The MAP search is
the weak link: a single L-BFGS start was measured stopping at `log p = −764` on an observation whose
truth scores −33, producing a confidently wrong reference (`r_hat` = 1.0006 at 14σ from truth).
Multi-start plus prior-dispersed chain initialization fixes the diagnosis but is slow.

**Intended replacement: NUTS via numpyro** (or blackjax) — mass-matrix adaptation handles the
anisotropy without needing a MAP, and it is battle-tested. Neither is currently a dependency. The
posterior std figures quoted in §3b come from the whitened MALA runs that *did* converge and should
be treated as indicative.

## 6. Commands

```bash
# datasets
scripts/simulate.py simulator=lotka_volterra model=lv_guided training=lv preprocessing=lv \
    data.n_simulations=20000
scripts/simulate.py simulator=lotka_volterra model=lv_guided training=lv preprocessing=lv \
    data.n_simulations=20000 data.dataset_name=test_data_2000.npz seed=999

# Arm B: conditional + guidance
scripts/train.py simulator=lotka_volterra model=lv_guided training=lv preprocessing=lv \
    data.n_simulations=20000 training.n_epochs=60

# Arm A: prior-only + guidance
scripts/train.py simulator=lotka_volterra model=lv_prior_only training=lv_prior_only \
    preprocessing=lv adapter=lv_unconditional data.n_simulations=20000 training.n_epochs=60

# Is guidance even large enough to matter? (cheap; run this FIRST)
scripts/diagnose_guidance.py ... model_dir=outputs/.../<ts> \
    +eval.guidance.t_on=1.0 +eval.guidance.scaling=norm_matched

# Paired guided vs unguided
scripts/evaluate_guided.py ... model_dir=outputs/.../<ts> \
    eval.n_guided_obs=24 eval.num_samples=300 \
    +eval.sample_kwargs.method=euler +eval.sample_kwargs.steps=200 \
    +eval.guidance.scaling=norm_matched +eval.guidance.guidance_strength=0.2 \
    +eval.guidance.t_on=1.0 +eval.guidance.t_full=1.0
```

Always run `diagnose_guidance` before a full evaluation: it reports `dx0_rel` and
`‖update‖/‖score‖` in seconds and will tell you immediately if a setting cannot possibly matter. It
warns explicitly when guidance is below 1% of the score throughout the window.

## 7. Next steps

1. **Full DPS** — differentiate through the network (`∇_{z_t} log p(x_obs | x̂₀(z_t))`), wrapping
   `jax.grad` around `self.score`. §3c suggests the dropped Jacobian is the binding constraint. Cost:
   several × slower per step.
2. **NUTS reference** (§5) so per-observation claims become quantitative (C2ST/MMD vs truth).
3. **`scaling=snr` sweep at moderate `t_on`** (0.3–0.5) — the principled cancellation of the σ²
   lever-arm loss, in the window where `x̂₀` is meaningful but σ is not yet tiny.
4. **A weakened Arm B** — deliberately under-train the summary network, so guidance has real headroom
   and double-counting is less dominant. This is the setting where guidance is most likely to help.
5. **Annealed/corrector steps** — BayesFlow's `corrector_steps` + `langevin` sampler may absorb a
   sharp likelihood better than an explicit ODE.

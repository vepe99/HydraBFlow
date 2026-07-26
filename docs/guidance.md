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

---

# Round 2 — why the gradient is so strong, and median-particle guidance

Round 1 left three open questions: *why* is the gradient so large, is the likelihood landscape jagged,
and can a less noisy direction be extracted. Each was measured. **Two hypotheses were refuted and one
was confirmed** — the refutations are as useful as the confirmation, because they close off
plausible-looking directions.

## R2.1 Why the gradient is so strong

Analytically, `∇_θ log p = Σ_{t,s}(z_ts/σ)·∂log f_ts/∂θ`, so
`‖∇‖ ~ (√(2T)/σ)·‖∂log f/∂θ‖·|z̄|`. Two factors dominate: `1/σ = 10`, and a Jacobian that accumulates
**coherently** across observation times (a rate change shifts the oscillation phase, and that error
compounds along the trajectory). Measured, the norm grows roughly **linearly** in the number of
observation times, while its alignment with the direction to the truth is flat:

| n_obs | 10 | 25 | 50 | 100 |
|---|---|---|---|---|
| median ‖∇‖ | 13 360 | 35 611 | 71 708 | 146 016 |
| cos(∇, →truth) | 0.545 | 0.597 | 0.585 | 0.582 |

**More observation times makes the gradient stronger, not tamer, and no more indicative.** Adding data
is therefore not a fix for guidance (it is still worth doing for posterior estimation — see R2.5).

## R2.2 The landscape is not jagged — it is anisotropic (hypothesis refuted)

The natural guess was that sparse sampling of an oscillatory ODE aliases the signal and produces a
multimodal, jagged likelihood. **It does not.** The prey oscillation period is 10.55 time units and
`n_obs=10` samples it 5.3×/period — comfortably above Nyquist. Measured over a 1-D slice spanning ±2
prior sd, at every `n_obs ∈ {10, 25, 50, 100}`:

* **exactly 1 local maximum** — the slice is unimodal;
* **multi-start MAP succeeds in 100% of trials** (6 restarts).

So the round-1 `log p = −764` MAP failure was a *single-start* artifact, already fixed by the
multi-start `laplace_preconditioner`. What is extreme is not ruggedness but **anisotropy: the Fisher
condition number is ≈1000**. The likelihood is a smooth, narrow, strongly-correlated ridge.

## R2.3 Fisher preconditioning fixes magnitude, not direction (hypothesis refuted)

Given the anisotropy, the natural fix is a natural-gradient/Gauss-Newton step
`(JᵀJ/σ² + prior_prec)⁻¹∇`. It does fix the scale — `‖H⁻¹∇‖ ≈ 1.5` versus `‖∇‖ ≈ 1e4`, a principled
O(1) step in natural units — but **not the direction**: `cos(H⁻¹∇, →truth) = 0.27` versus `0.33` for
the raw gradient. A local quadratic model is simply not valid 1σ out on a nonlinear forward map. Left
unimplemented; `max_grad_norm` / `norm_matched` already handle scale empirically.

## R2.4 Median-particle guidance (confirmed)

`x̂₀` is the *mean* of the denoising posterior `p(x₀|z_t)`. For a nonlinear forward model the gradient
*at the mean* is not the gradient *over the spread*. Sweeping an isotropic spread offline:

| spread (prior sd) | cos(g_point, g_avg) | cos(g_point,→truth) | cos(g_**mean**,→truth) | cos(g_**median**,→truth) |
|---|---|---|---|---|
| 0.1 | 0.997 | 0.291 | 0.367 | 0.308 |
| 0.3 | 0.984 | 0.514 | **0.676** | 0.589 |
| 1.0 | 0.675 | 0.374 | 0.543 | 0.593 |
| 2.0 | 0.098 | 0.547 | 0.282 | 0.525 |
| 4.0 | 0.000 | 0.553 | 0.000 | **0.652** |

Per-particle gradients are heavy-tailed: under averaging they **cancel** (the mean's alignment
collapses to 0.000 at a 4-sd spread) while the **componentwise median holds** (0.652). Hence
`guidance_particles` / `guidance_reduce` in `networks/guided_diffusion.py`, with `K` folded into the
batch axis (guidance is launch-bound, so `K=32` is cheap) and a **fixed** unit-normal set reused at
every step — common random numbers, so the ODE vector field stays smooth and sampling stays
reproducible without threading a PRNG key through `jax.lax` loops.

### The particle width must be the denoising-posterior std, not σ_t/α_t

The first implementation used `σ_t/α_t` as the cloud width. That is **wrong at large t** and was caught
only by the in-situ diagnostic: it reaches **80** at `t=1` against a prior std of 0.5, i.e. a cloud 160
prior-sd wide. Every particle then lands in the parameter soft-clip's saturation region, where the
derivative underflows to exactly zero in float32 — guidance silently became a **no-op** (`|grad|` = 0.0
for `t ≥ 0.8`).

The correct width is the exact Gaussian denoising-posterior std,
`s·σ_t/√(α_t²s² + σ_t²)` with `s = particle_data_std`, which equals `σ_t/α_t` at low noise but
**saturates at `s`** — a denoising posterior can never be wider than the prior it came from.

A consequence worth stating: with the correct width the cloud never exceeds ~1 prior sd, so the 2–4 sd
regime where the median dominated in the offline table is **physically unreachable**. The offline
sweep's most dramatic rows describe a spread the algorithm cannot produce.

### In-situ confirmation

Measured **counterfactually** — guidance evaluated along the *unguided* trajectory. This distinction
matters: with guidance applied, an unstable run leaves the prior within a few steps, the soft-clip
saturates, and every later gradient reads as zero, so the diagnostics end up describing a diverged
trajectory instead of the guidance. `diagnose_guidance` now records both (`cf_*` columns are the
counterfactual ones).

| trajectory stage | cos(point,→truth) | cos(**median**,→truth) | cos(pt,med) | width |
|---|---|---|---|---|
| early `t>0.5` | 0.614 | **0.651** | 0.955 | 0.500 |
| mid `0.2<t<0.5` | 0.646 | **0.731** | 0.923 | 0.237 |
| late `t<0.2` | 0.619 | 0.619 | 1.000 | 0.003 |

The median beats the point gradient at every stage, peaking at **0.913** alignment around `t=0.6` — the
direction is genuinely informative. The gain is real but modest (+0.04 to +0.09), smaller than the
offline sweep suggested, precisely because of the width cap above. Late in the trajectory the two
coincide, as they must (width → 0). `|grad|` stays ≈2.5e4 throughout, so **magnitude control remains
the open problem** — a better direction does not by itself make guidance usable.

### …and the better direction makes the posterior WORSE

Arm A, 8 observations × 200 draws, all at `t_on=1.0` (guidance throughout), `particle_data_std=0.5`:

| setting | rmse | post_std | calib |
|---|---|---|---|
| unguided (= the prior) | 0.4527 | 0.5001 | 0.1130 |
| **point** (K=1), norm_matched s=0.2 | **0.3665** | 0.5044 | **0.1175** |
| median K=8, norm_matched s=0.2 | 0.3737 | 0.5067 | 0.1259 |
| median K=32, norm_matched s=0.2 | 0.3837 | 0.4970 | 0.1387 |
| median K=64, norm_matched s=0.2 | 0.3987 | 0.5041 | 0.1552 |
| mean K=32, norm_matched s=0.2 | 0.3887 | 0.5011 | 0.1419 |
| point, none, clip=10 | 5.299 | 0.9415 | 0.3184 |
| median K=32, none, clip=10 | 6.108 | 0.9085 | 0.4092 |

**Both RMSE and calibration degrade monotonically in K** (0.3665 → 0.3737 → 0.3837 → 0.3987;
0.1175 → 0.1259 → 0.1387 → 0.1552). More particles is *worse*, and the plain point gradient wins.
No setting brings `post_std` below the prior's 0.500.

Monotonic-in-K rules out Monte-Carlo noise — more particles would reduce that. It is a **systematic
bias**: the particle median is a *smoothed* score, smoothed on the scale of the denoising posterior
(~0.5, i.e. the whole prior). Smoothing the score changes the stationary distribution of the reverse
ODE, and it destroys exactly the fine-scale structure that would concentrate the posterior.

**The most useful lesson is methodological: the gate metric was wrong.** `cos(g, θ_true − x̂₀)` asks
"does this point at the answer?", which is what a *point estimator* wants. A posterior sampler needs an
unbiased *score*, and those are different objectives — the median improved the first while damaging the
second. An in-situ direction diagnostic cannot substitute for measuring the posterior.

## R2.5 Observation density — the one thing that clearly helped

Arm B (conditional posterior network), **both models trained for 200 epochs** so the comparison is
matched. Numbers on the **full 2000-row test set** (`evaluate` stage, 500 draws each):

| n_obs | spacing | rmse | calibration error |
|---|---|---|---|
| 10 | 2.0 | 0.1052 | **0.0324** |
| **50** | 0.4 | **0.0731** | 0.0392 |

**5× more observation times: RMSE −31%.** Calibration is marginally worse but both values are small.

⚠ **Sample-size caveat — and a correction.** These were first measured on the 48-observation subset the
(expensive) guided runs use, which gave rmse 0.0596 → 0.0321, i.e. "−46%" and absolute errors ~1.8×
*better* than the truth. The full test set gives −31%. The direction of the conclusion held, the
magnitude did not. Absolute RMSEs quoted anywhere in this document from 8–48-observation runs should be
read with the same suspicion; the *paired* guided-vs-unguided comparisons are more trustworthy, since
both arms share the same observations and seeds, but their absolute levels are not.

Coverage on the dense model (`coverage.png`) sits **above** the diagonal for all four parameters —
credible intervals are conservative (over-wide), most visibly for `log_alpha` and `log_gamma`. Recovery
is excellent (`r = 0.994–0.998`). Both models still logged "loss still descending" at 200 epochs, so the
over-wide intervals are most likely residual under-training rather than a modelling defect.

Matching the epoch budget mattered. At 60 epochs the dense model looked *worse*-calibrated
(0.0516 → 0.0794) purely because it had not converged — the denser observable is a harder fit. The
naive 60-epoch comparison would have supported the opposite conclusion.

Note the contrast with R2.1, on the same change: denser observations **help the amortized posterior a
lot** and **do nothing for the guidance gradient** (norm grows ~linearly, alignment flat). The two
questions decouple, which is worth remembering before reaching for guidance at all — for this problem,
spending the effort on data and training beat every guidance variant tried.

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

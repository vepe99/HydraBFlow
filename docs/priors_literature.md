# Priors for stream-based inference of the Milky Way halo shape

*Literature review and recommendation, 2026-09-12. Diffed against the priors in
`conf/simulator/stream_agama_rnbody_ibata_m200c_v2.yaml` (and the `_wide` dataset that preceded it).
arXiv ids are given so every number can be re-checked; items the review could not verify from the
source text are marked UNVERIFIED.*

## 1. What the prior has to do in an SBI setting

In simulation-based inference the prior is not a regularizer added at the end: it *is* the training
set. Three consequences drive every recommendation below.

1. **Coverage beats sharpness.** A truth outside the prior support cannot be recovered, and a truth
   near the edge is pulled inward (the 2026 hierarchical stream papers, arXiv:2601.15373, measure
   exactly this bias toward q ≈ 1 with a U[0.5, 1.5] prior). The prior must contain *every*
   published posterior for the parameters we infer, not the one we believe.
2. **Nuisances get informative Gaussians, inferred parameters get broad supports.** Anything
   marginalized (solar frame, bulge amplitude, gas normalization, progenitor locals) should carry the
   best measurement as a Gaussian; anything we report (M200, c, γ, q, p, tilt, disc mass) should be
   uniform or log-uniform over the physically allowed range, so the posterior shape is data-driven.
3. **Degeneracies must be broken by observables, not by the prior.** The disc–halo degeneracy is the
   dominant systematic in every stream paper (Koposov+10, Bovy+16, Palau & Miralda-Escudé 23,
   Woudenberg & Helmi 24). A disc prior that is too narrow silently sets q; one that is too wide lets
   γ or Σ absorb the terminal-velocity tension (our own γ-railing symptom). The right fix is to feed
   K_z(1.1 kpc), the terminal velocities and the rotation curve as *data* (already done through
   `sigma_z`, `vterm_kms`, `vcirc_kms`) and keep the disc prior physically bounded.

## 2. What the literature does

### 2.1 Stream-based halo-shape papers use flat priors, in the density

No Milky Way stream paper uses a simulation-derived prior on the halo shape; all use flat (or grid)
priors, and simulations enter only as a post-hoc comparison. Density flattening is the modern
convention (Bovy+16, Malhan & Ibata 19, Vasiliev+21 with the explicit rationale that a density
flattening guarantees positivity, Koposov+23, Palau & M-E 23, Ibata+24, Woudenberg & Helmi 24,
Nibauer & Bonaca 25); potential flattening survives in logarithmic-halo work (Koposov+10, Bowden+15,
Erkal+19). Conversion near unity: (1 − q_ρ) ≈ 3 (1 − q_Φ), so a ±0.05 prior in q_Φ is ±0.15 in q_ρ.

| Paper (arXiv) | Free halo shape | Prior | Posterior |
|---|---|---|---|
| Koposov, Rix & Hogg 2010 (0907.1085) | q_Φ | flat grid, V_c N(229,18), R_0 N(8.4,0.4) | q_Φ = 0.87 (+0.07/−0.04); halo q_Φ > 0.89 |
| Küpper+2015 (1502.02658) | q_z | U[0.2, 1.8] | 0.95 (+0.16/−0.12) |
| Bowden+2015 (1502.00484) | q_Φ | U[0.5, 1.5], v_0 U[130, 290] | 0.90–0.91 |
| Bovy+2016 (1609.01298) | c/a (density) | U[1/2, 2], V_c U[200, 250] | c/a = 1.05 ± 0.14 |
| Malhan & Ibata 2019 (1807.05994) | q_ρ | grid [0.5, 1.5] | 0.82 (+0.25/−0.13), V_c = 244 ± 4 |
| Erkal+2019 (1812.08192) | q_Φ + direction | U(0.7,1) or U(1,1.3); l, b isotropic; M U(6,25)e11; r_s U(10,30) | 0.87 ± 0.04 or 1.20 (+0.04/−0.03) |
| Vasiliev+2021 (2009.10726) | q_in, q_out, p_out, β_out (density, r-dependent) | not printed (UNVERIFIED) | inner q ≈ 0.5–0.6 aligned with disc; outer 1.5–2 |
| Koposov+2023 (2211.04495) | q + direction, γ, β | q U(0,1) or U(1,2); direction N(0,1)³; γ U(0,2); β U(2,4); M log-U(1,30)e11 | q = 0.56 (+0.10/−0.08) or 1.40 (+0.12/−0.10) |
| Palau & M-E 2023 (2212.03587) — our three streams | q_ρ, α, β, a_1, ρ_0 | q U[0,6], α U[−3,3], β U[0,6], a_1 U[0,100], ρ_0 U[0,1.5e8] | q = 1.06 ± 0.06; α = 0.06 ± 0.22 (Pal5 0.73, NGC3201 0.68, M68 −0.23); heavy disc 6.1e10 |
| Ibata+2024 (2311.17202) | q_ρ, γ, r_0, (β) | positivity + M200 ∈ [0, 2e12] | q = 0.75 ± 0.03; γ = 0.97 (+0.17/−0.21); r_0 = 14.7 kpc; disc 4.2e10 |
| Woudenberg & Helmi 2024 (2407.21790) | p, q (density) | p U[1.0, 1.1], q U[0.7, 1.4], r_h U[10,30], M_discs U[3.5,5.5]e10 | p = 1.013 ± 0.006, q = 1.204 (+0.032/−0.036) |
| Nibauer & Bonaca 2025 (2504.07187) | c/a, b/a, pitch, yaw | only r_s ∈ [10,30] stated (rest UNVERIFIED) | axisym q = 0.81; triaxial 1 : 0.75 : 0.70, tilt 18°/23° |

The posterior spread across constant-q analyses, 0.56 to 1.40, is far wider than any single paper's
error bar and correlates with the baryonic model (Ibata: light disc → q = 0.75; Palau: heavy disc →
q = 1.06; W&H: disc mass bounded → q = 1.2). This is the empirical statement that the disc prior is
load-bearing.

### 2.2 Halo mass and concentration

* **M200 (200 ρ_crit), total.** The Gaia-era cluster is 1.0–1.2 × 10^12 Msun with ~20 % scatter
  (Callingham+19 1.17, Deason+21 1.01–1.16, Shen+22 1.08, Cautun+20 1.08, Palau+23 1.17, McMillan 17
  1.30, Posti & Helmi 19 1.3; low outliers Eadie & Jurić 19 0.70 and Bird+22 0.55 from tracer-profile
  systematics; Watkins+19 1.54 high). Wang+20's full compilation spans 0.5–2.0. Rotation-curve-only
  fits restricted to r < 30 kpc (Ou+24, Jiao+23: 0.2–0.7) are biased low by ~2× (Dado+26,
  2603.13516) and should not enter a prior.
* **Dark-halo versus total.** Only Cautun+20 separates them: M200_DM = 0.97 vs M200_total = 1.08
  (baryons inside R200: stars 5.0e10 + cold gas 1.2e10 + CGM 6.4e10). McMillan's M_v and Callingham's
  M200 are totals. Our `log10_M200_TwoPowerTriaxial_halo` is the **halo component alone** (unit-norm
  `enclosedMass(r200)` of the halo Spheroid; verified on six dataset rows, baryons add 4–17 % inside
  r200). So the prior must not be centred on total-mass numbers, and posteriors compared with the
  literature need the row's baryons added.
* **Concentration.** McMillan's prior ln c_v' ~ N(2.56, 0.272) is Boylan-Kolchin+10's log-normal at
  Δ ≈ 94 ρ_crit (verified); it corresponds to c200 ≈ 10. Planck DMO relations give c200 ≈ 8.0–8.5
  at 10^12 with 0.11–0.16 dex scatter (Dutton & Macciò 14, Ludlow+16, Diemer & Joyce 19), but an
  uncontracted profile forced to mimic the baryon-contracted Milky Way needs c200 ≈ 12–15 (Cautun's
  NFW alternative 13.3; McMillan 15.4). Our profile has no explicit contraction, so the DMO value
  would be wrong by ~0.15 dex; N(2.56, 0.272) already puts 15.4 at 0.6σ and is the right compromise.
* **Inner slope.** Contracted cusps behave like γ ≈ 1.3–1.7 inside 10 kpc (Cautun ×2 enclosed DM at
  8 kpc; Lazar+20 slope −1.5 at 1–2 % R_vir). Empirically γ runs from 0.06 (Palau, heavy disc)
  through 0.97 (Ibata) to 1.60 (Nitschai+21, light disc, fixed r_s). Negative γ (a central density
  hole) has no physical support in any of these.
* **Reparameterization.** Defining the concentration on r_{−2} (McMillan's c_v; our
  r_h = r200 / [c (2 − γ)]) is what the Einasto/gNFW degeneracy literature recommends: at fixed
  (M200, c_{−2}) γ changes the inner shape without moving the peak of the rotation curve. Its price is
  r_h → ∞ as γ → 2, so a cap at γ ≤ 1.7 is both numerical and physical.

### 2.3 Halo shape from simulations with baryons (5–30 kpc, MW mass)

| Study | c/a (density) | b/a | Verdict |
|---|---|---|---|
| Chua+2019 Illustris (1809.07255) | 0.70 ± 0.11 | 0.88 ± 0.10 | rounder and more oblate than DMO; ratios ~constant inside 0.15 R200 |
| Prada+2019 Auriga (1910.04045) | 0.65 [0.41–0.79] | 0.94 [0.54–0.99] | median T < 1/3 (oblate) at all radii; disc–minor-axis angle 19° ± 20° |
| Emami+2021 TNG50 (2009.09220) | 0.75 | 0.94 | T ≈ 0.24 (16–84 %: −0.11 to 0.31); 32 % aligned, 32 % twisted, 36 % stretched |
| Abadi+2010 (0902.2477) | isopotential 0.85 | — | "essentially oblate", aligned with disc |
| Kazantzidis+2004/2010 | +0.2–0.4 in axis ratios from cooling | | discs contributing < 50 % of V_c leave triaxiality intact |

Simulation prior would be c/a ≈ N(0.72, 0.12), b/a ≈ N(0.92, 0.06), tilt half-normal σ ≈ 20°. It
excludes half the observational literature (Bovy 1.05, Palau 1.06, W&H 1.20, Posti 1.30), so it is a
*check*, not a training prior. Radial variation of q sets in at r ≳ 30 kpc (Chua, Prada, Shao+21);
Vasiliev+21's inner-to-outer transition is Sgr/LMC-driven and "tentative". Keep q constant.

### 2.4 Baryons

* Thin disc: R_d = 2.6 ± 0.5 kpc photometric (Bland-Hawthorn & Gerhard 16), 2.15 ± 0.14 dynamical
  (Bovy & Rix 13); z_d = 300 ± 50 pc; mass 3.5 ± 1 × 10^10. Thick disc: R_d 2.0 ± 0.2 (α-defined)
  or 3.6 ± 0.7 (geometric), z_d = 900 ± 180 pc, mass 6 ± 3 × 10^9, local Σ ratio 12 ± 4 %. Total
  stellar mass 5 ± 1 (BHG16) or 6.08 ± 1.14 × 10^10 (Licquia & Newman 15). McMillan 17 posterior:
  Σ_0,thin = 896 Msun/pc², R = 2.50 kpc; thick 183, 3.02.
* Vertical force: Σ_1.1 = 71 ± 6 (Kuijken & Gilmore 91), 74 ± 6 (Holmberg & Flynn 04), 68 ± 4
  (Bovy & Rix 13); consensus 70 ± 5. Baryons alone 47 ± 3, so Σ_DM(|z| < 1.1) ≈ 23 ± 6 and
  ρ_DM,⊙ ≈ 0.008–0.011 Msun/pc³.
* Bulge: McMillan M_b = 8.9e9 ± 10 % (posterior 9.23e9); dynamical bulge-box mass 1.85 ± 0.05 × 10^10
  (Portail+17) if a bar is modelled. Bar pattern speed 33–41 km/s/kpc, R_CR ≈ 5.7–6.1 kpc. The bar
  changes Pal 5's length and along-track density (Pearson+17, Banik & Bovy 19, He+26) but only
  weakly its track and kinematics; no study isolates NGC3201/M68.
* Gas: McMillan's HI (1.1e10) is 40–50 % above Kalberla & Kerp 09 / Nakanishi & Sofue 16 (7–8e9);
  H2 1.2e9. Gas is ~14 of the 70 Msun/pc² in Σ_1.1.
* LMC: negligible at 5–30 kpc in a disc-centred frame (Nibauer+25: residual azimuthal acceleration
  ≈ 3–4 % of v_c²/R at GD-1; He+26: modest for Pal 5). Budget as a systematic, do not model.

### 2.5 Solar frame

R_0 = 8.178 ± 0.026 (GRAVITY 19) vs 8.275 ± 0.034 (GRAVITY 21, aberration-corrected), a 3σ
difference. z_⊙ = 20.8 ± 0.3 pc. (U, V, W)_⊙ = (11.1, 12.24, 7.25) ± (1.25, 2.05, 0.62) incl.
systematics (Schönrich+10). Sgr A* proper motion −6.411 ± 0.008 mas/yr (Reid & Brunthaler 20) pins
v_φ,⊙ = 4.74047 μ R_0 = 248.5 ± 0.7 km/s at 8.178 (251.5 at 8.275), independent of any potential.
With v_c(R_0) = 229–234 (Eilers+19, Zhou+23) that implies V_⊙,pec = 14–20 km/s, not 12.24: the
"V_⊙ = 12 vs 24" problem (Bovy+12: 26 ± 3).

## 3. Recommendation, parameter by parameter

Status: **keep** = current prior is supported; **change** = revise; **add** = new nuisance.

### Dark halo (inferred)

| Parameter | Current | Recommended | Status | Why |
|---|---|---|---|---|
| log10 M200 (halo only) | U[11.699, 12.398] (0.5–2.5e12) | **U[11.6, 12.4]** (0.4–2.5e12) | change (widen low edge) | halo-only convention sits ~10 % below the 1.0–1.2e12 total consensus; 0.4e12 covers Bird/Eadie once baryons are removed; keep uniform in log so the analytic compositional score is untouched |
| ln c_v' | N(2.56, 0.272) | keep | keep | Boylan-Kolchin log-normal, c200 ≈ 10; the right centre for an uncontracted profile mimicking a contracted MW (DMO 8.3 would be biased) |
| γ | U[−2, 1.5] | **U[0, 1.7]** | change | γ < 0 is a central density hole with no support; contracted cusps reach 1.3–1.7; 1.7 cap keeps r_h finite. With α free (v2) γ no longer has to do the outer-curve work |
| α (transition) | U[0.5, 2.0] (v2) | keep | keep | Zhao family; the honest knob for inner-steep/outer-declining |
| β | identity 3 | keep (or N(3, 0.3)) | keep | every MW fit returns ≈ 3; Ibata's 2.53 was limit-driven |
| q = c/a (density, z) | U[0.5, 1.5] | keep | keep | matches Malhan, Bowden, Bovy (½–2), Koposov (0–2); contains all posteriors 0.56–1.40; density flattening is the modern convention; uniform in q is what the field uses |
| p = b/a | U[0.75, 1.25] (v2) | keep | keep | sims 0.88–0.94, N&B 0.75, W&H 1.013; p > 1 is physically distinct from p < 1 because yaw is fixed and the Sun breaks the azimuthal symmetry |
| tilt | U[0, 30]° (v2) | **U[0, 45]°** | change | observed 18° (N&B), ~30° (Han+22), 43° (2510.00095); sims 19° ± 20°. Note the tilt axis is fixed (orientation = (0, tilt, 0)); N&B need pitch AND yaw. Freeing yaw is the next step if tilt is retained as a science parameter |
| q(r) | constant | constant | keep | published radial variation sets in at r ≳ 30 kpc |

### Baryons

| Parameter | Current | Recommended | Status | Why |
|---|---|---|---|---|
| Σ_Disk (thin) | U[1e7, 3e9] Msun/kpc² (10–3000 Msun/pc²) | **parameterize by mass**: M_thin U[1.5e10, 7e10] (or N(3.5e10, 1e10) truncated), Σ_0 derived from R_d; if Σ must stay: U[3e8, 1.6e9] | change | 3e9 Msun/kpc² with R_d = 2.6 is a 1.3e11 disc, 2× the entire stellar mass of the Galaxy; the uniform-in-Σ prior is a strongly non-uniform, R_d-coupled mass prior. The need for 3e9 (v_term) was a symptom of the single-disc model, see next row |
| Thick disc | absent (onedisk) | **restore**: r_thick N(3.6, 0.7) or N(2.0, 0.2), dz_thick > 0 with z_thick N(0.9, 0.18), M_thick N(6e9, 3e9) (or f_Σ N(0.12, 0.04)) | change | ~20 % of the local stellar Σ and the inner-disc mass the terminal velocities ask for; without it Σ_thin or γ absorb it |
| r_Disk | N(2.6, 0.5) | keep, truncate [1.5, 4.5] | keep | BHG16; note the dynamical 2.15 ± 0.14 (Bovy & Rix) at 1σ |
| z_Disk | N(0.3, 0.05) | keep | keep | BHG16 / McMillan fixed 0.30 |
| Bulge amplitude | N(9.93e10, 1e10) (v2) | keep | keep | ≡ McMillan M_b ± 10 % |
| Gas discs | fixed McMillan | **add** global factor f_gas N(1.0, 0.25), marginalized | add | McMillan HI 1.1e10 vs 7–8e9 in HI surveys |
| Bar | none | none (document a few-% along-track-density systematic for Pal 5) | keep | pericentre ≈ 7–8 kpc within 1.3 R_CR; track/kinematics weakly affected |
| LMC | none | none (3–5 % acceleration systematic) | keep | inner 30 kpc moves with the disc |

### Solar frame (marginalized)

| Parameter | Current (v2) | Recommended | Status | Why |
|---|---|---|---|---|
| R_0 | N(8.178, 0.026) | **N(8.23, 0.06)** (or pick 8.275 ± 0.034) | change | GRAVITY 2019 and 2021 disagree by 3σ |
| z_⊙ | fixed 20.8 pc | keep | keep | Bennett & Bovy |
| U_⊙, W_⊙ | N(11.1, 1.25), N(7.25, 0.62) | keep | keep | Schönrich incl. systematics |
| V_⊙ / v_φ,⊙ | V_pec N(12.24, 2.05), v_φ,⊙ = v_c(R_0) + V_pec in the row's potential | **set v_φ,⊙ = 4.74047 × 6.411 × R_0 (248.5 ± 0.8 at 8.178)** and drop the V_pec prior; if the current form is kept, widen to N(12, 5) | change | Sgr A*'s proper motion fixes v_φ,⊙ to 0.3 % independently of the potential; the current coupling imposes the low-V solution and moves the Sun's velocity with the halo parameters |

### Progenitors and stream ages (marginalized locals)

Corrected B&H18 masses and radii (`_prog`) are supported. t_end U[2, 10] Gyr for all three is the
broadest defensible envelope; Palau, Wang & Han 25 put M68 at 3.04 (+5.63/−0.29) Gyr and the
Gaia-Enceladus argument favours the short end, so a U[2, 6] alternative is worth a coverage check
rather than a prior change.

## 4. Coverage check: does the recommended prior contain every published posterior?

| Quantity | Literature extremes | Inside recommended support? |
|---|---|---|
| q_ρ | 0.56 (Koposov+23 oblate) – 1.40 (prolate); 1.30 ± 0.25 (Posti & Helmi) | yes for [0.5, 1.5]; Posti's +1σ (1.55) is out, acceptable |
| γ | 0.06 ± 0.22 (Palau) – 1.60 (Nitschai) | yes for [0, 1.7]; Palau's −1σ (−0.16) is out and unphysical |
| M200 (halo) | 0.45e12 (Bird total minus baryons) – 1.5e12 (Watkins) | yes for [0.4, 2.5]e12 |
| c200 | 8 (DMO) – 15 (McMillan) | yes, within 1.5σ of N(2.56, 0.272) in ln c_94 |
| tilt | 18° – 43° | yes for [0, 45]° |
| disc mass | 4.2e10 (Ibata) – 6.5e10 (Palau) | yes for M_thin U[1.5, 7]e10 + thick |
| p | 0.75 (N&B) – 1.013 (W&H) | yes |

## 5. Two conventions to state in every write-up

1. **M200 is the dark halo's mass inside its own r200** (200 ρ_crit, H0 = 70.4). Add the row's
   baryons inside r200 (4–17 %, measured) before comparing with total-mass literature values.
2. **q, p are density axis ratios** of the halo Spheroid, not potential flattenings; convert with
   (1 − q_ρ) ≈ 3 (1 − q_Φ) when comparing with Koposov+10, Bowden+15, Erkal+19 or GD-1 total q_Φ.

## Sources

Stream inference: 0907.1085, 1502.02658, 1502.00484, 1609.01298, 1807.05994, 1804.06854,
1812.08192, 2009.10726, 2211.04495, 2212.03587, 2311.17202, 2407.21790, 2504.07187, 2205.11767,
2303.17406, 2007.00356, 1703.04627, 1512.04536, 2601.15373, 2607.05510.
Halo mass and concentration: 1912.02599, 1608.00971, 1911.04557, 1808.10456, 1805.01408,
2010.13801, 2111.09327, 1810.10036, 2207.08839, 1804.11348, 2603.13516, 1402.7073, 1601.02624,
1809.07326, 0911.4484, 2001.07742, 2004.10817, 1306.0898, 2106.05286, 2303.12838, 2309.00048.
Halo shape in simulations: 1809.07255, 1910.04045, 2009.09220, 2005.03025, 2008.02404, 2302.08853,
2202.07662, 2309.07208, 1207.4555, astro-ph/0405189, 1006.0537, 0902.2477, 1107.5582.
Baryons and solar frame: 1602.07702, 1407.1078, 1309.0809, 1704.05063, 1711.03103, 1802.09291,
1608.07954, 1903.02009, 2107.10875, 1511.08877, 1904.05721, 2101.12098, 1809.03507, 0912.3693,
1209.0759, 2001.04386, 1810.09466, 2212.10393, 2607.17040, 2205.01688, 2107.13004, 1811.03631.

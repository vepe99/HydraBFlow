# Summary statistics for stellar streams — literature review and recommendations for ODISSEO II

Context: your current model embeds raw stream particles with a SetTransformer and does compositional score modelling. The posteriors are unstable across ML hyperparameters. The diagnosis this points to is a classic one for SBI on point clouds: the network is left to *discover* the potential-sensitive summaries itself from a variable-size, permutation-invariant, sampling-noisy set, and it can latch onto different (partly spurious) features under different seeds/hyperparameters. Injecting physically-motivated, low-dimensional, potential-sensitive summaries alongside the particles gives the network a robust "backbone" and typically stabilises and sharpens the posterior. This is exactly the historical trade-off the SBI-for-streams literature describes: bespoke summaries (robust, lossy) vs. raw data (in principle lossless, but hard to learn from).

---

## 1. The canonical set: 6D stream tracks in stream-aligned coordinates

The single most established summarisation of a smooth stream is the set of **1D tracks as functions of the along-stream angle φ1** in a stream-aligned (great-circle) coordinate frame (φ1, φ2). Bonaca & Hogg (2018), "The information content in cold stellar streams", show that for a smooth stream essentially *all* the Galactic-potential information is carried by these six tracks:

| Track | Symbol | Sensitivity |
|---|---|---|
| On-sky transverse position | φ2(φ1) | Stream track / potential shape, flattening |
| Heliocentric distance | D(φ1) | Radial potential gradient, 3D geometry |
| Line-of-sight velocity | v_los(φ1) | Potential depth / rotation |
| Proper motion along | μ_φ1(φ1) | Orbital energy, potential |
| Proper motion across | μ_φ2(φ1) | Halo flattening, misalignment |
| (radial-velocity gradient) | dv_los/dφ1 | Local potential curvature |

Practical encoding for your feature vector: fit each track with a low-order polynomial or spline in φ1 and pass the **coefficients** (e.g. degree 3–5 → ~4–6 numbers per track), or pass binned means over a fixed φ1 grid. This is compact (~30–40 numbers for all six), robust to particle count, and differentiable if you fit by least squares. Bonaca & Hogg also give a Cramér–Rao / Fisher-information framing you can reuse to argue *which* tracks matter most for your three streams.

## 2. Width, dispersion and length (second-moment tracks)

Beyond the mean tracks, the **spread** around them is potential-sensitive:

- **Transverse width** w(φ1) — the FWHM/σ of the φ2 density profile as a function of φ1. Physical width varies strongly (≈200 pc–1 kpc) and its variation encodes the potential and epicyclic structure; the connection between width and orbital inclination gives an independent handle on halo **flattening / symmetry axis** (Erkal et al.; DES stream work).
- **Velocity dispersion** σ_v(φ1) — often near-constant (~few km/s) even where width varies, so it anchors the progenitor mass and normalises the width signal.
- **Stream length / angular extent** and **leading–trailing asymmetry** — total φ1 span and the length ratio / density contrast between arms carry orbital-phase and potential information.
- Transverse profile is well described by a Gaussian core (~0.4° half-width) plus a broader exponential envelope — you can pass the core width and the envelope fraction as two extra numbers per bin if you want the "cocoon".

## 3. Action–angle / frequency-space summaries (most directly potential-sensitive)

This is the classical potential-constraint route and is worth passing explicitly because it converts geometry into a near-linear, potential-diagnostic structure:

- **Sanders & Binney (2013)** ("Stream–orbit misalignment II"): in the *true* potential the stars of a long, narrow stream lie on a **straight line in angle–frequency space**; the misalignment/curvature away from linearity is a direct wrongness-of-potential signal, independent of progenitor mass. Summaries: the direction of the best-fit line in frequency space, and the residual scatter / curvature about it.
- **Mean actions** (J_R, J_φ=L_z, J_z) and their dispersions. Note L_z is *exactly* conserved in an axisymmetric potential, so its distribution width is a clean diagnostic. Reyes/Yang, Malhan et al. and the "clustering in action space" line (arXiv:2007.00356, 1908.02336) constrain the potential by requiring stream stars to **cluster tightly in action space** — you can pass the action-space dispersion (or a clustering/entropy score) computed under your sampled potential θ_j.
- **Energy** distribution mean/spread.

Caveat: actions/frequencies must be computed *under some potential*. Two clean options for SBI: (a) compute them under a fixed **reference** potential so the summary is a deterministic function of the observed particles (the network learns how "wrong" the reference looks), or (b) compute them under the *sampled* θ_j inside the simulator and pass the resulting compactness — this is more informative but couples the summary to θ. AGAMA computes actions natively (Staeckel fudge), so this is cheap for you.

## 4. Orbital-pole / great-circle summaries

For each star (or for the stream as a whole): the **orbital pole** direction (from position × velocity), the **spread in orbital pole**, **deviation from a great circle**, and the **proper-motion misalignment angle** (angle between the velocity and the stream track). Sanders & Binney (2013, papers I & II) and the MW–LMC stream-population work (Brooks et al. 2024) show these are explicitly dependent on the assumed potential and are robust, low-dimensional, and cheap. A great-circle track corresponds to zero misalignment; departures scale with potential flattening and time-dependence.

## 5. Density-structure / power-spectrum summaries (mostly for substructure — lower priority for you)

The **linear density profile** ρ(φ1) and its **power spectrum / autocorrelation** of fluctuations (Bovy et al. 2017; the density-structure work, arXiv:1811.10084) are the standard summaries for **subhalo / dark-matter substructure** inference — this is what the "bespoke power spectrum of density perturbations" in the SBI-streams intros refers to. Since ODISSEO II targets the *global* smooth potential (+ circular-velocity curve) rather than gaps from subhalos, these are lower priority, but the smooth large-scale density gradient and any gap statistics are cheap to include and don't hurt.

## 6. Progenitor / global scalars

If your model marginalises or infers the progenitor: **progenitor phase-space position and velocity**, **estimated progenitor mass** (from σ_v and width), total **star count / luminosity**, and the **accretion-time proxy** from frequency separation (individual-stream frequency spread shrinks with time). These are natural conditioning scalars.

---

## Recommended concrete feature block to concatenate with the SetTransformer embedding

A compact, robust, physically-motivated vector per stream (order ~50–80 numbers):

1. Polynomial/spline coefficients of the six tracks φ2, D, v_los, μ_φ1, μ_φ2 vs φ1 (§1).
2. Width w(φ1) and velocity dispersion σ_v(φ1) on a fixed φ1 grid; stream length; leading/trailing length ratio (§2).
3. Mean and dispersion of actions (J_R, L_z, J_z) and energy, computed under a fixed reference potential (and optionally under θ_j); action-space compactness score (§3).
4. Mean orbital pole, orbital-pole spread, mean proper-motion misalignment, great-circle deviation (§4).
5. Progenitor scalars if applicable (§6).

Compute all of these with AGAMA in the same frame across simulations so they are comparable, and standardise (z-score) each summary using the prior-predictive distribution before feeding the network.

## Suggested experiment design (three variants, matched otherwise)

Run these with identical priors, simulator, noise/selection model, and training budget, and sweep ML hyperparameters for each:

- **A — Particles only (current):** SetTransformer on raw particles → score model.
- **B — Hybrid:** SetTransformer particle embedding **concatenated** with the standardised summary vector → score model. (Also worth trying: summaries injected as extra "tokens", or a small MLP on summaries fused with the set embedding.)
- **C — Summaries only:** MLP on the summary vector → score model (no per-particle set input).

What to measure — and note that your stated symptom (posterior disagreement across seeds/hyperparameters) is itself the key metric:

- **Robustness / reproducibility:** spread of the posterior (means, credible-interval widths) across random seeds and hyperparameter settings. Expect B and C to be markedly tighter/more consistent than A if the summaries carry the signal.
- **Calibration:** simulation-based calibration (SBC) rank histograms and/or **TARP** coverage; expected coverage probability.
- **Sharpness:** posterior contraction / credible-interval width **conditioned on being calibrated** (only compare sharpness among models that pass calibration).
- **Accuracy on held-out mocks:** posterior-mean error and log-posterior at truth; and on the real Pal 5 / NGC 3201 / M68 data, agreement with Palau & Miralda-Escudé (2023) and literature MW values.
- **Information attribution:** if B ≈ C, the summaries dominate and the raw particles add little (useful to report); if B > both A and C, the set and summaries are complementary — the ideal outcome and a clean story for the paper.

A likely finding, well-supported by the literature, is that the summaries stabilise the low-order potential parameters (mass, scale, flattening) while the raw particle set still helps for anything the fixed summaries throw away.

---

## Key references

- Bonaca & Hogg 2018, *The information content in cold stellar streams* — arXiv:1804.06854. The 6D-track / Fisher-information framing (§1).
- Sanders & Binney 2013, *Stream–orbit misalignment I & II* — arXiv:1305.1935, arXiv:1305.1937. Misalignment, angle–frequency linearity, potential algorithm (§3, §4).
- Reino/Yang et al. 2021 & Malhan et al., *Clustering in action space* — arXiv:2007.00356, arXiv:1908.02336. Action-space compactness as a potential constraint (§3).
- Bovy, Erkal & Sanders 2017, and *The Density Structure of Simulated Stellar Streams* — arXiv:1811.10084. Density power spectrum for substructure (§5).
- Alvey, Gerdes & Weniger 2023, *Albatross* (TMNRE/swyft, sstrax) — arXiv:2304.02032; and Hermans et al. 2021, *Towards constraining warm dark matter…* — arXiv:2011.14923. SBI-on-streams that move away from handcrafted summaries.
- Nibauer et al. 2024/25, *StreamSculptor* (differentiable streams, Hamiltonian perturbation theory).
- Sun et al. 2025, *Stream Members Only* (mixture density networks for track/width/density) — arXiv:2311.16960.
- Brooks et al. 2024, *LMC calls, Milky Way halo answers* — width, great-circle deviation, pole spread as potential-dependent stream-population statistics.
- Viterbo & Buck 2026 (Paper I) / *The dynamical memory of tidal stellar streams* — arXiv:2512.04600 (your flow-matching GD-1 work).
# Task: augment the dataset with HI terminal-velocity and vertical potential constraints

> **How to use this file (Claude Code):** this is a single self-contained brief. All data files live in the project's `assets/` folder. The rotation-curve datasets are embedded below as fenced ```csv blocks, each starting with a `# FILE: assets/<name>.csv` comment. The terminal-velocity and vertical-density data are external (download instructions given inline). Implement the helpers in *Reference implementation*, then follow *How to augment the dataset*.
>
> **STEP 0 — check what already exists before writing anything.** The current pipeline already has a rotation-curve (circular-velocity) modality, so some of this data may already be in the repo. Before creating files:
> 1. `ls assets/` and grep the data-loading / simulator code for existing rotation-curve, terminal-velocity, `Kz`/`sigma_z`/surface-density, and vertical-density inputs (search terms: `rotation`, `vcirc`, `circular`, `vterm`, `terminal`, `Kz`, `surface_density`, `Sigma`, `rho_z`, `Eilers`, `Zhou`, `Huang`).
> 2. If a rotation-curve file already exists, **do not overwrite it** — report what it is (source, R range, columns, units) and ask before replacing. If it is the old Eilers 2019 curve, replace it with the Zhou+Huang files below (that was the point of this change); otherwise reuse what is there.
> 3. Only write the embedded CSVs (and add the terminal-velocity / vertical modalities) for data that is **not already present**. Avoid duplicating datasets under different names.

## Goal

Extend the training dataset so that, alongside the stellar-stream particles and the circular-velocity (rotation-curve) modality, each simulated example also carries three new **potential-derived observables** from Ibata et al. 2023 (*Charting the Galactic acceleration field II*, arXiv:2311.17202):

1. **HI terminal-velocity curve** `v_term(ℓ)` (inner-Galaxy radial mass distribution)
2. **Vertical force / local surface density** `Σ(z=1.1 kpc)` (disk–halo decomposition)
3. **Vertical stellar density profile** `ρ(z)` at the Solar radius (shape only)

These are **deterministic functions of the AGAMA `Potential` object** built from the sampled parameters `θ` — no stream (particle-spray) simulation is needed to compute them. They are cheap (just force/density evaluations) and should be generated inside the existing simulator loop, right after the potential is constructed for each `θ`.

## Unit system

Use `agama.setUnits(mass=1, length=1, velocity=1)` → mass in **Msun**, length in **kpc**, velocity in **km/s**. In this system `pot.force(xyz)` returns the acceleration `−∇Φ` in units of `(km/s)² / kpc`, and `pot.density(xyz)` returns `Msun / kpc³`. Newton's constant is `G = 4.300917270e-6` (kpc·(km/s)²/Msun). Fix `R0 = 8.178` kpc (GRAVITY 2019), as Ibata does.

> **Check to run once:** verify `G` matches AGAMA's internal convention in these units by comparing `Σ(1.1 kpc)` for a known McMillan (2017) model against its published value (~71 Msun/pc²). If it is off by a constant factor, `G` or the unit system is wrong.

## Reference implementation

```python
"""
Ancillary Milky-Way potential observables from an AGAMA potential, following
Ibata et al. 2023 (arXiv:2311.17202), Eqs. 12-13 and Section 5.
Each function is a pure function of the AGAMA Potential object.
"""
import numpy as np
import agama

agama.setUnits(mass=1, length=1, velocity=1)   # Msun, kpc, km/s

G  = 4.300917270e-6    # kpc * (km/s)^2 / Msun  (must match AGAMA's convention)
R0 = 8.178             # kpc, fixed (GRAVITY 2019)


# --- Circular velocity  vc(R): building block for rotation curve & v_term ---
# vc^2 = R dPhi/dR = -R * F_R ; at (R,0,0) the radial force is force[...,0]
def vcirc(pot, R):
    R = np.atleast_1d(np.asarray(R, float))
    xyz = np.column_stack([R, np.zeros_like(R), np.zeros_like(R)])
    fR = pot.force(xyz)[:, 0]                 # radial (x) force, < 0
    return np.sqrt(np.maximum(-R * fR, 0.0))


# --- 1. HI terminal velocity  v_term(l)  (Eq. 13) ---
# v_term(l) = sgn(sin l) * vc(R0|sin l|) - vc(R0) * sin l
# Keep |sin l| > 0.5 (avoid the bar); data averaged over 2 deg; sigma = 6.2 km/s.
def terminal_velocity(pot, l_deg):
    l = np.radians(np.atleast_1d(np.asarray(l_deg, float)))
    sinl = np.sin(l)
    Rt = R0 * np.abs(sinl)                     # tangent-point radius
    return np.sign(sinl) * vcirc(pot, Rt) - vcirc(pot, R0) * sinl


# --- 2. Vertical force -> local surface density (Kuijken & Gilmore 1991) ---
# Sigma(z) = |K_z| / (2 pi G),  K_z = -dPhi/dz = force_z ; obs: 71 +/- 6 Msun/pc^2
def surface_density(pot, z=1.1, R=R0):
    Fz = pot.force([R, 0.0, z])[2]             # (km/s)^2/kpc, < 0 for z>0
    Kz = -Fz
    return Kz / (2.0 * np.pi * G) / 1.0e6       # Msun / pc^2


# --- 3. Vertical stellar density profile (Ibata et al. 2017b) ---
# SHAPE only (free normalization). Pass the disk component as its own Potential.
def vertical_density_profile(disk_pot, z_vals, R=R0):
    z_vals = np.atleast_1d(np.asarray(z_vals, float))
    xyz = np.column_stack([np.full_like(z_vals, R),
                           np.zeros_like(z_vals), z_vals])
    return disk_pot.density(xyz)                # Msun / kpc^3
```

## How to augment the dataset

For **every** sampled parameter vector `θ` already in / added to the dataset:

1. Build the AGAMA potential `pot` from `θ` exactly as the current simulator does. Keep a reference to the **disk component(s)** as separate `agama.Potential` objects so their density can be read individually for observable 3.
2. Evaluate the three observables on fixed grids (define these grids once, store in the dataset metadata so every example is aligned):
   - `l_grid`: Galactic longitudes with `|sin ℓ| > 0.5`, e.g. spanning the inner Galaxy in ~2° steps → `v_term = terminal_velocity(pot, l_grid)`.
   - `Sigma = surface_density(pot, z=1.1)` → a single scalar.
   - `z_grid`: e.g. `np.linspace(0.1, 5.0, N_z)` kpc → `rho_z = vertical_density_profile(disk_pot, z_grid)`; store the **shape** (normalization is marginalized at training/inference time).
3. Add noisy "observed" counterparts so these match how real measurements enter (mirroring the rotation-curve modality):
   - `v_term_obs = v_term + N(0, 6.2)` km/s per point.
   - `Sigma_obs  = Sigma + N(0, 6.0)` Msun/pc².
   - `rho_z_obs`: add per-point Gaussian noise at the assumed relative uncertainty; store `rho_err` too.
4. Store these as **new modalities** in each dataset record, e.g.:
   ```
   record["vterm"]   = {"l_grid": l_grid,  "value": v_term_obs, "sigma": 6.2}
   record["sigma_z"] = {"z": 1.1,          "value": Sigma_obs,  "sigma": 6.0}
   record["rho_z"]   = {"z_grid": z_grid,  "value": rho_z_obs,  "err": rho_err}
   ```
   Keep the grids identical across all records so tensors stack cleanly.

## Observational data (observed values + reported uncertainties)

These are the real measurements to use, and — crucially — the **reported uncertainties** to feed the resampling step. Convention: for each **simulated** example, evaluate the model observable from the potential and add Gaussian noise with the *reported observational σ* below (so the simulated "data" scatter matches real measurement precision). The observed *central values* are used only at inference time, on the real data. Match your model grid to each dataset's grid (`R`, `ℓ`, `z`).

### Rotation curve (replaces Eilers 2019)

The full data is embedded below. **Claude Code: after STEP 0, write each block verbatim to `assets/` (only if not already present), then load with `numpy.genfromtxt(fname, delimiter=",", names=True, comments="#")`.** Use Zhou for R ≤ 24 kpc and Huang for r > 25 kpc (Huang's inner HKG points overlap Zhou's range but are far less precise). Resample: `Vc_obs = Vc_model(R) + N(0, sigma_Vc)`.

Zhou et al. 2023 (ApJ 946, 73), Table 4. ~54,000 thin-disk LRGB stars, Jeans model. **R = 5.24 → 24.00 kpc**, 34 points. `sigma_Vc` is statistical (bootstrap), 0.16–4.94 km/s; plus ~1.36 km/s systematic (add in quadrature for a conservative floor). Adopted `R0 = 8.34 kpc`, `Vc(R0) = 234.04 km/s`.

```csv
# FILE: assets/rc_zhou2023.csv
# Rotation curve, Zhou et al. 2023 (ApJ 946, 73), Table 4.
# sigma_Vc = statistical (bootstrap) error only; systematic ~1.36 km/s at R0.
R_kpc,Vc_kms,sigma_Vc_kms,N_stars
5.24,225.10,0.69,845
5.74,233.53,0.68,692
6.25,234.30,0.62,704
6.77,233.17,0.60,759
7.23,236.19,0.45,1061
7.83,236.00,0.29,2288
8.21,233.19,0.26,2550
8.78,233.15,0.22,3281
9.26,232.15,0.17,4583
9.75,231.24,0.16,5061
10.25,230.34,0.17,4881
10.75,230.54,0.18,4564
11.25,229.11,0.19,4005
11.75,227.48,0.20,3431
12.24,226.69,0.25,2844
12.74,225.56,0.27,2312
13.25,224.90,0.27,2116
13.74,223.57,0.31,1825
14.23,221.10,0.40,1362
14.74,220.19,0.43,987
15.23,219.59,0.50,801
15.74,217.36,0.68,563
16.24,216.61,0.74,446
16.74,217.28,0.87,308
17.23,216.25,1.02,257
17.74,213.81,1.15,163
18.35,217.53,1.45,207
18.90,212.10,1.58,97
19.50,210.46,1.32,162
20.41,206.69,1.71,85
21.28,207.71,1.69,93
22.39,203.72,2.01,46
23.16,205.20,2.50,20
24.00,200.64,4.94,10
```

Huang et al. 2016 (MNRAS 463, 2623), Table 3 (final combined RC). Halo K giants, spherical Jeans equation. **Outer points r = 26 → 99 kpc**, 16 points, `sigma_Vc` ≈ 17–26 km/s. Adopted `R0 = 8.34 kpc`, `Vc(R0) = 239.89 km/s`. (One intermediate HKG point near ~88 kpc was not cleanly recoverable from the source extraction — verify against Huang+2016 Table 3 if you need it.)

```csv
# FILE: assets/rc_huang2016_outer.csv
# Outer RC, Huang et al. 2016 (MNRAS 463, 2623), Table 3; halo K giants (HKG).
# Only points r > 25 kpc kept (use Zhou 2023 below that). sigma_Vc = 1-sigma (MC).
r_kpc,Vc_kms,sigma_Vc_kms,tracer
26.22,242.02,18.66,HKG
28.71,224.11,16.97,HKG
31.29,211.20,16.43,HKG
33.73,217.93,17.66,HKG
36.19,219.33,18.44,HKG
38.73,213.31,17.29,HKG
41.25,200.05,17.72,HKG
43.93,190.15,18.65,HKG
46.43,198.95,20.70,HKG
48.71,192.91,19.24,HKG
51.56,198.90,21.74,HKG
57.03,185.88,21.56,HKG
62.55,173.89,22.87,HKG
69.47,196.36,25.89,HKG
79.27,175.05,22.71,HKG
98.97,147.72,23.55,HKG
```

> Unit caveat: both RC works adopt `R0 = 8.34 kpc`, whereas the AGAMA helper uses `R0 = 8.178 kpc` (GRAVITY). This shifts `vcirc(R0)` by <1%. Decide on ONE `R0` and use it consistently in both the data and `vcirc`; simplest is to set `R0 = 8.178` everywhere and treat the small offset as absorbed by the systematic error.

### HI terminal velocity

Source data (Ibata uses both quadrants):

- 4th quadrant: McClure-Griffiths & Dickey 2007 (ApJ 671, 427) — VizieR `J/ApJ/671/427`.
- 1st quadrant: McClure-Griffiths & Dickey 2016 (ApJ 831, 124) — VizieR `J/ApJ/831/124`.

First check STEP 0 — if a terminal-velocity file already exists in `assets/`, reuse it. Otherwise the raw curves are densely sampled (thousands of points). Ibata's processing: **average over 2° longitude intervals**, keep only `|sin ℓ| > 0.5` (avoids the Galactic bar), and adopt a **measurement uncertainty of 6.2 km/s** per (averaged) point. Pull, bin, and cache to `assets/terminal_velocity.csv` (columns `l_deg,vterm_kms,sigma_kms`), e.g.:

```python
from astroquery.vizier import Vizier
Vizier.ROW_LIMIT = -1
t4 = Vizier.get_catalogs("J/ApJ/671/427")[0]   # 4th quadrant (l, v_term)
t1 = Vizier.get_catalogs("J/ApJ/831/124")[0]   # 1st quadrant
# -> build l_grid in 2 deg bins with |sin l|>0.5; average v_term per bin; sigma=6.2
# -> save to assets/terminal_velocity.csv
```

Resample (simulated): `v_term_obs = terminal_velocity(pot, l_grid) + N(0, 6.2)`.

### Vertical force / local surface density

Single scalar, no grid: **Σ(z = 1.1 kpc) = 71 ± 6 M⊙ pc⁻²** (Kuijken & Gilmore 1991), at `R = R0`. Resample: `Sigma_obs = surface_density(pot) + N(0, 6.0)`.

### Vertical stellar density profile

Check STEP 0 first — reuse any existing vertical-density file in `assets/`. Otherwise: Ibata et al. 2017b (north Galactic cap): use the `b > 70°` sample, `z < 5 kpc`, decomposed into thin/thick disk by photometric metallicity. **Only the relative shape is constrained** (free multiplicative normalization). The profile is published as a figure (their Fig. 12f), so the numeric `ρ(z)` points must be digitized from that figure or requested from the authors; save them to `assets/vertical_density_ibata2017b.csv` as `(z_kpc, rho_rel, rho_err)`. Because the normalization is free, absolute units do not matter — only the per-point relative errors, which set the resampling scatter. Resample the *shape*: `rho_obs = A·rho_model(z_grid) + N(0, rho_err)` with `A` fixed by the best-fit normalization (see `vertical_density_chi2`).

The vertical-force datum (Kuijken & Gilmore 1991, `Σ(1.1 kpc) = 71 ± 6 M⊙/pc²`) is a single scalar — no file needed; hard-code it (or add a one-line `assets/vertical_force_kg91.txt` if you prefer everything in `assets/`).

## (Optional) explicit Gaussian likelihood terms

The dataset augmentation above is self-contained: each observable is stored as a
noisy "observed" vector plus its uncertainty, and whatever inference method is
used will learn from it like any other modality. **This section is optional** and
only needed if you also want the closed-form χ²-like log-likelihood terms (Ibata's
`lnL_ancillary`) — e.g. for a classical/MCMC cross-check or an explicit likelihood.
They are not required for the augmentation itself.

```python
def rotation_curve_chi2(pot, R_data, vc_data, vc_err):          # Eq. 12
    # Zhou 2023 (R<~25 kpc) + Huang 2016 (outer); no R<15 cut needed.
    return 0.5*np.sum(((vc_data - vcirc(pot, R_data))/vc_err)**2)

def terminal_velocity_chi2(pot, l_deg, vterm_data, sigma=6.2):  # Eq. 13
    l_deg = np.asarray(l_deg, float)
    m = np.abs(np.sin(np.radians(l_deg))) > 0.5
    return 0.5*np.sum(((vterm_data[m] - terminal_velocity(pot, l_deg[m]))/sigma)**2)

def vertical_force_chi2(pot, z=1.1, meas=71.0, err=6.0):
    return 0.5*((surface_density(pot, z) - meas)/err)**2

def vertical_density_chi2(disk_pot, z_vals, rho_data, rho_err, R=R0):
    """Shape-only: analytically marginalize a free scale factor A."""
    model = vertical_density_profile(disk_pot, z_vals, R)
    w = 1.0/rho_err**2
    A = np.sum(w*rho_data*model)/np.sum(w*model**2)             # best-fit norm
    return 0.5*np.sum(w*(rho_data - A*model)**2)

def lnL_ancillary(pot, disk_pot, rc=None, vterm=None,
                  sigma_z=(71.0, 6.0), vprof=None):
    chi2 = 0.0
    if rc      is not None: chi2 += rotation_curve_chi2(pot, *rc)
    if vterm   is not None: chi2 += terminal_velocity_chi2(pot, *vterm)
    if sigma_z is not None: chi2 += vertical_force_chi2(pot, 1.1, *sigma_z)
    if vprof   is not None: chi2 += vertical_density_chi2(disk_pot, *vprof)
    return -chi2
```

## Notes / gotchas

- `pot.force(xyz)` returns shape `(N,3)` for `(N,3)` input and `(3,)` for a single `(3,)` point — the helpers above handle both.
- The terminal-velocity formula assumes circular orbits and the tangent-point method; it is valid for the inner Galaxy only, hence the `|sin ℓ| > 0.5` cut (also avoids the bar).
- The surface-density term uses the plane-parallel slab approximation `Σ = |K_z|/(2πG)`, matching Ibata.
- For the vertical density profile, only the shape is informative (free normalization), so always marginalize `A` — do **not** fit an absolute normalization.
- Physically, the terminal-velocity term constrains the inner radial mass distribution and the vertical terms break the disk↔halo degeneracy that streams + the outer rotation curve leave open. That is the reason to add them.

## Sanity checks to include

Add a quick test (analytic limits) so regressions are caught:

- Flat rotation curve `vc = V = const` ⇒ `vcirc(R0) == V` and `v_term(30°) == 0.5·V`.
- Tune a constant `K_z` such that `Σ(1.1) == 71` and confirm the conversion.
- Compare `Σ(1.1 kpc)` and `vc(R0)` for a McMillan (2017) potential against published values.
"""Diagnostic: is the "concentrated, low-mass Milky Way" a prior/degeneracy artifact or data?

Symptom under investigation: the global posterior prefers a small halo scale radius
``a_TwoPowerTriaxial_halo`` (concentrated) and a low total mass. Two very different causes:

  (1) PRIOR / PARAMETRIZATION ARTIFACT -- a *uniform* prior on ``(rho, a)`` is strongly
      NON-uniform in the physically meaningful derived quantities (enclosed mass, concentration),
      and the streams+rotation-curve only pin the enclosed mass over a limited radial range, so
      ``a`` and ``rho`` are degenerate. If so, the "preference" is the prior pushforward projected
      onto a badly chosen axis -- the posterior on ``a`` just tracks the prior, and the real
      constraint (enclosed mass) is fine. Fix = reparametrize / report M_enc(r), not the network.

  (2) GENUINE DATA-DRIVEN low mass -- the real Gaia streams actually pull the enclosed mass below
      what simulated data does. If so it is either physics or (more likely) misspecification.

This script decides between them WITHOUT retraining, by comparing on derived quantities:
  Panel A  M_enc(r) at pivot radii: prior pushforward vs the real posterior (Combined + per-stream).
  Panel B  the (a_NFW, M_enc(20 kpc)) and (rho, a_NFW) planes: shows the degeneracy valley and
           where prior vs posterior draws sit in it.
  Panel C  simulated recovery of M_enc(20 kpc): truth distribution vs recovered posterior median --
           if these match, there is NO bias on well-specified data => symptom is real-data only.

It also reports M_enc at a large radius (default 100 kpc) with a WARNING: for the freed outer
slope beta in [2, 4], a TwoPowerTriaxial halo has INFINITE total mass when beta <= 3, so any
"total mass" number is ill-defined -- always quote M_enc at a fixed finite radius.

Standalone: agama + numpy + matplotlib (+ omegaconf to read configs). No hydrabflow / BayesFlow /
JAX import (mirrors scripts/ppc_rotation_curve.py), so it runs on a CPU-only cluster and starts
instantly. The host potential + rotation-curve constants are copied verbatim from
hydrabflow.simulators.stream_agama / stream_common.

Usage:
    uv run --no-sync python scripts/diagnose_mass_prior.py \
        --real-run outputs/stream_agama/stream_fusion_model5_spray_huang/2026-07-08_14-42-49 \
        --sim-run  outputs/stream_agama/stream_fusion_model5_spray_huang/2026-07-08_14-33-47 \
        --sim-truth data/streams/data_agama_spray_huang_hydrabflow/simulation_multistream_333.npz \
        --n-prior 4000 --pivot-kpc 20
"""

from __future__ import annotations

import argparse
import os

import numpy as np

# --------------------------------------------------------------------------------------------- #
# Host potential + rotation curve, copied verbatim from stream_agama / stream_common so the prior
# rejection here is identical to the simulator's, without importing the JAX-heavy package.
# --------------------------------------------------------------------------------------------- #
BULGE_PARAMS = dict(
    type="Spheroid", scaleRadius=75 / 1e3, densityNorm=9.6e10, gamma=0, alpha=1, beta=1.8,
    cutoffStrength=2, outerCutoffRadius=2.1, axisRatioY=1.0, axisRatioZ=0.5,
)
OBS_R_KPC = np.array([
    5.24, 5.74, 6.25, 6.77, 7.23, 7.83, 8.21, 8.78, 9.26, 9.75,
    10.25, 10.75, 11.25, 11.75, 12.24, 12.74, 13.25, 13.74, 14.23, 14.74,
    15.23, 15.74, 16.24, 16.74, 17.23, 17.74, 18.35, 18.90, 19.50, 20.41,
    21.28, 22.39, 23.16, 24.00,
])
OBS_VC_KMS = np.array([
    225.10, 233.53, 234.30, 233.17, 236.19, 236.00, 233.19, 233.15, 232.15, 231.24,
    230.34, 230.54, 229.11, 227.48, 226.69, 225.56, 224.90, 223.57, 221.10, 220.19,
    219.59, 217.36, 216.61, 217.28, 216.25, 213.81, 217.53, 212.10, 210.46, 206.69,
    207.71, 203.72, 205.20, 200.64,
])
OBS_SIGMA_VC = np.array([
    0.69, 0.68, 0.62, 0.60, 0.45, 0.29, 0.26, 0.22, 0.17, 0.16,
    0.17, 0.18, 0.19, 0.20, 0.25, 0.27, 0.27, 0.31, 0.40, 0.43,
    0.50, 0.68, 0.74, 0.87, 1.02, 1.15, 1.45, 1.58, 1.32, 1.71,
    1.69, 2.01, 2.50, 4.94,
])
_HUANG = np.array([
    [4.60, 231.24, 7.00], [5.08, 230.46, 7.00], [5.58, 230.01, 7.00],
    [6.10, 239.61, 7.00], [6.57, 246.27, 7.00], [7.07, 243.49, 7.00],
    [7.58, 242.71, 7.00], [8.04, 243.23, 7.00],
    [8.34, 239.89, 5.92], [8.65, 237.26, 6.29], [9.20, 235.30, 5.60],
    [9.62, 230.99, 5.49], [10.09, 228.41, 5.62], [10.58, 224.26, 5.87],
    [11.09, 224.94, 7.02], [11.58, 233.57, 7.65], [12.07, 240.02, 6.17],
    [12.73, 242.21, 8.64], [13.72, 261.78, 14.89], [14.95, 259.26, 30.84],
    [15.52, 268.57, 49.67], [16.55, 261.17, 50.91], [17.56, 240.66, 49.91],
    [18.54, 215.31, 24.80], [19.50, 214.99, 24.42], [21.25, 251.68, 19.50],
    [23.78, 259.65, 19.62], [26.22, 242.02, 18.66], [28.71, 224.11, 16.97],
    [31.29, 211.20, 16.43], [33.73, 217.93, 17.66], [36.19, 219.33, 18.44],
    [38.73, 213.31, 17.29], [41.25, 200.05, 17.72], [43.93, 190.15, 18.65],
    [46.43, 198.95, 20.70], [48.71, 192.91, 19.24], [51.56, 198.90, 21.74],
    [57.03, 185.88, 21.56], [62.55, 173.89, 22.87], [69.47, 196.36, 25.89],
    [79.27, 175.05, 22.71], [98.97, 147.72, 23.55],
])
HUANG_R_KPC, HUANG_VC_KMS, HUANG_SIGMA_VC = _HUANG.T

GLOBAL_KEYS = [
    "rho_TwoPowerTriaxial_halo", "gamma_TwoPowerTriaxial_halo", "a_TwoPowerTriaxial_halo",
    "beta_TwoPowerTriaxial_halo", "q_TwoPowerTriaxial_halo", "r_Disk", "z_Disk", "Sigma_Disk",
]


def extended_rotation_curve(split_kpc):
    split = float(OBS_R_KPC.max()) if split_kpc is None else float(split_kpc)
    hi = HUANG_R_KPC > split
    r = np.concatenate([OBS_R_KPC, HUANG_R_KPC[hi]])
    vc = np.concatenate([OBS_VC_KMS, HUANG_VC_KMS[hi]])
    sig = np.concatenate([OBS_SIGMA_VC, HUANG_SIGMA_VC[hi]])
    order = np.argsort(r)
    return r[order], vc[order], sig[order]


def _agama():
    import agama

    agama.setUnits(length=1, velocity=1, mass=1)
    return agama


def _host_potential(agama, p):
    return agama.Potential(
        BULGE_PARAMS,
        dict(type="Spheroid", scaleRadius=p["a_TwoPowerTriaxial_halo"],
             densityNorm=p["rho_TwoPowerTriaxial_halo"], gamma=p["gamma_TwoPowerTriaxial_halo"],
             alpha=1, beta=p["beta_TwoPowerTriaxial_halo"], cutoffStrength=2,
             outerCutoffRadius=np.inf, axisRatioY=1.0, axisRatioZ=p["q_TwoPowerTriaxial_halo"]),
        dict(type="Disk", scaleRadius=p["r_Disk"], scaleHeight=p["z_Disk"],
             surfaceDensity=p["Sigma_Disk"], sersicIndex=1, innerCutoffRadius=0),
    )


def _vcirc(pot, obs_r):
    pts = np.column_stack((obs_r, np.zeros_like(obs_r), np.zeros_like(obs_r)))
    v2 = -obs_r * pot.force(pts)[:, 0]
    return np.sqrt(np.where(v2 > 0, v2, np.nan))


def _menc(pot, radii):
    """Enclosed mass [Msun] at each radius; NaN on failure."""
    out = np.full(len(radii), np.nan)
    try:
        out[:] = pot.enclosedMass(np.asarray(radii, float))
    except Exception:
        for i, r in enumerate(radii):
            try:
                out[i] = float(pot.enclosedMass(float(r)))
            except Exception:
                pass
    return out


# --------------------------------------------------------------------------------------------- #
# Prior sampling + banded rotation-curve rejection (port of stream_agama._build_accept_bands /
# _vcirc_accept_worker, verbatim logic).
# --------------------------------------------------------------------------------------------- #
def _sample_prior_dict(priors, n, rng):
    out = {}
    for k, spec in priors.items():
        kind = spec["type"]
        pp = list(spec["prior_parameters"])
        if kind == "uniform":
            out[k] = rng.uniform(pp[0], pp[1], size=n)
        elif kind == "normal":
            out[k] = rng.normal(pp[0], pp[1], size=n)
        elif kind == "identity":
            out[k] = np.full(n, pp[0], float)
        else:
            raise ValueError(f"unknown prior type {kind}")
    return out


def _band_accept(vc, obs_r, obs_vc, obs_sig, bands):
    """True if this model curve passes every rejection band."""
    for b in bands:
        rmin = float(b.get("r_min_kpc", -np.inf))
        rmax = float(b.get("r_max_kpc", np.inf))
        m = (obs_r > rmin) & (obs_r <= rmax)
        if not m.any():
            continue
        vm, vo, so = vc[m], obs_vc[m], obs_sig[m]
        if not np.isfinite(vm).all():
            return False
        crit = b.get("criterion", "fracdev")
        stat = b.get("stat", "median")
        if crit == "fracdev":
            dev = np.abs(vm - vo) / vo
            thr = float(b["max_frac_dev"])
        elif crit == "sigma":
            dev = np.abs(vm - vo) / so
            thr = float(b["max_n_sigma"])
        else:
            raise ValueError(f"unknown criterion {crit}")
        s = np.median(dev) if stat == "median" else np.max(dev)
        if not (s < thr):
            return False
    return True


def sample_accepted_prior(priors, bands, obs_r, obs_vc, obs_sig, n_target, rng, cap=400_000):
    """Rejection-sample n_target accepted prior rows (or as many as fit in `cap` raw draws)."""
    agama = _agama()
    agama.setNumThreads(1)
    key0 = next(iter(priors))
    accepted = {k: [] for k in priors}
    n_raw = n_seen = 0
    batch = max(2000, n_target)
    while len(accepted[key0]) < n_target and n_raw < cap:
        draws = _sample_prior_dict(priors, batch, rng)
        n_raw += batch
        for i in range(batch):
            p = {k: float(draws[k][i]) for k in priors}
            try:
                vc = _vcirc(_host_potential(agama, p), obs_r)
            except Exception:
                continue
            n_seen += 1
            if bands and not _band_accept(vc, obs_r, obs_vc, obs_sig, bands):
                continue
            for k in priors:
                accepted[k].append(p[k])
            if len(accepted[next(iter(priors))]) >= n_target:
                break
    out = {k: np.asarray(v) for k, v in accepted.items()}
    n_acc = len(out[next(iter(out))])
    return out, n_raw, n_acc


def _load_post(path):
    d = np.load(path, allow_pickle=True)
    return {k: np.asarray(d[k]).reshape(np.asarray(d[k]).shape[0], -1) for k in d.files}


def _menc_stack(param_rows, radii):
    """param_rows: list of dicts -> (len, len(radii)) enclosed masses."""
    agama = _agama()
    agama.setNumThreads(1)
    out = np.full((len(param_rows), len(radii)), np.nan)
    for i, p in enumerate(param_rows):
        try:
            out[i] = _menc(_host_potential(agama, p), radii)
        except Exception:
            pass
    return out


def _rows_from_post(post, group, idx):
    return [{k: float(post[k][group, i]) for k in GLOBAL_KEYS} for i in idx]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real-run", required=True, help="evaluate_real composition=global run dir")
    ap.add_argument("--sim-run", default=None,
                    help="evaluate (simulated) run dir with compositional_posterior.npz")
    ap.add_argument("--sim-truth", default=None,
                    help="the simulation_multistream_*.npz test set (ground-truth globals)")
    ap.add_argument("--n-prior", type=int, default=4000, help="accepted prior draws for pushforward")
    ap.add_argument("--n-post", type=int, default=1000, help="posterior draws reused per group")
    ap.add_argument("--pivot-kpc", type=float, default=20.0, help="pivot radius for the M_enc panels")
    ap.add_argument("--no-reject", action="store_true", help="skip the vcirc rejection on the prior")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from omegaconf import OmegaConf

    cfg = OmegaConf.load(os.path.join(args.real_run, ".hydra", "config.yaml"))
    params = cfg.simulator.params
    priors = {k: OmegaConf.to_container(v) for k, v in params.priors_global.items()}
    split = float(params.get("obs_r_split_kpc", float(OBS_R_KPC.max())))
    extended = str(params.get("obs_r_grid", "")) == "extended"
    obs_r, obs_vc, obs_sig = (extended_rotation_curve(split) if extended
                              else (OBS_R_KPC, OBS_VC_KMS, OBS_SIGMA_VC))
    vr = params.get("vcirc_rejection")
    bands = None if (args.no_reject or vr is None) else OmegaConf.to_container(vr).get("bands")

    radii = np.array([15.0, args.pivot_kpc, 30.0, 50.0, 100.0])
    pivot_i = int(np.where(radii == args.pivot_kpc)[0][0])
    rng = np.random.default_rng(args.seed)

    print("=" * 90)
    print("MASS / CONCENTRATION PRIOR-vs-POSTERIOR DIAGNOSTIC")
    print("=" * 90)
    print(f"prior a_NFW range: {priors['a_TwoPowerTriaxial_halo']['prior_parameters']} kpc  |  "
          f"beta range: {priors['beta_TwoPowerTriaxial_halo']['prior_parameters']}")
    print(f"rejection: {'ON (' + str(len(bands)) + ' bands)' if bands else 'OFF'}  |  "
          f"grid: {'Zhou u Huang' if extended else 'Zhou'} ({obs_r.size} radii)")
    print(f"pivot radius for M_enc: {args.pivot_kpc} kpc")

    # --- Prior pushforward --------------------------------------------------------------------
    print(f"\n[1/3] rejection-sampling ~{args.n_prior} accepted prior draws ...")
    prior, n_raw, n_acc = sample_accepted_prior(
        priors, bands, obs_r, obs_vc, obs_sig, args.n_prior, rng)
    if bands:
        print(f"      accepted {n_acc} / {n_raw} raw draws ({n_acc / max(n_raw,1):.1%})")
    prior_rows = [{k: float(prior[k][i]) for k in GLOBAL_KEYS} for i in range(n_acc)]
    prior_menc = _menc_stack(prior_rows, radii)
    prior_a = prior["a_TwoPowerTriaxial_halo"]
    prior_rho = prior["rho_TwoPowerTriaxial_halo"]

    # --- Real posterior (Combined + per-stream) -----------------------------------------------
    print("[2/3] loading real posterior draws (reused, no re-sampling) ...")
    gpost = _load_post(os.path.join(args.real_run, "posterior.npz"))
    spost = _load_post(os.path.join(args.real_run, "single_stream_posterior.npz"))
    target = {str(k): int(v) for k, v in OmegaConf.to_container(params.target_streams).items()}
    idx_to_name = {v: k for k, v in target.items()}
    ndraw = gpost[GLOBAL_KEYS[0]].shape[1]
    npost = min(args.n_post, ndraw)
    pidx = rng.choice(ndraw, size=npost, replace=False)

    post_groups = [("Combined", _rows_from_post(gpost, 0, pidx))]
    for gi in range(spost[GLOBAL_KEYS[0]].shape[0]):
        post_groups.append((idx_to_name.get(gi, f"stream{gi}"), _rows_from_post(spost, gi, pidx)))
    post_menc = {name: _menc_stack(rows, radii) for name, rows in post_groups}
    post_a = {name: np.array([r["a_TwoPowerTriaxial_halo"] for r in rows])
              for name, rows in post_groups}
    post_rho = {name: np.array([r["rho_TwoPowerTriaxial_halo"] for r in rows])
                for name, rows in post_groups}

    def _fmt(x):
        return f"{np.nanmedian(x):.3g} [{np.nanpercentile(x,16):.3g}, {np.nanpercentile(x,84):.3g}]"

    print("\n  M_enc(%.0f kpc) [Msun]   (median [16,84])" % args.pivot_kpc)
    print(f"    prior            {_fmt(prior_menc[:, pivot_i]):>34s}")
    for name, _ in post_groups:
        print(f"    {name:14s}   {_fmt(post_menc[name][:, pivot_i]):>34s}")
    print("\n  a_NFW [kpc]")
    print(f"    prior            {_fmt(prior_a):>34s}")
    for name, _ in post_groups:
        print(f"    {name:14s}   {_fmt(post_a[name]):>34s}")

    a_lo, a_hi = priors["a_TwoPowerTriaxial_halo"]["prior_parameters"]
    comb_a_med = np.nanmedian(post_a["Combined"])
    frac_from_lo = (comb_a_med - a_lo) / (a_hi - a_lo)
    print(f"\n  >> Combined a_NFW median sits at {frac_from_lo:.0%} of the prior [{a_lo},{a_hi}] range.")
    print("     If ~0% and the a_NFW posterior width ~ prior width => a_NFW is UNCONSTRAINED and")
    print("     'low a' is a degeneracy/prior artifact (read M_enc instead). If well inside and")
    print("     much tighter than prior => genuinely data-driven.")
    print(f"\n  WARNING: beta in {priors['beta_TwoPowerTriaxial_halo']['prior_parameters']} => "
          "total mass DIVERGES for beta<=3; M_enc(100kpc) shown is finite-radius only, not M_tot.")

    # --- Simulated recovery bias --------------------------------------------------------------
    sim_truth_menc = sim_rec_menc = None
    if args.sim_run and args.sim_truth:
        print("\n[3/3] simulated recovery of M_enc (bias check on well-specified data) ...")
        cpost = _load_post(os.path.join(args.sim_run, "compositional_posterior.npz"))
        truth = np.load(args.sim_truth, allow_pickle=True)
        ntr = truth[GLOBAL_KEYS[0]].shape[0]
        ng = cpost[GLOBAL_KEYS[0]].shape[0]
        n = min(ntr, ng)
        truth_rows = [{k: float(np.asarray(truth[k]).reshape(ntr, -1)[i, 0]) for k in GLOBAL_KEYS}
                      for i in range(n)]
        rec_rows = [{k: float(np.median(cpost[k][i])) for k in GLOBAL_KEYS} for i in range(n)]
        sim_truth_menc = _menc_stack(truth_rows, radii)[:, pivot_i]
        sim_rec_menc = _menc_stack(rec_rows, radii)[:, pivot_i]
        good = np.isfinite(sim_truth_menc) & np.isfinite(sim_rec_menc)
        resid = np.log10(sim_rec_menc[good]) - np.log10(sim_truth_menc[good])
        print(f"      {good.sum()} groups | median log10(rec/truth) M_enc({args.pivot_kpc:.0f}) = "
              f"{np.median(resid):+.3f} dex  (0 = unbiased)")
        if n != ntr or ntr != ng:
            print(f"      note: truth={ntr}, posterior groups={ng}; paired first {n} "
                  "(distribution comparison is order-robust; per-pair may be approximate)")
        print("      If ~0 dex => NO low-mass bias on simulated data => symptom is real-data only "
              "(misspecification / degeneracy), not a training/prior bug.")

    # --- Plot ---------------------------------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"Combined": "k", "Pal5": "#d7191c", "NGC3201": "#2c7bb6", "M68": "#fdae61"}
    ncol = 3 if (sim_truth_menc is not None) else 2
    fig, axes = plt.subplots(1, ncol, figsize=(5.0 * ncol, 4.4))

    # Panel A: M_enc(pivot) prior pushforward vs posterior (log10 histogram)
    axA = axes[0]
    lm_prior = np.log10(prior_menc[:, pivot_i][np.isfinite(prior_menc[:, pivot_i])])
    axA.hist(lm_prior, bins=40, density=True, color="0.7", alpha=0.7, label="prior (accepted)")
    for name, _ in post_groups:
        v = post_menc[name][:, pivot_i]
        v = np.log10(v[np.isfinite(v)])
        if v.size:
            axA.hist(v, bins=40, density=True, histtype="step", lw=2,
                     color=colors.get(name, "purple"), label=name)
    axA.set_xlabel(rf"$\log_{{10}} M_\mathrm{{enc}}(<{args.pivot_kpc:.0f}\,\mathrm{{kpc}})\ [M_\odot]$")
    axA.set_ylabel("density")
    axA.set_title("A. Enclosed-mass: prior vs posterior")
    axA.legend(fontsize=7)

    # Panel B: (a_NFW, M_enc) degeneracy plane
    axB = axes[1]
    axB.scatter(prior_a, np.log10(prior_menc[:, pivot_i]), s=4, color="0.75", alpha=0.4,
                label="prior", rasterized=True)
    for name, _ in post_groups:
        axB.scatter(post_a[name], np.log10(post_menc[name][:, pivot_i]), s=5, alpha=0.5,
                    color=colors.get(name, "purple"), label=name, rasterized=True)
    axB.axvspan(a_lo, a_lo + 0.02 * (a_hi - a_lo), color="red", alpha=0.15)
    axB.set_xlabel(r"$a_\mathrm{NFW}$ [kpc]")
    axB.set_ylabel(rf"$\log_{{10}} M_\mathrm{{enc}}(<{args.pivot_kpc:.0f}\,\mathrm{{kpc}})$")
    axB.set_title("B. a-NFW / M_enc degeneracy")
    axB.legend(fontsize=7)

    # Panel C: simulated recovery bias
    if sim_truth_menc is not None:
        axC = axes[2]
        good = np.isfinite(sim_truth_menc) & np.isfinite(sim_rec_menc)
        lt, lr = np.log10(sim_truth_menc[good]), np.log10(sim_rec_menc[good])
        axC.scatter(lt, lr, s=8, alpha=0.5, color="#2c7bb6")
        lim = [min(lt.min(), lr.min()), max(lt.max(), lr.max())]
        axC.plot(lim, lim, "k--", lw=1, label="unbiased")
        axC.set_xlabel(rf"truth $\log_{{10}} M_\mathrm{{enc}}(<{args.pivot_kpc:.0f})$")
        axC.set_ylabel("recovered (posterior median)")
        axC.set_title(f"C. Sim recovery (bias {np.median(lr-lt):+.3f} dex)")
        axC.legend(fontsize=7)

    fig.suptitle("Milky Way mass/concentration diagnostic — prior pushforward vs posterior "
                 "(no re-sampling)", y=1.02)
    fig.tight_layout()
    out = args.out or os.path.join(args.real_run, "diagnose_mass_prior.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()

"""One-at-a-time sensitivity of the gala MW22 streams to each global potential parameter.

Around a MilkyWayPotential2022-like fiducial, each inferred global of `stream_gala_spray_mw22` is
moved to its prior 16th and 84th percentile with everything else fixed (same spray seed), and two
things are measured at the STARS' positions:

  A. the potential itself — median relative change of the acceleration |da|/|a| and of v_c(R) at the
     fiducial stream stars' 3-D positions (in-window stars);
  B. the observables — shift of the binned median track (phi2, mu_phi1, mu_phi2, v_los) in the
     real-fitted stream frame, evaluated in phi1 bins over the REAL members' footprint, in units of
     the real per-bin dispersion (the noise yardstick the network sees).

Outputs: <out>/sensitivity.json, <out>/sensitivity_tracks.png, <out>/sensitivity_field.png and a
printed ranking. gala is imported in-process (CPU only)."""
from __future__ import annotations
import argparse
import json
import os
import sys
import warnings
os.environ.setdefault("HYDRABFLOW_NUM_GPUS", "0"); os.environ.setdefault("JAX_PLATFORMS", "cpu")
warnings.simplefilter("ignore")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ppc_summary_statistics import NAMES, WINDOW, fit_frame, project
from ppc_particle_coverage import real_clouds
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

FID = dict(log10_M200_halo=11.98, c200_halo=13.3, q_rho_halo=1.0, m_disk=4.7717e10, h_R_disk=2.6, h_z_disk=0.3,
           m_bulge=5.0e9, c_bulge=1.0, m_nucleus=1.8142e9, c_nucleus=0.0688867)
PRIOR = dict(log10_M200_halo=("u", 11.6, 12.4), c200_halo=("u", 8.0, 20.0), q_rho_halo=("u", 0.5, 1.5),
             m_disk=("u", 3.0e10, 7.0e10), h_R_disk=("n", 2.6, 0.5), h_z_disk=("n", 0.3, 0.05))
LOCAL = {  # observed present-day phase space (prior means), mid-prior initial mass, t_end = 4 Gyr
    "Pal5": dict(ra=229.022, dec=-0.112, vr=-58.6, r=20.6, mu_ra_cosdec=-2.736, mu_dec=-2.646, m_progenitor=4.2e4, m_progenitor_final=1.34e4, a_progenitor=30.0, t_end=4.0),
    "NGC3201": dict(ra=154.403, dec=-46.412, vr=494.34, r=4.9, mu_ra_cosdec=8.324, mu_dec=-1.991, m_progenitor=2.25e5, m_progenitor_final=1.93e5, a_progenitor=30.0, t_end=4.0),
    "M68": dict(ra=189.867, dec=-26.744, vr=-92.99, r=10.3, mu_ra_cosdec=-2.752, mu_dec=1.762, m_progenitor=3.4e5, m_progenitor_final=1.28e5, a_progenitor=30.0, t_end=4.0),
}
OBS = ["phi2", "mu_phi1", "mu_phi2", "v_los"]; UNITS = ["deg", "mas/yr", "mas/yr", "km/s"]


def pct(spec, p):
    kind, a, b = spec
    if kind == "u":
        return a + p * (b - a)
    from scipy.stats import norm
    return a + b * norm.ppf(p)


def variants():
    out = [("fiducial", None, FID)]
    for k, spec in PRIOR.items():
        for tag, p in (("lo", 0.16), ("hi", 0.84)):
            th = dict(FID); th[k] = float(pct(spec, p)); out.append((f"{k}:{tag}", k, th))
    return out


def run_one(name, th, seed, n_particles, n_steps):
    from hydrabflow.simulators.stream_gala import _simulate_one_gala, RHO_CRIT_PLANCK18_MSUN_KPC3
    p = dict(th); p.update(LOCAL[name])
    opts = dict(n_steps=n_steps, n_particles_per_release=1, q_ref_r_kpc=15.0, c_phi_bracket=(0.6, 1.6),
                rho_c=RHO_CRIT_PLANCK18_MSUN_KPC3, mass_loss="linear", integrator="leapfrog", raise_errors=True)
    xv, vc, _, der, _, _ = _simulate_one_gala(p, n_particles, np.array([8.178]), seed, opts)
    return xv, float(vc[0]), der


def field_at(th, X, obs_r):
    """Acceleration |a| at 3-D positions X (N,3) and v_c at the stars' cylindrical R, from gala."""
    import astropy.units as u
    import gala.potential as gp
    from gala.units import galactic
    from hydrabflow.simulators.stream_gala import _build_potential, RHO_CRIT_PLANCK18_MSUN_KPC3
    opts = dict(q_ref_r_kpc=15.0, c_phi_bracket=(0.6, 1.6), rho_c=RHO_CRIT_PLANCK18_MSUN_KPC3)
    pot, _ = _build_potential(gp, u, galactic, th, opts)
    acc = pot.acceleration(X.T * u.kpc).to(u.km / u.s / u.Myr).value  # (3, N)
    R = np.hypot(X[:, 0], X[:, 1]); q = np.zeros((3, len(R))); q[0] = R
    vc = pot.circular_velocity(q * u.kpc).to(u.km / u.s).value
    return acc.T, vc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--n-particles", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=2026); ap.add_argument("--n-jobs", type=int, default=40)
    ap.add_argument("--k-bins", type=int, default=8)
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
    n_steps = args.n_particles // 2 - 1
    from joblib import Parallel, delayed
    V = variants()
    jobs = [(nm, lab, th) for nm in NAMES.values() for (lab, _, th) in V]
    res = Parallel(n_jobs=args.n_jobs)(delayed(run_one)(nm, th, args.seed, args.n_particles, n_steps) for nm, lab, th in jobs)
    sims = {(nm, lab): r for (nm, lab, _), r in zip(jobs, res)}
    from hydrabflow.simulators.stream_common import sky_projection
    real = real_clouds("assets/gaia/gaia_observed_streams_6Dwitherrors_cutNGC3201.npz", "stream_gala_spray_mw22", "stream_real_global", 300, 0)

    report = {"fiducial": FID, "variants": {}, "streams": {}}
    trk = {}  # (stream, param) -> array (2 dirs, 4 obs) of |dtrack| / real dispersion (median over bins)
    fld = {}  # (stream, param) -> (median |da|/|a|, median |dvc|/vc) averaged over lo/hi
    for j, nm in NAMES.items():
        rs, rvm = real[j]; R = fit_frame(rs[:, 0], rs[:, 1])
        r1, r2, rm1, rm2 = project(R, rs[:, 0], rs[:, 1], rs[:, 3], rs[:, 4])
        edges = np.quantile(r1, np.linspace(0, 1, args.k_bins + 1))
        # real per-bin dispersion (noise yardstick): median over bins of the per-bin std
        def binned(x, y, f=np.nanmedian, minc=3):
            out = np.full(len(edges) - 1, np.nan)
            for b in range(len(edges) - 1):
                s = (x >= edges[b]) & (x < edges[b + 1]) if b < len(edges) - 2 else (x >= edges[b]) & (x <= edges[b + 1])
                if s.sum() >= minc: out[b] = f(y[s])
            return out
        rvals = [r2, rm1, rm2, rs[:, 5]]
        rsel = [np.ones(len(rs), bool)] * 3 + [rvm]
        disp = [np.nanmedian(binned(r1[m], v[m], lambda a: np.std(a, ddof=1))) for v, m in zip(rvals, rsel)]

        def track(xv):
            lo_ra, hi_ra, lo_dec, hi_dec = WINDOW[j]
            P = sky_projection(xv[None])[0]
            w = (P[:, 0] >= lo_ra) & (P[:, 0] <= hi_ra) & (P[:, 1] >= lo_dec) & (P[:, 1] <= hi_dec) & np.isfinite(P).all(1)
            P = P[w]; p1, p2, m1, m2 = project(R, P[:, 0], P[:, 1], P[:, 3], P[:, 4])
            return [binned(p1, v) for v in (p2, m1, m2, P[:, 5])], w.sum(), P

        xv0, vc0, der0 = sims[(nm, "fiducial")]
        t0, nwin, P0 = track(xv0)
        X0 = xv0[np.isfinite(xv0).all(1)]
        # in-window fiducial stars for the field test
        # restrict the field test to the fiducial in-window stars
        Pall = sky_projection(X0[None])[0]; lo_ra, hi_ra, lo_dec, hi_dec = WINDOW[j]
        win = (Pall[:, 0] >= lo_ra) & (Pall[:, 0] <= hi_ra) & (Pall[:, 1] >= lo_dec) & (Pall[:, 1] <= hi_dec)
        Xw = X0[win][:, :3]; a0, vcs0 = field_at(FID, Xw, None)
        report["streams"][nm] = dict(real_dispersion=dict(zip(OBS, map(float, disp))), fiducial_in_window=int(nwin),
                                     fiducial_c_phi=der0["c_phi_halo_derived"], params={})
        for k in PRIOR:
            rows = []; frows = []
            for tag in ("lo", "hi"):
                lab = f"{k}:{tag}"; xv, vc, der = sims[(nm, lab)]; th = dict(FID); th[k] = float(pct(PRIOR[k], 0.16 if tag == "lo" else 0.84))
                t, nw, _ = track(xv)
                d = [np.nanmedian(np.abs(t[i] - t0[i])) for i in range(4)]
                rows.append([d[i] / disp[i] for i in range(4)])
                a1, vcs1 = field_at(th, Xw, None)
                frows.append([np.median(np.linalg.norm(a1 - a0, axis=1) / np.linalg.norm(a0, axis=1)), np.median(np.abs(vcs1 - vcs0) / vcs0)])
                report["streams"][nm]["params"].setdefault(k, {})[tag] = dict(value=th[k], in_window=int(nw), c_phi=der.get("c_phi_halo_derived"),
                    track_shift={OBS[i]: float(d[i]) for i in range(4)}, track_shift_over_disp={OBS[i]: float(rows[-1][i]) for i in range(4)},
                    median_rel_dacc=float(frows[-1][0]), median_rel_dvc=float(frows[-1][1]))
            trk[(nm, k)] = np.array(rows); fld[(nm, k)] = np.array(frows).mean(0)
    json.dump(report, open(os.path.join(args.out, "sensitivity.json"), "w"), indent=1, default=float)

    params = list(PRIOR); streams = list(NAMES.values())
    short = {"log10_M200_halo": "log10 M200", "c200_halo": "c200", "q_rho_halo": "q_rho", "m_disk": "m_disk", "h_R_disk": "h_R", "h_z_disk": "h_z"}
    # --- figure B: track shift / real dispersion, mean of lo & hi, one panel per observable
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
    for i, (ob, un) in enumerate(zip(OBS, UNITS)):
        M = np.array([[trk[(s, k)][:, i].mean() for k in params] for s in streams])
        im = axes[i].imshow(M, cmap="Blues", vmin=0, vmax=max(1.0, np.nanmax(M)), aspect="auto")
        for a in range(len(streams)):
            for b in range(len(params)):
                axes[i].text(b, a, f"{M[a, b]:.2f}", ha="center", va="center", fontsize=9, color="k" if M[a, b] < 0.6 * max(1.0, np.nanmax(M)) else "w")
        axes[i].set_xticks(range(len(params))); axes[i].set_xticklabels([short[k] for k in params], rotation=35, ha="right", fontsize=9)
        axes[i].set_yticks(range(len(streams))); axes[i].set_yticklabels(streams, fontsize=9)
        axes[i].set_title(f"{ob} track shift / real per-bin dispersion", fontsize=10)
        fig.colorbar(im, ax=axes[i], fraction=0.04)
    fig.suptitle("One-at-a-time sensitivity: |median track shift| when a global moves to its prior 16th/84th percentile (others at the MW2022 fiducial), "
                 "in units of the real per-bin dispersion (mean of the two directions)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "sensitivity_tracks.png"), dpi=130); plt.close(fig)
    # --- figure A: field
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, (lab, vmax) in enumerate((("median |Δa| / |a| at the in-window stream stars", None), ("median |Δv_c| / v_c at the stars' R", None))):
        M = np.array([[fld[(s, k)][i] for k in params] for s in streams])
        im = axes[i].imshow(M, cmap="Blues", vmin=0, aspect="auto")
        for a in range(len(streams)):
            for b in range(len(params)):
                axes[i].text(b, a, f"{100*M[a, b]:.1f}%", ha="center", va="center", fontsize=9, color="k" if M[a, b] < 0.6 * M.max() else "w")
        axes[i].set_xticks(range(len(params))); axes[i].set_xticklabels([short[k] for k in params], rotation=35, ha="right", fontsize=9)
        axes[i].set_yticks(range(len(streams))); axes[i].set_yticklabels(streams, fontsize=9); axes[i].set_title(lab, fontsize=10)
        fig.colorbar(im, ax=axes[i], fraction=0.04)
    fig.suptitle("Potential-level sensitivity at the stars' positions (prior 16th/84th pct moves, mean of both directions)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "sensitivity_field.png"), dpi=130); plt.close(fig)
    # --- printed ranking
    print(f"{'stream':8s} {'param':14s} {'|da|/|a|':>9s} {'|dvc|/vc':>9s} | " + " ".join(f"{o:>9s}" for o in OBS) + "   (track shift / real dispersion; mean lo/hi)")
    for s in streams:
        order = sorted(params, key=lambda k: -trk[(s, k)].mean())
        for k in order:
            print(f"{s:8s} {short[k]:14s} {100*fld[(s,k)][0]:8.1f}% {100*fld[(s,k)][1]:8.1f}% | " + " ".join(f"{trk[(s,k)][:, i].mean():9.2f}" for i in range(4)))
    print("real per-bin dispersion:", {s: {o: round(v, 3) for o, v in report['streams'][s]['real_dispersion'].items()} for s in streams})


if __name__ == "__main__":
    main()

"""Enclosed mass M(<r) of the full Milky Way potential for prior vs posterior draws (referee point 4).

For each group (prior + any number of posterior npz files, ``label=<file.npz>[:row]``; ``row`` picks one member of a
``single_stream_posterior.npz``, default 0) every draw's potential
is rebuilt with the simulator's ``_host_potential`` and M(<r) is evaluated on a log radius grid, plus the
total M200/R200 (same solve as ``posterior_total_m200.py``). Reports the contraction
``1 - var_post/var_prior`` of log10 M(<r) per radius: where the data constrain the mass and where the
M200 headline is extrapolation. Prior draws come straight from the run's ``priors_global`` spec.

Prior sensitivity: the pooled posterior is importance-reweighted to alternative log10_M200 priors on the
same support (uniform in M200 instead of log-uniform; a Gaussian in log10 M200 at each end of the range),
and the induced shift of M(<20 kpc) vs M200 is tabulated.

    python scripts/enclosed_mass_profile.py --runs pooled=<run>/posterior.npz \
        curve=<run_only_vcirc_kms>/single_stream_posterior.npz:0 Pal5=<run_only_sim_summary>/...:0 ...
"""

import argparse
import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(__file__))
from posterior_total_m200 import r200_of  # noqa: E402
from ppc_rotation_curve_ibata import (  # noqa: E402
    _build_pot_cfg, _identity_constants, _load_posterior, _log10_keys,
)

M200_KEY = "log10_M200_TwoPowerTriaxial_halo"
R_GRID = np.geomspace(2.0, 300.0, 60)
R_REPORT = [8.0, 15.0, 20.0, 25.0, 30.0, 50.0, 100.0]


def sample_prior(priors_global, n, rng):
    out = {}
    for name, spec in priors_global.items():
        t, p = str(spec["type"]), spec["prior_parameters"]
        if t == "uniform":
            out[name] = rng.uniform(p[0], p[1], n)
        elif t == "normal":
            out[name] = rng.normal(p[0], p[1], n)
    return out


def _profile(rows, pot_cfg, rho_t):
    from hydrabflow.simulators.stream_agama import _agama, _host_potential
    agama = _agama()
    out = []
    for row in rows:
        try:
            pot = _host_potential(agama, row, pot_cfg)
            m = np.asarray(pot.enclosedMass(R_GRID), float)
            r200, m200 = r200_of(pot, rho_t)
        except Exception:  # noqa: BLE001  (gamma >= 2 leaks give a non-positive scale radius)
            m, r200, m200 = np.full(len(R_GRID), np.nan), np.nan, np.nan
        out.append((m, r200, m200))
    return out


def profiles(draws, consts, pot_cfg, rho_t, n_jobs):
    n = len(next(iter(draws.values())))
    rows = [{**consts, **{k: float(v[i]) for k, v in draws.items()}} for i in range(n)]
    chunks = np.array_split(np.arange(n), max(1, n_jobs * 4))
    res = Parallel(n_jobs=n_jobs)(delayed(_profile)([rows[i] for i in c], pot_cfg, rho_t)
                                  for c in chunks if len(c))
    res = [r for part in res for r in part]
    return (np.stack([r[0] for r in res]), np.array([r[1] for r in res]),
            np.array([r[2] for r in res]))


def wquant(x, w, qs):
    ok = np.isfinite(x)
    x, w = x[ok], w[ok]
    i = np.argsort(x)
    c = np.cumsum(w[i]) / w.sum()
    return np.interp(qs, c, x[i])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True, help="label=npz[:row]; the first is reweighted")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-draws", type=int, default=4000)
    ap.add_argument("--n-jobs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from omegaconf import OmegaConf

    from hydrabflow.simulators.stream_agama import _agama, _resolve_pot_cfg

    runs = dict(s.split("=", 1) for s in args.runs)
    cfg = OmegaConf.load(os.path.join(os.path.dirname(next(iter(runs.values())).split(":")[0]),
                                      ".hydra", "config.yaml"))
    params = OmegaConf.to_container(cfg.simulator.params, resolve=True)
    pot_cfg = _resolve_pot_cfg(_build_pot_cfg(params))
    consts = _identity_constants(params["priors_global"])
    log10 = _log10_keys(OmegaConf.to_container(cfg, resolve=True))
    agama = _agama()
    H0 = float(pot_cfg["halo_H0_kms_mpc"]) / 1000.0
    rho_t = float(pot_cfg["halo_Delta_mass"]) * 3.0 * H0**2 / (8.0 * np.pi * agama.G)
    r_obs = np.asarray(params.get("obs_r_kpc") or [], float)

    rng = np.random.default_rng(args.seed)
    groups = {"prior": sample_prior(params["priors_global"], args.n_draws, rng)}
    for label, spec in runs.items():
        path, _, row = spec.partition(":")
        post = _load_posterior(path, log10)
        n = post[M200_KEY].shape[1]
        idx = rng.choice(n, min(n, args.n_draws), replace=False)
        groups[label] = {k: v[int(row or 0), idx] for k, v in post.items()}

    res = {}
    for label, draws in groups.items():
        cache = os.path.join(args.out, f"enclosed_mass_{label}.npz")
        if os.path.exists(cache):
            z = np.load(cache)
            res[label] = {k: z[k] for k in z.files}
            continue
        m, r200, m200 = profiles(draws, consts, pot_cfg, rho_t, args.n_jobs)
        res[label] = {"M_enc": m, "R200_total": r200, "M200_total": m200,
                      "log10_M200_halo": np.asarray(draws[M200_KEY], float)}
        np.savez(cache, r_grid=R_GRID, **res[label])
        print(f"{label}: {len(m)} draws, {np.isnan(m200).sum()} failed")

    lg = {k: np.log10(v["M_enc"]) for k, v in res.items()}
    var_pr = np.nanvar(lg["prior"], axis=0)
    summary = {"r_grid": R_GRID.tolist(), "groups": {}}
    for label in res:
        q = np.nanpercentile(lg[label], [16, 50, 84], axis=0)
        contr = 1.0 - np.nanvar(lg[label], axis=0) / var_pr
        lm200 = np.log10(res[label]["M200_total"])
        p = np.nanpercentile(lm200, [16, 50, 84])
        summary["groups"][label] = {
            "log10_M_enc_p16_p50_p84": {
                f"{r:g}": [float(np.interp(r, R_GRID, q[i])) for i in range(3)] for r in R_REPORT},
            "contraction": {f"{r:g}": float(np.interp(r, R_GRID, contr)) for r in R_REPORT},
            "log10_M200_total_p16_p50_p84": p.tolist(),
            "contraction_M200_total": float(1 - np.nanvar(lm200) / np.nanvar(
                np.log10(res["prior"]["M200_total"]))),
            "R200_total_median": float(np.nanmedian(res[label]["R200_total"])),
        }

    # prior sensitivity on the pooled posterior: reweight from the log-uniform M200 prior
    lo, hi = params["priors_global"][M200_KEY]["prior_parameters"]
    x = res[next(iter(runs))]["log10_M200_halo"]
    alts = {"log-uniform (fiducial)": np.ones_like(x),
            "uniform in M200": 10.0 ** (x - hi),
            f"N({lo + 0.1:.1f}, 0.15)": np.exp(-0.5 * ((x - lo - 0.1) / 0.15) ** 2),
            f"N({hi - 0.1:.1f}, 0.15)": np.exp(-0.5 * ((x - hi + 0.1) / 0.15) ** 2)}
    lab0 = next(iter(runs))
    m20 = np.log10([np.interp(20.0, R_GRID, row) for row in res[lab0]["M_enc"]])
    lm200 = np.log10(res[lab0]["M200_total"])
    sens = {}
    for name, w in alts.items():
        w = w / w.sum()
        sens[name] = {"ess": float(1 / np.sum(w**2)),
                      "log10_M_enc_20kpc": wquant(m20, w, [0.16, 0.5, 0.84]).tolist(),
                      "log10_M200_total": wquant(lm200, w, [0.16, 0.5, 0.84]).tolist()}
    summary["prior_sensitivity_" + lab0] = sens
    with open(os.path.join(args.out, "enclosed_mass_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif"})
    stream_c = dict(zip(["Pal5", "NGC3201", "M68"], plt.cm.RdYlBu_r(np.linspace(0, 1, 4))[1:]))
    colors = {"prior": "0.6", "pooled": "k", "curve": "tab:green", **stream_c}
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.2, 7.0), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1.2], "hspace": 0.05})
    for label in res:
        q = np.nanpercentile(lg[label], [16, 50, 84], axis=0)
        c = colors.get(label, "k")
        a1.fill_between(R_GRID, 10 ** q[0], 10 ** q[2], color=c, alpha=0.25 if label != "prior" else 0.15)
        a1.plot(R_GRID, 10 ** q[1], color=c, lw=1.4, label=label)
        a2.plot(R_GRID, (q[2] - q[0]) / 2, color=c, lw=1.6 if label != "prior" else 1.2,
                ls="-" if label != "prior" else "--")
        g = summary["groups"][label]
        lm = g["log10_M200_total_p16_p50_p84"]
        a2.plot([g["R200_total_median"]], [(lm[2] - lm[0]) / 2], "o", color=c, ms=5)
    for ax in (a1, a2):
        if len(r_obs):
            ax.axvspan(r_obs.min(), r_obs.max(), color="0.85", zorder=0, lw=0)
        ax.set_xscale("log")
    a1.set_yscale("log")
    a1.set_ylabel(r"$M(<r)\ [M_\odot]$")
    a1.legend(frameon=False, fontsize=9)
    a2.set_yscale("log")
    a2.set_ylabel(r"68 % half-width of" + "\n" + r"$\log_{10} M(<r)$ [dex]")
    a2.set_xlabel(r"$r$ [kpc]")
    a2.text(0.02, 0.92, r"dots: total $M_{200}$ at its $R_{200}$; dashed: prior", transform=a2.transAxes,
            ha="left", va="top", fontsize=8)
    fig.savefig(os.path.join(args.out, "enclosed_mass_profile.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(args.out, "enclosed_mass_profile.png"), dpi=200, bbox_inches="tight")
    print(json.dumps(summary["groups"], indent=1))
    print(json.dumps(sens, indent=1))


if __name__ == "__main__":
    main()

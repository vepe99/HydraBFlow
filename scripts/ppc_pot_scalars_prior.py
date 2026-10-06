"""Prior-predictive check of ``pot_scalars`` = [K_z, mu_l, M200/1e12] (scripts/add_pot_scalars.py).

1. Per scalar: the noise-convolved prior predictive (sim + N(0, err), as in training) vs the observed
   value; P(sim < obs), fraction of rows within 1 sigma, Spearman with every varied global.
2. Pairwise scatter of the three with the observed point.
3. Importance weights w_i = N(obs | sim_i, err) of every prior row for the scalars alone, the rotation
   curve alone (sigma = the simulator config's obs_sigma_vc, e.g. sys3) and both -- the exact
   likelihood-weighted posterior over the prior sample. ESS of each, and the weighted global
   parameters side by side: if the joint ESS collapses far below both single ones, the curve and the
   scalars want different Galaxies (tension); if it follows the smaller one, they agree.

Usage: python scripts/ppc_pot_scalars_prior.py --sim TRAIN.npz --simulator <sim cfg> --out DIR
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

os.environ.setdefault("HYDRABFLOW_NUM_GPUS", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

NAMES = ["K_z [km2 s-2 pc-1]", "mu_l [mas/yr]", "M200 [1e12 Msun]"]
OBS = np.array([2.00, -6.379, 1.17])
ERR = np.array([0.16, 0.026, 0.21])


def _ess(logw):
    w = np.exp(logw - logw.max())
    return float(w.sum() ** 2 / (w**2).sum()), w / w.sum()


def _wq(v, w, q):
    o = np.argsort(v)
    c = np.cumsum(w[o])
    return np.interp(q, c / c[-1], v[o])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True)
    ap.add_argument("--simulator", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--r-min", type=float, default=5.5, help="curve radii kept (mask_vcirc_radii)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from hydra import compose, initialize_config_dir
    from scipy.stats import spearmanr

    from hydrabflow.config import register_configs

    register_configs()
    with initialize_config_dir(config_dir=os.path.abspath("conf"), version_base=None):
        P = compose("config", overrides=[f"simulator={args.simulator}"]).simulator.params
    r = np.asarray(P.obs_r_kpc, float)
    keep = r >= args.r_min
    vobs, vsig = np.asarray(P.obs_vc_kms, float)[keep], np.asarray(P.obs_sigma_vc, float)[keep]

    ds = np.load(args.sim)
    s = np.asarray(ds["pot_scalars"], float)
    vc = np.asarray(ds["vcirc_kms"], float).reshape(len(s), -1)[:, keep]
    ok = np.isfinite(s).all(1) & np.isfinite(vc).all(1)
    s, vc = s[ok], vc[ok]
    n = len(s)
    globs = {}
    for k in ds.files:
        if k in ("sim_data_projected", "j") or k.endswith("_derived"):
            continue
        a = np.asarray(ds[k])
        if a.shape == (len(ok), 1) and np.ptp(a) > 0:
            globs[k] = a[ok, 0]

    noisy = s + np.random.default_rng(args.seed).normal(size=s.shape) * ERR
    rep = {"n_rows": n, "obs": OBS.tolist(), "err": ERR.tolist(), "scalars": {}}
    for j, name in enumerate(NAMES):
        rep["scalars"][name] = {
            "prior_pred_p5_p50_p95": np.percentile(noisy[:, j], [5, 50, 95]).round(4).tolist(),
            "P_sim_below_obs": round(float((noisy[:, j] < OBS[j]).mean()), 4),
            "frac_noise_free_within_1sigma": round(float((np.abs(s[:, j] - OBS[j]) < ERR[j]).mean()), 4),
            "spearman_vs_param": {k: round(float(spearmanr(v, s[:, j]).statistic), 3) for k, v in globs.items()},
        }

    logl_s = -0.5 * (((s - OBS) / ERR) ** 2).sum(1)
    logl_c = -0.5 * (((vc - vobs) / vsig) ** 2).sum(1)
    sets = {"scalars": logl_s, "curve": logl_c, "curve+scalars": logl_s + logl_c}
    for j, name in enumerate(NAMES):  # each scalar on its own: which one is the bottleneck
        sets[name.split()[0]] = -0.5 * ((s[:, j] - OBS[j]) / ERR[j]) ** 2
    W = {}
    rep["ess"] = {}
    for key, lw in sets.items():
        e, W[key] = _ess(lw)
        rep["ess"][key] = round(e, 1)
    rep["weighted_params"] = {
        k: {"prior": _wq(v, np.full(n, 1 / n), [0.16, 0.5, 0.84]).round(4).tolist(),
            **{key: _wq(v, W[key], [0.16, 0.5, 0.84]).round(4).tolist()
               for key in ("curve", "scalars", "curve+scalars")}}
        for k, v in globs.items()}
    # each data set's prediction of the other: does the curve-weighted prior predict the scalars?
    rep["curve_weighted_scalars_p16_50_84"] = {
        name: _wq(s[:, j], W["curve"], [0.16, 0.5, 0.84]).round(4).tolist() for j, name in enumerate(NAMES)}
    json.dump(rep, open(os.path.join(args.out, "pot_scalars_prior.json"), "w"), indent=1)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    for j, name in enumerate(NAMES):
        ax = axs[0, j]
        b = np.linspace(*np.percentile(noisy[:, j], [0.2, 99.8]), 80)
        ax.hist(noisy[:, j], bins=b, density=True, color="0.7", label="prior predictive (+noise)")
        ax.hist(s[:, j], bins=b, density=True, weights=W["curve"], histtype="step", lw=1.6, color="C0",
                label="curve-weighted")
        ax.axvspan(OBS[j] - ERR[j], OBS[j] + ERR[j], color="C3", alpha=0.25)
        ax.axvline(OBS[j], color="C3", label="observed +-1 sigma")
        ax.set_title(f"{name}: P(sim<obs) {rep['scalars'][name]['P_sim_below_obs']:.3f}")
        ax.set_xlabel(name)
    axs[0, 0].legend(fontsize=8)
    sub = np.random.default_rng(1).choice(n, size=min(n, 20000), replace=False)
    for ax, (a, b) in zip(axs[1], [(0, 1), (0, 2), (1, 2)]):
        ax.scatter(noisy[sub, a], noisy[sub, b], s=1, alpha=0.15, color="0.5", rasterized=True)
        ax.errorbar(OBS[a], OBS[b], xerr=ERR[a], yerr=ERR[b], fmt="o", color="C3", ms=6, capsize=3)
        ax.set_xlabel(NAMES[a])
        ax.set_ylabel(NAMES[b])
        ax.set_xlim(np.percentile(noisy[:, a], [0.5, 99.5]))
        ax.set_ylim(np.percentile(noisy[:, b], [0.5, 99.5]))
    fig.suptitle(f"pot_scalars prior predictive ({n} rows); ESS: "
                 + ", ".join(f"{k} {v:.0f}" for k, v in rep["ess"].items()))
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "pot_scalars_prior.png"), dpi=120)

    nc = 4
    nr = -(-len(globs) // nc)
    fig, axs = plt.subplots(nr, nc, figsize=(4 * nc, 3 * nr), squeeze=False)
    for ax, (k, v) in zip(axs.flat, globs.items()):
        b = np.linspace(*np.percentile(v, [0.5, 99.5]), 40)
        ax.hist(v, bins=b, density=True, color="0.8", label="prior")
        for key, c in (("curve", "C0"), ("scalars", "C1"), ("curve+scalars", "C3")):
            ax.hist(v, bins=b, density=True, weights=W[key], histtype="step", lw=1.6, color=c,
                    label=f"{key} (ESS {rep['ess'][key]:.0f})")
        ax.set_title(k, fontsize=9)
    for ax in list(axs.flat)[len(globs):]:
        ax.axis("off")
    axs.flat[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "pot_scalars_weighted_params.png"), dpi=120)
    print(json.dumps({k: rep[k] for k in ("ess", "curve_weighted_scalars_p16_50_84")}, indent=1))
    for name, d in rep["scalars"].items():
        print(name, d["prior_pred_p5_p50_p95"], "P(sim<obs)", d["P_sim_below_obs"])
    print(f"{'param':36s} {'prior':>22s} {'curve':>22s} {'scalars':>22s} {'curve+scalars':>22s}")
    for k, d in rep["weighted_params"].items():
        print(f"{k:36s} " + " ".join(f"{d[c][1]:8.4g} [{d[c][0]:.3g},{d[c][2]:.3g}]".rjust(22)
                                      for c in ("prior", "curve", "scalars", "curve+scalars")))


if __name__ == "__main__":
    main()

"""Is an observed rotation curve out of distribution for a training set? One row per observed curve:
prior band + 10 nearest rows (chi2) + data, and the typicality test: chi2 from the observed curve to its
nearest training row vs the same for training curves with the training noise N(0, sigma) added.

Training curves are cubic-interpolated from the stored ``vcirc_kms`` radii onto each observed grid.
Usage: scripts/rotation_curve_ood.py --sim <npz> --simulator <yaml name> --curve label=csv ... --out <png>
(``label=config`` takes the simulator's own obs_r_kpc / obs_vc_kms / obs_sigma_vc table.)
"""
import argparse

import matplotlib
import numpy as np
import yaml
from scipy.interpolate import CubicSpline

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--sim", required=True)
ap.add_argument("--simulator", required=True)
ap.add_argument("--curve", nargs="+", required=True, help="label=path.csv (R,V_c,e_V_c) or label=config")
ap.add_argument("--n-test", type=int, default=2000)
ap.add_argument("--k", type=int, default=300, help="nearest rows (smallest chi2) whose parameters are reported")
ap.add_argument("--out", required=True)
a = ap.parse_args()

def _params(name):  # follow same-group `defaults:` inheritance (child keys win)
    y = yaml.safe_load(open(f"conf/simulator/{name}.yaml"))
    out = {}
    for d in y.get("defaults") or []:
        if isinstance(d, str) and d != "_self_":
            out.update(_params(d))
    return {**out, **(y.get("params") or {})}


P = _params(a.simulator)
r_sim = np.array(P["obs_r_kpc"], float)
ds = np.load(a.sim)
vc_sim = ds["vcirc_kms"]
vc_sim = vc_sim.reshape(len(vc_sim), -1)
ok = np.isfinite(vc_sim).all(1)
vc_sim = vc_sim[ok]
glob = {k: np.asarray(ds[k]).reshape(-1)[ok] for k, v in P["priors_global"].items()
        if v["type"] != "identity" and k not in (P.get("marginalize") or []) and k in ds.files}
nn_idx = {}
rng = np.random.default_rng(0)
idx = rng.choice(len(vc_sim), a.n_test, replace=False)

fig, axs = plt.subplots(len(a.curve), 2, figsize=(14, 4.6 * len(a.curve)), squeeze=False,
                        gridspec_kw={"width_ratios": [1.6, 1]})
for row, spec in zip(axs, a.curve):
    lab, src = spec.split("=", 1)
    if src == "config":
        R, obs, sig = (np.array(P[k], float) for k in ("obs_r_kpc", "obs_vc_kms", "obs_sigma_vc"))
    else:
        R, obs, sig = np.loadtxt([s for s in open(src) if s[:1].isdigit()], delimiter=",").T
    vc = CubicSpline(r_sim, vc_sim, axis=1)(R)
    chi = (((vc - obs) / sig) ** 2).sum(1)
    z, o = vc / sig, obs / sig
    sq = (z**2).sum(1)
    nn = lambda q: (sq[None] + (q**2).sum(1)[:, None] - 2 * q @ z.T).min(1)  # noqa: E731
    sims = np.concatenate([nn(q) for q in np.array_split(z[idx] + rng.standard_normal(z[idx].shape), 20)])
    real = chi.min()
    nn_idx[lab] = np.argsort(chi)[: a.k]
    print(f"{lab}: k-NN (k={a.k}) parameters, prior median [16,84] -> NN median [16,84], shift / prior sd")
    for k, v in glob.items():
        pq, nq = np.percentile(v, [50, 16, 84]), np.percentile(v[nn_idx[lab]], [50, 16, 84])
        print(f"  {k:34s} {pq[0]:8.4g} [{pq[1]:7.4g},{pq[2]:7.4g}] -> {nq[0]:8.4g} [{nq[1]:7.4g},{nq[2]:7.4g}]"
              f"  {(nq[0] - pq[0]) / v.std():+.2f}")
    pct = 100 * (sims < real).mean()

    ax = row[0]
    band = np.percentile(vc, [5, 16, 50, 84, 95], axis=0)
    ax.fill_between(R, band[0], band[4], color="C0", alpha=0.15, label="prior 5-95 %")
    ax.fill_between(R, band[1], band[3], color="C0", alpha=0.3, label="prior 16-84 %")
    ax.plot(R, band[2], "C0-", label="prior median")
    for i in np.argsort(chi)[:10]:
        ax.plot(R, vc[i], color="C1", alpha=0.6, lw=0.9)
    ax.plot([], [], color="C1", label="10 nearest rows (chi2)")
    ax.errorbar(R, obs, yerr=sig, fmt="ko", ms=4, capsize=2, label=f"{lab} ({len(R)} radii)")
    ax.set(xlabel="R [kpc]", ylabel="v_c [km/s]", title=f"{lab}: best chi2 {real:.1f} / {len(R)} radii")
    ax.legend(fontsize=8, loc="lower left")

    ax = row[1]
    b = np.linspace(0, max(np.percentile(sims, 99.9), real) * 1.1, 60)
    ax.hist(sims, bins=b, color="0.6", label="training curve + training noise -> nearest row")
    ax.axvline(real, color="C3", lw=2, label=f"observed -> nearest row ({pct:.1f} pct)")
    ax.set(xlabel="chi2 to the nearest training row", ylabel="curves",
           title=f"{lab}: {'IN' if pct < 99 else 'OUT OF'} distribution")
    ax.legend(fontsize=8)
    print(f"{lab}: real {real:.1f}, sims median {np.median(sims):.1f} p99 {np.percentile(sims, 99):.1f}, pct {pct:.1f}")
fig.tight_layout()
fig.savefig(a.out, dpi=130)

nc = 4
fig, axs = plt.subplots(-(-len(glob) // nc), nc, figsize=(4 * nc, 3 * -(-len(glob) // nc)), squeeze=False)
for ax, (k, v) in zip(axs.flat, glob.items()):
    b = np.linspace(*np.percentile(v, [0.5, 99.5]), 40)
    ax.hist(v, bins=b, density=True, color="0.75", label="prior")
    for c, (lab, i) in enumerate(nn_idx.items()):
        ax.hist(v[i], bins=b, density=True, histtype="step", lw=1.8, color=f"C{c + 1}",
                label=f"{lab}: {len(i)} NN ({(np.median(v[i]) - np.median(v)) / v.std():+.2f} sd)")
    ax.set_title(k, fontsize=9)
    ax.legend(fontsize=6)
for ax in list(axs.flat)[len(glob):]:
    ax.axis("off")
fig.tight_layout()
fig.savefig(a.out.replace(".png", "_nearest_params.png"), dpi=120)

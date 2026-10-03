"""Rotation-curve typicality vs the systematic error added to the observed curve's errors.

For each error model sigma_eff = sqrt(sigma_stat^2 + floor^2), the observed curve's chi2 to its nearest
noise-free prior curve is compared with the same statistic for prior curves + one N(0, sigma_eff) draw
(the training noise) to their nearest OTHER prior curve. Writes a 3-panel figure: observed curve with
the error bars of each model over the prior band, the null histograms with the observed value, and the
percentile vs floor.

  python scripts/ppc_rotation_curve_sysfloor.py --sim <training npz> --simulator <yaml name> \
      --floors "stat" "1%" "2kms" "3%" --out <png>
"""
import argparse
import json

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def params(name):  # simulator yaml with same-group `defaults:` inheritance
    y = yaml.safe_load(open(f"conf/simulator/{name}.yaml"))
    out = {}
    for d in y.get("defaults") or []:
        if isinstance(d, str) and d != "_self_":
            out.update(params(d))
    return {**out, **(y.get("params") or {})}


ap = argparse.ArgumentParser()
ap.add_argument("--sim", required=True)
ap.add_argument("--simulator", required=True, help="yaml carrying the STATISTICAL errors")
ap.add_argument("--floors", nargs="+", default=["stat", "1%", "2kms", "3%"])
ap.add_argument("--r-min", type=float, default=5.5)
ap.add_argument("--n-rows", type=int, default=300000)
ap.add_argument("--n-test", type=int, default=1000)
ap.add_argument("--out", required=True)
a = ap.parse_args()

P = params(a.simulator)
r, vo, s0 = (np.asarray(P[k], float) for k in ("obs_r_kpc", "obs_vc_kms", "obs_sigma_vc"))
m = r > a.r_min
r, vo, s0 = r[m], vo[m], s0[m]
V = np.load(a.sim, mmap_mode="r")["vcirc_kms"][: a.n_rows, :, 0].astype(float)[:, m]
V = V[np.isfinite(V).all(1)]
rng = np.random.default_rng(0)
test = rng.choice(len(V), a.n_test, replace=False)


def sigma(f):
    if f == "stat":
        return s0
    if f.endswith("%"):
        return np.sqrt(s0 ** 2 + (float(f[:-1]) / 100 * vo) ** 2)
    return np.sqrt(s0 ** 2 + float(f.replace("kms", "")) ** 2)


res = {}
for f in a.floors:
    s = sigma(f)
    real = (((V - vo) / s) ** 2).sum(1).min()
    null = []
    for i in test:
        y = V[i] + rng.normal(0, s)
        c = (((V - y) / s) ** 2).sum(1)
        c[i] = np.inf
        null.append(c.min())
    null = np.asarray(null)
    res[f] = dict(sigma=s, real=float(real), null=null, pct=float(100 * (null <= real).mean()))
    print(f"{f:6s} observed chi2 {real:6.1f}  noisy-sim median {np.median(null):5.1f} [1%,99%] "
          f"{np.percentile(null, 1):5.1f}-{np.percentile(null, 99):5.1f}  -> percentile {res[f]['pct']:5.1f}")

label = {"stat": "quoted (stat. only)"}
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
lo, med, hi = np.percentile(V, [5, 50, 95], axis=0)
q16, q84 = np.percentile(V, [16, 84], axis=0)
ax[0].fill_between(r, lo, hi, color="0.85", label="prior 5-95 %")
ax[0].fill_between(r, q16, q84, color="0.7", label="prior 16-84 %")
ax[0].plot(r, med, color="0.4", lw=1, label="prior median")
cols = plt.cm.viridis(np.linspace(0, 0.85, len(a.floors)))
for k, (f, c) in enumerate(zip(a.floors, cols)):
    ax[0].errorbar(r + 0.06 * (k - len(a.floors) / 2), vo, yerr=res[f]["sigma"], fmt="o", ms=2.5, color=c,
                   lw=0.8, capsize=0, label=f"Eilers19, {label.get(f, '+' + f + ' syst.')}")
ax[0].set(xlabel="R [kpc]", ylabel=r"$v_c$ [km/s]", ylim=(150, 280), title="observed curve vs prior")
ax[0].legend(fontsize=7, loc="lower left")
bins = np.linspace(0, 120, 61)
for f, c in zip(a.floors, cols):
    ax[1].hist(np.clip(res[f]["null"], 0, 120), bins=bins, histtype="step", color=c, lw=1.4,
               label=f"{label.get(f, '+' + f)}: obs pct {res[f]['pct']:.1f}")
    ax[1].axvline(min(res[f]["real"], 119), color=c, ls="--", lw=1.4)
ax[1].set(xlabel=r"$\chi^2$ to nearest prior curve (37 radii)", ylabel="noisy simulated curves",
          title="typicality: dashed = observed Eilers19")
ax[1].legend(fontsize=7)
ax[2].bar(range(len(a.floors)), [res[f]["pct"] for f in a.floors], color=cols)
ax[2].axhspan(5, 95, color="green", alpha=0.08, label="typical (5-95 %)")
ax[2].set_xticks(range(len(a.floors)), [label.get(f, "+" + f) for f in a.floors], fontsize=8)
ax[2].set(ylabel="percentile of observed curve", ylim=(0, 100), title="in distribution?")
ax[2].legend(fontsize=8)
fig.tight_layout()
fig.savefig(a.out, dpi=140)
json.dump({f: {"real_chi2": v["real"], "null_median": float(np.median(v["null"])), "percentile": v["pct"],
               "sigma_kms": v["sigma"].round(3).tolist()} for f, v in res.items()},
          open(a.out.rsplit(".", 1)[0] + ".json", "w"), indent=1)
print("wrote", a.out)

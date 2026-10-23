"""Scoring table + figure for the compositional (pooled) posterior vs every single-input posterior on
the simulated test set: bias, RMSE, 68/95 % coverage, posterior contraction, CRPS per parameter, and
the multivariate energy score. Everything is in the network's native space (log10 for the disk
parameters, as in the truths) and normalized by the prior std (10^6 draws of the run's priors_global, in the
same native space).

Posteriors (rows group-major, j = 0..m-1 inside a group, m = 3):
  <stream> alone          eval_sim_only_sim_summary/base_posterior.npz
  curve alone             eval_sim_only_vcirc_kms/base_posterior.npz (member 0: the curve is per group)
  pooled                  eval_sim/compositional_posterior.npz

    python scripts/score_compositional_gain.py --trial <trial_dir> [--out <trial_dir>/compositional_gain]
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from enclosed_mass_profile import sample_prior  # noqa: E402
from plot_calibration_ecdf import LABEL, RECOVERY_COLOR, STREAM_COLOR, STREAMS, save  # noqa: E402
from ppc_rotation_curve_ibata import _log10_keys  # noqa: E402
from bayesflow.diagnostics.metrics import calibration_error  # noqa: E402

M = len(STREAMS)


def crps(x, y):
    """Sample CRPS per dataset: E|X - y| - E|X - X'|/2, x (n, S), y (n,)."""
    s = np.sort(x, axis=1)
    S = s.shape[1]
    w = 2 * np.arange(1, S + 1) - S - 1
    return np.mean(np.abs(x - y[:, None]), axis=1) - (s * w).sum(axis=1) / S**2


def energy(x, y, rng):
    """Energy score per dataset, x (n, S, d), y (n, d); E||X-X'|| from a random pairing of draws."""
    t1 = np.linalg.norm(x - y[:, None], axis=-1).mean(axis=1)
    x2 = x[:, rng.permutation(x.shape[1])]
    return t1 - 0.5 * np.linalg.norm(x - x2, axis=-1).mean(axis=1)


def scores(post, truth, sd, rng):
    """post {k: (n, S)}, truth {k: (n,)} -> per-parameter metric dict + per-dataset CRPS and ES."""
    out, per = {}, {}
    for k, x in post.items():
        z, y = x / sd[k], truth[k] / sd[k]
        mean = z.mean(axis=1)
        lo68, hi68, lo95, hi95 = np.percentile(z, [16, 84, 2.5, 97.5], axis=1)
        per[k] = crps(z, y)
        raw = x.mean(axis=1) - truth[k]
        out[k] = {"bias_raw": float(np.median(raw)), "rmse_raw": float(np.sqrt(np.mean(raw**2))),
                  "crps_raw": float(crps(x, truth[k]).mean()),
                  "bias": float(np.median(mean - y)), "rmse": float(np.sqrt(np.mean((mean - y) ** 2))),
                  "cov68": float(np.mean((y >= lo68) & (y <= hi68))),
                  "cov95": float(np.mean((y >= lo95) & (y <= hi95))),
                  "contraction": float(np.median(1 - z.var(axis=1))),
                  "cal_err": float(calibration_error(x[..., None], truth[k][:, None])["values"][0]),
                  "crps": float(per[k].mean())}
    keys = list(post)
    es = energy(np.stack([post[k] / sd[k] for k in keys], -1), np.stack([truth[k] / sd[k] for k in keys], -1), rng)
    out["joint"] = {"energy": float(es.mean())}
    return out, per, es


def tarp(post, truth, refs, sd):
    """TARP (Lemos+2023) coverage value per dataset: fraction of draws closer to a random reference
    than the truth, in prior-std units. Calibrated joint posterior <=> uniform on [0, 1]."""
    keys = list(post)
    x = np.stack([post[k] / sd[k] for k in keys], -1)
    y = np.stack([truth[k] / sd[k] for k in keys], -1)
    r = np.stack([refs[k] / sd[k] for k in keys], -1)
    d_x = np.linalg.norm(x - r[:, None], axis=-1)
    return np.mean(d_x < np.linalg.norm(y - r, axis=-1)[:, None], axis=1)


def latex_table(res, names, keys, path):
    """Bias (median over datasets) and RMSE of the posterior mean and mean CRPS, native units; best value per
    parameter in bold (smallest |bias|, RMSE, CRPS, after rounding to the printed precision)."""
    unit = {"r_Disk": " [dex]", "z_Disk": " [dex]", "Sigma_Disk": " [dex]"}
    nc = len(names)
    rows = []
    for met, title, f in (("bias_raw", "Bias", "{:+.3f}"), ("rmse_raw", "RMSE", "{:.3f}"), ("crps_raw", "CRPS", "{:.3f}"),
                          ("cal_err", "Calibration error", "{:.3f}"), ("contraction", "Posterior contraction", "{:.2f}")):
        rows += [r"\hline", rf"\multicolumn{{{nc + 1}}}{{l}}{{\textit{{{title}}}}} \\"]
        for k in keys:
            v = [f.format(res[nm][k][met]) for nm in names]
            score = [-float(t) if met == "contraction" else abs(float(t)) for t in v]  # contraction: higher is better
            cells = [rf"\textbf{{{t}}}" if sc == min(score) else t for t, sc in zip(v, score)]
            rows.append(f"{LABEL.get(k, k)}{unit.get(k, '')} & " + " & ".join(cells) + r" \\")
    heads = " & ".join(n.replace(" alone", "").replace("pooled", "compositional") for n in names)
    tex = "\n".join([
        r"\begin{table}", r"\centering",
        r"\caption{Bias (median over the simulated test datasets of posterior mean $-$ truth) and RMSE of the "
        r"posterior mean, and mean CRPS (continuous ranked probability score, a strictly proper score of the whole "
        r"marginal posterior; lower is better), calibration error (median over credibility levels of the absolute "
        r"difference between nominal and empirical coverage; lower is better) and posterior contraction "
        r"($1-\sigma^2_{\rm post}/\sigma^2_{\rm prior}$, median over datasets; higher is better), "
        r"for each single-stream posterior and the compositional posterior. "
        r"$r_D$, $z_D$ and "
        r"$\Sigma_D$ are in $\log_{10}$ (dex). Best value per parameter in bold.}",
        r"\label{tab:bias_rmse}",
        r"\begin{tabular}{l" + "r" * nc + "}", r"\hline",
        rf"Parameter & {heads} \\", *rows, r"\hline", r"\end{tabular}", r"\end{table}", ""])
    with open(path, "w") as f:
        f.write(tex)
    print(tex)


def load(path, keys):
    d = np.load(path)
    return {k: np.asarray(d[k])[..., 0] for k in keys}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trial", required=True)
    ap.add_argument("--out")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-refs", type=int, default=100, help="TARP reference sets")
    args = ap.parse_args()
    out = args.out or os.path.join(args.trial, "compositional_gain")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    J = lambda *p: os.path.join(args.trial, *p)  # noqa: E731

    path = J("eval_sim", "compositional_posterior.npz")
    pooled = load(path, np.load(path).files)
    keys = list(pooled)
    tr = np.load(J("eval_sim", "truths_flat.npz"))
    assert (tr["j"].reshape(-1, M) == np.arange(M)).all()  # group-major member rows
    truth = {k: tr[k][::M] for k in keys}
    n = len(truth[keys[0]])
    from omegaconf import OmegaConf
    cfg = OmegaConf.to_container(OmegaConf.load(J("eval_sim", ".hydra", "config.yaml")), resolve=True)
    pr = sample_prior(cfg["simulator"]["params"]["priors_global"], 10**6, rng)
    log10 = _log10_keys(cfg)
    with np.errstate(invalid="ignore"):  # normal priors on log10 keys can draw <= 0 (~1e-6): dropped
        pr = {k: np.log10(pr[k]) if k in log10 else pr[k] for k in keys}
    ok = np.all([np.isfinite(v) for v in pr.values()], axis=0)
    pr = {k: v[ok] for k, v in pr.items()}
    sd = {k: float(np.std(pr[k])) for k in keys}

    def member(path, j):
        return {k: v.reshape(n, M, -1)[:, j] for k, v in load(path, keys).items()}

    posts = {f"{s} alone": member(J("eval_sim_only_sim_summary", "base_posterior.npz"), j)
             for j, s in enumerate(STREAMS)}
    posts["curve alone"] = member(J("eval_sim_only_vcirc_kms", "base_posterior.npz"), 0)
    posts["pooled"] = pooled

    res, per, es = {}, {}, {}
    for name, p in posts.items():
        res[name], per[name], es[name] = scores(p, truth, sd, rng)

    # CRPS skill of pooled vs each reference: 1 - mean CRPS_pooled / mean CRPS_ref, bootstrap over datasets
    boot = rng.integers(0, n, (args.n_boot, n))
    skill = {}
    for name in posts:
        if name == "pooled":
            continue
        skill[name] = {}
        for k in keys + ["joint"]:
            a, b = (es["pooled"], es[name]) if k == "joint" else (per["pooled"][k], per[name][k])
            sk = 1 - a[boot].mean(1) / b[boot].mean(1)
            skill[name][k] = {"skill": float(1 - a.mean() / b.mean()), "p16_p84": np.percentile(sk, [16, 84]).tolist(),
                              "win_frac": float(np.mean(a < b))}
    with open(os.path.join(out, "compositional_gain.json"), "w") as f:
        json.dump({"n_datasets": n, "prior_sd": sd, "metrics": res, "crps_skill_of_pooled": skill}, f, indent=2)

    names = [p for p in posts if p != "curve alone"]  # the curve stays in the markdown table only
    latex_table(res, names, keys, os.path.join(out, "bias_rmse_table.tex"))

    # TARP on the joint posterior; references = independent prior draws in native space. TARP depends on
    # the reference set, so it is repeated over args.n_refs sets: median ECP + reference-to-reference spread
    from scipy.stats import kstest
    from bayesflow.utils.ecdf import simultaneous_ecdf_bands
    alpha = np.linspace(0, 1, 201)
    ecp, ks, atc = ({nm: [] for nm in names} for _ in range(3))
    for _ in range(args.n_refs):
        idx = rng.integers(0, len(pr[keys[0]]), n)
        refs = {k: pr[k][idx] for k in keys}
        for nm in names:
            c = tarp(posts[nm], truth, refs, sd)
            e = np.mean(c[:, None] <= alpha, axis=0)
            ecp[nm].append(e)
            ks[nm].append(kstest(c, "uniform").pvalue)
            atc[nm].append(np.trapezoid(e[100:] - alpha[100:], alpha[100:]))
    ecp = {nm: np.array(v) for nm, v in ecp.items()}
    tarp_sum = {nm: {"atc_median": float(np.median(atc[nm])), "atc_p16_p84": np.percentile(atc[nm], [16, 84]).tolist(),
                     "ks_pvalue_median": float(np.median(ks[nm])),
                     "frac_reference_sets_rejected_at_0.05": float(np.mean(np.array(ks[nm]) < 0.05))} for nm in names}
    with open(os.path.join(out, "tarp.json"), "w") as f:
        json.dump({"n_datasets": n, "n_reference_sets": args.n_refs, "references": "prior draws", **tarp_sum}, f, indent=2)
    print(json.dumps(tarp_sum, indent=1))

    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif"})
    col = {**{f"{s} alone": STREAM_COLOR[s] for s in STREAMS}, "pooled": RECOVERY_COLOR["compositional"]}
    lab = {nm: nm.replace(" alone", "").replace("pooled", "compositional") for nm in names}
    _, z, lo, hi = simultaneous_ecdf_bands(n)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.fill_between(z, lo, hi, color="grey", alpha=0.25, lw=0, label="95 % band")
    ax.plot([0, 1], [0, 1], color="0.5", lw=0.8, ls="--")
    for nm in names:
        ax.plot(alpha, np.median(ecp[nm], axis=0), color=col[nm], lw=2.6 if nm == "pooled" else 1.8, label=lab[nm])
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel=r"credibility level $1-\alpha$", ylabel="expected coverage")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, out, "tarp")

    # table: one block per metric, rows = posteriors, columns = parameters (+ joint energy score)
    cols = [k for k in keys]
    head = "| posterior | " + " | ".join(LABEL.get(k, k).strip("$") for k in cols) + " |"
    lines = []
    for met, fmt in (("bias", "{:+.2f}"), ("rmse", "{:.2f}"), ("cov68", "{:.2f}"), ("cov95", "{:.2f}"),
                     ("contraction", "{:.2f}"), ("cal_err", "{:.3f}"), ("crps", "{:.3f}")):
        lines += [f"\n**{met}**\n", head, "|" + "---|" * (len(cols) + 1)]
        lines += ["| " + name + " | " + " | ".join(fmt.format(res[name][k][met]) for k in cols) + " |" for name in posts]
    lines += ["\n**energy score (all parameters jointly, prior-std units)**\n", "| posterior | ES |", "|---|---|"]
    lines += [f"| {name} | {res[name]['joint']['energy']:.3f} |" for name in posts]
    lines += ["\n**CRPS skill of pooled = 1 - CRPS_pooled/CRPS_ref (win fraction)**\n",
              head.replace("| posterior |", "| reference |")[:-1] + "| joint (ES) |", "|" + "---|" * (len(cols) + 2)]
    lines += ["| " + name + " | " + " | ".join(f"{skill[name][k]['skill']:+.2f} ({skill[name][k]['win_frac']:.2f})"
                                                  for k in cols + ["joint"]) + " |" for name in skill]
    with open(os.path.join(out, "compositional_gain_table.md"), "w") as f:
        f.write(f"{n} simulated test datasets; bias/RMSE/CRPS/ES in prior-std units.\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))

    # figure: (a) CRPS skill of pooled, (b) contraction, (c) 68 % coverage
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif"})
    style = {**{f"{s} alone": dict(color=STREAM_COLOR[s], marker="o", mfc="none") for s in STREAMS},
             "pooled": dict(color="k", marker="*", ms=11)}
    fig, axs = plt.subplots(3, 1, figsize=(10, 9.5), sharex=True, gridspec_kw={"hspace": 0.08})
    x = np.arange(len(keys) + 1)
    off = (np.arange(len(names)) - (len(names) - 1) / 2) * 0.09
    for i, name in enumerate(names):
        st = {"ms": 6, "ls": "none", **style[name]}
        if name != "pooled":
            v = [skill[name][k] for k in keys + ["joint"]]
            yerr = np.abs(np.array([e["p16_p84"] for e in v]).T - [e["skill"] for e in v])
            axs[0].errorbar(x + off[i], [e["skill"] for e in v], yerr=yerr, label=name, **st)
        axs[1].plot(x[:-1] + off[i], [res[name][k]["contraction"] for k in keys], label=name, **st)
        axs[2].plot(x[:-1] + off[i], [res[name][k]["cov68"] for k in keys], **st)
    se = np.sqrt(0.68 * 0.32 / n)
    axs[2].axhspan(0.68 - 2 * se, 0.68 + 2 * se, color="0.85", lw=0)
    axs[2].axhline(0.68, color="0.5", lw=0.8)
    axs[0].axhline(0, color="0.5", lw=0.8)
    axs[1].axhline(0, color="0.5", lw=0.8)
    axs[0].set_ylabel("CRPS skill of pooled\n" + r"$1-\mathrm{CRPS}_{\rm pooled}/\mathrm{CRPS}_{\rm ref}$")
    axs[1].set_ylabel(r"contraction $1-\sigma^2_{\rm post}/\sigma^2_{\rm prior}$")
    axs[2].set_ylabel("68 % coverage")
    axs[2].set_xticks(x, [LABEL.get(k, k) for k in keys] + ["joint\n(energy score)"])
    axs[1].legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 2.3))
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
        for xi in x[:-1] + 0.5:
            ax.axvline(xi, color="0.92", lw=0.8, zorder=0)
    save(fig, out, "compositional_gain")


if __name__ == "__main__":
    main()

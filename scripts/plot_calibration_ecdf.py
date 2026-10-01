"""Paper versions of the calibration (all-parameter ECDF difference, one colour per parameter) and
recovery plots for the base (per-stream) and compositional (global) posteriors of a
composition=global sim-eval run, or per stream for a composition=local run (posterior.npz).
Adds the derived disk mass M_Disk = 2 pi Sigma_Disk r_Disk^2. Writes <run>/paper_plots/.

    python scripts/plot_calibration_ecdf.py EVAL_SIM_DIR
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import bayesflow as bf
from bayesflow.utils.ecdf import simultaneous_ecdf_bands

from replot_recovery import KEYS, load_physical

LABEL = {
    "gamma_TwoPowerTriaxial_halo": r"$\gamma$",
    "q_TwoPowerTriaxial_halo": r"$q$",
    "log10_M200_TwoPowerTriaxial_halo": r"$\log_{10}(M_{200}^{h})$",
    "ln_cvprime_TwoPowerTriaxial_halo": r"$\ln c_{V'}$",
    "Sigma_Disk": r"$\Sigma_D$",
    "z_Disk": r"$z_D$",
    "r_Disk": r"$r_D$",
    "M_Disk": r"$M_D$",
    # local (per-stream progenitor) parameters
    "m_progenitor": r"$M_{\rm prog}$",
    "t_end": r"$t_{\rm end}$",
    "vr": r"$v_{R}$",
    "r": r"$R$",
    "mu_ra_cosdec": r"$\mu_{\alpha}^{*}$",
    "mu_dec": r"$\mu_{\delta}$",
}
LOCAL_KEYS = ["m_progenitor", "t_end", "vr", "r", "mu_ra_cosdec", "mu_dec"]


# recovery colours: base black; compositional = the "Global" chain of evaluate_real's
# real_global_vs_streams_corner (RdYlBu_r(0), pipeline/evaluate_real.py)
RECOVERY_COLOR = {"base": "black", "compositional": plt.cm.RdYlBu_r(0.0)}
# per-stream colours of the same corner (scripts/replot_global_corner.py: Global + streams in j order)
STREAMS = ["Pal5", "NGC3201", "M68"]
STREAM_COLOR = dict(zip(STREAMS, plt.cm.RdYlBu_r(np.linspace(0, 1, 4))[1:]))


def fractional_ranks(est, targ):
    """(n_datasets,) fraction of posterior draws below the truth."""
    e, t = np.asarray(est)[..., 0], np.asarray(targ).reshape(-1, 1)
    return np.mean(e < t, axis=1)


def save(fig, out_dir, stem):
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=200)
        print(path)
    plt.close(fig)


def draw_ecdf(ax, est, targ, keys):
    ranks = {k: fractional_ranks(est[k], targ[k]) for k in keys}
    n = len(ranks[keys[0]])
    _, z, lo, hi = simultaneous_ecdf_bands(n)
    ax.fill_between(z, lo - z, hi - z, color="grey", alpha=0.2, label="95% band")
    colors = plt.cm.magma(np.linspace(0, 0.85, len(keys)))  # skip the near-white end
    for k, c in zip(keys, colors):
        r = np.sort(ranks[k])
        ax.plot(r, np.arange(1, n + 1) / n - r, color=c, lw=2.5, label=LABEL[k])
    ax.set(xlim=(0, 1), xlabel="Fractional rank statistic", ylabel="ECDF difference")
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_box_aspect(1)


def plot_ecdf(est, targ, keys, out_dir, stem):
    fig, ax = plt.subplots(figsize=(5, 5))
    draw_ecdf(ax, est, targ, keys)
    ax.legend(fontsize=8, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    save(fig, out_dir, stem)


def plot_ecdf_pair(panels, keys, out_dir, stem):
    """Base (left) and compositional (right) ECDF panels side by side, one legend right of the last."""
    fig, axs = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
    for ax, (est, targ) in zip(axs, panels):
        draw_ecdf(ax, est, targ, keys)
    for ax in axs[1:]:
        ax.set_ylabel("")
    axs[-1].legend(fontsize=8, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    save(fig, out_dir, stem)


def plot_ecdf_stack(panels, keys, out_dir, stem):
    """{title: (est, targ)} stacked vertically on shared x/y axes, one legend right of the top panel."""
    fig, axs = plt.subplots(len(panels), 1, figsize=(5, 4.3 * len(panels)), sharex=True, sharey=True,
                            gridspec_kw={"hspace": 0.12})
    for ax, (title, (est, targ)) in zip(axs, panels.items()):
        draw_ecdf(ax, est, targ, keys)
        ax.set_title(title)
    for ax in axs[:-1]:
        ax.set_xlabel("")
    axs[0].legend(fontsize=8, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    save(fig, out_dir, stem)


def plot_recovery(est, targ, keys, color, out_dir, stem):
    fig = bf.diagnostics.recovery(
        estimates={k: est[k] for k in keys}, targets={k: targ[k] for k in keys},
        variable_keys=keys, variable_names=[LABEL[k] for k in keys], color=color,
        metric_fontsize=24, title_fontsize=27,  # 1.5x the bayesflow defaults (16, 18)
        label_fontsize=20,  # 1.25x the default 16
    )
    for ax in fig.axes:
        ax.grid(False)
        for t in ax.texts:  # the r annotation
            t.set_bbox(dict(boxstyle="round", facecolor=(1, 1, 1, 0.9), edgecolor="black"))
    save(fig, out_dir, stem)


def load_local(run_dir):
    """(posterior, targets, j) of a composition=local sim-eval run, physical units, member rows."""
    from omegaconf import OmegaConf

    from hydrabflow.pipeline.adapter import fill_adapter_from_simulator, fill_stream_grid_from_simulator
    from hydrabflow.pipeline.compositional import flatten_members
    from hydrabflow.pipeline.evaluate import _load_test_data

    cfg = OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml"))
    fill_adapter_from_simulator(cfg)  # saved configs are pre-fill
    fill_stream_grid_from_simulator(cfg)
    test, pipeline = _load_test_data(cfg, cfg.model_dir)  # same rows as evaluate (drop_nan)
    flat = flatten_members(test, np.asarray(test["j"]).shape[1])
    keys = list(LOCAL_KEYS)
    targ = pipeline.inverse_transform({**{k: flat[k] for k in keys}, "j": flat["j"]})
    post = dict(np.load(os.path.join(run_dir, "posterior.npz")))  # saved in physical units
    assert post[keys[0]].shape[0] == np.asarray(targ[keys[0]]).shape[0]
    return post, targ, np.asarray(flat["j"]).reshape(-1).astype(int)


def main():
    run_dir = sys.argv[1]
    out_dir = os.path.join(run_dir, "paper_plots")
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(os.path.join(run_dir, "posterior.npz")):  # composition=local
        post, targ, j = load_local(run_dir)
        panels = {}
        for sid, name in enumerate(STREAMS):
            rows = j == sid
            est = {k: post[k][rows] for k in LOCAL_KEYS}
            tg = {k: np.asarray(targ[k])[rows] for k in LOCAL_KEYS}
            plot_ecdf(est, tg, LOCAL_KEYS, out_dir, f"{name}_calibration_ecdf")
            plot_recovery(est, tg, LOCAL_KEYS, STREAM_COLOR[name], out_dir, f"{name}_recovery")
            panels[name] = (est, tg)
        plot_ecdf_stack({n: panels[n] for n in ("M68", "NGC3201", "Pal5")}, LOCAL_KEYS, out_dir,
                        "calibration_ecdf_streams")
        return
    physical = load_physical(run_dir)
    for mode, (est, targ) in physical.items():
        for d in (est, targ):
            d["M_Disk"] = 2 * np.pi * np.asarray(d["Sigma_Disk"]) * np.asarray(d["r_Disk"]) ** 2
        keys = KEYS + ["M_Disk"]
        plot_ecdf(est, targ, keys, out_dir, f"{mode}_calibration_ecdf")
        plot_recovery(est, targ, keys, RECOVERY_COLOR[mode], out_dir, f"{mode}_recovery")
    plot_ecdf_pair([physical["base"], physical["compositional"]], keys, out_dir, "calibration_ecdf_base_compositional")


if __name__ == "__main__":
    main()

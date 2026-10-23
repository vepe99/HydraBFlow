"""Paper versions of the calibration (all-parameter ECDF difference, one colour per parameter) and
recovery plots for the base (per-stream) and compositional (global) posteriors of a
composition=global sim-eval run, or per stream for a composition=local run (posterior.npz).
Adds the derived disk mass M_Disk = 2 pi Sigma_Disk r_Disk^2. Writes <run>/paper_plots/.

    python scripts/plot_calibration_ecdf.py EVAL_SIM_DIR [RECOVERY_FONT_SCALE]

RECOVERY_FONT_SCALE (default 1) multiplies every font of the recovery plots (titles, x/y labels, ticks, r)
and of the ECDF plots (x/y labels, ticks, legend, panel titles).
"""

import os
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
import bayesflow as bf
from bayesflow.utils.ecdf import simultaneous_ecdf_bands

from replot_recovery import KEYS, load_physical

LABEL = {
    "gamma_TwoPowerTriaxial_halo": r"$\gamma$",
    "q_TwoPowerTriaxial_halo": r"$q$",
    "log10_M200_TwoPowerTriaxial_halo": r"$\log_{10}(M_{200}^{h})$",
    "ln_cvprime_TwoPowerTriaxial_halo": r"$\ln(C_{V'})$",
    "Sigma_Disk": r"$\Sigma_D$",
    "z_Disk": r"$z_D$",
    "r_Disk": r"$r_D$",
    "M_Disk": r"$M_D$",
    # local (per-stream progenitor) parameters
    "m_progenitor": r"$M_{\rm prog}$",
    "t_end": r"$t_{\rm end}$",
    "vr": r"$v_{R}$",
    "r": r"$R$",
    "mu_ra_cosdec": r"$\mu_{\alpha*}$",
    "mu_dec": r"$\mu_{\delta}$",
}
LOCAL_KEYS = ["m_progenitor", "t_end", "vr", "r", "mu_ra_cosdec", "mu_dec"]


# recovery colours: base black; compositional = the "Global" chain of evaluate_real's
# real_global_vs_streams_corner (RdYlBu_r(0), pipeline/evaluate_real.py)
RECOVERY_COLOR = {"base": "black", "compositional": plt.cm.RdYlBu_r(0.0)}
# per-stream colours of the same corner (scripts/replot_global_corner.py: Global + streams in j order)
STREAMS = ["Pal5", "NGC3201", "M68"]
STREAM_COLOR = dict(zip(STREAMS, plt.cm.RdYlBu_r(np.linspace(0, 1, 4))[1:]))
# ECDF font sizes, multiplied by the RECOVERY_FONT_SCALE argument in main()
FS = {"label": 14, "tick": 13, "legend": 11, "title": 14}


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
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fractional rank statistic", fontsize=FS["label"])
    ax.set_ylabel("ECDF difference", fontsize=FS["label"])
    ax.tick_params(labelsize=FS["tick"])
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_box_aspect(1)


def plot_ecdf(est, targ, keys, out_dir, stem):
    fig, ax = plt.subplots(figsize=(5, 5))
    draw_ecdf(ax, est, targ, keys)
    ax.legend(fontsize=FS["legend"], frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    save(fig, out_dir, stem)


def plot_ecdf_pair(panels, keys, out_dir, stem):
    """Base (left) and compositional (right) ECDF panels side by side, one legend right of the last."""
    fig, axs = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
    for ax, (est, targ) in zip(axs, panels):
        draw_ecdf(ax, est, targ, keys)
    for ax in axs[1:]:
        ax.set_ylabel("")
    axs[-1].legend(fontsize=FS["legend"], frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    save(fig, out_dir, stem)


def plot_ecdf_stack(panels, keys, out_dir, stem):
    """{title: (est, targ)} stacked vertically on shared x/y axes, one legend right of the top panel."""
    fig, axs = plt.subplots(len(panels), 1, figsize=(5, 4.3 * len(panels)), sharex=True, sharey=True,
                            gridspec_kw={"hspace": 0.12})
    for ax, (title, (est, targ)) in zip(axs, panels.items()):
        draw_ecdf(ax, est, targ, keys)
        ax.set_title(title, fontsize=FS["title"])
    for ax in axs[:-1]:
        ax.set_xlabel("")
    axs[0].legend(fontsize=FS["legend"], frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    save(fig, out_dir, stem)


def _interior_ticks(lo, hi, top=0.1, max_n=3):
    """2-3 round ticks at least 10 % inside [lo, hi] (`top` at the upper end): short labels, clear of
    the axis ends and of the 10^e text there. max_n=2: the two farthest-apart round candidates."""
    if max_n == 2:   # one round spacing, the pair spanning the most (ties -> coarser spacing)
        best = None
        for nb in (2, 3, 4, 5, 6):
            c = plt.MaxNLocator(nb, steps=[1, 2, 5, 10]).tick_values(lo, hi)
            c = c[(c > lo + 0.05 * (hi - lo)) & (c < hi - top * (hi - lo))]
            if len(c) >= 2 and (best is None or c[-1] - c[0] > best[-1] - best[0] + 1e-12):
                best = c[[0, -1]]
        if best is not None:
            return best
    for nbins in (3, 4, 5, 6):
        t = plt.MaxNLocator(nbins, steps=[1, 2, 5, 10]).tick_values(lo, hi)
        t = t[(t > lo + 0.1 * (hi - lo)) & (t < hi - top * (hi - lo))]
        if len(t) >= 2:
            t = t if len(t) <= 3 else t[::2]
            return t[[0, -1]] if len(t) > max_n else t
    return t


def _recovery(est, targ, keys, color, scale=1.0, xlabel="Ground truth", axes=None, units_on_axes=False,
              metric_scale=np.inf, unit_fmt="10^", tick_scale=1.0):
    """bayesflow recovery figure in the paper style; with `axes` it draws into those (one row of a
    larger figure) instead of a figure of its own."""
    # values with >2 integer digits are shown in units of 10^e, written into the panel title, or with
    # units_on_axes as a "10^e" at the end of both axes (two ticks there, the long labels collide)
    est, targ, names, exps = dict(est), dict(targ), [], {}
    for k in keys:
        e = int(np.floor(np.log10(np.nanmax(np.abs(targ[k])))))
        if e >= 2:
            est[k], targ[k] = np.asarray(est[k]) / 10.0**e, np.asarray(targ[k]) / 10.0**e
            exps[k] = e
        names.append(LABEL[k] + (rf" [$10^{{{e}}}$]" if e >= 2 and not units_on_axes else ""))
    bf_recovery = sys.modules[bf.diagnostics.recovery.__module__]   # the module, shadowed by the function
    orig = bf_recovery.prepare_plot_data
    if axes is not None:   # hand bayesflow our axes instead of the figure it creates
        def into_axes(*a, **k):
            d = orig(*a, **k)
            plt.close(d["fig"])
            d["fig"], d["axes"] = axes.flat[0].figure, np.asarray(axes).reshape(np.shape(d["axes"]))
            return d
        bf_recovery.prepare_plot_data = into_axes
    try:
        fig = bf.diagnostics.recovery(
            estimates={k: est[k] for k in keys}, targets={k: targ[k] for k in keys},
            variable_keys=keys, variable_names=names, color=color, xlabel=xlabel,
            metric_fontsize=24 * min(scale, metric_scale), title_fontsize=27 * scale,  # 1.5x the bayesflow defaults (16, 18)
            label_fontsize=20 * scale,  # 1.25x the default 16
            tick_fontsize=16 * scale * tick_scale,
        )
    finally:
        bf_recovery.prepare_plot_data = orig
    for ax in (fig.axes if axes is None else np.asarray(axes).flat):
        ax.grid(False)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        for t in ax.texts:  # the r annotation: bottom right, below the diagonal, clear of the title
            t.set_position((0.96, 0.04))
            t.set_ha("right")
            t.set_va("bottom")
            t.set_bbox(dict(boxstyle="round", facecolor=(1, 1, 1, 0.9), edgecolor="black"))
    if units_on_axes:
        for ax, k in zip(np.asarray(axes).flat if axes is not None else fig.axes, keys):
            for axis, (lo, hi), n, free, ufree in ((ax.xaxis, ax.get_xlim(), 2, 0.05, 0.3 if unit_fmt == "10^" else 0.2),
                                                   (ax.yaxis, ax.get_ylim(), 3, 0.1, 0.3)):
                # scaled axes keep their upper end free for the 10^e text; x labels side by side -> 2 ticks
                axis.set_major_locator(FixedLocator(_interior_ticks(lo, hi, ufree if k in exps else free, n)))
                axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace("-", "\u2212")))
            if k in exps:
                u = rf"$10^{{{exps[k]}}}$" if unit_fmt == "10^" else f"1e{exps[k]}"   # "1e": shorter
                # on the tick-label row/column: offset from the axis end by tick length + label pad
                py = plt.rcParams["ytick.major.size"] + plt.rcParams["ytick.major.pad"]
                ax._unit_text = u   # the x-axis copy is placed after layout, right after the last tick label
                ax.annotate(u, (0.0, 1.0), xycoords="axes fraction", xytext=(-py, 0), textcoords="offset points",
                            ha="right", va="top", fontsize=16 * scale * tick_scale)   # just below the top end
    return fig


def plot_recovery(est, targ, keys, color, out_dir, stem, scale=1.0):
    save(_recovery(est, targ, keys, color, scale), out_dir, stem)


def plot_recovery_stack(panels, keys, out_dir, stem, scale=1.0):
    """{stream: (est, targ)} as aligned rows of one figure: one title per column (top row), the 10^e
    units as text on the axes, the "Ground truth" x label only on the bottom row."""
    n = len(keys)
    fig, axs = plt.subplots(len(panels), n, figsize=(7.5 * n, 7.5 * len(panels)), squeeze=False)
    for r, (name, (est, targ)) in enumerate(panels.items()):
        last = r == len(panels) - 1
        _recovery(est, targ, keys, STREAM_COLOR[name], scale, xlabel="Ground truth" if last else "", axes=axs[r],
                  units_on_axes=True, metric_scale=scale / 1.2)   # r boxes one step smaller: they cover the data
        if r:
            for ax in axs[r]:
                ax.set_title("")
    _finish_grid(fig, axs, scale)
    save(fig, out_dir, stem)


def plot_recovery_pair(left, right, keys, out_dir, stem, scale=1.0, colors=None):
    """Two recovery figures, (est, targ) each, side by side in one grid (bayesflow's row x col layout per
    side); "Estimate" only on the left side, the 10^e units as text on the axes as in plot_recovery_stack."""
    n = len(keys)
    from bayesflow.utils.plot_utils import set_layout
    nr, nc = set_layout(n, None, None, False)   # bayesflow's own grid for n variables
    w = 8.2   # panel size (inches; fonts are absolute, so bigger = larger axes); sides at the inter-panel distance
    # cells narrower than tall: the square axes are height-limited (title + x label), so a square cell
    # would leave its extra width as white space between the panels
    fig, axs = plt.subplots(nr, 2 * nc, figsize=(0.85 * w * 2 * nc, w * nr), squeeze=False)
    colors = colors or [RECOVERY_COLOR["base"], RECOVERY_COLOR["compositional"]]
    for side, ((est, targ), c) in enumerate(zip((left, right), colors)):
        _recovery(est, targ, keys, c, scale, axes=axs[:, side * nc:(side + 1) * nc], units_on_axes=True, unit_fmt="1e",
                  tick_scale=0.7,   # smaller tick labels -> more of each panel for the data
                  metric_scale=scale / 1.2)
    for ax in axs[:, nc]:
        ax.set_ylabel("")
    _finish_grid(fig, axs, scale * 0.7, w_pad=2.2)
    save(fig, out_dir, stem)


def plot_recovery_overlay(panels, keys, out_dir, stem, scale=1.0, prob=0.95):
    """{label: (est, targ, color)} overlaid per parameter, the same information as bayesflow's recovery
    scatter: per test set the posterior median (a dot) and its `prob` credible interval, the intervals
    drawn as one band (fill_between over the test sets sorted by truth) instead of individual error
    bars. One r per label (corr of posterior median vs truth) in its colour."""
    from bayesflow.utils.plot_utils import set_layout
    nr, nc = set_layout(len(keys), None, None, False)
    w = 8.2
    fig, axs = plt.subplots(nr, nc, figsize=(0.8 * w * nc, w * nr), squeeze=False)
    fs = {"title": 27 * scale, "label": 20 * scale, "tick": 16 * scale * 0.7, "metric": 16 * scale * 0.8}
    for i, (ax, k) in enumerate(zip(axs.flat, keys)):
        e = int(np.floor(np.log10(np.nanmax(np.abs(next(iter(panels.values()))[1][k])))))
        f = 10.0**e if e >= 2 else 1.0
        lo_all, hi_all, rs = [], [], []
        for n, (label, (est, targ, color)) in enumerate(panels.items()):
            x = np.asarray(targ[k]).reshape(-1) / f
            draws = np.asarray(est[k])[..., 0] / f                     # (n_sets, n_draws)
            med = np.median(draws, axis=1)
            lo, hi = np.quantile(draws, [(1 - prob) / 2, (1 + prob) / 2], axis=1)
            o = np.argsort(x)
            # first label: filled band; later ones: lighter fill + edge lines, so overlaps stay readable
            ax.fill_between(x[o], lo[o], hi[o], color=color, alpha=0.25 if n == 0 else 0.15, lw=0)
            if n:
                ax.plot(x[o], lo[o], color=color, lw=0.8)
                ax.plot(x[o], hi[o], color=color, lw=0.8)
            ax.plot(x, med, "o", ms=3, color=color, alpha=0.6, mew=0)
            ax.plot([], [], "o", ms=3, color=color, label=label)   # opaque legend marker
            rs.append((np.corrcoef(x, med)[0, 1], color))
            lo_all += [x.min(), lo.min()]
            hi_all += [x.max(), hi.max()]
        lim = (min(lo_all), max(hi_all))
        pad = 0.03 * (lim[1] - lim[0])
        lim = (lim[0] - pad, lim[1] + pad)
        ax.plot(lim, lim, "k--", lw=1.2, zorder=0)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_box_aspect(1)
        ax.set_title(LABEL[k], fontsize=fs["title"])
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=fs["tick"])
        for axis, n, free, ufree in ((ax.xaxis, 2, 0.05, 0.2), (ax.yaxis, 3, 0.1, 0.3)):
            axis.set_major_locator(FixedLocator(_interior_ticks(*lim, ufree if f > 1 else free, n)))
            axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace("-", "\u2212")))
        if f > 1:
            ax._unit_text = f"1e{e}"
            py = plt.rcParams["ytick.major.size"] + plt.rcParams["ytick.major.pad"]
            ax.annotate(ax._unit_text, (0.0, 1.0), xycoords="axes fraction", xytext=(-py, 0),
                        textcoords="offset points", ha="right", va="top", fontsize=fs["tick"])
        # one r per label, stacked in the bottom-right corner in the label's colour
        from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker
        lines = [TextArea(f"$r = {r:.3f}$", textprops={"color": c, "fontsize": fs["metric"]}) for r, c in rs]
        box = AnchoredOffsetbox("lower right", child=VPacker(children=lines, align="right", pad=0, sep=2),
                                pad=0.3, borderpad=0.4, frameon=True)
        box.patch.set_boxstyle("round")
        box.patch.set_facecolor((1, 1, 1, 0.9))
        ax.add_artist(box)
        if i % nc == 0:
            ax.set_ylabel("Estimate", fontsize=fs["label"])
        if i // nc == nr - 1:
            ax.set_xlabel("Ground truth", fontsize=fs["label"])
    _finish_grid(fig, axs, scale * 0.7, w_pad=2.2)
    fig.legend(*axs.flat[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 1.0),
               ncol=len(panels), fontsize=fs["label"], frameon=False, markerscale=5)   # above the grid
    save(fig, out_dir, stem)


def _finish_grid(fig, axs, scale, w_pad=None):   # scale: of the x-axis unit text, = the tick labels
    """Square panels, layout, then the x-axis 10^e text on the tick-label row at the right end of the axis."""
    for ax in axs.flat:
        ax.set_box_aspect(1)
    fig.tight_layout(h_pad=2.0, w_pad=w_pad)   # bayesflow's own tight_layout ran before the later panels existed
    fig.canvas.draw()
    W, H = fig.bbox.width, fig.bbox.height
    for ax in axs.flat:
        if not hasattr(ax, "_unit_text"):
            continue
        row = [t for t in ax.get_xticklabels() if t.get_text()][-1].get_window_extent()
        txt = fig.text(ax.get_window_extent().x1 / W, row.y1 / H, ax._unit_text, ha="right", va="top",
                       fontsize=16 * scale)
        fig.canvas.draw()
        gap = 0.5 * txt.get_fontsize() * fig.dpi / 72   # half a character, in pixels
        if any(txt.get_window_extent().padded(gap).overlaps(t.get_window_extent())
               for t in ax.get_xticklabels() if t.get_text()):
            # the 10^e text runs into the last tick label: re-pick the ticks with more room at that end
            ax.xaxis.set_major_locator(FixedLocator(_interior_ticks(*ax.get_xlim(), 0.45, 2)))


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
    scale = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    FS.update({k: v * scale for k, v in FS.items()})
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
            plot_recovery(est, tg, LOCAL_KEYS, STREAM_COLOR[name], out_dir, f"{name}_recovery", scale)
            panels[name] = (est, tg)
        plot_ecdf_stack({n: panels[n] for n in ("M68", "NGC3201", "Pal5")}, LOCAL_KEYS, out_dir,
                        "calibration_ecdf_streams")
        plot_recovery_stack({n: panels[n] for n in ("M68", "NGC3201", "Pal5")}, LOCAL_KEYS, out_dir,
                            "recovery_streams", 1.8 * scale)   # paper figure: 1.8x the single-stream fonts
        return
    physical = load_physical(run_dir)
    for mode, (est, targ) in physical.items():
        for d in (est, targ):
            d["M_Disk"] = 2 * np.pi * np.asarray(d["Sigma_Disk"]) * np.asarray(d["r_Disk"]) ** 2
        keys = KEYS + ["M_Disk"]
        plot_ecdf(est, targ, keys, out_dir, f"{mode}_calibration_ecdf")
        plot_recovery(est, targ, keys, RECOVERY_COLOR[mode], out_dir, f"{mode}_recovery", scale)
    plot_ecdf_pair([physical["base"], physical["compositional"]], keys, out_dir, "calibration_ecdf_base_compositional")


if __name__ == "__main__":
    main()

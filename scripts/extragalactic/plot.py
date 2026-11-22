#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.5.5", "h5py==3.16.0", "matplotlib==3.11.2"]
# ///
"""Prior-predictive figures of the extragalactic stream dataset (reads data/, writes plots/ as PDF + PNG).

    uv run --script scripts/extragalactic/plot.py                    # every figure of the production set
    uv run --script scripts/extragalactic/plot.py --subdir pilot     # the pilot (figures go to plots/pilot/)
    uv run --script scripts/extragalactic/plot.py --only gallery,noise

Figures: marginals, energy, binned, gallery (3 pages), noise, tests (StreaMAX residuals + energy conservation).
`generate.py analyze` must have been run first (it writes data/analysis.h5).
"""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap

REPO = Path(__file__).resolve().parents[2]

# --- palette (roles, light surface): three categorical slots, one sequential hue, one diverging pair
SURFACE, INK, INK2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
C_ALL, C_BOUND, C_PASS = "#2a78d6", "#eb6834", "#1baf7a"  # all | bound | passing the three cuts
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div_blue_red", ["#104281", "#2a78d6", "#b9b8ae", "#e34948", "#8f1f1f"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "legend.frameon": False, "legend.fontsize": 8.5, "lines.linewidth": 2.0, "figure.dpi": 100,
})  # fmt: skip


# ------------------------------------------------------------------------------------------------------
# data
# ------------------------------------------------------------------------------------------------------
class Data:
    def __init__(self, out: Path, subdir: str):
        self.ddir = out / "data" / subdir if subdir else out / "data"
        self.pdir = out / "plots" / subdir if subdir else out / "plots"
        self.pdir.mkdir(parents=True, exist_ok=True)
        self._a = self._tracks = None

    @property
    def a(self) -> dict:
        if self._a is None:
            p = self.ddir / "analysis.h5"
            if not p.exists():
                raise SystemExit(f"{p} not found: run `generate.py analyze` first")
            with h5py.File(p, "r") as f:
                self._a = {k: f[k][:] for k in f}
                self.cfg = json.loads(f.attrs["config"])
                self.names = json.loads(f.attrs["theta_names"])
            a = self._a
            a["bound"] = ~a["unbound"].astype(bool) & ((a["status"] & 16) == 0)
            a["speed"] = np.linalg.norm(a["theta"][:, 9:12], axis=1)
        return self._a

    @property
    def tracks(self) -> dict:
        if self._tracks is None:
            cols: dict[str, list] = {}
            for p in sorted(self.ddir.glob("shard_*.h5")):
                with h5py.File(p, "r") as f:
                    cols.setdefault("index", []).append(f["index"][:])
                    for k in ("count", "r_mean", "r_std"):
                        cols.setdefault(k, []).append(f["track"][k][:])
            t = {k: np.concatenate(v) for k, v in cols.items()}
            o = np.argsort(t["index"])
            self._tracks = {k: v[o] for k, v in t.items()}
        return self._tracks

    def particles(self, idx: int):
        for p in sorted(self.ddir.glob("particles_*.h5")):
            with h5py.File(p, "r") as f:
                ii = f["index"][:]
                j = np.nonzero(ii == idx)[0]
                if len(j):
                    return f["xy"][j[0]], f["phi"][j[0]]
        return None

    def particle_indices(self) -> np.ndarray:
        out = []
        for p in sorted(self.ddir.glob("particles_*.h5")):
            with h5py.File(p, "r") as f:
                out.append(f["index"][:])
        return np.concatenate(out) if out else np.array([], dtype=int)

    @property
    def phi(self) -> np.ndarray:
        c = self.cfg
        return np.arange(-c["phi_max_deg"], c["phi_max_deg"], c["bin_deg"]) + c["bin_deg"] / 2


def save(fig, d: Data, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(d.pdir / f"{name}.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {d.pdir / name}.pdf / .png")


def step_hist(ax, x, bins, color, label=None, density=True, lw=2.0, fill=False):
    h, e = np.histogram(x[np.isfinite(x)], bins=bins, density=density)
    if fill:
        ax.stairs(h, e, color=color, alpha=0.25, fill=True, lw=0)
    ax.stairs(h, e, color=color, lw=lw, label=label)
    return h, e


# ------------------------------------------------------------------------------------------------------
# 1. marginals
# ------------------------------------------------------------------------------------------------------
def fig_marginals(d: Data) -> None:
    a, c = d.a, None
    c = d.cfg
    th, vec = a["theta"], a["c26_vec"]
    npdf = lambda x, s: np.exp(-0.5 * (x / s) ** 2) / (s * np.sqrt(2 * np.pi))
    uni = lambda ab: (lambda x: np.where((x >= ab[0]) & (x <= ab[1]), 1.0 / (ab[1] - ab[0]), 0.0))
    half = lambda s: (lambda x: np.where(x >= 0, 2 * npdf(x, s), 0.0))
    norm = lambda s: (lambda x: npdf(x, s))
    labels = [r"$\log_{10} M_{\rm halo}\ [M_\odot]$", r"$R_s$ [kpc]", r"$q$", r"$\cos i$", r"$\psi$ [rad]",
              r"$\log_{10} m_{\rm prog}\ [M_\odot]$", r"$r_s$ [kpc]", r"$x_0$ [kpc]", r"$z_0$ [kpc]",
              r"$v_{x0}$ [km/s]", r"$v_{y0}$ [km/s]", r"$v_{z0}$ [km/s]", r"$t_{\rm int}$ [Gyr]"]
    pdfs = [uni(c["log10_M_halo"]), uni(c["R_s"]), uni(c["q"]), uni((0, 1)), uni((0, 2 * np.pi)),
            uni(c["log10_m_prog"]), uni(c["r_s"]), half(c["sigma_pos"]), half(c["sigma_pos"]),
            norm(c["sigma_vel"]), half(c["sigma_vel"]), norm(c["sigma_vel"]), uni(c["t_int"])]
    cols = [th[:, i] for i in range(13)] + [vec[:, 0], vec[:, 1], vec[:, 2]]
    labels += [r"$\hat x$", r"$\hat y$", r"$\hat z$"]
    pdfs += [norm(1.0), norm(1.0), half(1.0)]
    fig, axes = plt.subplots(4, 4, figsize=(13, 10.5))
    for ax, x, lab, pdf in zip(axes.ravel(), cols, labels, pdfs):
        lo, hi = np.min(x), np.max(x)
        pad = 0.04 * (hi - lo)
        grid = np.linspace(lo - pad, hi + pad, 400)
        step_hist(ax, x, np.linspace(lo, hi, 51), C_ALL, fill=True, lw=1.5)
        ax.plot(grid, pdf(grid), color=INK, lw=1.2, ls="--")
        ax.set_xlabel(lab)
        ax.set_yticks([])
        ax.grid(False)
    axes[0, 0].plot([], [], color=C_ALL, lw=1.5, label=f"dataset (n = {len(th)})")
    axes[0, 0].plot([], [], color=INK, lw=1.2, ls="--", label="analytic prior")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("Sampled parameters against their priors; $(\\hat x,\\hat y,\\hat z)$ is the stored C26 vector "
                 "(StreaMAX convention)", y=1.02, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    save(fig, d, "prior_marginals")


# ------------------------------------------------------------------------------------------------------
# 2. energy
# ------------------------------------------------------------------------------------------------------
def fig_energy(d: Data) -> None:
    a = d.a
    ratio = a["E"] / np.abs(a["Phi0"])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw=dict(width_ratios=[1.15, 1]))
    hi = float(np.nanpercentile(ratio, 99))
    bins = np.linspace(-1, max(hi, 1.0), 81)
    h, e = np.histogram(np.clip(ratio[np.isfinite(ratio)], -1, bins[-1]), bins=bins)
    ctr = 0.5 * (e[:-1] + e[1:])
    ax.stairs(np.where(ctr < 0, h, 0), e, color=C_BOUND, fill=True, alpha=0.3, lw=0)
    ax.stairs(np.where(ctr >= 0, h, 0), e, color=C_ALL, fill=True, alpha=0.3, lw=0)
    ax.stairs(h, e, color=INK2, lw=1.2)
    ax.axvline(0, color=INK, lw=1.0)
    fb = float(np.mean(ratio < 0))
    ax.text(0.02, 0.95, f"bound  {100 * fb:.1f} %", transform=ax.transAxes, color=INK, va="top", ha="left")
    ax.text(0.98, 0.95, f"unbound  {100 * (1 - fb):.1f} %", transform=ax.transAxes, color=INK, va="top", ha="right")
    ax.set_xlabel(r"$E\,/\,|\Phi(x_0)|$   (last bin: all larger values)")
    ax.set_ylabel("streams")
    ax.set_title("Progenitor specific energy today")
    em = np.linspace(d.cfg["log10_M_halo"][0], d.cfg["log10_M_halo"][1], 13)
    ev = np.linspace(0, 900, 19)
    n, _, _ = np.histogram2d(a["theta"][:, 0], a["speed"], bins=[em, ev])
    nu, _, _ = np.histogram2d(a["theta"][:, 0][~a["bound"]], a["speed"][~a["bound"]], bins=[em, ev])
    with np.errstate(all="ignore"):
        frac = np.where(n >= 5, nu / n, np.nan)
    im = ax2.pcolormesh(em, ev, frac.T, cmap=SEQ, vmin=0, vmax=1, edgecolors=SURFACE, linewidth=0.6)
    ax2.grid(False)
    ax2.set_xlabel(r"$\log_{10} M_{\rm halo}\ [M_\odot]$")
    ax2.set_ylabel(r"speed today $|v_0|$ [km/s]")
    ax2.set_title("Unbound fraction (cells with fewer than 5 streams left blank)")
    fig.colorbar(im, ax=ax2, label="fraction unbound", pad=0.02)
    fig.tight_layout()
    save(fig, d, "energy")


# ------------------------------------------------------------------------------------------------------
# 3. binned values
# ------------------------------------------------------------------------------------------------------
def fig_binned(d: Data) -> None:
    a, t, c = d.a, d.tracks, d.cfg
    has = a["n_valid"] > 0
    groups = [("all", has, C_ALL), ("bound", has & a["bound"], C_BOUND),
              ("passing the three cuts", a["sel_all"].astype(bool), C_PASS)]
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8))
    specs = [("ang_len_deg", "angular length [deg]", np.linspace(0, 720, 73), False, c["sel_ang_deg"]),
             ("arc_len_kpc", "arc length [kpc]", np.logspace(0, 3.7, 60), True, c["sel_arc_kpc"]),
             ("r_track_mean", "mean track radius [kpc]", np.logspace(0, 3.5, 60), True, None),
             ("r_track_spread", "radial spread along the track [kpc]", np.logspace(-1.5, 3, 60), True, None)]
    for ax, (key, lab, bins, logx, cut) in zip(axes.ravel()[:4], specs):
        for name, m, col in groups:
            x = np.clip(a[key][m], bins[0], bins[-1])
            w = np.full(len(x), 1.0 / max(len(x), 1))
            h, e = np.histogram(x, bins=bins, weights=w)
            ax.stairs(h, e, color=col, lw=2.0, label=f"{name} (n = {int(m.sum())})")
        if cut is not None:
            ax.axvline(cut, color=INK2, lw=1.0, ls=":")
            ax.text(cut, ax.get_ylim()[1] * 0.97, " cut", color=INK2, va="top", ha="left", fontsize=8)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(lab + "   (range clipped)")
        ax.set_ylabel("fraction of the group")
    axes[0, 0].legend(loc="upper right")
    # particles per valid bin
    ax = axes[1, 1]
    valid = t["count"] >= c["n_valid_min"]
    bins = np.logspace(np.log10(c["n_valid_min"]), np.log10(c["n_particles"]), 50)
    for name, m, col in groups:
        x = t["count"][m][valid[m]]
        h, e = np.histogram(x, bins=bins, weights=np.full(len(x), 1.0 / max(len(x), 1)))
        ax.stairs(h, e, color=col, lw=2.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"particles per valid bin (valid: at least {c['n_valid_min']})")
    ax.set_ylabel("fraction of valid bins")
    # percentile bands of r(phi)
    ax = axes[1, 2]
    phi = d.phi
    for name, m, col in (groups[0], groups[2]):
        r = np.where(valid[m], t["r_mean"][m], np.nan).astype(np.float64)
        nn = np.isfinite(r).sum(0)
        use = nn >= max(20, 0.002 * m.sum())
        if not use.any():
            continue
        with np.errstate(all="ignore"):
            p5, p16, p50, p84, p95 = np.nanpercentile(r[:, use], [5, 16, 50, 84, 95], axis=0)
        ax.fill_between(phi[use], p5, p95, color=col, alpha=0.15, lw=0)
        ax.fill_between(phi[use], p16, p84, color=col, alpha=0.30, lw=0)
        ax.plot(phi[use], p50, color=col, lw=2.0, label=f"{name}: median, 16-84, 5-95")
    ax.set_yscale("log")
    ax.set_xlabel(r"unwrapped angle $\phi$ [deg]  (trailing < 0 < leading)")
    ax.set_ylabel(r"$r(\phi)$ over streams with a valid bin [kpc]")
    ax.legend(loc="upper center")
    fig.suptitle("Track summaries of the prior predictive (valid bin: "
                 f">= {c['n_valid_min']} particles)", fontsize=10.5)
    fig.tight_layout()
    save(fig, d, "binned_values")


# ------------------------------------------------------------------------------------------------------
# 4-5. gallery and the noisy examples
# ------------------------------------------------------------------------------------------------------
def draw_stream(d: Data, ax_xy, ax_rp, k: int, noise: float = 0.0, rng=None, tag: str = "") -> None:
    a, t, c = d.a, d.tracks, d.cfg
    row = int(np.searchsorted(a["index"], k))
    th = a["theta"][row]
    xy, phi = d.particles(k)
    cnt, rm, rs = t["count"][row], t["r_mean"][row].astype(float), t["r_std"][row].astype(float)
    valid = cnt >= c["n_valid_min"]
    pc = np.deg2rad(d.phi)
    fin = np.isfinite(phi) & np.isfinite(xy[:, 0])
    vmax = max(30.0, float(np.nanpercentile(np.abs(phi[fin]), 99))) if fin.any() else 180.0
    r = np.hypot(xy[:, 0], xy[:, 1])
    kw = dict(c=phi[fin], cmap=DIV, vmin=-vmax, vmax=vmax, s=1.2, lw=0, rasterized=True)
    ax_xy.scatter(xy[fin, 0], xy[fin, 1], **kw)
    ax_rp.scatter(phi[fin], r[fin], **kw)
    ang0 = np.arctan2(0.0, th[7])  # the progenitor sits at phi = 0, i.e. at polar angle 0 in the XY plane
    if noise > 0:
        robs = rm * (1.0 + noise * rng.standard_normal(rm.shape))
        err = noise * rm
        rv = np.where(valid, rm, np.nan)  # NaN breaks the line where bins are not valid
        ax_rp.plot(d.phi, rv, color=INK2, lw=1.0, label="noise-free track")
        ax_xy.plot(rv * np.cos(pc + ang0), rv * np.sin(pc + ang0), color=INK2, lw=1.0)
        yv, ev, lab = robs, err, f"with {100 * noise:g} % radial noise"
    else:
        yv, ev, lab = rm, rs, r"binned track, bars = $r_{\rm std}$"
    ax_rp.errorbar(d.phi[valid], yv[valid], yerr=ev[valid], fmt="o", ms=3.2, color=INK, ecolor=INK, elinewidth=0.9,
                   mfc=SURFACE, mew=0.9, label=lab, zorder=5)
    ux, uy = np.cos(pc[valid] + ang0), np.sin(pc[valid] + ang0)
    for x0_, y0_, lo, hi in zip(ux, uy, (yv - ev)[valid], (yv + ev)[valid]):
        ax_xy.plot([lo * x0_, hi * x0_], [lo * y0_, hi * y0_], color=INK, lw=0.9, zorder=5)
    ax_xy.plot(yv[valid] * ux, yv[valid] * uy, "o", ms=3.2, color=INK, mfc=SURFACE, mew=0.9, zorder=6)
    ax_xy.plot([th[7]], [0.0], marker="*", ms=11, color=INK, mfc="#eda100", mew=0.8, zorder=7, ls="none")
    ax_xy.plot([0], [0], marker="+", ms=7, color=INK2, zorder=4)
    lim = 1.08 * max(np.nanpercentile(r[fin], 99.5) if fin.any() else 1.0, abs(th[7]))
    ax_xy.set_xlim(-lim, lim)
    ax_xy.set_ylim(-lim, lim)
    ax_xy.set_aspect("equal")
    sin_i = np.sqrt(1 - th[3] ** 2)
    L = 0.45 * lim * sin_i
    if L > 0.02 * lim:
        ax_xy.annotate("", xy=(L * np.cos(th[4]), L * np.sin(th[4])), xytext=(0, 0),
                       arrowprops=dict(arrowstyle="-|>", color="#6250d6", lw=1.8), zorder=8)
    ax_xy.set_title(f"{tag}#{k}  q={th[2]:.2f}  cos i={th[3]:.2f}  $\\psi$={np.rad2deg(th[4]):.0f}$^\\circ$  "
                    f"log M={th[0]:.2f}", fontsize=8.2)
    ax_xy.set_xlabel("X [kpc]")
    ax_xy.set_ylabel("Y [kpc]")
    ax_rp.set_xlabel(r"$\phi$ [deg]")
    ax_rp.set_ylabel("r [kpc]")
    sel = fin & (np.abs(phi) <= vmax * 1.15)
    if sel.any():
        ax_rp.set_xlim(-vmax * 1.15, vmax * 1.15)
        ax_rp.set_ylim(0, 1.08 * np.nanpercentile(r[sel], 99.5))


def pick_gallery(d: Data, seed: int):
    a = d.a
    have = np.isin(a["index"], d.particle_indices())
    rng = np.random.default_rng(seed)
    sel = a["sel_all"].astype(bool)
    pools = [("pass", have & sel & a["bound"], 16), ("unbound", have & ~a["bound"] & (a["n_valid"] > 0), 4),
             ("bound, too short", have & a["bound"] & ~a["sel_ang"].astype(bool) & (a["n_valid"] > 0), 4)]
    out = []
    for name, m, n in pools:
        idx = a["index"][m]
        out.append((name, np.sort(rng.choice(idx, size=min(n, len(idx)), replace=False)) if len(idx) else idx))
    return out


def _legend_handles(fig) -> None:
    h = [plt.Line2D([], [], marker="*", ms=10, color=INK, mfc="#eda100", ls="none", label="progenitor today"),
         plt.Line2D([], [], color="#6250d6", lw=1.8, label=r"projected halo axis (length $\propto \sin i$)"),
         plt.Line2D([], [], marker="o", ms=4, color=INK, mfc=SURFACE, ls="none", label="binned track (valid bins)"),
         plt.Line2D([], [], marker="o", ms=4, color="#2a78d6", ls="none", label=r"particles, trailing ($\phi<0$)"),
         plt.Line2D([], [], marker="o", ms=4, color="#e34948", ls="none", label=r"particles, leading ($\phi>0$)")]
    fig.legend(handles=h, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.995), fontsize=8.5)


def fig_gallery(d: Data, seed: int) -> None:
    picks = pick_gallery(d, seed)
    items = [(f"{name} ", k) for name, ks in picks for k in ks]
    if not items:
        print("gallery: no stream with stored particles")
        return
    pages = [items[i : i + 8] for i in range(0, len(items), 8)]
    with PdfPages(d.pdir / "gallery.pdf") as pdf:
        for ip, page in enumerate(pages, 1):
            fig, axes = plt.subplots(4, 4, figsize=(17, 16.5))
            for j, (tag, k) in enumerate(page):
                draw_stream(d, axes[j // 2, 2 * (j % 2)], axes[j // 2, 2 * (j % 2) + 1], int(k), tag=tag)
            for j in range(len(page), 8):
                axes[j // 2, 2 * (j % 2)].axis("off")
                axes[j // 2, 2 * (j % 2) + 1].axis("off")
            _legend_handles(fig)
            fig.tight_layout(rect=(0, 0, 1, 0.975))
            pdf.savefig(fig, dpi=160, bbox_inches="tight")
            fig.savefig(d.pdir / f"gallery_p{ip}.png", dpi=130, bbox_inches="tight")
            plt.close(fig)
    print(f"wrote {d.pdir / 'gallery.pdf'} ({len(pages)} pages) and gallery_p*.png")


def fig_noise(d: Data, seed: int, noise: float) -> None:
    ks = pick_gallery(d, seed)[0][1][:3]
    if not len(ks):
        print("noise: no passing stream with stored particles")
        return
    rng = np.random.default_rng(seed + 1)
    fig, axes = plt.subplots(len(ks), 2, figsize=(11, 4.6 * len(ks)), squeeze=False)
    for i, k in enumerate(ks):
        draw_stream(d, axes[i, 0], axes[i, 1], int(k), noise=noise, rng=rng)
        axes[i, 1].legend(loc="best")
    fig.suptitle(f"What the network receives: valid bins with {100 * noise:g} % Gaussian radial noise per bin "
                 r"($\sigma_{\rm sys}=0$, the C26 likelihood)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    save(fig, d, "noise_examples")


# ------------------------------------------------------------------------------------------------------
# tests: StreaMAX residuals and energy conservation
# ------------------------------------------------------------------------------------------------------
def fig_tests(d: Data) -> None:
    tdir = d.ddir / "tests"
    p = tdir / "streamax_compare.npz"
    if p.exists():
        z = np.load(p)
        n = len(z["theta"])
        phi = z["phi_centres"]
        fig, axes = plt.subplots(2, n, figsize=(3.9 * n, 6.6), squeeze=False, sharex="col",
                                 gridspec_kw=dict(height_ratios=[2, 1]))
        for i in range(n):
            ax, axr = axes[0, i], axes[1, i]
            series = [("this code, same ordering", "this", C_ALL, "o"), ("StreaMAX 0.4.1 (float32)", "streamax", C_BOUND, "s"),
                      ("this code, C26 ordering", "this_c26", C_PASS, "^")]
            anyc = np.zeros(len(phi), bool)
            for lab, key, col, mk in series:
                cnt, rm, rs = z[f"count_{key}_{i}"], z[f"r_mean_{key}_{i}"], z[f"r_std_{key}_{i}"]
                v = cnt >= 100
                anyc |= v
                ax.errorbar(phi[v], rm[v], yerr=rs[v] / np.sqrt(np.maximum(cnt[v], 1)), fmt=mk, ms=4, color=col,
                            mfc=SURFACE, mew=1.2, elinewidth=1.0, label=lab, ls="none")
            for key, col, mk in (("this", C_ALL, "o"), ("this_c26", C_PASS, "^")):
                ca, cb = z[f"count_{key}_{i}"], z[f"count_streamax_{i}"]
                v = (ca >= 100) & (cb >= 100)
                res = (z[f"r_mean_{key}_{i}"] - z[f"r_mean_streamax_{i}"]) / np.sqrt(
                    z[f"r_std_{key}_{i}"] ** 2 / np.maximum(ca, 1) + z[f"r_std_streamax_{i}"] ** 2 / np.maximum(cb, 1))
                axr.plot(phi[v], res[v], mk, ms=4, color=col, mfc=SURFACE, mew=1.2)
            axr.axhline(0, color=INK2, lw=1.0)
            for s in (-2, 2):
                axr.axhline(s, color=INK2, lw=0.8, ls=":")
            if anyc.any():
                ax.set_xlim(phi[anyc].min() - 15, phi[anyc].max() + 15)
            th = z["theta"][i]
            ax.set_title(f"q={th[2]:.2f}  log M={th[0]:.2f}  t={th[12]:.1f} Gyr", fontsize=8.5)
            axr.set_xlabel(r"$\phi$ [deg]")
            axr.set_ylim(-6, 6)
        axes[0, 0].set_ylabel("mean radius per bin [kpc]")
        axes[1, 0].set_ylabel(r"(this $-$ StreaMAX) / $\sigma$")
        axes[0, 0].legend(loc="best", fontsize=7.5)
        fig.suptitle("Binned tracks, bins with >= 100 particles: this code against StreaMAX for the same parameters "
                     "(independent spray realisations; bars = error of the mean)", fontsize=10)
        fig.tight_layout()
        save(fig, d, "streamax_comparison")
    p = tdir / "energy.npz"
    if p.exists():
        z = np.load(p)
        keys = sorted((k for k in z.files if k.startswith("dE_")), key=lambda k: int(k[3:]))
        bound = z["E0"] < 0
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)
        bins = np.linspace(-12, 2, 43)
        for ax, m, title in ((axes[0], bound, f"bound draws (n = {int(bound.sum())})"),
                             (axes[1], ~bound, f"unbound draws (n = {int((~bound).sum())})")):
            for k, col in zip(keys, (C_ALL, C_BOUND, C_PASS)):
                x = np.log10(np.clip(z[k][m], 1e-12, 1e2))
                step_hist(ax, x, bins, col, label=f"{k[3:]} steps", density=False)
            ax.set_xlabel(r"$\log_{10}\,|E(-t_{\rm int}) - E(0)|\,/\,|E(0)|$ of the progenitor orbit")
            ax.set_title(title)
        axes[0].set_ylabel("prior draws")
        axes[0].legend(loc="upper left")
        fig.suptitle("Leapfrog energy conservation over the backward integration", fontsize=10.5)
        fig.tight_layout()
        save(fig, d, "energy_conservation")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="outputs/extragalactic/C26_test")
    ap.add_argument("--subdir", default="")
    ap.add_argument("--only", default="", help="comma list of: marginals, energy, binned, gallery, noise, tests")
    ap.add_argument("--seed", type=int, default=0, help="selection of the gallery streams")
    ap.add_argument("--noise", type=float, default=0.02)
    a = ap.parse_args()
    d = Data(REPO / a.out, a.subdir)
    figs = dict(marginals=lambda: fig_marginals(d), energy=lambda: fig_energy(d), binned=lambda: fig_binned(d),
                gallery=lambda: fig_gallery(d, a.seed), noise=lambda: fig_noise(d, a.seed, a.noise),
                tests=lambda: fig_tests(d))
    for name in [s for s in a.only.split(",") if s] or list(figs):
        figs[name]()


if __name__ == "__main__":
    main()

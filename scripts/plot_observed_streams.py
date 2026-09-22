#!/usr/bin/env python
"""Plot the three OBSERVED streams (real Gaia members only) in pure observation space.

One column per stream, one row per observable, everything plotted against ``--x`` (alpha by
default): neither axis depends on a great-circle frame fit, so this is the rawest view of the
catalogue and is directly comparable with any published sky figure.

Standalone by design: numpy + matplotlib only, no hydrabflow / agama / JAX import, so it starts in
under a second and can be re-run freely while trying out selection cuts.

Uncertainties
-------------
``--errors table`` (the DEFAULT) draws 1-sigma error bars. The npz stores a REAL per-star v_los
uncertainty (``vlos_error``), which is used as-is; the astrometric uncertainties are NOT stored, so
they are reconstructed the way the pipeline's observation model does it — the Gaia DR3
median-uncertainty-vs-G table (``assets/gaia/gaia_DR3_erorr_6D.txt``) linearly interpolated at each
star's stored G magnitude (``augmentation/streams.py::sample_obs_error``). They are therefore
TYPICAL errors for a star of that brightness, not the catalogue's own per-star values. sigma_ra and
sigma_dec are in mas (~0.01-1 mas), i.e. invisible on a degree-scale axis, by construction.
``--errors none`` turns the bars off, ``--errors vlos`` keeps only the real v_los ones.

Cuts
----
``--cut EXPR`` keeps the stars for which EXPR is true; repeat it to AND several together. The
expression is ordinary numpy over the per-star columns

  ra, dec, parallax, pmra, pmdec, vlos      the six catalogue observables
  g                                         Gaia G magnitude
  vlos_err                                  quoted v_los error (0 where unmeasured)
  has_vlos                                  bool, True for the stars with a measured v_los
  pm                                        sqrt(pmra^2 + pmdec^2)

Prefix a cut with ``<stream>:`` to apply it to that stream only, e.g.

  --cut "dec > -40" --cut "NGC3201: ra < 120" --cut "M68: has_vlos"

By default the rejected stars stay on the figure in light grey so the cut is visible; pass
``--hide-cut`` to drop them. Kept / rejected counts per stream are printed and written to
``<out>.json`` next to the figure.

Usage
-----
  .venv/bin/python scripts/plot_observed_streams.py --out data_local/observed_streams.png
  .venv/bin/python scripts/plot_observed_streams.py --cut "has_vlos" --out /tmp/vlos_only.png
  .venv/bin/python scripts/plot_observed_streams.py --x dec --y parallax,vlos --out /tmp/two.png
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_REAL = "assets/gaia/gaia_observed_streams_6Dwitherrors_cutNGC3201_desi_m68palau_main.npz"
NAMES = {0: "Pal5", 1: "NGC3201", 2: "M68"}
# sim_data_projected columns of the REAL npz (channel 2 is a parallax here, not a distance).
COLS = ["ra", "dec", "parallax", "pmra", "pmdec", "vlos"]
LABEL = {
    "ra": "alpha (RA) [deg]",
    "dec": "delta (Dec) [deg]",
    "parallax": "parallax [mas]",
    "pmra": "mu_alpha* [mas/yr]",
    "pmdec": "mu_delta [mas/yr]",
    "vlos": "v_los [km/s]",
    "g": "G [mag]",
    "vlos_err": "sigma_vlos [km/s]",
    "pm": "|mu| [mas/yr]",
}
COLOR = {0: "tab:blue", 1: "tab:orange", 2: "tab:green"}


def load_streams(path):
    """{j: {column: (N,) array}} of the attended real members of each stream."""
    d = np.load(path)
    sd = d["sim_data_projected"]
    sd = sd[0] if sd.ndim == 4 else sd                      # (3, P, 6)
    am = d["attention_mask"]
    am = am[:, 0, :] if am.ndim == 3 else am                # (3, P)
    js = np.asarray(d["j"]).reshape(-1).astype(int)
    mag = d["magnitudes"] if "magnitudes" in d.files else None
    verr = d["vlos_error"] if "vlos_error" in d.files else None
    vmask = d["vlos_mask"] if "vlos_mask" in d.files else None
    out = {}
    for i, j in enumerate(js):
        keep = am[i].astype(bool)
        s = {c: sd[i][keep, k] for k, c in enumerate(COLS)}
        s["g"] = mag[i][keep] if mag is not None else np.full(keep.sum(), np.nan)
        s["vlos_err"] = verr[i][keep] if verr is not None else np.zeros(keep.sum())
        vm = vmask[i] if vmask is not None else np.ones_like(am[i])
        vm = vm[:, 0] if vm.ndim == 2 else vm
        s["has_vlos"] = vm[keep].astype(bool)
        s["pm"] = np.hypot(s["pmra"], s["pmdec"])
        oe = d["obs_error"] if "obs_error" in d.files else None
        if oe is not None:                    # per-star catalogue uncertainties, same column order
            oe = oe[0] if oe.ndim == 4 else oe
            s["_err"] = {c: oe[i][keep, kk] for kk, c in enumerate(COLS)}
        out[int(j)] = s
    return out


# Gaia DR3 median uncertainty vs G, keyed by the npz column names (the table's own row labels are
# the pipeline's, `mu_ra`/`mu_dec`/`v_los`).
ERROR_TABLE = "assets/gaia/gaia_DR3_erorr_6D.txt"
ERROR_ROW = {"ra": "ra", "dec": "dec", "parallax": "parallax",
             "pmra": "mu_ra", "pmdec": "mu_dec", "vlos": "v_los"}
# sigma_ra / sigma_dec are tabulated in mas but the positions are plotted in degrees.
ERROR_SCALE = {"ra": 1.0 / 3.6e6, "dec": 1.0 / 3.6e6}


def load_error_table(path=ERROR_TABLE):
    """{row label: (mag_bins, sigma)} from the tab-separated Gaia DR3 uncertainty table."""
    with open(path) as fh:
        lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
    head = [c.strip().strip('"') for c in lines[0].split("\t")]
    mags = np.array([float(c) for c in head[1:-1]])       # first col = label, last = unit
    out = {}
    for ln in lines[1:]:
        f = [c.strip().strip('"') for c in ln.split("\t")]
        out[f[0]] = (mags, np.array([float(v) for v in f[1:1 + len(mags)]]))
    return out


def star_sigmas(star, mode, table_path=ERROR_TABLE):
    """{column: (N,) 1-sigma} for the plotted columns; empty dict when ``mode == 'none'``."""
    if mode == "none":
        return {}
    sig = {}
    if mode in ("real", "auto") and "_err" in star:
        # The file carries the catalogue's own per-star uncertainties (--astrometry dr3).
        sig = {c: v.copy() for c, v in star["_err"].items() if np.isfinite(v).any()}
        sig["pm"] = np.hypot(sig.get("pmra", 0.0), sig.get("pmdec", 0.0))
        if "vlos" in sig:
            sig["vlos"] = np.where(star["has_vlos"], sig["vlos"], np.nan)
        return sig
    if mode == "real":
        raise SystemExit("--errors real: this file has no obs_error array")
    if mode == "auto":
        mode = "table"
    if mode == "table":
        tbl = load_error_table(table_path)
        for col, row in ERROR_ROW.items():
            if row in tbl and col != "vlos":
                mags, vals = tbl[row]
                sig[col] = np.interp(star["g"], mags, vals) * ERROR_SCALE.get(col, 1.0)
        sig["pm"] = np.hypot(sig.get("pmra", 0.0), sig.get("pmdec", 0.0))
    # v_los: the npz carries the instrument's own per-star error, so prefer it over the table.
    ve = np.where(star["has_vlos"], star["vlos_err"], np.nan)
    if np.isfinite(ve).any():
        sig["vlos"] = ve
    elif mode == "table":
        mags, vals = load_error_table(table_path)["v_los"]
        sig["vlos"] = np.interp(star["g"], mags, vals)
    return sig


def parse_cuts(raw):
    """['dec > -40', 'M68: has_vlos'] -> {None: [...global...], 'M68': [...]} ."""
    by_stream = {}
    for c in raw or []:
        name, _, expr = c.partition(":")
        if expr and name.strip() in NAMES.values():
            by_stream.setdefault(name.strip(), []).append(expr.strip())
        else:
            by_stream.setdefault(None, []).append(c.strip())
    return by_stream


def apply_cuts(star, cuts, name):
    """Boolean keep-mask and the per-expression counts, for one stream."""
    keep = np.ones(len(star["ra"]), bool)
    per_cut = {}
    env = {**star, "np": np, **{k: getattr(np, k) for k in ("abs", "log10", "hypot", "sqrt")}}
    for expr in list(cuts.get(None, [])) + list(cuts.get(name, [])):
        m = np.asarray(eval(expr, {"__builtins__": {}}, env), bool)  # noqa: S307 (user's own cut)
        if m.shape != keep.shape:
            raise SystemExit(f"cut {expr!r} returned shape {m.shape}, expected {keep.shape}")
        per_cut[expr] = int(m.sum())
        keep &= m
    return keep, per_cut


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", default=DEFAULT_REAL, help="real member npz")
    ap.add_argument("--out", required=True, help="output .png")
    ap.add_argument("--x", default="ra", help="abscissa column (default: ra)")
    ap.add_argument("--y", default="dec,parallax,pmra,pmdec,vlos",
                    help="comma-separated ordinate columns, one row each")
    ap.add_argument("--cut", action="append", default=[],
                    help="keep stars where EXPR is true; repeatable; '<stream>: EXPR' scopes it")
    ap.add_argument("--streams", default=",".join(NAMES.values()),
                    help="comma-separated subset of the streams to plot")
    ap.add_argument("--errors", choices=("auto", "table", "real", "vlos", "none"), default="auto",
                    help="1-sigma error bars: 'auto' (default) = the file's own per-star "
                         "`obs_error` when it has one, else 'table'; 'real' = require it; "
                         "'table' = real v_los errors + astrometric "
                         "errors interpolated from the Gaia DR3 magnitude table; 'vlos' = the real "
                         "v_los errors only; 'none' = no bars")
    ap.add_argument("--error-table", default=ERROR_TABLE,
                    help="Gaia DR3 median-uncertainty-vs-G table used by --errors table")
    ap.add_argument("--hide-cut", action="store_true",
                    help="drop the rejected stars instead of drawing them in grey")
    ap.add_argument("--vlos-measured-only", dest="vlos_measured", action="store_true",
                    default=True, help="(default) show only measured stars in v_los panels")
    ap.add_argument("--vlos-all", dest="vlos_measured", action="store_false",
                    help="also show the imputed v_los values")
    ap.add_argument("--share-x", action="store_true",
                    help="one common x range for all three streams (default: per stream)")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    stars = load_streams(args.real)
    want = [s.strip() for s in args.streams.split(",") if s.strip()]
    js = [j for j in sorted(NAMES) if NAMES[j] in want]
    if not js:
        raise SystemExit(f"--streams matched nothing; choose from {list(NAMES.values())}")
    rows = [c.strip() for c in args.y.split(",") if c.strip()]
    for c in [args.x, *rows]:
        if c not in LABEL:
            raise SystemExit(f"unknown column {c!r}; choose from {sorted(LABEL)}")
    cuts = parse_cuts(args.cut)

    report = {"real": args.real, "x": args.x, "y": rows, "cuts": args.cut,
              "errors": args.errors, "streams": {}}
    keeps, sig = {}, {}
    for j in js:
        keep, per_cut = apply_cuts(stars[j], cuts, NAMES[j])
        keeps[j] = keep
        sig[j] = star_sigmas(stars[j], args.errors, args.error_table)
        s = stars[j]
        report["streams"][NAMES[j]] = {
            "n_total": int(len(s["ra"])), "n_kept": int(keep.sum()),
            "n_rejected": int((~keep).sum()),
            "n_kept_with_vlos": int((keep & s["has_vlos"]).sum()),
            "per_cut_kept": per_cut,
            "median_sigma_kept": {k: float(np.nanmedian(v[keep])) for k, v in sig[j].items()
                                  if keep.any() and np.isfinite(v[keep]).any()},
            "ra_range_kept": [float(s["ra"][keep].min()), float(s["ra"][keep].max())]
            if keep.any() else None,
        }

    fig, axes = plt.subplots(len(rows), len(js), figsize=(5.4 * len(js), 2.9 * len(rows)),
                             sharex="all" if args.share_x else "col", squeeze=False)
    for col, j in enumerate(js):
        s, keep = stars[j], keeps[j]
        for row, key in enumerate(rows):
            ax = axes[row][col]
            sel = keep.copy()
            drop = ~keep
            if key == "vlos" and args.vlos_measured:
                sel &= s["has_vlos"]
                drop &= s["has_vlos"]
            if not args.hide_cut and drop.any():
                ax.scatter(s[args.x][drop], s[key][drop], s=12, facecolors="none",
                           edgecolors="0.75", lw=0.6, label=f"cut out ({drop.sum()})")
            ax.scatter(s[args.x][sel], s[key][sel], s=13, c=COLOR[j], lw=0, zorder=3,
                       label=f"kept ({sel.sum()})")
            ey, ex = sig[j].get(key), sig[j].get(args.x)
            if sel.any() and (ey is not None or ex is not None):
                ax.errorbar(s[args.x][sel], s[key][sel],
                            yerr=None if ey is None else ey[sel],
                            xerr=None if ex is None else ex[sel],
                            fmt="none", ecolor=COLOR[j], elinewidth=0.7, alpha=0.55,
                            capsize=0, zorder=2)
            ax.set_ylabel(LABEL[key])
            ax.legend(fontsize=6, loc="best")
            if row == 0:
                ax.set_title(f"{NAMES[j]}  —  {keep.sum()}/{len(keep)} stars")
            if row == len(rows) - 1:
                ax.set_xlabel(LABEL[args.x])
    title = "Observed stream members in observation space"
    if args.errors != "none":
        title += "   (1-sigma bars)"
    if args.cut:
        title += "   |   cuts: " + "; ".join(args.cut)
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    with open(os.path.splitext(args.out)[0] + ".json", "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report["streams"], indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

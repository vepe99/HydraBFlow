#!/usr/bin/env python
"""Prior-predictive check in observation space, in the PUBLISHED STREAMFINDER stream frames.

Every other ppc_* script in this repo projects into a great-circle (phi1, phi2) frame FITTED to
whichever member catalogue is loaded (``ppc_summary_statistics.fit_frame``). That frame is a derived
quantity: its pole, handedness and zero-point all move with the catalogue, so a "misspecification"
seen there can be a property of the frame rather than of the model.

Here the frame is instead read off Ibata et al. (2024) Table 3 — the zero-point in R.A. and the pole
of the coordinate system the STREAMFINDER atlas itself used (``scripts/streamfinder_frame.py``;
Pal5 = Pal-5, NGC3201 = Gjoll, M68 = Fjorm). It is an external, fixed definition: nothing about it
depends on which members are loaded or on the simulation, so sim and real are measured with the
identical ruler and the numbers are comparable with the literature.

Observables (``--frame icrs`` keeps the raw catalogue columns instead):

    phi1 [deg]  phi2 [deg]  parallax [mas]  mu_phi1 [mas/yr]  mu_phi2 [mas/yr]  v_los [km/s]

Both sides go through their real pipelines — simulations through the training observation model up
to ``mask_vlos`` (window, member count, magnitudes, per-star errors, v_los selection), real members
through the real preset's ``sample_obs_error`` + ``impute_vlos`` — so the comparison is between what
the network is trained on and what it is shown.

Four views, in increasing order of how much structure they assume:

1. ``mmd``      model-free RBF-MMD coverage rank per observable subset. No binning, no frame, no
                abscissa. real-vs-sim MMD^2 against the sim-vs-sim null, as a percentile:
                100 = the real stream is farther from every simulation than simulations are from
                each other. This is the headline number.
2. ``quantile`` per-observable quantile envelopes (marginals only).
3. ``joint``    2-D overlays: the sky footprint, the proper-motion plane, v_los and parallax vs RA.
4. ``track``    per-phi1-bin median and robust dispersion of the other five observables. The bins
                are equal-count quantiles of the real members, so every bin is populated on the real
                side by construction; a simulated bin below ``min_count`` reads NaN, never a
                substituted 0 (the 2026-07-28 occupancy bias).

Caveat that survives the frame change: M68/Fjorm is not a great-circle stream in ANY frame (its
reference track has ~4 deg of phi2 rms even in Ibata's own), so for M68 phi2 mixes along-track and
across-track structure and is not the clean width coordinate it is for Pal5.

    .venv/bin/python scripts/ppc_observation_space.py \
        --sim <dataset>/test_multistream_333.npz --out <dataset>/ppc/observation_space \
        --simulator stream_trihedron_mcmillan17_v4 --frame streamfinder \
        --aug stream_global_v5 --real-aug stream_real_global_v5 \
        --real assets/gaia/gaia_observed_streams_6Dwitherrors_cutNGC3201_desi_m68palau_main.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

os.environ.setdefault("HYDRABFLOW_NUM_GPUS", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("HYDRABFLOW_SIM_QUIET", "1")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ppc_particle_coverage import (  # noqa: E402
    C_BAND,
    C_INK,
    C_REAL,
    C_SIM,
    QS,
    compose_aug,
    features,
    mmd2,
    real_clouds,
)
from ppc_summary_statistics import NAMES, augment_sim, fit_frame, project  # noqa: E402
from streamfinder_frame import frames as streamfinder_frames  # noqa: E402
from streamfinder_frame import mixed_frames  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Channel 2 is a heliocentric distance in the raw simulation and a parallax after
# `convert_distance_to_parallax`, which the training chain applies first — so by the time we see it
# both sides are parallaxes. (Reading `sim_data_projected` raw instead would compare kpc against mas.)
OBS_STREAM = ["phi1", "phi2", "parallax", "mu_phi1", "mu_phi2", "v_los"]
OBS_ICRS = ["ra", "dec", "parallax", "mu_ra_cosdec", "mu_dec", "v_los"]
UNITS = ["deg", "deg", "mas", "mas/yr", "mas/yr", "km/s"]
LABEL_STREAM = [r"$\phi_1$", r"$\phi_2$", r"$\varpi$", r"$\mu_{\phi_1}$", r"$\mu_{\phi_2}$", r"$v_{los}$"]
LABEL_ICRS = [r"$\alpha$", r"$\delta$", r"$\varpi$", r"$\mu_{\alpha*}$", r"$\mu_\delta$", r"$v_{los}$"]
SUBSETS = {"all": [0, 1, 2, 3, 4, 5], "sky": [0, 1], "pm": [3, 4], "vlos": [5], "parallax": [2]}
# 2-D views: (x, y) column pairs. The stream plane first, then the pm plane, then the two views
# against the along-track coordinate.
PAIRS = [(0, 1), (3, 4), (0, 5), (0, 2)]

# Filled by main() once --frame is known, so the figure helpers need no extra argument.
OBS, LABEL = OBS_STREAM, LABEL_STREAM


def stream_frames(kind: str, real_path: str) -> dict[int, np.ndarray] | None:
    """Rotation matrix per stream index, or ``None`` for the raw catalogue view.

    ``streamfinder`` and ``palau`` are PUBLISHED frames — fixed external definitions that do not
    move with the member catalogue or the simulation. ``fit`` reproduces what the other ppc_*
    scripts do (a great circle fitted to the loaded members) and is kept only for comparison.
    """
    if kind == "icrs":
        return None
    if kind == "streamfinder":
        by_name = streamfinder_frames()
    elif kind == "palau":
        by_name = mixed_frames()
    elif kind == "fit":
        d = np.load(real_path)
        s, jj = np.asarray(d["sim_data_projected"]), np.asarray(d["j"]).reshape(-1).astype(int)
        out = {}
        for row, j in enumerate(jj):
            st = s[0, row] if s.ndim == 4 else s[row]
            att = np.asarray(d["attention_mask"])
            a = (att[0, row] if att.ndim == 3 else att[row]).astype(bool)
            out[int(j)] = fit_frame(st[a][:, 0], st[a][:, 1])
        return out
    else:
        raise ValueError(f"unknown frame {kind!r}")
    name_to_j = {v: k for k, v in NAMES.items()}
    return {name_to_j[n]: R for n, R in by_name.items() if n in name_to_j}


def to_obs(R, stars: np.ndarray) -> np.ndarray:
    """(N,6) catalogue table -> the six analysis observables, projected when a frame is given."""
    stars = np.asarray(stars, dtype=float)[:, :6]
    if R is None:
        return stars
    phi1, phi2, mu1, mu2 = project(R, stars[:, 0], stars[:, 1], stars[:, 3], stars[:, 4])
    return np.column_stack([phi1, phi2, stars[:, 2], mu1, mu2, stars[:, 5]])


def binned(x, y, edges, f, min_count=3):
    """``f`` of ``y`` per ``x`` bin; NaN where the bin holds fewer than ``min_count`` finite values."""
    out = np.full(len(edges) - 1, np.nan)
    ok = np.isfinite(x) & np.isfinite(y)
    idx = np.digitize(x[ok], edges) - 1
    for b in range(len(edges) - 1):
        v = y[ok][idx == b]
        if v.size >= min_count:
            out[b] = f(v)
    return out


def mad_std(v):
    return 1.4826 * np.median(np.abs(v - np.median(v)))


def _nanmedian(a, axis=0):
    """``np.nanmedian`` without the all-NaN-slice warning: an empty bin legitimately has no median."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(a, axis=axis)


def _nanpercentile(a, q, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanpercentile(a, q, axis=axis)


def read_rows(npz_path, key, idx):
    """Rows ``idx`` of one array inside an uncompressed .npz, without loading the whole thing.

    ``np.savez`` stores each member as a STORED (uncompressed) zip entry, so its data is a flat
    C-order block and a row can be seek()ed to directly. The v4 training sets are ~5 GB and this box
    runs strict overcommit, so reading 180 rows must not cost 5 GB of commit charge. Falls back to a
    plain load for a compressed member.
    """
    import zipfile  # noqa: PLC0415

    from numpy.lib import format as npf  # noqa: PLC0415

    idx = np.asarray(idx, dtype=np.int64)
    with zipfile.ZipFile(npz_path) as z:
        name = key + ".npy"
        if z.getinfo(name).compress_type != zipfile.ZIP_STORED:
            return np.load(npz_path)[key][idx]
        with z.open(name) as fh:
            ver = npf.read_magic(fh)
            reader = npf.read_array_header_1_0 if ver == (1, 0) else npf.read_array_header_2_0
            shape, fortran, dtype = reader(fh)
            if fortran:
                raise SystemExit(f"{key}: Fortran-order member, cannot row-slice")
            start, nbytes = fh.tell(), int(np.prod(shape[1:])) * dtype.itemsize
            out = np.empty((len(idx), *shape[1:]), dtype=dtype)
            for t, i in enumerate(idx):
                fh.seek(start + int(i) * nbytes)
                out[t] = np.frombuffer(fh.read(nbytes), dtype=dtype).reshape(shape[1:])
    return out


def load_sim_groups(path, n_sim, rng):
    """(sim, j) as grouped arrays (G, S, P, 6) / (G, S), from a grouped OR a flat dataset.

    A flat training set has one stream per row, so ``G`` pseudo-groups are formed by drawing ``G``
    rows of each stream independently. Rows of a group then share no potential — which is exactly
    right here, since every stream is compared against its own simulated ensemble and the MMD never
    couples them.
    """
    with np.load(path) as d:
        j_raw = np.asarray(d["j"])
        shape = d["sim_data_projected"].shape if "sim_data_projected" in d.files else None
    if shape is None:
        raise SystemExit("--sim has no sim_data_projected")
    if len(shape) == 4:                                   # grouped multistream npz
        n = shape[0]
        pick = np.sort(rng.choice(n, size=min(n_sim, n), replace=False))
        with np.load(path) as d:
            return np.asarray(d["sim_data_projected"])[pick], j_raw.reshape(n, -1)[pick].astype(int)
    if len(shape) != 3:
        raise SystemExit(f"--sim must be (N,S,P,6) or flat (N,P,6); got {shape}")
    jf = j_raw.reshape(-1).astype(int)
    streams = sorted(set(jf.tolist()))
    per = [rng.choice(np.flatnonzero(jf == s), size=min(n_sim, int((jf == s).sum())),
                      replace=False) for s in streams]
    g = min(len(p) for p in per)
    rows = np.concatenate([np.sort(p[:g]) for p in per])
    flat = read_rows(path, "sim_data_projected", rows)
    sim = np.stack([flat[k * g:(k + 1) * g] for k in range(len(streams))], axis=1)
    jj = np.tile(np.array(streams, dtype=int), (g, 1))
    print(f"[sim] flat dataset: {g} pseudo-groups x {len(streams)} streams "
          f"({len(rows)} rows read of {len(jf)})")
    return sim, jj


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", required=True,
                    help="grouped multistream npz (N,S,P,6) or a flat training set (N,P,6), whose "
                         "rows are then drawn per stream into pseudo-groups")
    ap.add_argument("--out", required=True)
    ap.add_argument("--real",
                    default="assets/gaia/gaia_observed_streams_6Dwitherrors_cutNGC3201_desi_m68palau_main.npz")
    ap.add_argument("--simulator", default="stream_trihedron_mcmillan17_v4")
    ap.add_argument("--aug", default="stream_global_v5")
    ap.add_argument("--real-aug", default="stream_real_global_v5")
    ap.add_argument("--n-sim", type=int, default=60, help="simulated groups used (random subset)")
    ap.add_argument("--n-null", type=int, default=300, help="sim-vs-sim MMD pairs")
    ap.add_argument("--frame", choices=("streamfinder", "palau", "fit", "icrs"),
                    default="streamfinder",
                    help="stream frame: published Ibata+2024 Table 3 poles (default), Palau's M68 "
                         "frame with STREAMFINDER for the other two, a great circle fitted to the "
                         "loaded members, or no frame at all (raw catalogue columns)")
    ap.add_argument("--bins", type=int, default=8, help="along-track bins for the track view")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    global OBS, LABEL
    OBS = OBS_ICRS if args.frame == "icrs" else OBS_STREAM
    LABEL = LABEL_ICRS if args.frame == "icrs" else LABEL_STREAM
    R_of = stream_frames(args.frame, args.real)
    print(f"frame: {args.frame}"
          + ("" if R_of is None else "  poles " + ", ".join(
              f"{NAMES.get(j, j)}=({np.degrees(np.arctan2(R[2][1], R[2][0])) % 360:.2f}, "
              f"{np.degrees(np.arcsin(np.clip(R[2][2], -1, 1))):.2f})" for j, R in sorted(R_of.items()))))

    x, jj = load_sim_groups(args.sim, args.n_sim, rng)
    pick = np.arange(len(x))
    sim, attn, vmask = augment_sim(x, jj, aug_preset=args.aug, simulator=args.simulator,
                                   seed=args.seed)
    aug = compose_aug(args.simulator, args.aug)
    real = real_clouds(args.real, args.simulator, args.real_aug,
                       int(aug.params.get("max_particles", 300)), args.seed)

    report = {"sim": args.sim, "real": args.real, "frame": args.frame,
              "observables": OBS, "n_groups_used": int(len(pick)), "streams": {}}
    for j, name in NAMES.items():
        if j not in real:
            continue
        r_stars, r_vm = real[j]
        R = None if R_of is None else R_of.get(j)
        if R_of is not None and R is None:
            print(f"{NAMES[j]}: no published frame — skipped")
            continue
        Fr = to_obs(R, r_stars)
        clouds = []
        for g in range(len(pick)):
            hit = np.flatnonzero(jj[g] == j)
            if not len(hit):
                continue
            s = int(hit[0])
            a = attn[g, s].astype(bool)
            if a.sum() < 20:
                continue
            clouds.append((to_obs(R, sim[g, s][a]), vmask[g, s][a].astype(bool)))
        if len(clouds) < 5:
            print(f"{name}: only {len(clouds)} usable simulated clouds — skipped")
            continue

        # --- 1. quantile envelopes (marginals)
        qr = np.full((6, len(QS)), np.nan)
        qs = np.full((len(clouds), 6, len(QS)), np.nan)
        for c in range(6):
            v = Fr[:, c] if c != 5 else Fr[r_vm, c]
            v = v[np.isfinite(v)]
            if v.size > 3:
                qr[c] = np.quantile(v, QS)
            for gi, (F, vm) in enumerate(clouds):
                v = F[:, c] if c != 5 else F[vm, c]
                v = v[np.isfinite(v)]
                if v.size > 3:
                    qs[gi, c] = np.quantile(v, QS)
        qlo, qmed, qhi = np.nanpercentile(qs, [5, 50, 95], axis=0)
        inside = {OBS[c]: float(np.mean((qr[c] >= qlo[c]) & (qr[c] <= qhi[c]))) for c in range(6)}

        # --- 2. MMD coverage per observable subset (the model-free headline)
        mm = {}
        for sub, cols in SUBSETS.items():
            Xr = features(Fr, r_vm, cols)
            if len(Xr) < 10:
                continue
            mu = np.median(Xr, 0)
            sc = 1.4826 * np.median(np.abs(Xr - mu), 0) + 1e-9
            Zr = (Xr - mu) / sc
            d2 = np.sum((Zr[:, None] - Zr[None]) ** 2, -1)
            gamma = 1.0 / (2 * np.median(d2[d2 > 0]))
            Zs = [(features(F, vm, cols) - mu) / sc for F, vm in clouds]
            Zs = [Z for Z in Zs if len(Z) >= 10]
            if len(Zs) < 5:
                continue
            d_real = np.array([mmd2(Zr, Z, gamma) for Z in Zs])
            pairs = rng.integers(0, len(Zs), size=(args.n_null, 2))
            pairs = pairs[pairs[:, 0] != pairs[:, 1]]
            d_null = np.array([mmd2(Zs[a], Zs[b], gamma) for a, b in pairs])
            mm[sub] = dict(
                real_vs_sim_median=float(np.median(d_real)),
                sim_vs_sim_median=float(np.median(d_null)),
                sim_vs_sim_p95=float(np.percentile(d_null, 95)),
                percentile=float(100 * np.mean(d_null <= np.median(d_real))),
                frac_sims_farther_than_typical=float(np.mean(d_real > np.percentile(d_null, 95))),
            )

        # --- 4. Binned tracks along the frame's first coordinate (phi1, or RA under --frame icrs).
        #        Equal-count bins of the REAL members, so every bin is populated on the real side by
        #        construction; a sim bin below min_count reads NaN, never a substituted 0.
        edges = np.quantile(Fr[np.isfinite(Fr[:, 0]), 0], np.linspace(0, 1, args.bins + 1))
        edges[0] -= 1e-6
        edges[-1] += 1e-6
        tracks = {}
        for c in range(1, 6):
            sel_r = r_vm if c == 5 else np.ones(len(Fr), bool)
            med_r = binned(Fr[sel_r, 0], Fr[sel_r, c], edges, np.median)
            std_r = binned(Fr[sel_r, 0], Fr[sel_r, c], edges, mad_std)
            med_s = np.array([binned(F[vm if c == 5 else slice(None), 0],
                                     F[vm if c == 5 else slice(None), c], edges, np.median)
                              for F, vm in clouds])
            std_s = np.array([binned(F[vm if c == 5 else slice(None), 0],
                                     F[vm if c == 5 else slice(None), c], edges, mad_std)
                              for F, vm in clouds])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                pct_med = np.array([100 * np.nanmean(med_s[:, b] <= med_r[b]) for b in range(args.bins)])
                pct_std = np.array([100 * np.nanmean(std_s[:, b] <= std_r[b]) for b in range(args.bins)])
            tracks[OBS[c]] = dict(real_median=med_r, real_std=std_r, sim_median=med_s,
                                  sim_std=std_s, pct_median=pct_med, pct_std=pct_std)

        report["streams"][name] = dict(
            n_real=int(len(Fr)), n_real_vlos=int(r_vm.sum()), n_sim_clouds=len(clouds),
            quantile_inside_5_95=inside, mmd=mm,
            bin_edges=edges.tolist(),
            # A bin below min_count on every simulation gives an all-NaN column here. That is the
            # honest answer (M68 has 16 measured v_los across 8 bins), not a defect, so the warning
            # numpy raises for it is suppressed rather than worked around.
            track={k: {"real_median": v["real_median"].tolist(), "real_std": v["real_std"].tolist(),
                       "sim_median_p50": _nanmedian(v["sim_median"]).tolist(),
                       "sim_std_p50": _nanmedian(v["sim_std"]).tolist(),
                       "pct_real_median_in_sims": v["pct_median"].tolist(),
                       "pct_real_std_in_sims": v["pct_std"].tolist()}
                   for k, v in tracks.items()},
        )
        print(f"{name:8s} real {len(Fr)} stars ({int(r_vm.sum())} vlos) vs {len(clouds)} sims | "
              "MMD pct: " + " ".join(f"{k}={v['percentile']:.0f}" for k, v in mm.items()))

        fig_marginals(name, qr, qlo, qmed, qhi, inside, Fr, r_vm, clouds,
                      os.path.join(args.out, f"obs_marginals_{name}.png"), args.title)
        fig_tracks(name, edges, tracks, os.path.join(args.out, f"obs_tracks_{name}.png"), args.title)

    fig_mmd(report, os.path.join(args.out, "obs_mmd.png"), args.title)
    with open(os.path.join(args.out, "obs_coverage.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nwrote {args.out}/obs_coverage.json")


def fig_marginals(name, qr, qlo, qmed, qhi, inside, Fr, r_vm, clouds, path, title) -> None:
    """Row 1: per-observable quantile envelopes. Row 2: 2-D overlays in catalogue coordinates."""
    fig, axes = plt.subplots(2, 6, figsize=(22, 7.5))
    for c in range(6):
        ax = axes[0, c]
        ax.fill_between(QS, qlo[c], qhi[c], color=C_BAND, lw=0, label="sims 5-95 %")
        ax.plot(QS, qmed[c], color=C_SIM, lw=2, label="sims median")
        ax.plot(QS, qr[c], color=C_REAL, lw=2, marker="o", ms=4, label="real Gaia")
        ax.set_xlabel("quantile")
        ax.set_ylabel(f"{LABEL[c]} [{UNITS[c]}]")
        ax.set_title(f"{OBS[c]}  inside {inside[OBS[c]]:.0%}", fontsize=9, color=C_INK)
        if c == 0:
            ax.legend(fontsize=7, frameon=False)
    for k, (cx, cy) in enumerate(PAIRS):
        ax = axes[1, k]
        for F, vm in clouds[:20]:
            sel = (vm if 5 in (cx, cy) else np.ones(len(F), bool)) & np.isfinite(F[:, [cx, cy]]).all(1)
            ax.plot(F[sel, cx], F[sel, cy], ".", ms=1.2, alpha=0.10, color=C_SIM, rasterized=True)
        sel = (r_vm if 5 in (cx, cy) else np.ones(len(Fr), bool)) & np.isfinite(Fr[:, [cx, cy]]).all(1)
        ax.plot(Fr[sel, cx], Fr[sel, cy], ".", ms=3, color=C_REAL)
        ax.set_xlabel(f"{LABEL[cx]} [{UNITS[cx]}]")
        ax.set_ylabel(f"{LABEL[cy]} [{UNITS[cy]}]")
    for k in range(len(PAIRS), 6):
        axes[1, k].axis("off")
    fig.suptitle(f"{name}: observation-space coverage{(' — ' + title) if title else ''} "
                 "(orange = real Gaia, blue = simulations)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_tracks(name, edges, tracks, path, title) -> None:
    """Median and robust dispersion of each observable per RA bin — a catalogue-coordinate abscissa,
    so nothing here depends on a fitted stream frame."""
    keys = [k for k in OBS[1:] if k in tracks]
    ctr = 0.5 * (edges[:-1] + edges[1:])
    fig, axes = plt.subplots(2, len(keys), figsize=(3.4 * len(keys), 6.2), squeeze=False)
    for i, k in enumerate(keys):
        t = tracks[k]
        for row, (sim_arr, real_arr, lab) in enumerate((
            (t["sim_median"], t["real_median"], "median"),
            (t["sim_std"], t["real_std"], "robust dispersion"),
        )):
            ax = axes[row, i]
            lo, med, hi = _nanpercentile(sim_arr, [5, 50, 95])
            ax.fill_between(ctr, lo, hi, color=C_BAND, lw=0, label="sims 5-95 %")
            ax.plot(ctr, med, color=C_SIM, lw=2, label="sims median")
            ax.plot(ctr, real_arr, color=C_REAL, lw=2, marker="o", ms=4, label="real Gaia")
            ax.set_xlabel(f"{LABEL[0]} [deg]")
            unit = UNITS[OBS.index(k)]
            ax.set_ylabel(f"{lab} {LABEL[OBS.index(k)]} [{unit}]")
            if row == 0:
                ax.set_title(k, fontsize=10, color=C_INK)
            if i == 0 and row == 0:
                ax.legend(fontsize=7, frameon=False)
    fig.suptitle(f"{name}: binned observation-space tracks{(' — ' + title) if title else ''}",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_mmd(report, path, title) -> None:
    """The headline: how far the real stream sits from the simulations, per observable subset."""
    names = list(report["streams"])
    subs = list(SUBSETS)
    if not names:
        return
    fig, ax = plt.subplots(figsize=(1.6 * len(subs) * max(len(names), 1) / 2 + 3, 3.6))
    w = 0.8 / max(len(names), 1)
    for i, name in enumerate(names):
        mm = report["streams"][name]["mmd"]
        vals = [mm.get(s, {}).get("percentile", np.nan) for s in subs]
        ax.bar(np.arange(len(subs)) + i * w, vals, width=w, label=name)
    ax.axhline(95, color=C_INK, ls="--", lw=1)
    ax.set_xticks(np.arange(len(subs)) + 0.4 - w / 2)
    ax.set_xticklabels(subs)
    ax.set_ylabel("MMD percentile vs the sim-vs-sim null")
    ax.set_ylim(0, 102)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Observation-space misspecification "
                 f"(100 = real farther from every sim){(' — ' + title) if title else ''}",
                 fontsize=10, color=C_INK)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()

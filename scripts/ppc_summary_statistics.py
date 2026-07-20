#!/usr/bin/env python
"""Prior-predictive check of the stream-frame summary statistics: overlay simulated summary tracks
on the real Gaia streams.

Standalone (numpy + matplotlib only — no BayesFlow/JAX import, like ``scripts/ppc_rotation_curve.py``,
so it runs fast and on a GPU-less box). It fits the same per-stream data-driven great-circle frame
used by ``hydrabflow.augmentation.stream_summary`` (from the REAL members), projects both the real
members and a sample of simulated streams into ``(φ1, φ2, μ_φ1, μ_φ2)``, and overlays the binned
median±std tracks so you can confirm the summaries are realistic and localize which stream/quantity
is off.

The simulated draws come from a grouped multistream npz (``sim_data_projected`` ``(N,S,P,6)``, raw
model units, pre-observation-model); this script applies the per-stream observation window +
subsample to mimic the selection the training augmentation does, then projects with the real frame.

Usage:
  uv run python scripts/ppc_summary_statistics.py \
    --sim data/.../test_multistream_333.npz --out outputs/.../ppc_summary_statistics.png
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAMES = {0: "Pal5", 1: "NGC3201", 2: "M68"}
# Observation windows (ra_min, ra_max, dec_min, dec_max) + observed member counts, matching the
# reference stream configs (conf/augmentation/stream_global.yaml).
WINDOW = {
    0: (220.98008280733634, 250.0, -13.0, 10.0),
    1: (42.29847920913508, 140.0, -47.79545565626234, 20.812029301121846),
    2: (189.73694638139526, 280.0, -47.32109546202406, 70.0),
}
NOBS = {0: 129, 1: 195, 2: 297}
CH = dict(ra=0, dec=1, mu_ra=3, mu_dec=4, vlos=5)


def unit_vec(ra, dec):
    ra, dec = np.radians(ra), np.radians(dec)
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], -1)


def fit_frame(ra, dec):
    n = unit_vec(ra, dec)
    _, v = np.linalg.eigh(n.T @ n)
    pole = v[:, 0]
    mean = n.mean(0)
    mean = mean / np.linalg.norm(mean)
    x = mean - (mean @ pole) * pole
    x = x / np.linalg.norm(x)
    return np.stack([x, np.cross(pole, x), pole], 0)


def project(R, ra, dec, mura, mudec):
    n = unit_vec(ra, dec)
    rar, decr = np.radians(ra), np.radians(dec)
    e = np.stack([-np.sin(rar), np.cos(rar), np.zeros_like(rar)], -1)
    m = np.stack([-np.sin(decr) * np.cos(rar), -np.sin(decr) * np.sin(rar), np.cos(decr)], -1)
    v = mura[:, None] * e + mudec[:, None] * m
    npr, vpr = n @ R.T, v @ R.T
    phi1 = np.degrees(np.arctan2(npr[:, 1], npr[:, 0]))
    phi2 = np.degrees(np.arcsin(np.clip(npr[:, 2], -1, 1)))
    p1, p2 = np.radians(phi1), np.radians(phi2)
    ep = np.stack([-np.sin(p1), np.cos(p1), np.zeros_like(p1)], -1)
    mp = np.stack([-np.sin(p2) * np.cos(p1), -np.sin(p2) * np.sin(p1), np.cos(p2)], -1)
    return phi1, phi2, np.sum(vpr * ep, 1), np.sum(vpr * mp, 1)


def window_subsample(s, j, rng):
    ra, dec = s[:, CH["ra"]], s[:, CH["dec"]]
    lo_ra, hi_ra, lo_dec, hi_dec = WINDOW[j]
    idx = np.where((ra >= lo_ra) & (ra <= hi_ra) & (dec >= lo_dec) & (dec <= hi_dec))[0]
    if len(idx) == 0:
        return None
    return s[rng.choice(idx, size=min(NOBS[j], len(idx)), replace=False)]


def binned_median(x, y, edges):
    return np.array(
        [
            np.median(y[(x >= lo) & (x <= hi)]) if ((x >= lo) & (x <= hi)).sum() else np.nan
            for lo, hi in zip(edges[:-1], edges[1:])
        ]
    )


def binned_std(x, y, edges, min_count=3):
    """Per-bin sample std (ddof=1); NaN for bins with fewer than ``min_count`` points."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (x >= lo) & (x <= hi)
        out.append(np.std(y[sel], ddof=1) if sel.sum() >= min_count else np.nan)
    return np.array(out)


def binned_mad_std(x, y, edges, min_count=3):
    """Per-bin robust std (1.4826 x MAD) — insensitive to member-sample contamination, so the
    gap between this and :func:`binned_std` on the REAL data flags outlier-inflated bins."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (x >= lo) & (x <= hi)
        if sel.sum() >= min_count:
            v = y[sel]
            out.append(1.4826 * np.median(np.abs(v - np.median(v))))
        else:
            out.append(np.nan)
    return np.array(out)


def binned_mean_var(x, err, edges, min_count=3):
    """Per-bin mean squared measurement error — the noise floor to deconvolve from an observed
    per-bin std (Var_obs ~ Var_intrinsic + <err^2>)."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (x >= lo) & (x <= hi)
        out.append(np.mean(err[sel] ** 2) if sel.sum() >= min_count else np.nan)
    return np.array(out)


def augment_sim(sd, j, aug_preset="stream_global_ibata_grid", simulator="stream_agama",
                seed=0, upto="mask_vlos", aug_cfg=None):
    """Push raw grouped sim streams through the TRAINING observation model, so the per-bin std
    comparison against the real Gaia members is apples-to-apples (the network never sees
    noiseless streams).

    Applies the augmentation chain of ``aug_preset`` up to and including ``upto`` — for the
    stream_global presets that is: observational window -> member-count subsample -> compact ->
    Gaia member magnitudes -> DR3 per-star errors -> apply noise -> v_los measured-mask. NOT
    standalone (imports hydrabflow; CPU-safe — GPU probing is disabled).

    ``sd`` = (N, S, P, 6) raw ``sim_data_projected``; ``j`` = (N, S) stream indices. Returns
    ``(sim, attention_mask, vlos_mask)`` with shapes (N, S, P', 6) / (N, S, P') / (N, S, P'),
    P' = the training compaction length.
    """
    os.environ.setdefault("HYDRABFLOW_NUM_GPUS", "0")
    os.environ.setdefault("HYDRABFLOW_SIM_QUIET", "1")
    from omegaconf import OmegaConf

    from hydrabflow.augmentation.registry import build_augmentations

    if aug_cfg is None:  # compose the named preset (caller may instead pass a run's own node,
        # pre-resolved against its root config — e.g. the TRAIN augmentation of a model_dir)
        from hydra import compose, initialize_config_dir

        from hydrabflow.config.schema import register_configs

        register_configs()
        conf_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "conf")
        with initialize_config_dir(config_dir=conf_dir, version_base=None):
            cfg = compose(config_name="config", overrides=[
                f"simulator={simulator}", f"augmentation={aug_preset}", "composition=global",
            ])
        aug_cfg = OmegaConf.create(OmegaConf.to_container(cfg.augmentation, resolve=True))
    else:
        aug_cfg = OmegaConf.create(
            OmegaConf.to_container(aug_cfg, resolve=True)
            if OmegaConf.is_config(aug_cfg) else dict(aug_cfg)
        )
    steps = [str(s) for s in aug_cfg.steps]
    if upto not in steps:
        raise ValueError(f"step '{upto}' not in augmentation steps: {steps}")
    aug_cfg.steps = steps[: steps.index(upto) + 1]

    # Gaia resources (member magnitudes, DR3 error tables): fall back to the git-tracked
    # assets/gaia copy when the configured resources_dir (default 'data') is absent on this box.
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    res_dir = str(aug_cfg.params.get("resources_dir", "data"))
    member_table = str(aug_cfg.params.get("member_table", "apjad382dt1_mrt.txt"))
    if not os.path.exists(os.path.join(res_dir, member_table)):
        fallback = os.path.join(repo, "assets", "gaia")
        if os.path.exists(os.path.join(fallback, member_table)):
            aug_cfg.params["resources_dir"] = fallback

    chain = build_augmentations(aug_cfg, np.random.default_rng(seed), context={})
    n, s, p, c = sd.shape
    batch = {
        "sim_data_projected": np.asarray(sd, dtype=np.float32).reshape(n * s, p, c),
        "j": np.asarray(j, dtype=np.float32).reshape(n * s, 1),
    }
    for fn in chain:
        batch = fn(batch)

    sim = np.asarray(batch["sim_data_projected"])
    attn = np.asarray(batch["attention_mask"])[:, 0, :].astype(bool)
    vmask = np.asarray(batch["vlos_mask"])[:, 0, :].astype(bool)
    p2 = sim.shape[1]
    return sim.reshape(n, s, p2, c), attn.reshape(n, s, p2), vmask.reshape(n, s, p2)


def render(real, sim, out="ppc_summary_statistics.png", n_sim=40, k_track=10, k_vlos=3, seed=0,
           kind="Prior-predictive", sim_masks=None):
    """Overlay simulated per-stream summary tracks on the real Gaia streams and save to ``out``.

    ``real`` = observed-streams npz (defines the great-circle frames + real tracks); ``sim`` =
    grouped multistream npz path or (N,S,P,6) array. Reused by :func:`main`, by
    ``scripts/ppc_ancillary_observables.py`` and by ``scripts/ppc_posterior_summary_statistics.py``
    so prior- and posterior-predictive figures are directly comparable.

    ``sim_masks`` = optional ``(attention_mask, vlos_mask)`` (N,S,P) from :func:`augment_sim`:
    the sim is then treated as NOISE-CONVOLVED (training observation model already applied) —
    member selection uses the attention mask instead of the raw window subsample, the sim v_los
    bins use measured stars only, and the cold-stream comparison targets the RAW real std.
    """
    class args:  # keep the original body's `args.*` references intact
        pass
    args.real, args.sim, args.out = real, sim, out
    args.n_sim, args.k_track, args.k_vlos, args.seed = n_sim, k_track, k_vlos, seed

    rng = np.random.default_rng(args.seed)
    d = np.load(args.real)
    rsim = d["sim_data_projected"]
    rsim = rsim[0] if rsim.ndim == 4 else rsim
    ram = d["attention_mask"]
    ram = ram[:, 0, :] if ram.ndim == 3 else ram
    rvm = d["vlos_mask"]
    rvm = rvm[:, 0, :] if rvm.ndim == 3 else rvm
    rverr = d["vlos_error"] if "vlos_error" in d.files else None
    if rverr is not None and rverr.ndim == 3:
        rverr = rverr[:, 0, :]
    jarr = np.asarray(d["j"]).reshape(-1).astype(int)

    sd = np.asarray(sim) if isinstance(sim, np.ndarray) else np.load(sim)["sim_data_projected"]
    # (N,S,P,6)

    labels = ["phi2 [deg]", "mu_phi1", "mu_phi2", "vlos"]
    fig, axes = plt.subplots(3, 4, figsize=(18, 11))
    fig_s, axes_s = plt.subplots(3, 4, figsize=(18, 11))
    cold_report = []  # (stream, quantity, real_std, real_deconv, sim_med, frac_colder)
    for row in range(min(3, rsim.shape[0])):
        j = int(jarr[row])
        mem = ram[row].astype(bool)
        rs = rsim[row][mem]
        rvmask = (rvm[row].astype(bool) & mem)[mem]
        rverr_m = rverr[row][mem] if rverr is not None else None
        R = fit_frame(rs[:, CH["ra"]], rs[:, CH["dec"]])
        rphi1, rphi2, rm1, rm2 = project(R, rs[:, CH["ra"]], rs[:, CH["dec"]], rs[:, CH["mu_ra"]], rs[:, CH["mu_dec"]])
        rtracks = [rphi2, rm1, rm2, rs[:, CH["vlos"]]]

        te = np.quantile(rphi1, np.linspace(0, 1, args.k_track + 1))
        ve = np.quantile(rphi1[rvmask], np.linspace(0, 1, args.k_vlos + 1)) if rvmask.sum() > args.k_vlos else te
        tc, vc = 0.5 * (te[:-1] + te[1:]), 0.5 * (ve[:-1] + ve[1:])

        sim_med_std = [[] for _ in range(4)]  # per-draw median-over-bins std, per quantity
        picks = rng.choice(sd.shape[0], size=min(args.n_sim, sd.shape[0]), replace=False)
        for p in picks:
            if sim_masks is not None:  # noise-convolved: selection done by the training chain
                sel = sim_masks[0][p, row] & np.isfinite(sd[p, row][:, CH["ra"]])
                if sel.sum() < args.k_track:
                    continue
                s = sd[p, row][sel]
                svm = (sim_masks[1][p, row] & sim_masks[0][p, row])[sel]
            else:
                s = window_subsample(sd[p, row], j, rng)
                if s is None or len(s) < args.k_track:
                    continue
                svm = None
            p1, p2, m1, m2 = project(R, s[:, CH["ra"]], s[:, CH["dec"]], s[:, CH["mu_ra"]], s[:, CH["mu_dec"]])
            strk = [p2, m1, m2, s[:, CH["vlos"]]]
            for c in range(4):
                edges, cen = (ve, vc) if c == 3 else (te, tc)
                # sim v_los from MEASURED stars only when the vlos mask is available (matches
                # both the real data and what the training summary sees).
                cx, cy = (p1[svm], strk[c][svm]) if (c == 3 and svm is not None) else (p1, strk[c])
                axes[row, c].plot(cen, binned_median(cx, cy, edges), color="C0", alpha=0.25, lw=1)
                sstd = binned_std(cx, cy, edges)
                axes_s[row, c].plot(cen, sstd, color="C0", alpha=0.25, lw=1)
                if np.isfinite(sstd).any():
                    sim_med_std[c].append(np.nanmedian(sstd))

        for c in range(4):
            edges, cen = (ve, vc) if c == 3 else (te, tc)
            sx = rphi1[rvmask] if c == 3 else rphi1
            sy = rtracks[c][rvmask] if c == 3 else rtracks[c]

            ax = axes[row, c]
            ax.scatter(sx, sy, s=6, alpha=0.35, color="C3", zorder=3)
            ax.plot(cen, binned_median(sx, sy, edges), "k-o", lw=2.5, ms=5, zorder=4)

            # --- per-bin std ("cold stream" check): sim dispersion vs the real Gaia dispersion.
            # The real std includes measurement errors, so it is an UPPER bound on the intrinsic
            # dispersion; for vlos (the only per-star error shipped in the real npz) we also show
            # the error-deconvolved std — raw sim streams carry no observational noise.
            ax = axes_s[row, c]
            rstd = binned_std(sx, sy, edges)
            rmad = binned_mad_std(sx, sy, edges)
            ax.plot(cen, rstd, "k-o", lw=2.5, ms=5, zorder=4, label="real (incl. meas. err)")
            ax.plot(cen, rmad, "k--", lw=1.5, zorder=4, label="real robust (1.48 MAD)")
            rstd_dec = None
            if c == 3 and rverr_m is not None and rvmask.sum() >= 3:
                noise = binned_mean_var(sx, rverr_m[rvmask], edges)
                rstd_dec = np.sqrt(np.clip(rstd**2 - noise, 0.0, None))
                ax.plot(cen, rstd_dec, "-s", color="C3", lw=2.0, ms=4, zorder=4,
                        label="real (vlos err deconvolved)")
            if row == 0 and c in (0, 3):
                ax.legend(fontsize=7)

            for a in (axes[row, c], ax):
                if row == 0:
                    a.set_title(labels[c])
                if c == 0:
                    a.set_ylabel(f"{NAMES.get(j, j)}")
                a.set_xlabel("phi1 [deg]")
            ax.set_ylabel((f"{NAMES.get(j, j)}\n" if c == 0 else "") + "per-bin std")

            # noise-convolved sim is compared to the RAW real std; noiseless sim to the
            # deconvolved one (where available) since it lacks the measurement-error floor.
            if sim_masks is not None:
                real_ref = np.nanmedian(rstd)
            else:
                real_ref = np.nanmedian(rstd_dec if rstd_dec is not None else rstd)
            sim_arr = np.asarray(sim_med_std[c], dtype=float)
            cold_report.append((
                NAMES.get(j, str(j)), labels[c].split(" ")[0],
                float(np.nanmedian(rstd)),
                float(np.nanmedian(rmad)),
                float(np.nanmedian(rstd_dec)) if rstd_dec is not None else None,
                float(np.nanmedian(sim_arr)) if sim_arr.size else np.nan,
                float(np.mean(sim_arr < real_ref)) if sim_arr.size else np.nan,
            ))

    fig.suptitle(
        f"{kind} summary tracks: sim medians (blue) vs real Gaia (red pts, black median). "
        f"K_track={args.k_track}, K_vlos={args.k_vlos}",
        fontsize=13,
    )
    sim_desc = ("noise-convolved: training observation model applied" if sim_masks is not None
                else "noiseless")
    fig_s.suptitle(
        f"{kind} per-bin track std — 'too cold' check: sim dispersion (blue; {sim_desc}) vs real "
        "Gaia (black incl. measurement errors; red = vlos error-deconvolved). Sim persistently "
        "below the real level = too-cold streams.",
        fontsize=12,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    stem, ext = os.path.splitext(args.out)
    out_std = f"{stem}_std{ext or '.png'}"
    for f, path in ((fig, args.out), (fig_s, out_std)):
        f.tight_layout()
        f.savefig(path, dpi=110)
        print("saved", path)
    plt.close(fig)
    plt.close(fig_s)

    if sim_masks is not None:
        print("\nCold-stream check (median per-bin std over bins; sim is NOISE-CONVOLVED via the "
              "training observation model — a fair comparison against the raw real std; "
              "P(sim<real*) ~ 0.5 is well-bracketed, ~1.0 = too cold):")
    else:
        print("\nCold-stream check (median per-bin std over bins; sim is noiseless — expect sim "
              "slightly BELOW the raw real std, but far below the deconvolved level = too cold):")
    print(f"  {'stream':9s} {'quantity':8s} {'real':>8s} {'robust':>8s} {'deconv':>8s} "
          f"{'sim med':>8s} {'P(sim<real*)':>12s}")
    for stream, qty, r_raw, r_rob, r_dec, s_med, frac in cold_report:
        dec = f"{r_dec:8.3f}" if r_dec is not None else "       -"
        print(f"  {stream:9s} {qty:8s} {r_raw:8.3f} {r_rob:8.3f} {dec} {s_med:8.3f} {frac:12.2f}")

    return args.out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--real",
        default="assets/gaia/gaia_observed_streams_6Dwitherrors_cutNGC3201.npz",
        help="real observed-streams npz (defines the frames + real tracks)",
    )
    ap.add_argument("--sim", required=True, help="grouped multistream npz (N,S,P,6)")
    ap.add_argument("--out", default="ppc_summary_statistics.png")
    ap.add_argument("--n-sim", type=int, default=40)
    ap.add_argument("--k-track", type=int, default=10)
    ap.add_argument("--k-vlos", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", action="store_true",
                    help="apply the TRAINING observation model (Gaia window/subsample/DR3 noise/"
                    "vlos mask) to the sim before binning — fair 'too cold' comparison; imports "
                    "hydrabflow (not standalone)")
    ap.add_argument("--aug", default="stream_global_ibata_grid",
                    help="augmentation preset for --noise (the training chain to mimic)")
    ap.add_argument("--simulator", default="stream_agama",
                    help="simulator config for --noise (only target_streams matter)")
    args = ap.parse_args()

    sim, sim_masks = args.sim, None
    if args.noise:
        d = np.load(args.sim)
        sd, jarr = d["sim_data_projected"], np.asarray(d["j"]).reshape(d["sim_data_projected"].shape[:2])
        sim, attn, vmask = augment_sim(sd, jarr, aug_preset=args.aug, simulator=args.simulator,
                                       seed=args.seed)
        sim_masks = (attn, vmask)
    render(args.real, sim, args.out, args.n_sim, args.k_track, args.k_vlos, args.seed,
           sim_masks=sim_masks)


if __name__ == "__main__":
    main()

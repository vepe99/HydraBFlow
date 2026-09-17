"""Visual curation check of a FLAT stream training set: what does the network get fed?

1. Rotation curves: the sampled rows' ``vcirc_kms`` + dataset 68/95 % band vs the observed curve.
2. Per-stream in-window star counts (raw, before the observation model) vs the real member counts,
   NaN row fraction, and t_end / progenitor draws.
3. Stream panels + overlay through the TRAINING observation model — delegated to
   ``plot_fixed_potential_samples.py`` on a grouped (N, 3, P, 6) subsample written next to the
   figures.

Reads the npz members by memmap (np.savez stores them uncompressed), so a 24 GB file costs nothing.

    .venv/bin/python scripts/plot_training_set_samples.py \
        --npz data/.../training_data_1000000.npz --simulator stream_agama_ou24 \
        --aug stream_global_ibata_grid --out outputs/<run>/training_set_samples/sample
"""
from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys
import zipfile

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ppc_summary_statistics import NAMES, NOBS, WINDOW  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def npz_memmap(path, key):
    """Memmap one member of an UNCOMPRESSED npz (zip 'Stored' entries)."""
    zf = zipfile.ZipFile(path)
    info = zf.getinfo(f"{key}.npy")
    assert info.compress_type == zipfile.ZIP_STORED, f"{key} is compressed; np.load it instead"
    with open(path, "rb") as f:
        f.seek(info.header_offset)
        n_name, n_extra = struct.unpack("<HH", f.read(30)[26:30])
        data_start = info.header_offset + 30 + n_name + n_extra
        f.seek(data_start)
        version = np.lib.format.read_magic(f)
        reader = getattr(np.lib.format, f"read_array_header_{version[0]}_{version[1]}")
        shape, fortran, dtype = reader(f)
        offset = f.tell()
    return np.memmap(path, dtype=dtype, mode="r", shape=shape, offset=offset,
                     order="F" if fortran else "C")


def in_window_count(s, j):
    lo_ra, hi_ra, lo_dec, hi_dec = WINDOW[j]
    ra, dec = s[..., 0], s[..., 1]
    return ((ra >= lo_ra) & (ra <= hi_ra) & (dec >= lo_dec) & (dec <= hi_dec)).sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="flat training npz")
    ap.add_argument("--out", required=True, help="output prefix")
    ap.add_argument("--n-per-stream", type=int, default=36)
    ap.add_argument("--n-stats", type=int, default=20000, help="rows for the count/curve stats")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--simulator", required=True, help="simulator yaml (rotation-curve grid + streams)")
    ap.add_argument("--aug", default="stream_global_ibata_grid")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rng = np.random.default_rng(args.seed)

    j_all = np.asarray(npz_memmap(args.npz, "j")).reshape(-1).astype(int)
    sd = npz_memmap(args.npz, "sim_data_projected")
    vc = np.asarray(npz_memmap(args.npz, "vcirc_kms"))[..., 0]           # (n, n_r)
    n = len(j_all)
    streams = sorted(set(j_all.tolist()))

    # observed curve from the simulator yaml (single source of truth for the grid)
    from omegaconf import OmegaConf
    sim_cfg = OmegaConf.load(os.path.join(HERE, "..", "conf", "simulator", f"{args.simulator}.yaml"))
    r_obs = np.asarray(sim_cfg.params.obs_r_kpc, float)
    vc_obs = np.asarray(sim_cfg.params.obs_vc_kms, float)
    sig_obs = np.asarray(sim_cfg.params.obs_sigma_vc, float)

    # ------------------------------------------------------------------ stats subsample
    stat_idx = np.sort(rng.choice(n, size=min(args.n_stats, n), replace=False))
    counts, nan_frac = {}, {}
    for j in streams:
        idx = stat_idx[j_all[stat_idx] == j]
        rows = np.asarray(sd[idx])                                          # (k, P, 6)
        bad = ~np.isfinite(rows).all(axis=(1, 2))
        nan_frac[j] = bad.mean()
        counts[j] = in_window_count(np.nan_to_num(rows[~bad], nan=-1e9), j)
    vc_stat = vc[stat_idx]
    vc_bad = ~np.isfinite(vc_stat).all(1)
    vc_low = (np.nanmin(vc_stat, 1) < 3 * sig_obs.max())

    print(f"{n} rows; stats on {len(stat_idx)} random rows")
    print(f"vcirc: NaN rows {vc_bad.mean():.2%}, rows with min v_c < {3 * sig_obs.max():.0f} km/s "
          f"(noise can push them negative): {vc_low.mean():.2%}")
    for j in streams:
        c = counts[j]
        print(f"{NAMES[j]:8s} NaN rows {nan_frac[j]:.2%} | in-window stars (of {sd.shape[1]}): "
              f"median {np.median(c):.0f}, p5 {np.percentile(c, 5):.0f}, p95 {np.percentile(c, 95):.0f}, "
              f"< real count ({NOBS[j]}): {(c < NOBS[j]).mean():.1%}, < 20 stars: {(c < 20).mean():.1%}")

    # ------------------------------------------------------------------ plotted subsample
    pick = np.stack([np.sort(rng.choice(np.where(j_all == j)[0], args.n_per_stream, replace=False))
                     for j in streams], 1)                                  # (N, S)
    grouped = {"sim_data_projected": np.stack([np.asarray(sd[pick[:, c]]) for c in range(len(streams))], 1),
               "j": j_all[pick][..., None].astype(np.float32),
               "vcirc_kms": vc[pick[:, 0]][..., None]}
    for k in ("t_end", "m_progenitor", "a_progenitor", "ra", "dec", "vr", "r", "mu_ra_cosdec", "mu_dec"):
        try:
            grouped[k] = np.asarray(npz_memmap(args.npz, k))[pick]
        except KeyError:
            pass
    sub = f"{args.out}_subsample.npz"
    np.savez(sub, **grouped)
    print(f"wrote {sub}  (rows {pick.tolist()[:3]}... — NOTE one potential per ROW ONLY within a "
          f"column; the 3 streams of a row come from different potentials)")

    # ------------------------------------------------------------------ figure: rotation curves + counts
    fig, axes = plt.subplots(1, 1 + len(streams), figsize=(5 + 3.4 * len(streams), 4.0))
    ax = axes[0]
    good = vc_stat[~vc_bad]
    for q, a in ((2.5, 0.15), (16, 0.3)):
        ax.fill_between(r_obs, np.percentile(good, q, 0), np.percentile(good, 100 - q, 0),
                        color="C0", alpha=a, lw=0, label=f"dataset {100 - 2 * q:.0f}% band" if q == 16 else None)
    ax.plot(r_obs, np.median(good, 0), color="C0", lw=1.5, label="dataset median")
    for row in vc[pick.reshape(-1)]:
        ax.plot(r_obs, row, color="C1", lw=0.5, alpha=0.5)
    ax.plot([], [], color="C1", lw=0.5, label=f"{pick.size} plotted rows")
    ax.errorbar(r_obs, vc_obs, sig_obs, fmt="k.", capsize=2, label="observed")
    ax.set_xlabel("r [kpc]"); ax.set_ylabel("v_circ [km/s]"); ax.legend(fontsize=7)
    ax.set_title("rotation curve (stored, pre-noise)", fontsize=10)
    for k, j in enumerate(streams, 1):
        ax = axes[k]
        c = counts[j]
        ax.hist(c, bins=50, color="C0", alpha=0.7)
        ax.axvline(NOBS[j], color="k", ls="--", label=f"real members {NOBS[j]}")
        ax.set_yscale("log")
        ax.set_xlabel(f"in-window stars of {sd.shape[1]}")
        ax.set_title(f"{NAMES[j]}: NaN {nan_frac[j]:.1%}, <{NOBS[j]} stars {(c < NOBS[j]).mean():.0%}",
                     fontsize=10)
        ax.legend(fontsize=7)
    fig.suptitle(f"{os.path.basename(args.npz)} — {len(stat_idx)} random rows", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{args.out}_vcirc_counts.png", dpi=130)
    print(f"wrote {args.out}_vcirc_counts.png")

    # ------------------------------------------------------------------ stream panels via the existing plotter
    subprocess.run([sys.executable, os.path.join(HERE, "plot_fixed_potential_samples.py"),
                    "--sim", sub, "--out", args.out, "--seed", str(args.seed),
                    "--aug", args.aug, "--simulator", args.simulator,
                    "--label", "random TRAINING rows (one potential per row per stream)"], check=True)


if __name__ == "__main__":
    main()

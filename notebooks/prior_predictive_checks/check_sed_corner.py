"""Does the real SED land inside the training population *jointly*, not just band by band?

`check_sed_range` answers one wavelength at a time, which is the weaker question: a disk can sit
at the 50th percentile in every band and still be nowhere near the population if its *colours*
are wrong, because the bands are strongly correlated.  This is the joint version -- a corner plot
over a handful of representative bands, with the training population as 2D histograms and the
real disk as a crosshair -- plus the one number that summarises it, the Mahalanobis distance of
the real disk in log-flux space against the empirical distribution of the training rows' own
distances.

The SEDs come through the *full* configured forward model, so what is plotted is the noisy
population the network actually sees, including `sed_frac_cal_err`.  Run it before committing to
a training config: it is the cheap check that the SED branch is not being asked to extrapolate.

Run:
    uv run python notebooks/prior_predictive_checks/check_sed_corner.py [n] [hydra override ...]
    CHECK_DISK=hvtauc uv run python .../check_sed_corner.py 4000 experiment=protoplan_gausspsf
"""

import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _realdisk as rd  # noqa: E402  (imports hydrabflow: pins KERAS_BACKEND, picks a free GPU)
import make_real_npz as mrn  # noqa: E402

# Representative bands: the four imaged ones (3.9 / 450 / 880 / 1300 um) plus the optical/NIR and
# far-IR anchors that fix the SED's overall shape.  Nine axes is as wide as a corner plot stays
# readable; the Mahalanobis number below is computed on exactly these.
BANDS_UM = (0.5, 1.65, 3.9, 11.6, 24.0, 70.0, 450.0, 880.0, 1300.0)
DISKS = tuple(d for d in os.environ.get("CHECK_DISK", "hvtauc,oph163131").split(",") if d)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    overrides = tuple(sys.argv[2:]) or ("experiment=protoplan_gausspsf",)
    cfg = rd.load_cfg(overrides)

    data = rd.load_training_data(cfg)
    idx = np.random.default_rng(0).choice(len(data["seds"]), n, replace=False)
    idx.sort()
    rows = {k: np.asarray(v)[idx] for k, v in data.items()}
    batch = rd.configured_augmentation(cfg)(rows)

    lam = 10.0 ** np.squeeze(np.asarray(batch["sed_lams"])[0])
    # Nearest grid bin per requested band; the model grid is fixed, so this is exact enough.
    cols = [int(np.argmin(np.abs(np.log10(lam) - np.log10(b)))) for b in BANDS_UM]
    labels = [rf"{lam[c]:.3g} $\mu$m" for c in cols]
    X = np.squeeze(np.asarray(batch["seds"]), -1)[:, cols]     # already log10 Jy
    finite = np.isfinite(X).all(1)
    X = X[finite]
    print(f"{X.shape[0]} training rows x {len(cols)} bands (log10 Jy)")

    reals = {}
    for disk in DISKS:
        real = mrn.build(disk, overrides)[0]
        reals[disk] = np.squeeze(np.asarray(real["seds"]), -1)[0, cols]

    # Whiten on the training rows so every axis contributes on its own scale, then Mahalanobis.
    mu, sd = X.mean(0), X.std(0)
    Z_tr = (X - mu) / sd
    print()
    for disk, y in reals.items():
        d2_tr, d2, pct = rd.mahalanobis(Z_tr, (y - mu) / sd)
        print(f"{disk:10s} joint d2 = {d2:8.2f}  -> percentile {pct:5.1f} of the training rows' "
              f"own distances (median {np.median(d2_tr):.1f})")
        for c, lab, v in zip(cols, labels, y):
            print(f"    {lab:>12s}  {v:7.3f}  percentile {rd.percentile_rank(X[:, cols.index(c)], v):5.1f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = len(cols)
    colors = ("#0072B2", "#D55E00", "#009E73")
    fig, axes = plt.subplots(k, k, figsize=(1.55 * k, 1.55 * k))
    for i in range(k):
        for j in range(k):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                ax.hist(X[:, i], bins=60, color="0.75")
                for c, (disk, y) in zip(colors, reals.items()):
                    ax.axvline(y[i], color=c, lw=1.6)
                ax.set_yticks([])
            else:
                ax.hist2d(X[:, j], X[:, i], bins=60, cmap="Greys",
                          norm=matplotlib.colors.LogNorm())
                for c, (disk, y) in zip(colors, reals.items()):
                    ax.plot(y[j], y[i], marker="+", ms=11, mew=2.0, color=c)
            if i == k - 1:
                ax.set_xlabel(labels[j], fontsize=8)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(labels[i], fontsize=8)
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=6)
    for c, disk in zip(colors, reals):
        axes[0, k - 1].plot([], [], color=c, lw=2, label=disk)
    axes[0, k - 1].axis("off")
    axes[0, k - 1].legend(frameon=False, fontsize=10, loc="center")
    fig.suptitle(r"SED, log$_{10}$ $F_\nu$ [Jy] -- training population through the forward model",
                 fontsize=12)
    fig.tight_layout()
    out = rd.plot_dir("sed_corner", "_" + "_".join(DISKS))
    print("\n" + rd.save(fig, out, "sed_corner.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()

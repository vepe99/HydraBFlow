"""A plain gallery of forward-modelled training rows -- what the population actually looks like.

No model, no neighbour ranking, no scoring: stream N random training rows through the augmentation
exactly as configured for training (randomized observing prior and all) and plot them. This is the
eyeball version of `check_summary_range`'s question, and the thing to look at first when a
percentile says "outside" and you want to know what the inside looks like.

One PNG per band plus `population_sed.png` in `plots/population_<tag>/`, each panel asinh-stretched with that band's own
`simulator.params.asinh_sigma_*` -- the same transform the adapter applies, so the panels show the
network's view rather than a prettier one. `CHECK_DISK` adds the real disk as the first panel,
outlined, for comparison.

Run:
    uv run python notebooks/prior_predictive_checks/plot_population.py [n] [hydra override ...]
"""

import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _realdisk as rd  # noqa: E402  (imports hydrabflow: pins KERAS_BACKEND, picks a free GPU)

BANDS = [("im_jy_jwst", "JWST 3.9um", None), ("im_jy_alma_0", "B9 450um", 0),
         ("im_jy_alma_1", "B7 880um", 1), ("im_jy_alma_2", "B6 1300um", 2)]
DISK = os.environ.get("CHECK_DISK", "")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    overrides = tuple(sys.argv[2:]) or ("experiment=protoplan_mask_discrete",)
    cfg = rd.load_cfg(overrides)

    data = rd.load_training_data(cfg)
    # A random draw, not the first n rows: the caches are ordered by the RT grid, so a head slice
    # is a corner of parameter space rather than a sample of the population.
    idx = np.random.default_rng(0).choice(len(data["im_jy"]), n, replace=False)
    idx.sort()
    rows = {k: np.asarray(v)[idx] for k, v in data.items()}

    aug = rd.configured_augmentation(cfg, disk=DISK or None)
    batch = {k: np.asarray(v) for k, v in aug(rows).items()}
    print(f"{n} rows forward-modelled; keys {sorted(batch)}")

    real = None
    if DISK:
        import make_real_npz as mrn
        real = {k: np.asarray(v)[:1] for k, v in mrn.build(DISK, overrides)[0].items()}

    sig = {None: float(cfg.simulator.params.asinh_sigma_jwst)}
    sig.update(dict(enumerate(float(v) for v in cfg.simulator.params.asinh_sigma_alma)))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = rd.plot_dir("population", "_" + (DISK or "nodisk"))
    ncol = 16
    for key, label, ai in BANDS:
        imgs = np.arcsinh(batch[key] / sig[ai])
        panels = [(np.arcsinh(real[key][0] / sig[ai]), True)] if real is not None else []
        panels += [(imgs[i], False) for i in range(len(imgs))]
        nrow = int(np.ceil(len(panels) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(1.15 * ncol, 1.15 * nrow))
        for ax, panel in zip(axes.ravel(), panels + [None] * (nrow * ncol - len(panels))):
            ax.set_xticks([]); ax.set_yticks([])
            if panel is None:
                ax.axis("off"); continue
            img, is_real = panel
            # Each panel to its own peak: the population spans decades in amplitude, and a shared
            # scale would show one bright disk and 199 black squares. Amplitude is
            # `check_obs_range`'s question, shape is this one's.
            ax.imshow(np.squeeze(img), origin="lower", cmap="inferno",
                      vmin=float(np.min(img)), vmax=float(np.max(img)))
            for s in ax.spines.values():
                s.set_color("#00E5FF" if is_real else "0.85")
                s.set_linewidth(2.2 if is_real else 0.5)
            if is_real:
                ax.set_title(DISK, fontsize=7, color="#0072B2", pad=2)
        fig.suptitle(f"{label} -- {len(imgs)} training rows through the training forward model"
                     + (f"   (cyan = {DISK})" if real is not None else ""), fontsize=11)
        fig.tight_layout()
        print(rd.save(fig, out_dir, f"population_{key}.png"))
        plt.close(fig)

    # The SED, in the same units the network sees: log10 Jy on the log10 training grid, straight
    # out of `apply_sed_noise` + `log10_sed`, so the scatter shown is the *noisy* population.
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    lam = 10.0 ** np.squeeze(batch["sed_lams"][0])
    flux = 10.0 ** np.squeeze(batch["seds"])
    ax.plot(lam, flux.T, color="0.6", lw=0.5, alpha=0.5)
    ax.plot([], [], color="0.6", lw=0.5, label=f"{len(flux)} training rows")
    if real is not None:
        ax.plot(10.0 ** np.squeeze(real["sed_lams"][0]), 10.0 ** np.squeeze(real["seds"]),
                color="#0072B2", lw=2.0, label=DISK)
    ax.set(xscale="log", yscale="log", xlabel=r"$\lambda$ [$\mu$m]", ylabel=r"$F_\nu$ [Jy]",
           title="SED -- training population through the training forward model")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    print(rd.save(fig, out_dir, "population_sed.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()

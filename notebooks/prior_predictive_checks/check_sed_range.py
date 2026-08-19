"""Marimo notebook: is a real disk's SED inside the training population?

The SED arm of the prior-predictive checks.  Puts the training SEDs through exactly the two
`AugmentationsClass.preprocess` steps that touch them -- extinction correction, then the distance
rescale -- and reports where the real observation falls in the resulting per-wavelength
distribution.

Three arms, because they answer different questions:

* **clean** -- can the simulator's parameter prior produce this disk at all?
* **population log-normal** -- what the network actually saw.  `apply_sed_noise`'s width is measured
  across the five real disks, so it is wide enough to matter in the mid-IR where the SED is faint.
* **this disk's own error bars** -- the sharper question, using the measured uncertainties.

Everything stays in linear Jy; `log10_sed` is a compression for the network, not part of the
comparison.  No GPU and no real-disk pickle needed -- the SED is a text file in `assets/protoplan/`.

Each figure cell wraps its body in a function: marimo requires top-level names to be unique across
cells, and bare `for ax in ...` loops would collide.

Run with:  uv run marimo edit notebooks/prior_predictive_checks/check_sed_range.py
"""

import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    # Before importing _realdisk (which imports hydrabflow, which pins the backend and would
    # otherwise take a GPU): this notebook is interpolation plus one log-normal draw.
    os.environ["HYDRABFLOW_NUM_GPUS"] = "0"

    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

    import marimo as mo
    import numpy as np

    import _realdisk as rd

    disk = mo.ui.dropdown(options=sorted(rd.DISKS), value="oph163131", label="Real disk")
    n_curves = mo.ui.slider(0, 2000, value=400, step=100,
                            label="Individual sims drawn under the bands")
    # Scalar = one fixed extinction for the whole population (what the trained runs assumed);
    # "lo,hi" = draw one A_V per disk and marginalise over it, which is what `AugmentationsClass`
    # already does when `av` is a range.  A non-default value writes to its own plots/ subdirectory.
    av_text = mo.ui.text(value=os.environ.get("CHECK_AV", ""),
                         label="A_V override — blank = the disk's assumption; scalar; or 'lo,hi'")
    mo.vstack([disk, n_curves, av_text])
    return av_text, disk, mo, n_curves, np, rd


@app.cell
def _(av_text, disk, mo, rd):
    # The per-disk constants that are *assumptions*, not measurements -- stated up front so no
    # figure below can be read as more certain than its inputs.
    dcfg = rd.disk_config(disk.value)
    av = rd.parse_av(av_text.value, dcfg["av"])
    tag = rd.av_suffix(av, dcfg["av"])
    out_dir = rd.plot_dir(disk.value, tag)
    mo.md(
        f"""
        **{disk.value}** — `A_V = {av}`
        {'(**sampled per disk**, marginalised over)' if isinstance(av, tuple) else '(fixed)'},
        `dist_pc = {dcfg['dist_pc']}` (fiducial = no rescale),
        `sigma_jwst = {dcfg['sigma_jwst']:.5g}` MJy/sr.

        Missing ALMA channels:
        `{[rd.ALMA_CHANNELS[j][0] for j in rd.missing_alma_channels(disk.value)] or 'none'}`
        — treated as absent modalities.  Figures go to `{out_dir}`.
        """
    )
    return av, dcfg, out_dir, tag


@app.cell
def _(np, rd):
    # Composed from the repo's own config, so the data path and image variant are whatever the
    # trained runs used.  `load_dataset` memory-maps the images, so this costs almost nothing even
    # though only the SEDs are wanted here, and it applies the SED>0 row filter -- not optional,
    # since the raw cache holds exactly-zero fluxes and `apply_sed_noise` divides by the flux.
    cfg_run = rd.load_cfg()
    data = rd.load_training_data(cfg_run)
    train_lams = np.asarray(data["sed_lams"][0, :, 0], dtype=np.float64)
    sed_raw_jy = np.asarray(data["seds"][:, :, 0], dtype=np.float64)
    assert np.all(sed_raw_jy > 0), "load_dataset should have filtered non-positive SED rows"
    n_model_px = data["im_jy"].shape[1]
    (sed_raw_jy.shape, train_lams.round(3))
    return cfg_run, n_model_px, sed_raw_jy, train_lams


@app.cell
def _(av, dcfg, disk, n_model_px, np, rd, sed_raw_jy, train_lams):
    # An SED-only forward model: `has_jwst=False` skips the stpsf PSF entirely, so this notebook
    # needs neither $STPSF_PATH nor a device.
    aug_pop = rd.fixed_setup_augmentation(disk.value, n_model_px, av=av, has_jwst=False)

    # The two `preprocess` steps that touch the SED, applied by hand so each stage is inspectable.
    # `extinction_correction` mirrors `preprocess`' own branches: one correction for the whole
    # population when `av` is a scalar, one A_V drawn per row when it is a range.
    ext_corr = rd.extinction_correction(aug_pop, train_lams, av,
                                        rng=np.random.default_rng(0), n_rows=len(sed_raw_jy))
    dist_scale = (rd.spec.DIST_FID_PC / dcfg["dist_pc"]) ** 2
    clean = sed_raw_jy * ext_corr * dist_scale
    (ext_corr.shape, ext_corr[0].round(4), dist_scale)
    return aug_pop, clean, dist_scale, ext_corr


@app.cell
def _(aug_pop, av, clean, disk, n_model_px, np, rd, train_lams):
    def _noisy_arm(aug):
        batch = {"seds": np.asarray(clean, dtype=np.float32)[..., None],
                 "sed_lams": np.tile(np.asarray(train_lams, np.float32)[None, :, None],
                                     (clean.shape[0], 1, 1))}
        out = aug.apply_sed_noise(batch)
        return (np.asarray(out["seds"])[..., 0].astype(np.float64),
                np.asarray(out["sigma_sed_flux"], dtype=np.float64))

    # Arm 3 pins the log-normal to this disk's own measured error bars instead of the population's.
    err_own = rd.interp_real_sed(disk.value, train_lams)[1]
    noisy_pop, sigma_pop = _noisy_arm(aug_pop)
    noisy_own, sigma_own = _noisy_arm(rd.fixed_setup_augmentation(
        disk.value, n_model_px, av=av, has_jwst=False, sed_sigma_jy=err_own))
    (noisy_pop.shape, sigma_pop.shape)
    return err_own, noisy_own, noisy_pop, sigma_own, sigma_pop


@app.cell
def _(disk, rd, train_lams):
    flux_obs_grid, err_obs_grid, sed_table = rd.interp_real_sed(disk.value, train_lams)
    lam_obs, flux_obs, err_obs = sed_table[:, 0], sed_table[:, 1], sed_table[:, 2]
    # Index of the training bin nearest each measured band, in log wavelength.
    print(f"{disk.value}: {len(lam_obs)} measured bands, training grid has {len(train_lams)}")
    return err_obs, err_obs_grid, flux_obs, flux_obs_grid, lam_obs, sed_table


@app.cell
def _(lam_obs, np, train_lams):
    k_obs = np.array([int(np.argmin(np.abs(np.log10(train_lams) - np.log10(lam))))
                      for lam in lam_obs])
    k_obs
    return (k_obs,)


@app.cell
def _(clean, err_obs, flux_obs, k_obs, lam_obs, noisy_own, noisy_pop, np, rd, sigma_own,
      sigma_pop, train_lams):
    def _band_report(name, pop, sigma):
        """Per measured band: where the real flux sits in the population, in percent."""
        print(f"\n=== {name} ===")
        print(f"{'lam_obs':>9} {'lam_grid':>9} {'F_obs':>11} {'err_obs':>11} "
              f"{'F_med':>11} {'sig_med':>11} {'pct_clean':>10} {'pct_noisy':>10}")
        out = []
        for k, lo, fo, eo in zip(k_obs, lam_obs, flux_obs, err_obs):
            pct_c = rd.percentile_rank(clean[:, k], fo)
            pct_n = rd.percentile_rank(pop[:, k], fo)
            flag = "  <<<" if (pct_c < 0.15 or pct_c > 99.85) else ""
            print(f"{lo:9.3f} {train_lams[k]:9.3f} {fo:11.4g} {eo:11.4g} "
                  f"{np.median(clean[:, k]):11.4g} {np.median(sigma[:, k]):11.4g} "
                  f"{pct_c:10.2f} {pct_n:10.2f}{flag}")
            out.append((lo, pct_c, pct_n))
        return out

    ranks_pop = _band_report("population log-normal", noisy_pop, sigma_pop)
    ranks_own = _band_report("this disk's own error bars", noisy_own, sigma_own)
    return ranks_own, ranks_pop


@app.cell
def _():
    import matplotlib

    matplotlib.use("Agg")   # figures are saved and returned, never shown interactively
    import matplotlib.pyplot as plt

    PERCENTILES = [0.15, 2.5, 16, 50, 84, 97.5, 99.85]
    return PERCENTILES, plt


@app.cell
def _(PERCENTILES, clean, disk, err_obs, flux_obs, lam_obs, n_curves, noisy_own, noisy_pop, np,
      out_dir, plt, rd, train_lams):
    def _stages_figure():
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
        rng = np.random.default_rng(0)
        for ax, (title, pop) in zip(axes, [
                ("clean (prior reach)", clean),
                ("population log-normal (as trained)", noisy_pop),
                (f"{disk.value}'s own error bars", noisy_own)]):
            if n_curves.value:
                idx = rng.choice(pop.shape[0], size=min(n_curves.value, pop.shape[0]),
                                 replace=False)
                ax.plot(train_lams, pop[idx].T, color="0.6", lw=0.3, alpha=0.35)
            pct = np.percentile(pop, PERCENTILES, axis=0)
            for (a, b), alpha in (((0, 6), 0.15), ((1, 5), 0.25), ((2, 4), 0.40)):
                ax.fill_between(train_lams, pct[a], pct[b], color="tab:blue", alpha=alpha, lw=0)
            ax.plot(train_lams, pct[3], color="tab:blue", lw=1.6, label="training median")
            ax.errorbar(lam_obs, flux_obs, yerr=err_obs, fmt="o", ms=4, color="black",
                        ecolor="black", capsize=2, lw=1, label=disk.value, zorder=5)
            ax.set(xscale="log", yscale="log", xlabel=r"$\lambda$ [$\mu$m]")
            ax.set_title(title, fontsize=10)
        axes[0].set_ylabel("flux [Jy]")
        axes[0].legend(fontsize=8)
        # Bound the shared y axis by the *clean* population and the observation.  The log-normal
        # arms have tails reaching ~1e-14 Jy, which would otherwise squash the decade that matters
        # into a sliver.
        floor = min(np.percentile(clean, 0.15), flux_obs.min())
        ceil = max(np.percentile(clean, 99.85), flux_obs.max())
        axes[0].set_ylim(floor / 3, ceil * 3)
        fig.suptitle(f"{disk.value} vs the training SED population "
                     f"(bands: 0.15-99.85 / 2.5-97.5 / 16-84%)")
        fig.tight_layout()
        print(rd.save(fig, out_dir, "sed_training_vs_real.png", dpi=150))
        return fig

    fig_stages = _stages_figure()
    fig_stages
    return (fig_stages,)


@app.cell
def _(clean, disk, err_obs, flux_obs, lam_obs, np, out_dir, plt, rd, sigma_own, sigma_pop,
      train_lams):
    def _sigma_figure():
        frac_pop = sigma_pop / np.maximum(clean, 1e-30)
        frac_own = sigma_own / np.maximum(clean, 1e-30)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
        for ax, (lab, tr, own, ob) in zip(axes, [
                (r"$\sigma$ [Jy]", sigma_pop, sigma_own, err_obs),
                (r"$\sigma / F$", frac_pop, frac_own, err_obs / flux_obs)]):
            band = np.percentile(tr, [16, 84], axis=0)
            ax.fill_between(train_lams, band[0], band[1], color="tab:blue", alpha=0.3, lw=0,
                            label="injected 16-84% (population)")
            ax.plot(train_lams, np.median(tr, axis=0), color="tab:blue", lw=1.6)
            ax.plot(train_lams, np.median(own, axis=0), color="tab:green", lw=1.4, ls="--",
                    label="injected (this disk's sigma)")
            ax.plot(lam_obs, ob, "o", ms=4, color="black", label=f"{disk.value} measured")
            ax.set(xscale="log", yscale="log", xlabel=r"$\lambda$ [$\mu$m]", ylabel=lab)
            ax.legend(fontsize=8)
        fig.suptitle(f"{disk.value}: injected SED noise vs the measured error bars")
        fig.tight_layout()
        print(rd.save(fig, out_dir, "sed_noise_vs_err.png", dpi=150))
        return fig

    fig_sigma = _sigma_figure()
    fig_sigma
    return (fig_sigma,)


@app.cell
def _(clean, err_obs, flux_obs, k_obs, np):
    # A free multiplicative rescale per sim, in closed form (weighted least squares): with the
    # overall luminosity divided out, what is left is whether the SED *shape* is reachable.
    sub = clean[:, k_obs]
    sed_w = 1.0 / np.maximum(err_obs, 1e-30) ** 2
    scale = ((sub * flux_obs * sed_w).sum(axis=1)
             / np.maximum((sub * sub * sed_w).sum(axis=1), 1e-300))
    chi2 = (((scale[:, None] * sub - flux_obs) ** 2) * sed_w).sum(axis=1) / len(k_obs)
    print(f"chi2/N over the population: min={chi2.min():.3g} "
          f"p1={np.percentile(chi2, 1):.3g} median={np.median(chi2):.3g}")
    return chi2, scale, sed_w, sub


@app.cell
def _(chi2, clean, disk, err_obs, flux_obs, k_obs, lam_obs, np, out_dir, plt, rd, scale, sed_w,
      sub, train_lams):
    def _anomaly_figure():
        med = np.median(clean, axis=0)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))

        ax = axes[0]
        rel = clean / med
        lo, hi = np.percentile(rel, [2.5, 97.5], axis=0)
        ax.fill_between(train_lams, lo, hi, color="tab:blue", alpha=0.25, lw=0,
                        label="training 2.5-97.5%")
        ax.plot(train_lams, np.median(rel, axis=0), color="tab:blue", lw=1.5)
        ax.errorbar(lam_obs, flux_obs / med[k_obs], yerr=err_obs / med[k_obs],
                    fmt="o", ms=4, color="black", capsize=2)
        ax.axhline(1.0, color="0.4", lw=0.8, ls=":")
        ax.set(xscale="log", yscale="log", xlabel=r"$\lambda$ [$\mu$m]",
               ylabel=r"$F / F_{\rm med}(\lambda)$")
        ax.set_title("brightness divided out", fontsize=10)

        ax = axes[1]
        ranks = np.array([rd.percentile_rank(clean[:, k], f) for k, f in zip(k_obs, flux_obs)])
        ax.plot(lam_obs, np.clip(ranks, 0.05, 100), "o-", color="tab:red", ms=4)
        for y in (0.15, 2.5, 50.0, 97.5, 99.85):
            ax.axhline(y, color="0.7", lw=0.7, ls=":")
        ax.set(xscale="log", yscale="log", xlabel=r"$\lambda$ [$\mu$m]",
               ylabel="percentile rank in the clean population [%]")
        ax.set_title("per-band rank (clipped at 0.05%)", fontsize=10)

        ax = axes[2]
        best = np.argsort(chi2)[:5]
        for i in best:
            ax.plot(train_lams, scale[i] * clean[i], color="tab:blue", lw=0.9, alpha=0.8)
        ax.errorbar(lam_obs, flux_obs, yerr=err_obs, fmt="o", ms=4, color="black", capsize=2)
        ax.set(xscale="log", yscale="log", xlabel=r"$\lambda$ [$\mu$m]", ylabel="flux [Jy]")
        ax.set_title(f"5 best after a free rescale (min $\\chi^2/N$={chi2[best[0]]:.2f})",
                     fontsize=10)

        resid = (scale[best[0]] * sub[best[0]] - flux_obs) ** 2 * sed_w
        print("worst bands of the best sim [um]:", np.round(lam_obs[np.argsort(-resid)[:5]], 3))

        fig.suptitle(f"{disk.value}: SED anomaly")
        fig.tight_layout()
        print(rd.save(fig, out_dir, "sed_anomaly.png", dpi=150))
        return fig

    fig_anomaly = _anomaly_figure()
    fig_anomaly
    return (fig_anomaly,)


@app.cell
def _(chi2, disk, flux_obs, k_obs, lam_obs, np, out_dir, plt, rd, sub):
    def _color_color_figure():
        # Referenced to the geometric mean of *all* measured bands rather than one band: a single
        # reference band leaves a strong residual correlation with total luminosity, and what this
        # figure is for is shape.
        gm_tr = np.exp(np.log(np.maximum(sub, 1e-30)).mean(axis=1))
        gm_ob = np.exp(np.log(np.maximum(flux_obs, 1e-30)).mean())
        col_tr = np.log10(sub / gm_tr[:, None])
        col_ob = np.log10(flux_obs / gm_ob)

        nb = len(k_obs)
        keep = np.argsort(chi2)[:200]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, (i, j) in zip(axes, [(0, nb // 2), (nb // 2, nb - 1)]):
            ax.hexbin(col_tr[:, i], col_tr[:, j], gridsize=60, bins="log", cmap="Blues", mincnt=1)
            ax.plot(col_tr[keep, i], col_tr[keep, j], ".", ms=2, color="C1", alpha=0.6,
                    label=r"200 best $\chi^2$")
            ax.plot(col_ob[i], col_ob[j], "*", ms=16, color="black", label=disk.value)
            # The offset from the *local* median (sims within 0.3 dex in x) is the number to read.
            near = np.abs(col_tr[:, i] - col_ob[i]) < 0.3
            d = col_ob[j] - (np.median(col_tr[near, j]) if near.any() else np.nan)
            ax.set_xlabel(f"log10 F({lam_obs[i]:.3g}um) / geo-mean")
            ax.set_ylabel(f"log10 F({lam_obs[j]:.3g}um) / geo-mean")
            ax.set_title(f"{disk.value} is {d:+.2f} dex from the local median", fontsize=10)
            ax.legend(fontsize=8)
        fig.suptitle(f"{disk.value}: SED colour-colour "
                     f"(referenced to the geometric mean of all bands)")
        fig.tight_layout()
        print(rd.save(fig, out_dir, "sed_color_color.png", dpi=150))
        return fig

    fig_color = _color_color_figure()
    fig_color
    return (fig_color,)


@app.cell
def _(mo):
    # The corners are n_bands x n_bands panels (19x19 for oph163131) and take a few seconds each;
    # off by default so re-running the notebook stays quick.
    want_corner = mo.ui.checkbox(value=True, label="Draw the per-band corner plots (slow)")
    want_corner
    return (want_corner,)


@app.cell
def _(av, clean, dcfg, disk, flux_obs, k_obs, lam_obs, np, out_dir, plt, rd, want_corner):
    def _corner(shape_only):
        """
        Every observed band against every other, with the real disk overlaid.

        The pairwise view the colour-colour figure only samples: a joint gap anywhere in the
        n-band flux vector is visible rather than guessed at.  Raw `log10 F` rather than colours,
        because that *is* what the network is handed -- `log10_sed` compresses the flux itself, so
        "is this observation in support" is a question about this space.  Every panel then looks
        correlated, since luminosity and distance move all bands together, which is what
        `shape_only` removes by dividing each row by its own geometric mean.

        Diagonal: the marginal with the real disk's percentile rank.  Off-diagonal: the 2-D
        histogram with the real disk as a star.
        """
        import matplotlib.colors as mcolors

        lg_tr = np.log10(np.maximum(clean[:, k_obs], 1e-30))
        lg_ob = np.log10(np.maximum(flux_obs, 1e-30))
        if shape_only:
            lg_tr, lg_ob = lg_tr - lg_tr.mean(1, keepdims=True), lg_ob - lg_ob.mean()
        n = len(k_obs)
        fig, axes = plt.subplots(n, n, figsize=(1.35 * n, 1.35 * n), sharex="col")
        for i in range(n):
            for j in range(n):
                ax = axes[i, j]
                if j > i:
                    ax.axis("off")
                    continue
                if i == j:
                    ax.hist(lg_tr[:, i], bins=60, color="C0")
                    ax.axvline(lg_ob[i], color="k", lw=1.2)
                    ax.set_yticks([])
                    ax.set_title(f"{lam_obs[i]:g}$\\mu$m  {100 * (lg_tr[:, i] < lg_ob[i]).mean():.1f}%",
                                 fontsize=7, pad=2)
                else:
                    ax.hist2d(lg_tr[:, j], lg_tr[:, i], bins=60, cmap="Blues", norm=mcolors.LogNorm())
                    ax.plot(lg_ob[j], lg_ob[i], "*", color="k", ms=7, mec="w", mew=0.5)
                ax.tick_params(labelsize=5)
                if j == 0 and i > 0:
                    ax.set_ylabel(f"{lam_obs[i]:g}", fontsize=7)
                else:
                    ax.set_yticklabels([])
                if i == n - 1:
                    ax.set_xlabel(f"{lam_obs[j]:g}", fontsize=7)
        fig.suptitle(("log10 F / <F> per band [shape only]" if shape_only else
                      f"log10 F [Jy] per band, sims at {dcfg['dist_pc']:g} pc, $A_V$={av}")
                     + f" -- {len(clean)} sims, {disk.value} in black", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        name = "sed_corner_bands_shape.png" if shape_only else "sed_corner_bands.png"
        print(rd.save(fig, out_dir, name, dpi=110))
        return fig

    fig_corner_flux = _corner(False) if want_corner.value else None
    fig_corner_shape = _corner(True) if want_corner.value else None
    fig_corner_flux
    return fig_corner_flux, fig_corner_shape


if __name__ == "__main__":
    app.run()

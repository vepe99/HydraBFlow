"""Marimo notebook: is a real disk's *amplitude* inside the training population?

The scale arm of the prior-predictive checks, and the one that justifies itself through
`training.standardize: [all]`: BayesFlow standardises every network input by the mean/std of the
augmented training set, so what the CNN actually sees is `(x - train_mean) / train_std`.  If the
regridded real inputs live on a different scale -- or have zeros where the sims have a noise floor,
or the reverse -- the standardised values are off-distribution by construction and the posterior is
unreliable, however good the architecture is.

This notebook streams the **whole** training set through the augmentation as *configured for
training* (randomized observing prior; only the distance is pinned to the real source), pulling each
augmented batch straight back to NumPy so the device never holds more than one.  It keeps exact
running moments plus a bounded subsample per key -- never the ~20 GB of augmented pixels -- then puts
the real observation beside them, in training-z units.

Needs a GPU (autocvd picks one on import) and `$STPSF_PATH`.

Run with:  uv run marimo edit notebooks/prior_predictive_checks/check_obs_range.py
"""

import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

    import marimo as mo
    import numpy as np

    import _realdisk as rd   # imports hydrabflow, which pins KERAS_BACKEND and picks a free GPU

    disk = mo.ui.dropdown(options=sorted(rd.DISKS), value="oph163131", label="Real disk")
    # `OBS_N_ROWS` seeds the widget so a headless `python check_obs_range.py` can run a short pass.
    n_rows = mo.ui.number(0, 50000, value=int(os.environ.get("OBS_N_ROWS", 0)), step=1024,
                          label="Training rows (0 = all; use a few thousand to iterate)")
    batch_size = mo.ui.number(64, 4096, value=1024, step=64, label="Forward-model batch size")
    # Scalar, or "lo,hi" to draw one A_V per disk and marginalise over it.  Extinction is a
    # per-channel scalar and is unity at mm wavelengths, so this widens the JWST and SED
    # distributions and leaves every ALMA one untouched.
    av_text = mo.ui.text(value=os.environ.get("CHECK_AV", ""),
                         label="A_V override — blank = the disk's assumption; scalar; or 'lo,hi'")
    mo.vstack([disk, n_rows, batch_size, av_text])
    return av_text, batch_size, disk, mo, n_rows, np, rd


@app.cell
def _(av_text, disk, mo, rd):
    cfg = rd.load_cfg()
    dcfg = rd.disk_config(disk.value)
    av = rd.parse_av(av_text.value, dcfg["av"])
    out_dir = rd.plot_dir(disk.value, rd.av_suffix(av, dcfg["av"]))
    mo.md(
        f"""
        **{disk.value}** — `A_V = {av}`
        {'(**sampled per disk**)' if isinstance(av, tuple) else '(assumed)'},
        `dist_pc = {dcfg['dist_pc']}`, `sigma_jwst = {dcfg['sigma_jwst']:.5g}` MJy/sr.

        Missing ALMA channels:
        `{[rd.ALMA_CHANNELS[j][0] for j in rd.missing_alma_channels(disk.value)] or 'none'}`
        — no row, no histogram, no score.  Figures go to `{out_dir}`.
        """
    )
    return av, cfg, dcfg, out_dir


@app.cell
def _(cfg, rd):
    data = rd.load_training_data(cfg)
    n_model_px = data["im_jy"].shape[1]
    (len(data["im_jy"]), data["im_jy"].shape)
    return data, n_model_px


@app.cell
def _(cfg, data, disk, np, rd):
    # The real observation, regridded onto the **randomized prior's** ALMA grid -- not this disk's
    # own.  The two differ (162 px vs 122 px for oph163131), and comparing pixel distributions
    # across different samplings of the same field is not a like-for-like comparison.
    alma_px = rd.randomized_alma_obs_px(cfg)
    train_lams = np.asarray(data["sed_lams"][0, :, 0], dtype=np.float64)
    real, aux = rd.build_real_batch(disk.value, train_lams, alma_px=alma_px)
    return alma_px, aux, real, train_lams


@app.cell
def _(disk, rd):
    # Keys the networks actually consume, post-augmentation, minus this disk's absent bands.
    image_keys = rd.present_image_keys(disk.value)
    sed_keys = ["seds", "sigma_sed_flux", "sed_lams"]
    cond_keys = ["rot_deg", "sigma_jwst"]
    all_keys = image_keys + sed_keys + cond_keys
    all_keys
    return all_keys, cond_keys, image_keys, sed_keys


@app.cell
def _(all_keys, av, batch_size, cfg, data, disk, n_model_px, n_rows, np, rd, real):
    SAMPLE_CAP = 4_000_000

    def _stream():
        aug = rd.configured_augmentation(cfg, disk=disk.value, av=av)
        total = len(data["im_jy"]) if n_rows.value in (0, None) else min(n_rows.value,
                                                                        len(data["im_jy"]))
        n_batches = max(1, int(np.ceil(total / batch_size.value)))
        acc = {k: rd.Accum(SAMPLE_CAP // n_batches) for k in all_keys}
        shapes = {}
        for i, j, raw in rd.training_batches(data, batch_size.value, n_rows.value):
            out = aug(raw)
            for k in all_keys:
                arr = np.asarray(out[k])
                acc[k].update(arr)
                shapes.setdefault(k, arr.shape[1:])
            if (i // batch_size.value) % 10 == 0:
                print(f"  rows {j}/{total}", flush=True)
        return {k: a.finalize() for k, a in acc.items()}, \
               {k: a.samples for k, a in acc.items()}, shapes

    train_stats, train_samples, train_shapes = _stream()

    # The real ALMA/JWST images must be on the same grid as the augmented ones, or every statistic
    # below silently compares different samplings of the same field.
    for _k, _shape in train_shapes.items():
        if _k.startswith("im_jy_"):
            assert np.shape(real[_k])[1:] == _shape, (
                f"{_k}: real {np.shape(real[_k])[1:]} vs augmented {_shape} -- the regrid target "
                f"and the forward model's grid disagree")
    train_shapes
    return SAMPLE_CAP, train_samples, train_shapes, train_stats


@app.cell
def _(all_keys, data, n_rows, np, rd, real, train_stats):
    def _table():
        """Train vs real, and the real disk's extremes in *training-z* units."""
        hdr = (f"{'key':16s} {'who':6s} {'n':>10s} {'min':>11s} {'max':>11s} {'mean':>11s} "
               f"{'std':>11s} {'p1':>11s} {'p50':>11s} {'p99':>11s} {'%neg':>7s}")
        print(hdr)
        print("-" * len(hdr))
        flags = []
        for k in all_keys:
            tr = train_stats[k]
            sa = rd.stats(real[k])
            for who, d in (("train", tr), ("real", sa)):
                print(f"{k if who == 'train' else '':16s} {who:6s} {d['n']:10d} "
                      f"{d['min']:11.4g} {d['max']:11.4g} {d['mean']:11.4g} {d['std']:11.4g} "
                      f"{d['p1']:11.4g} {d['p50']:11.4g} {d['p99']:11.4g} "
                      f"{100 * d['frac_neg']:7.2f}")
            std = tr["std"] if tr["std"] > 0 else np.nan
            z_max = (sa["max"] - tr["mean"]) / std
            z_min = (sa["min"] - tr["mean"]) / std
            note = f"{'':16s} {'z':6s} min={z_min:+8.2f}  max={z_max:+8.2f}"
            if max(abs(z_min), abs(z_max)) > 5:
                note += "   [scale]"
                flags.append((k, "scale", float(z_min), float(z_max)))
            # The real images are 0 outside the footprint where the sims carry noise, so the
            # negative fraction is where that shows up.  A 10x disagreement either way matters.
            ftr, fsa = tr["frac_neg"], sa["frac_neg"]
            if (ftr > 0 and fsa > 0 and (fsa / ftr > 10 or ftr / fsa > 10)) or \
               (ftr > 0.01 and fsa == 0) or (fsa > 0.01 and ftr == 0):
                note += "   [neg]"
                flags.append((k, "neg", float(ftr), float(fsa)))
            print(note)
            print()
        return flags

    obs_flags = _table()
    print("FLAGS:", obs_flags or "none")
    if n_rows.value not in (0, None) and n_rows.value < len(data["im_jy"]):
        print(f"\n!! only {n_rows.value} of {len(data['im_jy'])} training rows were streamed. The "
              f"population's min/max grow with the sample, so the [scale] flags above are "
              f"pessimistic -- rerun with 0 rows before drawing a conclusion from them.")
    return (obs_flags,)


@app.cell
def _():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return (plt,)


@app.cell
def _(disk, image_keys, np, out_dir, plt, rd, real, sed_keys, train_samples):
    def _hist_overlays():
        paths = []
        for k in image_keys + sed_keys:
            tr = train_samples[k]
            sa = np.asarray(real[k], dtype=np.float64).ravel()
            tr = tr[np.isfinite(tr)]
            sa = sa[np.isfinite(sa)]
            lo = min(np.percentile(tr, 0.05), sa.min())
            hi = max(np.percentile(tr, 99.95), sa.max())
            if not hi > lo:
                lo, hi = lo - 1e-6, hi + 1e-6
            bins = np.linspace(lo, hi, 80)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(tr, bins=bins, density=True, color="tab:blue", alpha=0.55,
                    label="training (augmented)")
            ax.hist(sa, bins=bins, density=True, color="tab:red", alpha=0.55, label=disk.value)
            ax.set(xlabel=k, ylabel="density", yscale="log")
            ax.set_title(f"{k}: training population vs {disk.value}", fontsize=10)
            ax.legend(fontsize=8)
            fig.tight_layout()
            paths.append(rd.save(fig, out_dir, f"obs_range_{k}.png"))
            plt.close(fig)
        return paths

    obs_hist_paths = _hist_overlays()
    print("\n".join(obs_hist_paths))
    return (obs_hist_paths,)


@app.cell
def _(aux, disk, image_keys, np, out_dir, plt, rd, real):
    def _real_panel():
        """The regridded real images themselves -- the sanity check that the regrid did something
        sensible before any statistic is believed."""
        fig, axes = plt.subplots(1, len(image_keys), figsize=(4.2 * len(image_keys), 4))
        for ax, k in zip(np.atleast_1d(axes), image_keys):
            img = np.asarray(real[k])[0, :, :, 0]
            axis = aux["jwst_axis"] if k == "im_jy_jwst" else aux["alma_axis"]
            ext = [axis.max(), axis.min(), axis.min(), axis.max()]   # RA increases left
            im = ax.imshow(img, origin="lower", cmap="inferno", extent=ext)
            ax.set_title(f"{rd.BAND_LABEL[k]}\n{img.shape[0]}x{img.shape[1]} px", fontsize=9)
            ax.set_xlabel("RA offset [\"]")
            fig.colorbar(im, ax=ax, fraction=0.046, label="MJy/sr")
        axes[0].set_ylabel("Dec offset [\"]")
        fig.suptitle(f"{disk.value}: the regridded real observation the comparison uses")
        fig.tight_layout()
        return fig, rd.save(fig, out_dir, "obs_real_regridded.png")

    fig_real, p_real = _real_panel()
    print(p_real)
    fig_real
    return fig_real, p_real


if __name__ == "__main__":
    app.run()

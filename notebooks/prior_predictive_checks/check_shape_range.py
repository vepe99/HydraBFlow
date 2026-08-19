"""Marimo notebook: is a real disk's *morphology* reachable by the simulator?

The shape arm of the prior-predictive checks.  `check_obs_range` and `check_sed_range` ask whether
the disk is out of distribution in amplitude -- pixel ranges, negativity, per-band SED flux and
colour.  Neither asks whether its **shape** is reachable: whether any training disk, seen through
this disk's own beam, ever looks like that.  A posterior can be confidently wrong for exactly that
reason with every amplitude diagnostic passing.

The forward model here is pinned to the disk's own measured setup -- its beams, its noise, and the
disk rotated to its *measured* PA rather than a fresh draw.  The ALMA beam is elongated and
sky-fixed, so a sampled PA would inject disk-vs-beam-angle scatter the real disk does not have,
widening the null and making the test *less* able to see a mismatch.  Tick `random_rotation` to
restore the training-time marginal.

Every feature is computed twice, under an absolute `3 sigma` mask and a scale-invariant
`0.02 x peak` mask.  Extinction enters as one scalar per channel, so within a band a wrong A_V is a
pure multiplicative rescale and every feature here is invariant to it *by construction* -- the one
leak is the noise mask, since noise is injected after `preprocess` at fixed absolute sigma.  That is
what the two arms separate, and why the A_V scan at the end matters: `A_V = 4.0` is an assumption
this repo adopted, not a measurement.

Needs a GPU (autocvd picks one on import) and `$STPSF_PATH`.

Run with:  uv run marimo edit notebooks/prior_predictive_checks/check_shape_range.py
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
    n_rows = mo.ui.number(0, 50000, value=int(os.environ.get("SHAPE_N_ROWS", 0)), step=1024,
                          label="Training rows (0 = all; use a few thousand to iterate)")
    batch_size = mo.ui.number(64, 4096, value=1024, step=64, label="Forward-model batch size")
    random_rotation = mo.ui.checkbox(
        value=False, label="Sample PA per sim (training marginal) instead of the measured PA")
    n_neighbours = mo.ui.number(4, 16, value=8, step=1, label="Nearest neighbours to show")
    # Scalar = one fixed extinction; "lo,hi" = drawn per disk and marginalised over.  Note what this
    # can and cannot do here: extinction is a per-channel scalar, so over A_V in [1,5] the swing is
    # 1.7 dex at 0.5um but *exactly* 0.000 dex at 450/880/1300um -- every ALMA shape feature is
    # bit-identical whatever A_V does.  Only the JWST channel moves, and only through the absolute
    # mask (0.12 dex at 3.9um).
    av_text = mo.ui.text(value=os.environ.get("CHECK_AV", ""),
                         label="A_V override — blank = the disk's assumption; scalar; or 'lo,hi'")
    mo.vstack([disk, n_rows, batch_size, random_rotation, n_neighbours, av_text])
    return av_text, batch_size, disk, mo, n_neighbours, n_rows, np, random_rotation, rd


@app.cell
def _(av_text, disk, mo, random_rotation, rd):
    cfg = rd.load_cfg()
    dcfg = rd.disk_config(disk.value)
    av = rd.parse_av(av_text.value, dcfg["av"])
    out_dir = rd.plot_dir(disk.value, rd.av_suffix(av, dcfg["av"]))
    setup, rot_deg = rd.load_obs_setup(disk.value)
    mo.md(
        f"""
        **{disk.value}** — `A_V = {av}`
        {'(**sampled per disk**)' if isinstance(av, tuple) else '(assumed)'},
        `dist_pc = {dcfg['dist_pc']}`,
        `sigma_jwst = {dcfg['sigma_jwst']:.5g}` MJy/sr, `rot_deg = {rot_deg:.3f}`
        (from {dcfg['rot_band']}){', **sampled per sim**' if random_rotation.value else ''}.

        Missing ALMA channels:
        `{[rd.ALMA_CHANNELS[j][0] for j in rd.missing_alma_channels(disk.value)] or 'none'}`.
        Masks: absolute `{rd.N_SIGMA:g} sigma` and relative `{rd.REL_FRAC:g} x peak`.
        Figures go to `{out_dir}`.
        """
    )
    return av, cfg, dcfg, out_dir, rot_deg, setup


@app.cell
def _(cfg, rd):
    data = rd.load_training_data(cfg)
    n_model_px = data["im_jy"].shape[1]
    (len(data["im_jy"]), n_model_px)
    return data, n_model_px


@app.cell
def _(av, data, disk, n_model_px, np, random_rotation, rd, setup):
    # The forward model pinned to this disk's setup, and the real observation on the grid *it*
    # produces (this disk's own beams, not the randomized prior's).
    aug = rd.fixed_setup_augmentation(disk.value, n_model_px, av=av,
                                      random_rotation=random_rotation.value)
    train_lams = np.asarray(data["sed_lams"][0, :, 0], dtype=np.float64)
    real, aux = rd.build_real_batch(disk.value, train_lams)
    bands = rd.present_image_keys(disk.value)

    # Mask levels come from the same JSON as the beams, so mask and forward model cannot disagree.
    sigma = {"im_jy_jwst": float(rd.disk_config(disk.value)["sigma_jwst"])}
    sigma.update({f"im_jy_alma_{j}": float(s["sigma_mjy_sr"]) for j, s in setup.items()})
    axes_arc = {b: (aux["jwst_axis"] if b == "im_jy_jwst" else aux["alma_axis"]) for b in bands}

    for _b in bands:                      # a grid mismatch would compare different samplings
        assert np.shape(real[_b])[1] == len(axes_arc[_b]), _b
    (bands, {k: round(v, 4) for k, v in sigma.items()}, {b: len(a) for b, a in axes_arc.items()})
    return aug, aux, axes_arc, bands, real, sigma, train_lams


@app.cell
def _(aug, axes_arc, bands, batch_size, data, n_rows, np, rd, sigma):
    def _stream(aug_, sigma_):
        """
        Reduce each augmented batch to a handful of numbers per image and keep no pixels.

        The neighbour panels need images, but keeping all of them would be ~15 GB; they are
        re-observed for the few winning rows once the distances are known.
        """
        total = len(data["im_jy"]) if n_rows.value in (0, None) else min(n_rows.value,
                                                                        len(data["im_jy"]))
        feats = {b: {"abs": [], "rel": []} for b in bands}
        field = {b: [] for b in bands}
        for i, j, raw in rd.training_batches(data, batch_size.value, n_rows.value):
            out = aug_(raw)
            for b in bands:
                img = np.asarray(out[b])[..., 0]
                ax = axes_arc[b]
                feats[b]["abs"].append(rd.shape_features(img, ax, sigma_[b]))
                feats[b]["rel"].append(rd.shape_features(img, ax, sigma_[b],
                                                         rel_frac=rd.REL_FRAC))
                # Unmasked field mean, for the cross-band ratios: injected noise is zero-mean so it
                # adds scatter but no bias, whereas a threshold biases each band by its own SNR.
                field[b].append(img.mean(axis=(1, 2)))
            if (i // batch_size.value) % 10 == 0:
                print(f"  rows {j}/{total}", flush=True)
        return ({b: {m: np.concatenate(v) for m, v in d.items()} for b, d in feats.items()},
                {b: np.concatenate(v) for b, v in field.items()},
                total)

    F_train, field_train, n_train = _stream(aug, sigma)
    {b: F_train[b]["abs"].shape for b in bands}
    return F_train, field_train, n_train


@app.cell
def _(axes_arc, bands, np, rd, real, sigma):
    F_real = {b: {"abs": rd.shape_features(np.asarray(real[b]), axes_arc[b], sigma[b])[0],
                  "rel": rd.shape_features(np.asarray(real[b]), axes_arc[b], sigma[b],
                                           rel_frac=rd.REL_FRAC)[0]}
              for b in bands}
    for _b in bands:
        for _m in ("abs", "rel"):
            assert np.isfinite(F_real[_b][_m]).all(), \
                f"{_b}/{_m}: the real disk's features are not finite -- the mask kept nothing"
    {b: dict(zip(rd.FEATURES, F_real[b]["abs"].round(3))) for b in bands}
    return (F_real,)


@app.cell
def _(F_real, F_train, bands, np, rd):
    def _score():
        """Per-band ranks and joint d^2, plus an all-bands joint over rows usable in every band."""
        report = {}
        for mask in ("abs", "rel"):
            print(f"\n{'=' * 78}\n{rd.mask_label(mask == 'rel')}\n{'=' * 78}")
            per_band, d2 = {}, {}
            for b in bands:
                tr, sa = F_train[b][mask], F_real[b][mask]
                good = np.isfinite(tr).all(axis=1)
                tr_ok = tr[good]
                ranks = np.array([rd.percentile_rank(tr_ok[:, k], sa[k])
                                  for k in range(len(rd.FEATURES))])
                print(f"\n{rd.BAND_LABEL[b]}  ({good.sum()}/{len(tr)} usable rows)")
                print(f"  {'feature':14s} {'real':>11s} {'p1':>11s} {'p50':>11s} {'p99':>11s} "
                      f"{'rank':>7s}")
                for k, name in enumerate(rd.FEATURES):
                    p1, p50, p99 = np.percentile(tr_ok[:, k], [1, 50, 99])
                    flag = "  <<<" if (ranks[k] < 1 or ranks[k] > 99) else ""
                    print(f"  {name:14s} {sa[k]:11.4g} {p1:11.4g} {p50:11.4g} {p99:11.4g} "
                          f"{ranks[k]:7.2f}{flag}")
                Z, z = rd.normal_score_block(tr_ok, sa)
                d2_tr, d2_sa, pct = rd.mahalanobis(Z, z)
                print(f"  joint d^2 = {d2_sa:.2f}  (p{pct:.2f}; train p50={np.median(d2_tr):.2f}, "
                      f"p99={np.percentile(d2_tr, 99):.2f})")
                per_band[b] = dict(ranks=ranks, n_usable=int(good.sum()))
                d2[rd.BAND_LABEL[b]] = (d2_tr, d2_sa, pct)

            # All bands at once, on the rows usable in *every* band, so the rank transform and the
            # covariance see one consistent population.
            good_all = np.ones(len(F_train[bands[0]][mask]), dtype=bool)
            for b in bands:
                good_all &= np.isfinite(F_train[b][mask]).all(axis=1)
            tr_all = np.hstack([F_train[b][mask][good_all] for b in bands])
            sa_all = np.hstack([F_real[b][mask] for b in bands])
            Z, z = rd.normal_score_block(tr_all, sa_all)
            d2_tr, d2_sa, pct = rd.mahalanobis(Z, z)
            print(f"\nall bands ({tr_all.shape[1]}-d, {good_all.sum()} rows): "
                  f"d^2 = {d2_sa:.2f} (p{pct:.2f})")
            d2["all bands"] = (d2_tr, d2_sa, pct)
            report[mask] = dict(per_band=per_band, d2=d2, good_all=good_all,
                                tr_all=tr_all, sa_all=sa_all)
        return report

    scores = _score()
    return (scores,)


@app.cell
def _():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return (plt,)


@app.cell
def _(F_real, F_train, bands, disk, np, out_dir, plt, rd, scores):
    def _marginals():
        paths = []
        for mask in ("abs", "rel"):
            suffix = "" if mask == "abs" else "_rel"
            for b in bands:
                tr = F_train[b][mask]
                tr = tr[np.isfinite(tr).all(axis=1)]
                ranks = scores[mask]["per_band"][b]["ranks"]
                fig, axes = plt.subplots(2, 4, figsize=(15, 6))
                for k, ax in enumerate(axes.ravel()):
                    rd.feature_hist(ax, tr[:, k], F_real[b][mask][k], rd.FEATURES[k],
                                    ranks[k], disk=disk.value)
                fig.suptitle(f"{rd.BAND_LABEL[b]}: shape features, training vs {disk.value} "
                             f"({rd.mask_label(mask == 'rel')})")
                fig.tight_layout()
                paths.append(rd.save(fig, out_dir, f"shape_hist_{b}{suffix}.png"))
                plt.close(fig)
        return paths

    shape_hist_paths = _marginals()
    print("\n".join(shape_hist_paths))
    return (shape_hist_paths,)


@app.cell
def _(disk, np, out_dir, plt, rd, scores):
    def _mahalanobis_figures():
        paths = []
        for mask in ("abs", "rel"):
            d2 = scores[mask]["d2"]
            keys = list(d2)
            fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3.6))
            for ax, k in zip(np.atleast_1d(axes), keys):
                tr, sa, pct = d2[k]
                bins = np.logspace(np.log10(max(tr.min(), 1e-3)),
                                   np.log10(max(tr.max(), sa) * 1.1), 60)
                ax.hist(tr, bins=bins, color="tab:blue", alpha=0.6)
                ax.axvline(sa, color="tab:red", lw=2)
                ax.set(xscale="log", xlabel="Mahalanobis $d^2$")
                ax.set_title(f"{k}\n{disk.value} $d^2$={sa:.1f} (p{pct:.2f})", fontsize=9)
            fig.suptitle(f"{disk.value}: joint shape distance vs the training rows' own "
                         f"distances ({rd.mask_label(mask == 'rel')})")
            fig.tight_layout()
            paths.append(rd.save(fig, out_dir,
                                 f"shape_mahalanobis{'' if mask == 'abs' else '_rel'}.png"))
            plt.close(fig)
        return paths

    shape_d2_paths = _mahalanobis_figures()
    print("\n".join(shape_d2_paths))
    return (shape_d2_paths,)


@app.cell
def _(F_real, aug, axes_arc, bands, data, disk, n_neighbours, np, out_dir, rd, real, scores):
    def _neighbours():
        """
        The real disk beside its nearest training sims in the joint whitened shape space.

        The winning rows are re-observed rather than cached during the stream: keeping every
        augmented image would cost ~15 GB, and the noise redraw changes nothing about the shape the
        panel is there to show.
        """
        mask = "abs"
        s = scores[mask]
        Zw, zw = rd.whiten(*rd.normal_score_block(s["tr_all"], s["sa_all"]))
        order = np.argsort(np.linalg.norm(Zw - zw[None, :], axis=1))[:int(n_neighbours.value)]
        # `tr_all` was built from the rows usable in every band, so map back to dataset rows.
        rows = np.flatnonzero(s["good_all"])[order]
        dists = np.linalg.norm(Zw[order] - zw[None, :], axis=1)
        print(f"nearest rows: {rows.tolist()}")
        print(f"distances:    {np.round(dists, 3).tolist()}")

        sub = {k: np.asarray(data[k][rows]) for k in ("im_jy", "im_lams", "seds", "sed_lams")}
        out = aug(sub)
        labels = [f"row {r}  d={d:.2f}" for r, d in zip(rows, dists)]
        paths = []
        for b in bands:
            paths.append(rd.neighbour_panel(
                np.asarray(real[b])[0, :, :, 0], np.asarray(out[b])[..., 0], labels, b,
                axes_arc[b], out_dir, disk.value))
        return rows, dists, paths

    nb_rows, nb_dists, nb_paths = _neighbours()
    print("\n".join(nb_paths))
    return nb_dists, nb_paths, nb_rows


@app.cell
def _(av, bands, dcfg, disk, field_train, np, out_dir, plt, rd, real):
    def _flux_ratios():
        """
        Cross-band field-flux ratios, and the A_V each JWST ratio would imply.

        The JWST ratios are the A_V-sensitive ones -- extinction rescales the 3.9 um channel by up
        to 4x while leaving the mm channels untouched to 0.01%.  The ALMA-only ratios are a mm
        spectral index, immune to A_V by construction, and are therefore the control that separates
        "wrong extinction" from "wrong dust".  All bands share the same 3" field, so a ratio of field
        means *is* a ratio of aperture fluxes and the differing pixel counts cancel.

        One asymmetry the numbers carry and cannot remove: the real JWST cutout is 0 outside the IFU
        footprint where the sims carry noise, so its field mean is deflated (see
        `jwst_footprint_gap`).  The printed dex offset is reported alongside so the JWST rows can be
        read with it in mind.
        """
        ref = rd.av_ref(av)
        gap = rd.jwst_footprint_gap(disk.value)
        print(f"\nJWST field mean is deflated by the footprint gap: {100 * gap:.1f}% of pixels "
              f"are 0 where the sims have noise (~{np.log10(1 - gap):+.3f} dex)")
        pairs = [(f"{rd.BAND_LABEL[a]} / {rd.BAND_LABEL[b]}", a, b)
                 for a in bands for b in bands
                 if a != b and bands.index(a) < bands.index(b)]
        real_field = {b: float(np.asarray(real[b]).mean()) for b in bands}

        fig, axes = plt.subplots(1, len(pairs), figsize=(4.0 * len(pairs), 3.6), squeeze=False)
        report = {}
        print(f"\n{'ratio':34s} {'real':>9s} {'p1':>9s} {'p50':>9s} {'p99':>9s} {'rank':>7s} "
              f"{'implied A_V':>12s}")
        for ax, (name, a, b) in zip(axes.ravel(), pairs):
            with np.errstate(divide="ignore", invalid="ignore"):
                tr = np.log10(field_train[a] / field_train[b])
                sa = float(np.log10(real_field[a] / real_field[b]))
            tr = tr[np.isfinite(tr)]
            rank = rd.percentile_rank(tr, sa)
            p1, p50, p99 = np.percentile(tr, [1, 50, 99])
            # Only a ratio carrying the JWST channel can be moved by extinction.
            av_imp = (rd.implied_av(sa - p50, rd.JWST_LAM_UM, ref)
                      if a == "im_jy_jwst" else np.nan)
            print(f"{name:34s} {sa:9.3f} {p1:9.3f} {p50:9.3f} {p99:9.3f} {rank:7.2f} "
                  f"{av_imp:12.2f}")
            lo, hi = min(p1, sa), max(p99, sa)
            ax.hist(tr, bins=np.linspace(lo - 0.1, hi + 0.1, 60), color="tab:blue", alpha=0.6,
                    density=True)
            ax.axvline(sa, color="tab:red", lw=2)
            ax.set_title(f"{name}\n{disk.value} p{rank:.1f}"
                         + ("" if np.isnan(av_imp) else f", implied $A_V$={av_imp:.1f}"),
                         fontsize=8)
            ax.set_xlabel("log10 field-flux ratio")
            report[name] = dict(real=sa, rank=rank, implied_av=av_imp)
        fig.suptitle(f"{disk.value}: cross-band flux ratios (unmasked field means)")
        fig.tight_layout()
        print(rd.save(fig, out_dir, "shape_flux_ratios.png"))
        return fig, report

    fig_ratios, ratio_report = _flux_ratios()
    fig_ratios
    return fig_ratios, ratio_report


@app.cell
def _(F_train, av, axes_arc, dcfg, disk, np, out_dir, plt, rd, real, sigma):
    def _av_scan():
        """
        Percentile rank of each JWST feature against the A_V the *real* image is de-reddened with.

        Under the absolute mask a rescale moves the SNR, so a wrong A_V leaks into every feature
        through the mask.  A flat curve says the JWST shape mismatch is physical; one that walks
        back into the bulk says the assumed A_V is a live suspect -- which matters here because
        `A_V = 4.0` is inherited, not measured for this disk.
        """
        band = "im_jy_jwst"
        av_values = (0.0, 1.0, 2.0, 4.0, 6.0, 10.0, 15.0, 25.0)
        tr = F_train[band]["abs"]
        tr = tr[np.isfinite(tr).all(axis=1)]
        img = np.asarray(real[band])

        ranks = np.full((len(av_values), len(rd.FEATURES)), np.nan)
        for i, av_i in enumerate(av_values):
            # Rescaling the *real* image by corr(av)/corr(av_model) is equivalent to re-reddening
            # the sims, and needs no re-run of the forward model.
            scaled = img * rd.extinction_ratio(rd.JWST_LAM_UM, av_i, av_ref=rd.av_ref(av))
            f = rd.shape_features(scaled, axes_arc[band], sigma[band])[0]
            if np.isfinite(f).all():
                ranks[i] = [rd.percentile_rank(tr[:, k], f[k]) for k in range(len(rd.FEATURES))]

        fig, ax = plt.subplots(figsize=(8, 5))
        for k, name in enumerate(rd.FEATURES):
            ax.plot(av_values, ranks[:, k], "o-", ms=4, label=name)
        ax.axhspan(5, 95, color="0.85", zorder=0)
        ax.axvline(rd.av_ref(av), color="0.3", ls="--", lw=1)
        ax.set(xlabel=r"assumed $A_V$ [mag]", ylabel="percentile rank [%]", ylim=(-2, 102))
        ax.set_title(f"{disk.value}: JWST shape-feature rank vs assumed $A_V$ "
                     f"(dashed = the reference, {rd.av_ref(av):g})", fontsize=10)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        print(rd.save(fig, out_dir, "shape_av_scan_jwst.png"))
        return fig, dict(av_values=av_values, ranks=ranks)

    fig_av, av_report = _av_scan()
    fig_av
    return av_report, fig_av


if __name__ == "__main__":
    app.run()

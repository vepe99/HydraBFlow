"""Which held-out test rows sit closest to a real disk *in the network's summary space*?

`check_summary_range` answers "is the disk inside the population"; this answers "and which
simulations look like it to the network". Same space, same scoring toolkit -- the standardized
summary vector, rank-transformed per dimension and whitened, so Euclidean distance there is the
Mahalanobis distance the other notebook reports. The reference population is the **held-out tail**
(`tail_split`) by default -- the rows `evaluate` scores; `tail_split` is the only split, so its
`val` half is also the validation set, they are not two things. `CHECK_TRAIN_ROWS=<n>` uses the
first n rows of the *training* half instead, which buys a better-determined whitening covariance
and a denser pool at the price of asking about rows the network was fit on. Outputs from the
training pool are suffixed `_train<n>` so the two cannot clobber each other.

`CHECK_FIXED_SETUP=1` pins the beam and the noise to the disk's own measured values instead of
drawing them from the training prior, so the ranking asks about physics rather than about observing
conditions; outputs are suffixed `_fixedsetup`. The randomized default remains the honest model for
"is this disk in the population"; the pinned one is the honest model for "which sims look like it".

One difference from `check_summary_range`, and it is the point: the test rows are observed at the
**disk's own position angle** rather than a fresh draw per row. `apply_rotation` collapses to a
single angle when `rot_range_deg=(theta, theta)`, so this costs no new code. Sampling PA instead
would put disk-vs-beam angle scatter into the neighbour distances that the real disk, at one known
PA, does not have -- the neighbours would then be picked partly on orientation.

The absent-band columns are dropped from the distance (oph163131 has no B7: its embedding comes
from a zero image, which would otherwise dominate).

**The figure ranks per branch, not jointly**, and each column is ranked by its own block of the
summary vector -- so the cells in one row are different simulations, and the trade-off between the
bands and the SED is visible instead of averaged away. The joint distance is still computed,
printed, and cached (`summary_edge.py` and `check_summary_pca.py` read it), just not plotted: for a
disk this far outside the population it is a near-tie thousands of rows wide, and its argmin is a
compromise no single branch endorses. See the comment beside `branch_dist` for the hvtauc numbers.

Run:
    uv run python notebooks/prior_predictive_checks/summary_neighbours.py <model_dir> <out_dir> \
        [n_neighbours] [hydra override ...]
"""

import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _realdisk as rd  # noqa: E402  (imports hydrabflow: pins KERAS_BACKEND, picks a free GPU)
import check_summary_range as csr  # noqa: E402
import make_real_npz as mrn  # noqa: E402

DISK = os.environ.get("CHECK_DISK", "oph163131")

#: What to show per row: adapter image key -> (label, asinh knob index; None = JWST).
BANDS = [("im_jy_jwst", "JWST 3.9um", None), ("im_jy_alma_0", "B9 450um", 0),
         ("im_jy_alma_1", "B7 880um", 1), ("im_jy_alma_2", "B6 1300um", 2)]


def asinh(img, sigma):
    return np.arcsinh(np.asarray(img, dtype=np.float64) / sigma)


def main():
    model_dir, out_dir = sys.argv[1], sys.argv[2]
    k_nn = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    overrides = tuple(sys.argv[4:]) or ("experiment=protoplan_mask_discrete_av",)
    os.makedirs(out_dir, exist_ok=True)

    cfg = rd.load_cfg(overrides)
    cfg.model_dir = model_dir
    dcfg = rd.disk_config(DISK)
    _setup, rot_deg = rd.load_obs_setup(DISK)

    # `tail_split` is the only split there is: its `val` half *is* the held-out set `evaluate`
    # scores, so "validation" and "test" name the same 2000 rows. `CHECK_TRAIN_ROWS=<n>` swaps the
    # reference population for the first n rows of the *training* half instead -- a larger pool
    # makes the whitening covariance and the neighbour ranking better determined, at the cost of
    # asking about rows the network was fit on.
    n_test = next(int(s.get("n_test", 2000)) for s in cfg.preprocessing.steps
                  if s.name == "tail_split")
    full = rd.load_training_data(cfg)
    n_total = len(full["im_jy"])
    n_train_rows = int(os.environ.get("CHECK_TRAIN_ROWS", "0"))
    if n_train_rows:
        n_pool, offset, tag = n_train_rows, 0, f"_train{n_train_rows}"
        test = {k: v[:n_pool] for k, v in full.items()}
        pool_label = f"first {n_pool} training rows"
    else:
        n_pool, offset, tag = n_test, n_total - n_test, ""
        test = {k: v[-n_test:] for k, v in full.items()}
        pool_label = f"the {n_test} held-out (validation) rows"
    print(f"reference population: {pool_label}, observed at the disk's PA {rot_deg:.3f} deg")

    real, absent = mrn.build(DISK, overrides)
    real = {k: np.asarray(v)[:1] for k, v in real.items()}

    # Training's randomized observing prior (that is the grid the real image was regridded onto),
    # the disk's distance and A_V, and its measured PA held fixed.
    #
    # `random_flip=False` goes with the fixed angle and is not optional: the parity flip is a
    # *second*, independent draw, so pinning `rot_range_deg` alone leaves half the reference rows
    # mirrored while `build_real_batch` declares the real disk at `sky_flip=+1`. Mirrored rows are
    # then compared to the disk across a parity the network is told about, which is the same
    # orientation contamination fixing the angle was meant to remove.
    #
    # `CHECK_FIXED_SETUP=1` additionally pins the **beam and the noise** to this disk's measured
    # values, which is what makes the neighbour ranking a question about physics rather than about
    # observing conditions.  Under the randomized default every sim draws its own beam and noise
    # from the prior while the real row carries the measured ones (`build_real_batch` writes
    # `beam_maj_j` / `sigma_alma_j` from the FITS headers), and B7's prior alone spans a factor 2.9
    # in beam -- a large morphological change at fixed parameters.  Worse, those scalars are
    # conditions spliced into the CNN trunks, so each sim's embedding carries its own random
    # condition values too.  `configured_augmentation`'s docstring makes the case for the
    # randomized prior, and it is the right one for `check_summary_range`'s question ("what
    # population did the network see"); it is the wrong one for "which sims look like this disk".
    #
    # Note what is *not* pinned: `alma_obs_px_arcsec` stays at the config's value.
    # `fixed_setup_augmentation` derives its own grid from the measured beams (0.01427" -> 211 px
    # for hvtauc), but the real images are regridded onto the randomized prior's sampling because
    # that is the grid the CNN was trained on (`make_real_npz.build` -> `randomized_alma_obs_px`),
    # so taking the whole fixed-setup augmentation here would compare two different samplings and
    # feed the network a grid it never saw.
    fixed_setup = os.environ.get("CHECK_FIXED_SETUP", "") not in ("", "0")
    setup_kwargs = {}
    if fixed_setup:
        beams, noises = rd.measured_beams_noises(_setup)
        setup_kwargs = dict(randomize_alma_setup=False, alma_beams=beams,
                            alma_noises_jy_beam=noises,
                            jwst_noise_mjy_sr=(dcfg["sigma_jwst"], dcfg["sigma_jwst"]))
        tag += "_fixedsetup"
        print(f"CHECK_FIXED_SETUP: beams {[tuple(round(x, 4) for x in b) for b in beams]}, "
              f"noises {[float(f'{n:.4g}') for n in noises]} Jy/beam, "
              f"sigma_jwst {dcfg['sigma_jwst']} MJy/sr")
    aug = rd.configured_augmentation(cfg, disk=DISK, av=dcfg["av"],
                                     rot_range_deg=(rot_deg, rot_deg), random_flip=False,
                                     **setup_kwargs)
    batches = [(i, j, {k: np.asarray(v) for k, v in aug(raw).items()})
               for i, j, raw in rd.training_batches(test, 256, n_pool)]
    embed, keys, _approx = csr.summary_fn(cfg, csr.with_placeholders(batches[0][2], cfg))

    F = np.concatenate([embed(csr.with_placeholders(b, cfg)) for _i, _j, b in batches],
                       axis=0).astype(np.float64)
    f_real = embed(real)[0].astype(np.float64)
    blk = csr.blocks(keys, F.shape[1])
    assert blk is not None, f"summary width {F.shape[1]} not divisible by {len(keys)} branches"
    cols = np.concatenate([np.arange(F.shape[1])[blk[k]] for k in keys if k not in absent])
    print(f"summaries {F.shape}, branches {keys}, scoring on {len(cols)} dims "
          f"(dropped absent {absent})")

    # Same space `check_summary_range` reports its Mahalanobis distances in.
    Z, z = rd.normal_score_block(F[:, cols], f_real[cols])
    W, w = rd.whiten(Z, z)
    dist = np.sqrt(((W - w) ** 2).sum(1))
    order = np.argsort(dist)[:k_nn]

    # Per-branch distances in the same whitened space, and what the panel ranks on.
    #
    # The joint argmin is a compromise no branch endorses, because the disk sits several times the
    # population's own spacing outside *every* branch at once and the joint ranking is then a
    # near-tie broken by whichever block happens to be largest.  Measured on hvtauc against 47811
    # training rows: the joint nearest row is at 15.62 against a median of 19.62 and a p1 of 17.42,
    # so thousands of rows are effectively tied for first; the joint #1 is rank 26265/47811 in
    # `b9_input` and rank 90 in `sed_input`, i.e. it was picked for its SED and is worse than
    # median in B9 -- which is why its ALMA panels look like pure noise -- while the rows whose
    # ALMA morphology does match (rank 12 in b6, rank 1 in b7) sit near rank 20000 in the SED.
    # The joint top-3 shared no row with any single branch's top-3.
    #
    # Ranking inside one branch at a time is the question the figure can actually answer -- "which
    # sims look like this disk in *this* band" -- and the trade-off between the columns is then
    # visible instead of averaged away.  `dist` is still what the npz stores and what
    # `summary_edge.py` and `check_summary_pca.py` read.
    present = [k for k in keys if k not in absent]
    w_blk = len(cols) // len(present)
    branch_dist = {k: np.sqrt(((W[:, i * w_blk:(i + 1) * w_blk]
                                - w[i * w_blk:(i + 1) * w_blk]) ** 2).sum(1))
                   for i, k in enumerate(present)}
    branch_order = {k: np.argsort(d)[:k_nn] for k, d in branch_dist.items()}

    param_names = list(cfg.adapter.inference_variables)
    print(f"\n{k_nn} nearest rows in summary space ({pool_label}) "
          f"(median distance from the disk to a row {np.median(dist):.2f} -- this is the disk\'s "
          f"own distance distribution, NOT the population\'s inter-row spacing; "
          f"use summary_edge.py to ask whether the disk is interior):")
    head = f"  {'rank':>4s} {'row':>6s} {'dist':>7s} {'sil':>4s} {'cav':>4s}  " + \
           "".join(f"{p:>12s}" for p in param_names)
    print(head)
    for r, idx in enumerate(order):
        vals = "".join(f"{float(np.asarray(test[p][idx]).reshape(-1)[0]):12.3f}"
                       for p in param_names)
        print(f"  {r:4d} {offset + idx:6d} {dist[idx]:7.2f} "
              f"{float(np.asarray(test['sil_id'][idx]).reshape(-1)[0]):4.0f} "
              f"{float(np.asarray(test['has_cavity'][idx]).reshape(-1)[0]):4.0f} {vals}")

    # The same rows, and the panel's own ranking, branch by branch.  `joint r` is the rank that row
    # holds in the joint distance: a branch's own best row sitting at joint rank ~10000 is the
    # ranking being a near-tie, not a bug.
    joint_rank = np.empty(len(dist), int)
    joint_rank[np.argsort(dist)] = np.arange(len(dist))
    print(f"\nper-branch nearest rows -- what the figure's columns show:")
    for k in present:
        dk = branch_dist[k]
        print(f"  {k}: real min {dk.min():.2f}, p1 {np.percentile(dk, 1):.2f}, "
              f"p50 {np.percentile(dk, 50):.2f}")
        for r, idx in enumerate(branch_order[k]):
            print(f"    #{r + 1} row {offset + idx:6d}  d={dk[idx]:6.2f}  "
                  f"joint r{joint_rank[idx] + 1:6d} (d={dist[idx]:6.2f})  "
                  f"cav={float(np.asarray(test['has_cavity'][idx]).reshape(-1)[0]):.0f}")

    np.savez_compressed(
        os.path.join(out_dir, f"summary_neighbours{tag}.npz"),
        rows=np.array([offset + i for i in order]), dist=dist[order],
        all_dist=dist.astype(np.float32), branch_keys=np.array(keys), absent=np.array(absent),
        branch_present=np.array(present),
        branch_all_dist=np.stack([branch_dist[k] for k in present]).astype(np.float32),
        branch_rows=np.stack([offset + branch_order[k] for k in present]),
        rot_deg=rot_deg, real=f_real.astype(np.float32), test=F.astype(np.float32),
        **{p: np.array([float(np.asarray(test[p][i]).reshape(-1)[0]) for i in order])
           for p in param_names + ["sil_id", "has_cavity"]})

    # ── the figure: the disk on top, its neighbours beneath, band by band ─────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def row_batch(idx):
        """The augmented batch holding test row `idx`, and the offset of that row inside it."""
        for i, j, b in batches:
            if i <= idx < j:
                return b, idx - i
        raise IndexError(idx)

    #: figure column -> the summary branch that column's cells are ranked by.  The SED gets the
    #: last column, so the trade-off the joint ranking hides is one glance wide.
    col_branch = [mrn.BRANCH_NAME.get(ai, "jwst_input") for _k, _b, ai in BANDS] + ["sed_input"]

    real_lam = np.asarray(real["sed_lams"])[0].reshape(-1)
    real_sed = np.asarray(real["seds"])[0].reshape(-1)
    shown = sorted({int(i) for o in branch_order.values() for i in o})
    nb_sed = np.array([np.asarray(row_batch(i)[0]["seds"])[row_batch(i)[1]].reshape(-1)
                       for i in shown])
    sed_lo, sed_hi = min(real_sed.min(), nb_sed.min()) - 0.2, max(real_sed.max(), nb_sed.max()) + 0.2

    s_jwst = float(cfg.simulator.params.asinh_sigma_jwst)
    s_alma = [float(v) for v in cfg.simulator.params.asinh_sigma_alma]
    fig, axes = plt.subplots(k_nn + 1, len(BANDS) + 1,
                            figsize=(2.15 * (len(BANDS) + 1), 2.15 * (k_nn + 1)))
    for r in range(k_nn + 1):
        for c in range(len(BANDS) + 1):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            branch, is_sed = col_branch[c], c == len(BANDS)
            if r == 0:
                src, off = real, 0
            elif branch in absent:
                ax.axis("off")
                continue
            else:
                idx = branch_order[branch][r - 1]
                src, off = row_batch(idx)
            if is_sed:
                # Both the augmentation and `build_real_batch` emit the SED already
                # log10-compressed (`AugmentationsClass.log10_sed`), so this is a linear axis over
                # log10 quantities -- a log axis would silently drop every negative value, which is
                # most of them.
                ax.plot(real_lam, real_sed, color="0.6", lw=2.5, zorder=1)
                if r:
                    ax.plot(np.asarray(src["sed_lams"])[off].reshape(-1),
                            np.asarray(src["seds"])[off].reshape(-1),
                            color="#0072B2", lw=1.3, zorder=2)
                ax.set_ylim(sed_lo, sed_hi)
            else:
                key, _band, ai = BANDS[c]
                img = np.asarray(src[key])[off]
                img = img[..., 0] if img.ndim == 3 else img
                ax.imshow(asinh(img, s_jwst if ai is None else s_alma[ai]),
                          origin="lower", cmap="magma")
            if r == 0:
                head = "SED  (log10 F vs log10 lam)" if is_sed else BANDS[c][1]
                if branch in absent:
                    ax.set_title(f"{head}\n(absent -> zeros)", fontsize=8, color="#D55E00")
                else:
                    dk = branch_dist[branch]
                    ax.set_title(f"{head}\n{branch}: min {dk.min():.2f}, p50 "
                                 f"{np.percentile(dk, 50):.2f}", fontsize=8)
            else:
                ax.set_xlabel(f"row {offset + idx}  d={branch_dist[branch][idx]:.2f}  "
                              f"joint r{joint_rank[idx] + 1}", fontsize=7)
            if c == 0:
                ax.set_ylabel(f"{DISK}\n(real)" if r == 0 else f"#{r} in its own branch",
                              fontsize=8)
    fig.suptitle(f"{DISK} and the {k_nn} nearest rows *per branch* -- each column ranked by its "
                 f"own block of the summary vector, so the cells in a row are different sims\n"
                 f"{pool_label}, observed at the disk's PA {rot_deg:.2f} deg with "
                 f"{'the disk’s own measured beams and noise' if fixed_setup else 'the randomized training beam/noise prior'};"
                 f"  grey = the real disk's SED.  Joint ranking (a near-tie: min {dist.min():.2f} "
                 f"vs p50 {np.median(dist):.2f}) is printed, not plotted\n{model_dir}",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    print("\nfigure ->", rd.save(fig, out_dir, f"summary_neighbours{tag}.png"))

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.hist(dist, bins=60, color="0.75")
    for r, idx in enumerate(order):
        ax.axvline(dist[idx], color="#D55E00", lw=1)
    ax.set_xlabel("whitened rank-normal summary distance to " + DISK)
    ax.set_ylabel(pool_label)
    ax.set_title(f"the {k_nn} nearest (orange) against the whole pool", fontsize=10)
    print("figure ->", rd.save(fig, out_dir, f"summary_neighbour_dist{tag}.png"))


if __name__ == "__main__":
    main()

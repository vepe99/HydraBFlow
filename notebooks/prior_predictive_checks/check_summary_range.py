"""Is a real disk inside the training population *in the network's own summary space*?

The fourth prior-predictive check, and the one the other three could not do: `check_sed_range`,
`check_obs_range` and `check_shape_range` ask the question in spaces we chose (flux, amplitude,
morphology), this one asks it in the space the inference network actually conditions on. It needs a
finished training run, which is why it was not ported with the others.

What it compares is the **standardized summary vector**: every training row is pushed through the
augmentation as configured for training (randomized observing prior, distance pinned to the real
source), then through the adapter and the summary network with the approximator's own
standardization applied -- i.e. exactly the tensor the flow sees. The real disk goes through the
same summary network. With `fuse_head: false` that vector is the per-branch embeddings
concatenated in `sorted(backbones)` order, so it can be scored per instrument as well as globally,
which is what localizes an out-of-distribution verdict to a band.

Scoring reuses `_realdisk`'s toolkit so the numbers stay comparable to the other three notebooks:
rank-transform each dimension to normal scores through the training column's own ECDF, then
Mahalanobis distance against the *empirical* distribution of the training rows' own distances.

A band the disk lacks is a **positive control, not a result**: oph163131 has no B7, so its b7
embedding comes from a zero image and must come out extreme. If the b7 block is not flagged, the
method is not working.

Run:
    uv run python notebooks/prior_predictive_checks/check_summary_range.py <model_dir> [n_rows] \
        [hydra override ...]
"""

import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _realdisk as rd  # noqa: E402  (imports hydrabflow: pins KERAS_BACKEND, picks a free GPU)
import make_real_npz as mrn  # noqa: E402

DISK = os.environ.get("CHECK_DISK", "oph163131")


def with_placeholders(batch, cfg):
    """Zero-fill the target and indicator columns the adapter concatenates but we never read.

    `_realdisk.training_batches` materialises only the four keys the forward model needs, so the
    augmented batch has no parameter columns -- and `Adapter.concatenate` raises on a missing key.
    Only `summary_variables` is read out below, so what these hold is irrelevant; that they exist
    is not.
    """
    n = len(next(iter(batch.values())))
    out = dict(batch)
    for p_ in cfg.adapter.inference_variables:
        out.setdefault(p_, np.zeros((n, 1), dtype=np.float32))
    for c in cfg.adapter.inference_conditions:
        out.setdefault(c, np.zeros((n,), dtype=np.float32))
    return out


def summary_fn(cfg, probe):
    """`(batch_dict) -> (B, summary_dim)`, standardized exactly as the flow's conditions are."""
    from hydrabflow.pipeline import artifacts
    from hydrabflow.pipeline.workflow import build_workflow

    workflow = build_workflow(cfg)
    workflow.approximator = artifacts.load_approximator(
        cfg.model_dir, workflow=workflow, probe_batch={k: v[:2] for k, v in probe.items()})
    approx = workflow.approximator

    def embed(batch):
        adapted = approx.adapter(batch)
        sv = approx.standardizer.maybe_standardize(
            adapted["summary_variables"], key="summary_variables", stage="inference")
        return np.asarray(approx.summary_network(sv, training=False))

    # Branch layout: `ConditionedFusionNetwork` concatenates in `sorted(backbones)` order.
    return embed, sorted(approx.summary_network.backbones), approx


def blocks(keys, total):
    """Equal-width per-branch slices; every branch is built with the same `summary_dim`."""
    w = total // len(keys)
    if w * len(keys) != total:
        return None
    return {k: slice(i * w, (i + 1) * w) for i, k in enumerate(keys)}


def main():
    model_dir = sys.argv[1]
    n_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    overrides = tuple(sys.argv[3:]) or ("experiment=protoplan_mask_discrete",)

    cfg = rd.load_cfg(overrides)
    cfg.model_dir = model_dir
    dcfg = rd.disk_config(DISK)
    out_dir = rd.plot_dir(DISK, "_summary_" + pathlib.Path(model_dir).parent.name)

    data = rd.load_training_data(cfg)
    # `make_real_npz.build` is the only assembler of a real batch the *adapter* can read: on top of
    # the regrid it fills the zero image and in-prior placeholder conditions for a band the disk
    # lacks, which `build_real_batch` alone does not. It emits one row per indicator combination;
    # the summary network never sees the indicators, so row 0 is the observation.
    real, absent = mrn.build(DISK, overrides)
    real = {k: np.asarray(v)[:1] for k, v in real.items()}

    aug = rd.configured_augmentation(cfg, disk=DISK, av=dcfg["av"])
    # The probe has to be an *augmented* training batch: the adapter reads the augmentation's
    # per-instrument keys, which the raw cache does not have.
    probe = with_placeholders(
        {k: np.asarray(v) for k, v in aug(next(iter(rd.training_batches(data, 8, 8)))[2]).items()},
        cfg)
    embed, keys, approx = summary_fn(cfg, probe)

    emb, ref_raw = [], []
    total = len(data["im_jy"]) if n_rows in (0, None) else min(n_rows, len(data["im_jy"]))
    for i, j, raw in rd.training_batches(data, 512, n_rows):
        batch = with_placeholders({k: np.asarray(v) for k, v in aug(raw).items()}, cfg)
        # `summary_space_comparison` re-embeds raw data, so it cannot take the whole training set
        # (four image cubes x 49k rows). Two batches are what it gets; the full-population version
        # of the same statistic is `mmd()` below, on the embeddings we compute anyway.
        if len(ref_raw) < 2:
            ref_raw.append(batch)
        emb.append(embed(batch))
        if (i // 512) % 10 == 0:
            print(f"  rows {j}/{total}", flush=True)
    F_tr = np.concatenate(emb, axis=0).astype(np.float64)
    f_sa = embed(real)[0].astype(np.float64)
    print(f"\ntraining summaries {F_tr.shape}, real {f_sa.shape}, branches {keys}")

    blk = blocks(keys, F_tr.shape[1])
    assert blk is not None, f"summary width {F_tr.shape[1]} is not divisible by {len(keys)} branches"
    print(f"bands the disk lacks (positive control, excluded from the global score): {absent}")

    # ── per-dimension ranks ───────────────────────────────────────────────────────────
    ranks = np.array([rd.percentile_rank(F_tr[:, k], f_sa[k]) for k in range(F_tr.shape[1])])
    print("\nper-branch summary of the real disk's per-dimension percentile ranks:")
    print(f"  {'branch':12s} {'dims':>5s} {'min':>7s} {'median':>7s} {'max':>7s}  "
          f"{'outside 1-99':>12s}")
    for k, sl in blk.items():
        r = ranks[sl]
        out = int(((r < 1) | (r > 99)).sum())
        flag = "   <-- band absent" if k in absent else ("   <-- OOD dims" if out else "")
        print(f"  {k:12s} {len(r):5d} {r.min():7.2f} {np.median(r):7.2f} {r.max():7.2f} "
              f"{out:8d}/{len(r)}{flag}")

    # ── Mahalanobis, global (present bands only) and per branch ───────────────────────
    print("\nMahalanobis distance in rank-normal summary space (percentile against the training "
          "rows' own distances):")

    def score(cols, label):
        Z_tr, z_sa = rd.normal_score_block(F_tr[:, cols], f_sa[cols])
        d2_tr, d2_sa, pct = rd.mahalanobis(Z_tr, z_sa)
        # Nearest-neighbour ratio: robust where a 16-d Gaussian is not.
        W_tr, w_sa = rd.whiten(Z_tr, z_sa)
        sub = W_tr[:: max(1, len(W_tr) // 4000)]
        nn_sa = np.sqrt(((sub - w_sa) ** 2).sum(1)).min()
        d = np.sqrt(((sub[:, None, :] - sub[None, :400, :]) ** 2).sum(-1))
        np.fill_diagonal(d[:400, :400], np.inf)
        nn_tr = np.median(d.min(1))
        print(f"  {label:26s} d2={d2_sa:9.1f}  p{pct:6.2f} of training   "
              f"NN dist {nn_sa:6.2f} vs typical {nn_tr:5.2f}  (x{nn_sa / max(nn_tr, 1e-9):5.1f})")
        return d2_tr, d2_sa, pct

    present_cols = np.concatenate([np.arange(F_tr.shape[1])[blk[k]] for k in keys if k not in absent])
    d2_tr, d2_sa, pct_all = score(present_cols, "all present bands")
    per = {k: score(np.arange(F_tr.shape[1])[blk[k]], k) for k in keys}

    # ── MMD: the kernel two-sample statistic BayesFlow ships for exactly this question ────
    # `bf.diagnostics.metrics.summary_space_comparison` (Schmitt et al. 2021, model
    # misspecification) is `approximator.summarize()` followed by `bootstrap_comparison`, and
    # `embed` above already *is* summarize() -- adapter, inference-stage standardization, summary
    # network. So the cached embeddings go straight to the inner function: same statistic, same
    # bootstrap null, no second GPU pass, and the reference is the whole population rather than
    # what fits in memory as raw images. The high-level call is run too, on a small raw subsample,
    # because it is the number a reader following the BayesFlow docs would get.
    #
    # Whitened rank-normal space, not raw: the kernel is an IM-RBF mixture over hard-coded scales
    # on Euclidean distances, so on raw summaries the few high-variance directions set the
    # bandwidth and the rest of the vector stops mattering. Whitening also makes this directly
    # comparable to the Mahalanobis numbers above -- same space, different assumption (MMD needs
    # no ellipticity, which is what makes it worth having when the population is bimodal in
    # `has_cavity`).
    from bayesflow.diagnostics.metrics import bootstrap_comparison, summary_space_comparison
    from bayesflow.metrics.functional import maximum_mean_discrepancy

    print("\nMMD against the training population (bootstrap null over the training rows):")
    rng = np.random.default_rng(0)

    def mmd(cols, label, n_ref=2000, n_null=500):
        Z_tr, z_sa = rd.normal_score_block(F_tr[:, cols], f_sa[cols])
        W_tr, w_sa = rd.whiten(Z_tr, z_sa)
        # `bootstrap_comparison` recomputes the reference-reference kernel on every null draw, so
        # the reference is subsampled: at 49k rows that term alone is a 49k^2 matrix, 500 times.
        ref = W_tr[rng.choice(len(W_tr), min(n_ref, len(W_tr)), replace=False)].astype(np.float32)
        d, null = bootstrap_comparison(w_sa[None].astype(np.float32), ref,
                                       maximum_mean_discrepancy, num_null_samples=n_null)
        p = 100.0 * float((null < d).mean())
        print(f"  {label:26s} MMD={d:10.5f}  p{p:6.2f} of the null   "
              f"(null median {np.median(null):.5f}, max {null.max():.5f})", flush=True)
        return d, null, p

    mmd_all = mmd(present_cols, "all present bands")
    mmd_per = {k: mmd(np.arange(F_tr.shape[1])[blk[k]], k) for k in keys}

    ref_dict = {k: np.concatenate([b[k] for b in ref_raw]) for k in ref_raw[0]}
    d_ss, null_ss = summary_space_comparison(real, ref_dict, approx, num_null_samples=200)
    print(f"  {'summary_space_comparison':26s} MMD={d_ss:10.5f}  "
          f"p{100 * float((null_ss < d_ss).mean()):6.2f} of the null   "
          f"(library default: raw unwhitened summaries, {len(next(iter(ref_dict.values())))} rows)",
          flush=True)

    from bayesflow.diagnostics.plots import mmd_hypothesis_test
    rd.save(mmd_hypothesis_test(mmd_all[1], mmd_all[0]), out_dir, "summary_mmd")

    # ── save the embeddings ──────────────────────────────────────────────────────────
    # They cost a full augmented pass over the training set to produce, so anything else that
    # wants this space (a PCA, a neighbour hunt, another disk) reads them from here instead.
    np.savez_compressed(os.path.join(out_dir, "summaries.npz"), train=F_tr.astype(np.float32),
                        real=f_sa.astype(np.float32), branch_keys=np.array(keys),
                        absent=np.array(absent),
                        has_cavity=np.asarray(data["has_cavity"][:len(F_tr)]).reshape(-1),
                        sil_id=np.asarray(data["sil_id"][:len(F_tr)]).reshape(-1),
                        mmd_observed=np.array([mmd_all[0]]), mmd_null=mmd_all[1],
                        mmd_per_branch=np.array([mmd_per[k][0] for k in keys]),
                        mmd_per_branch_p=np.array([mmd_per[k][2] for k in keys]))

    # ── figure ───────────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    for ax, (label, (dt, ds, p)) in zip(axes.ravel(), [("all present bands",
                                                        (d2_tr, d2_sa, pct_all))] + list(per.items())):
        ax.hist(dt, bins=80, color="0.75", log=True)
        ax.axvline(ds, color="#D55E00" if p > 99 else "#0072B2", lw=2)
        ax.set_title(f"{label}\n{DISK} at p{p:.2f}" + ("  (band absent)" if label in absent else ""),
                     fontsize=10)
        ax.set_xlabel("Mahalanobis $d^2$ (rank-normal summary space)")
    fig.suptitle(f"Summary-space population check — {DISK} vs {len(F_tr)} augmented training rows\n"
                 f"{model_dir}", fontsize=11)
    fig.tight_layout()
    rd.save(fig, out_dir, "summary_mahalanobis")

    fig, ax = plt.subplots(figsize=(12, 3.4))
    x = np.arange(len(ranks))
    colours = ["#D55E00" if any(k in absent and blk[k].start <= i < blk[k].stop for k in keys)
               else "#0072B2" for i in x]
    ax.bar(x, ranks, color=colours, width=0.85)
    for k, sl in blk.items():
        ax.axvline(sl.stop - 0.5, color="0.7", lw=0.8)
        ax.text((sl.start + sl.stop) / 2 - 0.5, 103, k, ha="center", fontsize=9)
    ax.axhspan(1, 99, color="0.9", zorder=-5)
    ax.set_ylim(0, 108); ax.set_ylabel("percentile rank in training")
    ax.set_xlabel("summary dimension")
    ax.set_title(f"{DISK}: per-dimension position in the training summary distribution "
                 f"(grey band = central 98%)", fontsize=10)
    rd.save(fig, out_dir, "summary_dim_ranks")

    # ── PCA of the space the flow actually conditions on ─────────────────────────────
    # On the present bands only, and on the raw standardized summary vector rather than the
    # rank-normal one: this is the geometry the network sees, not a monotone re-mapping of it.
    # SVD rather than sklearn -- two lines, one fewer dependency.
    P = F_tr[:, present_cols]
    mu = P.mean(0)
    _u, sv, vt = np.linalg.svd(P - mu, full_matrices=False)
    var = sv ** 2 / (len(P) - 1)
    frac = var / var.sum()
    pcs = (P - mu) @ vt[:2].T
    pc_sa = (f_sa[present_cols] - mu) @ vt[:2].T
    print(f"\nPCA of the {len(present_cols)}-d present-band summary: "
          f"PC1 {100 * frac[0]:.1f}%, PC2 {100 * frac[1]:.1f}% of variance "
          f"({100 * frac[:2].sum():.1f}% together)")
    for a, name in ((0, "PC1"), (1, "PC2")):
        print(f"  {DISK} {name}: {pc_sa[a]:+.2f}  ({(pc_sa[a] - pcs[:, a].mean()) / pcs[:, a].std():+.2f}"
              f" sd, percentile {rd.percentile_rank(pcs[:, a], pc_sa[a]):.2f})")

    cav = np.asarray(data["has_cavity"][:len(F_tr)]).reshape(-1) > 0.5
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), sharex=True, sharey=True)
    axes[0].hexbin(pcs[:, 0], pcs[:, 1], gridsize=90, cmap="Greys", bins="log", mincnt=1)
    axes[0].set_title(f"all {len(pcs)} augmented training rows", fontsize=10)
    for on, colour, label in ((~cav, "#0072B2", "no cavity"), (cav, "#D55E00", "cavity")):
        axes[1].scatter(pcs[on, 0], pcs[on, 1], s=1.5, alpha=.12, c=colour, lw=0, label=label)
    # The cavity split is the open question from the real-disk corners: if the two clouds separate
    # here and the disk lands in one, that is the same information the has_cavity condition carries.
    axes[1].legend(markerscale=8, fontsize=9, loc="upper left", framealpha=.9)
    axes[1].set_title("coloured by the training row's has_cavity", fontsize=10)
    for ax in axes:
        ax.scatter(*pc_sa, marker="*", s=420, c="#111111", zorder=5)
        ax.scatter(*pc_sa, marker="*", s=200, c="#FFD400", zorder=6, label=DISK)
        ax.set_xlabel(f"PC1  ({100 * frac[0]:.1f}% of variance)")
    axes[0].set_ylabel(f"PC2  ({100 * frac[1]:.1f}%)")
    axes[0].annotate(DISK, pc_sa, textcoords="offset points", xytext=(14, 10), fontsize=10,
                     weight="bold")
    fig.suptitle(f"Summary-space PCA — {DISK} against the training population "
                 f"({', '.join(k for k in keys if k not in absent)})\n{model_dir}", fontsize=11)
    fig.tight_layout()
    rd.save(fig, out_dir, "summary_pca")

    print(f"\nfigures -> {out_dir}")


if __name__ == "__main__":
    main()

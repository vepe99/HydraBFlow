"""Is the real disk *interior* to the training population in summary space, or on its edge?

    uv run python notebooks/prior_predictive_checks/summary_edge.py [plot_dir ...]
    CHECK_TAG=_train47811 uv run python notebooks/prior_predictive_checks/summary_edge.py

Reads the `summary_neighbours{tag}.npz` that `summary_neighbours.py` already cached -- the reference
embeddings and the real disk's -- so this costs no GPU and no re-embedding.  `CHECK_TAG` picks the
pool: empty for the 2000 held-out rows, `_train<n>` for the training-row pool.

A low Mahalanobis d^2 says "not an outlier"; it does **not** say "interior".  d^2 is one radius, and
a radius is blind to *which* direction you went: a point can sit at an unremarkable distance from the
centroid in a direction where the training set has no rows at all.  Two statistics,
in the whitened rank-normal space `check_summary_range` and `summary_neighbours` score in:

* **Local sparsity** -- the query's k-th nearest-neighbour distance.  Large means it lives in a
  thinner part of the cloud than the population does.
* **Radius** -- for context, and because it is what d^2 already reported.

**Both are calibrated against a null**, and that is not optional.  `m_null` reference rows are
re-scored as queries against the same pool with the same transform, and what is reported is the real
disk's percentile *within that null*.  A disk at p50 is an ordinary member however extreme its raw
numbers look; only a high percentile is edge.

A third statistic was tried and **removed**: the fraction of rows projecting further out along the
unit vector from the centroid to the query.  It cannot discriminate.  The direction is derived from
the query itself, so every point maximises its own projection, and the null over interior sims comes
out p50 = 0.000, p95 = 0.001 -- identical to a real disk sitting on the boundary.  Do not re-add it
without a direction chosen independently of the query.

Nothing here is O(n^2): the null uses `m_null` sampled queries, never all pairs, so the pool can be
the whole training half.

Reported twice: over all dims, and in the PCA subspace holding 99% of the population's variance
(`check_summary_pca`'s decomposition, on the raw summaries -- see `_pca_subspace`).
"""

from __future__ import annotations

import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import _realdisk as rd  # noqa: E402

TAG = os.environ.get("CHECK_TAG", "")
PLOTS_8NN = pathlib.Path("assets/protoplan/extracted_fits/plots/8nn")


def _pca_subspace(F, f, var_frac: float = 0.99):
    """
    `(scores_train, score_real, k)` in the leading PCs of the **raw** summary space.

    The PCA has to happen before the whitening, not after: whitening sets the covariance to the
    identity, so every direction then carries equal variance and a 99% cut keeps all of them --
    a vacuous no-op.  `check_summary_pca` decomposes the raw space for the same reason.
    """
    mu = F.mean(0)
    _u, sv, vt = np.linalg.svd(F - mu, full_matrices=False)
    keep = int(np.searchsorted(np.cumsum(sv ** 2) / (sv ** 2).sum(), var_frac) + 1)
    V = vt[:keep].T
    return (F - mu) @ V, (f - mu) @ V, keep


def query_stats(W, q, mu, k: int, drop_self: bool = False):
    """
    `(radius, knn_dist)` for one query `q` against pool `W`.

    `drop_self` is for a null query, which *is* a member of the pool: its own zero distance would
    otherwise always be its own nearest neighbour.
    """
    d = np.linalg.norm(W - q[None, :], axis=1)
    if drop_self:
        d = d[d > 0]
    return float(np.linalg.norm(q - mu)), float(np.partition(d, k - 1)[k - 1])


def report(W, w, label: str, k: int = 7, m_null: int = 1000, seed: int = 0) -> dict:
    n, p = W.shape
    mu = W.mean(0)
    r_sa, knn_sa = query_stats(W, w, mu, k)

    # The null: reference rows re-scored as queries, same pool, same transform.
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(m_null, n), replace=False)
    null = np.array([query_stats(W, W[i], mu, k, drop_self=True) for i in idx])
    r_0, knn_0 = null.T

    def line(name, value, ref):
        pctile = rd.percentile_rank(ref, value)
        return (f"    {name:<18s} {value:9.3f}   null p50={np.median(ref):8.3f} "
                f"p95={np.percentile(ref, 95):8.3f}  -> p{pctile:5.1f}"
                + ("   <<< EDGE" if pctile > 99.0 else ""))

    print(f"\n  {label}  ({n} rows, {p} dims, null over {len(idx)} sims)")
    print(line("radius", r_sa, r_0))
    print(line(f"{k}-NN distance", knn_sa, knn_0))
    return dict(n=n, p=p, r=r_sa, knn=knn_sa,
                r_pct=rd.percentile_rank(r_0, r_sa),
                knn_pct=rd.percentile_rank(knn_0, knn_sa))


def check(plot_dir, k: int = 7, var_frac: float = 0.99, tag: str = TAG) -> dict:
    path = pathlib.Path(plot_dir) / f"summary_neighbours{tag}.npz"
    z = np.load(path)
    keys = [str(s) for s in z["branch_keys"]]
    absent = [str(s) for s in z["absent"]] if z["absent"].size else []
    F, f = np.asarray(z["test"], np.float64), np.asarray(z["real"], np.float64)

    # Same column selection the neighbour distances used: an absent band's branch is a zero image.
    import check_summary_range as csr
    blk = csr.blocks(keys, F.shape[1])
    cols = np.concatenate([np.arange(F.shape[1])[blk[b]] for b in keys if b not in absent])

    Fc, fc = F[:, cols], f[cols]
    print(f"\n{'=' * 78}\n{path.parent.name}{tag}  (PA {float(z['rot_deg']):.2f} deg, "
          f"branches {[b for b in keys if b not in absent]})\n{'=' * 78}")
    out = {"raw": report(*rd.whiten(*rd.normal_score_block(Fc, fc)), "all dims", k)}
    S, s_sa, keep = _pca_subspace(Fc, fc, var_frac)
    out["pca"] = report(*rd.whiten(*rd.normal_score_block(S, s_sa)),
                        f"PCA {100 * var_frac:g}%-variance subspace "
                        f"({keep} of {Fc.shape[1]} PCs)", k)
    return out


if __name__ == "__main__":
    for d in sys.argv[1:] or sorted(str(p) for p in PLOTS_8NN.glob("summary_*")):
        check(d)

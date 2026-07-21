"""t-SNE (and PCA) visualization of the fused summary space.

Embeds the simulated reference summaries together with the real observed
stream members so we can see where the true Gaia data lands relative to the
simulation manifold. Colors sim points by stream identity (j); overlays the
real members as large stars.

Standalone: numpy + scikit-learn + matplotlib only.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

STREAM_NAMES = {0: "Pal5", 1: "NGC3201", 2: "M68"}
STREAM_COLORS = {0: "#4C72B0", 1: "#DD8452", 2: "#55A868"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default="outputs/ibata_onedisk_grid_m200c/tuning/"
                    "best_trial36/eval_real/summaries.npz")
    ap.add_argument("--ref", default="outputs/ibata_onedisk_grid_m200c/tuning/"
                    "best_trial36/eval_sim/summaries.npz")
    ap.add_argument("--out", default="outputs/ibata_onedisk_grid_m200c/tuning/"
                    "best_trial36/eval_real/tsne_summaries.png")
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--n-neighbors", type=int, default=15)
    ap.add_argument("--min-dist", type=float, default=0.1)
    ap.add_argument("--method", choices=["tsne", "umap"], default="tsne")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    ref = np.load(args.ref, allow_pickle=True)
    real = np.load(args.real, allow_pickle=True)
    X_ref = np.asarray(ref["summaries"], dtype=np.float64)
    X_real = np.asarray(real["summaries"], dtype=np.float64)
    j_ref = np.asarray(ref["j"]).astype(int)
    j_real = np.asarray(real["j"]).astype(int)

    print(f"reference {X_ref.shape}  real {X_real.shape}")

    # Standardize using the reference (sim) statistics only.
    scaler = StandardScaler().fit(X_ref)
    Zr = scaler.transform(X_ref)
    Zo = scaler.transform(X_real)
    Z = np.vstack([Zr, Zo])

    # Nonlinear embedding on the combined set.
    if args.method == "umap":
        import umap
        reducer = umap.UMAP(n_components=2, n_neighbors=args.n_neighbors,
                            min_dist=args.min_dist, random_state=args.seed)
        emb = reducer.fit_transform(Z)
        nl_title = f"UMAP (n_neighbors={args.n_neighbors}, min_dist={args.min_dist})"
        nl_xl, nl_yl = "UMAP 1", "UMAP 2"
    else:
        perp = min(args.perplexity, (len(Z) - 1) / 3.0)
        reducer = TSNE(n_components=2, perplexity=perp, init="pca",
                       random_state=args.seed, learning_rate="auto")
        emb = reducer.fit_transform(Z)
        nl_title = f"t-SNE (perplexity={perp:.0f})"
        nl_xl, nl_yl = "t-SNE 1", "t-SNE 2"
    emb_ref, emb_real = emb[: len(Zr)], emb[len(Zr):]

    # PCA as a linear reference view.
    pca = PCA(n_components=2, random_state=args.seed).fit(Zr)
    pr, po = pca.transform(Zr), pca.transform(Zo)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for ax, (er, eo, title, xl, yl) in zip(
        axes,
        [(emb_ref, emb_real, nl_title, nl_xl, nl_yl),
         (pr, po, f"PCA (EVR {pca.explained_variance_ratio_[:2].sum():.2f})",
          "PC1", "PC2")],
    ):
        for j in sorted(STREAM_NAMES):
            m = j_ref == j
            ax.scatter(er[m, 0], er[m, 1], s=14, alpha=0.45,
                       c=STREAM_COLORS[j], label=f"sim {STREAM_NAMES[j]}",
                       edgecolors="none")
        for j in sorted(STREAM_NAMES):
            m = j_real == j
            if not m.any():
                continue
            ax.scatter(eo[m, 0], eo[m, 1], s=420, marker="*",
                       c=STREAM_COLORS[j], edgecolors="black", linewidths=1.6,
                       zorder=5, label=f"real {STREAM_NAMES[j]}")
        ax.set_title(title)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
    axes[0].legend(fontsize=8, loc="best", framealpha=0.9)
    fig.suptitle("Fused summary space: simulated reference vs. real Gaia streams",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

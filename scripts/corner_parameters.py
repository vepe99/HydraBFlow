"""Corner plot of parameter draws from a simulate dataset (offline helper — not a Hydra stage).

Reads one or more datasets (a flat ``simulate`` ``.npz``, a compositional ``simulate_multistream``
``.npz``, a resumable-simulator ``<dataset>.chunks/`` directory, or several of these) and corner-
plots the scalar parameter columns. This is handy as a prior-predictive view of the *effective*
prior the network trains on — e.g. after a ``vcirc_rejection`` cut the global-potential prior is
no longer the flat config box but a correlated, truncated distribution.

Parameter columns are auto-detected: every key whose array is one scalar per row (shape ``(n, 1)``,
or ``(n, m, 1)`` for a multistream file — flattened over the member axis) that is not an observable
or an index. Constant (``identity``) parameters are dropped automatically. Restrict or reorder with
``--params``; drop extras with ``--exclude``. Large-magnitude columns are auto-scaled by a power of
ten for readable axes (reflected in the label).

Usage:
    # inferred global potential params over all available training chunks
    python scripts/corner_parameters.py data/streams/data_agama_rnbody_hydrabflow/training_data_30000.chunks \
        OUT_DIR --params rho_TwoPowerTriaxial_halo gamma_TwoPowerTriaxial_halo \
        a_TwoPowerTriaxial_halo q_TwoPowerTriaxial_halo r_Disk z_Disk Sigma_Disk

    # everything non-constant in a finished dataset
    python scripts/corner_parameters.py data/.../training_data_30000.npz OUT_DIR

Writes ``corner.png`` (name override via ``--name``) and prints a percentile summary.
"""

from __future__ import annotations

import argparse
import glob
import math
import os

import matplotlib

matplotlib.use("Agg")
import corner  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Keys that are never parameters (observables + the stream index). Includes the Ibata ancillary
# potential observables (vterm_kms/sigma_z/rho_z) so they are not mistaken for scalar parameters.
NON_PARAM_KEYS = {
    "sim_data_projected", "sim_data_carthesian", "vcirc_kms", "vterm_kms", "sigma_z", "rho_z", "j"
}


def expand_inputs(paths: list[str]) -> list[str]:
    """Expand each input into concrete .npz files: a directory -> its sorted chunk_*.npz (or any
    .npz it holds); a glob -> its matches; a file -> itself."""
    files: list[str] = []
    for p in paths:
        if os.path.isdir(p):
            hits = sorted(glob.glob(os.path.join(p, "chunk_*.npz"))) or sorted(
                glob.glob(os.path.join(p, "*.npz"))
            )
            if not hits:
                raise FileNotFoundError(f"No .npz found in directory {p}")
            files.extend(hits)
        elif any(c in p for c in "*?["):
            files.extend(sorted(glob.glob(p)))
        else:
            files.append(p)
    if not files:
        raise FileNotFoundError(f"No datasets resolved from {paths}")
    return files


def detect_param_keys(sample: np.lib.npyio.NpzFile) -> list[str]:
    """Scalar-per-row keys that are not observables/indices, in file order."""
    keys = []
    for k in sample.files:
        if k in NON_PARAM_KEYS:
            continue
        shp = sample[k].shape
        if len(shp) >= 2 and shp[-1] == 1:  # (n,1) or (n,m,1)
            keys.append(k)
    return keys


def load_columns(files: list[str], keys: list[str]) -> dict[str, np.ndarray]:
    """Concatenate each key across files, flattening any member axis to one column per row."""
    cols = {k: [] for k in keys}
    for f in files:
        d = np.load(f, allow_pickle=True)
        for k in keys:
            cols[k].append(d[k].reshape(-1))
    return {k: np.concatenate(v) for k, v in cols.items()}


def autoscale(vals: np.ndarray, label: str) -> tuple[np.ndarray, str]:
    """Divide by a power of ten when the typical magnitude is large/small, and note it in the label."""
    med = np.median(np.abs(vals[np.isfinite(vals)])) if len(vals) else 0.0
    if med == 0 or not np.isfinite(med) or 1e-2 <= med < 1e4:
        return vals, label
    exp = int(math.floor(math.log10(med)))
    return vals / 10.0**exp, f"{label} [1e{exp}]"


def main() -> None:
    ap = argparse.ArgumentParser(description="Corner plot of simulate-dataset parameters.")
    ap.add_argument("inputs", nargs="+", help=".npz file(s), a chunks/ dir, or a glob")
    ap.add_argument("out_dir", help="output directory")
    ap.add_argument("--params", nargs="*", default=None, help="restrict/reorder to these keys")
    ap.add_argument("--exclude", nargs="*", default=[], help="drop these keys")
    ap.add_argument("--name", default="corner.png", help="output filename")
    ap.add_argument("--bins", type=int, default=40)
    ap.add_argument("--title", default=None, help="figure suptitle")
    args = ap.parse_args()

    # out_dir is the LAST positional; argparse put it in out_dir, inputs holds the rest.
    os.makedirs(args.out_dir, exist_ok=True)
    files = expand_inputs(args.inputs)

    sample = np.load(files[0], allow_pickle=True)
    keys = args.params if args.params else detect_param_keys(sample)
    keys = [k for k in keys if k not in set(args.exclude)]
    missing = [k for k in keys if k not in sample.files]
    if missing:
        raise KeyError(f"requested params not in dataset: {missing}")

    print(f"combining {len(files)} file(s); candidate params: {keys}", flush=True)
    raw = load_columns(files, keys)

    labels, data_cols = [], []
    for k in keys:
        v = raw[k]
        if np.allclose(v, v.flat[0]):  # constant (identity) -> skip
            print(f"  skip constant param '{k}' (= {v.flat[0]:.4g})")
            continue
        scaled, label = autoscale(v, k)
        data_cols.append(scaled)
        labels.append(label)

    if len(data_cols) < 2:
        raise SystemExit(f"need >=2 non-constant params to corner-plot, got {len(data_cols)}")

    data = np.column_stack(data_cols)
    n = data.shape[0]
    fig = corner.corner(
        data, labels=labels, bins=args.bins, color="C0",
        quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt=".2f",
        title_kwargs={"fontsize": 9}, label_kwargs={"fontsize": 10},
        plot_datapoints=False, fill_contours=True, smooth=1.0,
        hist_kwargs={"density": True},
    )
    fig.suptitle(args.title or f"Parameter draws (n={n:,}, {len(labels)} params)",
                 fontsize=12, y=1.02)
    path = os.path.join(args.out_dir, args.name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)

    for lab, c in zip(labels, data_cols):
        q = np.percentile(c, [16, 50, 84])
        print(f"  {lab:26s} median={q[1]:.4g}  16-84=[{q[0]:.4g}, {q[2]:.4g}]")


if __name__ == "__main__":
    main()

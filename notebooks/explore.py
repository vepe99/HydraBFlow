"""Marimo notebook: inspect a training run's posterior samples and diagnostics.

Run with:  uv run marimo edit notebooks/explore.py
Point RUN_DIR at a completed evaluate output dir (it must contain posterior.npz).
"""

import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    run_dir = mo.ui.text(
        value="outputs/two_moons/set_transformer+flow_matching/REPLACE_WITH_TIMESTAMP",
        label="Run dir (contains posterior.npz)",
        full_width=True,
    )
    run_dir
    return mo, run_dir


@app.cell
def _(run_dir):
    import os

    import numpy as np

    path = os.path.join(run_dir.value, "posterior.npz")
    posterior = dict(np.load(path)) if os.path.exists(path) else {}
    posterior_keys = list(posterior)
    posterior_keys
    return np, os, path, posterior, posterior_keys


@app.cell
def _(np, posterior, posterior_keys):
    import matplotlib.pyplot as plt

    if posterior_keys:
        fig, axes = plt.subplots(1, len(posterior_keys), figsize=(4 * len(posterior_keys), 3))
        axes = np.atleast_1d(axes)
        for ax, key in zip(axes, posterior_keys):
            ax.hist(np.asarray(posterior[key]).reshape(-1), bins=50)
            ax.set_title(key)
        fig.tight_layout()
        out = fig
    else:
        out = "No posterior.npz found — set the run dir above."
    out
    return (out,)


if __name__ == "__main__":
    app.run()

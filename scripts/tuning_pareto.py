"""Collate an Optuna study's trials into one table and plot its Pareto fronts.

    python scripts/tuning_pareto.py <study_dir> [--out <dir>]

<study_dir> = ${tuning.artifacts_dir} (holds trials/trial_XXXX/ and, one level up, <study>.log).
Per trial: Optuna objectives (val base RMSE / calibration, from the journal), the watcher's
eval_sim_333 base + compositional metrics, real-Gaia q_halo median/68% interval + MMD, and the
sampled hyperparameters. Writes trials.csv, pareto_val.png (Optuna objectives, front highlighted),
pareto_compositional.png (compositional RMSE vs calibration on the 333 test set) and
real_q_vs_calib.png. Re-run any time
trials without an evaluation yet appear with NaNs.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def collect(study_dir: str) -> pd.DataFrame:
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend

    name = os.path.basename(os.path.normpath(study_dir))
    log = os.path.join(os.path.dirname(os.path.normpath(study_dir)), f"{name}.log")
    study = optuna.load_study(study_name=name, storage=JournalStorage(JournalFileBackend(log)))
    front = {t.number for t in study.best_trials}
    rows = []
    for t in study.trials:
        if t.values is None:
            continue
        d = os.path.join(study_dir, "trials", f"trial_{t.number:04d}")
        row = {"trial": t.number, "state": str(t.state).split(".")[-1], "pareto_val": t.number in front,
               "val_rmse": t.values[0], "val_calib": t.values[1] if len(t.values) > 1 else np.nan,
               "duration_h": (t.duration.total_seconds() / 3600 if t.duration else np.nan)}
        for prefix in ("base", "compositional"):
            m = _load(os.path.join(d, "eval_sim_333", f"{prefix}_metrics.json"))
            row[f"{prefix}_rmse"] = m["rmse"]["mean"] if m else np.nan
            row[f"{prefix}_calib"] = m["calibration_error"]["mean"] if m else np.nan
        post = os.path.join(d, "eval_real", "posterior.npz")
        if os.path.exists(post):
            q = np.load(post)["q_TwoPowerTriaxial_halo"].reshape(-1)
            row["real_q16"], row["real_q50"], row["real_q84"] = np.percentile(q, [16, 50, 84])
        mmd = _load(os.path.join(d, "eval_real", "misspecification.json"))
        if mmd:
            row["real_mmd"] = mmd.get("mmd_observed")
            row["real_mmd_p_strat"] = mmd.get("p_value_stratified")
        row.update({k.split(".")[-1] if k.count(".") < 4 else ".".join(k.split(".")[-2:]): v
                    for k, v in t.params.items()})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("trial")


def _pareto_mask(x, y):
    pts = np.column_stack([x, y])
    ok = np.isfinite(pts).all(1)
    mask = np.zeros(len(pts), bool)
    for i in np.where(ok)[0]:
        mask[i] = not np.any((pts[ok] <= pts[i]).all(1) & (pts[ok] < pts[i]).any(1))
    return mask


def _scatter(df, x, y, front, path, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df[x], df[y], c="0.6", s=30, label="trials")
    f = df[front].sort_values(x)
    ax.plot(f[x], f[y], "o-", color="tab:red", ms=6, label="Pareto front")
    for _, r in df.iterrows():
        if np.isfinite(r[x]) and np.isfinite(r[y]):
            ax.annotate(int(r["trial"]), (r[x], r[y]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("study_dir")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or a.study_dir
    df = collect(a.study_dir)
    df.to_csv(os.path.join(out, "trials.csv"), index=False)
    _scatter(df, "val_rmse", "val_calib", df["pareto_val"].values, os.path.join(out, "pareto_val.png"),
             "Optuna objectives (val split, per-member)")
    if df["compositional_rmse"].notna().any():
        m = _pareto_mask(df["compositional_rmse"].values, df["compositional_calib"].values)
        _scatter(df, "compositional_rmse", "compositional_calib", m,
                 os.path.join(out, "pareto_compositional.png"), "Compositional (333-group test set)")
    if "real_q50" in df and df["real_q50"].notna().any():
        fig, ax = plt.subplots(figsize=(6, 5))
        ok = df["real_q50"].notna()
        ax.errorbar(df.loc[ok, "compositional_calib"], df.loc[ok, "real_q50"],
                    yerr=[df.loc[ok, "real_q50"] - df.loc[ok, "real_q16"], df.loc[ok, "real_q84"] - df.loc[ok, "real_q50"]],
                    fmt="o", color="tab:blue", ecolor="0.7")
        for _, r in df[ok].iterrows():
            ax.annotate(int(r["trial"]), (r["compositional_calib"], r["real_q50"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("compositional calibration error (sim)")
        ax.set_ylabel("real Gaia q_halo (pooled)")
        ax.grid(alpha=0.3)
        fig.savefig(os.path.join(out, "real_q_vs_calib.png"), dpi=130, bbox_inches="tight")
    cols = [c for c in ["trial", "state", "pareto_val", "val_rmse", "val_calib", "base_rmse", "base_calib",
                        "compositional_rmse", "compositional_calib", "real_q50", "real_mmd", "duration_h"] if c in df]
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n{len(df)} trials -> {out}/trials.csv, pareto_*.png")


if __name__ == "__main__":
    main()

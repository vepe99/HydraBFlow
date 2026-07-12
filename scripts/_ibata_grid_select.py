"""Milestone best-trial selector for the Ibata gridded-summary tuning study (offline helper for
scripts/tune_ibata_onedisk_grid.sh — NOT a Hydra run stage).

Opens the concurrency-safe Optuna JournalStorage study read-only and reconstructs, for three
milestones defined as trial-completion prefixes of the study — **1/3**, **1/2** and **all** of the
target trials — the best 3 trials as they would have stood at that point. "Best" = lowest
validation RMSE (the trial's ``rmse`` user attribute set by the tuning stage), tie-broken by
calibration error; trials that diverged (penalized RMSE) sort last and are only picked if fewer
than three finite trials exist.

Because Optuna records every trial's completion timestamp, the "at 1/3 of the run" state is fully
reconstructable after the fact — no need to pause the concurrent workers mid-study. This keeps the
GPUs saturated with tuning and defers all evaluation to the end.

Subcommands (all read-only):
  --list-trials   Print, one per line, the UNION of milestone-selected trials (deduplicated):
                      <trial_number>\t<artifact_dir>\t<space-joined hydra overrides>
                  Consumed by the bash driver to run evaluate / evaluate_real once per trial.
  --milestone NAME  Print the 3 selected trial numbers (rank order) for milestone NAME
                    (one of: third half full), space-separated.

Common args:
  --storage-log PATH     the study's JournalStorage .log
  --study-name NAME      Optuna study name
  --artifacts-dir DIR    tuning artifacts dir (trials live at DIR/trials/trial_XXXX)
  --total N              target total trial count (milestone cutoffs derive from it; default 50)
  --top-k K              trials per milestone (default 3)
"""

from __future__ import annotations

import argparse
import math
import os
import sys

_NAN_PENALTY_RMSE = 1.0e3  # must match pipeline/tune.py
_MILESTONES = ("third", "half", "full")


def _load_completed(storage_log: str, study_name: str):
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(
        study_name=study_name, storage=JournalStorage(JournalFileBackend(storage_log))
    )
    trials = [
        t
        for t in study.get_trials(deepcopy=False)
        if t.state == optuna.trial.TrialState.COMPLETE and t.datetime_complete is not None
    ]
    trials.sort(key=lambda t: t.datetime_complete)  # completion order = "as the run progressed"
    return trials


def _score(t):
    """Ranking key: (rmse, calibration_error); missing user-attrs fall back to objective values."""
    rmse = t.user_attrs.get("rmse")
    if rmse is None:
        rmse = t.values[0] if t.values else _NAN_PENALTY_RMSE
    cal = t.user_attrs.get("calibration_error")
    if cal is None:
        cal = t.values[1] if t.values and len(t.values) > 1 else 1.0
    return (float(rmse), float(cal))


def _cutoffs(total: int, n_available: int) -> dict[str, int]:
    raw = {"third": math.ceil(total / 3), "half": math.ceil(total / 2), "full": total}
    # Clamp to what actually completed (tuning may have been stopped early or overshot).
    return {name: min(c, n_available) for name, c in raw.items()}


def _best_for_cutoff(trials, cutoff: int, top_k: int):
    prefix = trials[:cutoff]
    ranked = sorted(prefix, key=_score)
    return ranked[:top_k]


def _artifact_dir(artifacts_dir: str, number: int) -> str:
    return os.path.join(artifacts_dir, "trials", f"trial_{number:04d}")


def _overrides(t) -> str:
    parts = []
    for path, value in sorted(t.params.items()):
        if isinstance(value, bool):
            value = str(value).lower()
        parts.append(f"{path}={value}")
    return " ".join(parts)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--storage-log", required=True)
    ap.add_argument("--study-name", required=True)
    ap.add_argument("--artifacts-dir", required=True)
    ap.add_argument("--total", type=int, default=50)
    ap.add_argument("--top-k", type=int, default=3)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list-trials", action="store_true")
    g.add_argument("--milestone", choices=_MILESTONES)
    args = ap.parse_args(argv)

    trials = _load_completed(args.storage_log, args.study_name)
    if not trials:
        sys.stderr.write("No completed trials found in study.\n")
        return 1
    cutoffs = _cutoffs(args.total, len(trials))

    if args.milestone:
        best = _best_for_cutoff(trials, cutoffs[args.milestone], args.top_k)
        print(" ".join(str(t.number) for t in best))
        return 0

    # --list-trials: union across all milestones, deduplicated, stable order (by trial number).
    selected = {}
    for name in _MILESTONES:
        for t in _best_for_cutoff(trials, cutoffs[name], args.top_k):
            selected.setdefault(t.number, t)
    for number in sorted(selected):
        t = selected[number]
        print(f"{number}\t{_artifact_dir(args.artifacts_dir, number)}\t{_overrides(t)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

"""Training-convergence inspection and Markdown report generation.

Two pure-Python helpers (no BayesFlow / Keras import, so they are cheap and safe to call from
any stage):

* :func:`inspect_history` — turns a Keras ``history.history`` dict into a structured convergence
  report (NaN check, overfitting, under-training) with a go/no-go verdict. Mirrors the
  amortized-workflow skill's ``inspect_training`` contract.
* :func:`write_report` — assembles a self-contained ``report.md`` from the artifacts an
  ``evaluate`` run leaves behind (``*metrics.json`` files) plus the training run's
  ``convergence.json`` (if reachable through ``model_dir``).

Both are deliberately defensive: they never raise on malformed input, because callers wire them
in as best-effort steps that must not abort a training or evaluation run.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

from hydrabflow.utils.logging import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------------------------- #
# Convergence inspection
# --------------------------------------------------------------------------------------------- #


def _finite(values) -> list[float]:
    out = []
    for v in values or []:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return out


def _has_nonfinite(values) -> bool:
    for v in values or []:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return True
        if not math.isfinite(f):
            return True
    return False


def inspect_history(history: dict, *, overfit_ratio: float = 1.1) -> dict[str, Any]:
    """Structured convergence report for a Keras ``history.history`` dict.

    Checks, each with an ``ok`` flag:

    * ``nan`` — any non-finite value in ``loss`` / ``val_loss``.
    * ``overfitting`` — final ``val_loss`` more than ``overfit_ratio`` x its best value.
    * ``under_training`` — training loss still descending over the last 10% of epochs
      (relative drop > 1% vs. the preceding window), i.e. more epochs would likely help.

    Returns ``{"checks": {...}, "overall": {"ok": bool, "issues": [str, ...]}}``. Never raises.
    """
    checks: dict[str, Any] = {}
    issues: list[str] = []

    loss = (history or {}).get("loss") or []
    val = (history or {}).get("val_loss")

    # --- NaN / Inf ---
    nan = _has_nonfinite(loss) or (val is not None and _has_nonfinite(val))
    checks["nan"] = {"ok": not nan}
    if nan:
        issues.append("Non-finite (NaN/Inf) values in the loss curve — training diverged.")

    # --- Overfitting (needs a validation curve) ---
    fval = _finite(val) if val is not None else []
    if len(fval) >= 2:
        best = min(fval)
        final = fval[-1]
        ratio = final / best if best > 0 else (final / best if best != 0 else float("inf"))
        ok = math.isfinite(ratio) and ratio <= overfit_ratio
        checks["overfitting"] = {"ok": ok, "final_over_best_val_loss": ratio}
        if not ok:
            issues.append(
                f"Possible overfitting: final val_loss is {ratio:.2f}x its best "
                f"(threshold {overfit_ratio:.2f}x)."
            )
    else:
        checks["overfitting"] = {"ok": None, "reason": "no validation curve"}

    # --- Under-training (training loss still descending at the end) ---
    floss = _finite(loss)
    if len(floss) >= 10:
        w = max(1, len(floss) // 10)
        last = sum(floss[-w:]) / w
        prev = sum(floss[-2 * w : -w]) / w
        rel_drop = (prev - last) / abs(prev) if prev != 0 else 0.0
        still_descending = rel_drop > 0.01
        checks["under_training"] = {"ok": not still_descending, "recent_relative_drop": rel_drop}
        if still_descending:
            issues.append(
                f"Loss still descending over the last {w} epoch(s) "
                f"({rel_drop * 100:.1f}% drop) — more epochs may help."
            )
    else:
        checks["under_training"] = {"ok": None, "reason": "too few epochs to assess"}

    return {"checks": checks, "overall": {"ok": not issues, "issues": issues}}


# --------------------------------------------------------------------------------------------- #
# Markdown report
# --------------------------------------------------------------------------------------------- #


def _load_json(path: str) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _rate(name: str, value: float) -> str:
    """Very light qualitative rating for the report (thresholds intentionally lenient)."""
    if not math.isfinite(value):
        return "n/a"
    if name == "calibration_error":
        return "excellent" if value < 0.05 else "fair" if value < 0.1 else "poor"
    if name == "rmse":  # NRMSE-like; raw RMSE scale is problem-dependent, so keep coarse
        return "good" if value < 0.5 else "fair" if value < 1.0 else "poor"
    return ""


def _metrics_table(metrics: dict, param_names: list[str]) -> list[str]:
    """Render one ``metrics.json`` payload as a Markdown table (per-parameter + mean)."""
    rows = ["| parameter | " + " | ".join(metrics.keys()) + " |",
            "|" + "---|" * (len(metrics) + 1)]
    n = 0
    for m in metrics.values():
        n = max(n, len(m.get("values", [])))
    names = param_names if len(param_names) == n else [f"param_{i}" for i in range(n)]
    for i, pname in enumerate(names):
        cells = []
        for mname, m in metrics.items():
            vals = m.get("values", [])
            v = vals[i] if i < len(vals) else float("nan")
            rating = _rate(mname, v)
            cells.append(f"{v:.4g}" + (f" ({rating})" if rating else ""))
        rows.append(f"| {pname} | " + " | ".join(cells) + " |")
    mean_cells = [f"{m.get('mean', float('nan')):.4g}" for m in metrics.values()]
    rows.append("| **mean** | " + " | ".join(mean_cells) + " |")
    return rows


def write_report(
    run_dir: str,
    *,
    param_names: list[str] | None = None,
    model_dir: str | None = None,
    title: str = "Evaluation report",
    extra_lines: list[str] | None = None,
) -> str | None:
    """Assemble ``report.md`` in ``run_dir`` from the diagnostics already written there.

    Scans ``run_dir`` for ``*metrics.json`` (base/compositional/plain), embeds each as a table,
    pulls the training ``convergence.json`` from ``model_dir`` if present, and lists the
    diagnostic figures found. Returns the report path, or ``None`` if nothing could be written.
    Never raises.
    """
    try:
        param_names = list(param_names or [])
        lines = [f"# {title}", "", f"- Run directory: `{run_dir}`"]
        if model_dir:
            lines.append(f"- Model directory: `{model_dir}`")
        lines.append("")

        # --- Convergence (from the training run) ---
        conv = _load_json(os.path.join(model_dir, "convergence.json")) if model_dir else None
        if conv:
            overall = conv.get("overall", {})
            verdict = "PASS" if overall.get("ok") else "REVIEW"
            lines += ["## Training convergence", "", f"**Verdict: {verdict}**", ""]
            for issue in overall.get("issues", []) or []:
                lines.append(f"- {issue}")
            if not overall.get("issues"):
                lines.append("- No convergence issues detected (NaN / overfitting / under-training).")
            lines.append("")

        # --- Diagnostic metrics (one section per *metrics.json) ---
        metric_files = sorted(
            f for f in os.listdir(run_dir) if f.endswith("metrics.json")
        )
        if metric_files:
            lines += ["## In-silico diagnostics", ""]
        for fname in metric_files:
            payload = _load_json(os.path.join(run_dir, fname))
            if not payload:
                continue
            label = fname[: -len("metrics.json")].rstrip("_") or "metrics"
            lines += [f"### {label}", ""]
            lines += _metrics_table(payload, param_names)
            lines.append("")

        # --- Figures ---
        figs = sorted(f for f in os.listdir(run_dir) if f.endswith(".png"))
        if figs:
            lines += ["## Figures", ""]
            lines += [f"- `{f}`" for f in figs]
            lines.append("")

        if extra_lines:
            lines += extra_lines + [""]

        path = os.path.join(run_dir, "report.md")
        with open(path, "w") as f:
            f.write("\n".join(lines))
        return path
    except Exception as exc:  # report generation must never abort a run
        log.warning("Could not write report.md: %s", exc)
        return None

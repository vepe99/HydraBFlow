"""Summary-space model misspecification test (observed group vs simulated reference).

Implements the MMD hypothesis test of Schmitt et al. (2021, arXiv:2112.08866) on the trained
summary network's outputs, generalizing the reference project's
``model_mispecification_rotationcurve_agama.py`` with two upgrades:

* the bootstrap null is **stratified by member identity** (``j``): the observed group holds one
  member per stream, so each null draw matches that composition instead of drawing members
  uniformly (uniform trios usually contain duplicate streams, which shifts the null when the
  summaries encode stream identity — and ours do, ``j`` is a concatenated feature);
* a **per-member OOD score**: the Mahalanobis distance of each observed member's summary to the
  reference summaries of the *same* stream, reported as the percentile within that stream's own
  reference-distance distribution. The pooled MMD says *whether* the observation is atypical;
  these scores say *which member* drives it.

The reference summaries are a byproduct of the standard simulated evaluation:
``evaluate composition=global`` saves ``summaries.npz`` (via :func:`save_member_summaries`), and
``evaluate_real composition=global`` consumes it through ``eval.misspecification_reference``
(the evaluate run dir, or a direct path to the ``.npz``). Both sides push their member rows
through the *same* trained approximator — adapter renames (``summary_attention_mask``) and
internal standardization included — so the two summary clouds are directly comparable.

Everything here is best-effort diagnostics: the public entry points never raise, so they cannot
abort a (possibly chained) evaluation run.
"""

from __future__ import annotations

import json
import os
from typing import Callable, Mapping

import numpy as np

from hydrabflow.utils.logging import get_logger
from hydrabflow.utils.paths import MISSPECIFICATION_JSON, MMD_PLOT, SUMMARIES

log = get_logger(__name__)


def _bf_mmd(x: np.ndarray, y: np.ndarray) -> float:
    """Maximum mean discrepancy via bayesflow's functional metric (lazy import)."""
    from bayesflow.metrics.functional import maximum_mean_discrepancy
    from keras import ops

    return float(
        ops.convert_to_numpy(
            maximum_mean_discrepancy(
                ops.convert_to_tensor(np.asarray(x, dtype="float32")),
                ops.convert_to_tensor(np.asarray(y, dtype="float32")),
            )
        )
    )


def member_summaries(approximator, flat: Mapping[str, np.ndarray]) -> np.ndarray:
    """Summary-network outputs for flat member rows (adapter applied, ``strict=False``)."""
    return np.asarray(approximator.summarize(dict(flat)))


def save_member_summaries(approximator, flat, run_dir: str, name: str = SUMMARIES):
    """Best-effort: save the member-row summaries (+ ``j``) so this run can serve as the
    reference set of a later misspecification test. Returns the summaries or ``None``."""
    try:
        summaries = member_summaries(approximator, flat)
        j = np.asarray(flat.get("j", np.zeros(summaries.shape[0]))).reshape(-1)
        np.savez(os.path.join(run_dir, name), summaries=summaries, j=j)
        log.info("Member summaries saved to %s %s", name, summaries.shape)
        return summaries
    except Exception as exc:  # diagnostics must never abort an evaluation
        log.warning("member summaries not saved: %s", exc)
        return None


def _null_mmd(
    reference: np.ndarray,
    draw_indices: Callable[[np.random.Generator], np.ndarray],
    comparison_fn: Callable[[np.ndarray, np.ndarray], float],
    num_null: int,
    rng: np.random.Generator,
) -> np.ndarray:
    return np.array(
        [comparison_fn(reference[draw_indices(rng)], reference) for _ in range(num_null)]
    )


def mmd_test(
    observed: np.ndarray,
    reference: np.ndarray,
    observed_j: np.ndarray | None = None,
    reference_j: np.ndarray | None = None,
    num_null: int = 200,
    comparison_fn: Callable[[np.ndarray, np.ndarray], float] = _bf_mmd,
    rng: np.random.Generator | None = None,
) -> dict:
    """MMD(observed, reference) with bootstrap null distribution(s).

    Always computes the plain null (uniform bootstrap of ``len(observed)`` reference rows, as in
    ``bf.diagnostics.metrics.bootstrap_comparison``); when strata are given, also the stratified
    null matching the observed ``j`` composition. p-values are upper-tail.
    """
    rng = rng or np.random.default_rng()
    observed = np.asarray(observed, dtype=float)
    reference = np.asarray(reference, dtype=float)
    n_obs, n_ref = observed.shape[0], reference.shape[0]

    mmd_observed = comparison_fn(observed, reference)

    plain = _null_mmd(
        reference, lambda r: r.integers(0, n_ref, size=n_obs), comparison_fn, num_null, rng
    )
    out = {
        "mmd_observed": float(mmd_observed),
        "num_null": int(num_null),
        "n_observed": int(n_obs),
        "n_reference": int(n_ref),
        "null_plain": plain,
        "p_value_plain": float((plain >= mmd_observed).mean()),
    }

    if observed_j is not None and reference_j is not None:
        obs_j = np.asarray(observed_j).reshape(-1).astype(int)
        ref_j = np.asarray(reference_j).reshape(-1).astype(int)
        pools = {j: np.flatnonzero(ref_j == j) for j in np.unique(obs_j)}
        if all(len(p) for p in pools.values()):

            def stratified(r: np.random.Generator) -> np.ndarray:
                return np.array([pools[j][r.integers(len(pools[j]))] for j in obs_j])

            null = _null_mmd(reference, stratified, comparison_fn, num_null, rng)
            out["null_stratified"] = null
            out["p_value_stratified"] = float((null >= mmd_observed).mean())
    return out


def per_member_scores(
    observed: np.ndarray,
    reference: np.ndarray,
    observed_j: np.ndarray,
    reference_j: np.ndarray,
    ridge: float = 1e-3,
) -> dict[int, dict]:
    """Mahalanobis OOD score of each observed member vs its own stream's reference cloud.

    ``percentile`` is the rank of the observed distance within the reference members' own
    distances (100 = farther than every reference member of that stream).
    """
    observed = np.asarray(observed, dtype=float)
    reference = np.asarray(reference, dtype=float)
    obs_j = np.asarray(observed_j).reshape(-1).astype(int)
    ref_j = np.asarray(reference_j).reshape(-1).astype(int)

    scores: dict[int, dict] = {}
    for i, j in enumerate(obs_j):
        pool = reference[ref_j == j]
        if pool.shape[0] < 2:
            continue
        mean = pool.mean(axis=0)
        cov = np.cov(pool, rowvar=False)
        cov += ridge * (np.trace(cov) / cov.shape[0]) * np.eye(cov.shape[0])
        prec = np.linalg.pinv(cov)

        delta = observed[i] - mean
        d_obs = float(np.sqrt(delta @ prec @ delta))
        d_ref = np.sqrt(np.einsum("ni,ij,nj->n", pool - mean, prec, pool - mean))
        scores[int(j)] = {
            "mahalanobis": d_obs,
            "reference_median": float(np.median(d_ref)),
            "percentile": float((d_ref <= d_obs).mean() * 100.0),
            "n_reference": int(pool.shape[0]),
        }
    return scores


def run_misspecification_test(
    cfg,
    approximator,
    flat: Mapping[str, np.ndarray],
    run_dir: str,
    observed_summaries: np.ndarray | None = None,
) -> dict | None:
    """Full best-effort test for an ``evaluate_real`` run: resolve the reference summaries,
    compare, name the members via the simulator's ``target_streams``, and save artifacts."""
    try:
        reference = str(getattr(cfg.eval, "misspecification_reference", "") or "")
        if not reference:
            log.info(
                "eval.misspecification_reference not set — skipping the summary-space "
                "misspecification test (point it at an `evaluate composition=global` run dir "
                "that saved %s).",
                SUMMARIES,
            )
            return None
        path = reference if reference.endswith(".npz") else os.path.join(reference, SUMMARIES)
        ref = np.load(path)
        ref_summaries, ref_j = ref["summaries"], ref["j"]

        if observed_summaries is None:
            observed_summaries = member_summaries(approximator, flat)
        obs_j = np.asarray(flat["j"]).reshape(-1).astype(int)

        rng = np.random.default_rng(int(getattr(cfg, "seed", 0)))
        num_null = int(getattr(cfg.eval, "misspecification_num_null", 200))
        results = mmd_test(
            observed_summaries, ref_summaries, obs_j, ref_j, num_null=num_null, rng=rng
        )
        results["per_member"] = per_member_scores(observed_summaries, ref_summaries, obs_j, ref_j)
        results["reference_path"] = path

        # Human names for the member ids, when the simulator declares them.
        try:
            from hydrabflow.simulators.registry import get_simulator

            streams = getattr(get_simulator(cfg.simulator), "target_streams", None) or {}
            names = {int(v): str(k) for k, v in streams.items()}
            results["per_member"] = {
                names.get(j, f"member_{j}"): s for j, s in results["per_member"].items()
            }
        except Exception:
            pass

        _save_artifacts(results, run_dir)
        headline = {
            "mmd_observed": round(results["mmd_observed"], 4),
            "p_plain": results["p_value_plain"],
            "p_stratified": results.get("p_value_stratified"),
            "per_member_percentile": {
                k: round(v["percentile"], 1) for k, v in results["per_member"].items()
            },
        }
        log.info("Misspecification test: %s", headline)
        return results
    except Exception as exc:  # diagnostics must never abort an evaluation
        log.warning("misspecification test failed: %s", exc)
        return None


def _save_artifacts(results: dict, run_dir: str) -> None:
    payload = {
        k: (v.tolist() if isinstance(v, np.ndarray) else v)
        for k, v in results.items()
        if not k.startswith("null_")
    }
    for key in ("null_plain", "null_stratified"):
        if key in results:
            null = np.asarray(results[key])
            payload[f"{key}_quantiles"] = {
                q: float(np.quantile(null, float(q))) for q in ("0.05", "0.5", "0.95", "0.99")
            }
    with open(os.path.join(run_dir, MISSPECIFICATION_JSON), "w") as f:
        json.dump(payload, f, indent=2)

    try:
        import matplotlib

        matplotlib.use("Agg")
        from bayesflow.diagnostics.plots import mmd_hypothesis_test

        null = results.get("null_stratified", results.get("null_plain"))
        fig = mmd_hypothesis_test(
            mmd_null=np.asarray(null), mmd_observed=float(results["mmd_observed"])
        )
        fig.savefig(os.path.join(run_dir, MMD_PLOT), bbox_inches="tight")
    except Exception as exc:
        log.warning("mmd plot failed: %s", exc)

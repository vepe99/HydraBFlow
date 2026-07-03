"""Targeted runtime fixes for known BayesFlow bugs (version-checked, applied once).

``CompositionalApproximator._prepare_compositional_conditions`` (bayesflow 2.0.12) reshapes the
*summary outputs* back to the grouped layout using the trailing shape of the *resolved
conditions*. With a summary network **and** direct inference conditions (e.g. the stream index
``j``), those widths differ (resolved = summaries + conditions), so compositional sampling
crashes with a reshape error — the same defect the reference project monkeypatched around.
The patched version below is identical except each tensor is restored with its own trailing
shape. Drop this module once the fix lands upstream.
"""

from __future__ import annotations

from hydrabflow.utils.logging import get_logger

log = get_logger(__name__)


def apply_bayesflow_patches() -> None:
    """Idempotently install the fixes. Called when a compositional workflow is built."""
    _patch_compositional_condition_reshape()


def _patch_compositional_condition_reshape() -> None:
    import keras
    import numpy as np  # noqa: F401  (kept for parity with the original module's imports)
    from bayesflow.approximators.compositional_approximator import CompositionalApproximator

    if getattr(CompositionalApproximator, "_hydrabflow_reshape_patch", False):
        return

    def _prepare_compositional_conditions(
        self, conditions, batch_size=None, summary_outputs=None, **kwargs
    ):
        if summary_outputs is not None:
            num_datasets, num_items = keras.ops.shape(summary_outputs)[:2]
            summary_outputs = keras.ops.reshape(
                summary_outputs,
                (num_datasets * num_items,) + keras.ops.shape(summary_outputs)[2:],
            )
            flattened_conditions = None
        elif conditions is not None:
            original_shapes = {}
            flattened_conditions = {}
            for key, value in conditions.items():
                original_shapes[key] = value.shape
                num_datasets, num_items = value.shape[:2]
                flattened_conditions[key] = value.reshape(
                    (num_datasets * num_items,) + value.shape[2:]
                )
            num_datasets, num_items = original_shapes[next(iter(original_shapes))][:2]
        else:
            raise ValueError(
                "At least one of 'conditions' or 'summary_outputs' must be provided for "
                "compositional sampling."
            )

        if num_items <= 1:
            raise ValueError(
                "At least two conditioning variables are required for compositional sampling, "
                f"got {num_items}. Use 'sample' instead."
            )

        resolved_conditions, adapted, summary_outputs = self._prepare_conditions(
            data=flattened_conditions,
            summary_outputs=summary_outputs,
            batch_size=batch_size,
            **kwargs,
        )

        # FIX vs upstream: restore each tensor with its *own* trailing shape (upstream reused
        # the resolved-conditions shape for the summary outputs).
        resolved_conditions = keras.ops.reshape(
            resolved_conditions,
            (num_datasets, num_items) + tuple(keras.ops.shape(resolved_conditions)[1:]),
        )
        if summary_outputs is not None:
            summary_outputs = keras.ops.reshape(
                summary_outputs,
                (num_datasets, num_items) + tuple(keras.ops.shape(summary_outputs)[1:]),
            )
        return resolved_conditions, adapted, summary_outputs

    CompositionalApproximator._prepare_compositional_conditions = (
        _prepare_compositional_conditions
    )
    CompositionalApproximator._hydrabflow_reshape_patch = True
    log.info("Applied BayesFlow compositional-conditions reshape fix.")

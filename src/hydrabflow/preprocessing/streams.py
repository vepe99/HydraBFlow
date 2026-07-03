"""Stream-specific preprocessing: per-stream normalization and rotation-curve trimming.

Ports the reference project's per-stream ("jonas_streamnorm") normalization into the
preprocessing module, split by what is being normalized:

* :class:`PerStreamParameterStandardize` — the *local parameters*: each stream's inferred
  phase-space parameters are z-scored with that stream's own prior mean/std (deterministic from
  config, invertible for posterior samples). This replaces the inline ``prior_local.yaml``
  renormalization the reference did in the local training/eval scripts.
* :class:`StreamObservationStats` — the *observations*: per-stream mean/std of the particle
  observable (reduced over rows and particles) and per-radial-bin stats of ``log10(vcirc)``,
  fitted once on the train split and saved with the preprocessing state. Its ``transform`` is
  the identity: application happens per batch **after** the physical-unit augmentations, through
  the ``per_stream_standardize`` augmentation, which reads this step's fitted state.
* :class:`MaskVcircRadii` — trim the rotation curve to radii ``r >= r_min`` on the observed
  grid (the reference kept ``R > 5.5 kpc``).

All steps key the stream by an integer index column (``j`` by default), so they work for both
flat ``(n, 1)`` and grouped/compositional ``(n, m, 1)`` layouts.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping

import numpy as np

from hydrabflow.preprocessing.base import Dataset, PreprocessStep
from hydrabflow.preprocessing.registry import register_step
from hydrabflow.simulators.stream_common import OBS_R_KPC, inferred_names


def _stream_index(data: Dataset, stream_key: str, like: np.ndarray) -> np.ndarray:
    """Integer stream ids broadcastable against ``like``.

    ``j``'s leading axes always match ``like``'s (rows, or (datasets, members)); posterior
    samples add trailing axes (num_samples, 1), so ``j`` is padded with trailing singleton
    dimensions until the ranks agree.
    """
    j = np.asarray(data[stream_key]).astype(int)
    while j.ndim < like.ndim:
        j = j[..., None]
    return j


@register_step("per_stream_parameter_standardize")
class PerStreamParameterStandardize(PreprocessStep):
    """z-score each stream's local parameters with that stream's prior mean/std."""

    name = "per_stream_parameter_standardize"

    def __init__(
        self,
        priors: Mapping[str, Mapping[str, Mapping]],
        target_streams: Mapping[str, int],
        keys: Iterable[str] | None = None,
        stream_key: str = "j",
    ) -> None:
        self.stream_key = stream_key
        self.keys = list(keys) if keys is not None else inferred_names(
            next(iter(priors.values()))
        )
        n_streams = max(int(v) for v in target_streams.values()) + 1
        # (n_streams, n_keys) lookups indexed by the j column.
        self.mean = np.zeros((n_streams, len(self.keys)))
        self.std = np.ones((n_streams, len(self.keys)))
        for name, j in target_streams.items():
            for i, key in enumerate(self.keys):
                spec = priors[name][key]
                if spec["type"] != "normal":
                    raise ValueError(
                        f"per_stream_parameter_standardize expects normal priors; "
                        f"'{key}' of stream '{name}' is '{spec['type']}'"
                    )
                self.mean[int(j), i] = float(spec["prior_parameters"][0])
                self.std[int(j), i] = float(spec["prior_parameters"][1])

    def transform(self, data: Dataset) -> Dataset:
        out = dict(data)
        for i, key in enumerate(self.keys):
            if key not in out:
                continue
            x = np.asarray(out[key])
            j = _stream_index(out, self.stream_key, x)
            out[key] = (x - self.mean[j, i]) / self.std[j, i]
        return out

    def inverse_transform(self, data: Dataset) -> Dataset:
        out = dict(data)
        for i, key in enumerate(self.keys):
            if key not in out:
                continue
            x = np.asarray(out[key])
            j = _stream_index(out, self.stream_key, x)
            out[key] = x * self.std[j, i] + self.mean[j, i]
        return out

    def state(self) -> Dict[str, np.ndarray]:
        return {"mean": self.mean, "std": self.std, "keys": np.array(self.keys)}

    def load_state(self, state: Dict[str, np.ndarray]) -> None:
        self.mean = np.asarray(state["mean"])
        self.std = np.asarray(state["std"])


@register_step("stream_observation_stats")
class StreamObservationStats(PreprocessStep):
    """Fit per-stream observation stats + log10(vcirc) per-bin stats on the (clean) train split.

    ``transform`` is the identity — the stats are applied per batch by the
    ``per_stream_standardize`` augmentation, which must run *after* the physical-unit
    augmentations (windows, measurement errors) and after ``log10_vcirc``, but *before* any
    feature concatenations (the stats cover the raw observable features only).
    """

    name = "stream_observation_stats"

    def __init__(
        self,
        observable: str = "sim_data_projected",
        vcirc_key: str = "vcirc_kms",
        stream_key: str = "j",
    ) -> None:
        self.observable = observable
        self.vcirc_key = vcirc_key
        self.stream_key = stream_key
        self.obs_mean: np.ndarray | None = None  # (n_streams, n_features)
        self.obs_std: np.ndarray | None = None
        self.vcirc_mean: np.ndarray | None = None  # (n_bins, 1), stats of log10(vcirc)
        self.vcirc_std: np.ndarray | None = None

    def fit(self, data: Dataset) -> None:
        obs = np.asarray(data[self.observable])
        j = np.asarray(data[self.stream_key]).astype(int)
        # Flatten grouped layouts (n, m, particles, feat) -> rows of (particles, feat).
        obs = obs.reshape(-1, *obs.shape[-2:])
        j = j.reshape(-1)
        n_streams = int(j.max()) + 1
        n_features = obs.shape[-1]
        self.obs_mean = np.zeros((n_streams, n_features))
        self.obs_std = np.ones((n_streams, n_features))
        for stream in np.unique(j):
            block = obs[j == stream]  # (rows, particles, feat)
            self.obs_mean[stream] = block.mean(axis=(0, 1))
            self.obs_std[stream] = np.where(
                block.std(axis=(0, 1)) == 0, 1.0, block.std(axis=(0, 1))
            )

        vcirc = np.asarray(data[self.vcirc_key])
        vcirc = vcirc.reshape(-1, *vcirc.shape[-2:])  # (rows, n_bins, 1)
        logv = np.log10(vcirc)
        self.vcirc_mean = logv.mean(axis=0)
        self.vcirc_std = logv.std(axis=0).clip(min=1e-8)

    def transform(self, data: Dataset) -> Dataset:
        return data

    def state(self) -> Dict[str, np.ndarray]:
        return {
            "obs_mean": self.obs_mean,
            "obs_std": self.obs_std,
            "vcirc_mean": self.vcirc_mean,
            "vcirc_std": self.vcirc_std,
        }

    def load_state(self, state: Dict[str, np.ndarray]) -> None:
        self.obs_mean = np.asarray(state["obs_mean"])
        self.obs_std = np.asarray(state["obs_std"])
        self.vcirc_mean = np.asarray(state["vcirc_mean"])
        self.vcirc_std = np.asarray(state["vcirc_std"])


@register_step("attach_observed_vcirc")
class AttachObservedVcirc(PreprocessStep):
    """Attach the *observed* Milky Way rotation curve to a real dataset that lacks one.

    Simulated datasets carry their model ``vcirc_kms``; real observations use the measured
    curve (Eilers-style values on the same radii grid). One copy per row, on the full grid —
    trim afterwards with ``mask_vcirc_radii`` exactly like the training data.
    """

    name = "attach_observed_vcirc"

    def __init__(self, vcirc_key: str = "vcirc_kms", values: Iterable[float] | None = None) -> None:
        from hydrabflow.simulators.stream_common import OBS_VC_KMS

        self.vcirc_key = vcirc_key
        self.values = np.asarray(values if values is not None else OBS_VC_KMS, dtype=float)

    def transform(self, data: Dataset) -> Dataset:
        if self.vcirc_key in data:
            return data
        out = dict(data)
        n = len(next(iter(data.values())))
        out[self.vcirc_key] = np.tile(self.values[None, :, None], (n, 1, 1))
        return out


@register_step("mask_vcirc_radii")
class MaskVcircRadii(PreprocessStep):
    """Keep only rotation-curve bins with ``r >= r_min`` on the observed radii grid."""

    name = "mask_vcirc_radii"

    def __init__(
        self,
        r_min: float = 5.5,
        vcirc_key: str = "vcirc_kms",
        radii: Iterable[float] | None = None,
    ) -> None:
        self.vcirc_key = vcirc_key
        self.radii = np.asarray(radii if radii is not None else OBS_R_KPC, dtype=float)
        self.mask = self.radii >= float(r_min)

    def transform(self, data: Dataset) -> Dataset:
        if self.vcirc_key not in data:
            return data
        out = dict(data)
        vcirc = np.asarray(out[self.vcirc_key])
        if vcirc.shape[-2] != self.mask.size:
            raise ValueError(
                f"{self.vcirc_key} has {vcirc.shape[-2]} radial bins but the configured grid "
                f"has {self.mask.size}"
            )
        out[self.vcirc_key] = vcirc[..., self.mask, :]
        return out

"""Per-batch augmentations turning clean simulated streams into Gaia-like observations.

NumPy port of the reference project's ``AugmentationsClass``
(``utils/utils_train_jax_new_rotationcurve_fixedvlosmask.py``), reorganized as registered,
config-driven augmentation factories. The physical chain (order matters, see
``conf/augmentation/stream_*.yaml``):

1.  ``convert_distance_to_parallax``  — distance [kpc] -> parallax [mas];
2.  ``observational_window``          — per-stream RA/Dec box -> boolean ``attention_mask``;
3.  ``observed_n_stars``              — subsample the mask to the observed member count;
4.  ``compact_to_attended``           — attended particles first, slice to ``max_particles``;
5.  ``sample_magnitudes``             — G magnitudes from a per-stream KDE of observed members;
6.  ``sample_obs_error``              — Gaia DR3 uncertainties interpolated at those magnitudes;
7.  ``apply_obs_error``               — perturb the observables with those uncertainties;
8.  ``mask_vlos``                     — only a subset of members keeps line-of-sight velocity;
9.  ``add_noise_to_vcirc``            — observed Vc(R) error bars on the model rotation curve;
10. ``log10_vcirc``                   — rotation curve to log10;
11. ``per_stream_standardize``        — per-stream z-scoring of observations (fitted stats from
                                        the ``stream_observation_stats`` preprocessing step);
12. ``concatenate_sigma_errors`` / ``concatenate_magnitudes`` / ``concatenate_vlos_mask`` /
    ``concatenate_stream_index``     — append per-particle features the network should see.

``override_vlos_error_with_real`` replaces sampled v_los uncertainties with the instrument's
real ones where available (real-data evaluation only).

Shared resources (Gaia member magnitudes, DR3 error tables) are loaded once per parameter set
from the files shipped in ``data/`` and cached at module level.
"""

from __future__ import annotations

import json
import os
from typing import Dict

import numpy as np

from hydrabflow.augmentation.registry import register_augmentation
from hydrabflow.simulators.stream_common import OBS_R_KPC, OBS_SIGMA_VC

_SIM_KEY = "sim_data_projected"
_RESOURCE_CACHE: Dict[str, "StreamResources"] = {}


def _sim_key(params) -> str:
    return str(params.get("observable", _SIM_KEY))


class StreamResources:
    """Gaia tables shared by several augmentations: member magnitudes (KDE per stream) and
    DR3 measurement uncertainties per magnitude bin."""

    def __init__(self, params: dict) -> None:
        from astropy import units as u
        from astropy.io import ascii as astro_ascii

        data_dir = params.get("resources_dir", "data")
        self.target_streams = {str(k): int(v) for k, v in params["target_streams"].items()}
        self.stream_by_j = {j: name for name, j in self.target_streams.items()}
        self.n_streams = max(self.stream_by_j) + 1

        # --- Observed member magnitudes per stream (Ibata+23 member table + id mapping) ------ #
        import pandas as pd

        tbl_members = astro_ascii.read(
            os.path.join(data_dir, params.get("member_table", "apjad382dt1_mrt.txt")),
            format="cds",
        )
        tbl_ids = pd.read_csv(
            os.path.join(data_dir, params.get("stream_id_table", "gaia_stream_id.csv")),
            sep="\t",
        )
        gaia_id = {str(k): str(v) for k, v in params["gaia_id"].items()}

        self.magnitudes: Dict[int, np.ndarray] = {}
        for name, j in self.target_streams.items():
            source_id = tbl_ids.loc[tbl_ids["Name"] == gaia_id[name], "s_ID"].values[0]
            subset = tbl_members[tbl_members["Stream"] == source_id]
            self.magnitudes[j] = np.asarray(subset["Gmag"], dtype=float)

        # --- Gaia DR3 uncertainty vs magnitude table ---------------------------------------- #
        tbl_err = astro_ascii.read(
            os.path.join(data_dir, params.get("error_table", "gaia_DR3_erorr_6D.txt")),
            format="tab",
        )
        tbl_err.remove_column("Unit")
        mag_bins = []
        for colname in tbl_err.colnames[1:]:
            clean = colname.replace("−", "-").replace("–", "-")
            if "-" in clean:
                lo, hi = clean.split("-")
                mag_bins.append((float(lo) + float(hi)) / 2.0)
            else:
                mag_bins.append(float(clean))
        self.error_mag_bins = np.asarray(mag_bins)

        error_values = {
            row["Quantity"].strip(): np.array(
                [row[col] for col in tbl_err.colnames[1:]], dtype=float
            )
            for row in tbl_err
        }
        self.error_keys = [
            str(k)
            for k in params.get(
                "error_keys", ["ra", "dec", "parallax", "mu_ra", "mu_dec", "v_los"]
            )
        ]
        self.error_values = np.stack([error_values[k] for k in self.error_keys], axis=0)
        # ra/dec uncertainties are tabulated in mas; the observables are in degrees.
        self.error_values[:2] *= u.mas.to(u.deg)

        # --- Per-stream config lookups ------------------------------------------------------ #
        def lookup(mapping, inner=None):
            out = np.zeros(self.n_streams)
            for name, j in self.target_streams.items():
                out[j] = float(mapping[name] if inner is None else mapping[name][inner])
            return out

        window = params["observational_window"]
        self.ra_min = lookup(window, "ra_min")
        self.ra_max = lookup(window, "ra_max")
        self.dec_min = lookup(window, "dec_min")
        self.dec_max = lookup(window, "dec_max")
        self.observed_n_stars = lookup(params["observed_n_stars"]).astype(int)
        self.min_star_with_vlos = lookup(params["min_star_with_vlos"]).astype(int)

        # --- Rotation-curve error bars on the (possibly trimmed) radii grid ----------------- #
        radii = np.asarray(params.get("obs_r_kpc", OBS_R_KPC), dtype=float)
        sigma = np.asarray(params.get("obs_sigma_vc", OBS_SIGMA_VC), dtype=float)
        keep = radii >= float(params.get("vcirc_r_min", 5.5))
        self.vcirc_sigma = sigma[keep][:, None]  # (n_bins, 1)


def _resources(params: dict) -> StreamResources:
    key = json.dumps({k: v for k, v in params.items()}, sort_keys=True, default=str)
    if key not in _RESOURCE_CACHE:
        _RESOURCE_CACHE[key] = StreamResources(params)
    return _RESOURCE_CACHE[key]


def _stream_ids(batch, key="j") -> np.ndarray:
    return np.asarray(batch[key]).astype(int).reshape(-1)


# ------------------------------------------------------------------------------------------- #
# Coordinate transforms
# ------------------------------------------------------------------------------------------- #


@register_augmentation("convert_distance_to_parallax")
def _convert_distance_to_parallax(params, rng):
    key = _sim_key(params)

    def aug(batch):
        sim = np.asarray(batch[key]).copy()
        sim[:, :, 2] = 1.0 / sim[:, :, 2]
        batch[key] = sim
        return batch

    return aug


@register_augmentation("remove_los_velocity")
def _remove_los_velocity(params, rng):
    key = _sim_key(params)

    def aug(batch):
        batch[key] = np.asarray(batch[key])[:, :, :5]
        return batch

    return aug


# ------------------------------------------------------------------------------------------- #
# Observational selection (window -> subsample -> compact)
# ------------------------------------------------------------------------------------------- #


@register_augmentation("observational_window")
def _observational_window(params, rng):
    res = _resources(params)
    key = _sim_key(params)

    def aug(batch):
        sim = np.asarray(batch[key])
        j = _stream_ids(batch)
        ra, dec = sim[:, :, 0], sim[:, :, 1]
        mask = (
            (ra >= res.ra_min[j][:, None])
            & (ra <= res.ra_max[j][:, None])
            & (dec >= res.dec_min[j][:, None])
            & (dec <= res.dec_max[j][:, None])
        )
        batch["attention_mask"] = mask[:, None, :]
        return batch

    return aug


def _keep_first_k_random(mask: np.ndarray, k_per_row: np.ndarray, rng) -> np.ndarray:
    """Randomly keep at most ``k`` True entries per row of a boolean mask (vectorized)."""
    n, p = mask.shape
    scores = rng.uniform(size=(n, p))
    scores = np.where(mask, scores, 2.0)  # push unattended past any threshold
    order = np.sort(scores, axis=1)
    k = np.clip(k_per_row, 1, p)
    threshold = order[np.arange(n), k - 1]
    return mask & (scores <= threshold[:, None])


@register_augmentation("observed_n_stars")
def _observed_n_stars(params, rng):
    res = _resources(params)

    def aug(batch):
        mask = np.asarray(batch["attention_mask"])[:, 0, :]
        j = _stream_ids(batch)
        batch["attention_mask"] = _keep_first_k_random(mask, res.observed_n_stars[j], rng)[
            :, None, :
        ]
        return batch

    return aug


@register_augmentation("compact_to_attended")
def _compact_to_attended(params, rng):
    key = _sim_key(params)
    max_particles = int(params.get("max_particles", 300))

    def aug(batch):
        sim = np.asarray(batch[key])
        mask = np.asarray(batch["attention_mask"])[:, 0, :]
        order = np.argsort(~mask, axis=1, kind="stable")  # attended first, original order kept
        sim = np.take_along_axis(sim, order[..., None], axis=1)[:, :max_particles, :]
        mask = np.take_along_axis(mask, order, axis=1)[:, :max_particles]
        batch[key] = sim
        batch["attention_mask"] = mask[:, None, :]
        return batch

    return aug


# ------------------------------------------------------------------------------------------- #
# Photometry and measurement errors
# ------------------------------------------------------------------------------------------- #


@register_augmentation("sample_magnitudes")
def _sample_magnitudes(params, rng):
    from scipy.stats import gaussian_kde

    res = _resources(params)
    key = _sim_key(params)
    kdes = {j: gaussian_kde(m) for j, m in res.magnitudes.items()}
    clip = {j: (m.min(), m.max()) for j, m in res.magnitudes.items()}

    def aug(batch):
        j = _stream_ids(batch)
        n, p = np.asarray(batch[key]).shape[:2]
        mags = np.empty((n, p))
        for stream in np.unique(j):
            rows = np.flatnonzero(j == stream)
            seed = int(rng.integers(0, 2**31 - 1))
            samples = kdes[stream].resample(rows.size * p, seed=seed)[0]
            lo, hi = clip[stream]
            mags[rows] = np.clip(samples.reshape(rows.size, p), lo, hi)
        batch["magnitudes"] = mags
        return batch

    return aug


@register_augmentation("sample_obs_error")
def _sample_obs_error(params, rng):
    res = _resources(params)

    def aug(batch):
        mags = np.asarray(batch["magnitudes"])
        sigmas = np.stack(
            [np.interp(mags, res.error_mag_bins, vals) for vals in res.error_values],
            axis=-1,
        )  # (n, particles, n_error_keys)
        noise = rng.normal(size=sigmas.shape)
        batch["obs_errors"] = sigmas * noise
        batch["sigma_errors"] = sigmas
        return batch

    return aug


@register_augmentation("apply_obs_error")
def _apply_obs_error(params, rng):
    res = _resources(params)
    key = _sim_key(params)

    def aug(batch):
        sim = np.asarray(batch[key]).copy()
        q = len(res.error_keys)
        sim[:, :, :q] += np.asarray(batch["obs_errors"])
        batch[key] = sim
        return batch

    return aug


@register_augmentation("mask_vlos")
def _mask_vlos(params, rng):
    res = _resources(params)
    key = _sim_key(params)

    def aug(batch):
        sim = np.asarray(batch[key]).copy()
        sigma = np.asarray(batch["sigma_errors"]).copy()
        mask = np.asarray(batch["attention_mask"])[:, 0, :]
        j = _stream_ids(batch)

        vlos_mask = _keep_first_k_random(mask, res.min_star_with_vlos[j], rng)

        vlos = sim[:, :, -1]
        kept = vlos_mask.sum(axis=1).clip(min=1)
        mean = np.where(vlos_mask, vlos, 0.0).sum(axis=1) / kept
        std = np.sqrt(
            np.where(vlos_mask, (vlos - mean[:, None]) ** 2, 0.0).sum(axis=1) / kept
        )

        sim[:, :, -1] = np.where(vlos_mask, vlos, mean[:, None])
        sigma[:, :, -1] = np.where(vlos_mask, sigma[:, :, -1], std[:, None])

        batch[key] = sim
        batch["sigma_errors"] = sigma
        batch["vlos_mask"] = vlos_mask[:, None, :]
        return batch

    return aug


@register_augmentation("override_vlos_error_with_real")
def _override_vlos_error_with_real(params, rng):
    """Real data only: where a member has a measured v_los, use the instrument's uncertainty."""

    def aug(batch):
        sigma = np.asarray(batch["sigma_errors"]).copy()
        vlos_error = np.asarray(batch["vlos_error"])[..., 0]
        vlos_mask = np.asarray(batch["vlos_mask"])[..., 0]
        sigma[:, :, -1] = np.where(vlos_mask, vlos_error, sigma[:, :, -1])
        batch["sigma_errors"] = sigma
        return batch

    return aug


# ------------------------------------------------------------------------------------------- #
# Rotation curve
# ------------------------------------------------------------------------------------------- #


@register_augmentation("add_noise_to_vcirc")
def _add_noise_to_vcirc(params, rng):
    res = _resources(params)
    key = str(params.get("vcirc_key", "vcirc_kms"))

    def aug(batch):
        vcirc = np.asarray(batch[key])
        batch[key] = vcirc + rng.normal(size=vcirc.shape) * res.vcirc_sigma
        return batch

    return aug


@register_augmentation("log10_vcirc")
def _log10_vcirc(params, rng):
    key = str(params.get("vcirc_key", "vcirc_kms"))

    def aug(batch):
        batch[key] = np.log10(np.asarray(batch[key]))
        return batch

    return aug


# ------------------------------------------------------------------------------------------- #
# Per-stream standardization (applies the stats fitted by preprocessing)
# ------------------------------------------------------------------------------------------- #


@register_augmentation("per_stream_standardize")
def _per_stream_standardize(params, rng, context):
    """z-score the observable per stream and the (log10) rotation curve per radial bin, using
    the state the ``stream_observation_stats`` preprocessing step fitted on the train split.
    Must run after the physical-unit augmentations and ``log10_vcirc``, before concatenations."""
    key = _sim_key(params)
    vcirc_key = str(params.get("vcirc_key", "vcirc_kms"))

    pipeline = (context or {}).get("pipeline")
    stats = pipeline.get_step("stream_observation_stats") if pipeline is not None else None
    if stats is None:
        raise ValueError(
            "per_stream_standardize needs the 'stream_observation_stats' preprocessing step "
            "(add it to preprocessing.steps so its stats are fitted and saved)."
        )

    def aug(batch):
        if stats.obs_mean is None:
            raise RuntimeError("stream_observation_stats has no fitted state")
        sim = np.asarray(batch[key])
        j = _stream_ids(batch)
        n_feat = stats.obs_mean.shape[1]
        sim = sim.copy()
        sim[..., :n_feat] = (sim[..., :n_feat] - stats.obs_mean[j][:, None, :]) / stats.obs_std[
            j
        ][:, None, :]
        batch[key] = sim
        if vcirc_key in batch:
            vc = np.asarray(batch[vcirc_key])
            batch[vcirc_key] = (vc - stats.vcirc_mean) / stats.vcirc_std
        return batch

    return aug


# ------------------------------------------------------------------------------------------- #
# Feature concatenations (must come last)
# ------------------------------------------------------------------------------------------- #


@register_augmentation("concatenate_sigma_errors")
def _concatenate_sigma_errors(params, rng):
    key = _sim_key(params)

    def aug(batch):
        batch[key] = np.concatenate(
            [np.asarray(batch[key]), np.asarray(batch["sigma_errors"])], axis=-1
        )
        return batch

    return aug


@register_augmentation("concatenate_magnitudes")
def _concatenate_magnitudes(params, rng):
    key = _sim_key(params)

    def aug(batch):
        batch[key] = np.concatenate(
            [np.asarray(batch[key]), np.asarray(batch["magnitudes"])[..., None]], axis=-1
        )
        return batch

    return aug


@register_augmentation("concatenate_vlos_mask")
def _concatenate_vlos_mask(params, rng):
    key = _sim_key(params)

    def aug(batch):
        sim = np.asarray(batch[key])
        vlos_mask = np.asarray(batch["vlos_mask"]).transpose(0, 2, 1).astype(sim.dtype)
        batch[key] = np.concatenate([sim, vlos_mask], axis=-1)
        return batch

    return aug


@register_augmentation("concatenate_stream_index")
def _concatenate_stream_index(params, rng):
    key = _sim_key(params)

    def aug(batch):
        sim = np.asarray(batch[key])
        j = np.asarray(batch["j"]).reshape(-1).astype(sim.dtype)
        j_feat = np.broadcast_to(j[:, None, None], (*sim.shape[:2], 1))
        batch[key] = np.concatenate([sim, j_feat], axis=-1)
        return batch

    return aug

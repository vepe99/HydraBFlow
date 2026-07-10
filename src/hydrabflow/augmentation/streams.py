"""Per-batch augmentations turning clean simulated streams into Gaia-like observations.

JAX port of the reference project's ``AugmentationsClass``
(``utils/utils_train_jax_new_rotationcurve_fixedvlosmask.py``), reorganized as registered,
config-driven augmentation factories. Each step's numeric core is compiled once (at closure-build
time, before the first batch) with ``jax.jit`` and reused every call, fusing its ops into a single
XLA program instead of dispatching each ``jnp`` op separately — this is what the reference project
itself did (``@partial(jit, static_argnums=(0,))`` on its class methods) and measurably raises GPU
utilization here too (dispatch overhead was leaving the GPU idle between ops). The physical chain
(order matters, see ``conf/augmentation/stream_*.yaml``):

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
from the files shipped in ``data/`` and cached at module level, alongside their JAX-side lookup
arrays and per-stream KDE objects (built once, reused every batch).

**RNG.** Each augmentation gets its own independent ``numpy.random.Generator`` child from the
registry (see ``augmentation/registry.py``); it is used once, at closure-build time, to seed a
``jax.random.PRNGKey``, which is then split on every call (mirroring the reference project's
``_split_key`` idiom) via a one-element mutable cell so each batch draws fresh, reproducible
randomness. The subkey is passed as a normal (traced) argument to each jitted function.

**Compiled-function caching.** ``jax.jit`` recompiles when input shapes change. Every step in this
chain sees a fixed shape per training run except ``sample_magnitudes``, whose particle count
changes once (1000 -> ``max_particles`` after ``compact_to_attended``); its jitted function is
therefore cached per distinct particle count (mirrors the reference's own per-``n_particles``
branch cache) rather than rebuilt every call.
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


def _vlos_impute(params) -> str:
    """Fill mode for stars without a measured v_los: ``mean`` (per-stream mean/std of the
    measured members — the historical behavior, and the fill baked into the real Gaia npz) or
    ``zero`` (constant 0.0 for value and sigma; the indicator channel carries the missingness,
    per Wang et al. 2024's most robust encoding)."""
    mode = str(params.get("vlos_impute", "mean"))
    if mode not in ("mean", "zero"):
        raise ValueError(f"params.vlos_impute must be 'mean' or 'zero', got {mode!r}")
    return mode


def _jax():
    import jax
    import jax.numpy as jnp

    return jax, jnp


def _key_cell(rng) -> list:
    """A fresh JAX PRNGKey, seeded from this augmentation's private NumPy generator, threaded
    through calls via a one-element mutable cell (`cell[0]` is split-and-replaced each call)."""
    jax, _ = _jax()
    seed = int(rng.integers(0, 2**31 - 1))
    return [jax.random.PRNGKey(seed)]


def _next_key(cell: list):
    jax, _ = _jax()
    cell[0], subkey = jax.random.split(cell[0])
    return subkey


class StreamResources:
    """Gaia tables shared by several augmentations: member magnitudes (KDE per stream) and
    DR3 measurement uncertainties per magnitude bin. Loaded once (NumPy/astropy/pandas); the
    JAX-side lookup arrays and per-stream KDE objects used every batch are built lazily and
    cached on first use (``jax_*`` / ``kde_streams`` attributes)."""

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

        self._jax_cache: dict = {}
        self._kde_cache: dict = {}
        self._kde_branch_cache: dict = {}

    # ---- JAX-side lookups, built once and cached ---------------------------------------- #

    def jax_lookups(self):
        if not self._jax_cache:
            _, jnp = _jax()
            self._jax_cache = {
                "ra_min": jnp.asarray(self.ra_min),
                "ra_max": jnp.asarray(self.ra_max),
                "dec_min": jnp.asarray(self.dec_min),
                "dec_max": jnp.asarray(self.dec_max),
                "observed_n_stars": jnp.asarray(self.observed_n_stars),
                "min_star_with_vlos": jnp.asarray(self.min_star_with_vlos),
                "error_mag_bins": jnp.asarray(self.error_mag_bins),
                "error_values": jnp.asarray(self.error_values),
                "vcirc_sigma": jnp.asarray(self.vcirc_sigma),
            }
        return self._jax_cache

    def kde_streams(self):
        """Per-stream ``jax.scipy.stats.gaussian_kde`` (built once, reused every batch)."""
        if not self._kde_cache:
            _, jnp = _jax()
            from jax.scipy.stats import gaussian_kde

            for j, mags in self.magnitudes.items():
                self._kde_cache[j] = gaussian_kde(jnp.asarray(mags)[None, :])
        return self._kde_cache

    def kde_branches(self, n_particles: int):
        """``jax.lax.switch`` branch functions (one per stream) with ``n_particles`` baked in
        as a static Python int, cached per distinct ``n_particles`` seen so far."""
        if n_particles not in self._kde_branch_cache:
            kdes = self.kde_streams()
            _, jnp = _jax()
            branches = []
            for j in range(self.n_streams):
                kde = kdes[j]
                lo, hi = float(self.magnitudes[j].min()), float(self.magnitudes[j].max())

                def branch_fn(key, kde=kde, lo=lo, hi=hi, n=n_particles):
                    samples = kde.resample(key, shape=(n,))[0]
                    return jnp.clip(samples, lo, hi)

                branches.append(branch_fn)
            self._kde_branch_cache[n_particles] = branches
        return self._kde_branch_cache[n_particles]


def _resources(params: dict) -> StreamResources:
    key = json.dumps({k: v for k, v in params.items()}, sort_keys=True, default=str)
    if key not in _RESOURCE_CACHE:
        _RESOURCE_CACHE[key] = StreamResources(params)
    return _RESOURCE_CACHE[key]


def _stream_ids_jax(batch, key="j"):
    _, jnp = _jax()
    return jnp.asarray(batch[key]).reshape(-1).astype(jnp.int32)


def _keep_first_k_random_jax(mask, k_per_row, key):
    """JAX version of the sort-and-threshold trick: randomly keep at most ``k`` True entries per
    row of a boolean mask. ``k_per_row`` may vary per row (indexed by stream)."""
    jax, jnp = _jax()
    n, p = mask.shape
    scores = jax.random.uniform(key, shape=(n, p))
    scores = jnp.where(mask, scores, 2.0)  # push unattended past any threshold
    sorted_scores = jnp.sort(scores, axis=1)
    k = jnp.clip(k_per_row, 1, p)
    threshold = sorted_scores[jnp.arange(n), k - 1]
    return mask & (scores <= threshold[:, None])


# ------------------------------------------------------------------------------------------- #
# Coordinate transforms
# ------------------------------------------------------------------------------------------- #


@register_augmentation("convert_distance_to_parallax")
def _convert_distance_to_parallax(params, rng):
    key = _sim_key(params)
    jax, jnp = _jax()

    @jax.jit
    def _run(sim):
        return sim.at[:, :, 2].set(1.0 / sim[:, :, 2])

    def aug(batch):
        batch[key] = _run(jnp.asarray(batch[key]))
        return batch

    return aug


@register_augmentation("remove_los_velocity")
def _remove_los_velocity(params, rng):
    key = _sim_key(params)
    jax, jnp = _jax()

    @jax.jit
    def _run(sim):
        return sim[:, :, :5]

    def aug(batch):
        batch[key] = _run(jnp.asarray(batch[key]))
        return batch

    return aug


# ------------------------------------------------------------------------------------------- #
# Observational selection (window -> subsample -> compact)
# ------------------------------------------------------------------------------------------- #


@register_augmentation("observational_window")
def _observational_window(params, rng):
    res = _resources(params)
    key = _sim_key(params)
    jax, jnp = _jax()
    lookups = res.jax_lookups()
    ra_min, ra_max = lookups["ra_min"], lookups["ra_max"]
    dec_min, dec_max = lookups["dec_min"], lookups["dec_max"]

    @jax.jit
    def _run(sim, j):
        ra, dec = sim[:, :, 0], sim[:, :, 1]
        return (
            (ra >= ra_min[j][:, None])
            & (ra <= ra_max[j][:, None])
            & (dec >= dec_min[j][:, None])
            & (dec <= dec_max[j][:, None])
        )[:, None, :]

    def aug(batch):
        batch["attention_mask"] = _run(jnp.asarray(batch[key]), _stream_ids_jax(batch))
        return batch

    return aug


@register_augmentation("observed_n_stars")
def _observed_n_stars(params, rng):
    res = _resources(params)
    cell = _key_cell(rng)
    jax, jnp = _jax()
    observed_n_stars = res.jax_lookups()["observed_n_stars"]

    @jax.jit
    def _run(mask, j, subkey):
        return _keep_first_k_random_jax(mask, observed_n_stars[j], subkey)[:, None, :]

    def aug(batch):
        mask = jnp.asarray(batch["attention_mask"])[:, 0, :]
        batch["attention_mask"] = _run(mask, _stream_ids_jax(batch), _next_key(cell))
        return batch

    return aug


@register_augmentation("compact_to_attended")
def _compact_to_attended(params, rng):
    key = _sim_key(params)
    max_particles = int(params.get("max_particles", 300))
    jax, jnp = _jax()

    @jax.jit
    def _run(sim, mask):
        order = jnp.argsort(~mask, axis=1)  # attended (False) first
        sim = jnp.take_along_axis(sim, order[..., None], axis=1)[:, :max_particles, :]
        mask = jnp.take_along_axis(mask, order, axis=1)[:, :max_particles]
        return sim, mask[:, None, :]

    def aug(batch):
        mask = jnp.asarray(batch["attention_mask"])[:, 0, :]
        sim, mask = _run(jnp.asarray(batch[key]), mask)
        batch[key] = sim
        batch["attention_mask"] = mask
        return batch

    return aug


# ------------------------------------------------------------------------------------------- #
# Photometry and measurement errors
# ------------------------------------------------------------------------------------------- #


@register_augmentation("sample_magnitudes")
def _sample_magnitudes(params, rng):
    res = _resources(params)
    key = _sim_key(params)
    cell = _key_cell(rng)
    jit_cache: dict = {}

    def _compiled(n_particles: int):
        compiled = jit_cache.get(n_particles)
        if compiled is None:
            jax, _ = _jax()
            branches = res.kde_branches(n_particles)

            @jax.jit
            def _run(j, keys):
                def sample_one(j_idx, k):
                    return jax.lax.switch(j_idx, branches, k)

                return jax.vmap(sample_one)(j, keys)

            compiled = _run
            jit_cache[n_particles] = compiled
        return compiled

    def aug(batch):
        jax, jnp = _jax()
        sim = jnp.asarray(batch[key])
        n_particles = int(sim.shape[1])
        j = _stream_ids_jax(batch)
        keys = jax.random.split(_next_key(cell), j.shape[0])
        batch["magnitudes"] = _compiled(n_particles)(j, keys)
        return batch

    return aug


@register_augmentation("sample_obs_error")
def _sample_obs_error(params, rng):
    res = _resources(params)
    cell = _key_cell(rng)
    jax, jnp = _jax()
    lookups = res.jax_lookups()
    mag_bins, values = lookups["error_mag_bins"], lookups["error_values"]

    @jax.jit
    def _run(mags, subkey):
        sigmas = jax.vmap(lambda vals: jnp.interp(mags, mag_bins, vals))(values)
        sigmas = jnp.transpose(sigmas, (1, 2, 0))  # (n, particles, n_error_keys)
        noise = jax.random.normal(subkey, shape=sigmas.shape)
        return sigmas * noise, sigmas

    def aug(batch):
        obs_errors, sigma_errors = _run(jnp.asarray(batch["magnitudes"]), _next_key(cell))
        batch["obs_errors"] = obs_errors
        batch["sigma_errors"] = sigma_errors
        return batch

    return aug


@register_augmentation("apply_obs_error")
def _apply_obs_error(params, rng):
    res = _resources(params)
    key = _sim_key(params)
    jax, jnp = _jax()
    q = len(res.error_keys)

    @jax.jit
    def _run(sim, errors):
        return sim.at[:, :, :q].add(errors)

    def aug(batch):
        batch[key] = _run(jnp.asarray(batch[key]), jnp.asarray(batch["obs_errors"]))
        return batch

    return aug


def _measured_vlos_stats(jnp, vlos, vlos_mask):
    """Per-row mean/std of v_los over the measured (mask=1) members."""
    kept = jnp.clip(vlos_mask.sum(axis=1), min=1)
    mean = jnp.where(vlos_mask, vlos, 0.0).sum(axis=1) / kept
    std = jnp.sqrt(jnp.where(vlos_mask, (vlos - mean[:, None]) ** 2, 0.0).sum(axis=1) / kept)
    return mean, std


@register_augmentation("mask_vlos")
def _mask_vlos(params, rng):
    res = _resources(params)
    key = _sim_key(params)
    cell = _key_cell(rng)
    jax, jnp = _jax()
    min_star_with_vlos = res.jax_lookups()["min_star_with_vlos"]
    impute = _vlos_impute(params)

    @jax.jit
    def _run(sim, sigma, mask, j, subkey):
        vlos_mask = _keep_first_k_random_jax(mask, min_star_with_vlos[j], subkey)

        vlos = sim[:, :, -1]
        if impute == "zero":
            fill_value = jnp.zeros(vlos.shape[0], dtype=vlos.dtype)
            fill_sigma = fill_value
        else:
            fill_value, fill_sigma = _measured_vlos_stats(jnp, vlos, vlos_mask)

        sim = sim.at[:, :, -1].set(jnp.where(vlos_mask, vlos, fill_value[:, None]))
        sigma = sigma.at[:, :, -1].set(
            jnp.where(vlos_mask, sigma[:, :, -1], fill_sigma[:, None])
        )
        return sim, sigma, vlos_mask[:, None, :]

    def aug(batch):
        sim = jnp.asarray(batch[key])
        sigma = jnp.asarray(batch["sigma_errors"])
        mask = jnp.asarray(batch["attention_mask"])[:, 0, :]
        j = _stream_ids_jax(batch)

        sim, sigma, vlos_mask = _run(sim, sigma, mask, j, _next_key(cell))
        batch[key] = sim
        batch["sigma_errors"] = sigma
        batch["vlos_mask"] = vlos_mask
        return batch

    return aug


@register_augmentation("override_vlos_error_with_real")
def _override_vlos_error_with_real(params, rng):
    """Real data only: where a member has a measured v_los, use the instrument's uncertainty."""
    jax, jnp = _jax()

    @jax.jit
    def _run(sigma, vlos_error, vlos_mask):
        return sigma.at[:, :, -1].set(jnp.where(vlos_mask, vlos_error, sigma[:, :, -1]))

    def aug(batch):
        sigma = jnp.asarray(batch["sigma_errors"])
        vlos_error = jnp.asarray(batch["vlos_error"])[..., 0]
        vlos_mask = jnp.asarray(batch["vlos_mask"])[..., 0]
        batch["sigma_errors"] = _run(sigma, vlos_error, vlos_mask)
        return batch

    return aug


@register_augmentation("impute_vlos")
def _impute_vlos(params, rng):
    """Re-apply the missing-v_los fill from the batch's existing ``vlos_mask`` (real data
    carries its own mask, and the shipped npz is pre-filled with the measured-star mean).
    ``vlos_impute: mean`` recomputes that mean — a value-preserving no-op on the shipped real
    data, sigma untouched — while ``zero`` writes 0.0 into value and sigma so real inputs match
    a model trained with ``mask_vlos`` in zero mode."""
    key = _sim_key(params)
    jax, jnp = _jax()
    impute = _vlos_impute(params)

    @jax.jit
    def _run(sim, sigma, vlos_mask):
        vlos = sim[:, :, -1]
        if impute == "zero":
            fill = jnp.zeros(vlos.shape[0], dtype=vlos.dtype)
            sigma = sigma.at[:, :, -1].set(
                jnp.where(vlos_mask, sigma[:, :, -1], fill[:, None])
            )
        else:
            fill, _ = _measured_vlos_stats(jnp, vlos, vlos_mask)
        sim = sim.at[:, :, -1].set(jnp.where(vlos_mask, vlos, fill[:, None]))
        return sim, sigma

    def aug(batch):
        sim = jnp.asarray(batch[key])
        sigma = jnp.asarray(batch["sigma_errors"])
        vlos_mask = jnp.asarray(batch["vlos_mask"])[:, 0, :].astype(bool)
        sim, sigma = _run(sim, sigma, vlos_mask)
        batch[key] = sim
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
    cell = _key_cell(rng)
    jax, jnp = _jax()
    vcirc_sigma = res.jax_lookups()["vcirc_sigma"]

    @jax.jit
    def _run(vcirc, subkey):
        noise = jax.random.normal(subkey, shape=vcirc.shape) * vcirc_sigma
        return vcirc + noise

    def aug(batch):
        batch[key] = _run(jnp.asarray(batch[key]), _next_key(cell))
        return batch

    return aug


@register_augmentation("log10_vcirc")
def _log10_vcirc(params, rng):
    key = str(params.get("vcirc_key", "vcirc_kms"))
    jax, jnp = _jax()

    @jax.jit
    def _run(vcirc):
        return jnp.log10(vcirc)

    def aug(batch):
        batch[key] = _run(jnp.asarray(batch[key]))
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
    jax, jnp = _jax()

    pipeline = (context or {}).get("pipeline")
    stats = pipeline.get_step("stream_observation_stats") if pipeline is not None else None
    if stats is None:
        raise ValueError(
            "per_stream_standardize needs the 'stream_observation_stats' preprocessing step "
            "(add it to preprocessing.steps so its stats are fitted and saved)."
        )
    if stats.obs_mean is None:
        raise RuntimeError("stream_observation_stats has no fitted state")

    # Preprocessing has already been fit_transform'd by the time augmentations are built (see
    # pipeline/train.py), so these stats are final for the whole run — safe to bake in as jit
    # constants rather than re-read every batch.
    obs_mean, obs_std = jnp.asarray(stats.obs_mean), jnp.asarray(stats.obs_std)
    vcirc_mean, vcirc_std = jnp.asarray(stats.vcirc_mean), jnp.asarray(stats.vcirc_std)
    n_feat = obs_mean.shape[1]

    @jax.jit
    def _run_sim(sim, j):
        return sim.at[..., :n_feat].set(
            (sim[..., :n_feat] - obs_mean[j][:, None, :]) / obs_std[j][:, None, :]
        )

    @jax.jit
    def _run_vcirc(vc):
        return (vc - vcirc_mean) / vcirc_std

    def aug(batch):
        batch[key] = _run_sim(jnp.asarray(batch[key]), _stream_ids_jax(batch))
        if vcirc_key in batch:
            batch[vcirc_key] = _run_vcirc(jnp.asarray(batch[vcirc_key]))
        return batch

    return aug


# ------------------------------------------------------------------------------------------- #
# Feature concatenations (must come last)
# ------------------------------------------------------------------------------------------- #


@register_augmentation("concatenate_sigma_errors")
def _concatenate_sigma_errors(params, rng):
    key = _sim_key(params)
    jax, jnp = _jax()

    @jax.jit
    def _run(sim, sigma_errors):
        return jnp.concatenate([sim, sigma_errors], axis=-1)

    def aug(batch):
        batch[key] = _run(jnp.asarray(batch[key]), jnp.asarray(batch["sigma_errors"]))
        return batch

    return aug


@register_augmentation("concatenate_magnitudes")
def _concatenate_magnitudes(params, rng):
    key = _sim_key(params)
    jax, jnp = _jax()

    @jax.jit
    def _run(sim, magnitudes):
        return jnp.concatenate([sim, magnitudes[..., None]], axis=-1)

    def aug(batch):
        batch[key] = _run(jnp.asarray(batch[key]), jnp.asarray(batch["magnitudes"]))
        return batch

    return aug


@register_augmentation("concatenate_vlos_mask")
def _concatenate_vlos_mask(params, rng):
    key = _sim_key(params)
    jax, jnp = _jax()

    @jax.jit
    def _run(sim, vlos_mask):
        vlos_mask = vlos_mask.transpose(0, 2, 1).astype(sim.dtype)
        return jnp.concatenate([sim, vlos_mask], axis=-1)

    def aug(batch):
        batch[key] = _run(jnp.asarray(batch[key]), jnp.asarray(batch["vlos_mask"]))
        return batch

    return aug


@register_augmentation("concatenate_stream_index")
def _concatenate_stream_index(params, rng):
    key = _sim_key(params)
    jax, jnp = _jax()

    @jax.jit
    def _run(sim, j):
        j = j.reshape(-1).astype(sim.dtype)
        j_feat = jnp.broadcast_to(j[:, None, None], (*sim.shape[:2], 1))
        return jnp.concatenate([sim, j_feat], axis=-1)

    def aug(batch):
        batch[key] = _run(jnp.asarray(batch[key]), jnp.asarray(batch["j"]))
        return batch

    return aug

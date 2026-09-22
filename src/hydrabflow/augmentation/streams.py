"""Per-batch augmentations turning clean simulated streams into Gaia-like observations.

JAX port of the reference project's ``AugmentationsClass``
(``utils/utils_train_jax_new_rotationcurve_fixedvlosmask.py``), reorganized as registered,
config-driven augmentation factories. Each step's numeric core is compiled once (at closure-build
time, before the first batch) with ``jax.jit`` and reused every call, fusing its ops into a single
XLA program instead of dispatching each ``jnp`` op separately — this is what the reference project
itself did (``@partial(jit, static_argnums=(0,))`` on its class methods) and measurably raises GPU
utilization here too (dispatch overhead was leaving the GPU idle between ops). The physical chain
(order matters, see ``conf/augmentation/stream_*.yaml``):

1.  ``convert_distance_to_parallax``  — distance [kpc] -> parallax [mas]
2.  ``observational_window``          — per-stream RA/Dec box -> boolean ``attention_mask``
3.  ``observed_n_stars``              — subsample the mask to the observed member count
4.  ``compact_to_attended``           — attended particles first, slice to ``max_particles``
5.  ``sample_magnitudes``             — G magnitudes from a per-stream KDE of observed members
6.  ``sample_obs_error``              — Gaia DR3 uncertainties interpolated at those magnitudes
7.  ``apply_obs_error``               — perturb the observables with those uncertainties
8.  ``mask_vlos``                     — only a subset of members keeps line-of-sight velocity
9.  ``add_noise_to_vcirc``            — observed Vc(R) error bars on the model rotation curve
10. ``log10_vcirc``                   — rotation curve to log10
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
registry (see ``augmentation/registry.py``)
it is used once, at closure-build time, to seed a
``jax.random.PRNGKey``, which is then split on every call (mirroring the reference project's
``_split_key`` idiom) via a one-element mutable cell so each batch draws fresh, reproducible
randomness. The subkey is passed as a normal (traced) argument to each jitted function.

**Compiled-function caching.** ``jax.jit`` recompiles when input shapes change. Every step in this
chain sees a fixed shape per training run except ``sample_magnitudes``, whose particle count
changes once (1000 -> ``max_particles`` after ``compact_to_attended``)
its jitted function is
therefore cached per distinct particle count (mirrors the reference's own per-``n_particles``
branch cache) rather than rebuilt every call.
"""

from __future__ import annotations

import json
import os
from typing import Dict

import numpy as np

from hydrabflow.registry import register_augmentation
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


def _next_key(cell:
    list):
    jax, _ = _jax()
    cell[0], subkey = jax.random.split(cell[0])
    return subkey


class StreamResources:
    """Gaia tables shared by several augmentations: member magnitudes (KDE per stream) and
    DR3 measurement uncertainties per magnitude bin. Loaded once (NumPy/astropy/pandas)
    the
    JAX-side lookup arrays and per-stream KDE objects used every batch are built lazily and
    cached on first use (``jax_*`` / ``kde_streams`` attributes)."""

    def __init__(self, params:
        dict) -> None:
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

        # `magnitude_source: real_streams` instead builds the per-stream magnitude KDE from the G
        # magnitudes stored in `real_streams_file` — the very members the observation model is being
        # matched to. The default (`member_table`) is the Ibata+2024 atlas, which is the right source
        # only when the real set IS that atlas: for another selection its magnitude distribution can
        # differ by ~1 mag, and since every per-star uncertainty is interpolated at these magnitudes
        # that propagates straight into the simulated parallax / proper-motion scatter.
        if str(params.get("magnitude_source", "member_table")) == "real_streams":
            real_file = params.get("real_streams_file")
            if not real_file:
                raise ValueError("magnitude_source=real_streams needs params.real_streams_file")
            with np.load(real_file) as d:
                mags = np.asarray(d["magnitudes"], dtype=float)
                am = np.asarray(d["attention_mask"])
                am = am[:, 0, :] if am.ndim == 3 else am
                js = np.asarray(d["j"]).reshape(-1).astype(int)
            for row, j in enumerate(js):
                g = mags[row][am[row].astype(bool)]
                g = g[np.isfinite(g) & (g > 0)]
                if int(j) in self.stream_by_j and g.size >= 5:
                    self.magnitudes[int(j)] = g

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

    def kde_branches(self, n_particles:
        int):
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


def _resources(params:
    dict) -> StreamResources:
    key = json.dumps({k: v for k, v in params.items()}, sort_keys=True, default=str)
    if key not in _RESOURCE_CACHE:
        _RESOURCE_CACHE[key] = StreamResources(params)
    return _RESOURCE_CACHE[key]


_REAL_VLOS_CACHE: Dict[str, "RealVlosModel"] = {}


class RealVlosModel:
    """Per-stream v_los observation model read off the REAL member npz (2026-09-21):

    * ``has_vlos`` vs G  -> a logistic fit p(G) per stream (which members get a velocity is strongly
      magnitude-dependent: Pal5 91 % of the brightest quartile vs 3 % of the faintest);
    * (G, sigma_vlos) pairs of the measured members, sorted by G -> the empirical, heteroscedastic
      error distribution (literature high-res spectroscopy at G 16-18, DESI at G 18-20.5), which the
      Gaia DR3 error table cannot describe because Gaia RVS does not observe these stars.

    Arrays are padded to a common length across streams so they can be indexed by ``j`` in JAX.
    """

    def __init__(self, params:
        dict) -> None:
        real_file = params.get(
            "real_streams_file",
            os.path.join(params.get("resources_dir", "assets/gaia"),
                         "gaia_observed_streams_6Dwitherrors_cutNGC3201.npz"),
        )
        target = {str(k): int(v) for k, v in params["target_streams"].items()}
        self.n_streams = max(target.values()) + 1
        d = np.load(real_file)
        am = np.asarray(d["attention_mask"])
        am = am[:, 0, :] if am.ndim == 3 else am
        vm = np.asarray(d["vlos_mask"])
        vm = vm[:, 0, :] if vm.ndim == 3 else vm
        mag = np.asarray(d["magnitudes"], dtype=float)
        ve = np.asarray(d["vlos_error"], dtype=float)
        jarr = np.asarray(d["j"]).reshape(-1).astype(int)

        self.logit_coef = np.zeros((self.n_streams, 2))  # p(has_vlos | G) = sigmoid(a + b G)
        g_sorted, s_sorted = [], []
        for row in range(am.shape[0]):
            j = int(jarr[row])
            if j >= self.n_streams:
                continue
            mem = am[row].astype(bool)
            g, y = mag[row][mem], (vm[row].astype(bool) & mem)[mem].astype(float)
            self.logit_coef[j] = self._fit_logistic(g, y)
            meas = y > 0
            order = np.argsort(g[meas])
            g_sorted.append(g[meas][order])
            s_sorted.append(ve[row][mem][meas][order])
        L = max((len(a) for a in g_sorted), default=1)
        self.n_measured = np.array([len(a) for a in g_sorted] + [0] * (self.n_streams - len(g_sorted)))
        self.g_meas = np.full((self.n_streams, L), np.nan)
        self.sigma_meas = np.full((self.n_streams, L), np.nan)
        for j, (g, sg) in enumerate(zip(g_sorted, s_sorted)):
            self.g_meas[j, : len(g)] = g
            self.sigma_meas[j, : len(g)] = sg
            if len(g) < L:
                # pad by repeating the faintest measured star (only reached via clipping)
                self.g_meas[j, len(g):] = g[-1]
                self.sigma_meas[j, len(g):] = sg[-1]
        self._jax_cache: dict = {}

    @staticmethod
    def _fit_logistic(g, y, n_iter=50, ridge=1e-3):
        """IRLS fit of y ~ sigmoid(a + b (G - <G>)); returns (a - b <G>, b). Falls back to a flat
        fraction if one class is empty."""
        if y.min() == y.max():
            frac = float(np.clip(y.mean(), 1e-3, 1 - 1e-3))
            return np.array([np.log(frac / (1 - frac)), 0.0])
        g0 = g.mean()
        X = np.column_stack([np.ones_like(g), g - g0])
        w = np.zeros(2)
        for _ in range(n_iter):
            p = 1.0 / (1.0 + np.exp(-(X @ w)))
            W = p * (1 - p) + 1e-9
            H = X.T @ (X * W[:, None]) + ridge * np.eye(2)
            grad = X.T @ (y - p) - ridge * w
            step = np.linalg.solve(H, grad)
            w = w + step
            if np.abs(step).max() < 1e-8:
                break
        return np.array([w[0] - w[1] * g0, w[1]])

    def jax_lookups(self):
        if not self._jax_cache:
            _, jnp = _jax()
            self._jax_cache = {
                "logit_coef": jnp.asarray(self.logit_coef),
                "g_meas": jnp.asarray(self.g_meas),
                "sigma_meas": jnp.asarray(self.sigma_meas),
                "n_measured": jnp.asarray(self.n_measured),
            }
        return self._jax_cache


class RealAstrometricErrorModel:
    """Per-stream ASTROMETRIC observation errors read off the REAL member npz.

    The default error model interpolates the Gaia DR3 *median* uncertainty-vs-G relation
    (``gaia_DR3_erorr_6D.txt``) at each simulated star's magnitude. That relation is a median over
    all of DR3, so it captures neither a given selection's magnitude distribution (see
    ``magnitude_source``) nor the scatter of sigma at fixed G (crowding, scan coverage, colour).

    When the real file carries the catalogue's own per-star uncertainties — an ``obs_error``
    (1, S, P, 6) array, columns as ``sim_data_projected``, written by
    ``scripts/build_palau23_members.py --astrometry dr3`` — this model instead draws (G, sigma)
    PAIRS from the real members of the same stream, exactly as ``RealVlosModel`` does for v_los.

    Arrays are padded to a common length across streams so they can be indexed by ``j`` in JAX.
    """

    def __init__(self, params: dict, columns) -> None:
        real_file = params.get("real_streams_file")
        if not real_file:
            raise ValueError("sample_obs_error_empirical needs params.real_streams_file")
        target = {str(k): int(v) for k, v in params["target_streams"].items()}
        self.n_streams = max(target.values()) + 1
        self.columns = list(columns)
        with np.load(real_file) as d:
            if "obs_error" not in d.files:
                raise ValueError(
                    f"{real_file} has no per-star `obs_error`; build it with "
                    "scripts/build_palau23_members.py --astrometry dr3, or drop "
                    "`sample_obs_error_empirical` from the chain"
                )
            oe = np.asarray(d["obs_error"], dtype=float)
            oe = oe[0] if oe.ndim == 4 else oe
            am = np.asarray(d["attention_mask"])
            am = am[:, 0, :] if am.ndim == 3 else am
            mag = np.asarray(d["magnitudes"], dtype=float)
            jarr = np.asarray(d["j"]).reshape(-1).astype(int)

        g_sorted, s_sorted = [[] for _ in range(self.n_streams)], [[] for _ in range(self.n_streams)]
        for row in range(am.shape[0]):
            j = int(jarr[row])
            if j >= self.n_streams:
                continue
            mem = am[row].astype(bool)
            g, sig = mag[row][mem], oe[row][mem][:, self.columns]
            ok = np.isfinite(g) & (g > 0) & np.isfinite(sig).all(1) & (sig > 0).all(1)
            order = np.argsort(g[ok])
            g_sorted[j], s_sorted[j] = g[ok][order], sig[ok][order]
        self.n_measured = np.array([len(a) for a in g_sorted])
        if (self.n_measured < 5).any():
            raise ValueError(f"{real_file}: a stream has <5 usable per-star errors "
                             f"{self.n_measured.tolist()}")
        length = int(self.n_measured.max())
        self.g_meas = np.zeros((self.n_streams, length))
        self.sigma_meas = np.zeros((self.n_streams, length, len(self.columns)))
        for j, (g, sig) in enumerate(zip(g_sorted, s_sorted)):
            self.g_meas[j, :len(g)] = g
            self.g_meas[j, len(g):] = g[-1] if len(g) else 0.0
            self.sigma_meas[j, :len(sig)] = sig
            self.sigma_meas[j, len(sig):] = sig[-1] if len(sig) else 0.0
        self._jax_cache: dict = {}

    def jax_lookups(self):
        if not self._jax_cache:
            _, jnp = _jax()
            self._jax_cache = {"g_meas": jnp.asarray(self.g_meas),
                               "sigma_meas": jnp.asarray(self.sigma_meas),
                               "n_measured": jnp.asarray(self.n_measured)}
        return self._jax_cache


_REAL_ASTROM_CACHE: dict = {}


def _real_astrom_model(params: dict, columns) -> RealAstrometricErrorModel:
    key = json.dumps({"real": params.get("real_streams_file"), "cols": list(columns),
                      "target": {str(k): int(v) for k, v in params["target_streams"].items()}},
                     sort_keys=True)
    if key not in _REAL_ASTROM_CACHE:
        _REAL_ASTROM_CACHE[key] = RealAstrometricErrorModel(params, columns)
    return _REAL_ASTROM_CACHE[key]


def _real_vlos_model(params:
    dict) -> RealVlosModel:
    key = json.dumps({"real": params.get("real_streams_file"), "res": params.get("resources_dir"),
                      "target": {str(k): int(v) for k, v in params["target_streams"].items()}}, sort_keys=True)
    if key not in _REAL_VLOS_CACHE:
        _REAL_VLOS_CACHE[key] = RealVlosModel(params)
    return _REAL_VLOS_CACHE[key]


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


def _keep_top_k_weighted_jax(mask, k_per_row, log_weight, key):
    """Sample exactly ``k`` True entries per row WITHOUT replacement with probability proportional
    to ``exp(log_weight)`` (Gumbel-top-k). Unattended entries are never selected."""
    jax, jnp = _jax()
    mask = mask.astype(bool)
    n, p = mask.shape
    gumbel = -jnp.log(-jnp.log(jnp.clip(jax.random.uniform(key, shape=(n, p)), 1e-12, 1 - 1e-12)))
    scores = jnp.where(mask, log_weight + gumbel, -jnp.inf)
    sorted_desc = -jnp.sort(-scores, axis=1)
    k = jnp.clip(k_per_row, 1, p)
    threshold = sorted_desc[jnp.arange(n), k - 1]
    return mask & (scores >= threshold[:, None]) & jnp.isfinite(scores)


# ------------------------------------------------------------------------------------------- #
# Coordinate transforms
# ------------------------------------------------------------------------------------------- #


@register_augmentation("convert_distance_to_parallax")
def _convert_distance_to_parallax(params, rng, context=None):
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
def _remove_los_velocity(params, rng, context=None):
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
def _observational_window(params, rng, context=None):
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
def _observed_n_stars(params, rng, context=None):
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
def _compact_to_attended(params, rng, context=None):
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
def _sample_magnitudes(params, rng, context=None):
    res = _resources(params)
    key = _sim_key(params)
    cell = _key_cell(rng)
    jit_cache: dict = {}

    def _compiled(n_particles:
        int):
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
def _sample_obs_error(params, rng, context=None):
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
def _apply_obs_error(params, rng, context=None):
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
def _mask_vlos(params, rng, context=None):
    res = _resources(params)
    key = _sim_key(params)
    cell = _key_cell(rng)
    jax, jnp = _jax()
    min_star_with_vlos = res.jax_lookups()["min_star_with_vlos"]
    impute = _vlos_impute(params)
    # Which members get a velocity: `random` (historical) or `magnitude` — probability ∝ the
    # logistic p(has_vlos | G) fitted per stream on the real members (RealVlosModel); the count
    # per row is still exactly min_star_with_vlos[j].
    selection = str(params.get("vlos_selection", "random"))
    if selection not in ("random", "magnitude"):
        raise ValueError(f"vlos_selection must be 'random' or 'magnitude', got {selection!r}")
    logit_coef = _real_vlos_model(params).jax_lookups()["logit_coef"] if selection == "magnitude" else None

    @jax.jit
    def _run(sim, sigma, mask, j, subkey, mags):
        if selection == "magnitude":
            a, b = logit_coef[j][:, 0:1], logit_coef[j][:, 1:2]
            log_w = jax.nn.log_sigmoid(a + b * mags)
            vlos_mask = _keep_top_k_weighted_jax(mask, min_star_with_vlos[j], log_w, subkey)
        else:
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

        mags = jnp.asarray(batch["magnitudes"]) if selection == "magnitude" else jnp.zeros(mask.shape)
        sim, sigma, vlos_mask = _run(sim, sigma, mask, j, _next_key(cell), mags)
        batch[key] = sim
        batch["sigma_errors"] = sigma
        batch["vlos_mask"] = vlos_mask
        return batch

    return aug


@register_augmentation("sample_vlos_error_empirical")
def _sample_vlos_error_empirical(params, rng, context=None):
    """Replace the v_los column of ``sigma_errors`` (and the corresponding noise draw in
    ``obs_errors``) with a draw from the REAL measured members' (G, sigma_vlos) pairs of the same
    stream: for each star, one of the ``vlos_error_neighbours`` real stars nearest in G, chosen at
    random. Run after ``sample_obs_error`` and before ``apply_obs_error``. Rationale: the Gaia DR3
    error table is Gaia-RVS-only, but the velocities in the member catalogue come from literature
    spectroscopy and DESI at magnitudes Gaia RVS never reaches (2026-09-21)."""
    model = _real_vlos_model(params)
    cell = _key_cell(rng)
    jax, jnp = _jax()
    lk = model.jax_lookups()
    g_meas, sigma_meas, n_meas = lk["g_meas"], lk["sigma_meas"], lk["n_measured"]
    n_nb = int(params.get("vlos_error_neighbours", 5))
    vlos_idx = -1  # error_keys end with v_los

    @jax.jit
    def _run(mags, j, subkey):
        k1, k2 = jax.random.split(subkey)
        g_j, s_j, n_j = g_meas[j], sigma_meas[j], n_meas[j]  # (rows, L), (rows,)

        def one(gm, gj, sj, nj, key):
            pos = jnp.searchsorted(gj, gm)  # (P,)
            off = jax.random.randint(key, gm.shape, -(n_nb // 2), n_nb - n_nb // 2)
            idx = jnp.clip(pos + off, 0, jnp.maximum(nj - 1, 0))
            return sj[idx]

        sigma_v = jax.vmap(one)(mags, g_j, s_j, n_j, jax.random.split(k1, mags.shape[0]))
        noise_v = sigma_v * jax.random.normal(k2, sigma_v.shape)
        return sigma_v, noise_v

    def aug(batch):
        sigma = jnp.asarray(batch["sigma_errors"])
        obs = jnp.asarray(batch["obs_errors"])
        sigma_v, noise_v = _run(jnp.asarray(batch["magnitudes"]), _stream_ids_jax(batch), _next_key(cell))
        batch["sigma_errors"] = sigma.at[:, :, vlos_idx].set(sigma_v)
        batch["obs_errors"] = obs.at[:, :, vlos_idx].set(noise_v)
        return batch

    return aug


@register_augmentation("sample_obs_error_empirical")
def _sample_obs_error_empirical(params, rng, context=None):
    """Replace the ASTROMETRIC columns of ``sigma_errors`` (and their noise draw in ``obs_errors``)
    with a draw from the REAL members' own (G, sigma) pairs of the same stream — for each simulated
    star, one of the ``obs_error_neighbours`` real stars nearest in G, chosen at random. The v_los
    column is untouched (``sample_vlos_error_empirical`` owns it). Run after ``sample_obs_error``
    and before ``apply_obs_error``.

    ``params.obs_error_columns`` (default parallax / mu_ra / mu_dec) names which of ``error_keys``
    to replace; ra and dec are left alone because the default table gives them zero noise, which is
    the intended convention (Gaia positions are exact at this scale).

    Rationale (2026-09-22): the DR3 median-sigma-vs-G table gets the WIDTH of the simulated parallax
    and proper-motion distributions only as right as its magnitude model, and it has no scatter of
    sigma at fixed G. A catalogue that ships per-star uncertainties can supply both directly.
    """
    res = _resources(params)
    names = [str(c) for c in params.get("obs_error_columns", ["parallax", "mu_ra", "mu_dec"])]
    missing = [c for c in names if c not in res.error_keys]
    if missing:
        raise ValueError(f"obs_error_columns {missing} not in error_keys {res.error_keys}")
    err_idx = [res.error_keys.index(c) for c in names]
    # Columns of the real file's `obs_error` are those of sim_data_projected.
    OBS_COL = {"ra": 0, "dec": 1, "parallax": 2, "mu_ra": 3, "mu_dec": 4, "v_los": 5}
    model = _real_astrom_model(params, [OBS_COL[c] for c in names])
    cell = _key_cell(rng)
    jax, jnp = _jax()
    lk = model.jax_lookups()
    g_meas, sigma_meas, n_meas = lk["g_meas"], lk["sigma_meas"], lk["n_measured"]
    n_nb = int(params.get("obs_error_neighbours", 5))
    idx = jnp.asarray(err_idx)

    @jax.jit
    def _run(mags, j, subkey):
        k1, k2 = jax.random.split(subkey)
        g_j, s_j, n_j = g_meas[j], sigma_meas[j], n_meas[j]

        def one(gm, gj, sj, nj, key):
            pos = jnp.searchsorted(gj, gm)
            off = jax.random.randint(key, gm.shape, -(n_nb // 2), n_nb - n_nb // 2)
            return sj[jnp.clip(pos + off, 0, jnp.maximum(nj - 1, 0))]

        sigma = jax.vmap(one)(mags, g_j, s_j, n_j, jax.random.split(k1, mags.shape[0]))
        return sigma, sigma * jax.random.normal(k2, sigma.shape)

    def aug(batch):
        sigma_new, noise_new = _run(jnp.asarray(batch["magnitudes"]), _stream_ids_jax(batch),
                                    _next_key(cell))
        batch["sigma_errors"] = jnp.asarray(batch["sigma_errors"]).at[:, :, idx].set(sigma_new)
        batch["obs_errors"] = jnp.asarray(batch["obs_errors"]).at[:, :, idx].set(noise_new)
        return batch

    return aug


@register_augmentation("stream_track_width_cut")
def _stream_track_width_cut(params, rng, context=None):
    """Mirror a member-selection WIDTH cut on the simulations: for each stream listed in
    ``params.track_width_cut`` ({name: half_width_deg}), drop stars farther than that from the
    realization's OWN binned-median phi2 track (in the great-circle frame fitted to the real members),
    touching ``attention_mask`` only. Streams not listed are untouched. Run after
    ``observational_window`` and before ``observed_n_stars``, so the member subsample is drawn from
    the thin component only.

    The track is the realization's own (per-row quantile phi1 bins of its in-window stars, per-bin
    median phi2, linear interpolation), NOT the real data's: the split separates a thin component
    from its envelope wherever the stream happens to lie, which is what a main/envelope selection
    does on the data, and does not penalise a realization whose orbit is offset from the observed one.

    Motivation (2026-09-21): the recommended M68 member set is the MAIN component of Palau &
    Miralda-Escude 2025 with their 92-star envelope removed; in the real-fitted frame the main
    component lies within |dphi2| <= 1.8 deg of its track while the envelope starts at 1.1 deg
    (5th percentile 1.6), so a 1.5 deg half-width reproduces their split. Training on all stripped
    stars while evaluating on the thin component would bias every width statistic."""
    from hydrabflow.augmentation.stream_summary import _np_fit_frame

    key = _sim_key(params)
    cuts = {str(k): float(v) for k, v in (params.get("track_width_cut") or {}).items()}
    target = {str(k): int(v) for k, v in params["target_streams"].items()}
    n_streams = max(target.values()) + 1
    k_bins = int(params.get("track_width_bins", 10))
    real_file = params.get(
        "real_streams_file",
        os.path.join(params.get("resources_dir", "assets/gaia"),
                     "gaia_observed_streams_6Dwitherrors_cutNGC3201.npz"),
    )
    d = np.load(real_file)
    sim = np.asarray(d["sim_data_projected"], dtype=float)
    sim = sim[0] if sim.ndim == 4 else sim
    am = np.asarray(d["attention_mask"])
    am = am[:, 0, :] if am.ndim == 3 else am
    jarr = np.asarray(d["j"]).reshape(-1).astype(int)
    R = np.repeat(np.eye(3)[None], n_streams, axis=0)
    half = np.full(n_streams, np.inf)
    for name, w in cuts.items():
        j = target[name]
        row = int(np.flatnonzero(jarr == j)[0])
        s_j = sim[row][am[row].astype(bool)]
        R[j] = _np_fit_frame(s_j[:, 0], s_j[:, 1])
        half[j] = w
    jax, jnp = _jax()
    R_j, half_j = jnp.asarray(R), jnp.asarray(half)
    qs = jnp.linspace(0.0, 1.0, k_bins + 1)

    def _own_track(phi1, phi2, mask):
        """One row: per-bin median phi2 over the row's own phi1 quantile bins; NaN-free by
        construction (every bin holds ~N/K attended stars), evaluated at every star's phi1."""
        p1 = jnp.where(mask, phi1, jnp.nan)
        edges = jnp.nanquantile(p1, qs)
        mids = 0.5 * (edges[1:] + edges[:-1])
        idx = jnp.clip(jnp.searchsorted(edges[1:-1], phi1, side="right"), 0, k_bins - 1)

        def med(b):
            return jnp.nanmedian(jnp.where(mask & (idx == b), phi2, jnp.nan))

        track = jax.vmap(med)(jnp.arange(k_bins))
        ok = jnp.isfinite(track)
        track = jnp.where(ok, track, jnp.nanmedian(jnp.where(mask, phi2, jnp.nan)))
        return jnp.interp(phi1, mids, track)

    @jax.jit
    def _run(sim, mask, j):
        ra, dec = jnp.radians(sim[:, :, 0]), jnp.radians(sim[:, :, 1])
        n = jnp.stack([jnp.cos(dec) * jnp.cos(ra), jnp.cos(dec) * jnp.sin(ra), jnp.sin(dec)], -1)
        v = jnp.einsum("rpk,rmk->rpm", n, R_j[j])
        phi1 = jnp.degrees(jnp.arctan2(v[..., 1], v[..., 0]))
        phi2 = jnp.degrees(jnp.arcsin(jnp.clip(v[..., 2], -1, 1)))
        trk = jax.vmap(_own_track)(phi1, phi2, mask)
        keep = jnp.abs(phi2 - trk) <= half_j[j][:, None]
        return mask & keep

    def aug(batch):
        if not cuts:
            return batch
        mask = jnp.asarray(batch["attention_mask"])[:, 0, :].astype(bool)
        j = _stream_ids_jax(batch)
        batch["attention_mask"] = _run(jnp.asarray(batch[key]), mask, j)[:, None, :]
        return batch

    return aug


@register_augmentation("override_vlos_error_with_real")
def _override_vlos_error_with_real(params, rng, context=None):
    """Real data only: where a member has a measured v_los, use the instrument's uncertainty
    (``vlos_error`` from the real npz) in ``sigma_errors`` instead of the Gaia-table value.

    Layouts (as ``evaluate_real._prepare_real_members`` emits them): ``vlos_mask`` ``(rows, 1, P)``
    (or ``(rows, P)``), ``vlos_error`` ``(rows, P)`` (or with a trailing/middle singleton axis);
    both are normalised to ``(rows, P)`` here."""
    jax, jnp = _jax()

    def _flat(a, n_rows):
        a = jnp.asarray(a)
        return a.reshape(n_rows, -1)

    @jax.jit
    def _run(sigma, vlos_error, vlos_mask):
        return sigma.at[:, :, -1].set(jnp.where(vlos_mask, vlos_error, sigma[:, :, -1]))

    def aug(batch):
        sigma = jnp.asarray(batch["sigma_errors"])
        n_rows = sigma.shape[0]
        vlos_error = _flat(batch["vlos_error"], n_rows)
        vlos_mask = _flat(batch["vlos_mask"], n_rows).astype(bool)
        batch["sigma_errors"] = _run(sigma, vlos_error, vlos_mask)
        return batch

    return aug


@register_augmentation("override_obs_error_with_real")
def _override_obs_error_with_real(params, rng, context=None):
    """Real data only: take the ASTROMETRIC entries of ``sigma_errors`` from the catalogue's own
    per-star uncertainties (the real npz's ``obs_error`` array) instead of the Gaia-table value.

    The real-data counterpart of ``sample_obs_error_empirical``: that step makes the SIMULATED
    sigma channels a draw from the real (G, sigma) distribution, and this one makes the REAL sigma
    channels the actual measurements, so the two sides of `concatenate_sigma_errors` mean the same
    thing. Pair them; using either alone leaves sim and real on different error models.

    ``obs_error`` columns follow ``sim_data_projected`` (ra, dec, parallax, mu_ra, mu_dec, v_los);
    ``params.obs_error_columns`` selects which to override (default parallax / mu_ra / mu_dec).
    Non-finite or non-positive entries keep the table value. No-op when the file has no
    ``obs_error`` — so a real set without per-star errors still runs, on the table model."""
    res = _resources(params)
    names = [str(c) for c in params.get("obs_error_columns", ["parallax", "mu_ra", "mu_dec"])]
    missing = [c for c in names if c not in res.error_keys]
    if missing:
        raise ValueError(f"obs_error_columns {missing} not in error_keys {res.error_keys}")
    OBS_COL = {"ra": 0, "dec": 1, "parallax": 2, "mu_ra": 3, "mu_dec": 4, "v_los": 5}
    err_idx = [res.error_keys.index(c) for c in names]
    obs_idx = [OBS_COL[c] for c in names]
    jax, jnp = _jax()

    @jax.jit
    def _run(sigma, real_sigma):
        for k, c in enumerate(err_idx):
            col = real_sigma[:, :, k]
            sigma = sigma.at[:, :, c].set(
                jnp.where(jnp.isfinite(col) & (col > 0), col, sigma[:, :, c])
            )
        return sigma

    def aug(batch):
        if "obs_error" not in batch:
            return batch
        sigma = jnp.asarray(batch["sigma_errors"])
        oe = jnp.asarray(batch["obs_error"])
        oe = oe.reshape(sigma.shape[0], sigma.shape[1], -1)
        batch["sigma_errors"] = _run(sigma, oe[:, :, jnp.asarray(obs_idx)])
        return batch

    return aug


@register_augmentation("impute_vlos")
def _impute_vlos(params, rng, context=None):
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
def _add_noise_to_vcirc(params, rng, context=None):
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
def _log10_vcirc(params, rng, context=None):
    key = str(params.get("vcirc_key", "vcirc_kms"))
    jax, jnp = _jax()

    # Floor at 1 km/s: `add_noise_to_vcirc` is Gaussian, so a model curve with v_c ~ 40 km/s
    # (legacy broad halo prior at 25 kpc, sigma 17 km/s) goes negative ~1e-5 of the time -> NaN -> a
    # NaN loss that TerminateOnNaN turns into a dead run. A finite floor is harmless: no observed
    # or physical curve is anywhere near it.
    floor = float(params.get("vcirc_floor_kms", 1.0))

    @jax.jit
    def _run(vcirc):
        return jnp.log10(jnp.maximum(vcirc, floor))

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
def _concatenate_sigma_errors(params, rng, context=None):
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
def _concatenate_magnitudes(params, rng, context=None):
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
def _concatenate_vlos_mask(params, rng, context=None):
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
def _concatenate_stream_index(params, rng, context=None):
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


# ------------------------------------------------------------------------------------------- #
# Ibata (2023) ancillary potential observables — observational-error resampling.
#
# Mirror ``add_noise_to_vcirc``: the simulator stores the noise-free MODEL observable (a
# deterministic function of the sampled potential), and these per-batch augmentations add the
# reported observational scatter so the simulated "data" match real measurement precision. Each
# is a no-op unless the corresponding key is present (so they can sit in a shared step list; only
# the Ibata datasets carry vterm_kms / sigma_z / rho_z). Defaults from ``stream_common``.
# ------------------------------------------------------------------------------------------- #


@register_augmentation("add_noise_to_vterm")
def _add_noise_to_vterm(params, rng, context=None):
    from hydrabflow.simulators.stream_common import VTERM_SIGMA_KMS

    key = str(params.get("vterm_key", "vterm_kms"))
    sigma = float(params.get("vterm_sigma_kms", VTERM_SIGMA_KMS))
    cell = _key_cell(rng)
    jax, jnp = _jax()

    @jax.jit
    def _run(vterm, subkey):
        return vterm + jax.random.normal(subkey, shape=vterm.shape) * sigma

    def aug(batch):
        if key in batch:
            batch[key] = _run(jnp.asarray(batch[key]), _next_key(cell))
        return batch

    return aug


@register_augmentation("add_noise_to_sigma_z")
def _add_noise_to_sigma_z(params, rng, context=None):
    from hydrabflow.simulators.stream_common import SIGMA_Z_ERR_MSUN_PC2

    key = str(params.get("sigma_z_key", "sigma_z"))
    sigma = float(params.get("sigma_z_err", SIGMA_Z_ERR_MSUN_PC2))
    cell = _key_cell(rng)
    jax, jnp = _jax()

    @jax.jit
    def _run(sig, subkey):
        return sig + jax.random.normal(subkey, shape=sig.shape) * sigma

    def aug(batch):
        if key in batch:
            batch[key] = _run(jnp.asarray(batch[key]), _next_key(cell))
        return batch

    return aug


@register_augmentation("add_noise_to_rho_z")
def _add_noise_to_rho_z(params, rng, context=None):
    """Resample the vertical stellar-density profile at an assumed per-point RELATIVE uncertainty
    (rho(z) is a shape observable with a free normalization). Noise sigma = rel_err * |rho|."""
    from hydrabflow.simulators.stream_common import RHO_Z_REL_ERR

    key = str(params.get("rho_z_key", "rho_z"))
    rel_err = float(params.get("rho_z_rel_err", RHO_Z_REL_ERR))
    cell = _key_cell(rng)
    jax, jnp = _jax()

    @jax.jit
    def _run(rho, subkey):
        return rho + jax.random.normal(subkey, shape=rho.shape) * rel_err * jnp.abs(rho)

    def aug(batch):
        if key in batch:
            batch[key] = _run(jnp.asarray(batch[key]), _next_key(cell))
        return batch

    return aug


@register_augmentation("log10_rho_z")
def _log10_rho_z(params, rng, context=None):
    """rho(z) spans orders of magnitude over z; feed its log10 to the network (as for vcirc).
    Runs AFTER add_noise_to_rho_z
    the small chance of a negative noised value is clipped."""
    key = str(params.get("rho_z_key", "rho_z"))
    jax, jnp = _jax()

    @jax.jit
    def _run(rho):
        return jnp.log10(jnp.maximum(rho, 1.0))

    def aug(batch):
        if key in batch:
            batch[key] = _run(jnp.asarray(batch[key]))
        return batch

    return aug


# ------------------------------------------------------------------------------------------- #
# Member contamination (interlopers that survive the stream-finder selection)
# ------------------------------------------------------------------------------------------- #


@register_augmentation("contaminate_members")
def _contaminate_members(params, rng, context=None):
    """Replace a prior-drawn fraction of the selected members with interlopers.

    Real stream member catalogues are contaminated, and it is quantified for our streams. Ibata et al.
    (2020) reject 1 of 5 spectroscopic Gjoll (NGC 3201) targets on radial velocity and find a
    metallicity spread ~30x the cluster's intrinsic sigma, stating plainly that not all of the
    candidates were stripped from NGC 3201
    modern Pal 5 samples still flag dozens of candidates as
    deviating from the known kinematic/CMD trends. For M68 — where the simulated-vs-real dispersion
    gap is worst — there is no published width or velocity-dispersion measurement at all, so the
    target the forward model is being asked to match is itself unvalidated.

    Modelling choice: interlopers are drawn as **broadened look-alikes of that row's own members**,
    not as uniform field stars. Contaminants in a real catalogue are precisely the objects that
    passed the stream-finder's kinematic + CMD cuts, so they resemble members with inflated scatter;
    drawing them from the raw field would make them trivially separable and would understate their
    effect. Sky positions are redrawn uniformly across the stream's observational window (an
    interloper carries no memory of the track), while parallax / proper motions / v_los are redrawn
    from the member mean with the dispersion inflated by ``contamination_inflate``.

    Marginalising this fraction is the remedy the misspecification literature recommends when the
    mismatch is traceable to a specific process (Kelly et al. 2025), and it guards against tuning the
    physical priors to a width that is partly observational.

    Params (a no-op when ``contamination_max_frac`` is 0, the default, so the step is safe to leave in
    a chain): ``contamination_max_frac`` — upper end of the per-row Uniform[0, f] contaminated
    fraction
    ``contamination_inflate`` — dispersion inflation factor for the redrawn kinematics.

    Runs after ``compact_to_attended`` and before ``sample_magnitudes`` so contaminants pick up
    magnitudes and Gaia uncertainties through exactly the same path as members (they are, after all,
    real Gaia stars).
    """
    res = _resources(params)
    key = _sim_key(params)
    max_frac = float(params.get("contamination_max_frac", 0.0))
    inflate = float(params.get("contamination_inflate", 3.0))
    cell = _key_cell(rng)
    jax, jnp = _jax()
    lookups = res.jax_lookups()
    ra_min, ra_max = lookups["ra_min"], lookups["ra_max"]
    dec_min, dec_max = lookups["dec_min"], lookups["dec_max"]

    @jax.jit
    def _run(sim, attn, j, subkey):
        attended = attn[:, 0, :].astype(bool)  # (n, P)
        n, P = attended.shape
        k_frac, k_pick, k_pos, k_kin = jax.random.split(subkey, 4)

        # per-row contaminated fraction ~ U[0, max_frac]
        frac = jax.random.uniform(k_frac, (n, 1)) * max_frac
        # pick which attended slots become interlopers (uniform draw per slot, thresholded)
        picked = (jax.random.uniform(k_pick, (n, P)) < frac) & attended

        # member statistics per row, over the attended stars only
        w = attended.astype(sim.dtype)
        cnt = jnp.maximum(w.sum(1, keepdims=True), 1.0)
        mean = (sim * w[..., None]).sum(1) / cnt  # (n, 6)
        var = (((sim - mean[:, None, :]) ** 2) * w[..., None]).sum(1) / cnt
        std = jnp.sqrt(jnp.maximum(var, 0.0))  # (n, 6)

        # interloper sky position: uniform across this stream's observational window
        u = jax.random.uniform(k_pos, (n, P, 2))
        ra = ra_min[j][:, None] + u[..., 0] * (ra_max[j] - ra_min[j])[:, None]
        dec = dec_min[j][:, None] + u[..., 1] * (dec_max[j] - dec_min[j])[:, None]
        # interloper kinematics: member mean + inflated scatter
        kin = mean[:, None, :] + inflate * std[:, None, :] * jax.random.normal(k_kin, (n, P, 6))

        replacement = jnp.concatenate([ra[..., None], dec[..., None], kin[..., 2:]], axis=-1)
        return jnp.where(picked[..., None], replacement, sim)

    def aug(batch):
        if max_frac <= 0.0:
            return batch
        batch[key] = _run(
            jnp.asarray(batch[key]),
            jnp.asarray(batch["attention_mask"]),
            _stream_ids_jax(batch),
            _next_key(cell),
        )
        return batch

    return aug

"""Parameter spec, condition layout and adapter for the protoplanetary-disk RT dataset.

Ported from `ProtoplanetaryDisk_SBI`'s `params.py` + `npe/setup.py`.  Underscore-prefixed so
`registry.discover()` skips it; `simulators/protoplan.py` imports it.

The encoding is fixed to the one that arm settled on, and the three choices are coupled:

* **discrete-as-conditions** -- `sil_id` and a derived `has_cavity` are NPE *conditions*, not
  targets, so the estimator is `p(theta_17 | data, sil_id, has_cavity)` and inference enumerates
  the four combinations.  A `FlowMatching` posterior cannot represent a delta atom; the
  alternative was smearing both indicators into continuous clusters, and the `log_r_cav` jitter
  range that went with it was *disjoint* from where real cavity-bearing disks live.
* **conditions in the summary networks** -- the geometry and beam/noise scalars are spliced into
  the CNN trunks rather than concatenated onto the already-compressed summary vector.  Whether a
  faint ring survived the beam is a question about the *image*, and the CNN is the only network
  that sees the image.  `inference_conditions` then holds only the two discrete indicators, which
  are conditions on the density rather than descriptions of the measurement.
* **asinh images** -- see `_asinh_images`.

Numpy only: this module must be importable before the GPU/backend is pinned.
"""

from __future__ import annotations

import os

import numpy as np

# The 18 physical parameters.  The leading "num" placeholder column of the raw params array is
# dropped before this list is used.
PARAMS = [
    "rc", "h_c", "gamma", "psi", "log_Mgas", "f_lg", "XH_lg", "T_star",
    "L_star", "inc", "log_amin", "log_amax_lg", "log_amax_sm", "sil_id",
    "X_sil", "log_X_H2O", "log_r_cav", "log_delta_cav",
]

#: Derived, not a column of `params_combined.npy`: `log_r_cav != NO_CAVITY_LOG_R_CAV`.
HAS_CAVITY = "has_cavity"

#: The two discrete parameters conditioned on rather than estimated, in adapter column order.
DISCRETE_CONDITION_PARAMS = ["sil_id", HAS_CAVITY]

#: What the posterior is over: 17 targets, `sil_id` removed (`has_cavity` was never in `PARAMS`).
NPE_TARGET_PARAMS = [p for p in PARAMS if p != "sil_id"]

#: The raw grid's own sentinel for "no cavity" on `log_r_cav`, exact in every raw row.
NO_CAVITY_LOG_R_CAV = -2.0
#: The real, continuous branch's support, measured off the raw grid: every row with
#: `log_r_cav != NO_CAVITY_LOG_R_CAV` falls in here.
CAVITY_LOG_R_CAV_RANGE = (0.45, 1.90)

N_ALMA = 3           # im_jy_alma_0/1/2  <-> 450 / 880 / 1300 um

#: The three ALMA channels, in the order the image cache's channel axis has them (channel 1+j of
#: `im_jy`), each with its own summary branch.  The bands are given their own networks rather than
#: stacked as channels of one CNN because they are observed with *different beams*: a shared
#: convolutional trunk would have to serve three different point-spread functions at once, and the
#: per-band beam/noise conditions could only be handed to it pooled.
ALMA_BANDS = ("b9", "b7", "b6")          # 450 um -> B9, 880 um -> B7, 1300 um -> B6
ALMA_BAND_LAMS_UM = (450.0, 880.0, 1300.0)

#: `{band: input key}`, i.e. what the adapter names each band's image and what the summary network
#: keys its backbone by.
ALMA_INPUT_KEYS = {band: f"{band}_input" for band in ALMA_BANDS}
IMAGE_INPUT_KEYS = ["jwst_input", *(ALMA_INPUT_KEYS[b] for b in ALMA_BANDS)]

#: Distance the RT models were run at [pc].  Every cached image and SED is "as seen from here";
#: `AugmentationsClass(dist_pc=...)` rescales to a target source's distance.
DIST_FID_PC = 140.0

#: The RT output covers a fixed 3.0" field of view; resized caches keep the FOV and change only
#: the sampling, so arcsec/pixel must be recomputed per cache.  `AugmentationsClass` sizes its
#: convolution kernels from `px_arcsec_mod`, *not* from the array shape, so feeding a resized
#: cache without updating it is a silent beam error.
FOV_ARCSEC_MOD = 3.0

# ── Image asinh softening scales, MJy/sr ──────────────────────────────────────────────
# The per-channel scale at which `asinh(x / sigma)` turns over from linear to logarithmic.
# These are one specific disk's measured per-band noise with *uncorrelated* Gaussian noise, so
# they are only the right knee for a run whose configured noise is of that order -- override
# them from config (`simulator.params.asinh_sigma_*`) when the noise changes.  asinh needs the
# knee in the right ballpark, not exactly right, but far enough off and the transform degenerates
# into either a near-identity (knee >> noise) or a pure log (knee << noise).
ASINH_SIGMA_JWST = 0.9801783804396227
#: One entry per `ALMA_BANDS`, in that order (450 / 880 / 1300 um).
ASINH_SIGMA_ALMA = (1744.6539538736572, 154.95994745585898, 5.556490901687079)


# ══════════════════════════════════════════════════════════════════════════════════════
# Loading
# ══════════════════════════════════════════════════════════════════════════════════════

def filter_and_jitter_params(data_path, rng: np.random.Generator | None = None):
    """Read `params_combined.npy` and apply the discrete-as-conditions encoding.

    The "num" placeholder column is dropped, and so are the 18 of 49829 rows whose SED holds an
    exactly-zero flux: clipping them to 1e-10 does not help, since the SED sigma then becomes
    ~1e40 relative to the flux and the log10 compression turns the result into ~-22 against a
    typical -3, i.e. an extreme outlier for the standardizer. Dropping 0.036% of rows is cheaper
    and more honest. `keep` is the row mask to apply to every other array in `data_path`.

    `has_cavity` is derived *before* `log_r_cav` is touched.  `log_r_cav` stays a target and on
    the no-cavity rows is drawn `Uniform(*CAVITY_LOG_R_CAV_RANGE)` -- the real cavity branch.
    Conditional on `has_cavity=0` the parameter is unidentified, so a posterior that returns the
    prior there is the honest answer; drawing from the real branch is what makes that statement
    true rather than an artefact of an off-support range.  Holding the sentinel instead would
    reintroduce exactly the atom this avoids.  Neither indicator is jittered -- a condition is a
    plain concatenation into the network, never a density target.

    Shapes differ by role and that is load-bearing: targets come back `(N, 1)`, the two
    conditions come back **flat `(N,)`**, because `build_adapter` runs `to_array` +
    `expand_dims(axis=-1)` on every scalar condition.  An `(N, 1)` condition would arrive at the
    concatenation as `(N, 1, 1)`.

    Returns `(target_names, columns, keep)`; `columns` holds the 17 targets *and* the two
    conditions, which are in `columns` but not in `target_names`.
    """
    uniform = np.random.uniform if rng is None else rng.uniform

    params_data = np.load(os.path.join(data_path, "params_combined.npy"), allow_pickle=True)
    keep = ~np.any(
        np.asarray(np.load(os.path.join(data_path, "sed_flx_jy_combined.npy"), mmap_mode="r")) <= 0,
        axis=1)
    params_data = params_data[keep]
    columns = {p: params_data[:, 1 + i].copy() for i, p in enumerate(PARAMS)}

    log_r_cav = columns["log_r_cav"]
    no_cavity = log_r_cav == NO_CAVITY_LOG_R_CAV
    has_cavity = (~no_cavity).astype(log_r_cav.dtype)
    log_r_cav[no_cavity] = uniform(*CAVITY_LOG_R_CAV_RANGE, size=int(no_cavity.sum()))

    targets = list(NPE_TARGET_PARAMS)
    out = {p: columns[p].reshape(-1, 1) for p in targets}
    out["sil_id"] = columns["sil_id"].reshape(-1)
    out[HAS_CAVITY] = has_cavity.reshape(-1)
    return targets, out, keep


# ══════════════════════════════════════════════════════════════════════════════════════
# Condition layout
# ══════════════════════════════════════════════════════════════════════════════════════
# Which observing-setup scalars are concatenated into which summary-group key.  The split is per
# instrument: each CNN is told the setup that produced *its own* image and nothing else.
# `rot_*`/`sky_flip` are shared, because one rotation was applied to every channel of the same
# model image -- so they get their own key and both branches read it.  `Concatenate` *pops* the
# keys it consumes, so one key cannot feed two groups; the reuse happens inside the network
# (`networks.protoplan.ConditionedFusionNetwork.routing`).

#: Geometry nuisances.  These describe how the *source* was placed on the grid.
GEOMETRY_CONDITION_KEYS = ["rot_sin", "rot_cos", "sky_flip"]

GEOMETRY_SUMMARY_KEY = "geom_cond"
JWST_SUMMARY_COND_KEY = "jwst_cond"

#: `{band: its condition-group key}`.  Per band, not pooled: each band's CNN is told the beam and
#: noise that produced *its own* map and nothing else.
ALMA_SUMMARY_COND_KEYS = {band: f"{band}_cond" for band in ALMA_BANDS}

JWST_NOISE_CONDITION_KEYS = ["sigma_jwst"]


def alma_band_condition_keys(j: int) -> list[str]:
    """The five batch keys describing ALMA channel `j`'s beam and noise, in column order."""
    return [f"sigma_alma_{j}", f"beam_maj_{j}", f"beam_min_{j}",
            f"beam_pa_sin_{j}", f"beam_pa_cos_{j}"]


#: Sampled log-uniformly and spanning decades, so logged before the standardizer sees them.
#: Angles are conditioned as sin/cos (a raw angle has a discontinuity at the wrap that the network
#: would have to learn around), and are therefore *not* in here.
LOG_CONDITION_KEYS = (
    ["sigma_jwst"]
    + [f"sigma_alma_{j}" for j in range(N_ALMA)]
    + [f"beam_maj_{j}" for j in range(N_ALMA)]
    + [f"beam_min_{j}" for j in range(N_ALMA)]
)

#: `inference_conditions`, in column order.  Deliberately not log'd: 0/1 indicators, not scales.
DISCRETE_CONDITION_KEYS = list(DISCRETE_CONDITION_PARAMS)


def summary_condition_groups() -> dict[str, list[str]]:
    """`{summary_group_key: [scalar batch keys, in concatenation order]}`."""
    groups = {
        GEOMETRY_SUMMARY_KEY: list(GEOMETRY_CONDITION_KEYS),
        JWST_SUMMARY_COND_KEY: list(JWST_NOISE_CONDITION_KEYS),
    }
    for j, band in enumerate(ALMA_BANDS):
        groups[ALMA_SUMMARY_COND_KEYS[band]] = alma_band_condition_keys(j)
    return groups


def summary_condition_routing() -> dict[str, list[str]]:
    """`{backbone key: [summary condition keys it receives]}` -- derived from the same source as
    the adapter's groups, so the two cannot disagree.

    Geometry is read by every image branch, which is why it is its own key: one rotation was
    applied to every channel of the same model image, and `Concatenate` *pops* the keys it
    consumes, so a single key cannot feed several groups.  The reuse happens inside the network.
    """
    routing = {"jwst_input": [GEOMETRY_SUMMARY_KEY, JWST_SUMMARY_COND_KEY]}
    for band in ALMA_BANDS:
        routing[ALMA_INPUT_KEYS[band]] = [GEOMETRY_SUMMARY_KEY, ALMA_SUMMARY_COND_KEYS[band]]
    return routing


def summary_variable_keys() -> list[str]:
    """Every member of `summary_variables`: the five modalities plus the condition groups.

    The condition groups are members so `standardize="all"` covers them: `Standardize` flattens
    the group and keeps one mean/sd per leaf key and column, so the log-sigmas get their own
    statistics rather than sharing the images'.
    """
    return [*IMAGE_INPUT_KEYS, "sed_input", *summary_condition_groups()]


# ══════════════════════════════════════════════════════════════════════════════════════
# Adapter
# ══════════════════════════════════════════════════════════════════════════════════════

def _asinh_images(adapter, sigma_jwst=None, sigma_alma=None):
    """Compress each image key as `asinh(x / sigma)`, per band.

    Without it the image keys reach the networks untransformed and their only preprocessing is
    `standardize="all"` -- one global mean/sd per channel, since `Standardize._update_moments`
    reduces over every axis but the last.  That cannot serve a channel whose pixels span five
    decades: the sd is set by the inner rim, so measured over the full training set the p10-p90
    span of *background* pixels is 0.004 sd on JWST and 0.16 on ALMA 1300um, against a rim
    reaching 533 sd and a skew of 195.  Under asinh the same spans are 2.04 and 1.67, with
    max|z| < 13.

    asinh and not log/log1p because ~40-48% of post-noise pixels are negative: asinh is odd and
    finite for x < 0, so a -2 sigma excursion is treated exactly as +2 sigma.

    Built from stock `.scale` + `.apply` so the adapter stays serializable with no registered
    functions to keep alive for the lifetime of every checkpoint (`apply_serializable` would
    demand exactly that, and a renamed function then silently breaks reloading).  `arcsinh` is
    not in `NumpyTransform.INVERSE_METHODS`, so the inverse is named explicitly; it is a real
    inverse, which matters because the adapter must round-trip.
    """
    alma = ASINH_SIGMA_ALMA if sigma_alma is None else np.reshape(sigma_alma, (-1,))
    if len(alma) != N_ALMA:
        raise ValueError(f"asinh_sigma_alma needs {N_ALMA} entries (one per ALMA band, "
                         f"{ALMA_BAND_LAMS_UM} um), got {len(alma)}")

    # float32 *arrays* of shape (1,), not the scalars they could be: a scalar survives
    # `serialize_keras_object` only as a Python float, so a reloaded adapter would silently
    # promote that channel to float64.
    sigma = {"jwst_input": np.asarray(
        [ASINH_SIGMA_JWST if sigma_jwst is None else sigma_jwst], dtype=np.float32)}
    for j, band in enumerate(ALMA_BANDS):
        sigma[ALMA_INPUT_KEYS[band]] = np.asarray([alma[j]], dtype=np.float32)

    for key, s in sigma.items():
        adapter = adapter.scale(key, by=1.0 / s)
    return adapter.apply(list(sigma), forward="arcsinh", inverse="sinh")


def build_adapter(params=None, conditions=None, asinh_sigma_jwst=None, asinh_sigma_alma=None):
    """Map the augmented batch dict onto BayesFlow's variable groups.

    `params` defaults to `NPE_TARGET_PARAMS` and `conditions` to `DISCRETE_CONDITION_KEYS`.  The
    layout is the one the module docstring fixes: the discrete indicators in
    `inference_conditions`, everything else about the measurement routed into `summary_variables`
    alongside the three modalities.

    Dropping an indicator from `conditions` does not make it a target -- it makes the posterior
    marginal over it.  `[has_cavity]` alone estimates `p(theta | data, has_cavity)`, i.e. the
    silicate type is integrated out rather than conditioned on.  Keep
    `model.inference_network.params.discrete_condition_dim` in step: it is the width of the
    always-observed leading condition group, and `GroupedFlowMatching` raises at the first
    training step if the two disagree.
    """
    import bayesflow as bf

    params = list(NPE_TARGET_PARAMS if params is None else params)
    summary_groups = summary_condition_groups()
    routed = [k for keys in summary_groups.values() for k in keys]
    scalars = list(DISCRETE_CONDITION_KEYS if conditions is None else conditions)
    unknown = [k for k in scalars if k not in DISCRETE_CONDITION_KEYS]
    if unknown:
        raise ValueError(
            f"inference_conditions {unknown} are not discrete indicators; this adapter routes "
            f"every other condition into the summary networks. Choose from "
            f"{DISCRETE_CONDITION_KEYS}.")

    adapter = (
        bf.adapters.Adapter()
        .concatenate(params, into="inference_variables")
        .convert_dtype("float64", "float32")
    )

    # Every scalar condition arrives as (N,) and must become (N, 1) before concatenation.
    for k in scalars + routed:
        adapter = adapter.to_array(k)
    adapter = adapter.to_array("sigma_sed_flux")

    adapter = adapter.log(LOG_CONDITION_KEYS)

    for k in scalars + routed:
        adapter = adapter.expand_dims(k, axis=-1)

    adapter = adapter.concatenate(scalars, into="inference_conditions")
    for group, keys in summary_groups.items():
        adapter = adapter.concatenate(keys, into=group)

    # One key per band, not the three ALMA channels concatenated: each gets its own CNN, because
    # each was observed with its own beam.  The augmentation already splits `im_jy` into
    # `im_jy_jwst` + `im_jy_alma_{0,1,2}` (channel 1+j), so this is a pure rename.
    adapter = adapter.rename("im_jy_jwst", "jwst_input")
    for j, band in enumerate(ALMA_BANDS):
        adapter = adapter.rename(f"im_jy_alma_{j}", ALMA_INPUT_KEYS[band])

    # Before the group, so the keys still exist at the top level.
    adapter = _asinh_images(adapter, asinh_sigma_jwst, asinh_sigma_alma)

    return (
        adapter
        .expand_dims("sigma_sed_flux", axis=-1)
        .concatenate(["seds", "sigma_sed_flux", "sed_lams"], into="sed_input")
        .group(summary_variable_keys(), into="summary_variables")
    )

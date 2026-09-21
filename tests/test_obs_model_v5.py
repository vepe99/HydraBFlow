"""v5 observation model (2026-09-21): magnitude-weighted v_los selection, empirical v_los errors,
the stream-frame width cut, and the v5 presets."""
from __future__ import annotations

import numpy as np

REAL = "assets/gaia/gaia_observed_streams_6Dwitherrors_cutNGC3201_desi_m68palau_main.npz"
SIM_KEY = "sim_data_projected"


def _params(**extra):
    p = {
        "observable": SIM_KEY,
        "resources_dir": "assets/gaia",
        "target_streams": {"Pal5": 0, "NGC3201": 1, "M68": 2},
        "gaia_id": {"Pal5": "Pal-5", "NGC3201": "Gjoll", "M68": "Fjorm"},
        "observational_window": {
            "Pal5": dict(ra_min=220.98, ra_max=250.0, dec_min=-13.0, dec_max=10.0),
            "NGC3201": dict(ra_min=42.3, ra_max=140.0, dec_min=-47.8, dec_max=20.8),
            "M68": dict(ra_min=189.74, ra_max=280.0, dec_min=-47.3, dec_max=70.0),
        },
        "observed_n_stars": {"Pal5": 129, "NGC3201": 195, "M68": 195},
        "min_star_with_vlos": {"Pal5": 77, "NGC3201": 48, "M68": 16},
        "real_streams_file": REAL,
    }
    p.update(extra)
    return p


def _build(name, params, seed=0):
    from hydrabflow.registry import AUGMENTATIONS

    return AUGMENTATIONS.get(name)(params, np.random.default_rng(seed), {})


def _batch(n=6, particles=300, attended=250, seed=0):
    rng = np.random.default_rng(seed)
    sim = np.zeros((n, particles, 6), dtype=np.float32)
    sim[..., 0] = rng.uniform(225.0, 245.0, (n, particles))
    sim[..., 1] = rng.uniform(-5.0, 5.0, (n, particles))
    sim[..., 5] = rng.normal(-58.0, 2.0, (n, particles))
    mask = np.zeros((n, 1, particles), dtype=bool)  # bool, as compact_to_attended emits it
    mask[:, 0, :attended] = True
    mags = rng.uniform(14.0, 20.5, (n, particles)).astype(np.float32)
    sigma = np.ones((n, particles, 6), dtype=np.float32)
    return {
        SIM_KEY: sim, "attention_mask": mask, "magnitudes": mags, "sigma_errors": sigma,
        "obs_errors": np.zeros_like(sigma), "j": np.array([[i % 3] for i in range(n)]),
    }


def test_real_vlos_model_fits_the_recommended_set():
    from hydrabflow.augmentation.streams import RealVlosModel

    m = RealVlosModel(_params())
    assert list(m.n_measured) == [77, 48, 16]
    # brighter stars are more likely to have a velocity in every stream (negative slope in G)
    assert (m.logit_coef[:, 1] < 0).all()
    # measured sigma arrays are sorted by G and finite
    assert np.all(np.diff(m.g_meas[0, :77]) >= 0) and np.isfinite(m.sigma_meas).all()


def test_mask_vlos_magnitude_mode_keeps_exact_count_and_prefers_bright_stars():
    b = _batch()
    out = _build("mask_vlos", _params(vlos_selection="magnitude"))(b)
    vm = np.asarray(out["vlos_mask"])[:, 0, :].astype(bool)
    att = b["attention_mask"][:, 0, :].astype(bool)
    for i, k in enumerate([77, 48, 16] * 2):
        assert vm[i].sum() == k and not vm[i][~att[i]].any()
        assert np.median(b["magnitudes"][i][vm[i]]) < np.median(b["magnitudes"][i][att[i] & ~vm[i]])


def test_mask_vlos_random_mode_unchanged():
    b = _batch()
    out = _build("mask_vlos", _params(vlos_selection="random"))(b)
    vm = np.asarray(out["vlos_mask"])[:, 0, :].astype(bool)
    assert [int(v.sum()) for v in vm[:3]] == [77, 48, 16]


def test_empirical_vlos_error_tracks_magnitude_and_stays_in_real_range():
    b = _batch(n=3)
    out = _build("sample_vlos_error_empirical", _params())(b)
    sig = np.asarray(out["sigma_errors"])[:, :, -1]
    real = np.load(REAL)
    am = real["attention_mask"][:, 0, :].astype(bool)
    vm = real["vlos_mask"].astype(bool)
    for i in range(3):
        ve = real["vlos_error"][i][am[i] & vm[i]]
        assert sig[i].min() >= ve.min() - 1e-6 and sig[i].max() <= ve.max() + 1e-6
        # only the v_los column changed; the noise draw was regenerated with the new sigma
        assert np.allclose(np.asarray(out["sigma_errors"])[i, :, :5], 1.0)
        assert np.abs(np.asarray(out["obs_errors"])[i, :, -1]).max() > 0
    # heteroscedastic: faint stars carry larger errors on average (Pal5: 0.1 -> 3.4 km/s by quartile)
    g = b["magnitudes"][0]
    assert sig[0][g > 19].mean() > sig[0][g < 17].mean()


def test_track_width_cut_removes_envelope_around_the_realizations_own_track():
    from hydrabflow.augmentation.stream_summary import _np_fit_frame, _np_unit_vec

    real = np.load(REAL)
    am = real["attention_mask"][:, 0, :].astype(bool)
    s = real["sim_data_projected"][0][2][am[2]]  # real M68 main-component members
    R = _np_fit_frame(s[:, 0], s[:, 1])
    v = _np_unit_vec(s[:, 0], s[:, 1]) @ R.T
    phi1 = np.arctan2(v[:, 1], v[:, 0])
    phi2 = np.arcsin(v[:, 2])

    def to_sky(p1, p2):
        xyz = np.stack([np.cos(p2) * np.cos(p1), np.cos(p2) * np.sin(p1), np.sin(p2)], -1) @ R
        return np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0])) % 360, np.degrees(np.arcsin(xyz[:, 2]))

    n_env = len(s) // 4  # a 25 % envelope 4 deg off the thin component, so the median stays on it
    rng = np.random.default_rng(0)
    e1 = rng.choice(phi1, n_env)
    e2 = np.interp(e1, np.sort(phi1), phi2[np.argsort(phi1)]) + np.radians(4.0) * rng.choice([-1, 1], n_env)
    ra_e, dec_e = to_sky(e1, e2)
    # shift the WHOLE realization 3 deg off the real track: a thin stream that is merely offset
    # from the observed one must keep its main component (the cut is about its own track)
    ra_s, dec_s = to_sky(phi1, phi2 + np.radians(3.0))
    ra_se, dec_se = to_sky(e1, e2 + np.radians(3.0))
    particles = len(s) + n_env
    sim = np.zeros((3, particles, 6), dtype=np.float32)
    sim[0, : len(s), 0], sim[0, : len(s), 1] = s[:, 0], s[:, 1]
    sim[0, len(s):, 0], sim[0, len(s):, 1] = ra_e, dec_e
    sim[1, : len(s), 0], sim[1, : len(s), 1] = ra_s, dec_s
    sim[1, len(s):, 0], sim[1, len(s):, 1] = ra_se, dec_se
    sim[2] = sim[0]  # Pal5 index: identical stars, must NOT be cut
    mask = np.ones((3, 1, particles), dtype=bool)
    out = _build("stream_track_width_cut", _params(track_width_cut={"M68": 1.5}))(
        {SIM_KEY: sim, "attention_mask": mask, "j": np.array([[2], [2], [0]])}
    )
    m = np.asarray(out["attention_mask"])[:, 0, :]
    for row in (0, 1):
        assert m[row, : len(s)].mean() > 0.95, row
        assert not m[row, len(s):].any(), row
    assert m[2].all()


def test_track_width_cut_without_config_is_a_no_op():
    b = _batch(n=3)
    out = _build("stream_track_width_cut", _params())(b)
    np.testing.assert_array_equal(np.asarray(out["attention_mask"]), b["attention_mask"])


def test_v5_presets_compose_and_match_the_real_set(compose):
    cfg = compose(
        ["simulator=stream_agama_spray_massloss_ibata_m200c_v4", "model=stream_fusion_2modal",
         "adapter=stream_2modal", "augmentation=stream_global_v5",
         "preprocessing=stream_global_log10_ibata_sumstats", "composition=global"]
    )
    steps = list(cfg.augmentation.steps)
    assert steps.index("observational_window") < steps.index("stream_track_width_cut") < steps.index("observed_n_stars")
    assert steps.index("sample_obs_error") < steps.index("sample_vlos_error_empirical") < steps.index("apply_obs_error")
    real = np.load(cfg.augmentation.params.real_streams_file)
    am = real["attention_mask"][:, 0, :]
    vm = real["vlos_mask"]
    p = cfg.augmentation.params
    for j, nm in enumerate(("Pal5", "NGC3201", "M68")):
        assert int(am[j].sum()) == p.observed_n_stars[nm]
        assert int((am[j] * vm[j]).sum()) == p.min_star_with_vlos[nm]
    rc = compose(["simulator=stream_agama_spray_massloss_ibata_m200c_v4", "augmentation=stream_real_global_v5"])
    assert "override_vlos_error_with_real" in list(rc.augmentation.steps)
    assert "stream_track_width_cut" not in list(rc.augmentation.steps)
    assert rc.augmentation.params.summary_vlos_bins == p.summary_vlos_bins == 2
    assert rc.augmentation.params.real_streams_file == p.real_streams_file


def test_override_vlos_error_with_real_uses_the_instrument_error_in_real_layout():
    """Real batches carry vlos_mask (rows,1,P) and vlos_error (rows,P) (evaluate_real layout); the
    measured members must end up with exactly their catalogue error in sigma_errors."""
    real = np.load(REAL)
    rows, P = real["vlos_error"].shape
    sigma = np.full((rows, P, 6), 7.0, dtype=np.float32)
    batch = {
        "sigma_errors": sigma, "vlos_error": real["vlos_error"],
        "vlos_mask": real["vlos_mask"][:, None, :], "j": real["j"][0],
    }
    out = np.asarray(_build("override_vlos_error_with_real", _params())(batch)["sigma_errors"])
    vm = real["vlos_mask"].astype(bool)
    np.testing.assert_allclose(out[:, :, -1][vm], real["vlos_error"][vm], rtol=1e-6)
    assert np.all(out[:, :, -1][~vm] == 7.0) and np.all(out[:, :, :5] == 7.0)


def test_v5_nodisp_presets_drop_the_dispersions_and_the_model_contract_matches(compose):
    """stream_global_v5_nodisp emits the 9-channel medians-only grid and the nodisp model's channel
    indices address exactly that layout (counts at 5/6, medians at 0-3 and 4)."""
    cfg = compose(
        ["simulator=stream_agama_spray_massloss_ibata_m200c_v4", "model=stream_fusion_2modal_nodisp",
         "adapter=stream_2modal", "augmentation=stream_global_v5_nodisp",
         "preprocessing=stream_global_log10_ibata_sumstats", "composition=global"]
    )
    p = cfg.augmentation.params
    assert p.summary_include_std is False and p.summary_include_occupancy is True
    assert p.observed_n_stars.M68 == 195 and p.track_width_cut.M68 == 1.5  # v5 inherited
    assert "stream_track_width_cut" in list(cfg.augmentation.steps)
    bb = cfg.model.summary_network.params.backbones.sim_summary.params
    assert bb.count_channel == 5 and bb.vlos_count_channel == 6
    assert list(bb.stat_channels) == [0, 1, 2, 3] and list(bb.vlos_stat_channels) == [4]
    assert cfg.model.summary_network.params.head is None
    assert cfg.model.name == "stream_fusion_2modal_nodisp"
    real = compose(["simulator=stream_agama_spray_massloss_ibata_m200c_v4",
                    "augmentation=stream_real_global_v5_nodisp"])
    assert real.augmentation.params.summary_include_std is False
    assert "override_vlos_error_with_real" in list(real.augmentation.steps)

    # the chain really emits 9 channels: run the grid step on a toy batch with the nodisp params
    from omegaconf import OmegaConf

    params = OmegaConf.to_container(cfg.augmentation, resolve=True)["params"]
    params["resources_dir"] = "assets/gaia"
    n, P = 3, 60
    rng = np.random.default_rng(0)
    real_np = np.load(REAL)
    am = real_np["attention_mask"][:, 0, :].astype(bool)
    sim = np.zeros((n, P, 6), dtype=np.float32)
    for i in range(n):
        s = real_np["sim_data_projected"][0][i][am[i]][:P]
        sim[i, : len(s)] = s + rng.normal(0, 1e-3, s.shape)
    batch = {
        SIM_KEY: sim, "attention_mask": np.ones((n, 1, P), bool),
        "vlos_mask": np.ones((n, 1, P), bool), "j": np.arange(n)[:, None],
    }
    out = _build("stream_summary_grid", params)(batch)["sim_summary"]
    assert np.asarray(out).shape == (n, params["summary_track_bins"], 9)

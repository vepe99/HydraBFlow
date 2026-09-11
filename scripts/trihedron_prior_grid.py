#!/usr/bin/env python
"""400 streams across the halo prior via the Frenet-Serret remap, as a phi1-vs-phi2 panel grid.

Where ``compare_trihedron_vs_spray.py`` sweeps ONE global over a handful of values and runs all
three forward models at each, this draws MANY potentials from the halo prior and forward-models them
with the trihedron remap alone -- the only arm that is milliseconds per potential. One 20x20 grid
per stream, one panel per draw, real Gaia members in grey behind every panel.

That is affordable only because the expensive part is already done: the fiducial 1e6-particle
restricted-N-body runs are cached by ``compare_trihedron_vs_spray.py`` under
``<outdir>/<stream>/cache/<name>_rnbody_fid_P<N>_s<seed>.npz``. One template is built per stream from
those, and each of the N draws is then a single progenitor-orbit integration plus an einsum. This
script REFUSES to run the N-body itself (``--allow-rnbody`` overrides), so a missing cache is an
error naming the command that would create it rather than a silent 8-15 min simulation per stream.

Drawn per row, uniform and SHARED across the three streams (one Galaxy per panel index, as in a real
dataset row), over the training-prior ranges of ``conf/simulator/stream_agama.yaml`` and
``stream_agama_ibata_onedisk_beta3.yaml``:

    rho   U[1.5e6, 5.0e7] Msun/kpc^3      a     U[1, 30] kpc       gamma U[-2, 1.5]
    beta  U[2, 4]                         q     U[0.5, 1.5]

Everything else -- disks, bulge, gas, progenitors -- stays at the fiducial config's values. The
solar frame is potential-dependent (``_solar_frame`` uses ``v_circ(R0) + V_Sun``), so it is
recomputed per draw, as is the progenitor's Galactocentric state from its FIXED observed ICRS
coordinates.

ACCURACY CAVEAT, carried into every figure. The remap is exact at the fiducial and degrades away
from it, asymmetrically and stream-specifically; on the MASS axis (``rho``) the 2026-08-28 sweep
measured it to be WORSE than particle spray on all three streams, because ``t_hat`` is frozen and a
mass change is mostly an orbital-period rescaling, i.e. exactly the along-track redistribution the
remap cannot represent. The usable window beating spray was q in [1.0,1.4] (M68), [0.85,1.2] (Pal5),
[1.0,1.05] (NGC3201) and NO mass range on any stream. A full-prior sample therefore contains many
panels the remap gets wrong, and there is no ground truth in the figure. These are a look at what
the remap produces across the prior, not a validated forward model.

Usage
-----
    uv run python scripts/trihedron_prior_grid.py --n-draws 4 --n-remap 20000 --outdir /tmp/tpg
    uv run python scripts/trihedron_prior_grid.py --n-draws 400 --grid-cols 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Must precede the hydrabflow imports: simulator auto-discovery otherwise probes for a free GPU,
# which stalls for many minutes on a busy box (and this script is pure CPU/agama anyway).
os.environ.setdefault("HYDRABFLOW_NUM_GPUS", "0")
os.environ.setdefault("HYDRABFLOW_SIM_QUIET", "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from trihedron import Template, build_template, remap  # noqa: E402
from ppc_summary_statistics import NAMES, augment_sim  # noqa: E402
from plot_fixed_potential_samples import (  # noqa: E402
    phi2_grid_figure,
    project_sample,
    robust_lim,
    stream_frame,
)
from compare_trihedron_vs_spray import (  # noqa: E402
    edge_centre_ratio,
    fiducial_row,
    halo_m200,
    load_or_run,
    load_simulator,
    observed_sample,
    progenitor_state,
    project_xv,
    run_rnbody,
)

from hydrabflow.simulators.stream_agama import _agama, _host_potential  # noqa: E402

# Halo prior: (key, low, high). Order fixes the column order of prior_grid_draws.npz.
HALO_PRIOR = [
    ("rho_TwoPowerTriaxial_halo", 1.5e6, 5.0e7),
    ("a_TwoPowerTriaxial_halo", 1.0, 30.0),
    ("gamma_TwoPowerTriaxial_halo", -2.0, 1.5),
    ("beta_TwoPowerTriaxial_halo", 2.0, 4.0),
    ("q_TwoPowerTriaxial_halo", 0.5, 1.5),
]
# Short tags for the per-panel corner annotation.
SHORT = {"rho_TwoPowerTriaxial_halo": "rho", "a_TwoPowerTriaxial_halo": "a",
         "gamma_TwoPowerTriaxial_halo": "g", "beta_TwoPowerTriaxial_halo": "b",
         "q_TwoPowerTriaxial_halo": "q"}

CAVEAT = ("remap is EXACT only at the fiducial and degrades away from it "
          "(worst along rho: there it loses to particle spray) -- no ground truth in these panels")


def subsample_template(tpl: Template, n: int, seed: int) -> Template:
    """Keep ``n`` random stars of a template, as a new Template sharing the same knot grid.

    The observation model keeps only the observed member count (129-297 stars) per realization, so
    remapping and projecting the full 1e6 particles for every (draw, stream) pair buys nothing and
    costs ~20 min plus several GB. The indices are drawn once and reused for every draw, so panels
    stay comparable star for star.
    """
    if n <= 0 or n >= tpl.n_stars:
        return tpl
    idx = np.random.default_rng(seed).choice(tpl.n_stars, size=n, replace=False)
    return Template(t_hat=tpl.t_hat[idx], c=tpl.c[idx], w=tpl.w[idx], t_knots=tpl.t_knots,
                    velocity=tpl.velocity, diagnostics=dict(tpl.diagnostics))


def draw_halo_rows(n: int, seed: int) -> list[dict]:
    """``n`` uniform draws of the halo prior, shared by all three streams."""
    rng = np.random.default_rng(seed)
    return [{k: float(v) for k, v in zip([p[0] for p in HALO_PRIOR], row)}
            for row in np.column_stack([rng.uniform(lo, hi, n) for _, lo, hi in HALO_PRIOR])]


def panel_label(i: int, draw: dict, n_stars: int | None) -> str:
    """Corner annotation: the draw index and the halo it was made in.

    Two significant figures on three short lines — a single line of five values overruns a 2.4 inch
    panel at the 5 pt the grid uses and collides with its neighbour.
    """
    g = draw["rho_TwoPowerTriaxial_halo"]
    return (f"#{i}" + (f" n={n_stars}" if n_stars is not None else "")
            + f"\nrho={g:.1e} a={draw['a_TwoPowerTriaxial_halo']:.1f}"
            + f"\ng={draw['gamma_TwoPowerTriaxial_halo']:.2f}"
            + f" b={draw['beta_TwoPowerTriaxial_halo']:.2f}"
            + f" q={draw['q_TwoPowerTriaxial_halo']:.2f}")


def zoom_lim(lim, factor: float):
    """Expand a range about its centre by ``factor``."""
    c, half = 0.5 * (lim[0] + lim[1]), 0.5 * (lim[1] - lim[0])
    return (c - factor * half, c + factor * half)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulator", default="stream_agama_rnbody_cautun_fixed",
                    help="config supplying the fiducial potential and the progenitors")
    ap.add_argument("--streams", nargs="+", default=["Pal5", "NGC3201", "M68"])
    ap.add_argument("--n-draws", type=int, default=400)
    ap.add_argument("--include-fiducial", action="store_true",
                    help="prepend the untouched fiducial halo as draw #0 and assert the remap "
                         "reproduces the cached N-body there (the identity self-test)")
    ap.add_argument("--n-particles", type=int, default=1000000,
                    help="particle count of the CACHED fiducial N-body run to build the template on")
    ap.add_argument("--n-remap", type=int, default=100000,
                    help="template stars actually remapped per draw (0 = all); the observation "
                         "model keeps only the observed member count anyway")
    ap.add_argument("--seed", type=int, default=2026,
                    help="also selects the cached N-body run (its filename carries the seed)")
    ap.add_argument("--draw-seed", type=int, default=None,
                    help="seed for the halo draws; default = --seed")
    ap.add_argument("--T-myr", type=float, default=200.0,
                    help="half-width of the orbit window the trihedron is built on")
    ap.add_argument("--n-knots", type=int, default=2001)
    ap.add_argument("--velocity", choices=["relative", "absolute"], default="relative")
    ap.add_argument("--agama-threads", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=20,
                    help="draws pushed through the observation model at once; the batch is "
                         "(chunk, n_streams, n_remap, 6), so all 400 at once would be ~7 GB")
    ap.add_argument("--no-noise", action="store_true",
                    help="plot the raw window-cut streams instead of the observation-model output")
    ap.add_argument("--aug", default="stream_global_ibata_grid",
                    help="augmentation preset supplying the observation model")
    ap.add_argument("--cache-root", default="data_local/trihedron_1e6",
                    help="where compare_trihedron_vs_spray.py left <stream>/cache/")
    ap.add_argument("--outdir", default="data_local/trihedron_1e6/prior_grid")
    ap.add_argument("--out", default="prior_grid", help="figure filename stem inside --outdir")
    ap.add_argument("--real",
                    default="assets/gaia/gaia_observed_streams_6Dwitherrors_cutNGC3201.npz")
    ap.add_argument("--grid-cols", type=int, default=20)
    ap.add_argument("--phi2-lim", nargs=2, type=float, default=None,
                    help="fix the phi2 range of every panel; default = the real members' robust "
                         "range expanded by --phi2-zoom")
    ap.add_argument("--phi2-zoom", type=float, default=6.0,
                    help="how much wider than the real members' phi2 range each panel is. Pooling "
                         "all draws instead (as the phi1 range does) is stretched by a handful of "
                         "wildly off-track halos until every stream is a flat line; the fraction "
                         "of stars falling outside is reported per draw as 'phi2_outside_frac'")
    ap.add_argument("--replot", action="store_true",
                    help="redraw the figures from a previous run's <out>_samples_<stream>.npz "
                         "instead of remapping again (seconds instead of ~35 min)")
    ap.add_argument("--panel-w", type=float, default=2.4)
    ap.add_argument("--panel-h", type=float, default=2.0)
    ap.add_argument("--dpi", type=int, default=0)
    ap.add_argument("--allow-rnbody", action="store_true",
                    help="permit running the fiducial N-body if its cache is missing (slow)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    agama = _agama()
    agama.setNumThreads(int(args.agama_threads))
    tu = agama.getUnits()["time"]
    time_unit_gyr = float(getattr(tu, "value", tu)) / 1e3
    myr_to_agama = 1e-3 / time_unit_gyr

    sim = load_simulator(args.simulator)
    pot_cfg = sim._pot_cfg
    opts = sim._rnbody_opts()
    streams = {name: sim.target_streams[name] for name in args.streams}

    # --- halo draws (shared across streams) -----------------------------------------------------
    draws = draw_halo_rows(int(args.n_draws), args.draw_seed if args.draw_seed is not None
                           else args.seed)
    if args.include_fiducial:
        row0 = fiducial_row(sim, args.streams[0])
        draws.insert(0, {k: float(row0[k]) for k, _, _ in HALO_PRIOR})
    n_draws = len(draws)
    print(f"{n_draws} halo draws x {len(streams)} streams, "
          f"{args.n_remap or args.n_particles} particles remapped per draw")

    # --- real members define the great-circle frames ---------------------------------------------
    d = np.load(args.real)
    rsim = d["sim_data_projected"]
    rsim = rsim[0] if rsim.ndim == 4 else rsim
    ram = d["attention_mask"]
    ram = ram[:, 0, :] if ram.ndim == 3 else ram
    jreal = np.asarray(d["j"]).reshape(-1).astype(int)

    # --- one template per stream, from the cached fiducial N-body --------------------------------
    templates, frames, per_stream = {}, {}, {}
    for name, j in streams.items():
        rrow = int(np.where(jreal == j)[0][0])
        mem = ram[rrow].astype(bool)
        R, real = stream_frame(rsim[rrow][mem])
        per_stream[name] = dict(j=j, R=R, real=real, samples=[], template={})
        if args.replot:
            continue
        cache = Path(args.cache_root) / name / "cache"
        path = cache / f"{name}_rnbody_fid_P{args.n_particles}_s{args.seed}.npz"
        if not path.exists() and not args.allow_rnbody:
            raise SystemExit(
                f"missing fiducial N-body cache {path}\n"
                f"create it with:\n"
                f"  uv run python scripts/compare_trihedron_vs_spray.py --streams {name} "
                f"--n-particles {args.n_particles} --seed {args.seed} --check-fiducial "
                f"--outdir {args.cache_root}\n"
                f"(or pass --allow-rnbody to run it here, ~8-15 min per stream at 1e6 particles)")
        cache.mkdir(parents=True, exist_ok=True)

        row_fid = fiducial_row(sim, name)
        pot_fid = _host_potential(agama, row_fid, pot_cfg)
        posvel_fid, _ = progenitor_state(agama, pot_fid, row_fid)
        xv_fid, _ = load_or_run(
            path, False,
            lambda: run_rnbody(agama, pot_fid, posvel_fid, row_fid, opts,
                               args.n_particles, args.seed, time_unit_gyr),
            label=f"{name} rnbody fiducial (template)",
        )
        tpl = build_template(agama, pot_fid, posvel_fid, np.asarray(xv_fid, dtype=float),
                             args.T_myr * myr_to_agama, args.n_knots, velocity=args.velocity)
        print(f"  {name} template: "
              + ", ".join(f"{k}={v:.4g}" for k, v in tpl.diagnostics.items()))
        if tpl.diagnostics["boundary_frac"] > 1e-3:
            print(f"  WARNING: {tpl.diagnostics['boundary_frac']:.1%} of {name} stars pin to the "
                  f"window boundary -- raise --T-myr")

        if args.include_fiducial:
            err = float(np.nanmax(np.linalg.norm(
                remap(agama, pot_fid, posvel_fid, tpl)[:, 0:3] - np.asarray(xv_fid)[:, 0:3],
                axis=1)))
            print(f"  {name} fiducial round-trip: max |dx| = {err:.3e} kpc")
            assert err < 1e-9, f"remap is not exact at the fiducial for {name}: {err:.3e} kpc"

        templates[name] = subsample_template(tpl, int(args.n_remap), args.seed)
        frames[name] = (row_fid, pot_cfg)
        per_stream[name]["template"] = tpl.diagnostics

    names = list(streams)

    # --- remap every draw, in chunks through the observation model --------------------------------
    if args.replot:
        for name in names:
            per_stream[name]["samples"] = load_samples(outdir / f"{args.out}_samples_{name}.npz")
            print(f"  [replot] {name}: {len(per_stream[name]['samples'])} cached draws")
        n_remapped = None  # not recomputed on a replot; the previous run's json has it
    else:
        n_remapped = templates[names[0]].n_stars
        for lo in range(0, n_draws, int(args.chunk)):
            hi = min(lo + int(args.chunk), n_draws)
            proj = np.full((hi - lo, len(names), n_remapped, 6), np.nan)
            for di in range(lo, hi):
                for si, name in enumerate(names):
                    row_fid, _ = frames[name]
                    row = dict(row_fid)
                    row.update(draws[di])
                    pot = _host_potential(agama, row, pot_cfg)
                    posvel, frame = progenitor_state(agama, pot, row)
                    xv = remap(agama, pot, posvel, templates[name])
                    proj[di - lo, si] = project_xv(np.asarray(xv, dtype=float), frame)

            if args.no_noise:
                # match the observation model's first step so channel 2 is a parallax on both sides
                for di in range(hi - lo):
                    for si, name in enumerate(names):
                        s, _ = observed_sample(proj[di, si], per_stream[name]["j"],
                                               argparse.Namespace(noise=False), args.seed)
                        per_stream[name]["samples"].append(s)
            else:
                jj = np.tile(np.array([[per_stream[n]["j"] for n in names]]), (hi - lo, 1))
                sd, attn, _ = augment_sim(proj, jj, aug_preset=args.aug, simulator=args.simulator,
                                          seed=args.seed + lo)
                for di in range(hi - lo):
                    for si, name in enumerate(names):
                        per_stream[name]["samples"].append(
                            sd[di, si][attn[di, si].astype(bool)])
            print(f"  draws {lo}-{hi - 1} done", flush=True)

        # The observed samples are the only expensive product; caching them makes --replot a
        # seconds-long redraw instead of a ~35 minute re-remap.
        for name in names:
            save_samples(outdir / f"{args.out}_samples_{name}.npz", per_stream[name]["samples"])

    # --- figures ----------------------------------------------------------------------------------
    kind = ("raw (noiseless, window cut only)" if args.no_noise
            else "through the training observation model")
    metrics = {}
    for name in names:
        ps = per_stream[name]
        real = ps["real"]
        samples = [project_sample(ps["R"], s) if s is not None and len(s) else None
                   for s in ps["samples"]]
        ok = [s for s in samples if s is not None]
        # phi1 is pooled over the draws (the arms genuinely differ in length and that is the
        # point); phi2 is anchored on the REAL members, because pooling it lets a handful of
        # wildly off-track halos stretch the axis until every panel is a flat line.
        lims = dict(phi1=robust_lim(real["phi1"], *[s["phi1"] for s in ok]),
                    phi2=(tuple(args.phi2_lim) if args.phi2_lim
                          else zoom_lim(robust_lim(real["phi2"]), args.phi2_zoom)))
        metrics[name] = [panel_metrics(s, lims["phi2"]) for s in samples]
        labels = [panel_label(i, draws[i], None if s is None else len(s["phi1"]))
                  for i, s in enumerate(samples)]
        phi2_grid_figure(
            real, samples, lims, str(outdir / f"{args.out}_{name}_phi2.png"),
            ncol=args.grid_cols, panel_w=args.panel_w, panel_h=args.panel_h, dpi=args.dpi,
            labels=labels,
            suptitle=f"{name} — {n_draws} draws of the halo prior (rho, a, gamma, beta, q), "
                     f"trihedron remap from the {args.simulator} fiducial, {kind}\n"
                     f"grey = real Gaia members;  CAVEAT: {CAVEAT}",
        )

    # --- draws + metrics --------------------------------------------------------------------------
    m200 = [halo_m200(agama, {**fiducial_row(sim, names[0]), **dr}, pot_cfg) for dr in draws]
    np.savez(outdir / f"{args.out}_draws.npz",
             **{k: np.array([dr[k] for dr in draws]) for k, _, _ in HALO_PRIOR},
             **{k: np.array([m[k] for m in m200]) for k in ("m200_msun", "r200_kpc", "c200")})
    print(f"wrote {outdir / f'{args.out}_draws.npz'}")

    out = dict(
        simulator=args.simulator, n_draws=n_draws, n_remapped=n_remapped,
        n_particles_template=args.n_particles, seed=args.seed, noise=not args.no_noise,
        prior={k: [lo, hi] for k, lo, hi in HALO_PRIOR}, caveat=CAVEAT,
        draws=[{**dr, **m} for dr, m in zip(draws, m200)],
        streams={name: dict(j=per_stream[name]["j"], template=per_stream[name]["template"],
                            per_draw=metrics[name]) for name in names},
    )
    path = outdir / f"{args.out}_metrics.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")


def panel_metrics(p, phi2_lim) -> dict:
    """Distributional summary of one panel, from its already-projected stars."""
    m = {"n_in_window": int(0 if p is None else len(p["phi1"]))}
    if p is None or len(p["phi1"]) < 10:
        return m
    m["phi1_extent_deg"] = float(np.ptp(p["phi1"]))
    # edge/centre density, NOT the extent, is the meaningful arm-length statistic for
    # NGC3201/M68 -- their in-window extent is saturated by the observation window.
    m["edge_centre_ratio"] = edge_centre_ratio(p["phi1"])
    m["phi2_median_deg"] = float(np.nanmedian(p["phi2"]))
    # How much of this draw the panel's real-member-anchored phi2 range cuts off, so a stream
    # that has left the observed track entirely is recorded rather than silently blank.
    m["phi2_outside_frac"] = float(np.mean((p["phi2"] < phi2_lim[0]) | (p["phi2"] > phi2_lim[1])))
    return m


def save_samples(path: Path, samples: list) -> None:
    """Cache one stream's observed samples, NaN-padded to the longest draw, plus their counts."""
    n = max((0 if s is None else len(s)) for s in samples) if samples else 0
    arr = np.full((len(samples), n, 6), np.nan, dtype=np.float32)
    cnt = np.zeros(len(samples), dtype=np.int32)
    for i, s in enumerate(samples):
        if s is not None and len(s):
            arr[i, : len(s)] = s
            cnt[i] = len(s)
    np.savez_compressed(path, samples=arr, counts=cnt)
    print(f"wrote {path}")


def load_samples(path: Path) -> list:
    if not path.exists():
        raise SystemExit(f"--replot needs {path}; run once without it first")
    d = np.load(path)
    arr, cnt = d["samples"], d["counts"]
    return [None if c == 0 else np.asarray(arr[i, :c], dtype=float) for i, c in enumerate(cnt)]


if __name__ == "__main__":
    main()

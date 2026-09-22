#!/usr/bin/env python
"""Build a real-observation npz from the Palau & Miralda-Escude (2023) member tables.

Input (all shipped in ``assets/gaia/``, transcribed from the paper's appendix tables):

  ``Pal5.txt``        Table B1 — 126 GDR2 stars compatible with the Palomar 5 stream model
  ``Pal5_radvel.txt`` Table B2 — the 14 of those with a radial velocity from Ibata et al. (2017),
                      joined onto Table B1 by ``source_id``
  ``NGC3201.txt``     Table C1 — 170 GDR2 stars for the NGC 3201 (Gjoll) stream
  ``M68.txt``         Table E1 — 115 GDR2 stars for the M68 (Fjorm) stream; the table's own header
                      states GDR2 gives NO radial velocity for any of them

``--streamfinder-rv`` (ON by default) additionally cross-matches every member against the
STREAMFINDER catalogue of Ibata et al. (2024) (``apjad382dt1_mrt.txt``, 24540 stars, 2178 with a
``VHel``) and adopts its heliocentric radial velocity where the Palau23 tables have none — matching
first on ``source_id``, then on sky position within ``--match-arcsec`` (the Palau tables are GDR2,
the atlas DR3, so a few ids differ), and requiring the atlas row to carry that stream's own label
(Pal-5 61, Gjoll 34, Fjorm 47). ``e_VHel = 300`` is the atlas' NULL flag and is treated as "no
velocity". This is what gives NGC 3201 and M68 any velocities at all: Palau23 lists none for them.

Output: the same layout every other real-observation file in ``assets/gaia/`` uses, so it is a
drop-in for ``--real`` / ``data.real_data_path`` —

  ``sim_data_projected`` (1, 3, P, 6)   alpha, delta, parallax, mu_alpha*, mu_delta, v_los
  ``attention_mask``     (3, 1, P)      1 for a real member, 0 for padding (padding is all-zero)
  ``magnitudes``         (3, P)         Gaia G
  ``vlos_error``         (3, P)         1-sigma v_los uncertainty
  ``vlos_mask``          (3, P)         1 where v_los is MEASURED
  ``j``                  (1, 3, 1)      stream index: Pal5 0, NGC3201 1, M68 2

Two things are not in the tables and are therefore choices, both reported in the sidecar JSON:

* **v_los for the stars without one.** The existing files fill them with the per-stream MEAN of the
  measured values and set their error to the sample STD (what ``mask_vlos`` does), and this script
  does the same for Pal 5. NGC 3201 and M68 have no measured velocity at all under Palau23, so the
  fill falls back to the cluster's systemic v_los (``SYSTEMIC``, the ``vr`` identity priors of
  ``conf/simulator/stream_agama.yaml``) with ``--vlos-fill-sigma`` as its error. ``vlos_mask`` is 0
  for every one of them either way, so a masked / zero-imputing model never sees these numbers.
* **The observation window.** The forward model only ever produces stars inside each stream's
  RA/Dec window, so members outside it can never be matched by a simulation. ``--window cut``
  (the default) drops them; ``--window keep`` keeps the tables verbatim.  M68's box is narrower here than the
  pipeline's (``190 < alpha < 260``, ``dec >= -8``) — see ``WINDOW``; ``--window-box`` changes any
  of them, e.g. ``--window-box "M68:189.74,280,-47.32,70"`` restores the pipeline box.

``--astrometry dr3`` replaces the tables' GDR2 astrometry with the star's **Gaia DR3** values,
resolved through the official ``gaiadr3.dr2_neighbourhood`` crossmatch over the ESA Gaia TAP service
(the DR2 and DR3 ``source_id`` of a star are NOT guaranteed equal). The response is cached as a CSV
beside the tables, so it is fetched once and every later build is offline. This mode also carries
the catalogue's own PER-STAR uncertainties into an extra ``obs_error`` (3, P, 6) array — the real
reason to do it, since the GDR2 tables list no errors at all and everything else has to fall back on
the DR3 median-uncertainty-vs-G table. Gaia DR3's own RVS radial velocities are picked up too.
Note this keeps PALAU'S membership but not the data it was selected on: their cuts were applied to
GDR2 astrometry. DR2 and DR3 values are never mixed in one file — ``--astrometry dr2`` (the default)
is pure GDR2, ``dr3`` drops any star the crossmatch cannot resolve.

Standalone: numpy only (the TAP query uses stdlib ``urllib``).

Usage:
  .venv/bin/python scripts/build_palau23_members.py --out assets/gaia/gaia_observed_streams_palau23.npz
"""

from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np

ASSETS = "assets/gaia"
# stream -> (index j, member table, radial-velocity table or None)
TABLES = {
    "Pal5": (0, "Pal5.txt", "Pal5_radvel.txt"),
    "NGC3201": (1, "NGC3201.txt", None),
    "M68": (2, "M68.txt", None),
}
# Observation windows (RA_lo, RA_hi, Dec_lo, Dec_hi). Pal5 and NGC3201 are identical to
# ppc_summary_statistics.WINDOW and the `observational_window` augmentation; M68 is DELIBERATELY
# narrower here (user request 2026-09-22): 190 < alpha < 260 and dec >= -8, which removes the
# southern group around dec ~ -45 and the stars beyond alpha 260. The pipeline-wide M68 window
# (189.73694638139526, 280.0, -47.32109546202406, 70.0) is unchanged — this only selects which
# Palau23 members enter the file. Override any of them with --window-box.
WINDOW = {
    0: (220.98008280733634, 250.0, -13.0, 10.0),
    1: (42.29847920913508, 140.0, -47.79545565626234, 20.812029301121846),
    2: (190.0, 260.0, -8.0, 70.0),
}
# ESA Gaia TAP, for --astrometry dr3.
GAIA_TAP = "https://gea.esac.esa.int/tap-server/tap/sync"
GAIA_CACHE = "palau23_gaiadr3.csv"
DR3_QUERY = """SELECT n.dr2_source_id, n.dr3_source_id, n.angular_distance, n.magnitude_difference,
 g.ra, g.dec, g.ra_error, g.dec_error, g.parallax, g.parallax_error,
 g.pmra, g.pmra_error, g.pmdec, g.pmdec_error, g.phot_g_mean_mag,
 g.radial_velocity, g.radial_velocity_error
FROM gaiadr3.dr2_neighbourhood AS n
JOIN gaiadr3.gaia_source AS g ON g.source_id = n.dr3_source_id
WHERE n.dr2_source_id IN ({ids})"""

# Ibata et al. (2024) STREAMFINDER atlas: fixed-width table + this project's stream labels.
STREAMFINDER_TABLE = "apjad382dt1_mrt.txt"
SF_LABEL = {0: 61, 1: 34, 2: 47}          # Pal-5, Gjoll, Fjorm
SF_NULL_EVHEL = 300.0                     # the table's NULL flag for e_VHel

# Cluster systemic line-of-sight velocities [km/s] (the `vr` identity priors of
# conf/simulator/stream_agama.yaml), used only to fill streams with no measured velocity at all.
SYSTEMIC = {0: -58.60, 1: 495.38, 2: -93.11}
# Table B1/C1/E1 column order after `N` and `source_id`.
COLS = ["parallax", "dec", "ra", "pmdec", "pmra", "bp_rp", "g", "chisel"]


def _num(tok: str) -> float:
    """Float from a table token: unicode minus/en-dash, thin spaces and split exponents."""
    t = (tok.replace("−", "-").replace("–", "-").replace("—", "-")
            .replace(" ", "").replace(" ", "").replace(" ", ""))
    return float(t)


def read_members(path):
    """{column: (N,) array} from a Palau23 member table (rows start with an integer index)."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            f = [c.strip() for c in ln.rstrip("\n").split("\t")]
            if len(f) < len(COLS) + 2 or not re.fullmatch(r"\d+", f[0]):
                continue
            rows.append(([int(f[0]), int(f[1])], [_num(x) for x in f[2:2 + len(COLS)]]))
    if not rows:
        raise SystemExit(f"no data rows parsed from {path}")
    # source_id is a 19-digit Gaia id: it must stay an exact int64, never a float (> 2**53).
    ids = np.array([r[0] for r in rows], dtype=np.int64)
    a = np.array([r[1] for r in rows], dtype=float)
    out = {"n": ids[:, 0], "source_id": ids[:, 1]}
    out.update({c: a[:, k] for k, c in enumerate(COLS)})
    return out


def read_radvel(path):
    """{source_id: (v_los, sigma)} from a Palau23 radial-velocity table."""
    out = {}
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            f = [c.strip() for c in ln.rstrip("\n").split("\t") if c.strip()]
            if len(f) < 4 or not re.fullmatch(r"\d+", f[0]):
                continue
            out[int(f[1])] = (_num(f[2]), _num(f[3]))
    return out


def read_streamfinder(path):
    """Columns of the STREAMFINDER MRT table, as arrays (fixed-width, byte ranges from its header)."""
    sid, ra, dec, plx, pmra, pmdec, g, vh, ev, ref, lab = ([] for _ in range(11))
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            if len(ln) < 97 or not ln[:19].strip().isdigit():
                continue
            sid.append(int(ln[0:19]))
            ra.append(float(ln[20:30]))
            dec.append(float(ln[31:41]))
            plx.append(float(ln[42:48]))
            pmra.append(float(ln[49:56]))
            pmdec.append(float(ln[57:64]))
            g.append(float(ln[65:70]))
            vh.append(float(ln[77:84]))
            ev.append(float(ln[85:91]))
            ref.append(int(ln[92:94]))
            lab.append(int(ln[95:97]))
    return dict(source_id=np.array(sid, dtype=np.int64), ra=np.array(ra), dec=np.array(dec),
                parallax=np.array(plx), pmra=np.array(pmra), pmdec=np.array(pmdec),
                g=np.array(g), vlos=np.array(vh), e_vlos=np.array(ev), ref=np.array(ref),
                label=np.array(lab))


def match_streamfinder(m, sf, label, arcsec):
    """Index into ``sf`` for each member of ``m`` (-1 = unmatched): source_id first, then sky."""
    idx = {int(s): i for i, s in enumerate(sf["source_id"])}
    out = np.full(len(m["ra"]), -1)
    n_id = 0
    for k in range(len(m["ra"])):
        i = idx.get(int(m["source_id"][k]), -1)
        if i >= 0:
            n_id += 1
        else:                                   # GDR2 vs DR3 ids differ for a few stars
            d = np.hypot((sf["ra"] - m["ra"][k]) * np.cos(np.radians(m["dec"][k])),
                         sf["dec"] - m["dec"][k]) * 3600.0
            i = int(d.argmin())
            i = i if d[i] < arcsec else -1
        if i >= 0 and sf["label"][i] != label:   # matched a star of a DIFFERENT stream: reject
            i = -1
        out[k] = i
    return out, n_id


def fetch_dr3(dr2_ids, cache, chunk=150, timeout=300):
    """Rows of :data:`DR3_QUERY` for ``dr2_ids``, from ``cache`` if it exists else from the TAP."""
    import csv  # noqa: PLC0415  (only this path needs it)
    import urllib.parse  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415

    if os.path.exists(cache):
        with open(cache, newline="") as fh:
            rows = list(csv.DictReader(fh))
        if {int(r["dr2_source_id"]) for r in rows} >= set(dr2_ids):
            print(f"[dr3] {len(rows)} cached rows from {cache}")
            return rows
        print(f"[dr3] {cache} does not cover every id — re-querying")
    rows, header = [], None
    ids = sorted(set(int(i) for i in dr2_ids))
    for a in range(0, len(ids), chunk):
        q = DR3_QUERY.format(ids=", ".join(str(i) for i in ids[a:a + chunk]))
        body = urllib.parse.urlencode({"REQUEST": "doQuery", "LANG": "ADQL",
                                       "FORMAT": "csv", "QUERY": q}).encode()
        print(f"[dr3] TAP query {a // chunk + 1}/{-(-len(ids) // chunk)} "
              f"({len(ids[a:a + chunk])} ids) ...", flush=True)
        with urllib.request.urlopen(GAIA_TAP, data=body, timeout=timeout) as r:
            text = r.read().decode()
        lines = text.splitlines()
        header = header or lines[0]
        rows.extend(lines[1:])
    with open(cache, "w") as fh:
        fh.write("\n".join([header, *rows]) + "\n")
    print(f"[dr3] wrote {cache} ({len(rows)} rows)")
    with open(cache, newline="") as fh:
        return list(csv.DictReader(fh))


def resolve_dr3(rows):
    """{dr2_source_id: row} keeping, for a DR2 id with several DR3 counterparts, the nearest one
    (ties broken by the smaller |magnitude difference|), plus the ids that were ambiguous."""
    best, ambiguous = {}, set()

    def score(r):
        return (float(r["angular_distance"] or 0.0), abs(float(r["magnitude_difference"] or 0.0)))

    for r in rows:
        k = int(r["dr2_source_id"])
        if k in best:
            ambiguous.add(k)
            if score(r) >= score(best[k]):
                continue
        best[k] = r
    return best, ambiguous


def _f(row, key):
    v = row.get(key, "")
    return float(v) if v not in ("", None) else np.nan


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets", default=ASSETS, help="directory holding the Palau23 tables")
    ap.add_argument("--out", default=os.path.join(ASSETS, "gaia_observed_streams_palau23.npz"))
    ap.add_argument("--window", choices=("cut", "keep"), default="cut",
                    help="drop (default) or keep members outside the stream's observation window")
    ap.add_argument("--window-box", action="append", default=[],
                    metavar="STREAM:RA_LO,RA_HI,DEC_LO,DEC_HI",
                    help="replace one stream's window, e.g. 'M68:190,260,-8,70'; repeatable")
    ap.add_argument("--n-pad", type=int, default=1000,
                    help="padded particle axis length (1000 matches the other real npz files)")
    ap.add_argument("--astrometry", choices=("dr2", "dr3"), default="dr2",
                    help="'dr2' (default) keeps the tables' GDR2 values; 'dr3' replaces them with "
                         "Gaia DR3 via the dr2_neighbourhood crossmatch, with per-star errors")
    ap.add_argument("--gaia-cache", default=os.path.join(ASSETS, GAIA_CACHE),
                    help="CSV cache of the DR3 TAP response")
    ap.add_argument("--no-streamfinder-rv", dest="streamfinder_rv", action="store_false",
                    default=True,
                    help="do NOT take radial velocities from the Ibata+2024 STREAMFINDER atlas")
    ap.add_argument("--streamfinder-table",
                    default=os.path.join(ASSETS, STREAMFINDER_TABLE))
    ap.add_argument("--match-arcsec", type=float, default=2.0,
                    help="positional tolerance for members whose source_id is not in the atlas")
    ap.add_argument("--vlos-fill-sigma", type=float, default=20.0,
                    help="error assigned to filled v_los when a stream has no measured velocity")
    args = ap.parse_args()

    window = dict(WINDOW)
    for spec in args.window_box:
        name, _, box = spec.partition(":")
        if name.strip() not in TABLES:
            raise SystemExit(f"--window-box: unknown stream {name!r}")
        vals = [float(v) for v in box.split(",")]
        if len(vals) != 4:
            raise SystemExit(f"--window-box {spec!r}: expected RA_LO,RA_HI,DEC_LO,DEC_HI")
        window[TABLES[name.strip()][0]] = tuple(vals)

    n_pad = args.n_pad
    sim = np.zeros((3, n_pad, 6))
    att = np.zeros((3, n_pad))
    mag = np.zeros((3, n_pad))
    verr = np.zeros((3, n_pad))
    vmask = np.zeros((3, n_pad))
    obs_err = np.zeros((3, n_pad, 6))        # per-star 1-sigma, columns as sim_data_projected
    report = {"source_tables": {}, "window": args.window, "streams": {}}

    dr3, dr3_amb = {}, set()
    if args.astrometry == "dr3":
        all_ids = []
        for _, member_file, _ in TABLES.values():
            all_ids += list(read_members(os.path.join(args.assets, member_file))["source_id"])
        dr3, dr3_amb = resolve_dr3(fetch_dr3(all_ids, args.gaia_cache))
        print(f"[dr3] resolved {len(dr3)}/{len(set(all_ids))} DR2 ids "
              f"({len(dr3_amb)} had several DR3 counterparts)")

    for name, (j, member_file, rv_file) in TABLES.items():
        m = read_members(os.path.join(args.assets, member_file))
        rv = read_radvel(os.path.join(args.assets, rv_file)) if rv_file else {}
        report["source_tables"][name] = {"members": member_file, "radvel": rv_file,
                                         "n_rows": int(len(m["n"])), "n_radvel_rows": len(rv)}

        lo_ra, hi_ra, lo_dec, hi_dec = window[j]
        inside = ((m["ra"] >= lo_ra) & (m["ra"] <= hi_ra)
                  & (m["dec"] >= lo_dec) & (m["dec"] <= hi_dec))
        keep = inside if args.window == "cut" else np.ones_like(inside)
        if keep.sum() > n_pad:
            raise SystemExit(f"{name}: {keep.sum()} members exceed --n-pad {n_pad}")

        err = np.full((len(m["ra"]), 6), np.nan)
        n_dr3 = n_amb = n_lost = n_partial = 0
        if args.astrometry == "dr3":
            hit = np.array([int(sid) in dr3 for sid in m["source_id"]])
            n_dr3, n_amb = int(hit.sum()), int(sum(int(sid) in dr3_amb for sid in m["source_id"]))
            n_lost = int((~hit).sum())
            for kk in np.flatnonzero(hit):
                r = dr3[int(m["source_id"][kk])]
                m["ra"][kk], m["dec"][kk] = _f(r, "ra"), _f(r, "dec")
                m["parallax"][kk] = _f(r, "parallax")
                m["pmra"][kk], m["pmdec"][kk] = _f(r, "pmra"), _f(r, "pmdec")
                m["g"][kk] = _f(r, "phot_g_mean_mag")
                err[kk] = [_f(r, "ra_error") / 3.6e6, _f(r, "dec_error") / 3.6e6,
                           _f(r, "parallax_error"), _f(r, "pmra_error"), _f(r, "pmdec_error"),
                           _f(r, "radial_velocity_error")]
            keep &= hit                      # never mix DR2 and DR3 values in one file
            # DR3 two-parameter solutions carry no parallax/proper motion: they would enter the
            # file as NaN, so drop them rather than leave a hole in the observables.
            full = np.isfinite(m["parallax"]) & np.isfinite(m["pmra"]) & np.isfinite(m["pmdec"])
            n_partial = int((keep & ~full).sum())
            keep &= full

        v = np.array([rv.get(int(s), (np.nan, np.nan)) for s in m["source_id"]])
        n_rvs = 0
        if args.astrometry == "dr3":         # Gaia DR3's own RVS velocities
            for kk in np.flatnonzero(keep & ~np.isfinite(v[:, 0])):
                r = dr3[int(m["source_id"][kk])]
                if np.isfinite(_f(r, "radial_velocity")):
                    v[kk] = [_f(r, "radial_velocity"), _f(r, "radial_velocity_error")]
                    n_rvs += 1
        n_sf_added = n_sf_match = n_sf_id = 0
        sf_refs = {}
        if args.streamfinder_rv:
            sf = read_streamfinder(args.streamfinder_table)
            mi, n_sf_id = match_streamfinder(m, sf, SF_LABEL[j], args.match_arcsec)
            n_sf_match = int((mi >= 0).sum())
            ok = (mi >= 0) & keep
            ok[ok] = sf["e_vlos"][mi[ok]] < SF_NULL_EVHEL
            take = ok & ~np.isfinite(v[:, 0])        # the paper's own value wins where both exist
            v[take] = np.stack([sf["vlos"][mi[take]], sf["e_vlos"][mi[take]]], axis=-1)
            n_sf_added = int(take.sum())
            for i in mi[take]:
                r = int(sf["ref"][i])
                sf_refs[r] = sf_refs.get(r, 0) + 1
        has_v = np.isfinite(v[:, 0]) & keep
        if has_v.any():                      # Pal 5: mean / std of the measured values, as mask_vlos
            fill, fill_sig = float(v[has_v, 0].mean()), float(v[has_v, 0].std())
        else:                                # NGC 3201, M68: no velocity in Palau23 at all
            fill, fill_sig = SYSTEMIC[j], args.vlos_fill_sigma
        vlos = np.where(np.isfinite(v[:, 0]), v[:, 0], fill)
        vsig = np.where(np.isfinite(v[:, 1]), v[:, 1], fill_sig)

        k = keep.nonzero()[0]
        n = len(k)
        sim[j, :n] = np.stack([m["ra"][k], m["dec"][k], m["parallax"][k],
                               m["pmra"][k], m["pmdec"][k], vlos[k]], axis=-1)
        att[j, :n] = 1.0
        mag[j, :n] = m["g"][k]
        verr[j, :n] = vsig[k]
        vmask[j, :n] = has_v[k].astype(float)
        e = err[k]
        e[:, 5] = vsig[k]                    # v_los error: the adopted one, whatever its source
        obs_err[j, :n] = np.nan_to_num(e)
        report["streams"][name] = {
            "j": j, "n_members": int(n), "window_box": list(window[j]), "n_outside_window": int((~inside).sum()),
            "n_vlos_measured": int(has_v[k].sum()),
            "n_vlos_from_paper": int(sum(int(sid) in rv for sid in m["source_id"][k])),
            "astrometry": args.astrometry,
            "n_dr3_resolved": n_dr3, "n_dr3_ambiguous": n_amb, "n_dr3_unresolved": n_lost,
            "n_dr3_two_parameter_dropped": n_partial,
            "n_vlos_from_gaia_rvs": n_rvs,
            "n_vlos_from_streamfinder": n_sf_added,
            "n_streamfinder_matched": n_sf_match, "n_streamfinder_id_matched": n_sf_id,
            "streamfinder_vlos_sources": sf_refs,
            "vlos_fill": fill, "vlos_fill_sigma": fill_sig,
            "ra_range": [float(m["ra"][k].min()), float(m["ra"][k].max())],
            "dec_range": [float(m["dec"][k].min()), float(m["dec"][k].max())],
            "g_range": [float(m["g"][k].min()), float(m["g"][k].max())],
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out,
             j=np.arange(3).reshape(1, 3, 1),
             sim_data_projected=sim[None],
             attention_mask=att[:, None, :],
             magnitudes=mag, vlos_error=verr, vlos_mask=vmask, obs_error=obs_err[None])
    with open(os.path.splitext(args.out)[0] + ".json", "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

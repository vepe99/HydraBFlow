#!/usr/bin/env python
"""Collate ``compare_trihedron_vs_spray`` sweeps into one table + ``summary.json``.

Reads ``<outdir>/<stream>/<sweep>_metrics.json`` for every stream and sweep and answers the only
question the sweeps were run to answer: over what range of a parameter is the Frenet-Serret remap
an acceptable stand-in for the restricted N-body, and does it beat particle spray there.

    uv run python scripts/summarize_trihedron.py data_local/trihedron_1e6 --sweeps q rho
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def rows(res: dict) -> list[dict]:
    out = []
    for value, m in res["per_q"].items():
        e = m.get("trihedron_vs_rnbody", {})
        out.append({
            "value": float(value),
            "m200_msun": m.get("m200_msun"),
            "c200": m.get("c200"),
            "tri_track_deg": m.get("trihedron", {}).get("track_offset_deg"),
            "spray_track_deg": m.get("spray", {}).get("track_offset_deg"),
            "dx_perp_kpc": e.get("dx_perp_kpc_median"),
            "dphi2_deg": e.get("dphi2_deg_median"),
            "edge_centre": {a: m.get(a, {}).get("edge_centre_ratio") for a in
                            ("rnbody", "trihedron", "spray")},
            "n_in_window": {a: m.get(a, {}).get("n_in_window") for a in
                            ("rnbody", "trihedron", "spray")},
            "m_bound_final": m.get("m_bound_final"),
        })
    return sorted(out, key=lambda r: r["value"])


def fmt(x, spec=".3f"):
    return "  n/a" if x is None or x != x else format(x, spec)


STREAM_COLORS = {"Pal5": "tab:blue", "NGC3201": "tab:orange", "M68": "tab:green"}


def make_combined_figure(summary: dict, outdir: Path) -> None:
    """All streams on one axis per sweep -- the script itself can only draw the stream it ran.

    Each sweep runs one stream per process, so ``<outdir>/<sweep>_error_vs_param.png`` holds
    whichever process finished last. This redraws it from the collated metrics.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sweeps = [s for s in summary if summary[s]]
    fig, axes = plt.subplots(1, len(sweeps), figsize=(6.0 * len(sweeps), 4.6), squeeze=False)
    for ax, sweep in zip(axes[0], sweeps):
        for name, res in summary[sweep].items():
            c = STREAM_COLORS.get(name, "k")
            # x in M200 for the mass sweep, in the parameter itself for the shape sweep
            x = [(r["m200_msun"] if sweep == "rho" else r["value"]) for r in res["rows"]]
            # the remap is exactly 0 at the fiducial by construction; on a log axis that is -inf,
            # so drop it rather than let the line dive off the plot
            tri = [(r["tri_track_deg"] or None) or None for r in res["rows"]]
            tri = [None if (t is None or t == 0.0) else t for t in tri]
            ax.plot(x, tri, "o-", c=c, label=f"{name} trihedron")
            ax.plot(x, [r["spray_track_deg"] for r in res["rows"]], "s--", c=c, alpha=0.55,
                    label=f"{name} spray")
        ax.set_yscale("log")
        ax.set_ylabel("median |d phi2 track| vs restricted N-body [deg]")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, ncol=2)
        if sweep == "rho":
            ax.set_xscale("log")
            ax.set_xlabel("halo M200 [Msun]")
            ax.set_title("halo mass sweep (rho scaled at fixed a)")
        else:
            ax.set_xlabel("q_halo")
            ax.set_title("halo flattening sweep")
    fig.suptitle("Frenet-Serret remap vs particle spray, against the restricted N-body truth "
                 "(1e6 particles, Cautun+2020 fixed potential)", fontsize=11)
    fig.tight_layout()
    path = outdir / "combined_error_vs_param.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--sweeps", nargs="+", default=["q", "rho"])
    args = ap.parse_args()
    outdir = Path(args.outdir)

    summary, lines = {}, []
    for sweep in args.sweeps:
        for path in sorted(outdir.glob(f"*/{sweep}_metrics.json")):
            name = path.parent.name
            res = json.loads(path.read_text())
            r = rows(res)
            summary.setdefault(sweep, {})[name] = {
                "param": res.get("param"), "param_fid": res.get("param_fid"),
                "template": res.get("template"), "rows": r,
            }
            lines.append(f"\n### {name} -- {res.get('param')} "
                         f"(fiducial {res.get('param_fid'):.6g})")
            lines.append("| value | M200 [Msun] | trihedron |d phi2| | spray |d phi2| | ratio | "
                         "perp |dx| [kpc] | edge/centre rnbody / tri / spray |")
            lines.append("|---|---|---|---|---|---|---|")
            for x in r:
                t, s = x["tri_track_deg"], x["spray_track_deg"]
                ratio = (t / s) if (t and s and s == s and s > 0) else None
                ec = x["edge_centre"]
                lines.append(
                    f"| {x['value']:.6g} | {fmt(x['m200_msun'], '.3e')} | {fmt(t)} | {fmt(s)} | "
                    f"{fmt(ratio, '.2f')} | {fmt(x['dx_perp_kpc'], '.2f')} | "
                    f"{fmt(ec['rnbody'], '.2f')} / {fmt(ec['trihedron'], '.2f')} / "
                    f"{fmt(ec['spray'], '.2f')} |")

    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    make_combined_figure(summary, outdir)
    text = "\n".join(lines)
    print(text)
    print(f"\nwrote {outdir / 'summary.json'}")


if __name__ == "__main__":
    main()

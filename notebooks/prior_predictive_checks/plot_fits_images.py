"""
Plot the central field of every band in `assets/protoplan/extracted_fits/`.

    uv run python notebooks/prior_predictive_checks/plot_fits_images.py            # both disks
    uv run python notebooks/prior_predictive_checks/plot_fits_images.py oph163131  # just one

One PNG per disk in `assets/protoplan/extracted_fits/plots/`, four panels: the JWST 3.9 um slice
and the three ALMA bands, all cut to the same +/-1.5" the model field covers and all in MJy/sr, so
the panels are directly comparable.  Each ALMA panel draws its own beam (lower left) and every panel
marks the flux centroid -- the bands are cut about their own reference pixel, and those do not all
agree, so the centroid is how a mis-registration shows itself.

The stretch is asinh with the knee at 3x the cut's own MAD noise, i.e. the same reason the pipeline
compresses images with asinh: a linear scale is set by the inner rim and shows nothing else, and a
log one throws away the ~40-48% of post-noise pixels that are negative.
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import numpy as np                                                 # noqa: E402
from matplotlib.colors import AsinhNorm                            # noqa: E402
from matplotlib.patches import Ellipse                             # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import _fitsdisk as F                                              # noqa: E402
import _realdisk as rd                                             # noqa: E402

PLOT_DIR = F.FITS_DIR / "plots"
PANELS = ["jwst", "alma_0", "alma_1", "alma_2"]


def _noise(img):
    """MAD noise of the cut, from the pixels outside the central half -- robust to the disk."""
    n = img.shape[0]
    edge = np.concatenate([img[:n // 4].ravel(), img[-n // 4:].ravel()])
    edge = edge[np.isfinite(edge)]
    mad = np.median(np.abs(edge - np.median(edge))) if edge.size else 0.0
    return 1.4826 * mad or float(np.nanstd(img)) or 1.0


def plot_band(ax, b):
    """One band into one axis: asinh image, beam ellipse, centroid marker."""
    img, ra, dec = b["img"], b["rRA"], b["rDEC"]
    sigma = _noise(img)
    # imshow wants the axis limits, and rRA runs *down* the columns (East left), so the left edge
    # of the frame is the largest RA offset -- passing the extent in that order is what keeps the
    # sky orientation instead of silently mirroring it.
    extent = (ra[0], ra[-1], dec[0], dec[-1])
    # NaN reads as white on `inferno`, i.e. as the *brightest* value -- the opposite of "no data".
    # The JWST cubes are stored North-up while the IFU aperture was rotated (PA_APER ~ 222/237 deg),
    # so the corners of the array were never observed; grey them instead.
    cmap = matplotlib.colormaps["inferno"].with_extremes(bad="0.35")
    ax.imshow(img, origin="lower", extent=extent, cmap=cmap,
              norm=AsinhNorm(linear_width=3.0 * sigma,
                             vmin=float(np.nanpercentile(img, 1)), vmax=float(np.nanmax(img))))

    if "fwhm_maj" in b:
        # Sky PA is measured North through East; the ellipse's angle is CCW from +x (= -RA), which
        # is the same rotation sense on a frame with East to the left.
        ax.add_patch(Ellipse((ra[0] * 0.75, dec[0] * 0.75), b["fwhm_min"], b["fwhm_maj"],
                             angle=b["bpa_deg"], facecolor="none", edgecolor="cyan", lw=1.2))
    cx, cy = b["centroid"]
    ax.plot(cx, cy, "+", color="lime", ms=9, mew=1.4)
    ax.plot(0, 0, "x", color="white", ms=6, mew=1.0, alpha=0.7)

    beam = (f"\n{b['fwhm_maj']:.3f}x{b['fwhm_min']:.3f}\" @ {b['bpa_deg']:+.0f}deg"
            if "fwhm_maj" in b else "")
    ax.set_title(f"{b['label']}  (true {b['lam_um']:.0f}um){beam}\n"
                 f"centroid ({cx:+.2f}, {cy:+.2f})\"  peak {np.nanmax(img):.3g} MJy/sr",
                 fontsize=8)
    ax.set_xlabel("ΔRA [\"]", fontsize=8)
    ax.tick_params(labelsize=7)


def read_vendored(disk: str, half_arcsec: float = F.HALF_ARCSEC) -> dict:
    """
    The same band dicts out of the *vendored* `realimg_<disk>.npz` -- the pickle route.

    Read directly rather than through `_realdisk.load_real_images`, which goes via the `DISKS`
    registry: hvtauc is not in it, and the point here is to compare the two readers on both disks.
    """
    z = np.load(F.ASSETS_DIR / f"realimg_{disk}.npz")
    out = {}
    for key, prefix in [("jwst", "jwst")] + [(f"alma_{j}", f"alma_{j}") for j in range(3)]:
        if f"{prefix}_img" not in z:
            continue
        img, ra, dec = F._cut(z[f"{prefix}_img"], np.asarray(z[f"{prefix}_rRA"], float),
                              np.asarray(z[f"{prefix}_rDEC"], float), half_arcsec)
        lam = F.JWST_LAM_UM if key == "jwst" else rd.ALMA_CHANNELS[int(key[-1])][1]
        out[key] = dict(img=img, rRA=ra, rDEC=dec, lam_um=lam,
                        label=rd.BAND_LABEL["im_jy_jwst" if key == "jwst" else f"im_jy_{key}"],
                        px_arcsec=abs(float(ra[1] - ra[0])), centroid=F._centroid(img, ra, dec))
    return out


def plot_disk(disk: str, half_arcsec: float = F.HALF_ARCSEC, out_dir=PLOT_DIR,
              source: str = "fits") -> pathlib.Path:
    """
    Every band of one disk, cut to +/-`half_arcsec`, into one PNG.

    `source="vendored"` plots `realimg_<disk>.npz` (the upstream pickles) instead of the FITS, on
    the same axes and stretch -- which is how the two readers get compared, NaN footprint included.
    """
    bands = (F.read_disk(disk, half_arcsec=half_arcsec) if source == "fits"
             else read_vendored(disk, half_arcsec=half_arcsec))
    keys = [k for k in PANELS if k in bands]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.4 * len(keys), 3.9), constrained_layout=True)
    for ax, key in zip(np.atleast_1d(axes), keys):
        plot_band(ax, bands[key])
    np.atleast_1d(axes)[0].set_ylabel("\u0394DEC [\"]", fontsize=8)
    fig.suptitle(f"{disk} [{source}] — central {2 * half_arcsec:g}\"  (asinh, knee = 3σ; "
                 f"+ centroid, × field centre; grey = no data)", fontsize=10)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{disk}_{source}_central{2 * half_arcsec:g}arcsec.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or F.disks()
    sources = ["fits", "vendored"] if "--vendored" in sys.argv else ["fits"]
    for name in names:
        for src in sources:
            print(plot_disk(name, source=src))

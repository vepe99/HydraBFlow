"""
Vendor the real-disk images into `assets/protoplan/` -- run once, then the notebooks need no pickles.

The upstream `imdata_<disk>.pkl` files are 100-300 MB each, almost all of it the JWST spectral cube
and the full ALMA fields; `load_real_images` reads four arrays per band from them.  This writes those
arrays, cropped to +/-`HALF_ARCSEC`, into one compressed `.npz` per disk -- a few MB, small enough to
live in the repo.

The crop is twice the +/-1.5" model field (`_realdisk.OBS_HALF_EXTENT`), so `regrid_to_model` never
reaches the edge.  Verified against the source FITS: `imdata_Jybm` in the pickles is byte-identical
to the ALMA `*.pbcor.fits` images, and the JWST `image_3.9um_<disk>.npz` is the single 3.9um slice of
the same cube the pickle carries (the pickle's `im39` is a narrow-band average of it).

    uv run python notebooks/prior_predictive_checks/_vendor_real_images.py
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _realdisk import ALMA_CHANNELS, ASSETS_DIR, BAND_LABEL, OBS_DATA_DIR  # noqa: E402

HALF_ARCSEC = 3.0

#: disk -> (pickle, jwst attribute, ALMA channels present).  Superset of `_realdisk.DISKS`: the
#: pickles hold four disks and vendoring all of them costs a few MB.
SOURCES = {
    "oph163131": ("imdata_oph163131.pkl", "g395", (0, 2)),
    "saucer":    ("imdata_saucer.pkl",    "g395", (0, 1, 2)),
    "hvtauc":    ("imdata_hvtauc.pkl",    "g395", (1, 2)),
    "esoha574":  ("imdata_esoha574.pkl",  "g395", (1, 2)),
}


def _crop(img, rRA, rDEC):
    """`(img, ra_1d, dec_1d)` cut to +/-HALF_ARCSEC.  Accepts 1-D or 2-D coordinate arrays."""
    ra = np.asarray(rRA, float)
    dec = np.asarray(rDEC, float)
    ra_1d = ra[0, :] if ra.ndim == 2 else ra
    dec_1d = dec[:, 0] if dec.ndim == 2 else dec
    ir = np.abs(ra_1d) <= HALF_ARCSEC
    idec = np.abs(dec_1d) <= HALF_ARCSEC
    return (np.asarray(img, np.float32)[np.ix_(idec, ir)],
            ra_1d[ir].astype(np.float32), dec_1d[idec].astype(np.float32))


def vendor(disk: str) -> pathlib.Path:
    import dill

    pkl, jwst_attr, channels = SOURCES[disk]
    with open(OBS_DATA_DIR / pkl, "rb") as fh:
        obs = dill.load(fh)

    out = {}
    jw = getattr(obs, jwst_attr)
    out["jwst_img"], out["jwst_rRA"], out["jwst_rDEC"] = _crop(jw.im_cent, jw.rRA, jw.rDEC)
    for j in channels:
        b = getattr(obs, ALMA_CHANNELS[j][0])
        (out[f"alma_{j}_img"], out[f"alma_{j}_rRA"],
         out[f"alma_{j}_rDEC"]) = _crop(b.im_cent_MJyster, b.rRA, b.rDEC)

    path = ASSETS_DIR / f"realimg_{disk}.npz"
    np.savez_compressed(path, **out)
    print(f"{path.name}: {path.stat().st_size / 1e6:.2f} MB  "
          + "  ".join(f"{k.rsplit('_', 1)[0]}{v.shape}" for k, v in out.items() if k.endswith("img")))
    return path


if __name__ == "__main__":
    for name in sys.argv[1:] or SOURCES:
        vendor(name)

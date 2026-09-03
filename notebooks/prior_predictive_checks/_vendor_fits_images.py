"""
Vendor the real disks from `extracted_fits/` into the `.npz` the rest of this directory reads.

The FITS route's counterpart to `_vendor_real_images.py`, which does the same job from the upstream
`dill` pickles.  Output keys are identical, so `_realdisk.load_real_images` -> `build_real_batch` ->
`make_real_npz.py` -> `hydrabflow-evaluate data.real_data_path=...` needs no change:

    jwst_img / jwst_rRA / jwst_rDEC          MJy/sr, arcsec
    alma_{j}_img / alma_{j}_rRA / alma_{j}_rDEC

plus what the pickle route had no way to carry, and the vendoring step is the right place to fix:

    <band>_observed        bool, False where the pixel was never observed and has been noise-filled
    <band>_bkg             (mean, sigma) of the background those fills were drawn from, MJy/sr
    jwst_measured_sigma    what the annulus actually read, when `JWST_NOISE_MJY_SR` overrode it
    alma_{j}_beam          (fwhm_maj, fwhm_min, bpa_deg) arcsec/deg, straight from the header
    alma_{j}_lam_um        the map's *true* wavelength, which is not always its band's slot
    alma_{j}_theta_deg     the emission's position angle, which `load_obs_setup` returns as rot_deg
    <band>_source_radec    the ICRS position *that band* was cut about (its own centroid, since
                           `read_disk` recentres per band -- see `_fitsdisk.measure_source_radec`)
    source_radec           the shared starting guess those iterations began from

Cropped to +/-`HALF_ARCSEC` = 3.0", twice the +/-1.5" model field, so `regrid_to_model` never
reaches an edge -- the same margin `_vendor_real_images.py` uses.

    uv run python notebooks/prior_predictive_checks/_vendor_fits_images.py            # both disks
    uv run python notebooks/prior_predictive_checks/_vendor_fits_images.py --overwrite

Without `--overwrite` this writes `realimg_<disk>_fits.npz` and leaves the pickle-derived
`realimg_<disk>.npz` alone -- they are *different reductions* (the JWST fluxes differ by ~20%, and
oph163131's B9 beam is 0.174" there against 0.160" here), so which one a run used is worth keeping
straight.  `--overwrite` promotes these to `realimg_<disk>.npz`, which is what `load_real_images`
actually reads; the pickle versions are regenerable with `_vendor_real_images.py`.
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import _fitsdisk as F  # noqa: E402

HALF_ARCSEC = 3.0


def vendor(disk: str, overwrite: bool = False, half_arcsec: float = HALF_ARCSEC) -> pathlib.Path:
    bands = F.read_disk(disk, half_arcsec=half_arcsec)
    out = {"source_radec": np.asarray(F.SOURCE_RADEC[disk], np.float64)}
    for key, b in bands.items():
        out[f"{key}_img"] = np.asarray(b["img"], np.float32)
        out[f"{key}_rRA"] = np.asarray(b["rRA"], np.float32)
        out[f"{key}_rDEC"] = np.asarray(b["rDEC"], np.float32)
        out[f"{key}_observed"] = np.asarray(b["observed"], bool)
        out[f"{key}_bkg"] = np.asarray(b["bkg"], np.float64)
        out[f"{key}_source_radec"] = np.asarray(b["source_radec"], np.float64)
        if "measured_sigma" in b:
            out[f"{key}_measured_sigma"] = np.float64(b["measured_sigma"])
        if "fwhm_maj" in b:
            out[f"{key}_beam"] = np.asarray([b["fwhm_maj"], b["fwhm_min"], b["bpa_deg"]])
            out[f"{key}_lam_um"] = np.float64(b["lam_um"])
            out[f"{key}_theta_deg"] = np.float64(b["theta_deg"])

    name = f"realimg_{disk}.npz" if overwrite else f"realimg_{disk}_fits.npz"
    path = F.ASSETS_DIR / name
    np.savez_compressed(path, **out)
    print(f"{path.name}: {path.stat().st_size / 1e6:.2f} MB")
    for key, b in bands.items():
        filled = int((~b["observed"]).sum())
        mean, sigma = b["bkg"]
        print(f"   {key:<7} {str(b['img'].shape):>12} @ {b['px_arcsec']:.5f}\"  "
              f"peak {np.nanmax(b['img']):.4g} MJy/sr  "
              f"noise-filled {filled:5d} px ({filled / b['observed'].size:5.1%}) "
              f"from N({mean:.3g}, {sigma:.3g})"
              + (f"  [sigma assumed; annulus read {b['measured_sigma']:.3g}]"
                 if b.get("measured_sigma", sigma) != sigma else ""))
    return path


if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or F.disks()
    for name in names:
        vendor(name, overwrite="--overwrite" in sys.argv)

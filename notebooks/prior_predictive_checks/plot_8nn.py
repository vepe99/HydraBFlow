"""
The real disk beside its N nearest training sims, all at the disk's own measured PA.

    uv run python notebooks/prior_predictive_checks/plot_8nn.py              # both disks, 7 NN
    uv run python notebooks/prior_predictive_checks/plot_8nn.py hvtauc --n 7 --rows 8192

One PNG per band in `assets/protoplan/extracted_fits/plots/8nn/`: the real observation first, then
the 7 nearest training rows in the joint whitened shape space, each panel stretched to its own peak
(amplitude is `check_obs_range`'s question, not this one's).

This is `check_shape_range.py`'s neighbour panel as a script -- same forward model, same features,
same distance -- so the two cannot drift.  Three things it pins deliberately:

* **One angle, the disk's own.**  `fixed_setup_augmentation` defaults to `random_rotation=False`,
  which collapses `rot_range_deg` to `(theta, theta)` and turns `random_flip` off.  Every sim is
  observed at the real disk's PA through the real disk's beams, so the panels are like-for-like;
  sampling either would put disk-vs-beam-angle scatter in the comparison that the real disk, at one
  known PA and one known parity, does not have.
* **The full instrument model runs**, noise included -- these are mock *observations* of the
  training rows, not clean RT images, which is the only fair comparison to a real map.
* **The winning rows are re-observed, not cached.**  Keeping every augmented image would cost
  ~15 GB; the noise redraw changes nothing about the shape the panel exists to show.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import _realdisk as rd    # imports hydrabflow, which pins KERAS_BACKEND and picks a free GPU
import _fitsdisk as F     # noqa: E402

OUT_DIR = F.FITS_DIR / "plots" / "8nn"


def shape_space(disk: str, n_rows: int = 0, batch_size: int = 256):
    """
    `(rows_usable, Zw, zw, aug, real, axes_arc, bands, data)` for one disk.

    Streams the training set through the disk-pinned forward model, reducing each augmented image
    to `rd.shape_features` and keeping no pixels, then whitens the joint normal-score space so a
    Euclidean distance in it *is* the Mahalanobis distance the notebook reports.
    """
    cfg = rd.load_cfg()
    data = rd.load_training_data(cfg)
    n_model_px = data["im_jy"].shape[1]
    setup, rot_deg = rd.load_obs_setup(disk)

    aug = rd.fixed_setup_augmentation(disk, n_model_px)      # fixed PA, no flip -- see module docs
    train_lams = np.asarray(data["sed_lams"][0, :, 0], dtype=np.float64)
    real, aux = rd.build_real_batch(disk, train_lams)
    bands = rd.present_image_keys(disk)

    sigma = {"im_jy_jwst": float(rd.disk_config(disk)["sigma_jwst"])}
    sigma.update({f"im_jy_alma_{j}": float(s["sigma_mjy_sr"]) for j, s in setup.items()})
    axes_arc = {b: (aux["jwst_axis"] if b == "im_jy_jwst" else aux["alma_axis"]) for b in bands}
    for b in bands:                       # a grid mismatch would compare different samplings
        assert np.shape(real[b])[1] == len(axes_arc[b]), b

    print(f"{disk}: rot_deg={rot_deg:.3f} (from {rd.disk_config(disk)['rot_band']}), "
          f"bands={bands}, n_model_px={n_model_px}", flush=True)

    total = len(data["im_jy"]) if not n_rows else min(n_rows, len(data["im_jy"]))
    feats = {b: [] for b in bands}
    for i, j, raw in rd.training_batches(data, batch_size, n_rows):
        out = aug(raw)
        for b in bands:
            feats[b].append(rd.shape_features(np.asarray(out[b])[..., 0], axes_arc[b], sigma[b]))
        if (i // batch_size) % 20 == 0:
            print(f"  rows {j}/{total}", flush=True)
    F_train = {b: np.concatenate(v) for b, v in feats.items()}
    F_real = {b: rd.shape_features(np.asarray(real[b]), axes_arc[b], sigma[b])[0] for b in bands}
    for b in bands:
        assert np.isfinite(F_real[b]).all(), f"{b}: the real disk's features are not finite"

    # Rows usable in *every* band, so the rank transform and the covariance see one population.
    good = np.ones(len(F_train[bands[0]]), dtype=bool)
    for b in bands:
        good &= np.isfinite(F_train[b]).all(axis=1)
    tr = np.hstack([F_train[b][good] for b in bands])
    sa = np.hstack([F_real[b] for b in bands])
    Zw, zw = rd.whiten(*rd.normal_score_block(tr, sa))
    print(f"  {good.sum()}/{len(good)} rows usable in every band, {tr.shape[1]}-d shape space")
    return np.flatnonzero(good), Zw, zw, aug, real, axes_arc, bands, data


def plot_disk(disk: str, n_neighbours: int = 7, n_rows: int = 0,
              batch_size: int = 256, out_dir=OUT_DIR) -> list[str]:
    rows_ok, Zw, zw, aug, real, axes_arc, bands, data = shape_space(disk, n_rows, batch_size)

    d = np.linalg.norm(Zw - zw[None, :], axis=1)
    order = np.argsort(d)[:n_neighbours]
    rows, dists = rows_ok[order], d[order]
    print(f"  nearest rows: {rows.tolist()}\n  distances:    {np.round(dists, 3).tolist()}")

    sub = {k: np.asarray(data[k][rows]) for k in ("im_jy", "im_lams", "seds", "sed_lams")}
    out = aug(sub)
    labels = [f"row {r}  d={x:.2f}" for r, x in zip(rows, dists)]

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return [rd.neighbour_panel(np.asarray(real[b])[0, :, :, 0], np.asarray(out[b])[..., 0],
                               labels, b, axes_arc[b], str(out_dir), disk, suffix=f"_{disk}")
            for b in bands]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("disks", nargs="*", default=None, help="default: every disk in extracted_fits/")
    ap.add_argument("--n", type=int, default=7, help="neighbours to show beside the real disk")
    ap.add_argument("--rows", type=int, default=0, help="training rows to search (0 = all)")
    ap.add_argument("--batch-size", type=int, default=256)
    a = ap.parse_args()
    for name in a.disks or F.disks():
        print("\n".join(plot_disk(name, a.n, a.rows, a.batch_size)))

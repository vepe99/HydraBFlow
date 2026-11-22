#!/usr/bin/env python
"""Pal 5 x Milky-Way halo: amortized normalizing-flow posterior (BayesFlow) for the SAME prior and the
SAME likelihood as the nested-sampling run of ``scripts/pal5_galax_ns.py``, + importance sampling.

ONE file, staged (``all`` runs them in order, each with the interpreter it needs, skipping finished ones):

  simulate  prior draws -> galax stream -> the NOISE-FREE simulation: the NS bin model per phi1 bin (particle
            count, mean phi1, track centre, slope, variance) + up to 512 of the stream particles of each bin;
            one worker per free GPU (autocvd), chunked, resumable.                              [galax env]
  train     coupling-flow posterior + MLP summary network (BayesFlow). Every batch re-draws the NOISE of the
            likelihood (and the nuisance parameters, which do not enter the simulation) on top of the stored
            simulations; then posterior draws for the observed data with their density q(theta | data). [.venv]
  is        the nested-sampling likelihood THAT THE MODEL WAS TRAINED ON for every posterior draw (multi-GPU)
            -> importance weights w = L(theta) pi(theta) / q(theta | data), effective sample size, evidence,
            reweighted posterior, corner plot (+ the NS posterior when it exists).              [galax env]
  check     numerical test that the training noise model is the NS likelihood (star and bin).  [galax env]

``--noise`` selects the likelihood the mock data are drawn from:
  gaussian  (default) the NS ``--likelihood star``: every observed star ~ (1 - f_out) N(bin track at the star's
            phi1, model width^2 (1 + pi/2/n) + Gaia error^2 + sigma_inflate^2) + f_out U(outlier box). With
            ``--features stars`` the network sees every star's value (593 numbers): exactly the NS likelihood.
  nsbin     the NS ``--likelihood bin``: the observed bin centre of every (phi1 bin, observable) cell
            (``pal5_galax_ns.bin_data``) ~ N(model bin centre at the cell's phi1, standard error of the observed
            centre^2 + model width^2 pi/2/n + sigma_inflate^2); no outliers (f_out keeps its prior). The network
            sees the cell centres (<= 50 numbers). Importance sampling then uses the NS bin likelihood.
  nsbspline the B-spline-track likelihood: stars ~ N(bin track, star variance) WITHOUT outliers, data = their
            core080 B-spline tracks t_k = H_k y_k (`bspline_operators`). A linear map of Gaussians, so
            logL = sum_k log N(V_k^T y_k; V_k^T mu_k, V_k^T diag(tot_k) V_k), V_k = orthonormal basis of the row
            space of H_k (70 dimensions; `NoiseModel.logpdf`). The network sees the 100 track values + the 70 z.
            The `is` stage evaluates this density itself (NS forward model + bin model on the GPU).
  particles NOT an NS likelihood: mock stars = random stream particles of the star's bin + Gaia error +
            inflation (+ outliers), usually with ``--features bins`` (per-bin mean and scatter). It has no
            closed-form likelihood, so its importance weights use the NS star likelihood, a different target.

Relation to the nested sampling:
  * prior = ``pal5_galax_ns.prior_sample`` restricted to the draws the NS likelihood does not veto (failed
    integration / an observed phi1 bin with < N_MIN_BIN particles get logL = -1e6 there, i.e. zero posterior mass);
  * forward model = ``pal5_galax_ns.simulate_stream`` with its fixed spray key, 10^4 particles, float64;
  * the uniform outlier component (constant density 1/prod(OUTLIER_WIDTH) in the NS star likelihood) is a box of
    that size centred on the observed mid-range, which contains every observed star.
  * parameters are trained in u = probit(prior CDF) (uniform priors) / z-score (normal priors); the inputs are
    squashed with arcsinh((y - observed mean) / scale); affine couplings (the spline flow went NaN).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs/pal5_galax_ns/sbi_comparison"
NS_OUT = REPO / "outputs/pal5_galax_ns"
GALAX_SITE = REPO / ".venv-galax/lib/python3.12/site-packages"
BF_PYTHON = REPO / ".venv/bin/python"
SEED_BASE = 77000


# ======================================================================================================
# Interpreters / GPUs
# ======================================================================================================
def galax_cmd() -> tuple[list[str], dict]:
    """Interpreter + env for the galax stages. ``.venv-galax`` links /usr/bin/python3.12, which only exists
    on some hosts; elsewhere any Python 3.12 (uv-managed) runs the same site-packages through PYTHONPATH."""
    env = dict(os.environ)
    py = REPO / ".venv-galax/bin/python"
    if py.exists():
        return [str(py)], env
    cands = sorted(glob.glob(os.path.expanduser("~/.local/share/uv/python/cpython-3.12*/bin/python3.12")))
    if not cands:
        sys.exit("no Python 3.12 for the galax env: run `uv python install 3.12`")
    env["PYTHONPATH"] = f"{GALAX_SITE}:{env.get('PYTHONPATH', '')}".rstrip(":")
    return [cands[-1]], env


def pick_free_gpus(n: int) -> list[str]:
    """Free GPUs from autocvd. Fallback (autocvd crashes when `nvidia-smi -L` errors on a broken card, as on
    compgpu6): the idle GPUs of an nvidia-smi query, addressed by UUID so a dead card cannot shift the indices."""
    if os.environ.get("PAL5_GPU_IDS"):  # explicit GPUs (ids or UUIDs), e.g. to share the cards of our OWN jobs
        gpus = os.environ["PAL5_GPU_IDS"].split(",")[:n]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
        return gpus
    try:
        from autocvd import autocvd

        autocvd(num_gpus=n)
        return os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    except Exception as e:
        print(f"autocvd failed ({type(e).__name__}); falling back to an nvidia-smi query", flush=True)
    while True:
        q = subprocess.run(["nvidia-smi", "--query-gpu=uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True).stdout
        free = []
        for line in q.splitlines():
            c = [s.strip() for s in line.split(",")]
            if len(c) == 3 and c[0].startswith("GPU-") and float(c[1]) < 100 and float(c[2]) < 5:
                free.append(c[0])
        if len(free) >= n:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(free[:n])
            return free[:n]
        print(f"only {len(free)} free GPUs, need {n}: waiting", flush=True)
        time.sleep(60)


def import_ns():
    """The nested-sampling module (prior, simulator, likelihood). Call only after the GPU is pinned."""
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import pal5_galax_ns as ns

    return ns


def run_workers(mode: str, out: Path, n_gpus: int, extra: list[str]) -> None:
    """One ``mode`` worker per free GPU; wait for all of them."""
    gpus = pick_free_gpus(n_gpus)
    print(f"[{mode}] free GPUs from autocvd: {gpus}", flush=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    cmd0, env0 = galax_cmd()
    procs = []
    for w, gpu in enumerate(gpus):
        env = dict(env0, CUDA_VISIBLE_DEVICES=gpu, XLA_PYTHON_CLIENT_PREALLOCATE="false")
        env.pop("JAX_PLATFORMS", None)
        cmd = cmd0 + [__file__, mode, "--out", str(out), "--worker", str(w), "--n-workers", str(len(gpus))] + extra
        log = open(out / "logs" / f"{mode}_{w:02d}.log", "a")
        procs.append(subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT))
        print(f"  worker {w} -> GPU {gpu} (pid {procs[-1].pid})", flush=True)
    codes = [p.wait() for p in procs]
    if max(codes) != 0:
        sys.exit(f"[{mode}] worker exit codes {codes}: see {out / 'logs'}")


# ======================================================================================================
# 1) simulate: prior draws -> noise-free bin model
# ======================================================================================================
def cmd_sim_worker(args):
    ns = import_ns()
    import jax
    import jax.numpy as jnp
    import jax.random as jr

    out = Path(args.out) / "sims"
    out.mkdir(parents=True, exist_ok=True)
    data = ns.load_data()
    edges = jnp.asarray(data["edges"])

    nb, K = edges.shape[0] - 1, args.pool

    def one(theta, key):
        obs, ok, _ = ns.simulate_stream(theta)
        n, mphi, centre, slope, var = ns.bin_model(obs, edges)
        # the stream particles themselves: per phi1 bin a random subset of (up to) K of them, each moved along
        # the bin's linear track to the bin's mean phi1 (value = y - slope (phi1 - <phi1>)); the first
        # min(n, K) entries of a bin are distinct particles, the rest is padding
        phi1, y = obs[:, 0], obs[:, 1:]
        idx = jnp.searchsorted(edges, phi1) - 1
        inbin = (idx[None, :] == jnp.arange(nb)[:, None]) & (phi1[None, :] > edges[0]) & (phi1[None, :] < edges[-1])
        score = jnp.where(inbin, jr.uniform(key, inbin.shape), -1.0)
        sel = jnp.argsort(-score, axis=1)[:, :K]  # (B, K)
        pool = y[sel] - slope[:, None, :] * (phi1[sel] - mphi[:, None])[:, :, None]
        return ok, n, mphi, centre, slope, var, pool.astype(jnp.float32)

    f = jax.jit(jax.vmap(one))
    quota = int(np.ceil(args.n_sims / args.n_workers))
    kept, c = 0, 0
    print(f"worker {args.worker} on GPU {os.environ.get('CUDA_VISIBLE_DEVICES')} {jax.devices()}: quota {quota}", flush=True)
    while kept < quota:
        path = out / f"w{args.worker:02d}_c{c:05d}.npz"
        if path.exists():
            kept += int(np.load(path)["keep"].sum())
            c += 1
            continue
        t0 = time.time()
        key = jr.fold_in(jr.key(SEED_BASE + args.worker), c)
        theta = ns.prior_sample(key, args.batch)
        ok, n, mphi, centre, slope, var, pool = (np.asarray(a) for a in f(theta, jr.split(jr.fold_in(key, 1), args.batch)))
        finite = np.all(np.isfinite(centre) & np.isfinite(slope) & np.isfinite(var), axis=(1, 2)) & np.all(np.isfinite(mphi), 1)
        ok = ok & finite
        keep = ok & np.all(n >= ns.N_MIN_BIN, axis=1)  # exactly the draws the NS likelihood does not veto
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, theta=np.asarray(theta), ok=ok, keep=keep, n=n, mphi=mphi, centre=centre, slope=slope, var=var, pool=pool)
        os.replace(tmp, path)
        kept += int(keep.sum())
        c += 1
        print(f"chunk {c:5d} | kept {kept:6d}/{quota} | this chunk {int(keep.sum())}/{args.batch} "
              f"(failed {int((~ok).sum())}) | {time.time() - t0:.1f}s", flush=True)


def merge_sims(out: Path, n_sims: int) -> None:
    files = sorted((out / "sims").glob("w*_c*.npz"))
    parts = []
    for f in files:
        p = dict(np.load(f))
        p["pool"] = p["pool"][p["keep"]]  # the big one: kept rows only
        parts.append(p)
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    keep = cat["keep"]
    pool_row = np.cumsum(keep) - 1  # row of `pool` for a kept draw
    # interleave the workers' chunks so that truncating to n_sims does not drop one worker's tail only
    idx = np.random.default_rng(0).permutation(np.where(keep)[0])[:n_sims]
    # observed data + prior definition, so that the training stage needs neither galax nor gala
    os.environ["JAX_PLATFORMS"] = "cpu"
    ns = import_ns()
    d = ns.load_data()
    stats = dict(n_draws=int(len(keep)), n_failed=int((~cat["ok"]).sum()), n_vetoed_undersampled=int((cat["ok"] & ~keep).sum()),
                 n_kept=int(keep.sum()), n_saved=int(len(idx)))
    np.savez(
        out / "training_set.npz",
        theta=cat["theta"][idx], n=cat["n"][idx], mphi=cat["mphi"][idx], centre=cat["centre"][idx],
        slope=cat["slope"][idx], var=cat["var"][idx], pool=cat["pool"][pool_row[idx]],
        obs_phi1=d["phi1"], obs_y=d["y"], obs_err=d["err"], obs_mask=d["mask"], obs_bin=d["bin"], edges=d["edges"],
        names=np.array(ns.NAMES), kind_u=ns._KIND_U, prior_a=ns._A, prior_b=ns._B,
        outlier_width=ns.OUTLIER_WIDTH, obs_names=np.array(ns.OBS_NAMES), stats=json.dumps(stats),
    )
    (out / "simulate_stats.json").write_text(json.dumps(stats, indent=1))
    print(f"[simulate] {stats} -> {out / 'training_set.npz'}", flush=True)


def cmd_simulate(args):
    out = Path(args.out)
    if (out / "training_set.npz").exists():
        print("[simulate] training_set.npz exists: skipping")
        return
    t0 = time.time()
    run_workers("sim-worker", out, args.gpus, ["--n-sims", str(args.n_sims), "--batch", str(args.batch), "--pool", str(args.pool)])
    merge_sims(out, args.n_sims)
    print(f"[simulate] done in {(time.time() - t0) / 3600:.2f} h", flush=True)


# ======================================================================================================
# B-spline track summary of the main HydraBFlow pipeline (`stream_bspline_grid`, preset core080)
# ======================================================================================================
def bspline_operators(phi1, y, mask, n_grid=20, n_int=20, n_int_v=6, core_frac=0.8, lam_scale=10.0, ridge=1e-3):
    """The `stream_global_streamfinder_bspline_core080` estimator as fixed linear maps. Per observable: a
    penalized cubic B-spline (uniformly extended knots, `n_int` interior over the core; v_los: `n_int_v`
    over the measured stars) fitted to the stars inside the central `core_frac` phi1 quantile range of the
    real members, lambda = `lam_scale` x scipy's GCV choice on the real members, extended linearly to the
    real range ends with the chord slope over the outer 10 % of the core, evaluated on `n_grid` uniform phi1
    points of the real range. The mock stars sit at the OBSERVED phi1, so the design matrix is the same for
    every simulation and the whole fit is one matrix per observable: track_k = H[k] @ y[:, k].
    Returns H (5, n_grid, S), the grids (5, n_grid) and the lambdas. Differences from the main pipeline:
    Pal5PriceWhelan18 frame (not the STREAMFINDER great circle), pm_ra*/pm_dec (not rotated into the stream
    frame), no `valid` flags (constant here)."""
    from hydrabflow.augmentation.stream_bspline import _extended_knots, _gcv_lambda, _np_basis

    S = len(phi1)
    H, grids, lams = np.zeros((5, n_grid, S)), np.zeros((5, n_grid)), np.zeros(5)
    for k in range(5):
        w = mask[:, k] > 0
        x = phi1[w]
        lo, hi = x.min(), x.max()
        c0, c1 = np.quantile(x, [(1 - core_frac) / 2, (1 + core_frac) / 2])
        t, om = _extended_knots(np.array([[c0, c1]]), n_int_v if k == 4 else n_int)
        t, om = t[0], om[0]
        lam = lam_scale * _gcv_lambda(x, y[w, k], np.array([lo, hi]))
        grid = lo + (hi - lo) * np.linspace(0.0, 1.0, n_grid)
        h = 0.1 * (c1 - c0)
        q = np.concatenate([np.clip(grid, c0, c1), [c0, c0 + h, c1 - h, c1]])
        wc = (w & (phi1 >= c0) & (phi1 <= c1)).astype(float)
        B, Bq = _np_basis(phi1, t).astype(float), _np_basis(q, t).astype(float)
        A = B.T @ (B * wc[:, None]) + ridge * np.eye(B.shape[1]) + lam * om
        Hq = Bq @ np.linalg.solve(A, (B * wc[:, None]).T)  # (n_grid + 4, S)
        f, e = Hq[:n_grid], Hq[n_grid:]
        lo_s, hi_s = (e[1] - e[0]) / h, (e[3] - e[2]) / h
        H[k] = (f + np.where(grid < c0, grid - c0, 0.0)[:, None] * lo_s[None]
                + np.where(grid > c1, grid - c1, 0.0)[:, None] * hi_s[None])
        grids[k], lams[k] = grid, lam
    return H, grids, lams


# ======================================================================================================
# The likelihood's noise model in numpy (what `make_loglike` evaluates, turned into a sampler)
# ======================================================================================================
class NoiseModel:
    def __init__(self, z, features: str = "stars", binned: bool = False, tracks: bool = False, ops=None):
        self.binned = binned  # data = one bin centre per (phi1 bin, observable) cell: the NS `--likelihood bin`
        self.tracks = tracks  # data = the B-spline tracks of the stars (Gaussian stars, no outliers)
        self.names = [str(s) for s in z["names"]]
        self.kind_u, self.a, self.b = z["kind_u"].astype(bool), z["prior_a"], z["prior_b"]
        self.phi1, self.y, self.bin = z["obs_phi1"], z["obs_y"], z["obs_bin"]
        self.err2 = (z["obs_err"] ** 2).astype(np.float32)
        self.mask = z["obs_mask"] > 0
        self.width = z["outlier_width"].astype(np.float32)
        lo = np.array([self.y[self.mask[:, k], k].min() for k in range(5)])
        hi = np.array([self.y[self.mask[:, k], k].max() for k in range(5)])
        assert np.all(hi - lo < self.width), "an observed star lies outside the outlier box"
        self.box_lo = (0.5 * (lo + hi) - 0.5 * self.width).astype(np.float32)
        self.i_sig = [self.names.index(k) for k in ("sig_phi2", "sig_plx", "sig_pmra", "sig_pmdec", "sig_vlos")]
        self.i_f = self.names.index("f_out")
        self.i_nuis = self.i_sig + [self.i_f]
        self.features = features
        self.nb = int(self.bin.max()) + 1
        self.onehot = (self.bin[:, None] == np.arange(self.nb)[None]).astype(np.float32)  # (S, B)
        self.w = self.onehot[:, :, None] * self.mask[:, None, :]  # (S, B, 5): star s contributes to cell (b, k)
        self.cnt = self.w.sum(0)  # observed stars per (bin, observable)
        self.scale = np.array([1.0, 0.3, 0.5, 0.5, 20.0], np.float32)  # deg, mas, mas/yr, mas/yr, km/s
        self.y0 = np.array([self.y[self.mask[:, k], k].mean() for k in range(5)], np.float32)
        if features == "bspline":
            if ops is not None and Path(ops).exists():  # the galax env has no hydrabflow: operators from `train`
                H = np.load(ops)["H"]
            else:
                H, self.bs_grid, self.bs_lam = bspline_operators(self.phi1, self.y, self.mask)
                if ops is not None:
                    np.savez(ops, H=H, grid=self.bs_grid, lam=self.bs_lam)
            self.H = H.astype(np.float32)  # (5, G, S)
            # The tracks t_k = H_k y_k live in the row space of H_k (rank 15, 15, 15, 15, 10 of 20: the points
            # beyond the 80 % core are linear extrapolations). V_k = orthonormal basis of that row space, so
            # z_k = V_k^T y_k is a one-to-one, well-conditioned re-expression of the track t_k = (H_k V_k) z_k.
            self.V = []
            for k in range(5):
                _, sv, vt = np.linalg.svd(H[k], full_matrices=False)
                self.V.append(vt[sv > 1e-10 * sv[0]].T)  # (S, r_k), float64
            self.V32 = [v.astype(np.float32) for v in self.V]

    # ---- noise-free part, per star -----------------------------------------------------------------
    def per_star(self, n, mphi, centre, slope, var, dtype=np.float32):
        """Bin model -> per-star (per-cell if binned) predictive mean and model variance, as `make_loglike`:
        star: width^2 (1 + pi/2/n) (dispersion + Monte-Carlo error of the median); bin: width^2 pi/2/n."""
        b = self.bin
        mc = 0.5 * np.pi / np.maximum(n[:, b], 1.0)
        if self.binned:  # phi1 is per cell (B, 5)
            mu = centre[:, b] + slope[:, b] * (self.phi1[None] - mphi[:, b][:, :, None])
            vmod = var[:, b] * mc[:, :, None]
        else:
            mu = centre[:, b] + slope[:, b] * (self.phi1[None, :] - mphi[:, b])[:, :, None]
            vmod = var[:, b] * (1.0 + mc)[:, :, None]
        return mu.astype(dtype), vmod.astype(dtype)

    # ---- parameters <-> unconstrained --------------------------------------------------------------
    def to_u(self, theta):
        """Uniform priors -> probit of the prior CDF, normal priors -> z-score: u ~ N(0, 1) under the prior
        (a logit has logistic tails out to |u| ~ 16, on which the spline flow went NaN)."""
        from scipy.special import ndtri

        p = np.clip((theta - self.a) / (self.b - self.a), 1e-7, 1 - 1e-7)
        return np.where(self.kind_u, ndtri(p), (theta - self.a) / self.b)

    def to_theta(self, u):
        from scipy.special import ndtr

        u = np.asarray(u, np.float64)
        return np.where(self.kind_u, self.a + (self.b - self.a) * ndtr(u), self.a + self.b * u)

    def log_du_dtheta(self, u):
        """log |det du/dtheta| at theta(u): probit -> 1 / ((b - a) phi(u)); normal -> 1 / b."""
        u = np.asarray(u, np.float64)
        width = np.where(self.kind_u, np.abs(self.b - self.a), self.b)
        return np.where(self.kind_u, 0.5 * u**2 + 0.5 * np.log(2 * np.pi), 0.0).sum(-1) - np.log(width).sum()

    def logprior(self, theta):
        inside = (theta >= self.a) & (theta <= self.b)
        lu = np.where(inside, -np.log(np.abs(self.b - self.a)), -np.inf)
        ln = -0.5 * ((theta - self.a) / self.b) ** 2 - np.log(self.b * np.sqrt(2 * np.pi))
        return np.where(self.kind_u, lu, ln).sum(-1)

    def resample_nuisance(self, theta, rng):
        """sigma_inflate (5) and f_out do not enter the simulation and are independent a priori (uniform)."""
        theta = theta.copy()
        k = self.i_nuis
        theta[:, k] = self.a[k] + (self.b[k] - self.a[k]) * rng.random((len(theta), len(k)))
        return theta

    # ---- noise part ----------------------------------------------------------------------------------
    def stars_from_particles(self, pool, n, slope, mphi, rng, rows=None):
        """One mock star per observed star: a random stream particle of the star's phi1 bin, moved along the
        bin's linear track to the star's own phi1 (the likelihood conditions on the observed phi1)."""
        rows = np.arange(len(pool)) if rows is None else rows  # simulations of this batch (no copy of `pool`)
        b, K = self.bin, pool.shape[2]
        navail = np.minimum(n[rows][:, b], K)  # (B, S) distinct particles stored for that bin
        j = np.floor(rng.random(navail.shape) * navail).astype(np.int64)
        d = pool[rows[:, None], b[None, :], j]  # (B, S, 5)
        return (d + slope[rows][:, b] * (self.phi1[None, :] - mphi[rows][:, b])[:, :, None]).astype(np.float32)

    def draw(self, theta, mu, vmod, rng):
        """One data set per row: star ~ (1 - f) N(mu, vmod + Gaia + inflation) + f U(outlier box).
        `mu` = mock stars drawn from the particles (vmod = 0) or the bin track (vmod = model variance)."""
        sig2 = (theta[:, self.i_sig] ** 2).astype(np.float32)
        tot = vmod + self.err2[None] + sig2[:, None, :]
        y = mu + np.sqrt(tot) * rng.standard_normal(mu.shape, dtype=np.float32)
        if self.binned or self.tracks:  # the bin and B-spline likelihoods have no outlier component
            return y
        out = rng.random(mu.shape[:2]) < theta[:, self.i_f][:, None]
        yo = self.box_lo + self.width * rng.random(mu.shape, dtype=np.float32)
        return np.where(out[:, :, None], yo, y)

    def logpdf(self, theta, mu, vmod, y):
        """log density of the data `y` under `draw` (float64): what `make_loglike` returns."""
        theta, mu, vmod, y = (np.asarray(a, np.float64) for a in (theta, mu, vmod, y))
        tot = vmod + self.err2[None].astype(np.float64) + (theta[:, self.i_sig] ** 2)[:, None, :]
        logn = np.sum(self.mask[None] * (-0.5 * (y - mu) ** 2 / tot - 0.5 * np.log(2 * np.pi * tot)), -1)
        if self.binned:
            return logn.sum(-1)
        if self.tracks:  # sum_k log N(V_k^T y_k; V_k^T mu_k, V_k^T diag(tot_k) V_k): the density of the tracks
            out = np.zeros(len(theta))  # (in the coordinates z_k = V_k^T y_k; 70 dimensions in total)
            for k, v in enumerate(self.V):
                r = (y[:, :, k] - mu[:, :, k]) @ v  # (n, r_k)
                c = np.einsum("sa,ns,sb->nab", v, tot[:, :, k], v)
                L = np.linalg.cholesky(c)
                a = np.linalg.solve(L, r[:, :, None])[:, :, 0]
                out += -0.5 * (a * a).sum(-1) - np.log(np.diagonal(L, axis1=1, axis2=2)).sum(-1) - 0.5 * v.shape[1] * np.log(2 * np.pi)
            return out
        f = theta[:, self.i_f][:, None]
        log_out = -np.sum(self.mask * np.log(self.width.astype(np.float64)), -1)[None]
        return np.logaddexp(np.log1p(-f) + logn, np.log(f) + log_out).sum(-1)

    def featurize(self, y):
        """(n, S, 5) -> network input. 'stars': every available value; 'bins': per-bin mean and scatter."""
        # Inputs are squashed with a one-to-one map, arcsinh((y - observed mean) / scale): no information is
        # lost, but the prior's heavy tails (tracks up to 115 deg / 1200 km/s from the data) no longer blow up
        # the standardised inputs (the raw values gave NaN losses in epoch 3).
        if self.features == "stars":
            return np.arcsinh((y - self.y0) / self.scale)[:, self.mask].astype(np.float32)
        if self.features == "bspline":  # B-spline tracks on the phi1 grid: (n, 5 * G)
            c = ((y - self.y0) * self.mask).astype(np.float32)
            tr = np.einsum("nsk,kgs->nkg", c, self.H)
            x = np.arcsinh(tr / self.scale[None, :, None]).reshape(len(y), -1)
            if self.tracks:  # + the same information in the coordinates of the likelihood (70 numbers): the
                # tracks alone hide the less-smooth directions at 1e-3 .. 1e-5 of the leading ones
                x = np.concatenate([x] + [np.arcsinh(c[:, :, k] @ v / self.scale[k]) for k, v in enumerate(self.V32)], -1)
            return x.astype(np.float32)
        cnt = np.maximum(self.cnt, 1.0)
        c = (y - self.y0) * self.mask  # centred on the observed means: E[c^2] - E[c]^2 stays accurate in float32
        m = np.einsum("nsk,sb->nbk", c, self.onehot) / cnt
        s2 = np.maximum(np.einsum("nsk,sb->nbk", c * c, self.onehot) / cnt - m * m, 0.0)
        m, ls = np.arcsinh(m / self.scale), np.log(np.sqrt(s2) / self.scale + 1e-3)
        # a mean needs >= 1 star in the cell, a scatter >= 2 (cells without are constant and dropped)
        return np.concatenate([m[:, self.cnt >= 1], ls[:, self.cnt >= 2]], -1).astype(np.float32)


def obs_arrays(z, binned=None):
    """What `NoiseModel` reads: prior definition + the observed stars, or (binned) the observed bin centres."""
    d = {k: z[k] for k in ("names", "kind_u", "prior_a", "prior_b", "outlier_width", "obs_phi1", "obs_y", "obs_err",
                           "obs_mask", "obs_bin")}
    if binned is not None:
        d.update(obs_phi1=binned["phi1"], obs_y=binned["y"], obs_err=binned["err"], obs_mask=binned["mask"],
                 obs_bin=binned["bin"])
    return d


def cmd_bin_data(args):
    """`pal5_galax_ns.bin_data` of the observed stars -> binned_data.npz (the training env has no galax)."""
    os.environ["JAX_PLATFORMS"] = "cpu"
    ns = import_ns()
    d = ns.bin_data(ns.load_data())
    np.savez(Path(args.out) / "binned_data.npz", **{k: np.asarray(d[k]) for k in ("phi1", "y", "err", "mask", "bin", "n_obs")})


def cmd_check(args):
    """The training noise model must BE the NS likelihood: density of the observed data under `NoiseModel`
    (stored bin model of training rows) vs `pal5_galax_ns.make_loglike` at the same parameters."""
    pick_free_gpus(1)
    ns = import_ns()
    import jax
    import jax.numpy as jnp

    z = np.load(Path(args.out) / "training_set.npz")
    k = args.batch
    theta = z["theta"][:k]
    sim = [z[q][:k] for q in ("n", "mphi", "centre", "slope", "var")]
    stars = ns.load_data()
    for name, binned in (("star", False), ("bin", True)):
        data = ns.bin_data(stars) if binned else stars
        nm = NoiseModel(obs_arrays(z, data if binned else None), binned=binned)
        mu, vmod = nm.per_star(*sim, dtype=np.float64)
        mine = nm.logpdf(theta, mu, vmod, nm.y[None])
        loglike, _ = ns.make_loglike(data, binned=binned)
        ref = np.asarray(jax.jit(jax.vmap(loglike))(jnp.asarray(theta)))
        print(f"[check] {name:4s} likelihood, {k} training rows: NS logL in [{ref.min():.1f}, {ref.max():.1f}], "
              f"max |noise-model logpdf - NS logL| = {np.abs(mine - ref).max():.3g}", flush=True)


# ======================================================================================================
# 2) train
# ======================================================================================================
def cmd_train(args):
    out = Path(args.out)
    if not args.cpu:
        pick_free_gpus(1)
    else:
        os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["KERAS_BACKEND"] = "jax"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import bayesflow as bf
    import keras
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keras.utils.set_random_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    z = np.load(out / "training_set.npz")
    zb = None
    if args.noise == "nsbin":  # observed bin centres, from the NS script (galax env)
        if not (out / "binned_data.npz").exists():
            gcmd, genv = galax_cmd()
            subprocess.check_call(gcmd + [__file__, "bin-data", "--out", str(out)], env=genv)
        zb = np.load(out / "binned_data.npz")
        args.features = "stars"  # the cell centres themselves
    if args.noise == "nsbspline":
        args.features = "bspline"
    nm = NoiseModel(obs_arrays(z, zb), args.features, binned=zb is not None, tracks=args.noise == "nsbspline",
                    ops=out / "bspline_ops.npz" if args.features == "bspline" else None)
    theta = z["theta"][: args.n_train_max]
    N = len(theta)
    if args.noise == "particles":  # mock stars = stream particles + errors
        sims = {"pool": z["pool"][:N], "n": z["n"][:N], "slope": z["slope"][:N].astype(np.float32),
                "mphi": z["mphi"][:N].astype(np.float32)}
    else:  # mock data = the NS likelihood's Gaussian about the bin track (per star, or per bin centre)
        mu, vmod = nm.per_star(z["n"][:N], z["mphi"][:N], z["centre"][:N], z["slope"][:N], z["var"][:N])
        sims = {"mu": mu, "vmod": vmod}
    n_val = min(args.n_val, len(theta) // 10)
    tr, va = slice(n_val, None), slice(0, n_val)
    print(f"[train] {len(theta) - n_val} training + {n_val} validation simulations; noise = {args.noise}, features = {args.features}", flush=True)

    def augment(batch):
        """Per batch: fresh noise realisation of the likelihood (and fresh nuisance parameters)."""
        rows = batch["row"][:, 0]  # the dataset only carries row numbers: bayesflow's per-batch np.take over the
        th = theta[rows]  # big arrays costs ~1 s per batch, indexing them here ~1 ms
        if not args.keep_nuisance:
            th = nm.resample_nuisance(th, rng)
        if args.noise == "particles":
            mu, vmod = nm.stars_from_particles(sims["pool"], sims["n"], sims["slope"], sims["mphi"], rng, rows), 0.0
        else:
            mu, vmod = sims["mu"][rows], sims["vmod"][rows]
        y = nm.draw(th, mu, vmod, rng)
        return {"u": nm.to_u(th).astype(np.float32), "x": nm.featurize(y)}

    row = np.arange(N)[:, None]
    train = {"row": row[tr]}
    val = augment({"row": row[va]})
    x_obs = nm.featurize(nm.y[None].astype(np.float32))
    print(f"[train] input dimension {x_obs.shape[1]}, {theta.shape[1]} parameters", flush=True)

    class MLPSummary(bf.networks.SummaryNetwork):
        def __init__(self, widths, summary_dim, dropout, **kwargs):
            super().__init__(**kwargs)
            self.mlp = bf.networks.MLP(widths, dropout=dropout)
            self.head = keras.layers.Dense(summary_dim)

        def call(self, x, training=False, **kwargs):
            return self.head(self.mlp(x, training=training))

    adapter = bf.Adapter().keep(["u", "x"]).rename("u", "inference_variables").rename("x", "summary_variables")
    workflow = bf.BasicWorkflow(
        adapter=adapter,
        inference_network=bf.networks.CouplingFlow(
            depth=args.depth, transform=args.transform, subnet_kwargs=dict(widths=(args.width, args.width), dropout=0.0)),
        summary_network=MLPSummary(tuple(args.summary_widths), args.summary_dim, args.dropout),
        initial_learning_rate=args.lr,
        standardize=["inference_variables", "summary_variables"],
    )
    t0 = time.time()
    best = out / "approximator.weights.h5"  # the weights of the best validation epoch (long runs overfit:
    ckpt = keras.callbacks.ModelCheckpoint(str(best), monitor="val_loss", save_best_only=True, save_weights_only=True)
    hist = workflow.fit_offline(train, epochs=args.epochs, batch_size=args.batch_size, validation_data=val,
                                augmentations=augment, verbose=2, callbacks=[ckpt])  # the bin model after ~360)
    hours = (time.time() - t0) / 3600
    h = {k: [float(v) for v in vs] for k, vs in hist.history.items()}
    (out / "history.json").write_text(json.dumps(h))
    if not np.isfinite(h["loss"][-1]):
        sys.exit("[train] the loss is not finite: no posterior written (try --transform affine or a lower --lr)")
    best_epoch = int(np.argmin(h["val_loss"])) + 1
    workflow.approximator.load_weights(str(best))
    print(f"[train] best validation loss {min(h['val_loss']):.4f} at epoch {best_epoch}: using those weights", flush=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for k in ("loss", "val_loss"):
        if k in h:
            ax.plot(h[k], label=k)
    ax.set(xlabel="epoch", ylabel="negative log posterior density (standardised)")
    ax.legend()
    fig.savefig(out / "loss.png", dpi=130, bbox_inches="tight")

    # ---- amortized diagnostics on held-out simulations -------------------------------------------
    try:
        k = min(500, n_val)
        post = workflow.sample(conditions={"x": val["x"][:k]}, num_samples=500)["u"]
        figs = {
            "recovery": bf.diagnostics.plots.recovery({"u": post}, {"u": val["u"][:k]}, variable_names=nm.names),
            "calibration_ecdf": bf.diagnostics.plots.calibration_ecdf(
                {"u": post}, {"u": val["u"][:k]}, variable_names=nm.names, difference=True),
        }
        for name, f in figs.items():
            f.savefig(out / f"val_{name}.png", dpi=110, bbox_inches="tight")
    except Exception as e:  # diagnostics must not cost the trained model
        print(f"[train] diagnostics skipped: {e!r}", flush=True)

    # ---- posterior for the observed stars, with its density ----------------------------------------
    u = np.concatenate([
        workflow.sample(conditions={"x": x_obs}, num_samples=m)["u"][0]
        for m in [args.n_post // 10] * 10
    ]).astype(np.float32)
    logq_u = np.concatenate([
        np.asarray(workflow.log_prob({"u": u[i:i + 5000], "x": np.repeat(x_obs, len(u[i:i + 5000]), 0)}))
        for i in range(0, len(u), 5000)
    ]).astype(np.float64)
    th = nm.to_theta(u)
    logq = logq_u + nm.log_du_dtheta(u)  # density in theta
    np.savez(out / "posterior_npe.npz", names=np.array(nm.names), theta=th, u=u, logq=logq, logprior=nm.logprior(th), x_obs=x_obs)
    info = dict(vars(args), train_hours=hours, n_train=len(theta) - n_val, n_val=n_val, input_dim=int(x_obs.shape[1]),
                final_loss=h["loss"][-1], best_epoch=best_epoch, best_val_loss=min(h["val_loss"]), final_val_loss=h.get("val_loss", [None])[-1])
    (out / "train_info.json").write_text(json.dumps(info, indent=1, default=str))
    print(f"[train] {args.epochs} epochs in {hours:.2f} h; {len(th)} posterior draws -> posterior_npe.npz", flush=True)


# ======================================================================================================
# 3) importance sampling
# ======================================================================================================
def trained_noise(out: Path) -> str:
    f = out / "train_info.json"
    return json.loads(f.read_text()).get("noise", "gaussian") if f.exists() else "gaussian"


def cmd_is_worker(args):
    ns = import_ns()
    import jax
    import jax.numpy as jnp

    out = Path(args.out)
    theta = np.load(out / "posterior_npe.npz")["theta"]
    idx = np.arange(len(theta))[args.worker :: args.n_workers]
    noise = trained_noise(out)  # the likelihood the model was trained on
    binned = noise == "nsbin"
    data = ns.bin_data(ns.load_data()) if binned else ns.load_data()
    if noise == "nsbspline":  # B-spline-track likelihood: the NS forward + bin model on the GPU, the Gaussian of
        z = np.load(out / "training_set.npz")  # the tracks in numpy (`NoiseModel.logpdf`, float64)
        nm = NoiseModel(obs_arrays(z), "bspline", tracks=True, ops=out / "bspline_ops.npz")
        edges = jnp.asarray(data["edges"])

        def one(th):
            obs, ok, _ = ns.simulate_stream(th)
            return (ok,) + tuple(ns.bin_model(obs, edges))

        g = jax.jit(jax.vmap(one))

        def f(th):
            ok, n, mphi, centre, slope, var = (np.asarray(a) for a in g(th))
            good = ok & np.all(np.isfinite(centre) & np.isfinite(slope) & np.isfinite(var), axis=(1, 2)) & np.all(n >= ns.N_MIN_BIN, axis=1)
            ll = np.full(len(good), -1e6)  # failed / under-sampled: vetoed, as in the NS likelihoods
            if good.any():
                mu, vmod = nm.per_star(n[good], mphi[good], centre[good], slope[good], var[good], dtype=np.float64)
                ll[good] = nm.logpdf(np.asarray(th)[good], mu, vmod, nm.y[None])
            return ll
    else:
        loglike, _ = ns.make_loglike(data, binned=binned)
        f = jax.jit(jax.vmap(loglike))
    print(f"likelihood: {noise}", flush=True)
    (out / "is_parts").mkdir(exist_ok=True)
    for c, i in enumerate(range(0, len(idx), args.batch)):
        path = out / "is_parts" / f"w{args.worker:02d}_c{c:05d}.npz"
        if path.exists():
            continue
        t0 = time.time()
        sel = idx[i : i + args.batch]
        pad = np.concatenate([sel, np.full(args.batch - len(sel), sel[-1])])  # fixed batch shape: one compilation
        ll = np.asarray(f(jnp.asarray(theta[pad])))[: len(sel)]
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, idx=sel, logl=ll)
        os.replace(tmp, path)
        print(f"chunk {c} ({i + len(sel)}/{len(idx)}) {time.time() - t0:.1f}s", flush=True)


def wquantile(x, w, q):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return np.interp(q, (c - 0.5 * w[o]) / c[-1], x[o])


def cmd_is(args):
    out = Path(args.out)
    z = np.load(out / "posterior_npe.npz")
    theta, logq, logp, names = z["theta"], z["logq"], z["logprior"], [str(s) for s in z["names"]]
    n = len(theta)
    done = sum(len(np.load(f)["idx"]) for f in (out / "is_parts").glob("w*_c*.npz")) if (out / "is_parts").exists() else 0
    if done < n:
        run_workers("is-worker", out, args.gpus, ["--batch", str(args.batch)])
    logl = np.full(n, np.nan)
    for f in (out / "is_parts").glob("w*_c*.npz"):
        p = np.load(f)
        logl[p["idx"]] = p["logl"]
    assert np.all(np.isfinite(logl)), "missing likelihood evaluations"
    veto = logl < -1e5  # failed / under-sampled models: zero weight (as in the nested sampling)
    logw = np.where(veto | ~np.isfinite(logp), -np.inf, logl + logp - logq)
    m = logw.max()
    w = np.exp(logw - m)
    ess = w.sum() ** 2 / np.sum(w**2)
    wn = w / w.sum()
    logz = m + np.log(w.mean())  # evidence: E_q[L pi / q]
    logz_err = float(np.sqrt(np.sum((wn - 1.0 / n) ** 2)))  # delta-method standard error of log mean(w)
    rng = np.random.default_rng(0)
    res = theta[rng.choice(n, size=n, p=wn)]  # equal-weight draws of the reweighted posterior
    np.savez(out / "posterior_is.npz", names=np.array(names), theta=theta, logq=logq, logprior=logp, logl=logl,
             logw=logw, weight=wn, theta_resampled=res)
    qs = [0.16, 0.5, 0.84]
    table = {}
    for k, name in enumerate(names):
        table[name] = dict(npe=[float(v) for v in np.quantile(theta[:, k], qs)],
                           is_reweighted=[float(v) for v in wquantile(theta[:, k], wn, qs)])
    summary = dict(n_samples=n, ess=float(ess), ess_fraction=float(ess / n), logZ_is=float(logz), logZ_is_err=logz_err,
                   n_vetoed=int(veto.sum()), n_outside_prior=int((~np.isfinite(logp)).sum()), max_weight=float(wn.max()))
    noise = trained_noise(out)
    summary["likelihood"] = {"nsbin": "NS bin (make_loglike binned=True)",
                             "nsbspline": "B-spline tracks of Gaussian stars (NoiseModel.logpdf, tracks=True)"}.get(noise, "NS star (make_loglike)")
    summary["likelihood_is_the_training_likelihood"] = noise != "particles"
    summary["logl_pct_2.5_50_97.5_max"] = [float(v) for v in np.percentile(logl[~veto], [2.5, 50, 97.5, 100])]

    # ---- nested-sampling posterior, if it has been post-processed ---------------------------------
    ns_theta = ns_w = None
    ns_dir = NS_OUT / {"nsbin": "binned_likelihood", "nsbspline": "bspline_likelihood"}.get(noise, "")  # the NS run with the same likelihood
    if (ns_dir / "posterior.npz").exists():
        p = np.load(ns_dir / "posterior.npz")
        ns_theta, ns_w = p["theta"], np.exp(p["logw"] - p["logw"].max())
        ns_w /= ns_w.sum()
        for k, name in enumerate(names):
            table[name]["nested_sampling"] = [float(v) for v in wquantile(ns_theta[:, k], ns_w, qs)]
        if (ns_dir / "summary.json").exists():
            summary["nested_sampling_summary"] = {k: v for k, v in json.loads((ns_dir / "summary.json").read_text()).items()
                                                  if "logz" in k.lower() or "ess" in k.lower()}
    summary["quantiles_16_50_84"] = table
    (out / "is_summary.json").write_text(json.dumps(summary, indent=1))

    print(f"[is] likelihood: {summary['likelihood']}; trained with --noise {noise}"
          + ("" if noise != "particles" else " (NOT this likelihood: the weights target a different posterior)"))
    print(f"[is] N = {n}, ESS = {ess:.1f} ({100 * ess / n:.2f} %), logZ = {logz:.3f} +- {logz_err:.3f}, "
          f"vetoed {int(veto.sum())}, max weight {wn.max():.4f}")
    print(f"{'parameter':14s} {'NPE median [16, 84]':>34s} {'IS-reweighted':>34s}" + (f" {'nested sampling':>34s}" if ns_theta is not None else ""))
    fmt = lambda v: f"{v[1]:10.4g} [{v[0]:9.4g}, {v[2]:9.4g}]"
    for name in names:
        t = table[name]
        print(f"{name:14s} {fmt(t['npe']):>34s} {fmt(t['is_reweighted']):>34s}" + (f" {fmt(t['nested_sampling']):>34s}" if ns_theta is not None else ""))

    try:
        import corner
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.lines as mlines

        z0 = np.load(out / "training_set.npz")
        a, b, ku = z0["prior_a"], z0["prior_b"], z0["kind_u"].astype(bool)
        rngs = [(a[k], b[k]) if ku[k] else (a[k] - 4 * b[k], a[k] + 4 * b[k]) for k in range(len(names))]
        kw = dict(labels=names, range=rngs, bins=30, smooth=1.0, plot_datapoints=False, plot_density=False,
                  levels=(0.393, 0.865), hist_kwargs=dict(density=True))
        hk = lambda: dict(hist_kwargs=dict(density=True))  # corner stores the colour in hist_kwargs: one copy per call
        kw.pop("hist_kwargs")
        fig = corner.corner(theta, color="tab:blue", **kw, **hk())
        corner.corner(theta, weights=wn, color="tab:red", fig=fig, **kw, **hk())
        handles = [mlines.Line2D([], [], color="tab:blue", label="normalizing flow (NPE)"),
                   mlines.Line2D([], [], color="tab:red", label=f"importance reweighted (ESS {ess:.0f}/{n})")]
        if ns_theta is not None:
            corner.corner(ns_theta, weights=ns_w, color="k", fig=fig, **kw, **hk())
            handles.append(mlines.Line2D([], [], color="k", label="nested sampling"))
        fig.legend(handles=handles, loc="upper right", fontsize=18)
        fig.savefig(out / "corner_npe_is.png", dpi=80, bbox_inches="tight")
    except Exception as e:
        print(f"[is] corner plot skipped: {e!r}")


# ======================================================================================================
def ns_posterior(ns_dir: Path):
    """Weighted posterior of the nested-sampling runs in `ns_dir` (finished or not: an unfinished run contributes its
    dead points and its current live points). Runs are merged through their birth/death contours; the prior volume
    shrinks by its expectation exp(-1/n_live) per point (numpy only, no blackjax). Returns theta, weights, summary."""
    from scipy.special import logsumexp

    def weights(logl, birth):
        o = np.argsort(logl, kind="stable")
        birth = np.where(np.isnan(birth), -np.inf, birth)  # the initial live points are stored with a NaN birth contour
        ll, n_live = logl[o], np.searchsorted(np.sort(birth), logl[o], side="left") - np.arange(len(logl))
        n_live = np.maximum(n_live, 1)
        logx = -np.cumsum(1.0 / n_live)
        lw = ll + np.concatenate([[0.0], logx[:-1]]) + np.log1p(-np.exp(-1.0 / n_live))
        out = np.empty(len(logl))
        out[o] = lw
        return out

    runs, tags = [], set()
    for f in sorted(ns_dir.glob("run*_final.npz")) + sorted(ns_dir.glob("run*_ckpt.npz")):
        tag = f.name.split("_")[0]
        if tag in tags:
            continue
        tags.add(tag)
        z = np.load(f)
        fin = "theta" in z.files
        get = (lambda k: z[k]) if fin else (lambda k: np.concatenate([z["dead_" + k], z["live_" + k]]))
        runs.append(dict(tag=tag, finished=fin, theta=get("theta"), logl=get("logl"), birth=get("logl_birth")))
    if not runs:
        return None
    for r in runs:
        r["logZ"] = float(logsumexp(weights(r["logl"], r["birth"])))
    theta, logl = np.concatenate([r["theta"] for r in runs]), np.concatenate([r["logl"] for r in runs])
    lw = weights(logl, np.concatenate([r["birth"] for r in runs]))
    w = np.exp(lw - lw.max())
    w /= w.sum()
    lz = np.array([r["logZ"] for r in runs])
    info = dict(n_runs=len(runs), n_finished=int(sum(r["finished"] for r in runs)), n_points=int(len(w)),
                logZ=float(logsumexp(lw)), logZ_runs=[round(float(v), 3) for v in lz],
                logZ_run_scatter=float(lz.std(ddof=1)) if len(lz) > 1 else float("nan"), ess=float(1.0 / np.sum(w**2)),
                max_logL=float(logl.max()))
    return theta, w, info


REPORT_RUNS = [  # likelihood, title, NS folder (relative to NS_OUT), SBI folder (relative to OUT) with the same likelihood
    ("star", "per-star likelihood (593 values, outlier mixture)", "", "stars_1000ep"),
    ("bin", "bin likelihood (50 bin centres)", "binned_likelihood", "nsbin_300ep"),
    ("bspline", "B-spline likelihood (70 track coordinates)", "bspline_likelihood", "nsbspline_1000ep"),
]


def cmd_report(args):
    """Nested sampling vs importance-reweighted flow, one block per likelihood: figures + tables in <out>/report."""
    import corner
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt

    root, rep = Path(args.out), Path(args.out) / "report"
    rep.mkdir(parents=True, exist_ok=True)
    qs, min_ess = [0.16, 0.5, 0.84], 50.0  # below this ESS the reweighted sample is a handful of points: not drawn
    blocks, names = [], None
    for lik, title, ns_sub, run in REPORT_RUNS:
        p = np.load(root / run / "posterior_is.npz")
        names = [str(s) for s in p["names"]]
        s = json.loads((root / run / "is_summary.json").read_text())
        t = json.loads((root / run / "train_info.json").read_text())
        ok = np.all(np.isfinite(p["theta"]), axis=1)
        b = dict(lik=lik, title=title, run=run, npe=p["theta"][ok], is_w=p["weight"][ok] / p["weight"][ok].sum(),
                 ess=s["ess"], logz_is=s["logZ_is"], logz_is_err=s["logZ_is_err"], epochs=t["epochs"],
                 n_draws=int(len(p["theta"])), max_weight=s["max_weight"], ns=ns_posterior(NS_OUT / ns_sub))
        b["sets"] = [("nested sampling", "k", b["ns"][0], b["ns"][1])] if b["ns"] else []
        b["sets"].append(("normalizing flow (NPE)", "tab:blue", b["npe"], np.full(len(b["npe"]), 1.0 / len(b["npe"]))))
        if b["ess"] >= min_ess:
            b["sets"].append((f"flow, importance reweighted (ESS {b['ess']:.0f})", "tab:red", b["npe"], b["is_w"]))
        blocks.append(b)
    nd = len(names)
    lo = np.array([min(wquantile(x[:, k], w, 0.002) for b in blocks for _, _, x, w in b["sets"]) for k in range(nd)])
    hi = np.array([max(wquantile(x[:, k], w, 0.998) for b in blocks for _, _, x, w in b["sets"]) for k in range(nd)])
    pad = 0.05 * (hi - lo)
    rng_ = [(lo[k] - pad[k], hi[k] + pad[k]) for k in range(nd)]

    # ---- 1-d marginals: two rows of eight parameters per likelihood, the same axis range in every block -------
    ncol = 8
    nrow = int(np.ceil(nd / ncol))
    fig, axs = plt.subplots(nrow * len(blocks), ncol, figsize=(2.6 * ncol, 2.1 * nrow * len(blocks) + 1.2))
    for ib, b in enumerate(blocks):
        for k in range(nd):
            ax = axs[ib * nrow + k // ncol, k % ncol]
            for lab, col, x, w in b["sets"]:
                h, e = np.histogram(x[:, k], bins=40, range=rng_[k], weights=w, density=True)
                ax.stairs(h, e, color=col, lw=1.3 if col != "k" else 1.8)
            ax.set_yticks([])
            ax.set_xlabel(names[k], fontsize=9)
            ax.tick_params(labelsize=7)
            if k == 0:
                ax.set_title(f"{b['title']}  |  SBI run {b['run']}", fontsize=11, loc="left", fontweight="bold")
    handles = [mlines.Line2D([], [], color="k", lw=1.8, label="nested sampling"),
               mlines.Line2D([], [], color="tab:blue", label="normalizing flow (NPE)"),
               mlines.Line2D([], [], color="tab:red", label="flow, importance reweighted")]
    fig.legend(handles=handles, loc="upper right", ncol=3, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(rep / "marginals_by_likelihood.png", dpi=110)
    plt.close(fig)

    # ---- median and 68 % interval of every parameter, all likelihoods and methods side by side ------------------
    fig, axs = plt.subplots(4, 4, figsize=(15, 11))
    for k, ax in enumerate(axs.ravel()):
        y, ticks = 0, []
        for b in blocks:
            for lab, col, x, w in b["sets"]:
                q = wquantile(x[:, k], w, qs)
                ax.errorbar(q[1], y, xerr=[[q[1] - q[0]], [q[2] - q[1]]], fmt="o", color=col, ms=4, capsize=2)
                y -= 1
            ticks.append((y + 0.5 * len(b["sets"]) + 0.5, b["lik"]))
            ax.axhline(y, color="0.85", lw=0.8)
            y -= 1
        ax.set_yticks([t_[0] for t_ in ticks], [t_[1] for t_ in ticks], fontsize=8)
        ax.set_title(names[k], fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
    fig.legend(handles=handles, loc="upper right", ncol=3, fontsize=11)
    fig.suptitle("median and 16-84 % interval", x=0.02, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(rep / "intervals_by_likelihood.png", dpi=110)
    plt.close(fig)

    # ---- corner plot per likelihood (halo, orbit and progenitor parameters) -------------------------------------
    sel = [names.index(n_) for n_ in ("log10_m_nfw", "r_s", "q", "dist", "pmra", "pmdec", "vr", "log10_m_prog", "a_prog", "t_end")]
    for b in blocks:
        fig = None
        kw = dict(labels=[names[k] for k in sel], range=[rng_[k] for k in sel], bins=30, smooth=1.0, plot_datapoints=False,
                  plot_density=False, levels=(0.393, 0.865))
        for lab, col, x, w in b["sets"][::-1]:  # corner stores the colour in hist_kwargs: a fresh dict per call
            fig = corner.corner(x[:, sel], weights=w, color=col, fig=fig, hist_kwargs=dict(density=True), **kw)
        fig.legend(handles=[mlines.Line2D([], [], color=col, label=lab) for lab, col, _, _ in b["sets"]], loc="upper right", fontsize=16)
        fig.suptitle(b["title"], x=0.02, y=1.02, ha="left", fontsize=20)
        fig.savefig(rep / f"corner_{b['lik']}.png", dpi=70, bbox_inches="tight")
        plt.close(fig)

    # ---- tables --------------------------------------------------------------------------------------------------
    cell = lambda q: f"{q[1]:.4g} [{q[0]:.4g}, {q[2]:.4g}]"
    lines = ["| likelihood | SBI run | epochs | ESS of the flow draws | max weight | logZ importance sampling | logZ nested sampling "
             "| per-run scatter | NS runs finished | NS posterior ESS |", "|---|---|---|---|---|---|---|---|---|---|"]
    for b in blocks:
        n = b["ns"][2] if b["ns"] else None
        lines.append(f"| {b['lik']} | `{b['run']}` | {b['epochs']} | {b['ess']:.1f} of {b['n_draws']} | {b['max_weight']:.3f} | "
                     f"{b['logz_is']:.2f} ± {b['logz_is_err']:.2f} | " + (f"{n['logZ']:.2f} | {n['logZ_run_scatter']:.2f} | "
                     f"{n['n_finished']} of {n['n_runs']} | {n['ess']:.0f} |" if n else "– | – | – | – |"))
    for b in blocks:
        lines += ["", f"### {b['title']}", "", "| parameter | " + " | ".join(lab for lab, *_ in b["sets"]) + " |",
                  "|---|" + "---|" * len(b["sets"])]
        for k in range(nd):
            lines.append(f"| {names[k]} | " + " | ".join(cell(wquantile(x[:, k], w, qs)) for _, _, x, w in b["sets"]) + " |")
    (rep / "tables.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    for b in blocks:
        if b["ns"]:
            print(b["lik"], b["ns"][2])
    print(f"wrote {rep}")


def cmd_all(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gcmd, genv = galax_cmd()
    common = ["--out", str(out)]
    stages = [
        ("simulate", gcmd, genv, ["--gpus", str(args.gpus), "--n-sims", str(args.n_sims), "--batch", str(args.batch),
                                 "--pool", str(args.pool)],
         out / "training_set.npz"),
        ("train", [str(BF_PYTHON)], dict(os.environ), ["--epochs", str(args.epochs), "--features", args.features, "--noise", args.noise],
         out / "posterior_npe.npz"),
        ("is", gcmd, genv, ["--gpus", str(args.gpus), "--batch", str(args.batch)], out / "is_summary.json"),
    ]
    for name, cmd, env, extra, product in stages:
        if product.exists():
            print(f"[all] {name}: {product.name} exists, skipping", flush=True)
            continue
        print(f"[all] {name} ...", flush=True)
        code = subprocess.call(cmd + [__file__, name] + common + extra, env=env)
        if code != 0:
            sys.exit(f"[all] stage {name} failed ({code})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    cmds = {"all": cmd_all, "simulate": cmd_simulate, "sim-worker": cmd_sim_worker, "train": cmd_train, "is": cmd_is,
            "is-worker": cmd_is_worker, "bin-data": cmd_bin_data, "check": cmd_check, "report": cmd_report}
    for name in cmds:
        sp = sub.add_parser(name)
        sp.add_argument("--out", default=str(OUT))
        sp.add_argument("--gpus", type=int, default=8, help="free GPUs used by simulate / is")
        sp.add_argument("--n-sims", type=int, default=200_000, help="prior draws kept (not vetoed by the likelihood)")
        sp.add_argument("--batch", type=int, default=64, help="streams per GPU batch")
        sp.add_argument("--worker", type=int, default=0)
        sp.add_argument("--n-workers", type=int, default=1)
        # training
        sp.add_argument("--epochs", type=int, default=200)
        sp.add_argument("--batch-size", type=int, default=512)
        sp.add_argument("--lr", type=float, default=5e-4)
        sp.add_argument("--pool", type=int, default=512, help="stream particles stored per phi1 bin")
        sp.add_argument("--features", choices=["stars", "bins", "bspline"], default="stars",
                        help="network input: per-bin mean and scatter of the mock stars | every star's values")
        sp.add_argument("--noise", choices=["gaussian", "nsbin", "nsbspline", "particles"], default="gaussian",
                        help="NS star likelihood | NS bin likelihood (bin centres) | B-spline tracks of Gaussian stars "
                             "(no outliers) | stream particles + errors")
        sp.add_argument("--keep-nuisance", action="store_true", help="do not re-draw sigma_inflate / f_out per batch")
        sp.add_argument("--summary-widths", type=int, nargs="+", default=[512, 512, 256])
        sp.add_argument("--summary-dim", type=int, default=64)
        sp.add_argument("--dropout", type=float, default=0.05)
        sp.add_argument("--depth", type=int, default=8, help="coupling layers")
        sp.add_argument("--width", type=int, default=256, help="coupling subnet width")
        sp.add_argument("--transform", choices=["affine", "spline"], default="affine", help="coupling transform")
        sp.add_argument("--n-val", type=int, default=5000)
        sp.add_argument("--n-train-max", type=int, default=10**9)
        sp.add_argument("--n-post", type=int, default=20_000, help="posterior draws for the observed data")
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    cmds[args.mode](args)


if __name__ == "__main__":
    main()

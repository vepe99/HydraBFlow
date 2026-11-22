#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "galax==0.1.0",
#     "jax[cuda12]==0.12.1",
#     "xmmutablemap==0.2.2",
#     "unxt==2.2.0",
#     "coordinax==0.24.3",
#     "quax==0.4.6",
#     "quaxed==0.10.7",
#     "equinox==0.13.8",
#     "diffrax==0.7.2",
#     "plum-dispatch==2.10.1",
#     "dataclassish==0.9.2",
#     "jaxtyping==0.3.11",
#     "numpy==2.5.5",
#     "scipy==1.18.1",
#     "h5py==3.16.0",
#     "matplotlib==3.11.2",
#     "astropy==8.0.3",
#     "autocvd==0.2.1",
#     "StreaMAX @ git+https://github.com/David-Chemaly/StreaMAX.git@9fd520ea2904b617c701e1ee5b8740a9a0fd148e",
# ]
#
# [tool.uv]
# exclude-newer = "2026-11-20T00:00:00Z"
# ///
"""Extragalactic stellar-stream tracks for simulation-based inference (Chemaly et al. 2026a/b forward model).

One file, run with uv (the header above is the environment):

    uv run --script scripts/extragalactic/generate.py test     # step 4: unit / symmetry / StreaMAX checks
    uv run --script scripts/extragalactic/generate.py pilot    # step 5: 1,000 streams on one GPU
    uv run --script scripts/extragalactic/generate.py run      # step 6: 100,000 streams, one worker per free GPU
    uv run --script scripts/extragalactic/generate.py analyze  # step 7: per-stream diagnostics + the summary table

Add ``--cpu`` to any of them to stay off the GPUs (slow; meant for smoke tests). Every field of `Config`
is a command-line option (``--n-streams 2000 --progenitor-mass-loss linear`` ...).

Forward model (mirrors StreaMAX @ 9fd520e, the code of C26 = arXiv:2601.15373 and 2607.05510):
  * host: axisymmetric NFW, flattening q in the DENSITY along the unit vector n (Chandrasekhar integral with the
    same 12-node Gauss-Legendre rule as StreaMAX `NFWFlatDensity`); M is the NFW scale mass 4 pi rho0 R_s^3
    (Phi = -G M ln(1 + r/R_s) / r at q = 1). `halo_flattening=potential` gives the legacy StreaMAX `NFW`.
  * progenitor: galax `PlummerPotential`, constant mass (`progenitor_mass_loss=none`) or StreaMAX's ramp
    (`linear`: log10 m goes linearly in time from log10 m_prog to 0, i.e. to 1 Msun).
  * spray: galax `FardalStreamDF` (same k-constants as StreaMAX: 2.0/0.3 means, 0.5 dispersions), tidal radius
    (G m / (Omega^2 - d2Phi/dr2))^(1/3); StreaMAX release schedule = n_steps + 1 epochs uniform over
    [-t_int, 0] (both ends included), n_particles / (n_steps + 1) particles per epoch alternating between the
    outer (trailing) and inner (leading) Lagrange point.
  * integration: kick-drift-kick leapfrog, n_steps equal steps per particle from its release to today
    (step = remaining time / n_steps); the progenitor is re-integrated from each release epoch with that
    epoch's step, exactly like StreaMAX `precompute_prog_trajectories`.
  * track: projection on XY, per-particle unwrapped angle (C26 sec. 2.2), fixed 10 deg bins on [-1440, 1440).

galax extensions written here (everything else is galax): `OrientedNFWPotential` (galax has a density-
flattened NFW but only along z and with a different quadrature variable), the fixed-step KDK leapfrog `kdk`
(galax integrates with diffrax solvers, none of which is a KDK leapfrog with a per-particle step), and the
StreaMAX release schedule around galax's `FardalStreamDF`.
"""

import argparse
import dataclasses
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STREAMAX_COMMIT = "9fd520ea2904b617c701e1ee5b8740a9a0fd148e"


# ======================================================================================================
# Settings (one dataclass; every field is a CLI option)
# ======================================================================================================
@dataclass(frozen=True)
class Config:
    out: str = "outputs/extragalactic/C26_test"  # relative to the project root
    seed: int = 2601  # master key; stream k uses jax.random.fold_in(key(seed), k)
    n_streams: int = 100_000
    shard_size: int = 1000
    # --- forward model
    n_particles: int = 10_000
    n_steps: int = 99  # leapfrog steps per particle; n_steps + 1 release epochs (StreaMAX-fit default)
    halo_flattening: str = "density"  # density (NFWFlatDensity) | potential (legacy NFW)
    progenitor_mass_loss: str = "none"  # none (C26) | linear (2026b, streams with no visible progenitor)
    # --- prior (C26 table 1)
    log10_M_halo: tuple = (11.0, 14.0)
    R_s: tuple = (10.0, 25.0)  # kpc
    q: tuple = (0.5, 1.5)
    log10_m_prog: tuple = (7.0, 9.0)
    r_s: tuple = (1.0, 5.0)  # kpc
    sigma_pos: float = 150.0  # kpc      x0, z0 ~ |N(0, sigma_pos)|, y0 = 0
    sigma_vel: float = 250.0  # km/s     vx0, vz0 ~ N(0, sigma_vel), vy0 ~ |N(0, sigma_vel)|
    t_int: tuple = (1.0, 4.0)  # Gyr
    # --- track
    track_ordering: str = "c26"  # c26 (per-particle unwrapping) | gibbons14 (StreaMAX get_stream_ordered)
    bin_deg: float = 10.0
    phi_max_deg: float = 1440.0
    # --- diagnostics / selection (C26 sec. 2.3)
    n_valid_min: int = 100  # particles for a bin to be valid
    sel_ang_deg: float = 90.0
    sel_arc_kpc: float = 100.0
    sel_r_min: float = 10.0
    sel_r_max: float = 500.0
    n_fine: int = 2000  # steps of the reference progenitor orbit used by the diagnostics
    # --- running
    batch_size: int = 0  # 0 = choose from a memory probe (GPU) / 8 (CPU)
    max_batch: int = 256
    mem_fraction: float = 0.6  # of the device memory the probe may plan for
    n_save_particles: int = 5000  # projected particles are stored for the first this-many streams
    pilot_n: int = 1000
    subdir: str = ""  # data/<subdir>; the pilot writes to data/pilot
    gpus: str = ""  # comma-separated GPU indices/UUIDs to use instead of the free ones
    cpu: bool = False
    overwrite: bool = False
    # --- tests
    n_energy: int = 200
    n_test_streams: int = 32  # prior draws simulated by `test` to find bound streams for the StreaMAX check
    test_batch: int = 8

    @property
    def n_bins(self) -> int:
        return int(round(2 * self.phi_max_deg / self.bin_deg))

    @property
    def n_epochs(self) -> int:
        return self.n_steps + 1

    def physics_hash(self) -> str:
        keys = ("seed n_particles n_steps halo_flattening progenitor_mass_loss log10_M_halo R_s q log10_m_prog "
                "r_s sigma_pos sigma_vel t_int track_ordering bin_deg phi_max_deg n_fine").split()
        blob = json.dumps({k: getattr(self, k) for k in keys}, sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]

    def check(self) -> None:
        ne = self.n_epochs
        if self.n_particles % ne or (self.n_particles // ne) % 2:
            raise SystemExit(
                f"n_particles ({self.n_particles}) must be an even multiple of n_steps + 1 ({ne}): StreaMAX "
                f"releases n_particles / (n_steps + 1) particles per epoch, half from each Lagrange point."
            )
        assert self.halo_flattening in ("density", "potential")
        assert self.progenitor_mass_loss in ("none", "linear")
        assert self.track_ordering in ("c26", "gibbons14")


THETA_NAMES = ["log10_M_halo", "R_s", "q", "cos_i", "psi", "log10_m_prog", "r_s",
               "x0", "z0", "vx0", "vy0", "vz0", "t_int"]

# status bits
ST_UNBOUND, ST_BAD_PARTICLES, ST_TIDAL, ST_OVERFLOW, ST_FAILED, ST_NONMONO, ST_NO_VALID = 1, 2, 4, 8, 16, 32, 64
STATUS_BITS = {
    "unbound": ST_UNBOUND,  # progenitor energy today >= 0
    "bad_particles": ST_BAD_PARTICLES,  # some particles are non-finite (excluded from the track)
    "tidal_undefined": ST_TIDAL,  # Omega^2 - d2Phi/dr2 <= 0 at some release epoch (StreaMAX: NaN)
    "grid_overflow": ST_OVERFLOW,  # some particles fall outside [-phi_max, phi_max)
    "failed": ST_FAILED,  # non-finite progenitor orbit or no finite particle at all
    "prog_angle_nonmonotonic": ST_NONMONO,  # projected progenitor angle not monotonic (C26 assumption)
    "no_valid_bin": ST_NO_VALID,  # no bin with >= n_valid_min particles
}


# ======================================================================================================
# Devices: must run before jax is imported
# ======================================================================================================
def free_gpus() -> list[str]:
    """UUIDs of every GPU that is free right now.

    autocvd has no call that returns *all* free devices (`autocvd(num_gpus=N)` waits for exactly N), so the
    devices are listed with nvidia-smi and filtered with autocvd's own test, `gpu_is_free` (= no compute
    process on the card). Devices are addressed by UUID because a broken card (compgpu6 has one) shifts the
    CUDA indices relative to nvidia-smi's. Under Slurm the allocation (CUDA_VISIBLE_DEVICES) is used as is.
    """
    if os.environ.get("SLURM_JOB_ID") and os.environ.get("CUDA_VISIBLE_DEVICES"):
        return [g for g in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if g]
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout
    except FileNotFoundError:  # no NVIDIA driver on this machine: no GPU to offer
        return []
    rows = [m.groups() for m in (re.match(r"\s*(\d+),\s*(GPU-\S+)", ln) for ln in out.splitlines()) if m]
    try:
        from autocvd.nvidia_smi_calls import gpu_is_free
    except Exception:  # autocvd missing: same test by hand
        busy = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.split()
        return [uuid for _, uuid in rows if uuid not in busy]
    free = []
    for idx, uuid in rows:
        try:
            if gpu_is_free(int(idx)):
                free.append(uuid)
        except Exception:  # a card nvidia-smi cannot query is not usable
            pass
    return free


def _early_device_setup() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if "--cpu" in sys.argv or cmd in ("run", "analyze", "streamax", ""):
        os.environ["JAX_PLATFORMS"] = "cpu"  # launchers and CPU runs never touch a GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif cmd in ("test", "pilot"):
        gpus = _cli_gpus() or free_gpus()
        if not gpus:
            raise SystemExit("no free GPU right now; wait, or pass --cpu for a (slow) CPU run")
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus[0]
    # worker: CUDA_VISIBLE_DEVICES was set by the launcher


def _cli_gpus() -> list[str]:
    for i, a in enumerate(sys.argv):
        if a == "--gpus" and i + 1 < len(sys.argv):
            return [g for g in sys.argv[i + 1].split(",") if g]
        if a.startswith("--gpus="):
            return [g for g in a.split("=", 1)[1].split(",") if g]
    return []


if __name__ == "__main__":
    _early_device_setup()

import warnings  # noqa: E402

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

if not (len(sys.argv) > 1 and sys.argv[1] == "streamax"):  # StreaMAX forces float32: it gets its own process
    import equinox as eqx
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import jax.random as jr
    import unxt as u
    from unxt.quantity import AllowValue
    from xmmutablemap import ImmutableMap

    import galax.dynamics as gd
    import galax.potential as gp
    from galax.potential._src.base import default_constants

    # kpc, Gyr, Msun: G = 4.498502151469553e-06 kpc^3 / (Msun Gyr^2), identical to StreaMAX's constants.G
    USYS = u.unitsystem(u.unit("kpc"), u.unit("Gyr"), u.unit("Msun"), u.unit("rad"))

KPC_KM, GYR_S = 3.0856775814913673e16, 3.15576e16
KMS = GYR_S / KPC_KM  # km/s -> kpc/Gyr (= StreaMAX KMS_TO_KPCGYR)
EPS = 1e-12  # StreaMAX EPSILON
_gx, _gw = np.polynomial.legendre.leggauss(12)  # StreaMAX _N_GL_DENSFLAT
GL_V, GL_W = 0.5 * (_gx + 1.0), 0.5 * _gw


# ======================================================================================================
# galax extension 1: NFW flattened (in the density, or in the potential) along an arbitrary axis
# ======================================================================================================
def _nfw_phi(G, M, Rs, q, R2, Z, flatten: str):
    """Potential of the axisymmetric NFW at cylindrical (R^2, Z) about its symmetry axis. Mirrors StreaMAX
    `NFW_density_flat_potential` (flatten='density') / `NFW_potential` with a = b = 1, c = q ('potential')."""
    if flatten == "potential":
        r = jnp.sqrt(R2 + (Z / q) ** 2 + EPS)
        return -G * M * jnp.log(1.0 + r / Rs) / (r + EPS)
    q2 = q * q
    xi2 = ((R2 + EPS) / Rs**2)[..., None]
    zeta2 = ((Z / Rs) ** 2)[..., None]
    D = q2 + (1.0 - q2) * GL_V**2
    m = GL_V * jnp.sqrt(xi2 / D + zeta2 / q2 + EPS)
    return -(G * M * q2 / Rs) * jnp.sum(GL_W / ((1.0 + m) * D), axis=-1)


def _nfw_grad(G, M, Rs, q, n, xyz, flatten: str):
    """Analytic gradient of `_nfw_phi` (StreaMAX `NFW_density_flat_acceleration`, with the opposite sign)."""
    Z = jnp.sum(xyz * n, axis=-1, keepdims=True)
    perp = xyz - Z * n
    R2 = jnp.sum(perp * perp, axis=-1, keepdims=True)
    if flatten == "potential":
        r = jnp.sqrt(R2 + (Z / q) ** 2 + EPS)
        dphi_dr = G * M * (jnp.log(1.0 + r / Rs) / r**2 - 1.0 / (r * (Rs + r)))
        return dphi_dr * (perp + Z * n / q**2) / r
    q2 = q * q
    R = jnp.sqrt(R2 + EPS)
    xi, zeta = R / Rs, Z / Rs
    D = q2 + (1.0 - q2) * GL_V**2
    m = GL_V * jnp.sqrt(xi**2 / D + zeta**2 / q2 + EPS)
    base = GL_V**2 / ((1.0 + m) ** 2 * m + EPS)
    dI_dxi = -xi * jnp.sum(GL_W * base / D**2, axis=-1, keepdims=True)
    dI_dzeta = -(zeta / q2) * jnp.sum(GL_W * base / D, axis=-1, keepdims=True)
    pref = -(G * M * q2 / Rs) / Rs
    return pref * dI_dxi * perp / R + pref * dI_dzeta * n


if "gp" in globals():

    class OrientedNFWPotential(gp.AbstractSinglePotential):
        r"""Axisymmetric NFW whose flattening `q` acts along the unit vector `axis`.

        flatten='density':   rho = M / (4 pi r_s^3) / [(m/r_s)(1 + m/r_s)^2],  m^2 = R^2 + Z^2 / q^2
        flatten='potential': Phi = -G M ln(1 + m/r_s) / m with the same m
        where Z = x . axis and R^2 = |x|^2 - Z^2. `m` is the NFW scale mass (not a virial mass).
        """

        m: gp.params.AbstractParameter = gp.params.ParameterField(dimensions="mass", doc="Scale mass.")  # type: ignore[assignment]
        r_s: gp.params.AbstractParameter = gp.params.ParameterField(dimensions="length", doc="Scale radius.")  # type: ignore[assignment]
        q: gp.params.AbstractParameter = gp.params.ParameterField(dimensions="dimensionless", doc="Axis ratio.")  # type: ignore[assignment]
        axis: jax.Array = eqx.field(converter=jnp.asarray)

        _: dataclasses.KW_ONLY
        units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
        constants: ImmutableMap = eqx.field(default=default_constants, converter=ImmutableMap)
        flatten: str = eqx.field(default="density", static=True)

        def _pars(self, t):
            t = u.Q.from_(t, self.units["time"])
            return (self.constants["G"].value, self.m(t, ustrip=self.units["mass"]),
                    self.r_s(t, ustrip=self.units["length"]), u.ustrip(AllowValue, u.unit(""), self.q(t)))

        def _potential(self, xyz, t, /):
            xyz = u.ustrip(AllowValue, self.units["length"], xyz)
            G, M, Rs, q = self._pars(t)
            Z = jnp.sum(xyz * self.axis, axis=-1)
            R2 = jnp.sum(xyz * xyz, axis=-1) - Z * Z
            return _nfw_phi(G, M, Rs, q, jnp.maximum(R2, 0.0), Z, self.flatten)

        def _gradient(self, xyz, t, /):
            xyz = u.ustrip(AllowValue, self.units["length"], xyz)
            G, M, Rs, q = self._pars(t)
            return _nfw_grad(G, M, Rs, q, self.axis, xyz, self.flatten)


# ======================================================================================================
# galax extension 2: kick-drift-kick leapfrog with a fixed number of steps
# ======================================================================================================
def kdk(acc, x, v, dt, n: int):
    """`n` KDK steps of size `dt` (may be negative, may be an array broadcasting against x). Returns the
    positions and velocities after every step, shapes (n, ...)."""

    def step(c, _):
        x, v, a = c
        vh = v + 0.5 * dt * a
        x = x + dt * vh
        a = acc(x)
        v = vh + 0.5 * dt * a
        return (x, v, a), (x, v)

    _, (xs, vs) = jax.lax.scan(step, (x, v, acc(x)), None, length=n)
    return xs, vs


def _wrap(a):
    """Angle into [-pi, pi)."""
    return (a + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


# ======================================================================================================
# Step 1: prior
# ======================================================================================================
def sample_prior(key, cfg: Config):
    """One draw of the 13 parameters (order = THETA_NAMES). Each parameter has its own sub-key."""
    k = [jr.fold_in(key, j) for j in range(len(THETA_NAMES))]
    U = lambda kk, ab: jr.uniform(kk, (), minval=ab[0], maxval=ab[1])
    N = lambda kk, s: s * jr.normal(kk, ())
    return jnp.stack([
        U(k[0], cfg.log10_M_halo), U(k[1], cfg.R_s), U(k[2], cfg.q),
        U(k[3], (0.0, 1.0)), U(k[4], (0.0, 2.0 * np.pi)),
        U(k[5], cfg.log10_m_prog), U(k[6], cfg.r_s),
        jnp.abs(N(k[7], cfg.sigma_pos)), jnp.abs(N(k[8], cfg.sigma_pos)),
        N(k[9], cfg.sigma_vel), jnp.abs(N(k[10], cfg.sigma_vel)), N(k[11], cfg.sigma_vel),
        U(k[12], cfg.t_int),
    ])  # fmt: skip


def stream_keys(cfg: Config, idx):
    """(prior key, spray key) of streams `idx`: both derive from fold_in(master, k) only."""
    master = jr.key(cfg.seed)
    kk = jax.vmap(lambda i: jr.fold_in(master, i))(jnp.asarray(idx, dtype=jnp.uint32))
    return jax.vmap(lambda a: jr.fold_in(a, 0))(kk), jax.vmap(lambda a: jr.fold_in(a, 1))(kk)


def draw_prior(cfg: Config, idx) -> np.ndarray:
    kp, _ = stream_keys(cfg, idx)
    return np.asarray(jax.vmap(lambda a: sample_prior(a, cfg))(kp))


def halo_axis(theta: np.ndarray) -> np.ndarray:
    """n = (sin i cos psi, sin i sin psi, cos i): the physical symmetry axis of the halo."""
    ci, psi = theta[..., 3], theta[..., 4]
    si = np.sqrt(1.0 - ci**2)
    return np.stack([si * np.cos(psi), si * np.sin(psi), ci], axis=-1)


def c26_vector(theta: np.ndarray, cfg: Config) -> np.ndarray:
    """The (xhat, yhat, zhat) that makes StreaMAX build this halo.

    Length: r = F^-1(u), u = (q - q_min) / (q_max - q_min), F = Maxwell(1) cdf = P(3/2, r^2/2).
    Direction: StreaMAX evaluates the halo-frame potential at R x with R = get_mat(dir) the rotation that
    takes z onto dir, so the symmetry axis of its halo is R^T z = (-dirx, -diry, dirz) / |dir| -- the sky-plane
    components are mirrored. Hence (xhat, yhat, zhat) = r * (-n_x, -n_y, n_z). (The prior is unchanged.)
    """
    from scipy.special import gammaincinv

    uq = (theta[..., 2] - cfg.q[0]) / (cfg.q[1] - cfg.q[0])
    r = np.sqrt(2.0 * gammaincinv(1.5, np.clip(uq, 0.0, 1.0)))
    return r[..., None] * halo_axis(theta) * np.array([-1.0, -1.0, 1.0])


def theta_to_inputs(theta: np.ndarray):
    """13 parameters -> simulator inputs (internal units kpc, kpc/Gyr, Gyr)."""
    z = np.zeros_like(theta[..., 0])
    wf = np.stack([theta[..., 7], z, theta[..., 8], theta[..., 9] * KMS, theta[..., 10] * KMS,
                   theta[..., 11] * KMS], axis=-1)
    return dict(logM=theta[..., 0], Rs=theta[..., 1], q=theta[..., 2], nvec=halo_axis(theta),
                logm=theta[..., 5], rs=theta[..., 6], wf=wf, t_int=theta[..., 12])


# ======================================================================================================
# Steps 2-3: simulator and track extraction
# ======================================================================================================
def bin_track(phi_deg, r, ok, cfg: Config):
    """Fixed bins of `bin_deg` on [-phi_max, phi_max): count, mean, std (ddof 0), median and the 16-84
    half-width (the two statistics StreaMAX `get_track_2D` returns). Empty bins are NaN."""
    nb, N = cfg.n_bins, r.shape[0]
    idx = jnp.floor((jnp.where(ok, phi_deg, 0.0) + cfg.phi_max_deg) / cfg.bin_deg).astype(jnp.int32)
    inr = ok & (idx >= 0) & (idx < nb)
    seg = jnp.where(inr, idx, nb)
    rs = jnp.where(inr, r, 0.0)
    count = jnp.zeros(nb + 1, jnp.int32).at[seg].add(1)
    cnt = jnp.maximum(count, 1)
    mean = jnp.zeros(nb + 1).at[seg].add(rs) / cnt
    std = jnp.sqrt(jnp.zeros(nb + 1).at[seg].add((rs - mean[seg]) ** 2) / cnt)
    r_sorted = rs[jnp.lexsort((rs, seg))]
    start = jnp.cumsum(count) - count

    def quantile(f):  # linear interpolation between order statistics, as jnp.nanquantile
        p = start + f * (count - 1)
        lo = jnp.floor(p)
        w = p - lo
        lo = jnp.clip(lo.astype(jnp.int32), 0, N - 1)
        hi = jnp.clip(jnp.minimum(lo + 1, start + count - 1), 0, N - 1)
        return r_sorted[lo] * (1.0 - w) + r_sorted[hi] * w

    empty = count[:nb] == 0
    nan = lambda a: jnp.where(empty, jnp.nan, a[:nb]).astype(jnp.float32)
    return dict(count=count[:nb], r_mean=nan(mean), r_std=nan(std), r_med=nan(quantile(0.5)),
                r_hw=nan(0.5 * (quantile(0.84) - quantile(0.16))),
                n_overflow=jnp.sum(ok & ~inr).astype(jnp.int32))


def gibbons14_angle(x, y, g, ok):
    """Hook for the 2026b ordering (Gibbons+14 sec. 3.4 as coded in StreaMAX `get_stream_ordered`): sort by
    g = sum over steps of (|r_prog| - |r_particle|), unwrap the polar angle along that order, zero it at the
    particle with the smallest |g|. Not validated against the 2026b fits."""
    gs = jnp.where(ok, g, jnp.inf)
    order = jnp.argsort(gs)
    th = jnp.arctan2(y, x)[order]
    d = jnp.where(ok[order][1:], _wrap(jnp.diff(th)), 0.0)
    thu = jnp.concatenate([th[:1], th[0] + jnp.cumsum(d)])
    thu = thu - thu[jnp.argmin(jnp.abs(gs[order]))]
    return jnp.zeros_like(thu).at[order].set(thu)


def make_simulator(cfg: Config):
    """Returns the jitted, vmapped simulator: batched inputs of `theta_to_inputs` + spray keys -> dict."""
    cfg.check()
    n, ne = cfg.n_steps, cfg.n_epochs
    n_per = cfg.n_particles // ne
    N = cfg.n_particles
    sign = jnp.tile(jnp.array([1.0, -1.0]), N // 2)[:, None]  # +1: outer point (trailing), -1: inner (leading)
    rep = lambda a: jnp.repeat(a, n_per, axis=0)
    df = gd.FardalStreamDF()

    def sim(logM, Rs, q, nvec, logm, rs, wf, t_int, key):
        host = OrientedNFWPotential(m=10.0**logM, r_s=Rs, q=q, axis=nvec, units=USYS, flatten=cfg.halo_flattening)
        prog = gp.PlummerPotential(m_tot=1.0, r_s=rs, units=USYS)  # linear in its mass: scaled by m(t) below
        acc_h = lambda x: -host._gradient(x, 0.0)
        energy = lambda x, v: 0.5 * jnp.sum(v * v, -1) + host._potential(x, 0.0)
        xf, vf = wf[:3], wf[3:]

        # --- progenitor: today -> -t_int, then forward on the n + 1 release epochs (StreaMAX does both)
        dt = t_int / n
        xb, vb = kdk(acc_h, xf, vf, -dt, n)
        xs, vs = kdk(acc_h, xb[-1], vb[-1], dt, n)
        x_sat = jnp.concatenate([xb[-1:], xs])
        v_sat = jnp.concatenate([vb[-1:], vs])
        t_sat = dt * jnp.arange(ne)
        logm_f = logm if cfg.progenitor_mass_loss == "none" else 0.0 * logm
        logm_sat = logm + (logm_f - logm) * jnp.arange(ne) / n

        # --- release: galax Fardal+15 DF. StreaMAX's tidal radius is NaN where Omega^2 - d2Phi/dr2 <= 0.
        r2 = jnp.sum(x_sat**2, -1)
        om2 = jnp.sum(jnp.cross(x_sat, v_sat) ** 2, -1) / r2**2
        d2 = jnp.einsum("ei,eij,ej->e", x_sat, host._hessian(x_sat, 0.0), x_sat) / r2
        tidal_ok = (om2 - d2) > 0
        xl, vl, xt, vt = df._sample(key, host, u.Q(rep(x_sat), "kpc"), u.Q(rep(v_sat), "kpc/Gyr"),
                                    u.Q(rep(10.0**logm_sat), "Msun"), u.Q(rep(t_sat), "Gyr"))
        xl, xt = u.ustrip("kpc", xl), u.ustrip("kpc", xt)
        vl, vt = u.ustrip("kpc/Gyr", vl), u.ustrip("kpc/Gyr", vt)
        x0 = jnp.where(sign > 0, xt, xl)
        v0 = jnp.where(sign > 0, vt, vl)

        # --- the progenitor seen by the particles of epoch e: re-integrated with that epoch's step
        dt_e = (t_int - t_sat) / n
        xe, _ = jax.vmap(lambda x, v, d: kdk(acc_h, x, v, d, n))(x_sat, v_sat, dt_e)
        ppos = jnp.swapaxes(jnp.concatenate([x_sat[:, None], xe], axis=1), 0, 1)  # (boundary, epoch, 3)
        s = jnp.arange(n + 1)[:, None]
        pmass = 10.0 ** (logm_sat[None] - s * (logm_sat[None] - logm_f) / n)  # (boundary, epoch)

        # --- particles: n KDK steps each, host + moving Plummer; angle and Gibbons coordinate on the fly
        dtp = rep(dt_e)[:, None]
        acc = lambda x, rp, m: acc_h(x) - m[:, None] * prog._gradient(x - rp, 0.0)

        def step(c, inp):
            x, v, a, th, dphi, g = c
            rp0, rp1, m1 = inp
            vh = v + 0.5 * dtp * a
            x = x + dtp * vh
            a = acc(x, rep(rp1), rep(m1))
            v = vh + 0.5 * dtp * a
            th1 = jnp.arctan2(x[:, 1], x[:, 0])
            g = g + rep(jnp.linalg.norm(rp0, axis=-1)) - jnp.linalg.norm(x, axis=-1)
            return (x, v, a, th1, dphi + _wrap(th1 - th), g), None

        th0 = jnp.arctan2(x0[:, 1], x0[:, 0])
        c0 = (x0, v0, acc(x0, rep(ppos[0]), rep(pmass[0])), th0, jnp.zeros(N), jnp.zeros(N))
        (xN, _, _, _, dphi, g), _ = jax.lax.scan(step, c0, (ppos[:-1], ppos[1:], pmass[1:]))

        # --- C26 sec. 2.2: progenitor's unwrapped angle at release + the particle's own offset and
        #     accumulated angle, minus the progenitor's final unwrapped angle
        thp = jnp.arctan2(x_sat[:, 1], x_sat[:, 0])
        dthp = _wrap(jnp.diff(thp))
        thp_u = jnp.concatenate([thp[:1], thp[0] + jnp.cumsum(dthp)])
        phi = rep(thp_u) + _wrap(th0 - rep(thp)) + dphi - thp_u[-1]
        ok = jnp.all(jnp.isfinite(xN), -1) & jnp.isfinite(phi) & rep(tidal_ok)
        if cfg.track_ordering == "gibbons14":
            phi = gibbons14_angle(xN[:, 0], xN[:, 1], g, ok)
        phi_deg = jnp.rad2deg(phi)
        out = bin_track(phi_deg, jnp.hypot(xN[:, 0], xN[:, 1]), ok, cfg)

        # --- diagnostics from a finely sampled backward orbit
        xfi, vfi = kdk(acc_h, xf, vf, -t_int / cfg.n_fine, cfg.n_fine)
        X = jnp.concatenate([xf[None], xfi])
        rr = jnp.linalg.norm(X, axis=-1)
        dth = _wrap(jnp.diff(jnp.arctan2(X[:, 1], X[:, 0])))
        cr, dot = jnp.linalg.norm(jnp.cross(X[:-1], X[1:]), axis=-1), jnp.sum(X[:-1] * X[1:], -1)
        E0 = energy(xf, vf)
        E_n, E_fine = energy(x_sat[0], v_sat[0]), energy(xfi[-1], vfi[-1])
        out.update(
            n_bad=jnp.sum(~ok).astype(jnp.int32),
            tidal_bad=jnp.any(~tidal_ok),
            prog_nonmono=jnp.any(dthp <= 0),
            prog_finite=jnp.all(jnp.isfinite(x_sat)) & jnp.all(jnp.isfinite(v_sat)),
            E=E0 / KMS**2, Phi0=host._potential(xf, 0.0) / KMS**2,  # (km/s)^2
            dE_rel=jnp.abs(E_n - E0) / jnp.abs(E0), dE_rel_fine=jnp.abs(E_fine - E0) / jnp.abs(E0),
            dx_rel=jnp.linalg.norm(x_sat[0] - xfi[-1]) / jnp.linalg.norm(xfi[-1]),
            r_min=jnp.min(rr), r_max=jnp.max(rr),
            n_peri=jnp.sum((rr[1:-1] < rr[:-2]) & (rr[1:-1] <= rr[2:])).astype(jnp.int32),
            n_apo=jnp.sum((rr[1:-1] > rr[:-2]) & (rr[1:-1] >= rr[2:])).astype(jnp.int32),
            swept_proj_deg=jnp.rad2deg(-jnp.sum(dth)),  # forward in time, signed
            swept_3d_deg=jnp.rad2deg(jnp.sum(jnp.arctan2(cr, dot))),
            part_xy=xN[:, :2].astype(jnp.float32), part_phi=phi_deg.astype(jnp.float32),
            part_g=g.astype(jnp.float32),
            sat_xy=x_sat[:, :2].astype(jnp.float32),
        )
        return out

    fn = jax.jit(jax.vmap(sim))

    def run(inp: dict, keys):
        return fn(*(jnp.asarray(inp[k]) for k in ("logM", "Rs", "q", "nvec", "logm", "rs", "wf", "t_int")), keys)

    return run


# ======================================================================================================
# Step 7 (host side): track summaries, selection flags, status word
# ======================================================================================================
PARTICLE_KEYS = ("part_xy", "part_phi")
TRACK_KEYS = ("count", "r_mean", "r_std", "r_med", "r_hw")
DIAG_KEYS = ("E", "Phi0", "dE_rel", "dE_rel_fine", "dx_rel", "r_min", "r_max", "n_peri", "n_apo",
             "swept_proj_deg", "swept_3d_deg", "n_bad", "n_overflow")
SUMMARY_KEYS = ("n_valid", "ang_len_deg", "ang_span_deg", "arc_len_kpc", "r_track_mean", "r_track_spread",
                "r_track_min", "r_track_max")
FLAG_KEYS = ("unbound", "completed_orbit", "contiguous", "sel_ang", "sel_arc", "sel_rad", "sel_all")


def track_summaries(count: np.ndarray, r_mean: np.ndarray, cfg: Config) -> dict:
    """Per-stream summaries of the binned track; a valid bin has >= n_valid_min particles.

    ang_len   = number of valid bins x bin width         (ang_span = first to last valid bin)
    arc_len   = length of the polyline through the track points of ADJACENT valid bins, in the XY plane
    mean/spread/min/max = statistics of r_mean over the valid bins
    """
    valid = count >= cfg.n_valid_min
    nv = valid.sum(1)
    has = nv > 0
    first = np.argmax(valid, axis=1)
    last = valid.shape[1] - 1 - np.argmax(valid[:, ::-1], axis=1)
    span = np.where(has, last - first + 1, 0)
    r = np.where(valid, r_mean, np.nan).astype(np.float64)
    dphi = np.deg2rad(cfg.bin_deg)
    chord = np.sqrt(np.maximum(r[:, :-1] ** 2 + r[:, 1:] ** 2 - 2 * r[:, :-1] * r[:, 1:] * np.cos(dphi), 0.0))
    arc = np.nansum(chord, axis=1)  # NaN wherever one of the two neighbours is not valid
    with np.errstate(all="ignore"):
        rmean, rstd = np.nanmean(r, 1), np.nanstd(r, 1)
        rmin, rmax = np.nanmin(np.where(valid, r, np.inf), 1), np.nanmax(np.where(valid, r, -np.inf), 1)
    rmin, rmax = np.where(has, rmin, np.nan), np.where(has, rmax, np.nan)
    ang = nv * cfg.bin_deg
    sel_ang = ang > cfg.sel_ang_deg
    sel_arc = arc > cfg.sel_arc_kpc
    sel_rad = has & (rmin >= cfg.sel_r_min) & (rmax <= cfg.sel_r_max)
    return dict(n_valid=nv.astype(np.int32), ang_len_deg=ang.astype(np.float32),
                ang_span_deg=(span * cfg.bin_deg).astype(np.float32), arc_len_kpc=arc.astype(np.float32),
                r_track_mean=rmean.astype(np.float32), r_track_spread=rstd.astype(np.float32),
                r_track_min=rmin.astype(np.float32), r_track_max=rmax.astype(np.float32),
                contiguous=has & (span == nv), sel_ang=sel_ang, sel_arc=sel_arc, sel_rad=sel_rad,
                sel_all=sel_ang & sel_arc & sel_rad)


def finalize(raw: dict, cfg: Config) -> dict:
    """Simulator output (numpy) -> everything stored per stream except the parameters."""
    out = {k: raw[k] for k in TRACK_KEYS + DIAG_KEYS}
    out.update(track_summaries(raw["count"], raw["r_mean"], cfg))
    unbound = ~(raw["E"] < 0)  # NaN energy counts as not bound
    failed = ~raw["prog_finite"] | (raw["n_bad"] >= cfg.n_particles) | ~np.isfinite(raw["E"])
    # one full radial period seen inside t_int = two turning points of the same kind
    out["unbound"] = unbound
    out["completed_orbit"] = (raw["n_peri"] >= 2) | (raw["n_apo"] >= 2)
    out["status"] = (ST_UNBOUND * unbound + ST_BAD_PARTICLES * (raw["n_bad"] > 0) + ST_TIDAL * raw["tidal_bad"]
                     + ST_OVERFLOW * (raw["n_overflow"] > 0) + ST_FAILED * failed
                     + ST_NONMONO * raw["prog_nonmono"] + ST_NO_VALID * (out["n_valid"] == 0)).astype(np.int32)
    return out


# ======================================================================================================
# Running: batches, shards, workers
# ======================================================================================================
def data_dir(cfg: Config) -> Path:
    d = REPO / cfg.out / "data"
    return d / cfg.subdir if cfg.subdir else d


def device_info() -> dict:
    d = jax.devices()[0]
    info = dict(platform=d.platform, device=str(d), visible=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    try:
        info.update({k: int(v) for k, v in d.memory_stats().items() if k in ("peak_bytes_in_use", "bytes_limit")})
    except Exception:
        pass
    return info


class Simulator:
    """Simulates arbitrary stream indices in fixed-size batches (one compilation per batch size)."""

    def __init__(self, cfg: Config, batch: int | None = None):
        self.cfg, self.fn = cfg, make_simulator(cfg)
        self.batch = batch or cfg.batch_size or self._probe()
        self.compile_s = 0.0

    def _probe(self) -> int:
        """Batch size from a memory probe: run 8 streams, read the peak allocation, scale to
        `mem_fraction` of the device memory (capped at max_batch, rounded down to a power of two)."""
        cfg = self.cfg
        if jax.devices()[0].platform != "gpu":
            return 8
        self._run(np.arange(8), 8)
        try:
            st = jax.devices()[0].memory_stats()
            peak, limit = int(st["peak_bytes_in_use"]), int(st["bytes_limit"])
        except Exception:  # no allocator statistics on this backend: a conservative default
            print(json.dumps(dict(event="probe", batch=32, note="no memory statistics")), flush=True)
            return min(32, cfg.max_batch)
        b = int(cfg.mem_fraction * limit / (peak / 8))
        b = max(8, min(cfg.max_batch, 2 ** int(np.log2(max(b, 1)))))
        print(json.dumps(dict(event="probe", peak_bytes_8=peak, bytes_limit=limit, batch=b)), flush=True)
        return b

    def _run(self, idx: np.ndarray, batch: int, keep_particles: bool = False, theta=None, inputs=None):
        cfg = self.cfg
        pad = np.concatenate([idx, np.full(batch - len(idx), idx[-1])])
        th = draw_prior(cfg, pad) if theta is None else theta
        inp = theta_to_inputs(th) if inputs is None else inputs
        _, ks = stream_keys(cfg, pad)
        res = self.fn(inp, ks)
        skip = () if keep_particles else PARTICLE_KEYS + ("part_g", "sat_xy")
        raw = {k: np.asarray(v)[: len(idx)] for k, v in res.items() if k not in skip}
        return th[: len(idx)], raw

    def simulate(self, idx: np.ndarray, keep_particles: bool = False):
        """-> theta (n, 13), raw outputs (numpy). Falls back to half the batch on an out-of-memory error."""
        thetas, raws = [], []
        i = 0
        while i < len(idx):
            chunk = idx[i : i + self.batch]
            try:
                th, raw = self._run(chunk, self.batch, keep_particles)
            except Exception as e:  # noqa: BLE001
                if "RESOURCE_EXHAUSTED" in str(e) or "out of memory" in str(e).lower():
                    if self.batch <= 1:
                        raise
                    self.batch //= 2
                    print(json.dumps(dict(event="oom", new_batch=self.batch)), flush=True)
                    continue
                raise
            thetas.append(th)
            raws.append(raw)
            i += len(chunk)
        return np.concatenate(thetas), {k: np.concatenate([r[k] for r in raws]) for k in raws[0]}


def write_shard(path: Path, cfg: Config, idx: np.ndarray, theta: np.ndarray, fin: dict, meta: dict) -> None:
    import h5py

    tmp = path.with_suffix(".tmp")
    comp = dict(compression="gzip", compression_opts=4, shuffle=True)
    with h5py.File(tmp, "w") as f:
        f.attrs["config"] = json.dumps(dataclasses.asdict(cfg))
        f.attrs["physics_hash"] = cfg.physics_hash()
        f.attrs["meta"] = json.dumps(meta)
        f.attrs["theta_names"] = json.dumps(THETA_NAMES)
        f.attrs["status_bits"] = json.dumps(STATUS_BITS)
        f.create_dataset("index", data=idx.astype(np.int64))
        f.create_dataset("theta", data=theta.astype(np.float32))
        f.create_dataset("axis", data=halo_axis(theta).astype(np.float32))
        f.create_dataset("c26_vec", data=c26_vector(theta, cfg).astype(np.float32))
        f.create_dataset("status", data=fin["status"])
        f.create_dataset("track/count", data=fin["count"].astype(np.int16), **comp)
        for k in TRACK_KEYS[1:]:
            f.create_dataset(f"track/{k}", data=fin[k].astype(np.float32), **comp)
        for k in DIAG_KEYS:
            a = fin[k]
            f.create_dataset(f"diag/{k}", data=a if a.dtype.kind == "i" else a.astype(np.float32))
        for k in SUMMARY_KEYS:
            f.create_dataset(f"summary/{k}", data=fin[k])
        for k in FLAG_KEYS:
            f.create_dataset(f"flags/{k}", data=fin[k].astype(bool))
    tmp.rename(path)


def write_particles(path: Path, idx: np.ndarray, raw: dict) -> None:
    import h5py

    tmp = path.with_suffix(".tmp")
    with h5py.File(tmp, "w") as f:
        f.create_dataset("index", data=idx.astype(np.int64))
        f.create_dataset("xy", data=raw["part_xy"])  # kpc
        f.create_dataset("phi", data=raw["part_phi"])  # deg, unwrapped, progenitor at 0
    tmp.rename(path)


def shard_range(cfg: Config, s: int) -> np.ndarray:
    return np.arange(s * cfg.shard_size, min((s + 1) * cfg.shard_size, cfg.n_streams))


def run_shards(cfg: Config, shards: list[int], tag: str = "") -> dict:
    """Simulate the given shards (skipping finished ones) on the current device. Returns timing."""
    import h5py

    ddir = data_dir(cfg)
    ddir.mkdir(parents=True, exist_ok=True)
    todo = []
    for s in shards:
        p = ddir / f"shard_{s:05d}.h5"
        if p.exists() and not cfg.overwrite:
            with h5py.File(p, "r") as f:
                if f.attrs["physics_hash"] != cfg.physics_hash():
                    raise SystemExit(f"{p} was made with other settings; use --overwrite or another --out/--subdir")
            continue
        todo.append(s)
    stats = dict(n_done=0, sim_s=0.0, compile_s=0.0, batch=0)
    if not todo:
        return stats
    t0 = time.time()
    sim = Simulator(cfg)
    sim._run(np.arange(sim.batch), sim.batch)  # compile at the production batch size
    stats["compile_s"] = time.time() - t0
    stats["batch"] = sim.batch
    for s in todo:
        t1 = time.time()
        idx = shard_range(cfg, s)
        keep = idx[0] < cfg.n_save_particles
        theta, raw = sim.simulate(idx, keep_particles=keep)
        fin = finalize(raw, cfg)
        dt = time.time() - t1
        meta = dict(shard=s, wall_s=dt, batch=sim.batch, **device_info(), tag=tag, time=time.strftime("%F %T"))
        if keep:
            m = idx < cfg.n_save_particles
            write_particles(ddir / f"particles_{s:05d}.h5", idx[m], {k: raw[k][m] for k in PARTICLE_KEYS})
        write_shard(ddir / f"shard_{s:05d}.h5", cfg, idx, theta, fin, meta)
        stats["n_done"] += len(idx)
        stats["sim_s"] += dt
        print(json.dumps(dict(event="shard", shard=s, n=len(idx), wall_s=round(dt, 2),
                              rate=round(len(idx) / dt, 2), batch=sim.batch, tag=tag)), flush=True)
    stats.update(device_info())
    return stats


def write_manifest(cfg: Config, extra: dict | None = None) -> dict:
    import h5py
    from importlib.metadata import version

    ddir = data_dir(cfg)
    shards = []
    for p in sorted(ddir.glob("shard_*.h5")):
        with h5py.File(p, "r") as f:
            idx = f["index"][:]
            meta = json.loads(f.attrs["meta"])
            shards.append(dict(file=p.name, start=int(idx[0]), stop=int(idx[-1]) + 1, n=len(idx),
                               wall_s=meta["wall_s"], device=meta.get("visible", ""),
                               particles=(f"particles_{meta['shard']:05d}.h5"
                                          if (ddir / f"particles_{meta['shard']:05d}.h5").exists() else None)))
    man = dict(
        created=time.strftime("%F %T"), config=dataclasses.asdict(cfg), physics_hash=cfg.physics_hash(),
        n_streams=sum(s["n"] for s in shards), complete=sum(s["n"] for s in shards) == cfg.n_streams,
        theta_names=THETA_NAMES, status_bits=STATUS_BITS,
        phi_edges_deg=[-cfg.phi_max_deg, cfg.phi_max_deg, cfg.bin_deg],
        versions={p: version(p) for p in ("galax", "jax", "unxt", "numpy", "h5py")},
        streamax_commit=STREAMAX_COMMIT, shards=shards, **(extra or {}),
    )
    (ddir / "manifest.json").write_text(json.dumps(man, indent=1))
    return man


def cli_overrides(cfg: Config) -> list[str]:
    """Command-line flags reproducing every non-default field (handed to the workers)."""
    out = []
    for f in dataclasses.fields(Config):
        v = getattr(cfg, f.name)
        if v == f.default or f.name in ("gpus", "cpu"):
            continue
        flag = "--" + f.name.replace("_", "-")
        out += [flag] if isinstance(v, bool) else [flag, ",".join(map(str, v)) if isinstance(v, tuple) else str(v)]
    return out


def cmd_run(cfg: Config) -> None:
    """Step 6: one worker per free GPU, each with a contiguous, disjoint block of shards."""
    ddir = data_dir(cfg)
    (ddir / "logs").mkdir(parents=True, exist_ok=True)
    n_shards = -(-cfg.n_streams // cfg.shard_size)
    devices = [""] if cfg.cpu else ([g for g in cfg.gpus.split(",") if g] or free_gpus())
    if not devices:
        raise SystemExit("no free GPU right now (pass --gpus to force a choice, or --cpu)")
    blocks = [b for b in np.array_split(np.arange(n_shards), len(devices)) if len(b)]
    print(f"{cfg.n_streams} streams in {n_shards} shards on {len(blocks)} worker(s): {devices[:len(blocks)]}", flush=True)
    t0 = time.time()
    procs = []
    for dev, blk in zip(devices, blocks):
        env = dict(os.environ)
        env.pop("JAX_PLATFORMS", None)
        env["CUDA_VISIBLE_DEVICES"] = dev
        cmd = [sys.executable, __file__, "worker", "--shards", f"{blk[0]}:{blk[-1] + 1}", *cli_overrides(cfg)]
        cmd += ["--cpu"] if cfg.cpu else []
        log = open(ddir / "logs" / f"worker_{blk[0]:05d}.log", "a")
        procs.append((subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), dev, blk))
    fail = [(dev, p.returncode) for p, dev, _ in procs if p.wait() != 0]
    wall = time.time() - t0
    man = write_manifest(cfg, dict(run=dict(wall_s=wall, devices=devices[: len(blocks)], failed=fail)))
    print(f"done in {wall:.0f} s: {man['n_streams']} / {cfg.n_streams} streams on disk"
          + (f"; FAILED workers {fail} (see {ddir / 'logs'}; rerun to resume)" if fail else ""), flush=True)


def cmd_worker(cfg: Config, shards: str) -> None:
    a, b = map(int, shards.split(":"))
    if not cfg.cpu and jax.devices()[0].platform != "gpu":
        raise SystemExit("worker: JAX did not initialise the GPU (refusing to fall back to CPU silently)")
    print(json.dumps(dict(event="start", shards=[a, b], **device_info())), flush=True)
    st = run_shards(cfg, list(range(a, b)), tag=os.environ.get("CUDA_VISIBLE_DEVICES", "cpu"))
    print(json.dumps(dict(event="end", **st)), flush=True)


def cmd_pilot(cfg: Config) -> None:
    """Step 5: `pilot_n` streams on one device, written to data/pilot; reports rate, memory, NaN rate."""
    cfg = dataclasses.replace(cfg, n_streams=cfg.pilot_n, subdir=cfg.subdir or "pilot")
    ddir = data_dir(cfg)
    n_shards = -(-cfg.n_streams // cfg.shard_size)
    t0 = time.time()
    st = run_shards(cfg, list(range(n_shards)), tag="pilot")
    wall = time.time() - t0
    write_manifest(cfg)
    table = cmd_analyze(cfg, quiet=True)
    rate = st["n_done"] / st["sim_s"] if st["sim_s"] else float("nan")
    n_free = 1 if cfg.cpu else max(1, len(free_gpus()) + 1)  # + the one this pilot holds
    rep = dict(
        n_streams=st["n_done"], device=st.get("device"), platform=st.get("platform"), batch=st["batch"],
        compile_s=round(st["compile_s"], 1), sim_s=round(st["sim_s"], 1), wall_s=round(wall, 1),
        streams_per_s=round(rate, 2),
        peak_mem_GB=round(st.get("peak_bytes_in_use", 0) / 2**30, 2),
        device_mem_GB=round(st.get("bytes_limit", 0) / 2**30, 2),
        nan_rate=dict(failed=table["all"]["failed"], any_bad_particle=table["extra"]["any_bad_particle"],
                      mean_bad_particle_fraction=table["extra"]["mean_bad_particle_fraction"]),
        gpus_free_now=n_free,
        projected_wall_s_1e5=round(1e5 / (rate * n_free) + st["compile_s"], 0),
        projected_wall_s_1e5_one_device=round(1e5 / rate + st["compile_s"], 0),
    )
    (ddir / "pilot_report.json").write_text(json.dumps(rep, indent=1))
    print("\n=== pilot ===\n" + json.dumps(rep, indent=1))
    print("\n" + (ddir / "summary_table.md").read_text())


# ======================================================================================================
# Step 7: analyze
# ======================================================================================================
def load_shards(ddir: Path, with_tracks: bool = False) -> dict:
    import h5py

    cols: dict[str, list] = {}
    groups = ["diag", "summary", "flags"] + (["track"] if with_tracks else [])
    for p in sorted(ddir.glob("shard_*.h5")):
        with h5py.File(p, "r") as f:
            for k in ("index", "theta", "axis", "c26_vec", "status"):
                cols.setdefault(k, []).append(f[k][:])
            for g in groups:
                for k in f[g]:
                    cols.setdefault(k, []).append(f[g][k][:])
    if not cols:
        raise SystemExit(f"no shards in {ddir}")
    d = {k: np.concatenate(v) for k, v in cols.items()}
    order = np.argsort(d["index"])
    return {k: v[order] for k, v in d.items()}


def summary_table(d: dict, cfg: Config) -> dict:
    """The step-7 table: fractions over all streams and over the bound ones."""
    st = d["status"]
    unbound = d["unbound"].astype(bool)
    failed = (st & ST_FAILED) > 0
    bound = ~unbound & ~failed
    speed = np.linalg.norm(d["theta"][:, 9:12], axis=1)

    def block(m):
        n = max(int(m.sum()), 1)
        f = lambda a: float((a & m).sum() / n)
        return dict(
            n=int(m.sum()), unbound=f(unbound), bound_no_completed_orbit=f(bound & ~d["completed_orbit"].astype(bool)),
            failed=f(failed), any_valid_bin=f(d["n_valid"] > 0),
            sel_ang=f(d["sel_ang"].astype(bool)), sel_arc=f(d["sel_arc"].astype(bool)),
            sel_rad=f(d["sel_rad"].astype(bool)), sel_all=f(d["sel_all"].astype(bool)),
        )

    edges_m = np.linspace(cfg.log10_M_halo[0], cfg.log10_M_halo[1], 7)
    edges_v = np.array([0, 100, 200, 300, 400, 500, 700, np.inf])
    im = np.clip(np.digitize(d["theta"][:, 0], edges_m) - 1, 0, len(edges_m) - 2)
    iv = np.clip(np.digitize(speed, edges_v) - 1, 0, len(edges_v) - 2)
    frac = lambda sel: float(unbound[sel].mean()) if sel.any() else float("nan")
    return dict(
        all=block(np.ones_like(bound)), bound=block(bound),
        extra=dict(
            any_bad_particle=float(((st & ST_BAD_PARTICLES) > 0).mean()),
            mean_bad_particle_fraction=float(d["n_bad"].mean() / cfg.n_particles),
            tidal_undefined=float(((st & ST_TIDAL) > 0).mean()),
            grid_overflow=float(((st & ST_OVERFLOW) > 0).mean()),
            prog_angle_nonmonotonic=float(((st & ST_NONMONO) > 0).mean()),
            sel_all_unbound=float((d["sel_all"].astype(bool) & unbound).sum() / max(d["sel_all"].sum(), 1)),
            unresolved_orbit=float((d["dE_rel"] > 1e-2).mean()),
            unresolved_orbit_bound=float((d["dE_rel"][bound] > 1e-2).mean()) if bound.any() else float("nan"),
            unresolved_orbit_sel_all=float((d["dE_rel"][d["sel_all"].astype(bool)] > 1e-2).mean())
            if d["sel_all"].any() else float("nan"),
            sel_all_and_contiguous=float((d["sel_all"].astype(bool) & d["contiguous"].astype(bool)).mean()),
            dE_rel_p50_p99_bound=[float(x) for x in np.nanpercentile(d["dE_rel"][bound], [50, 99])] if bound.any() else None,
        ),
        unbound_vs_log10M=dict(edges=edges_m.tolist(), frac=[frac(im == i) for i in range(len(edges_m) - 1)],
                               n=[int((im == i).sum()) for i in range(len(edges_m) - 1)]),
        unbound_vs_speed=dict(edges=[float(e) for e in edges_v[:-1]] + ["inf"],
                              frac=[frac(iv == i) for i in range(len(edges_v) - 1)],
                              n=[int((iv == i).sum()) for i in range(len(edges_v) - 1)]),
        unbound_vs_log10M_speed=[[frac((im == i) & (iv == j)) for j in range(len(edges_v) - 1)]
                                 for i in range(len(edges_m) - 1)],
    )


def table_markdown(t: dict, cfg: Config) -> str:
    rows = [("unbound (E >= 0)", "unbound"), ("bound, no completed radial orbit in t_int", "bound_no_completed_orbit"),
            ("failed / NaN", "failed"), ("at least one valid bin", "any_valid_bin"),
            (f"angular length > {cfg.sel_ang_deg:g} deg", "sel_ang"), (f"arc length > {cfg.sel_arc_kpc:g} kpc", "sel_arc"),
            (f"valid radii in [{cfg.sel_r_min:g}, {cfg.sel_r_max:g}] kpc", "sel_rad"), ("all three cuts", "sel_all")]
    pc = lambda x: f"{100 * x:.2f} %"
    md = [f"| fraction of streams | all (n = {t['all']['n']}) | bound only (n = {t['bound']['n']}) |", "|---|---|---|"]
    md += [f"| {lab} | {pc(t['all'][k])} | {pc(t['bound'][k])} |" for lab, k in rows]
    e = t["extra"]
    md += ["", f"Other flags (all streams): any non-finite particle {pc(e['any_bad_particle'])} "
           f"(mean particle fraction {pc(e['mean_bad_particle_fraction'])}), tidal radius undefined at some epoch "
           f"{pc(e['tidal_undefined'])}, particles outside the phi grid {pc(e['grid_overflow'])}, projected "
           f"progenitor angle not monotonic {pc(e['prog_angle_nonmonotonic'])}, all three cuts and contiguous "
           f"valid bins {pc(e['sel_all_and_contiguous'])}.", ""]
    md += [f"Of the streams passing all three cuts, {pc(e['sel_all_unbound'])} are unbound. Progenitor orbit not "
           f"resolved by {cfg.n_steps} leapfrog steps (relative energy error > 1 %): {pc(e['unresolved_orbit'])} of all, "
           f"{pc(e['unresolved_orbit_bound'])} of the bound, {pc(e['unresolved_orbit_sel_all'])} of those passing the "
           f"cuts.", ""]
    um, uv = t["unbound_vs_log10M"], t["unbound_vs_speed"]
    md += ["| log10 M_halo | " + " | ".join(f"{a:.1f}-{b:.1f}" for a, b in zip(um["edges"][:-1], um["edges"][1:])) + " |",
           "|---|" + "---|" * len(um["frac"]), "| unbound | " + " | ".join(pc(x) for x in um["frac"]) + " |", ""]
    md += ["| speed today [km/s] | " + " | ".join(f"{a:g}-{b:g}" if b != "inf" else f">{a:g}" for a, b in
                                                    zip(uv["edges"][:-1], uv["edges"][1:])) + " |",
           "|---|" + "---|" * len(uv["frac"]), "| unbound | " + " | ".join(pc(x) for x in uv["frac"]) + " |", ""]
    return "\n".join(md)


def cmd_analyze(cfg: Config, quiet: bool = False) -> dict:
    """Collect the per-stream diagnostics of every shard into data/analysis.h5, recompute the track
    summaries and selection flags with the current thresholds, and write the step-7 table."""
    import h5py

    ddir = data_dir(cfg)
    d = load_shards(ddir, with_tracks=True)
    d.update(track_summaries(d["count"], d["r_mean"], cfg))  # thresholds may differ from generation time
    d["status"] = ((d["status"] & ~ST_NO_VALID) | ST_NO_VALID * (d["n_valid"] == 0)).astype(np.int32)
    table = summary_table(d, cfg)
    with h5py.File(ddir / "analysis.h5", "w") as f:
        f.attrs["config"] = json.dumps(dataclasses.asdict(cfg))
        f.attrs["theta_names"] = json.dumps(THETA_NAMES)
        f.attrs["status_bits"] = json.dumps(STATUS_BITS)
        f.attrs["summary"] = json.dumps(table)
        for k, v in d.items():
            if k not in TRACK_KEYS:
                f.create_dataset(k, data=v)
    (ddir / "summary_table.json").write_text(json.dumps(table, indent=1))
    md = table_markdown(table, cfg)
    (ddir / "summary_table.md").write_text(md + "\n")
    if not quiet:
        print(md)
    return table


# ======================================================================================================
# Step 4: tests
# ======================================================================================================
def _check(report: dict, name: str, ok: bool, **info) -> None:
    report[name] = dict(passed=bool(ok), **info)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: " + ", ".join(f"{k}={v}" for k, v in info.items()), flush=True)


def test_prior(cfg: Config, rep: dict) -> None:
    from scipy import stats
    from scipy.special import erf

    th = draw_prior(cfg, np.arange(100_000))
    vec = c26_vector(th, cfg)
    p = dict(xhat=stats.kstest(vec[:, 0], "norm").pvalue, yhat=stats.kstest(vec[:, 1], "norm").pvalue,
             zhat=stats.kstest(vec[:, 2], "halfnorm").pvalue)
    _check(rep, "prior_c26_vector_KS", min(p.values()) > 1e-3, **{k: f"{v:.3f}" for k, v in p.items()})
    U = lambda ab: stats.uniform(ab[0], ab[1] - ab[0]).cdf
    cdfs = [U(cfg.log10_M_halo), U(cfg.R_s), U(cfg.q), U((0, 1)), U((0, 2 * np.pi)), U(cfg.log10_m_prog), U(cfg.r_s),
            stats.halfnorm(scale=cfg.sigma_pos).cdf, stats.halfnorm(scale=cfg.sigma_pos).cdf,
            stats.norm(scale=cfg.sigma_vel).cdf, stats.halfnorm(scale=cfg.sigma_vel).cdf,
            stats.norm(scale=cfg.sigma_vel).cdf, U(cfg.t_int)]
    pm = {n: stats.kstest(th[:, i], c).pvalue for i, (n, c) in enumerate(zip(THETA_NAMES, cdfs))}
    _check(rep, "prior_marginals_KS", min(pm.values()) > 1e-3, min_p=f"{min(pm.values()):.3f}",
           worst=min(pm, key=pm.get))
    # the stored vector maps back to q through C26 eq. 11
    r = np.linalg.norm(vec, axis=1)
    q_back = cfg.q[0] + (cfg.q[1] - cfg.q[0]) * (erf(r / np.sqrt(2)) - np.sqrt(2 / np.pi) * r * np.exp(-r**2 / 2))
    _check(rep, "prior_c26_vector_to_q", np.abs(q_back - th[:, 2]).max() < 1e-9,
           max_abs_err=f"{np.abs(q_back - th[:, 2]).max():.1e}")
    # reproducibility: stream k does not depend on which other streams are drawn with it
    a, b = draw_prior(cfg, np.array([7, 123, 99_999])), draw_prior(cfg, np.arange(99_990, 100_000))
    _check(rep, "prior_fold_in_reproducible", np.array_equal(a[2], b[9]) and np.array_equal(a[0], th[7]))


def test_potential(cfg: Config, rep: dict) -> None:
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(size=(200, 3)) * 40.0)
    n = np.array([0.3, -0.5, 0.81])
    n /= np.linalg.norm(n)
    G = 4.498502151469553e-06
    for q in (0.5, 0.8, 1.0, 1.3, 1.5):
        pot = OrientedNFWPotential(m=1e12, r_s=15.0, q=q, axis=n, units=USYS)
        # (a) analytic gradient = autodiff gradient of the potential
        ga = pot._gradient(x, 0.0)
        gn = jax.vmap(jax.grad(lambda xx: pot._potential(xx, 0.0)))(x)
        e_grad = float(jnp.max(jnp.linalg.norm(ga - gn, axis=-1) / jnp.linalg.norm(gn, axis=-1)))
        # (b) Poisson: laplacian / 4 pi G = NFW density on m^2 = R^2 + Z^2/q^2 with rho0 = M / (4 pi r_s^3)
        lap = jax.vmap(lambda xx: jnp.trace(jax.hessian(lambda y: pot._potential(y, 0.0))(xx)))(x)
        Z = x @ n
        m = jnp.sqrt(jnp.sum(x * x, -1) - Z**2 + Z**2 / q**2) / 15.0
        rho = 1e12 / (4 * np.pi * 15.0**3) / (m * (1 + m) ** 2)
        e_rho = float(jnp.max(jnp.abs(lap / (4 * np.pi * G) / rho - 1)))
        # (c) galax's own density-flattened NFW (50-node quadrature, flattened along z), rotated by hand
        zax = OrientedNFWPotential(m=1e12, r_s=15.0, q=q, axis=np.array([0.0, 0.0, 1.0]), units=USYS)
        tri = gp.TriaxialNFWPotential(m=1e12, r_s=15.0, q1=1.0, q2=q, units=USYS, integration_order=50)
        e_gx = float(jnp.max(jnp.abs(zax._potential(x, 0.0) / jax.vmap(lambda xx: tri._potential(xx, 0.0))(x) - 1)))
        _check(rep, f"potential_q{q}", e_grad < 1e-8 and e_rho < 2e-2 and e_gx < 1e-4,
               grad_vs_autodiff=f"{e_grad:.1e}", density_vs_NFW=f"{e_rho:.1e}", phi_vs_galax_Triaxial=f"{e_gx:.1e}")
    sph = OrientedNFWPotential(m=1e12, r_s=15.0, q=1.0, axis=n, units=USYS)
    nfw = gp.NFWPotential(m=1e12, r_s=15.0, units=USYS)
    e = float(jnp.max(jnp.abs(sph._potential(x, 0.0) / jax.vmap(lambda xx: nfw._potential(xx, 0.0))(x) - 1)))
    _check(rep, "potential_q1_vs_galax_NFW", e < 1e-6, max_rel=f"{e:.1e}")


def test_energy(cfg: Config, rep: dict) -> dict:
    """Progenitor orbit with n_steps (and 100) leapfrog steps against 2,000, on `n_energy` prior draws."""
    th = draw_prior(cfg, np.arange(cfg.n_energy))
    inp = theta_to_inputs(th)

    def one(logM, Rs, q, nvec, wf, t_int):
        host = OrientedNFWPotential(m=10.0**logM, r_s=Rs, q=q, axis=nvec, units=USYS, flatten=cfg.halo_flattening)
        acc = lambda x: -host._gradient(x, 0.0)
        E = lambda x, v: 0.5 * jnp.sum(v * v) + host._potential(x, 0.0)
        E0, res = E(wf[:3], wf[3:]), {}
        for ns in sorted({cfg.n_steps, 100, 2000}):
            xs, vs = kdk(acc, wf[:3], wf[3:], -t_int / ns, ns)
            res[ns] = (jnp.abs(E(xs[-1], vs[-1]) - E0) / jnp.abs(E0), xs[-1])
        return E0, {ns: (dE, jnp.linalg.norm(xe - res[2000][1]) / jnp.linalg.norm(res[2000][1])) for ns, (dE, xe) in res.items()}

    E0, res = jax.jit(jax.vmap(one))(*(jnp.asarray(inp[k]) for k in ("logM", "Rs", "q", "nvec", "wf", "t_int")))
    bound = np.asarray(E0) < 0
    pct = lambda a: [float(f"{x:.3g}") for x in np.nanpercentile(np.asarray(a), [50, 90, 99, 100])]
    out = dict(n=cfg.n_energy, n_bound=int(bound.sum()), percentiles=[50, 90, 99, 100])
    for ns, (dE, dx) in res.items():
        out[f"dE_rel_{ns}_steps"] = dict(all=pct(dE), bound=pct(np.asarray(dE)[bound]), unbound=pct(np.asarray(dE)[~bound]))
        out[f"dx_rel_{ns}_vs_2000"] = dict(all=pct(dx), bound=pct(np.asarray(dx)[bound]), unbound=pct(np.asarray(dx)[~bound]))
    out["raw"] = dict(E0=np.asarray(E0), **{f"dE_{ns}": np.asarray(v[0]) for ns, v in res.items()},
                      **{f"dx_{ns}": np.asarray(v[1]) for ns, v in res.items()}, theta=th)
    med = out[f"dE_rel_{cfg.n_steps}_steps"]["all"][0]
    _check(rep, "energy_conservation", med < 1e-2, n_steps=cfg.n_steps, n_bound=out["n_bound"],
           dE_rel_p50_p90_p99_max=out[f"dE_rel_{cfg.n_steps}_steps"]["all"],
           dE_rel_2000=out["dE_rel_2000_steps"]["all"], dx_rel_vs_2000=out[f"dx_rel_{cfg.n_steps}_vs_2000"]["all"])
    return out


def _rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_symmetries(cfg: Config, sim: "Simulator", th: np.ndarray, rep: dict) -> None:
    """q = 1 is orientation-free; n and -n are the same halo; a rotation about z leaves r(phi) unchanged."""
    B = sim.batch
    base = np.repeat(th[:1], B, axis=0)
    inp = theta_to_inputs(base)
    R = _rotz(1.234)
    n0 = inp["nvec"][0]
    # rows: 0 q=1 axis A | 1 q=1 axis B | 2 reference (q of the draw) | 3 flipped axis | 4 rotated about z
    inp["q"] = inp["q"].copy()
    inp["q"][:2] = 1.0
    inp["nvec"] = inp["nvec"].copy()
    inp["nvec"][1] = np.array([0.6, 0.0, 0.8])
    inp["nvec"][3] = -n0
    inp["nvec"][4] = R @ n0
    inp["wf"] = inp["wf"].copy()
    inp["wf"][4] = np.concatenate([R @ inp["wf"][2, :3], R @ inp["wf"][2, 3:]])
    idx = np.zeros(B, dtype=int)
    _, raw = sim._run(idx, B, keep_particles=True, theta=base, inputs=inp)
    rr = np.hypot(raw["part_xy"][..., 0], raw["part_xy"][..., 1]).astype(np.float64)

    def cmp(i, j):
        m = np.isfinite(rr[i]) & np.isfinite(rr[j])
        dr = np.abs(rr[i] - rr[j])[m].max() / np.median(rr[i][m])
        dphi = np.abs(raw["part_phi"][i] - raw["part_phi"][j])[m].max()
        dtrack = np.nanmax(np.abs(raw["r_mean"][i] - raw["r_mean"][j]))
        same = np.array_equal(raw["count"][i], raw["count"][j])
        return dr, dphi, dtrack, same

    for name, (i, j) in dict(q1_orientation_invariance=(0, 1), axis_flip_n_minus_n=(2, 3), rotation_about_z=(2, 4)).items():
        dr, dphi, dtrack, same = cmp(i, j)
        # particles are stored in float32: agreement is limited to ~1e-6 relative
        _check(rep, name, dr < 1e-4 and dphi < 1e-2 and same, max_rel_dr=f"{dr:.1e}", max_dphi_deg=f"{dphi:.1e}",
               max_dr_mean_kpc=f"{dtrack:.1e}", counts_identical=same)


def test_streamax(cfg: Config, sim: "Simulator", th: np.ndarray, rep: dict, tdir: Path) -> None:
    """Five bound parameter sets through StreaMAX itself (float32, its own process) and through this code."""
    B = sim.batch
    k = len(th)
    pad = np.concatenate([th, np.repeat(th[-1:], B - k, axis=0)])
    idx = np.arange(B)
    _, raw = sim._run(idx, B, keep_particles=True, theta=pad)
    np.savez(tdir / "streamax_in.npz", theta=th, c26_vec=c26_vector(th, cfg), n_steps=cfg.n_steps,
             n_particles=cfg.n_particles, flatten=cfg.halo_flattening, mass_loss=cfg.progenitor_mass_loss)
    env = dict(os.environ, JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([sys.executable, __file__, "streamax", str(tdir / "streamax_in.npz"),
                        str(tdir / "streamax_out.npz")], env=env, capture_output=True, text=True)
    if r.returncode != 0:
        _check(rep, "streamax_comparison", False, error=r.stderr.strip().splitlines()[-1][:200] if r.stderr else "?")
        return
    sm = np.load(tdir / "streamax_out.npz")
    out = dict(theta=th, phi_centres=np.arange(-cfg.phi_max_deg, cfg.phi_max_deg, cfg.bin_deg) + cfg.bin_deg / 2)
    zs, zs_c26, dsat = [], [], []
    rezero = lambda x, y, j, xp, yp: (np.arctan2(y[j], x[j]) - np.arctan2(yp, xp) + np.pi) % (2 * np.pi) - np.pi
    binned = lambda ph, x, y, okm: {a: np.asarray(b) for a, b in bin_track(
        jnp.asarray(np.rad2deg(ph), dtype=float), jnp.asarray(np.hypot(x, y), dtype=float), jnp.asarray(okm), cfg).items()}

    def zscore(a, b):
        v = (a["count"] >= cfg.n_valid_min) & (b["count"] >= cfg.n_valid_min)
        z = (a["r_mean"] - b["r_mean"]) / np.sqrt(a["r_std"] ** 2 / np.maximum(a["count"], 1)
                                                  + b["r_std"] ** 2 / np.maximum(b["count"], 1))
        return z, v

    for i in range(k):
        # deterministic part: the progenitor orbit on the release epochs
        dsat.append(np.abs(raw["sat_xy"][i] - sm["xv_sat"][i][:, :2]).max() / np.abs(raw["sat_xy"][i]).max())
        xp, yp = sm["xv_sat"][i][-1, :2]
        # StreaMAX: its own ordering (get_stream_ordered), re-zeroed on the progenitor instead of on a particle
        x, y, t = sm["x"][i], sm["y"][i], sm["theta"][i]
        ts = binned(t + rezero(x, y, np.argmin(np.abs(t)), xp, yp), x, y, np.isfinite(x) & np.isfinite(t))
        # this code, same ordering algorithm (isolates the simulator from the ordering) ...
        xm, ym, gm = (raw["part_xy"][i][:, 0].astype(float), raw["part_xy"][i][:, 1].astype(float),
                      raw["part_g"][i].astype(float))
        okm = np.isfinite(xm) & np.isfinite(raw["part_phi"][i])
        pg = np.asarray(gibbons14_angle(jnp.asarray(xm), jnp.asarray(ym), jnp.asarray(gm), jnp.asarray(okm)))
        tg = binned(pg + rezero(xm, ym, np.argmin(np.where(okm, np.abs(gm), np.inf)), xp, yp), xm, ym, okm)
        # ... and with the C26 per-particle unwrapping that the dataset stores
        tc = {a: raw[a][i] for a in ("count", "r_mean", "r_std")}
        z, v = zscore(tg, ts)
        zc, vc = zscore(tc, ts)
        zs.append(z[v])
        zs_c26.append(zc[vc])
        for a in ("count", "r_mean", "r_std"):
            out[f"{a}_this_{i}"], out[f"{a}_this_c26_{i}"], out[f"{a}_streamax_{i}"] = tg[a], tc[a], ts[a]
        out[f"xy_this_{i}"], out[f"xy_streamax_{i}"] = raw["part_xy"][i], np.stack([x, y], -1)
        out[f"valid_{i}"], out[f"valid_c26_{i}"] = v, vc
    np.savez(tdir / "streamax_compare.npz", **out)
    zall = np.concatenate(zs) if zs else np.array([np.nan])
    zc26 = np.concatenate(zs_c26) if zs_c26 else np.array([np.nan])
    _check(rep, "streamax_progenitor_orbit", max(dsat) < 1e-3, max_rel_dx=f"{max(dsat):.1e}")
    _check(rep, "streamax_binned_tracks", np.isfinite(zall).all() and np.abs(np.mean(zall)) < 0.4 and np.std(zall) < 1.5,
           ordering="same (gibbons14) on both", n_bins=len(zall), z_mean=f"{np.mean(zall):.2f}",
           z_std=f"{np.std(zall):.2f}", z_abs_max=f"{np.abs(zall).max():.2f}",
           valid_bins_per_stream=[int(len(z)) for z in zs])
    # informational: the dataset's C26 ordering against StreaMAX's current (2026b) ordering
    rep["streamax_binned_tracks_c26_vs_gibbons_ordering"] = dict(
        passed=True, n_bins=len(zc26), z_mean=float(np.mean(zc26)), z_std=float(np.std(zc26)),
        z_abs_max=float(np.abs(zc26).max()), frac_abs_z_gt_3=float((np.abs(zc26) > 3).mean()))
    print("[INFO] C26 ordering vs StreaMAX ordering:", rep["streamax_binned_tracks_c26_vs_gibbons_ordering"], flush=True)


def cmd_streamax(fin: str, fout: str) -> None:
    """Runs in its own process: importing StreaMAX switches JAX to float32."""
    import jax as _jax
    import jax.numpy as _jnp
    import StreaMAX

    d = np.load(fin)
    n_steps, n_particles = int(d["n_steps"]), int(d["n_particles"])
    host = "NFWFlatDensity" if str(d["flatten"]) == "density" else "NFW"
    res = dict(x=[], y=[], theta=[], xv_sat=[])
    for th, vec in zip(d["theta"], d["c26_vec"]):
        logM, Rs, q, _, _, logm, rs, x0, z0, vx, vy, vz, t = (float(a) for a in th)
        ph = dict(logM=logM, Rs=Rs, a=1.0, b=1.0, c=q, dirx=float(vec[0]), diry=float(vec[1]), dirz=float(vec[2]),
                  x_origin=0.0, y_origin=0.0, z_origin=0.0)
        ps = dict(logM=logm, Rs=rs, x_origin=x0, y_origin=0.0, z_origin=z0)
        xv_f = _jnp.array([x0, 0.0, z0, vx * StreaMAX.KMS_TO_KPCGYR, vy * StreaMAX.KMS_TO_KPCGYR, vz * StreaMAX.KMS_TO_KPCGYR])
        m_f = logm if str(d["mass_loss"]) == "none" else 0.0
        _, xv_sat, xv, xhi = StreaMAX.generate_stream(xv_f, host, ph, "Plummer", ps, t, 1.0, n_steps, n_particles,
                                                      1, m_f_sat=m_f)
        xo, yo, tho, _, _ = StreaMAX.get_stream_ordered(xv[:, 0], xv[:, 1], xhi)
        for k, v in zip(("x", "y", "theta", "xv_sat"), (xo, yo, tho, xv_sat)):
            res[k].append(np.asarray(v))
    # the rotation convention behind `c26_vector`: StreaMAX's symmetry axis is R^T z
    vec = d["c26_vec"][0]
    R = np.asarray(StreaMAX.get_mat(*(float(a) for a in vec)))
    np.savez(fout, axis_streamax=R.T @ np.array([0.0, 0.0, 1.0]), x64=bool(_jax.config.jax_enable_x64),
             **{k: np.stack(v) for k, v in res.items()})


def cmd_test(cfg: Config) -> None:
    tdir = data_dir(cfg) / "tests"
    tdir.mkdir(parents=True, exist_ok=True)
    rep: dict = {}
    t0 = time.time()
    test_prior(cfg, rep)
    test_potential(cfg, rep)
    en = test_energy(cfg, rep)
    np.savez(tdir / "energy.npz", **en.pop("raw"))
    # a handful of prior draws through the full simulator: smoke test + bound streams for StreaMAX
    sim = Simulator(cfg, batch=cfg.test_batch)
    t1 = time.time()
    theta, raw = sim.simulate(np.arange(cfg.n_test_streams), keep_particles=False)
    fin = finalize(raw, cfg)
    wall = time.time() - t1
    bound = ~fin["unbound"] & ((fin["status"] & ST_FAILED) == 0)
    _check(rep, "simulator_smoke", np.isfinite(raw["E"]).all() and (raw["count"].sum(1) + raw["n_bad"] + raw["n_overflow"]
                                                                      == cfg.n_particles).all(),
           n=cfg.n_test_streams, bound=int(bound.sum()), sel_all=int(fin["sel_all"].sum()),
           with_bad_particles=int((raw["n_bad"] > 0).sum()), wall_s=f"{wall:.0f}", device=device_info()["platform"])
    # Five bound sets whose progenitor orbit is resolved by n_steps (energy error < 1 %): those passing the C26
    # cuts first, then the longest tracks. Deeply bound draws with many orbits inside t_int are NOT resolved by
    # ~100 steps (see the energy test); their leapfrog map is chaotic, so round-off (and StreaMAX's float32)
    # changes individual particles by O(r) and neither an exact symmetry nor a noise-level comparison exists.
    good = bound & (raw["dE_rel"] < 1e-2) & (fin["n_valid"] > 0)
    order = np.lexsort((-fin["n_valid"], ~fin["sel_all"]))
    pick = [i for i in order if good[i]][:5]
    th5 = theta[pick]
    if len(pick):
        test_symmetries(cfg, sim, theta[[min(pick, key=lambda i: raw["dE_rel"][i])]], rep)
        test_streamax(cfg, sim, th5, rep, tdir)
        if (tdir / "streamax_out.npz").exists():
            ax = np.load(tdir / "streamax_out.npz")["axis_streamax"]
            _check(rep, "streamax_axis_convention", np.allclose(ax, halo_axis(th5[:1])[0], atol=1e-5),
                   axis_streamax=np.round(ax, 4).tolist(), axis_here=np.round(halo_axis(th5[:1])[0], 4).tolist())
    rep["_meta"] = dict(wall_s=round(time.time() - t0, 1), energy=en, config=dataclasses.asdict(cfg),
                        picked_streams=[int(i) for i in pick], **device_info())
    (tdir / "test_report.json").write_text(json.dumps(rep, indent=1, default=str))
    n_fail = sum(1 for k, v in rep.items() if not k.startswith("_") and not v["passed"])
    print(f"\n{len(rep) - 1 - n_fail} passed, {n_fail} failed in {time.time() - t0:.0f} s -> {tdir / 'test_report.json'}")
    plot = Path(__file__).with_name("plot.py")
    if plot.exists():
        subprocess.run([sys.executable, str(plot), "--out", cfg.out, "--only", "tests"]
                       + (["--subdir", cfg.subdir] if cfg.subdir else []), check=False)


# ======================================================================================================
# CLI
# ======================================================================================================
def parse_args() -> tuple[str, Config, argparse.Namespace]:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["test", "pilot", "run", "analyze", "worker", "streamax"])
    ap.add_argument("paths", nargs="*", help=argparse.SUPPRESS)
    ap.add_argument("--shards", default="", help=argparse.SUPPRESS)
    for f in dataclasses.fields(Config):
        flag = "--" + f.name.replace("_", "-")
        if isinstance(f.default, bool):
            ap.add_argument(flag, action="store_true", default=f.default)
        elif isinstance(f.default, tuple):
            ap.add_argument(flag, type=lambda s: tuple(float(x) for x in s.split(",")), default=f.default)
        else:
            ap.add_argument(flag, type=type(f.default), default=f.default)
    a = ap.parse_args()
    cfg = Config(**{f.name: getattr(a, f.name) for f in dataclasses.fields(Config)})
    return a.command, cfg, a


def main() -> None:
    cmd, cfg, a = parse_args()
    if cmd == "streamax":
        return cmd_streamax(*a.paths)
    cfg.check()
    {"test": cmd_test, "pilot": cmd_pilot, "run": cmd_run, "analyze": cmd_analyze,
     "worker": lambda c: cmd_worker(c, a.shards)}[cmd](cfg)


if __name__ == "__main__":
    main()

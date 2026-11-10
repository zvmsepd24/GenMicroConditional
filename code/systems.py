"""The processing ranges studied: four Gaussian-random-field ranges and three Cahn-Hilliard ranges.

Every range has raw parameters theta (GRF: correlation lengths lx, ly in pixels; CH: mean
composition u_bar and ln t) drawn uniformly over a box. Models see each parameter scaled to
[-1, 1] over its own range, so the extent of a range in knob units is invisible to the network.

`late_red` is the `late` range with ln t written as the sum of two knobs, ln t = a + b, both in
[ln t0 / 2, ln t1 / 2]: three nominal parameters, two effective ones. For a training field with
ln t = s, a is drawn uniformly over the values that keep both knobs in range.
"""
import math
import sys

import torch

from genmicro.ch import simulate_at  # noqa: E402
from genmicro.chstats import descriptors as ch_descriptors  # noqa: E402

import grf  # noqa: E402
import potts  # noqa: E402

N_PX = 64
GRF_LAGS = (1, 2, 3, 4, 6, 8, 12, 16)
POTTS_LAGS = (1, 2, 3, 4, 6, 8, 12, 16)
POTTS_DIAG = (1, 2, 4)

SYSTEMS = {
    # Strong-response follow-up: separate identities preserve the original a=0.15 results.
    "osc_strong_m8": dict(kind="mix", box=((-1.0, 1.0), (-1.0, 1.0)), s=0.25, a=0.8, m=8),
    "osc_strong_m8_raw": dict(kind="mix", box=((-1.0, 1.0), (-1.0, 1.0)), s=0.25, a=0.8, m=8, embed="raw"),
    # 2 x 2 design: information (high / low) x texture scale (fine / coarse); exact I in nats
    "gA": dict(kind="grf", box=((2.0, 3.0), (2.0, 3.0))),     # fine, high information
    "gB": dict(kind="grf", box=((5.0, 12.0), (5.0, 12.0))),   # coarse, high information
    "gC": dict(kind="grf", box=((2.4, 2.6), (2.4, 2.6))),     # fine, low information
    "gD": dict(kind="grf", box=((7.3, 8.7), (7.3, 8.7))),     # coarse, low information
    "onset": dict(kind="ch", box=((-0.12, 0.12), (math.log(16.0), math.log(39.0)))),
    "late": dict(kind="ch", box=((-0.12, 0.12), (math.log(100.0), math.log(244.0)))),
    "late_red": dict(kind="chred", box=((-0.12, 0.12), (math.log(100.0), math.log(244.0)))),
    # v3: a narrow pilot inside the late range (a quarter of its width in each parameter), and a range
    # of the same extent and times as the late range that crosses from connected domains to droplets
    "late_narrow": dict(kind="ch", box=((-0.03, 0.03), (math.log(139.7), math.log(174.6)))),
    "transition": dict(kind="ch", box=((0.10, 0.34), (math.log(100.0), math.log(244.0)))),
    # Potts grain growth: temperature T (units of J) and ln(sweeps); sweeps are rounded to integers
    "potts_wide": dict(kind="potts", box=((0.4, 1.4), (math.log(24.0), math.log(96.0)))),
    "potts_narrow": dict(kind="potts", box=((0.775, 1.025), (math.log(40.4), math.log(57.1)))),
    "potts_early": dict(kind="potts", box=((0.4, 1.4), (math.log(8.0), math.log(24.0)))),
    # v4 accordion (Gaussian random fields, grf.mix_*): knobs are the scaled coordinates u themselves;
    # s sets the extent of the straight sheet, a the depth of the folds and m the number of folds
    "acc_m0": dict(kind="mix", box=((-1.0, 1.0), (-1.0, 1.0)), s=0.25, a=0.0, m=0),
    "acc_m1": dict(kind="mix", box=((-1.0, 1.0), (-1.0, 1.0)), s=0.25, a=0.15, m=1),
    "acc_m2": dict(kind="mix", box=((-1.0, 1.0), (-1.0, 1.0)), s=0.25, a=0.15, m=2),
    "acc_m4": dict(kind="mix", box=((-1.0, 1.0), (-1.0, 1.0)), s=0.25, a=0.15, m=4),
    # crossover test (added after the first accordion results): folds fine enough to cost the flow model
    "acc_m8": dict(kind="mix", box=((-1.0, 1.0), (-1.0, 1.0)), s=0.25, a=0.15, m=8),
    "acc_m16": dict(kind="mix", box=((-1.0, 1.0), (-1.0, 1.0)), s=0.25, a=0.15, m=16),
    # embedding test: same images, but the model gets the raw scaled knobs (no sinusoidal features,
    # whose frequencies pi 2^k include every fold frequency above)
    "acc_m0_raw": dict(kind="mix", box=((-1.0, 1.0), (-1.0, 1.0)), s=0.25, a=0.0, m=0, embed="raw"),
    "acc_m8_raw": dict(kind="mix", box=((-1.0, 1.0), (-1.0, 1.0)), s=0.25, a=0.15, m=8, embed="raw"),
    "acc_m16_raw": dict(kind="mix", box=((-1.0, 1.0), (-1.0, 1.0)), s=0.25, a=0.15, m=16, embed="raw"),
    # v3 ranges with the raw embedding (same images, pools and references via study.base_of)
    "gA_raw": dict(kind="grf", box=((2.0, 3.0), (2.0, 3.0)), embed="raw"),
    "gD_raw": dict(kind="grf", box=((7.3, 8.7), (7.3, 8.7)), embed="raw"),
    "onset_raw": dict(kind="ch", box=((-0.12, 0.12), (math.log(16.0), math.log(39.0))), embed="raw"),
    "late_raw": dict(kind="ch", box=((-0.12, 0.12), (math.log(100.0), math.log(244.0))), embed="raw"),
    "potts_wide_raw": dict(kind="potts", box=((0.4, 1.4), (math.log(24.0), math.log(96.0))), embed="raw"),
    "potts_narrow_raw": dict(kind="potts", box=((0.775, 1.025), (math.log(40.4), math.log(57.1))), embed="raw"),
    # encoding bandwidth sweep: sinusoidal features cut at the highest frequency pi 2^K ("f<K>")
    "acc_m0_f0": dict(kind="mix", box=((-1.0, 1.0), (-1.0, 1.0)), s=0.25, a=0.0, m=0, embed="f0"),
    "acc_m0_f2": dict(kind="mix", box=((-1.0, 1.0), (-1.0, 1.0)), s=0.25, a=0.0, m=0, embed="f2"),
    "acc_m4_f2": dict(kind="mix", box=((-1.0, 1.0), (-1.0, 1.0)), s=0.25, a=0.15, m=4, embed="f2"),
    "acc_m8_f0": dict(kind="mix", box=((-1.0, 1.0), (-1.0, 1.0)), s=0.25, a=0.15, m=8, embed="f0"),
    "acc_m8_f2": dict(kind="mix", box=((-1.0, 1.0), (-1.0, 1.0)), s=0.25, a=0.15, m=8, embed="f2"),
    # v5: raw-parameter versions of the sloppy GRF range and of the redundant range
    "gC_raw": dict(kind="grf", box=((2.4, 2.6), (2.4, 2.6)), embed="raw"),
    "late_red_raw": dict(kind="chred", box=((-0.12, 0.12), (math.log(100.0), math.log(244.0))), embed="raw"),
    "acc_m0_x4": dict(kind="mix", box=((-1.0, 1.0), (-1.0, 1.0)), s=1.0, a=0.0, m=0),
    "acc_m2_x4": dict(kind="mix", box=((-1.0, 1.0), (-1.0, 1.0)), s=1.0, a=0.15, m=2),
}


def lo_hi(sys_):
    b = torch.tensor(sys_["box"], dtype=torch.float64)
    return b[:, 0], b[:, 1]


def snap(sys_, theta):
    """Potts sweeps are integers: condition on ln of the rounded number of sweeps, which is what the
    simulator runs. Other systems are unchanged."""
    if sys_["kind"] != "potts":
        return theta
    theta = theta.clone()
    theta[:, 1] = torch.log(torch.round(torch.exp(theta[:, 1])))
    return theta


def draw(sys_, n, gen):
    """Raw parameters uniform over the box, (n, 2) float64 on the CPU."""
    lo, hi = lo_hi(sys_)
    return snap(sys_, lo + (hi - lo) * torch.rand(n, 2, generator=gen, dtype=torch.float64))


def normalize(sys_, theta):
    lo, hi = lo_hi(sys_)
    return (2 * (theta - lo) / (hi - lo) - 1).float()


def denormalize(sys_, c):
    lo, hi = lo_hi(sys_)
    return lo + (hi - lo) * (c.double().cpu() + 1) / 2


def render(sys_, theta, gen_cuda):
    """One field per parameter row, (n, N_PX, N_PX) float32 on the GPU."""
    if sys_["kind"] == "grf":
        return grf.sample(theta.cuda(), N_PX, gen_cuda)
    if sys_["kind"] == "mix":
        return grf.mix_sample(theta, sys_["s"], sys_["a"], sys_["m"], N_PX, gen_cuda)
    if sys_["kind"] == "potts":
        return potts.render(theta[:, 0].float(), torch.round(torch.exp(theta[:, 1])).long(), N_PX, generator=gen_cuda)
    return simulate_at(theta[:, 0].float(), torch.exp(theta[:, 1]).float(), n=N_PX, generator=gen_cuda)


def red_split(sys_, s, gen):
    """For ln t = s, draw knob a uniformly over [max(h0, s - h1), min(h1, s - h0)] and return the
    normalized pair (a, b) with b = s - a; h0, h1 = ln t0 / 2, ln t1 / 2."""
    h0, h1 = sys_["box"][1][0] / 2, sys_["box"][1][1] / 2
    a_lo = torch.clamp(s - h1, min=h0)
    a_hi = torch.clamp(s - h0, max=h1)
    a = a_lo + (a_hi - a_lo) * torch.rand(len(s), generator=gen, dtype=torch.float64)
    b = s - a
    return torch.stack([2 * (a - h0) / (h1 - h0) - 1, 2 * (b - h0) / (h1 - h0) - 1], 1).float()


def condition(sys_, theta, gen):
    """Network condition (n, k) in [-1, 1]: k = 2, or 3 for the redundant range."""
    c = normalize(sys_, theta)
    if sys_["kind"] != "chred":
        return c
    return torch.cat([c[:, :1], red_split(sys_, theta[:, 1], gen)], 1)


def n_cond(sys_):
    return 3 if sys_["kind"] == "chred" else 2


def grf_descriptors(x):
    """Directional autocorrelation at GRF_LAGS along x and y and log variance, (B, 17) float64."""
    x = x.double()
    x = x - x.mean((1, 2), keepdim=True)
    n = x.shape[-1]
    f = torch.fft.rfft2(x)
    s2 = torch.fft.irfft2(f * f.conj(), s=(n, n)) / (n * n)
    var = s2[:, 0, 0]
    cx = torch.stack([s2[:, 0, l] for l in GRF_LAGS], 1) / var[:, None]
    cy = torch.stack([s2[:, l, 0] for l in GRF_LAGS], 1) / var[:, None]
    return torch.cat([cx, cy, torch.log(var)[:, None]], 1)


def potts_descriptors(x):
    """Grain-boundary images (boundary where x > 0) -> (B, 13) float64: boundary fraction; two-point
    function of the boundary phase at POTTS_LAGS along the axes (averaged) and at POTTS_DIAG along the
    diagonals (averaged); log mean linear intercept of grain interiors along rows and columns."""
    b = (x > 0).float()
    n = b.shape[-1]
    f = torch.fft.rfft2(b)
    s2 = torch.fft.irfft2(f * f.conj(), s=(n, n)) / (n * n)
    ax = torch.stack([(s2[:, l, 0] + s2[:, 0, l]) / 2 for l in POTTS_LAGS], 1)
    dg = torch.stack([(s2[:, l, l] + s2[:, l, n - l]) / 2 for l in POTTS_DIAG], 1)
    inside = 1 - b
    runs = ((inside == 1) & (torch.roll(inside, 1, 2) == 0)).float().sum((1, 2)) \
        + ((inside == 1) & (torch.roll(inside, 1, 1) == 0)).float().sum((1, 2))
    intercept = 2 * inside.sum((1, 2)) / runs.clamp_min(1)
    return torch.cat([b.mean((1, 2))[:, None], ax, dg, torch.log(intercept.clamp_min(0.5))[:, None]], 1).double()


def descriptors(sys_, x):
    if sys_["kind"] in ("grf", "mix"):
        return grf_descriptors(x)
    if sys_["kind"] == "potts":
        return potts_descriptors(x)
    return ch_descriptors(x)


def augment(sys_, x, c, gen):
    """Exact symmetries. x: (B, 1, n, n), c: (B, k). Random periodic shifts and reflections for all
    systems; a transpose swaps the axes, which for the GRF swaps lx and ly in the condition."""
    b, _, n, _ = x.shape
    dy, dx = torch.randint(0, n, (2, b), device=x.device, generator=gen)
    ar = torch.arange(n, device=x.device)
    ys = (ar[None] + dy[:, None]) % n
    xs = (ar[None] + dx[:, None]) % n
    x = x[torch.arange(b, device=x.device)[:, None, None], 0, ys[:, :, None], xs[:, None, :]][:, None]
    f = torch.rand(3, b, device=x.device, generator=gen) < 0.5
    x = torch.where(f[0, :, None, None, None], x.flip(-1), x)
    x = torch.where(f[1, :, None, None, None], x.flip(-2), x)
    x = torch.where(f[2, :, None, None, None], x.transpose(-1, -2), x)
    if sys_["kind"] == "grf":
        c = torch.where(f[2, :, None], c.flip(1), c)
    if sys_["kind"] == "mix":  # a transpose swaps the x- and y-elongated components: u2 -> -u2
        c = torch.where(f[2, :, None], c * torch.tensor([1.0, -1.0], device=c.device), c)
    return x, c

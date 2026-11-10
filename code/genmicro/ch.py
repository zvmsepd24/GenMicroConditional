"""Batched Cahn-Hilliard solver on the GPU, in dimensionless form.

    du/dt = lap( u^3 - eps u - lap u ),   u = 2c - 1, periodic square grid.

Lengths are in units of the interface length xi = sqrt(kappa / |f''|) and time in units of
xi^2 / (M |f''|). An image of the field is then set by three natural variables:
* the mean composition u_bar;
* the dimensionless time t, which sets the coarsening stage and with it the ratio of the
  domain size to the interface width, L/xi;
* the interface resolution xi/h, the number of pixels per interface length (h is the grid
  spacing in units of xi). With a fixed physical pixel size this is the physical interface width.

In dimensional terms, t = t_phys M |f''|^2 / kappa and xi/h = sqrt(kappa / |f''|) / h_phys.
The mobility only enters through t; kappa enters through both.

eps > 0 is the quench depth (Landau form, eps ~ (Tc - T)/Tc): equilibrium phases are
u = +/- sqrt(eps), the interface width scales as 1/sqrt(eps) and the time scale as 1/eps^2.
eps = 1 is the reference quench.

Time stepping: linearly stabilised semi-implicit Fourier scheme (Eyre-type splitting). The scheme
under-estimates the growth rate of the initial pattern at large steps, so the step is held at
dt_early = 0.01 until t = 100; afterwards it grows by 2% per step up to 0.5, because coarsening
slows down (domain size ~ t^(1/3)). Halving dt_early changes the mean |u| by < 1.5% at t = 16 and
< 0.3% for t >= 25, and the characteristic length by < 0.2%.
"""
import math

import torch


def wavenumbers(n, h, device, dtype):
    """|k|^2 on an n x n grid for each grid spacing in h (B,) -> (B, n, n//2+1)."""
    k = 2 * math.pi * torch.fft.fftfreq(n, device=device, dtype=dtype)
    kr = 2 * math.pi * torch.fft.rfftfreq(n, device=device, dtype=dtype)
    base = k[:, None] ** 2 + kr[None, :] ** 2  # for unit grid spacing
    return base[None] / h.reshape(-1, 1, 1) ** 2


@torch.no_grad()
def simulate(u_bar, times, n=64, resolution=1.0, noise=0.05, dt0=0.01, dt_max=0.5, growth=1.02,
             stab=2.0, eps=1.0, generator=None, device="cuda", dtype=torch.float32, per_field=False,
             dt_early=0.01, t_early=100.0):
    """Run len(u_bar) independent fields.

    u_bar: (B,) mean compositions. resolution: scalar or (B,) pixels per interface length, xi/h.
    eps: scalar or (B,) quench depth. noise: initial fluctuation amplitude per interface-length
    cell (equal to the per-pixel amplitude at xi/h = 1).
    times: increasing output times (dimensionless); returns (len(times), B, n, n).
    With per_field=True, times is a (B,) list of one output time per field; returns (B, n, n).
    float32 is used because consumer GPUs run float64 far slower; the mean is re-imposed each
    step so round-off cannot drift the composition.
    """
    u_bar = torch.as_tensor(u_bar, dtype=dtype, device=device)
    b = len(u_bar)
    h = 1.0 / torch.as_tensor(resolution, dtype=dtype, device=device).expand(b)
    # noise is the amplitude per interface-length cell; cell-averaged white noise on a grid of
    # spacing h has standard deviation proportional to 1/h, so the per-pixel amplitude is noise*xi/h
    # and the physical initial condition does not change with resolution.
    sigma = (noise / h).reshape(-1, 1, 1)
    u = u_bar[:, None, None] + sigma * torch.randn(b, n, n, dtype=dtype, device=device, generator=generator)
    u = u - u.mean((1, 2), keepdim=True) + u_bar[:, None, None]  # exact mean
    k2 = wavenumbers(n, h, device, dtype)
    eps = torch.as_tensor(eps, dtype=dtype, device=device).reshape(-1, 1, 1)
    stab = stab * eps.max().item()
    if per_field:
        field_t = torch.as_tensor(times, dtype=torch.float64)
        out_t, inv = torch.unique(field_t, sorted=True, return_inverse=True)
        out_t = out_t.tolist()
        result = torch.empty(b, n, n, dtype=torch.float32, device=device)
    else:
        out_t = list(times)
        result = []
    t, dt, i = 0.0, dt0, 0
    while i < len(out_t):
        step = min(dt, out_t[i] - t)
        uh = torch.fft.rfft2(u)
        nl = torch.fft.rfft2(u ** 3 - (eps + stab) * u)
        uh = (uh - step * k2 * nl) / (1 + step * k2 * (stab + k2))
        u = torch.fft.irfft2(uh, s=(n, n))
        u = u - u.mean((1, 2), keepdim=True) + u_bar[:, None, None]
        t += step
        if abs(t - out_t[i]) < 1e-9:
            if per_field:
                idx = (inv == i).nonzero()[:, 0].to(device)
                result[idx] = u[idx].float()
            else:
                result.append(u.float().clone())
            i += 1
        # small steps while the pattern is still growing (the stabilised scheme under-estimates growth
        # rates at large steps); after t_early the step grows geometrically up to dt_max
        dt = min(dt * growth, dt_max) if t >= t_early else dt_early
    return result if per_field else torch.stack(result)


@torch.no_grad()
def simulate_at(u_bar, t, batch=1024, **kw):
    """One independent field per condition (u_bar[i], t[i]); returns (B, n, n) float32."""
    out = [simulate(u_bar[s:s + batch], t[s:s + batch], per_field=True, **kw) for s in range(0, len(u_bar), batch)]
    return torch.cat(out)


def structure_length(u):
    """Characteristic length 2*pi / <k> from the radially averaged structure factor, in pixels.

    u: (B, n, n). Returns (B,) lengths. Uses the first moment of S(k) over k > 0.
    """
    n = u.shape[-1]
    s = torch.fft.fft2(u - u.mean((-2, -1), keepdim=True)).abs().square()
    f = torch.fft.fftfreq(n, device=u.device)
    k = 2 * math.pi * torch.sqrt(f[:, None] ** 2 + f[None, :] ** 2)
    k1 = (s * k).sum((-2, -1)) / s.sum((-2, -1))
    return 2 * math.pi / k1

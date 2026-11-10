"""Anisotropic Gaussian random fields with an exact likelihood.

A field on an n x n periodic grid has a circulant covariance whose eigenvalues, the spectrum
lam_k, are

    S_k = exp(-(lx^2 kx^2 + ly^2 ky^2) / 2),   lam_k = (1 - tau2) S_k / mean(S) + tau2,

a Gaussian correlation with lengths (lx, ly) in pixels plus a white nugget tau2. The mean of lam_k
is 1, so every field has unit variance and the parameters act only through the shape of the
correlation. Because the covariance is circulant, sampling, the likelihood and the Fisher
information are exact in Fourier space:

    x = ifft2(sqrt(lam) * fft2(w)),  w ~ N(0, I)
    log p(x | theta) = -1/2 [ sum_k |fft2(x)_k|^2 / (n^2 lam_k) + sum_k log lam_k + n^2 log 2 pi ]
    G_ij(theta) = 1/2 sum_k d_i log lam_k  d_j log lam_k
"""
import math

import torch

TAU2 = 0.01


def kgrid(n, device="cuda", dtype=torch.float64):
    k = 2 * math.pi * torch.fft.fftfreq(n, device=device, dtype=dtype)
    return k[:, None], k[None, :]  # ky along rows (axis 0), kx along columns (axis 1)


def spectrum(theta, n, tau2=TAU2):
    """theta: (B, 2) = (lx, ly) in pixels -> lam (B, n, n)."""
    ky, kx = kgrid(n, theta.device, theta.dtype)
    lx, ly = theta[:, 0, None, None], theta[:, 1, None, None]
    s = torch.exp(-0.5 * (lx ** 2 * kx ** 2 + ly ** 2 * ky ** 2))
    return (1 - tau2) * s / s.mean((1, 2), keepdim=True) + tau2


@torch.no_grad()
def sample(theta, n=64, generator=None):
    """One exact field per row of theta; returns float32 (B, n, n)."""
    lam = spectrum(theta.double(), n)
    w = torch.randn(len(theta), n, n, device=theta.device, dtype=torch.float64, generator=generator)
    return torch.fft.ifft2(lam.sqrt() * torch.fft.fft2(w)).real.float()


def periodogram(x):
    """|fft2(x)|^2 / n^2, (B, n, n) float64."""
    n = x.shape[-1]
    return torch.fft.fft2(x.double()).abs().square() / n ** 2


def loglik_matrix(per, lam):
    """per: (M, n, n) periodograms; lam: (G, n, n) spectra -> (M, G) exact log-likelihoods."""
    n2 = per.shape[-1] * per.shape[-2]
    quad = per.flatten(1) @ (1 / lam).flatten(1).T
    logdet = torch.log(lam).flatten(1).sum(1)
    return -0.5 * (quad + logdet[None] + n2 * math.log(2 * math.pi))


def fisher(theta, n=64):
    """Exact Fisher information of one field, (B, 2, 2), in the units of theta."""
    theta = theta.double()
    # per-mode derivatives by central differences in float64, step 1e-5 of each length
    eps = 1e-5
    out = []
    for i in range(2):
        d = torch.zeros_like(theta)
        d[:, i] = eps * theta[:, i]
        lp = torch.log(spectrum((theta + d).detach(), n))
        lm = torch.log(spectrum((theta - d).detach(), n))
        out.append(((lp - lm) / (2 * d[:, i, None, None])).flatten(1))
    j = torch.stack(out, -1)  # (B, n^2, 2)
    return 0.5 * j.transpose(1, 2) @ j


def box_grid(box, m, device="cuda"):
    """Midpoint grid of m x m points over box ((lx0, lx1), (ly0, ly1)) -> (m*m, 2) float64."""
    ax = [torch.linspace(lo, hi, 2 * m + 1, device=device, dtype=torch.float64)[1::2] for lo, hi in box]
    gx, gy = torch.meshgrid(ax[0], ax[1], indexing="ij")
    return torch.stack([gx.flatten(), gy.flatten()], 1)


def volume(box, n=64, m=64):
    """Number of distinguishable fields in the box, V = int sqrt(det G) dtheta (midpoint rule)."""
    th = box_grid(box, m)
    g = fisher(th, n)
    cell = (box[0][1] - box[0][0]) * (box[1][1] - box[1][0]) / m ** 2
    return torch.sqrt(torch.linalg.det(g).clamp_min(0)).sum().item() * cell


@torch.no_grad()
def exact_mi(box, n=64, samples=4096, m=160, seed=0, chunk=256):
    """I(theta; X) in nats for theta uniform on the box, by Monte Carlo over (theta, x) and
    midpoint quadrature over theta' for the marginal p(x). Returns (mean, standard error)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    lo = torch.tensor([b[0] for b in box], device="cuda", dtype=torch.float64)
    hi = torch.tensor([b[1] for b in box], device="cuda", dtype=torch.float64)
    grid = box_grid(box, m)
    lam_grid = spectrum(grid, n)
    vals = []
    for s in range(0, samples, chunk):
        b = min(chunk, samples - s)
        th = lo + (hi - lo) * torch.rand(b, 2, device="cuda", dtype=torch.float64, generator=g)
        x = sample(th, n, g)
        per = periodogram(x)
        ll_true = loglik_matrix(per, spectrum(th, n)).diagonal()
        ll_grid = loglik_matrix(per, lam_grid)
        log_px = torch.logsumexp(ll_grid, 1) - math.log(len(grid))
        vals.append(ll_true - log_px)
    v = torch.cat(vals)
    return v.mean().item(), (v.std() / math.sqrt(len(v))).item()


# ---------------------------------------------------------------------------------------------
# v4 accordion family: spectra that are mixtures of five fixed unit-mean spectra, so that the mean
# of every second-order descriptor moves exactly as the mixture weights do. With knobs u in
# [-1, 1]^2 (the scaled coordinates the model sees):
#   coarse / fine        weights (1 -/+ s u1) / 4      a straight sheet ...
#   x- / y-elongated     weights (1 -/+ s u2) / 4
#   ring (wavelength 8)  weight  w_R = a (1 - cos(pi m u1)) / 2, the others scaled by 1 - w_R
# ... folded m times along u1 toward a labyrinth-like ring texture.

def _iso(l, n, device):
    ky, kx = kgrid(n, device)
    return torch.exp(-0.5 * l ** 2 * (kx ** 2 + ky ** 2))


def _aniso(lx, ly, n, device):
    ky, kx = kgrid(n, device)
    return torch.exp(-0.5 * (lx ** 2 * kx ** 2 + ly ** 2 * ky ** 2))


def _ring(k0, w, n, device):
    ky, kx = kgrid(n, device)
    return torch.exp(-0.5 * ((torch.sqrt(kx ** 2 + ky ** 2) - k0) / w) ** 2)


def mix_components(n=64, device="cuda", tau2=TAU2):
    """(5, n, n): fine, coarse, x-elongated, y-elongated, ring; each with mean 1."""
    raw = [_iso(2.0, n, device), _iso(4.0, n, device), _aniso(3.5, 2.0, n, device), _aniso(2.0, 3.5, n, device),
           _ring(2 * math.pi / 8, 0.12, n, device)]
    return torch.stack([(1 - tau2) * s / s.mean() + tau2 for s in raw])


def mix_weights(u, s, a, m):
    """u: (B, 2) scaled knobs -> (B, 5) mixture weights."""
    u = u.double()
    wr = a * (1 - torch.cos(math.pi * m * u[:, 0])) / 2
    base = torch.stack([1 + s * u[:, 0], 1 - s * u[:, 0], 1 + s * u[:, 1], 1 - s * u[:, 1]], 1) / 4
    return torch.cat([base * (1 - wr[:, None]), wr[:, None]], 1)


def mix_spectrum(u, s, a, m, n=64):
    comp = mix_components(n, u.device)
    return torch.einsum("bj,jxy->bxy", mix_weights(u, s, a, m).to(comp.device), comp)


@torch.no_grad()
def mix_sample(u, s, a, m, n=64, generator=None):
    lam = mix_spectrum(u.cuda().double(), s, a, m, n)
    w = torch.randn(len(u), n, n, device="cuda", dtype=torch.float64, generator=generator)
    return torch.fft.ifft2(lam.sqrt() * torch.fft.fft2(w)).real.float()

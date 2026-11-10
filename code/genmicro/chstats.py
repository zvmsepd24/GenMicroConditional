"""Descriptors and Fisher information for Cahn-Hilliard fields.

Descriptors of one field u (B, n, n), computed on the phase u > 0 unless noted:
phase fraction; two-point function at separations 1-32 px (average of the two axes); interface
density; mean |u| (gray-level contrast); log characteristic length.

For a set of states with descriptor means mu and covariances Sigma, the Fisher information of one
image about the parameters is G = J^T Sigma^-1 J with J = d mu / d theta. Descriptors are first
divided by a fixed scale (their spread across all states of the reference map), and a ridge
RIDGE is added to Sigma: changes smaller than sqrt(RIDGE) of that spread count as invisible.
"""
import torch

from genmicro.ch import structure_length

LAGS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
RIDGE = 1e-3
NAMES = ["phase_fraction"] + [f"S2_{l}" for l in LAGS] + ["interface_density", "mean_abs_u", "log_length"]


def descriptors(u):
    """u: (B, n, n) fields -> (B, D) float64 descriptors."""
    u = u.float()
    m = (u > 0).float()
    n = u.shape[-1]
    f = torch.fft.rfft2(m)
    s2 = torch.fft.irfft2(f * f.conj(), s=(n, n)) / (n * n)
    s2r = torch.stack([(s2[:, l, 0] + s2[:, 0, l]) / 2 for l in LAGS], 1)
    phi = m.mean((1, 2))
    interface = ((m != m.roll(1, 1)).float().mean((1, 2)) + (m != m.roll(1, 2)).float().mean((1, 2))) / 2
    contrast = u.abs().mean((1, 2))
    length = structure_length(u)
    return torch.cat([phi[:, None], s2r, interface[:, None], contrast[:, None], torch.log(length)[:, None]], 1).double()


def shrink_cov(x):
    """Covariance shrunk 10% towards its diagonal, for stability with ~100 samples."""
    c = torch.cov(x.T)
    return 0.9 * c + 0.1 * torch.diag(torch.diag(c))


def standardized(mu, cov, scale):
    eye = torch.eye(mu.shape[-1], dtype=cov.dtype, device=cov.device)
    return mu / scale, cov / (scale[:, None] * scale[None, :]) + RIDGE * eye


def fisher(mu, cov, scale, spacing):
    """mu: (*grid, D), cov: (*grid, D, D), spacing per grid axis -> G (*grid, P, P)."""
    mu, cov = standardized(mu, cov, scale)
    jac = torch.stack([torch.gradient(mu, spacing=(s,), dim=a)[0] for a, s in enumerate(spacing)], -1)
    return jac.transpose(-1, -2) @ torch.linalg.solve(cov, jac)


def scatter_distance(mu_a, mu_b, cov_ref, scale):
    """Distance between two descriptor means in units of the realization-to-realization scatter
    of the reference: sqrt((a-b)^T Sigma^-1 (a-b)), after standardisation. Same units as the
    Fisher distance: 1 is about the smallest difference a single image can reveal."""
    eye = torch.eye(mu_a.shape[-1], dtype=cov_ref.dtype, device=cov_ref.device)
    c = cov_ref / (scale[:, None] * scale[None, :]) + RIDGE * eye
    d = (mu_a - mu_b) / scale
    return torch.sqrt((d[..., None, :] @ torch.linalg.solve(c, d[..., None]))[..., 0, 0])

"""q-state Potts model of grain growth on the GPU.

Spins s in {0, ..., q-1} on an n x n periodic square lattice; each spin interacts with its 8 nearest
and next-nearest neighbors (Moore neighborhood), energy J * (number of unlike neighbors), J = 1.
Monte Carlo: the lattice is split into four sublattices by (row parity, column parity), so that no two
sites updated together are neighbors; each site proposes the spin of a randomly chosen neighbor and
accepts with the Metropolis probability min(1, exp(-dE / T)). A site inside a grain proposes its own
spin, so grains have no interior fluctuations and only boundaries move. One sweep updates all four
sublattices. Fields start from independent random spins and coarsen; the image of a field is its
grain-boundary map (+1 where a pixel differs from any of its 4 nearest neighbors, -1 elsewhere).
"""
import math

import torch

Q = 64
SHIFTS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def neighbors(s):
    return torch.stack([torch.roll(s, (dy, dx), (1, 2)) for dy, dx in SHIFTS])  # (8, B, n, n)


@torch.no_grad()
def simulate(T, sweeps, n=64, q=Q, generator=None, device="cuda"):
    """One field per entry of T (B,) and integer sweeps (B,); returns spins (B, n, n) int64."""
    T = torch.as_tensor(T, dtype=torch.float32, device=device)
    sweeps = torch.as_tensor(sweeps, dtype=torch.long, device=device)
    b = len(T)
    s = torch.randint(0, q, (b, n, n), device=device, generator=generator)
    beta = (1 / T).view(b, 1, 1)
    yy, xx = torch.meshgrid(torch.arange(n, device=device), torch.arange(n, device=device), indexing="ij")
    masks = [((yy % 2 == py) & (xx % 2 == px)) for py in (0, 1) for px in (0, 1)]
    out = torch.empty_like(s)
    max_sweeps = int(sweeps.max())
    for t in range(1, max_sweeps + 1):
        for m in masks:
            nb = neighbors(s)
            k = torch.randint(0, 8, (1, b, n, n), device=device, generator=generator)
            prop = torch.gather(nb, 0, k)[0]
            e_old = (nb != s).sum(0)
            e_new = (nb != prop).sum(0)
            de = (e_new - e_old).float()
            u = torch.rand(b, n, n, device=device, generator=generator)
            accept = (de <= 0) | (u < torch.exp(-beta * de))
            s = torch.where(m & accept, prop, s)
        done = sweeps == t
        if done.any():
            out[done] = s[done]
    return out


def boundary_image(s):
    """+1 on grain boundaries (a 4-neighbor differs), -1 inside grains; float32 (B, n, n)."""
    diff = torch.zeros_like(s, dtype=torch.bool)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        diff |= s != torch.roll(s, (dy, dx), (1, 2))
    return diff.float() * 2 - 1


def render(T, sweeps, n=64, generator=None, batch=1024):
    out = []
    for i in range(0, len(T), batch):
        out.append(boundary_image(simulate(T[i:i + batch], sweeps[i:i + batch], n, generator=generator)))
    return torch.cat(out)

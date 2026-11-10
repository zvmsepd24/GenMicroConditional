"""v5: geometry of a stochastic microstructure manifold, from data and from a conditional flow model.

A processing range is a family of image distributions P_u, with u the scaled processing parameters in
[-1, 1]^k that the models receive. Its natural metric is the Fisher information of u,
G(u) = E[d_u log p(x|u) d_u log p(x|u)^T]: how far apart nearby settings are in units of the spread of
the ensemble. Three estimates:

- exact_fisher: Gaussian random fields, G_jk = 1/2 sum_k d_j lam_k d_k lam_k / lam_k^2.
- model_fisher: from a model's conditional denoiser xhat(y, u) = E[x | x + sigma eps, u], through
      d_u xhat = Cov(x, d_u log p(x|u) | y)   and   dJ_gamma/dgamma = E[d_u xhat d_u xhat^T],
  where J_gamma is the Fisher information of u in sqrt(gamma) x + eps (gamma = 1 / sigma^2), so
      G(u) = int_0^inf E[d_u xhat d_u xhat^T] dgamma = int E[d_u xhat d_u xhat^T] 2 sigma^-2 dln sigma.
  The derivative is a central difference in the network's condition input, in float32.
- desc_fisher: J^T C^-1 J for a vector of descriptors T with mean derivative J and covariance C, a lower
  bound on the image Fisher metric (Cramer-Rao for the statistic T, then data processing). J comes from
  paired samples at u +- h e_j (common random numbers: the same noise for real renders, the same
  initial noise for generated images); the bias of the plug-in estimate from the scatter of the
  paired differences is subtracted.
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import grf  # noqa: E402
import study as St  # noqa: E402
import systems as S  # noqa: E402

torch.backends.cudnn.allow_tf32 = False  # TF32 rounding (1e-3) swamps the finite differences in the condition
torch.backends.cuda.matmul.allow_tf32 = False
SIGMAS = np.geomspace(0.01, 100.0, 29)
SIG_VIS = 0.5


def exact_lam(sys_, u):
    """Spectrum (B, n, n) float64 at scaled parameters u (B, 2)."""
    u = u.double()
    if sys_["kind"] == "grf":
        return grf.spectrum(S.denormalize(sys_, u).cuda(), S.N_PX)
    return grf.mix_spectrum(u.cuda(), sys_["s"], sys_["a"], sys_["m"], S.N_PX)


def exact_fisher(sys_, u, h=1e-4, sigmas=SIGMAS):
    """Exact metric (B, 2, 2) and its integrand per dln sigma (B, S, 2, 2) on the sigma grid."""
    lam = exact_lam(sys_, u)
    dl = []
    for j in range(2):
        e = torch.zeros_like(u, dtype=torch.float64)
        e[:, j] = h
        dl.append((exact_lam(sys_, u + e) - exact_lam(sys_, u - e)) / (2 * h))
    dl = torch.stack(dl, 1)  # (B, 2, n, n)
    G = 0.5 * torch.einsum("bjxy,bkxy->bjk", dl / lam[:, None], dl / lam[:, None])
    integ = []
    for s in sigmas:
        w = 2 * s ** 2 / (lam + s ** 2) ** 3
        integ.append(torch.einsum("bjxy,bkxy->bjk", dl * w[:, None], dl))
    return G.cpu(), torch.stack(integ, 1).cpu()


def exact_fisher_h(sys_, u, h, sigmas=SIGMAS):
    """Exact counterpart of model_fisher at a finite step h: the optimal linear denoisers at u +- h e_j
    applied to noisy images of the ensemble at u, E||xhat_+ - xhat_-||^2 / (2h)^2 integrated over
    gamma. Tends to the Fisher metric as h -> 0. Returns (B, 2, 2)."""
    lam = exact_lam(sys_, u)
    integ = []
    for s in sigmas:
        dx = []
        for j in range(2):
            e = torch.zeros_like(u, dtype=torch.float64)
            e[:, j] = h
            lp, lm = exact_lam(sys_, u + e), exact_lam(sys_, u - e)
            dx.append((lp / (lp + s ** 2) - lm / (lm + s ** 2)) / (2 * h))
        dx = torch.stack(dx, 1)
        integ.append(torch.einsum("bjxy,bkxy->bjk", dx * (lam + s ** 2)[:, None], dx) * 2 / s ** 2)
    return integrate(torch.stack(integ, 1).cpu(), sigmas)


def integrate(integrand, sigmas=SIGMAS):
    """Trapezoid over ln sigma of an integrand (..., S, k, k) given per dln sigma."""
    ls = torch.tensor(np.log(sigmas), dtype=integrand.dtype)
    return torch.trapezoid(integrand, ls, dim=-3)


@torch.no_grad()
def model_fisher(model, x, c, sigmas=SIGMAS, h=2e-3, seed=0, batch=128, keep_vis=False):
    """Fisher metric of the model at one setting from real images x (R, n, n) with conditions c (R, k).
    Returns the integrand per dln sigma (S, k, k) and, if keep_vis, d_u xhat of the first image at
    sigma = SIG_VIS (k, n, n) with that image and its noisy version."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    R, k = c.shape
    out, vis = [], None
    for s in sigmas:
        t = 1.0 / (1.0 + s)
        M = torch.zeros(k, k, dtype=torch.float64)
        for b0 in range(0, R, batch):
            xb, cb = x[b0:b0 + batch, None], c[b0:b0 + batch]
            nb = len(xb)
            xt = (1 - t) * torch.randn(xb.shape, device="cuda", generator=g) + t * xb
            tt = torch.full((nb,), t, device="cuda")
            one = torch.ones(nb, device="cuda")
            D = []
            for j in range(k):
                e = torch.zeros_like(cb)
                e[:, j] = h
                vp = model(xt, tt, St.flagged(cb + e, one))
                vm = model(xt, tt, St.flagged(cb - e, one))
                D.append(((1 - t) * (vp - vm) / (2 * h)).flatten(1).double())
            D = torch.stack(D, 1)  # (nb, k, P)
            M += torch.einsum("bjp,bkp->jk", D, D).cpu()
            if keep_vis and vis is None and abs(s - SIG_VIS) < 1e-9 + 0.06 * SIG_VIS:
                vis = dict(d_xhat=D[0].reshape(k, *xb.shape[-2:]).float().cpu(), x=xb[0, 0].cpu(), y=(xt[0, 0] / t).cpu())
        out.append(M / R * 2 / s ** 2)
    return torch.stack(out), vis


def grid_model_fisher(model, sys_, theta, R=128, seed=8300, sigmas=SIGMAS, keep_vis=False, h=2e-3):
    """Model metric at each row of theta: renders R real images there. Returns G (G, k, k), the
    integrand (G, S, k, k) and, if keep_vis, one tangent picture per setting."""
    integ, vis = [], []
    for i, row in enumerate(theta):
        th = row[None].repeat(R, 1)
        x = S.render(sys_, th, torch.Generator(device="cuda").manual_seed(seed + i))
        c = S.condition(sys_, th, torch.Generator().manual_seed(seed + 50 + i)).cuda()
        f, v = model_fisher(model, x, c, sigmas=sigmas, h=h, seed=seed + 100 + i, keep_vis=keep_vis)
        integ.append(f)
        vis.append(v)
    integ = torch.stack(integ)
    return integrate(integ, sigmas), integ, vis


def desc_fisher_from_pairs(T_plus, T_minus, h):
    """T_plus, T_minus: lists over directions j of paired descriptor arrays (R, D) at u +- h e_j.
    Returns the debiased J^T C^-1 J (k, k)."""
    k = len(T_plus)
    d = [(tp - tm) / (2 * h) for tp, tm in zip(T_plus, T_minus)]
    R, Dm = d[0].shape
    C = torch.stack([torch.cov(t.T) for t in T_plus + T_minus]).mean(0)
    C = C + 1e-8 * torch.trace(C) / Dm * torch.eye(Dm, dtype=C.dtype, device=C.device)
    Ci = torch.linalg.inv(C) * (R * 2 * k - Dm - 2) / (R * 2 * k - 1)  # unbiased inverse (Wishart)
    J = torch.stack([x.mean(0) for x in d])  # (k, D)
    G = J @ Ci @ J.T
    for a in range(k):
        for b in range(k):
            cab = ((d[a] - d[a].mean(0)).T @ (d[b] - d[b].mean(0))) / (R - 1)
            G[a, b] -= torch.trace(Ci @ cab) / R
    return G


def real_desc_fisher(sys_, theta_row, R, h, seed):
    """Descriptor metric of the real ensemble at one setting, in the coordinates of the range's own
    scaled parameters (2 of them; for the redundant range use the late range and transform)."""
    u0 = S.normalize(sys_, theta_row[None]).double()
    Tp, Tm = [], []
    for j in range(2):
        for sign, store in ((1, Tp), (-1, Tm)):
            u = u0.clone()
            u[0, j] += sign * h
            th = S.denormalize(sys_, u).repeat(R, 1)
            x = S.render(sys_, th, torch.Generator(device="cuda").manual_seed(seed))
            store.append(S.descriptors(sys_, x).double())
    return desc_fisher_from_pairs(Tp, Tm, h)


@torch.no_grad()
def gen_desc_fisher(model, sys_, theta_row, R, h, seed):
    """Descriptor metric of the model's generated ensemble at one setting, in the network's condition
    coordinates; plus and minus share the initial noise."""
    c0 = S.condition(sys_, theta_row[None].repeat(R, 1), torch.Generator().manual_seed(seed)).cuda()
    k = c0.shape[1]
    Tp, Tm = [], []
    for j in range(k):
        for sign, store in ((1, Tp), (-1, Tm)):
            c = c0.clone()
            c[:, j] += sign * h
            x0 = torch.randn(R, 1, S.N_PX, S.N_PX, device="cuda", generator=torch.Generator(device="cuda").manual_seed(seed))
            x = St.sample(model, x0, St.SAMPLER["steps"], cond=St.flagged(c, torch.ones(R, device="cuda")), rho=St.SAMPLER["rho"])  # float32
            store.append(S.descriptors(sys_, x[:, 0].float()).double())
    return desc_fisher_from_pairs(Tp, Tm, h)

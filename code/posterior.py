"""Variational (Barber-Agakov) estimate of I(theta; X) from N fields, one per condition.

The parameters, scaled to [-1, 1] over the range, are mapped to the real line by a logit,
z = logit((c + 1) / 2). A small translation-invariant CNN outputs a Gaussian posterior q(z | x)
with full covariance. For any q, I(theta; X) = I(z; X) >= h(z) + E[log q(z | x)] on fields not
used to fit q, and h(z) = 2 nats per parameter for c uniform on [-1, 1].

Cross-fitting uses only the N fields: they are split into two halves; q is fitted on one half
(early stopping on a quarter of it) and scored on the other, and the roles are swapped. The fitted
networks are also scored on a large independent test set, which removes the scoring noise but not
the fitting noise. At scoring, q is averaged over eight label-preserving symmetries of the field
(a valid density, so the bound holds).

Usage: python posterior.py NAME[:binary] [NAME ...]
"""
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import systems as S  # noqa: E402

OUT = HERE.parent / "results"
N_EST = [128, 512, 2048]
REPLICATES = {128: 1, 512: 1, 2048: 1}
POOL, TEST = 2048, 1024


BIN_EDGES = (0, 1, 2, 3, 4, 6, 8, 11, 16, 23, 33)  # |k| index bins along each axis (n = 64)


def spectral_features(x):
    """Log power spectrum averaged over bins of |k_x| and |k_y|, (B, 100). Second-order statistics
    of a stationary field; a transpose of the field transposes the bin matrix."""
    n = x.shape[-1]
    p = torch.fft.fft2(x[:, 0] - x[:, 0].mean((-2, -1), keepdim=True)).abs().square() / n ** 2
    idx = torch.arange(n, device=x.device)
    fold = torch.minimum(idx, n - idx)
    b = torch.bucketize(fold, torch.tensor(BIN_EDGES[1:], device=x.device), right=True)
    nb = len(BIN_EDGES) - 1
    onehot = F.one_hot(b, nb).float()  # (n, nb)
    s = torch.einsum("bij,ip,jq->bpq", p, onehot, onehot) / torch.einsum("ip,jq->pq", onehot, onehot)
    return torch.log(s + 1e-6).flatten(1)


class PostNet(nn.Module):
    """Convolutional features (spatially averaged) and the binned log power spectrum -> Gaussian
    posterior with full covariance over the logit-transformed parameters."""

    def __init__(self, k, w=32):
        super().__init__()
        conv = lambda a, b, s=1: nn.Conv2d(a, b, 3, stride=s, padding=1, padding_mode="circular")
        gn = lambda c: nn.GroupNorm(8, c)
        self.body = nn.Sequential(conv(1, w), gn(w), nn.SiLU(), conv(w, w), gn(w), nn.SiLU(),
                                  conv(w, 2 * w, 2), gn(2 * w), nn.SiLU(), conv(2 * w, 2 * w), gn(2 * w), nn.SiLU(),
                                  conv(2 * w, 4 * w, 2), gn(4 * w), nn.SiLU(), conv(4 * w, 4 * w), gn(4 * w), nn.SiLU())
        nf = (len(BIN_EDGES) - 1) ** 2
        self.spec = nn.Sequential(nn.LayerNorm(nf), nn.Linear(nf, 128), nn.SiLU())
        self.k = k
        self.head = nn.Sequential(nn.Linear(4 * w + 128, 128), nn.SiLU(), nn.Linear(128, k + k * (k + 1) // 2))
        self.tril = torch.tril_indices(k, k)

    def forward(self, x):
        out = self.head(torch.cat([self.body(x).mean((2, 3)), self.spec(spectral_features(x))], 1))
        m, raw = out[:, :self.k], out[:, self.k:]
        L = torch.zeros(len(x), self.k, self.k, device=x.device)
        L[:, self.tril[0], self.tril[1]] = raw
        diag = F.softplus(torch.diagonal(L, dim1=1, dim2=2)) + 1e-3
        L = torch.tril(L, -1) + torch.diag_embed(diag)
        return m, L


def log_q(m, L, z):
    r = torch.linalg.solve_triangular(L, (z - m)[..., None], upper=False)[..., 0]
    return -0.5 * r.square().sum(1) - torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(1) - 0.5 * z.shape[1] * math.log(2 * math.pi)


def to_z(c):
    u = ((c + 1) / 2).clamp(1e-6, 1 - 1e-6)
    return torch.log(u) - torch.log1p(-u)


def symmetries(x):
    """Eight label-preserving copies: flips along x, y and a half-period shift."""
    n = x.shape[-1]
    out = []
    for fx in (False, True):
        for fy in (False, True):
            for sh in (0, n // 2):
                y = x.flip(-1) if fx else x
                y = y.flip(-2) if fy else y
                out.append(y.roll((sh, sh), (-2, -1)))
    return out


@torch.no_grad()
def score(net, x, z, batch=512):
    """log of the symmetry-averaged posterior density at the true z, per field."""
    out = []
    for s in range(0, len(x), batch):
        xb, zb = x[s:s + batch, None], z[s:s + batch]
        lq = torch.stack([log_q(*net(v), zb) for v in symmetries(xb)])
        out.append(torch.logsumexp(lq, 0) - math.log(len(lq)))
    return torch.cat(out)


def fit(sys_, x, c, seed):
    """Fit q on fields x (n, N_PX, N_PX) with conditions c (n, 2); early stopping on 25%."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    n = len(x)
    perm = torch.randperm(n, device="cuda", generator=g)
    n_val = max(4, n // 4)
    vi, ti = perm[:n_val], perm[n_val:]
    net = PostNet(c.shape[1]).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    steps = int(min(1500, max(400, 2 * len(ti))))
    warm = int(0.4 * steps)  # mean by squared error first: the Gaussian likelihood alone first inflates
    best, best_state = -1e9, None  # the variance and then barely moves the mean
    for step in range(steps):
        idx = ti[torch.randint(0, len(ti), (64,), device="cuda", generator=g)]
        xb, cb = S.augment(sys_, x[idx, None], c[idx], g)
        m, L = net(xb)
        z = to_z(cb)
        loss = (m - z).square().mean() if step < warm else -log_q(m, L, z).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step >= warm and (step + 1) % 50 == 0:
            net.eval()
            v = score(net, x[vi], to_z(c[vi])).mean().item()
            net.train()
            if v > best:
                best, best_state = v, {k: t.clone() for k, t in net.state_dict().items()}
    net.load_state_dict(best_state)
    return net.eval()


def data(name, sys_, n, seed, binary):
    gen = torch.Generator().manual_seed(seed)
    theta = S.draw(sys_, n, gen)
    x = S.render(sys_, theta, torch.Generator(device="cuda").manual_seed(seed + 1))
    if binary:
        x = torch.where(x > 0, 1.0, -1.0)
    return x, S.normalize(sys_, theta).cuda()


def run(job):
    name, _, mode = job.partition(":")
    binary = mode == "binary"
    sys_ = S.SYSTEMS[name]
    k = 2
    t0 = time.perf_counter()
    base = 50000 + sum(map(ord, name)) * 10
    x, c = data(name, sys_, POOL, base, binary)
    xt, ct = data(name, sys_, TEST, base + 5, binary)
    rows = []
    for n in N_EST:
        for r in range(REPLICATES[n]):
            sl = slice(r * n, (r + 1) * n)
            xa, ca = x[sl], c[sl]
            half = n // 2
            folds = [(slice(0, half), slice(half, n)), (slice(half, n), slice(0, half))]
            cross, test = [], []
            for f, (tr, te) in enumerate(folds):
                net = fit(sys_, xa[tr], ca[tr], seed=base + 100 * n + 10 * r + f)
                cross.append(score(net, xa[te], to_z(ca[te])))
                test.append(score(net, xt, to_z(ct)).mean().item())
            lq = torch.cat(cross)
            rows.append(dict(name=name, binary=binary, n=n, replicate=r,
                             i_cross=2 * k + lq.mean().item(), i_cross_se=(lq.std() / math.sqrt(len(lq))).item(),
                             i_test=2 * k + sum(test) / 2))
            print(json.dumps({a: (round(v, 3) if isinstance(v, float) else v) for a, v in rows[-1].items()}), flush=True)
    tag = f"{name}_binary" if binary else name
    (OUT / f"posterior_{tag}.json").write_text(json.dumps(dict(name=name, binary=binary, seconds=time.perf_counter() - t0, rows=rows), indent=1))


if __name__ == "__main__":
    torch.cuda.set_per_process_memory_fraction(0.3)
    torch.backends.cudnn.benchmark = True
    for job in sys.argv[1:]:
        run(job)

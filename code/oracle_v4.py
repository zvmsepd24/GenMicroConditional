"""Gaussian oracle for the Gaussian-random-field ranges: the best linear-denoiser conditional model.

A Gaussian random field is fixed by its spectrum, and the optimal denoiser at each setting is
linear, so a conditional model of these ranges only has to estimate the spectrum lam(u) as a function
of the scaled knobs u. The oracle does so by a kernel smoother of the training periodograms,

    lam_hat(u) = sum_i K_h(u - u_i) P_i / sum_i K_h(u - u_i),   P_i = |fft2(x_i)|^2 / n^2,

with a Gaussian kernel of widths (h1, h2) along the two knobs, using the same training pools as the
flow models plus their transposed copies (the flow models see transposes as augmentation; a
transpose swaps lx and ly for the GRF ranges and negates u2 for the accordion ranges). At each
training-set size the widths are chosen from a grid to give the lowest trend error against the
reference (an oracle choice). Fields drawn from lam_hat are scored exactly as the flow models are.

Usage: python oracle_v4.py RANGE [RANGE ...]   -> results/oracle_v4_{RANGE}.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import grf  # noqa: E402
import study as St  # noqa: E402
import systems as S  # noqa: E402

OUT = HERE.parent / "results"
WIDTHS = [0.05, 0.1, 0.2, 0.35, 0.6, 1.0, 3.0]


@torch.no_grad()
def score(sys_, lam, ref, scale, seed):
    """Trend error (median over the grid) of fields drawn from spectra lam (G, n, n)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    n = lam.shape[-1]
    w = torch.randn(len(lam), St.N_GEN, n, n, device="cuda", dtype=torch.float64, generator=g)
    x = torch.fft.ifft2(lam[:, None].clamp_min(1e-8).sqrt() * torch.fft.fft2(w)).real.float()
    gs = St.gen_stats(sys_, x)
    return St.split_distance(gs, ref, scale)[1].median().item()


def run(name):
    sys_ = S.SYSTEMS[name]
    ref = St.reference(name, sys_, 100)
    scale = St.scale_of(ref)
    ug = S.normalize(sys_, St.grid_theta(sys_)).cuda().double()
    sets = sorted({int(p.stem.rsplit("_", 1)[1]) for p in OUT.glob(f"study_{name}_*.json") if p.stem.rsplit("_", 1)[0] == f"study_{name}"})
    sets = sets or [0, 1]  # before any flow run: the training pools the flow models will use
    res = dict(range=name, sets=sets, widths=WIDTHS, rows=[])
    for split in sets:
        theta, cond, fields = St.pool(name, sys_, split, max(St.LADDER))
        u = S.normalize(sys_, theta).cuda().double()
        per = grf.periodogram(fields)
        for n in St.LADDER:
            uu = torch.cat([u[:n], u[:n].flip(1) if sys_["kind"] == "grf" else u[:n] * torch.tensor([1.0, -1.0], device="cuda", dtype=torch.float64)])
            pp = torch.cat([per[:n], per[:n].transpose(-1, -2)])
            best = None
            for h1 in WIDTHS:
                for h2 in WIDTHS:
                    k = torch.exp(-0.5 * (((ug[:, None, 0] - uu[None, :, 0]) / h1) ** 2 + ((ug[:, None, 1] - uu[None, :, 1]) / h2) ** 2))
                    k = k / k.sum(1, keepdim=True).clamp_min(1e-300)
                    lam = torch.einsum("gi,ixy->gxy", k, pp)
                    e = score(sys_, lam, ref, scale, seed=31)
                    if best is None or e < best[0]:
                        best = (e, h1, h2)
            res["rows"].append(dict(split=split, n=n, processing_median=best[0], h1=best[1], h2=best[2]))
            print(name, split, n, [round(b, 3) if isinstance(b, float) else b for b in best], flush=True)
    (OUT / f"oracle_v4_{name}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    torch.cuda.set_per_process_memory_fraction(0.6)
    for nm in sys.argv[1:]:
        run(nm)

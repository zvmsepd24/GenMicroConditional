"""Exact I(theta; sqrt(gamma) X + noise) against log gamma for the GRF ranges."""
import json, math, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).parent))
import systems as S
from grf import spectrum, sample, periodogram, loglik_matrix, box_grid

@torch.no_grad()
def mi_noisy(box, gamma, n=64, samples=2048, m=128, seed=0, chunk=256):
    g = torch.Generator(device="cuda").manual_seed(seed)
    lo = torch.tensor([b[0] for b in box], device="cuda", dtype=torch.float64)
    hi = torch.tensor([b[1] for b in box], device="cuda", dtype=torch.float64)
    grid = box_grid(box, m); lam_grid = gamma * spectrum(grid, n) + 1
    vals = []
    for s in range(0, samples, chunk):
        th = lo + (hi - lo) * torch.rand(chunk, 2, device="cuda", dtype=torch.float64, generator=g)
        x = sample(th, n, g).double()
        y = math.sqrt(gamma) * x + torch.randn(x.shape, device="cuda", dtype=torch.float64, generator=g)
        per = periodogram(y)
        llt = loglik_matrix(per, gamma * spectrum(th, n) + 1).diagonal()
        llg = loglik_matrix(per, lam_grid)
        vals.append(llt - (torch.logsumexp(llg, 1) - math.log(len(grid))))
    return torch.cat(vals).mean().item()

out = {}
lams = [-6, -4, -2, -1, 0, 1, 2, 3, 4, 5, 6, 8, 10]
for name in ("gA", "gB", "gC", "gD"):
    box = S.SYSTEMS[name]["box"]
    out[name] = [mi_noisy(box, math.exp(l)) for l in lams]
    print(name, [round(v, 3) for v in out[name]], flush=True)
(Path(__file__).parent.parent / "results" / "exact_snr_curve.json").write_text(json.dumps(dict(log_snr=lams, I=out), indent=1))

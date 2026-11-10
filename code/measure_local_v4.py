"""Local fiber dimension d(sigma | theta) at every evaluation setting of a range (added during the historical v4 plan
after the range-averaged measurement: local intrinsic dimension varies over the manifold, and a range
average blurs regions where fiber directions collapse with regions where they do not).

For each grid setting theta_i, R real images are rendered at theta_i and the conditional denoising
error of each saved N = 512 model gives d(sigma | theta_i) = E||x - xhat||^2 / sigma^2 (see
measure_v4.py). For Gaussian random fields the exact local value sum_k lam_k(theta_i) /
(lam_k(theta_i) + sigma^2) is stored alongside. Compared in analyze_v4.py with the local trend error
of the models at the same settings.

Usage: python measure_local_v4.py RANGE [RANGE ...]   -> results/local_v4_{RANGE}.json
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import grf  # noqa: E402
import study as St  # noqa: E402
import systems as S  # noqa: E402
from measure_v4 import SIGMAS, dsm_spectrum, load_model  # noqa: E402

OUT = HERE.parent / "results"
R = 64


def run(name):
    sys_ = S.SYSTEMS[name]
    sets = sorted(int(p.stem.rsplit("_", 1)[1]) for p in OUT.glob(f"model_{name}_*.pt") if p.stem.rsplit("_", 1)[0] == f"model_{name}")
    th = St.grid_theta(sys_)
    g = torch.Generator(device="cuda").manual_seed(8100)
    x = S.render(sys_, th.repeat_interleave(R, 0), g)
    c = S.condition(sys_, th.repeat_interleave(R, 0), torch.Generator().manual_seed(8200)).cuda()
    res = dict(range=name, sets=sets, sigmas=SIGMAS.tolist(), grid=th.tolist(), d_local=[], exact_local=None)
    for split in sets:
        model = load_model(name, split)
        per = []
        for i in range(len(th)):
            dc, _ = dsm_spectrum(model, x[i * R:(i + 1) * R], c[i * R:(i + 1) * R], seed=31 + i)
            per.append(dc)
        res["d_local"].append(per)
        del model
    if sys_["kind"] in ("grf", "mix"):
        lam = grf.spectrum(th.cuda().double(), S.N_PX) if sys_["kind"] == "grf" else \
            grf.mix_spectrum(S.normalize(sys_, th).cuda().double(), sys_["s"], sys_["a"], sys_["m"], S.N_PX)
        res["exact_local"] = [[(lam[i] / (lam[i] + s ** 2)).sum().item() for s in SIGMAS] for i in range(len(th))]
    (OUT / f"local_v4_{name}.json").write_text(json.dumps(res))
    print(name, "done", flush=True)


if __name__ == "__main__":
    torch.cuda.set_per_process_memory_fraction(0.6)
    for nm in sys.argv[1:]:
        run(nm)

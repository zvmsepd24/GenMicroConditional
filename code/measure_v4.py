"""v4 measurements on saved models (N = 512): fiber dimension spectrum and bias-variance split.

* Fiber dimension spectrum d(sigma). On the straight path x_t = (1 - t) noise + t x, y = x_t / t is
  the clean image plus Gaussian noise of standard deviation sigma = (1 - t) / t, and the model's
  denoiser is xhat = x_t + (1 - t) v. The denoising error per image, mmse(sigma) = E||x - xhat||^2
  summed over pixels, gives d(sigma) = mmse / sigma^2: the number of directions in which the images
  still vary at scale sigma (the denoising-loss estimator of local intrinsic dimension, Yeats et al.
  2025; the slope of a rate-distortion curve). A trained denoiser can only raise mmse, so d is an
  upper bound. With the condition given (flag 1) it is the dimension of the fiber; without it
  (flag 0), of the whole range. For Gaussian random fields the conditional value is exact,
  d(sigma) = sum_k lam_k / (lam_k + sigma^2).
* Bias-variance split of the trend error across independent training sets: the trend deviations
  p_s = d_s - mean_grid(d_s) of set s; V = sum_s ||p_s - pbar||^2 / (k - 1) (minus the sampling
  noise of 128 generated images) is the variance of one model, B = ||pbar||^2 - V / k its bias.

Usage: python measure_v4.py RANGE [RANGE ...]   -> results/measure_v4_{RANGE}.json
"""
import json
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
from genmicro.net import FlowUNet  # noqa: E402

OUT = HERE.parent / "results"
SIGMAS = np.geomspace(0.05, 5.0, 15)


def load_model(name, split):
    ck = torch.load(OUT / f"model_{name}_{split}.pt", weights_only=False)
    m = FlowUNet(cond_dim=ck["cond_dim"], **ck["model_config"])
    if ck.get("embed"):
        m.cond = St.cond_embedding(ck["cond_dim"], m.cond.mlp[-1].out_features, ck["embed"])
    m = m.cuda()
    m.load_state_dict(ck["ema"])
    return m.eval()


@torch.no_grad()
def dsm_spectrum(model, x, c, seed=21, batch=128):
    """Conditional and unconditional d(sigma) on images x (n, n_px, n_px) with conditions c."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    out_c, out_u = [], []
    for sig in SIGMAS:
        t = 1.0 / (1.0 + sig)
        ec = eu = 0.0
        for s in range(0, len(x), batch):
            xb, cb = x[s:s + batch, None], c[s:s + batch]
            b = len(xb)
            eps = torch.randn(xb.shape, device="cuda", generator=g)
            xt = (1 - t) * eps + t * xb
            tt = torch.full((b,), t, device="cuda")
            with torch.autocast("cuda", torch.bfloat16):
                vc = model(xt, tt, St.flagged(cb, torch.ones(b, device="cuda"))).float()
                vu = model(xt, tt, St.flagged(cb, torch.zeros(b, device="cuda"))).float()
            ec += (xb - (xt + (1 - t) * vc)).square().sum().item()
            eu += (xb - (xt + (1 - t) * vu)).square().sum().item()
        out_c.append(ec / len(x) / sig ** 2)
        out_u.append(eu / len(x) / sig ** 2)
    return out_c, out_u


def exact_spectrum(sys_, theta):
    """Exact conditional d(sigma) for Gaussian random fields, averaged over the settings."""
    if sys_["kind"] == "grf":
        lam = grf.spectrum(theta.cuda().double(), S.N_PX)
    elif sys_["kind"] == "mix":
        lam = grf.mix_spectrum(theta.cuda().double(), sys_["s"], sys_["a"], sys_["m"], S.N_PX)
    else:
        return None
    return [(lam / (lam + sig ** 2)).sum((1, 2)).mean().item() for sig in SIGMAS]


def processing_parts(mus, ref, scale, n_gen):
    """Bias and variance (squared distances, per grid setting) of the trend deviations of k sets."""
    c = St.metric(ref, scale)
    s2 = scale[:, None] * scale[None, :]
    p = [(mu - ref["mu"]) / scale for mu in mus]
    p = torch.stack([q - q.mean(0) for q in p])  # (k, G, D)
    k = len(p)
    pbar = p.mean(0)
    q = lambda v: (v[..., None, :] @ torch.linalg.solve(c, v[..., None]))[..., 0, 0]
    noise = torch.diagonal(torch.linalg.solve(c, ref["cov"] / s2 / n_gen), dim1=-2, dim2=-1).sum(-1)
    var = (q(p - pbar).sum(0) / (k - 1) - noise).clamp_min(0)
    ref_noise = torch.diagonal(torch.linalg.solve(c, ref["cov"] / s2 / ref["n"]), dim1=-2, dim2=-1).sum(-1)
    bias = (q(pbar) - var / k - noise / k - ref_noise).clamp_min(0)
    return bias, var


def bias_variance_saved(name, sys_, ref, scale, sets):
    th = St.grid_theta(sys_)
    mus = []
    for split in sets:
        model = load_model(name, split)
        mus.append(St.gen_stats(sys_, St.generate(model, sys_, th, St.N_GEN, seed=177 + split))["mu"])
        del model
    b, v = processing_parts(mus, ref, scale, St.N_GEN)
    return dict(bias=b.tolist(), var=v.tolist(), bias_share=float((b / (b + v).clamp_min(1e-12)).median()))


def bias_variance_ladder(name, ref, scale, sets):
    """For v4 runs, which saved the generated means at every N."""
    rows = [json.loads((OUT / f"study_{name}_{s}.json").read_text())["rows"] for s in sets]
    out = []
    for j in range(len(rows[0])):
        mus = [torch.tensor(r[j]["gen_mu"], dtype=torch.float64, device=ref["mu"].device) for r in rows]
        b, v = processing_parts(mus, ref, scale, St.N_GEN)
        out.append(dict(n=rows[0][j]["n"], bias_median=float(b.median()), var_median=float(v.median()),
                        bias_share=float((b / (b + v).clamp_min(1e-12)).median()), bias=b.tolist(), var=v.tolist()))
    return out


def run(name):
    sys_ = S.SYSTEMS[name]
    sets = sorted(int(p.stem.rsplit("_", 1)[1]) for p in OUT.glob(f"model_{name}_*.pt") if p.stem.rsplit("_", 1)[0] == f"model_{name}")
    ref = St.reference(name, sys_, 100)
    scale = St.scale_of(ref)
    theta, cond, fields = St.held_out(name, sys_)
    res = dict(range=name, sets=sets, sigmas=SIGMAS.tolist(), exact=exact_spectrum(sys_, theta), d_cond=[], d_uncond=[])
    for split in sets:
        dc, du = dsm_spectrum(load_model(name, split), fields, cond)
        res["d_cond"].append(dc)
        res["d_uncond"].append(du)
    res["bias_variance_512"] = bias_variance_saved(name, sys_, ref, scale, sets) if len(sets) > 1 else None
    if sys_["kind"] == "mix" and len(sets) > 1:
        res["bias_variance_ladder"] = bias_variance_ladder(name, ref, scale, sets)
    (OUT / f"measure_v4_{name}.json").write_text(json.dumps(res, indent=1))
    print(name, "d_cond(0.5) =", [round(np.interp(math.log(0.5), np.log(SIGMAS), d), 1) for d in res["d_cond"]],
          "exact", None if res["exact"] is None else round(float(np.interp(math.log(0.5), np.log(SIGMAS), res["exact"])), 1),
          "bias share", res["bias_variance_512"] and round(res["bias_variance_512"]["bias_share"], 2), flush=True)


if __name__ == "__main__":
    torch.cuda.set_per_process_memory_fraction(0.6)
    for nm in sys.argv[1:]:
        run(nm)

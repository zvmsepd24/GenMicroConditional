"""Training data needed by a conditional flow model, and the processing information it absorbs.

For one range and one training pool (set), 512 parameter values are drawn uniformly and one field
is rendered per value. A conditional flow model is trained on nested subsets N = 16 ... 512, each
size continuing from the model of the next smaller size. After each size:

* error: at a 3 x 3 grid of interior conditions (normalized coordinates -2/3, 0, 2/3), the
  distance between the mean descriptors of 128 generated fields and of reference fields, in units
  of the reference scatter between fields (Mahalanobis, covariance shrunk 10% to its diagonal
  plus a ridge of RIDGE in standardized units), with the expected contribution of sampling noise
  in both means subtracted. 0 is a perfect model; 1 means the average generated field at a
  condition is one single-field scatter away from the true average.
  The difference of means at condition i, d_i, is split into its average over the grid, b (the
  texture error, common to the whole range), and d_i - b (the processing error, which measures
  how well the model follows the change of the microstructure across the range). The effect size
  is the processing error of a model that ignores the condition: the distance of each reference
  mean from the grid average of reference means.
* at the largest N only, the I-MMSE estimate of I(theta; X) from the model's own conditional and
  unconditional denoisers on 256 held-out real fields,
      I = 1/2 int gamma E|| xhat(y, theta) - xhat(y) ||^2 dlog(gamma),
  where y = sqrt(gamma) x + noise, and the same with the difference of the two denoising losses.
  For exact denoisers both equal I(theta; X). The model is trained with the condition dropped
  for 10% of samples (flag channel 0), so one network gives both denoisers.

Usage: python study.py RANGE:SET [RANGE:SET ...]
"""
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import systems as S  # noqa: E402
from genmicro.net import ConditionEmbedding, FlowUNet  # noqa: E402
from genmicro.train import sample, train  # noqa: E402

OUT = HERE.parent / "results"
LADDER = [16, 32, 64, 128, 256, 512]
FIRST_STEPS, STAGE_STEPS, BATCH = 1200, 600, 64
MODEL = dict(channels=1, patch=1, widths=(32, 64, 128, 128), depths=(1, 2, 2, 4), periodic=True)
DROP = 0.1
GRID = (-2 / 3, 0.0, 2 / 3)
N_GEN = 128
N_REF = {"grf": 512, "mix": 512, "ch": 256, "chred": 256, "potts": 256}
GRID_U1_MIX = tuple((2 * j - 8) / 9 for j in range(9))  # v4 accordion: 9 settings along the fold axis
RIDGE = 1e-2
LOG_SNR = torch.linspace(-6.0, 10.0, 33)
N_HELD = 256
LADDER_DIR = Path(os.environ["LADDER_DIR"]) if os.environ.get("LADDER_DIR") else None  # v5: save every rung there
SAMPLER = dict(steps=32, rho=1.0)  # pilot: 16 steps at rho 0.7 smoothed GRF fields at pixel scale


def base_of(name):
    """Range whose images, references and pools a range shares (late_red: late; *_raw, *_f0 ...: same
    images, other condition encoding)."""
    name = re.sub(r"_(raw|f\d)$", "", name)
    return "late" if name == "late_red" else name


def cond_embedding(cond_dim, width, embed):
    """Condition encoding: None, the default sinusoidal features at pi 2^k, k = -1 ... 4 in steps of 1/3;
    "raw", the scaled knobs only; "f<K>", the same features cut at the highest frequency pi 2^K."""
    if embed == "raw":
        return ConditionEmbedding(cond_dim, width, n_freq=0)
    kmax = int(embed[1:])
    n = 3 * (kmax + 1) + 1
    emb = ConditionEmbedding(cond_dim, width, n_freq=n)
    emb.freq = math.pi * 2.0 ** torch.linspace(-1, kmax, n)
    return emb


def make_model(sys_):
    model = FlowUNet(cond_dim=S.n_cond(sys_) + 1, **MODEL)
    if sys_.get("embed"):
        model.cond = cond_embedding(S.n_cond(sys_) + 1, model.cond.mlp[-1].out_features, sys_["embed"])
    return model


def flagged(c, keep):
    return torch.cat([c * keep[:, None], keep[:, None]], 1)


def grid_theta(sys_):
    g = torch.tensor(GRID, dtype=torch.float64)
    g1 = torch.tensor(GRID_U1_MIX, dtype=torch.float64) if sys_["kind"] == "mix" else g
    c = torch.stack(torch.meshgrid(g1, g, indexing="ij"), -1).reshape(-1, 2)
    return S.snap(sys_, S.denormalize(sys_, c))


def ref_kind(sys_):
    return "ch" if sys_["kind"] == "chred" else sys_["kind"]


def reference(name, sys_, seed):
    """Descriptor mean, covariance and count at the grid conditions, cached on disk."""
    base = base_of(name)
    path = OUT / f"ref_{base}_{seed}.pt"
    if path.exists():
        return torch.load(path)
    th = grid_theta(sys_)
    r = N_REF[sys_["kind"]]
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = S.render(sys_, th.repeat_interleave(r, 0), g)  # all grid settings in large batches
    mus, covs = [], []
    for i in range(len(th)):
        d = S.descriptors(sys_, x[i * r:(i + 1) * r])
        mus.append(d.mean(0))
        covs.append(torch.cov(d.T))
    out = dict(mu=torch.stack(mus), cov=torch.stack(covs), n=r)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(out, path)
    return out


def scale_of(ref):
    """Pooled within-condition standard deviation of each descriptor."""
    return torch.diagonal(ref["cov"], dim1=-2, dim2=-1).mean(0).sqrt()


def metric(ref, scale):
    s = scale[:, None] * scale[None, :]
    c = ref["cov"] / s
    return 0.9 * c + 0.1 * torch.diag_embed(torch.diagonal(c, dim1=-2, dim2=-1)) + RIDGE * torch.eye(len(scale), dtype=c.dtype, device=c.device)


def mahalanobis(d, c, noise):
    """Debiased distance sqrt(max(0, d^T C^-1 d - tr(C^-1 noise))) per row, plus its two parts."""
    d2 = (d[..., None, :] @ torch.linalg.solve(c, d[..., None]))[..., 0, 0]
    bias = torch.diagonal(torch.linalg.solve(c, noise), dim1=-2, dim2=-1).sum(-1)
    return torch.sqrt(torch.clamp(d2 - bias, min=0)), d2, bias


def distance(gen, ref, scale):
    """Debiased Mahalanobis distance per grid condition, in units of the reference scatter."""
    s = scale[:, None] * scale[None, :]
    noise = gen["cov"] / s / gen["n"] + ref["cov"] / s / ref["n"]
    return mahalanobis((gen["mu"] - ref["mu"]) / scale, metric(ref, scale), noise)


def split_distance(gen, ref, scale):
    """Texture error (grid-average difference, scalar) and processing error (per condition)."""
    s = scale[:, None] * scale[None, :]
    c = metric(ref, scale)
    m = len(ref["mu"])
    d = (gen["mu"] - ref["mu"]) / scale
    noise = gen["cov"] / s / gen["n"] + ref["cov"] / s / ref["n"]  # covariance of each d_i
    b = d.mean(0)
    noise_b = noise.mean(0) / m
    texture = mahalanobis(b, c.mean(0), noise_b)[0]
    processing = mahalanobis(d - b, c, noise * (1 - 2 / m) + noise_b)[0]
    return texture, processing


def effect_size(ref, scale):
    """Processing error of a condition-blind model that returns the grid average of the reference."""
    s = scale[:, None] * scale[None, :]
    m = len(ref["mu"])
    mu = ref["mu"] / scale
    noise = ref["cov"] / s / ref["n"]
    return mahalanobis(mu - mu.mean(0), metric(ref, scale), noise * (1 - 2 / m) + noise.mean(0) / m)[0]


@torch.no_grad()
def generate(model, sys_, theta, n, seed):
    """n generated fields at each parameter row of theta -> (len(theta), n, N_PX, N_PX)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    gc = torch.Generator().manual_seed(seed)
    out = []
    for row in theta:
        c = S.condition(sys_, row[None].repeat(n, 1), gc).cuda()
        x0 = torch.randn(n, 1, S.N_PX, S.N_PX, device="cuda", generator=g)
        with torch.autocast("cuda", torch.bfloat16):
            x = sample(model, x0, SAMPLER["steps"], cond=flagged(c, torch.ones(n, device="cuda")), rho=SAMPLER["rho"])
        out.append(x[:, 0].float())
    return torch.stack(out)


def gen_stats(sys_, fields):
    mus, covs = [], []
    for x in fields:
        d = S.descriptors(sys_, x)
        mus.append(d.mean(0))
        covs.append(torch.cov(d.T))
    return dict(mu=torch.stack(mus), cov=torch.stack(covs), n=fields.shape[1])


@torch.no_grad()
def absorbed_information(model, x1, c, seed=11, batch=256):
    """I-MMSE estimate of I(theta; X) from the model's conditional and unconditional denoisers.
    On the straight path x_t = (1-t) x0 + t x1, y = x_t / (1-t) has snr gamma = (t / (1-t))^2 and
    the denoiser is xhat = x_t + (1-t) v. Returns the difference-of-denoisers and the
    difference-of-losses estimates and their integrands per log-snr point."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    n = len(x1)
    diff, loss = [], []
    for lam in LOG_SNR.tolist():
        r = math.exp(lam / 2)  # sqrt(gamma) = t / (1 - t)
        t = r / (1 + r)
        acc_d = acc_l = 0.0
        for s in range(0, n, batch):
            xb, cb = x1[s:s + batch], c[s:s + batch]
            b = len(xb)
            x0 = torch.randn(xb.shape, device="cuda", generator=g)
            xt = (1 - t) * x0 + t * xb
            tt = torch.full((b,), t, device="cuda")
            with torch.autocast("cuda", torch.bfloat16):
                vc = model(xt, tt, flagged(cb, torch.ones(b, device="cuda"))).float()
                vu = model(xt, tt, flagged(cb, torch.zeros(b, device="cuda"))).float()
            xc, xu = xt + (1 - t) * vc, xt + (1 - t) * vu
            acc_d += (xc - xu).square().sum().item()
            acc_l += ((xb - xu).square().sum() - (xb - xc).square().sum()).item()
        diff.append(0.5 * math.exp(lam) * acc_d / n)
        loss.append(0.5 * math.exp(lam) * acc_l / n)
    ls = LOG_SNR.numpy()
    return dict(i_diff=float(np.trapezoid(diff, ls)), i_loss=float(np.trapezoid(loss, ls)), integrand_diff=diff, integrand_loss=loss)


def pool(name, sys_, split, n):
    """Training pool: parameters, conditions and fields. late_red reuses the late pool exactly."""
    base = base_of(name)
    gen = torch.Generator().manual_seed(1000 + 17 * split + sum(map(ord, base)))
    theta = S.draw(sys_, n, gen)
    fields = S.render(sys_, theta, torch.Generator(device="cuda").manual_seed(2000 + 17 * split + sum(map(ord, base))))
    cond = S.condition(sys_, theta, torch.Generator().manual_seed(5 + split))
    return theta, cond.cuda(), fields


def held_out(name, sys_):
    base = base_of(name)
    gen = torch.Generator().manual_seed(9000 + sum(map(ord, base)))
    theta = S.draw(sys_, N_HELD, gen)
    fields = S.render(sys_, theta, torch.Generator(device="cuda").manual_seed(9100 + sum(map(ord, base))))
    return theta, S.condition(sys_, theta, torch.Generator().manual_seed(9200)).cuda(), fields


def run(name, split):
    sys_ = S.SYSTEMS[name]
    t_all = time.perf_counter()
    ref, ref2 = reference(name, sys_, 100), reference(name, sys_, 200)
    scale = scale_of(ref)
    floor = distance(ref2, ref, scale)[0]
    floor_tex, floor_proc = split_distance(ref2, ref, scale)
    effect = effect_size(ref, scale)
    theta, cond, fields = pool(name, sys_, split, max(LADDER))
    _, cond_h, fields_h = held_out(name, sys_)
    th_grid = grid_theta(sys_)
    model = make_model(sys_).cuda().to(memory_format=torch.channels_last)
    g = torch.Generator(device="cuda").manual_seed(3000 + split)
    rows = []
    for k, n in enumerate(LADDER):
        def batch_fn(b, step, n=n):
            idx = torch.randint(0, n, (b,), device="cuda", generator=g)
            x, c = S.augment(sys_, fields[idx, None], cond[idx], g)
            keep = (torch.rand(b, device="cuda", generator=g) > DROP).float()
            return x.contiguous(memory_format=torch.channels_last), flagged(c, keep)

        steps = FIRST_STEPS if k == 0 else STAGE_STEPS
        t1 = time.perf_counter()
        ema, info = train(model, batch_fn, steps, batch=BATCH, seed=split * 100 + k, log_every=10 ** 9)
        ema.eval()
        t2 = time.perf_counter()
        gen_fields = generate(ema, sys_, th_grid, N_GEN, seed=77)
        gs = gen_stats(sys_, gen_fields)
        err, d2, bias = distance(gs, ref, scale)
        tex, proc = split_distance(gs, ref, scale)
        tr = lambda cv: torch.diagonal(cv, dim1=-2, dim2=-1).div(scale ** 2).sum(-1)
        spread = torch.sqrt(tr(gs["cov"]) / tr(ref["cov"]))
        row = dict(range=name, split=split, n=n, steps=steps, train_s=t2 - t1,
                   error_median=err.median().item(), error_max=err.max().item(), error_mean=err.mean().item(),
                   texture=tex.item(), processing_median=proc.median().item(), processing_max=proc.max().item(),
                   processing_mean=proc.mean().item(), effect_median=effect.median().item(),
                   floor_median=floor.median().item(), floor_texture=floor_tex.item(), floor_processing=floor_proc.median().item(),
                   spread_median=spread.median().item(), error_grid=err.tolist(), processing_grid=proc.tolist(),
                   effect_grid=effect.tolist(), d2_grid=d2.tolist(), bias_grid=bias.tolist())
        if sys_["kind"] == "mix":  # v4: keep generated means for the bias-variance split across sets
            row.update(gen_mu=gs["mu"].tolist(), gen_cov=gs["cov"].tolist())
        if n == max(LADDER) and sys_["kind"] != "mix":
            row.update(absorbed_information(ema, fields_h[:, None], cond_h))
        row["eval_s"] = time.perf_counter() - t2
        rows.append(row)
        if LADDER_DIR is not None:  # v5 ladder runs: keep the model of every training-set size
            torch.save(dict(model_config=MODEL, cond_dim=S.n_cond(sys_) + 1, embed=sys_.get("embed"), ema=ema.state_dict()),
                       LADDER_DIR / f"model_{name}_{split}_n{n}.pt")
        print(json.dumps({a: (round(v, 3) if isinstance(v, float) else v) for a, v in row.items()
                          if not a.endswith("grid") and not a.startswith(("integrand", "gen_"))}), flush=True)
    keep_idx = [0, 13, 26] if sys_["kind"] == "mix" else [0, 4, 8]
    np.savez_compressed((LADDER_DIR or OUT) / f"samples_{name}_{split}.npz", generated=gen_fields[keep_idx, :8].cpu().numpy(),
                        real=fields[:16].cpu().numpy(), real_theta=theta[:16].numpy(), grid_theta=th_grid[keep_idx].numpy())
    ((LADDER_DIR or OUT) / f"study_{name}_{split}.json").write_text(json.dumps(dict(
        range=name, system={k: v for k, v in sys_.items()}, model=MODEL, ladder=LADDER, first_steps=FIRST_STEPS,
        stage_steps=STAGE_STEPS, batch=BATCH, drop=DROP, n_gen=N_GEN, n_ref=N_REF[sys_["kind"]], ridge=RIDGE,
        log_snr=LOG_SNR.tolist(), total_s=time.perf_counter() - t_all, rows=rows), indent=1))
    if LADDER_DIR is None:
        torch.save(dict(model_config=MODEL, cond_dim=S.n_cond(sys_) + 1, embed=sys_.get("embed"), ema=ema.state_dict()), OUT / f"model_{name}_{split}.pt")


if __name__ == "__main__":
    torch.cuda.set_per_process_memory_fraction(0.6)
    torch.backends.cudnn.benchmark = True
    OUT.mkdir(parents=True, exist_ok=True)
    for job in sys.argv[1:]:
        name, split = job.split(":")
        run(name, int(split))

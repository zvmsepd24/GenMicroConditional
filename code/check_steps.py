"""Is the error at N = 512 limited by training steps? Train from scratch on the 512-field gA pool for
longer and compare; also report the error of a model that ignores the condition (grid mean)."""
import json, sys, time, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import study as T, systems as S
from genmicro.net import FlowUNet
from genmicro.train import train
torch.backends.cudnn.benchmark = True
name = sys.argv[1]; steps_list = [int(s) for s in sys.argv[2:]]
sys_ = S.SYSTEMS[name]
ref, ref2 = T.reference(name, sys_, 100), T.reference(name, sys_, 200)
scale = T.scale_of(ref)
uncond = dict(mu=ref2["mu"].mean(0, keepdim=True).expand_as(ref2["mu"]), cov=ref2["cov"], n=10**9)
print("uncond error per grid", [round(v, 2) for v in T.distance(uncond, ref, scale)[0].tolist()])
theta, cond, fields = T.pool(name, sys_, 0, 512)
th_grid = T.grid_theta(sys_)
for steps in steps_list:
    model = FlowUNet(cond_dim=S.n_cond(sys_) + 1, **T.MODEL).cuda().to(memory_format=torch.channels_last)
    g = torch.Generator(device="cuda").manual_seed(4000)
    def bf(b, step):
        idx = torch.randint(0, 512, (b,), device="cuda", generator=g)
        x, c = S.augment(sys_, fields[idx, None], cond[idx], g)
        keep = (torch.rand(b, device="cuda", generator=g) > T.DROP).float()
        return x.contiguous(memory_format=torch.channels_last), T.flagged(c, keep)
    t0 = time.perf_counter()
    ema, info = train(model, bf, steps, batch=T.BATCH, seed=9, log_every=10**9)
    gs = T.gen_stats(sys_, T.generate(ema.eval(), sys_, th_grid, T.N_GEN, 77))
    err = T.distance(gs, ref, scale)[0]
    tex, trend = T.split_distance(gs, ref, scale)
    row = dict(range=name, steps=steps, seconds=round(time.perf_counter() - t0, 1), error_median=round(err.median().item(), 3),
               texture=round(tex.item(), 3), trend_median=round(trend.median().item(), 3), error_grid=[round(v, 2) for v in err.tolist()])
    print(json.dumps(row), flush=True)
    out = T.OUT / "steps_check.json"
    rows = json.loads(out.read_text()) if out.exists() else []
    out.write_text(json.dumps(rows + [row], indent=1))

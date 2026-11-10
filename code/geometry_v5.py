"""v5 measurements of manifold geometry on the evaluation grid of each range (fisher_v5.py).

  python geometry_v5.py real  NAME ...   real descriptor metric (and exact metric for GRF ranges)
                                          -> results/geom_v5_real_{NAME}.json
  python geometry_v5.py model NAME ...   denoiser-read metric of every saved model of the range, and the
                                          generated descriptor metric of training set 0
                                          -> results/geom_v5_model_{NAME}.json, geom_v5_vis_{NAME}.pt
  python geometry_v5.py ladder NAME ...  as model, for the checkpoints in results/ladder (E3), plus the
                                          local fiber dimension at every N -> results/geom_v5_ladder_{NAME}.json

The redundant range late_red has no real metric of its own: its images are those of late, and its
three network inputs (u1, a, b) map to late's (u1, u2) by u2 = (a + b) / 2.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import fisher_v5 as F  # noqa: E402
import study as St  # noqa: E402
import systems as S  # noqa: E402
from measure_v4 import SIGMAS as DSIG, dsm_spectrum, load_model  # noqa: E402

OUT = HERE.parent / "results"
LAD = OUT / "ladder"
R_REAL, R_MODEL, R_GEN = 256, 64, 128


def step(sys_):
    return 0.01 if sys_["kind"] == "mix" else 0.1


def sets_of(name, folder=OUT):
    return sorted(int(p.stem.rsplit("_", 1)[1]) for p in folder.glob(f"model_{name}_*.pt") if p.stem.rsplit("_", 1)[0] == f"model_{name}")


def real(name):
    sys_ = S.SYSTEMS[name]
    th = St.grid_theta(sys_)
    res = dict(range=name, theta=th.tolist(), h=step(sys_), r=R_REAL, G_desc=[])
    for i, row in enumerate(th):
        res["G_desc"].append(F.real_desc_fisher(sys_, row, R_REAL, step(sys_), seed=8500 + i).tolist())
    if sys_["kind"] in ("grf", "mix"):
        G, integ = F.exact_fisher(sys_, S.normalize(sys_, th).double())
        res.update(G_exact=G.tolist(), integrand_exact=integ.tolist(), sigmas=F.SIGMAS.tolist())
    (OUT / f"geom_v5_real_{name}.json").write_text(json.dumps(res))
    print(name, "real done", flush=True)


def model_geometry(name, model, sys_, th, vis=False, gen=True, seed=8300, R=R_MODEL, hs=(2e-3,)):
    out = {}
    for h in hs:
        G, integ, v = F.grid_model_fisher(model, sys_, th, R=R, seed=seed, keep_vis=vis and h == hs[0], h=h)
        if h == hs[0]:
            out.update(G=G.tolist(), integrand=integ.tolist())
            vv = v
        else:
            out.setdefault("G_by_h", {})[str(h)] = G.tolist()
    if gen:
        out["G_gen_desc"] = [F.gen_desc_fisher(model, sys_, row, R_GEN, step(sys_), seed=8700 + i).tolist() for i, row in enumerate(th)]
    return out, vv


def model(spec):
    """spec = NAME@opts with opts from: all (every saved set; default set 0 only), gen (generated
    descriptor metric of the first set), hscan (also steps 0.01, 0.05, 0.2 on the first set),
    R<number> (real images per setting)."""
    name, _, opts = spec.partition("@")
    opts = opts.split(",") if opts else []
    R = next((int(o[1:]) for o in opts if o.startswith("R")), R_MODEL)
    sys_ = S.SYSTEMS[name]
    th = St.grid_theta(sys_)
    sets = sets_of(name)
    sets = sets if "all" in opts else sets[:1]
    res = dict(range=name, theta=th.tolist(), sigmas=F.SIGMAS.tolist(), sets=sets, r=R, per_set=[])
    for split in sets:
        m = load_model(name, split)
        first = split == sets[0]
        hs = (2e-3, 1e-2, 5e-2, 0.2) if first and "hscan" in opts else (2e-3,)
        g, v = model_geometry(name, m, sys_, th, vis=first, gen=first and "gen" in opts, R=R, hs=hs)
        res["per_set"].append(dict(split=split, **g))
        if first:
            torch.save(v, OUT / f"geom_v5_vis_{name}.pt")
        del m
        print(name, split, "model done", flush=True)
    (OUT / f"geom_v5_model_{name}.json").write_text(json.dumps(res))


def ladder(name):
    """E3: geometry and held-out fiber dimension of the checkpoint saved at every training-set size."""
    sys_ = S.SYSTEMS[name]
    th = St.grid_theta(sys_)
    res = dict(range=name, theta=th.tolist(), sigmas=F.SIGMAS.tolist(), dsig=DSIG.tolist(), rows=[])
    _, cond_h, fields_h = St.held_out(name, sys_)
    for n in St.LADDER:
        m = load_model_path(LAD / f"model_{name}_0_n{n}.pt")
        g, _ = model_geometry(name, m, sys_, th, gen=n == max(St.LADDER))
        dc, _ = dsm_spectrum(m, fields_h, cond_h)
        res["rows"].append(dict(n=n, d_cond=dc, **g))
        del m
        print(name, n, "ladder done", flush=True)
    (OUT / f"geom_v5_ladder_{name}.json").write_text(json.dumps(res))


def ladder_small(name, R=32, idx=4):
    """E3 at small scale: the denoiser metric at one setting (the centre) of the checkpoint saved at
    every training-set size, and for GRF ranges the held-out fiber dimension."""
    sys_ = S.SYSTEMS[name]
    th = St.grid_theta(sys_)[idx:idx + 1]
    res = dict(range=name, theta=th.tolist(), setting_index=idx, r=R, sigmas=F.SIGMAS.tolist(), dsig=DSIG.tolist(), rows=[])
    held = St.held_out(name, sys_) if sys_["kind"] == "grf" else None
    for n in St.LADDER:
        m = load_model_path(LAD / f"model_{name}_0_n{n}.pt")
        G, integ, _ = F.grid_model_fisher(m, sys_, th, R=R, seed=8300)
        row = dict(n=n, G=G.tolist())
        if held is not None:
            row["d_cond"] = dsm_spectrum(m, held[2], held[1])[0]
        res["rows"].append(row)
        del m
    (OUT / f"geom_v5_ladder_{name}.json").write_text(json.dumps(res))
    print(name, "ladder done", flush=True)


def embed(spec):
    """E4: metric on a fine 9 x 9 grid of u in [-1, 1]^2 for an embedding of the manifold.
    spec = NAME (exact for GRF ranges, else the first saved model) or NAME@model (model even for GRF)."""
    name, _, opt = spec.partition("@")
    opts = opt.split(",") if opt else []
    n = next((int(o[1:]) for o in opts if o.startswith("n")), 9)
    R = next((int(o[1:]) for o in opts if o.startswith("R")), 32)
    sys_ = S.SYSTEMS[name]
    g = torch.linspace(-1, 1, n, dtype=torch.float64)
    u = torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    th = S.denormalize(sys_, u)
    if sys_["kind"] in ("grf", "mix") and "model" not in opts:
        G, _ = F.exact_fisher(sys_, u)
        tag = name + "_exact"
    else:
        m = load_model(name, sets_of(name)[0])
        G, _, _ = F.grid_model_fisher(m, sys_, th, R=R, seed=9300)
        tag = name + "_model"
    (OUT / f"geom_v5_embed_{tag}.json").write_text(json.dumps(dict(range=name, u=u.tolist(), theta=th.tolist(), G=G.tolist())))
    print(tag, "embed done", flush=True)


def load_model_path(path):
    from genmicro.net import FlowUNet
    ck = torch.load(path, weights_only=False)
    m = FlowUNet(cond_dim=ck["cond_dim"], **ck["model_config"])
    if ck.get("embed"):
        m.cond = St.cond_embedding(ck["cond_dim"], m.cond.mlp[-1].out_features, ck["embed"])
    m = m.cuda()
    m.load_state_dict(ck["ema"])
    return m.eval()


if __name__ == "__main__":
    torch.cuda.set_per_process_memory_fraction(0.6)
    what, names = sys.argv[1], sys.argv[2:]
    for nm in names:
        globals()[what](nm)

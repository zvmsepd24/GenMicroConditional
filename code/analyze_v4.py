"""v4 analysis (base and fiber): the accordion ranges, fiber dimension spectra, bias-variance splits,
the Gaussian oracle, and the predictions P1-P8 of the historical v4 tests -> results/summary_v4.json, tests_v4.json.

Deviation from the plan, recorded here and in the manuscript: v3 kept models only at N = 512, so the
"variance at N = 16" half of P7 is evaluated on the low-trend accordion range acc_m0 (trend size 1.1,
like gC and gD) instead of gC and gD.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from analyze import curvature, curve, fit_n0, load_studies, n_needed, noise_dimension  # noqa: E402

RES = HERE.parent / "results"
ACC = ["acc_m0", "acc_m1", "acc_m2", "acc_m4", "acc_m0_x4", "acc_m2_x4", "acc_m8", "acc_m16", "acc_m0_raw", "acc_m8_raw", "acc_m16_raw"]
V3 = ["gA", "gB", "gC", "gD", "late", "late_narrow", "onset", "transition", "late_red", "potts_wide", "potts_narrow", "potts_early"]
BW = ["acc_m0_f0", "acc_m0_f2", "acc_m4_f2", "acc_m8_f0", "acc_m8_f2"]
RAW3 = ["gA_raw", "gD_raw", "late_raw", "onset_raw", "potts_wide_raw", "potts_narrow_raw"]
SIGMA_STAR = 0.5


def geo(x):
    return float(np.exp(np.mean(np.log(x))))


def at_sigma(m, d, sig=SIGMA_STAR):
    return float(np.interp(math.log(sig), np.log(m["sigmas"]), d))


@torch.no_grad()
def fold_geometry(name, n_line=97, n_ref=256):
    """Along the fold axis u1 (u2 = 0): reference descriptor means on a fine line, their departure
    from a straight line (fold amplitude, scatter units), the RMS second derivative d2mu/du1^2
    (scatter units), the fold direction (top principal direction of the departures) and the variance
    of the reference along it. Returns also the departures, for window averages."""
    import study as St
    import systems as S
    sys_ = S.SYSTEMS[name]
    ref = St.reference(name, sys_, 100)
    sc = St.scale_of(ref)
    C = St.metric(ref, sc)[13]  # centre of the 9 x 3 grid
    u1 = torch.linspace(-1, 1, n_line, dtype=torch.float64)
    th = torch.stack([u1, torch.zeros_like(u1)], 1)
    g = torch.Generator(device="cuda").manual_seed(4)
    x = S.render(sys_, th.repeat_interleave(n_ref, 0), g)
    mu = S.descriptors(sys_, x).reshape(n_line, n_ref, -1).mean(1) / sc
    L = torch.linalg.cholesky(torch.linalg.inv(C))  # whitening: |v|_C = |L^T v|
    w = mu @ L
    A = torch.stack([torch.ones_like(u1), u1], 1).to(w.device)
    res = w - A @ torch.linalg.lstsq(A, w).solution
    h = float(u1[1] - u1[0])
    d2 = (w[2:] - 2 * w[1:-1] + w[:-2]) / h ** 2
    if float(res.norm(dim=1).max()) > 1e-9:
        f = torch.linalg.svd(res - res.mean(0), full_matrices=False).Vh[0]
    else:
        f = torch.zeros(w.shape[1], dtype=w.dtype, device=w.device)
    return dict(u1=u1.numpy(), proj=(res @ f).cpu().numpy(), fold_amp=float(res.norm(dim=1).max()),
                fold_rms=float(res.norm(dim=1).pow(2).mean().sqrt()), d2_rms=float(d2.norm(dim=1).pow(2).mean().sqrt()),
                direction=(L @ f).cpu(), sc=sc.cpu(), C=C.cpu())


def spread_along(name, fg, rows):
    """Generated over reference variance along the fold direction at N = 512, median over the grid,
    with the generated covariance of each training set."""
    import study as St
    import systems as S
    ref = St.reference(name, S.SYSTEMS[name], 100)
    sc = fg["sc"]
    v = fg["direction"].double()
    s2 = sc[:, None] * sc[None, :]
    out = []
    for r in rows:
        G = torch.tensor(r["gen_cov"], dtype=torch.float64) / s2
        R = ref["cov"].cpu() / s2
        out.append(float(torch.median((v @ G @ v) / (v @ R @ v))))
    return out


def window_variance(fg, h):
    """Variance of the fold departure (scatter units, along the fold direction) under a Gaussian
    window of width h in u1, averaged over the evaluation settings |u1| <= 8/9."""
    u, p = fg["u1"], fg["proj"]
    vals = []
    for c in np.linspace(-8 / 9, 8 / 9, 9):
        w = np.exp(-0.5 * ((u - c) / h) ** 2)
        w /= w.sum()
        m = (w * p).sum()
        vals.append((w * (p - m) ** 2).sum())
    return float(np.mean(vals))


def oracle_n1(name):
    p = RES / f"oracle_v4_{name}.json"
    if not p.exists():
        return None
    o = json.loads(p.read_text())
    out = []
    for s in o["sets"]:
        rows = [r for r in o["rows"] if r["split"] == s]
        n = np.array([r["n"] for r in rows], float)
        e = np.array([r["processing_median"] for r in rows], float)
        out.append(n_needed(n, e, 1.0)[0])
    return out


def per_range(st):
    out = {}
    for name in ACC + V3 + RAW3 + BW:
        ds = sorted(st.get(name, []), key=lambda d: d["rows"][0]["split"])
        if not ds:
            continue
        n1 = [n_needed(*curve(d, "processing_median"), 1.0) for d in ds]
        fits = [fit_n0(d) for d in ds]
        o = dict(sets=len(ds), trend_size=ds[0]["rows"][0]["effect_median"], d_eff=noise_dimension(name),
                 n1=[v for v, _ in n1], n1_flag=[f for _, f in n1], n0=geo([a for a, _ in fits]), slope=[b for _, b in fits],
                 trend_512=[d["rows"][-1]["processing_median"] for d in ds],
                 spread_by_n=[[r["spread_median"] for r in d["rows"]] for d in ds], ladder=[r["n"] for r in ds[0]["rows"]])
        if name in V3:
            o["curvature"] = curvature(name)
        o["trend_low_n"] = [geo([r["processing_median"] for r in d["rows"] if r["n"] <= 128]) for d in ds]
        m = RES / f"measure_v4_{name}.json"
        if m.exists():
            mm = json.loads(m.read_text())
            o["d_star"] = geo([at_sigma(mm, d) for d in mm["d_cond"]])
            o["d_star_uncond"] = geo([at_sigma(mm, d) for d in mm["d_uncond"]])
            o["d_star_exact"] = at_sigma(mm, mm["exact"]) if mm["exact"] else None
            if mm["bias_variance_512"]:
                o["bias_share_512"] = mm["bias_variance_512"]["bias_share"]
            if "bias_variance_ladder" in mm:
                o["bias_share_by_n"] = [r["bias_share"] for r in mm["bias_variance_ladder"]]
        on = oracle_n1(name)
        if on:
            o["n1_oracle"] = on
        if name in ACC:
            fg = fold_geometry(name)
            o.update(fold_amp=fg["fold_amp"], fold_rms=fg["fold_rms"])
            if fg["fold_amp"] > 0.5:
                o["spread_fold_512"] = spread_along(name, fg, [d["rows"][-1] for d in ds])
                share = o["d_eff"] / o["n0"]                      # averaging share of the range (v3)
                h = math.sqrt(2 * share / math.pi)               # Gaussian window with that share of [-1, 1]^2
                o["window_h"] = h
                o["fold_window_var"] = window_variance(fg, h)
        out[name] = o
    return out


def tests(r):
    t = {}
    g = lambda nm, k="n1": geo(r[nm][k])
    # P1: N1 rises with fold count
    seq = [g(nm) for nm in ["acc_m0", "acc_m1", "acc_m2", "acc_m4"]]
    t["P1"] = dict(n1=seq, held=bool(all(b > a for a, b in zip(seq, seq[1:]))))
    # P2: unresolved folds merge fibers
    if "spread_fold_512" in r.get("acc_m4", {}):
        meas = geo(r["acc_m4"]["spread_fold_512"])
        base = r["acc_m0"].get("spread_fold_512")
        excess = meas - 1
        pred = r["acc_m4"]["fold_window_var"]
        t["P2"] = dict(spread_fold_m4=meas, excess_var=excess, predicted_var=pred,
                       ratio=excess / pred if pred > 0 else None,
                       held=bool(excess > 0 and pred > 0 and 0.5 <= excess / pred <= 2))
    # P3: extent does not cost data
    t["P3"] = dict(m0=g("acc_m0_x4") / g("acc_m0"), m2=g("acc_m2_x4") / g("acc_m2"),
                   trend_m0=r["acc_m0_x4"]["trend_size"] / r["acc_m0"]["trend_size"],
                   trend_m2=r["acc_m2_x4"]["trend_size"] / r["acc_m2"]["trend_size"])
    t["P3"]["held"] = bool(max(t["P3"]["m0"], t["P3"]["m2"]) < 1.5)
    # P4: model d(sigma) matches exact within 25% on sigma in [0.1, 3] (GRF ranges)
    worst = {}
    for nm in ["gA", "gB", "gC", "gD"]:
        mm = json.loads((RES / f"measure_v4_{nm}.json").read_text())
        s = np.array(mm["sigmas"])
        k = (s >= 0.1) & (s <= 3)
        ex = np.array(mm["exact"])[k]
        worst[nm] = float(max(np.max(np.abs(np.array(d)[k] / ex - 1)) for d in mm["d_cond"]))
    t["P4"] = dict(worst_rel_error=worst, held=bool(max(worst.values()) < 0.25))
    # P5: fiber dimension ratio predicts the N0 ratio in the fine/coarse random-field pairs
    p5 = {}
    for fine, coarse in [("gA", "gB"), ("gC", "gD")]:
        dr = r[fine]["d_star_exact"] / r[coarse]["d_star_exact"]
        nr = r[fine]["n0"] / r[coarse]["n0"]
        der = r[fine]["d_eff"] / r[coarse]["d_eff"]
        p5[f"{fine}/{coarse}"] = dict(d_ratio=dr, n0_ratio=nr, pred_over_obs=dr / nr, d_eff_ratio=der)
    t["P5"] = dict(pairs=p5, held=bool(all(0.5 <= v["pred_over_obs"] <= 2 for v in p5.values())))
    # P6: across the 12 v3 ranges, N0 / d(sigma*) vs trend size controlling for curvature
    names = [nm for nm in V3 if "d_star" in r[nm]]
    y = stats.rankdata([r[nm]["n0"] / r[nm]["d_star"] for nm in names])
    x = stats.rankdata([r[nm]["trend_size"] for nm in names])
    z = stats.rankdata([r[nm]["curvature"] for nm in names])
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    t["P6"] = dict(n=len(names), partial_rho=rho, held=bool(abs(rho) < 0.3))
    # P7: bias at N = 512 for m = 4, variance at N = 16 for the low-trend accordion range
    b4 = r["acc_m4"].get("bias_share_by_n", [None])[-1]
    v0 = r["acc_m0"].get("bias_share_by_n", [None])[0]
    t["P7"] = dict(bias_share_m4_512=b4, variance_share_m0_16=None if v0 is None else 1 - v0,
                   held=bool(b4 is not None and v0 is not None and b4 > 0.5 and (1 - v0) > 0.5))
    # P8: flow vs Gaussian oracle
    ratios = {nm: g(nm) / geo(r[nm]["n1_oracle"]) for nm in ["gA", "gB", "gC", "gD"] + ACC[:6] if "n1_oracle" in r.get(nm, {})}
    t["P8"] = dict(ratios=ratios, n_outside=int(sum(not (0.5 <= v <= 2) for v in ratios.values())),
                   held=bool(sum(not (0.5 <= v <= 2) for v in ratios.values()) < 2))
    # crossover test C1-C4 (written after P1-P8 were scored, before the m = 8, 16 flow runs)
    if "acc_m16" in r and "acc_m8" in r:
        floor = geo([geo(r[nm]["n1"]) for nm in ["acc_m0", "acc_m1", "acc_m2", "acc_m4"]])
        e16 = r["acc_m16"]["trend_512"]
        b16 = r["acc_m16"].get("bias_share_by_n", [None])[-1]
        s16 = r["acc_m16"].get("spread_fold_512")
        s4 = r["acc_m4"].get("spread_fold_512")
        n8 = geo(r["acc_m8"]["n1"])
        ratio8 = n8 / geo(r["acc_m8"]["n1_oracle"])
        t["C1"] = dict(trend_512=e16, n1_flags=r["acc_m16"]["n1_flag"], held=bool(min(e16) > 1))
        t["C2"] = dict(bias_share_512=b16, spread_fold_m16=None if s16 is None else geo(s16), spread_fold_m4=None if s4 is None else geo(s4),
                       held=bool(b16 is not None and s16 is not None and b16 > 0.5 and geo(s16) > geo(s4)))
        t["C3"] = dict(n1_m8=n8, n1_m8_sets=r["acc_m8"]["n1"], floor=floor, held=bool(n8 >= floor))
        t["C4"] = dict(flow_over_oracle_m8=ratio8, flow_over_oracle_m4=geo(r["acc_m4"]["n1"]) / geo(r["acc_m4"]["n1_oracle"]),
                       held=bool(ratio8 < 2.3))
    # embedding test E1-E4 (written after the crossover results, before the *_raw runs)
    if all(nm in r for nm in ["acc_m0_raw", "acc_m8_raw", "acc_m16_raw"]):
        e1 = g("acc_m0_raw") / g("acc_m0")
        t["E1"] = dict(n1_raw=g("acc_m0_raw"), n1=g("acc_m0"), ratio=e1, held=bool(1 / 1.5 <= e1 <= 1.5))
        t["E2"] = dict(n1_raw=g("acc_m8_raw"), n1_raw_sets=r["acc_m8_raw"]["n1"], flags=r["acc_m8_raw"]["n1_flag"],
                       n1=g("acc_m8"), held=bool(g("acc_m8_raw") > 1.5 * g("acc_m8")))
        e16 = r["acc_m16_raw"]["trend_512"]
        t["E3"] = dict(trend_512=e16, held=bool(min(e16) > 1))
        b = r["acc_m16_raw"].get("bias_share_by_n", [None])[-1]
        s_raw, s_fe = r["acc_m16_raw"].get("spread_fold_512"), r["acc_m16"].get("spread_fold_512")
        t["E4"] = dict(bias_share_512=b, spread_fold_raw=None if s_raw is None else geo(s_raw), spread_fold=None if s_fe is None else geo(s_fe),
                       held=bool(b is not None and s_raw is not None and b > 0.5 and geo(s_raw) > geo(s_fe)))
    # v3 ranges with the raw embedding R1-R4 (written after the embedding test, before the runs)
    if all(nm in r for nm in RAW3):
        ratio = {nm: g(nm + "_raw") / g(nm) for nm in ["gA", "late", "potts_wide", "onset"]}
        t["R1"] = dict(ratio=ratio, n1_raw={nm: r[nm + "_raw"]["n1"] for nm in ratio}, held=bool(all(ratio[nm] <= 0.5 for nm in ["gA", "late", "potts_wide"])))
        low = {nm: g(nm + "_raw", "trend_low_n") / g(nm, "trend_low_n") for nm in ["gD", "potts_narrow"]}
        t["R2"] = dict(ratio=low, held=bool(all(v <= 0.6 for v in low.values())))
        b = r["onset_raw"].get("bias_share_512")
        t["R3"] = dict(ratio_onset=ratio["onset"], bias_share_512=b, held=bool(all(ratio["onset"] > ratio[nm] for nm in ["gA", "late", "potts_wide"]) and b is not None and b > 0.5))
        dd = {nm: r[nm + "_raw"]["d_star"] / r[nm]["d_star"] for nm in ["gA", "gD", "late", "onset", "potts_wide", "potts_narrow"] if "d_star" in r[nm + "_raw"]}
        t["R4"] = dict(d_ratio=dd, held=bool(len(dd) == 6 and all(abs(v - 1) < 0.05 for v in dd.values())))
    # encoding bandwidth sweep S1-S4 (written after the manuscript rewrite, before the runs); set 0 throughout
    if all(nm in r for nm in BW):
        n0 = lambda nm: r[nm]["n1"][0]
        seq = [n0("acc_m0_raw"), n0("acc_m0_f0"), n0("acc_m0_f2"), n0("acc_m0")]
        t["S1"] = dict(n1_raw_f0_f2_full=seq, held=bool(seq[0] <= seq[1] < seq[2] < seq[3] and seq[1] <= 1.5 * seq[0]))
        for k, nm in (("S2", "acc_m8_f0"), ("S3", "acc_m8_f2")):
            t[k] = dict(trend_512=r[nm]["trend_512"][0], flag=r[nm]["n1_flag"][0], held=bool(r[nm]["n1_flag"][0] == ">"))
        t["S4"] = dict(n1_f2=n0("acc_m4_f2"), n1_full=n0("acc_m4"), held=bool(n0("acc_m4_f2") < n0("acc_m4") / 1.5))
    return t


def local(st):
    """Within each range: local fiber dimension d(sigma* | theta_i) against the local trend error of
    the models at the same setting (geometric mean over training sets), at every training-set size.
    Within-range Spearman correlations, and a pooled correlation of within-range centred logs."""
    out, pooled = {}, {}
    for p in sorted(RES.glob("local_v4_*.json")):
        m = json.loads(p.read_text())
        nm = m["range"]
        ds = sorted(st.get(nm, []), key=lambda d: d["rows"][0]["split"])
        sg = np.log(m["sigmas"])
        dl = np.exp(np.mean([[math.log(np.interp(math.log(SIGMA_STAR), sg, di)) for di in per] for per in m["d_local"]], 0))
        ladder = [r["n"] for r in ds[0]["rows"]]
        rows = []
        for j, n in enumerate(ladder):
            e = np.exp(np.mean([np.log(np.maximum(np.array(d["rows"][j]["processing_grid"]), 1e-3)) for d in ds], 0))
            rho = float(stats.spearmanr(dl, e)[0]) if np.ptp(dl) > 0 else None
            rows.append(dict(n=n, rho=rho))
            pooled.setdefault(n, []).append((np.log(dl) - np.log(dl).mean(), np.log(e) - np.log(e).mean()))
        out[nm] = dict(d_local=dl.tolist(), spread_d=float(dl.max() / dl.min()), by_n=rows,
                       exact_local=None if m["exact_local"] is None else
                       [float(np.interp(math.log(SIGMA_STAR), sg, x)) for x in m["exact_local"]])
    pool = {}
    for n, v in pooled.items():
        x = np.concatenate([a for a, _ in v])
        y = np.concatenate([b for _, b in v])
        pool[n] = float(np.corrcoef(x, y)[0, 1])
    return dict(per_range=out, pooled_r=pool)


if __name__ == "__main__":
    st = load_studies()
    r = per_range(st)
    t = tests(r)
    t["local"] = local(st)
    (RES / "summary_v4.json").write_text(json.dumps(r, indent=1))
    (RES / "tests_v4.json").write_text(json.dumps(t, indent=1))
    for k, v in t.items():
        print(k, json.dumps(v)[:400])

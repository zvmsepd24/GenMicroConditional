"""Three-system analysis (v3): data needed per range, the pre-registered tests of the historical v3 tests, and
summary_v3.json for the figures and the manuscript."""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from analyze import PAIRS, curvature, curve, fit_n0, load_studies, n_needed, noise_dimension  # noqa: E402

RES = HERE.parent / "results"
NAMES = ["gA", "gB", "gC", "gD", "late", "late_narrow", "onset", "transition", "late_red", "potts_wide", "potts_narrow", "potts_early"]


def geo(x):
    return float(np.exp(np.mean(np.log(x))))


def per_range(st):
    out = {}
    exact = json.loads((RES / "exact_grf.json").read_text())
    for name in NAMES:
        ds = sorted(st.get(name, []), key=lambda d: d["rows"][0]["split"])
        if not ds:
            continue
        n1 = [n_needed(*curve(d, "processing_median"), 1.0) for d in ds]
        n05 = [n_needed(*curve(d, "processing_median"), 0.5) for d in ds]
        fits = [fit_n0(d) for d in ds]
        o = dict(sets=len(ds), trend_size=ds[0]["rows"][0]["effect_median"], d_eff=noise_dimension(name), curvature=curvature(name),
                 n1=[v for v, _ in n1], n1_flag=[f for _, f in n1], n05=[v for v, _ in n05], n05_flag=[f for _, f in n05],
                 n0=geo([a for a, _ in fits]), n0_sets=[a for a, _ in fits], slope=[b for _, b in fits],
                 trend_512=[d["rows"][-1]["processing_median"] for d in ds], texture_512=[d["rows"][-1]["texture"] for d in ds],
                 spread_512=[d["rows"][-1]["spread_median"] for d in ds],
                 ratio_16=[d["rows"][0]["processing_median"] / d["rows"][0]["effect_median"] for d in ds],
                 ratio_512=[d["rows"][-1]["processing_median"] / d["rows"][-1]["effect_median"] for d in ds],
                 i_diff=[d["rows"][-1].get("i_diff") for d in ds])
        n, _ = curve(ds[0], "processing_median")
        rat = [curve(d, "processing_median")[1] / curve(d, "effect_median")[1] for d in ds]
        o["invented_until"] = [n_needed(n, r, 1.0)[0] if r[0] > 1 else None for r in rat]
        if name in exact:
            o["I_exact"] = exact[name]["I"]
        out[name] = o
    return out


def tests(r):
    t = {}
    # 1. wide vs narrow pairs
    pairs = []
    for w, nrw, lab in PAIRS:
        if w in r and nrw in r:
            pairs.append(dict(pair=lab, wide=w, narrow=nrw, n1_wide=geo(r[w]["n1"]), n1_narrow=geo(r[nrw]["n1"]),
                              ratio=geo(r[w]["n1"]) / geo(r[nrw]["n1"]), size_ratio=r[w]["trend_size"] / r[nrw]["trend_size"],
                              n0_ratio=r[w]["n0"] / r[nrw]["n0"]))
    t["pairs"] = pairs
    # 2. within-pair slope of log N1 on log trend size, set-level points, fixed effect per pair
    y, x, g = [], [], []
    for k, p in enumerate(pairs):
        for name in (p["wide"], p["narrow"]):
            for v in r[name]["n1"]:
                y.append(math.log(v)), x.append(math.log(r[name]["trend_size"])), g.append(k)
    y, x, g = map(np.array, (y, x, g))
    yc = y - np.array([y[g == k].mean() for k in g])
    xc = x - np.array([x[g == k].mean() for k in g])
    beta = float((xc * yc).sum() / (xc ** 2).sum())
    resid = yc - beta * xc
    df = len(y) - len(pairs) - 1
    se = float(math.sqrt((resid ** 2).sum() / df / (xc ** 2).sum()))
    tq = stats.t.ppf(0.975, df)
    t["slope_trend_size"] = dict(beta=beta, ci=[beta - tq * se, beta + tq * se], df=df, set_scatter_sd=float(math.sqrt((resid ** 2).sum() / df)))
    # 3. averaging: N0.5 / N1 (only uncensored)
    ratios = []
    for name, o in r.items():
        for a, b, fa, fb in zip(o["n05"], o["n1"], o["n05_flag"], o["n1_flag"]):
            if not fa and not fb:
                ratios.append(a / b)
    t["n05_over_n1"] = dict(geo_mean=geo(ratios), range=[min(ratios), max(ratios)], n=len(ratios))
    t["slopes"] = dict(range=[min(s for o in r.values() for s in o["slope"]), max(s for o in r.values() for s in o["slope"])])
    # 4. transition and onset vs late; redundant vs late
    for name in ("transition", "onset", "late_red"):
        if name in r and "late" in r:
            t[f"{name}_vs_late"] = dict(n1_ratio=geo(r[name]["n1"]) / geo(r["late"]["n1"]),
                                        trend512=[r[name]["trend_512"], r["late"]["trend_512"]],
                                        curvature=[r[name]["curvature"], r["late"]["curvature"]],
                                        trend_size=[r[name]["trend_size"], r["late"]["trend_size"]])
    # 5. N0 vs noise dimension and curvature across ranges (log scale)
    names = [n for n in r if n != "late_red"]
    ln0 = np.log([r[n]["n0"] for n in names])
    t["n0_vs_d_eff"] = dict(spearman=float(stats.spearmanr([r[n]["d_eff"] for n in names], ln0)[0]))
    t["n0_vs_curvature"] = dict(spearman=float(stats.spearmanr([r[n]["curvature"] for n in names], ln0)[0]))
    t["n0_vs_trend_size"] = dict(spearman=float(stats.spearmanr([r[n]["trend_size"] for n in names], ln0)[0]))
    return t


if __name__ == "__main__":
    st = load_studies()
    r = per_range(st)
    t = tests(r)
    (RES / "summary_v3.json").write_text(json.dumps(r, indent=1))
    (RES / "tests_v3.json").write_text(json.dumps(t, indent=1))
    for name, o in r.items():
        print(f"{name:13s} sets {o['sets']} size {o['trend_size']:5.2f} D {o['d_eff']:4.1f} curv {o['curvature']:5.2f}  N1 "
              + ", ".join(f"{f}{v:.0f}" for v, f in zip(o["n1"], o["n1_flag"])) + f"  N0 {o['n0']:.0f}  slope "
              + ", ".join(f"{s:.2f}" for s in o["slope"]) + "  trend512 " + ", ".join(f"{v:.2f}" for v in o["trend_512"])
              + "  spread " + ", ".join(f"{v:.2f}" for v in o["spread_512"]))
    print(json.dumps(t, indent=1))

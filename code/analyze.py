"""Summaries of the sweeps and estimators: errors against N, data needed, information estimates."""
import json
import sys
import math
from pathlib import Path

import numpy as np

RES = Path(__file__).parent.parent / "results"
RANGES = ["gA", "gB", "gC", "gD", "onset", "late", "late_narrow", "transition", "late_red", "potts_wide", "potts_narrow", "potts_early"]
AREA_UNITS = {"gA": "px^2", "gB": "px^2", "gC": "px^2", "gD": "px^2"}


def load_studies():
    out = {}
    for p in sorted(RES.glob("study_*.json")):
        d = json.loads(p.read_text())
        out.setdefault(d["range"], []).append(d)
    return out


def load_posteriors():
    out = {}
    for p in sorted(RES.glob("posterior_*.json")):
        d = json.loads(p.read_text())
        out[p.stem.replace("posterior_", "")] = d["rows"]
    return out


def curve(d, key):
    return np.array([r["n"] for r in d["rows"]], float), np.array([r[key] for r in d["rows"]], float)


def n_needed(n, e, thr):
    """Smallest N at which the error falls to thr, by linear interpolation in (log N, log error).
    Returns (value, flag): flag '<' if already below at the first N, '>' if never reached."""
    if e[0] <= thr:
        return n[0], "<"
    for i in range(1, len(n)):
        if e[i] <= thr:
            a = (math.log(thr) - math.log(e[i - 1])) / (math.log(e[i]) - math.log(e[i - 1]))
            return math.exp(math.log(n[i - 1]) + a * (math.log(n[i]) - math.log(n[i - 1]))), ""
    return n[-1], ">"


def info_estimate(post, name, which="i_test"):
    """Posterior-network estimate at the largest N_est (mean over replicates)."""
    rows = [r for r in post.get(name, []) if r["n"] == max(r2["n"] for r2 in post[name])]
    return float(np.mean([r[which] for r in rows])) if rows else float("nan")


def geo_mean_curve(ds, key):
    n = curve(ds[0], key)[0]
    e = np.exp(np.mean([np.log(np.maximum(curve(d, key)[1], 1e-3)) for d in ds], 0))
    return n, e


def fit_n0(d, key="processing_median", n_min=32):
    """Fit error = (N0 / N)^(1/2) and error = a N^-s by least squares in log space over N >= n_min."""
    n, e = curve(d, key)
    m = n >= n_min
    ln, le = np.log(n[m]), np.log(e[m])
    n0 = float(np.exp(np.mean(2 * le + ln)))
    s, b = np.polyfit(ln, le, 1)
    return n0, float(-s)


def noise_dimension(name):
    """E[d^T C^-1 d] for d drawn from the single-field scatter, median over the grid: the number of
    independent ways one field's descriptors fluctuate, as seen by the error metric."""
    import torch
    import study as T
    base = T.base_of(name)
    ref = torch.load(RES / f"ref_{base}_100.pt")
    sc = T.scale_of(ref)
    R = ref["cov"] / (sc[:, None] * sc[None, :])
    return float(torch.diagonal(torch.linalg.solve(T.metric(ref, sc), R), dim1=-2, dim2=-1).sum(-1).median())


def curvature(name):
    """How sharply the mean microstructure bends inside the range: Mahalanobis length of the second
    difference mu(-2/3) - 2 mu(0) + mu(2/3) along each row and column of the 3 x 3 reference grid
    (single-image scatter units), median over the six lines. A straight-line change gives zero."""
    import torch
    import study as T
    base = T.base_of(name)
    ref = torch.load(RES / f"ref_{base}_100.pt")
    sc = T.scale_of(ref)
    mu = (ref["mu"] / sc).reshape(3, 3, -1)
    C = T.metric(ref, sc).reshape(3, 3, len(sc), len(sc))[1, 1]
    lines = [mu[i, 0] - 2 * mu[i, 1] + mu[i, 2] for i in range(3)] + [mu[0, j] - 2 * mu[1, j] + mu[2, j] for j in range(3)]
    d = torch.stack(lines)
    return float(torch.sqrt((d[:, None, :] @ torch.linalg.solve(C, d[:, :, None]))[:, 0, 0]).median())


PAIRS = [("gA", "gC", "GRF fine"), ("gB", "gD", "GRF coarse"), ("late", "late_narrow", "Cahn-Hilliard"),
         ("potts_wide", "potts_narrow", "Potts")]


def summary():
    studies, post = load_studies(), load_posteriors()
    exact = json.loads((RES / "exact_grf.json").read_text())
    out = {}
    for name in RANGES:
        ds = sorted(studies.get(name, []), key=lambda d: d["rows"][0]["split"])
        if not ds:
            continue
        o = dict(sets=len(ds), effect=ds[0]["rows"][0]["effect_median"], d_eff=noise_dimension(name), curvature=curvature(name))
        o["n_needed_1"] = [n_needed(*curve(d, "processing_median"), 1.0) for d in ds]
        o["n_needed_05"] = [n_needed(*curve(d, "processing_median"), 0.5) for d in ds]
        o["fit"] = [fit_n0(d) for d in ds]
        o["processing"] = [curve(d, "processing_median")[1].tolist() for d in ds]
        o["ratio"] = [(curve(d, "processing_median")[1] / curve(d, "effect_median")[1]).tolist() for d in ds]
        o["texture"] = [curve(d, "texture")[1].tolist() for d in ds]
        o["total"] = [curve(d, "error_median")[1].tolist() for d in ds]
        o["spread_512"] = [d["rows"][-1]["spread_median"] for d in ds]
        o["i_diff"] = [d["rows"][-1].get("i_diff") for d in ds]
        o["i_loss"] = [d["rows"][-1].get("i_loss") for d in ds]
        if name in exact:
            o["I_exact"] = exact[name]["I"]
        out[name] = o
    out["posterior"] = post
    (RES / "summary.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        s = summary()
        for name in RANGES:
            if name in s:
                o = s[name]
                print(f"{name:9s} effect {o['effect']:.2f} D_eff {o['d_eff']:.2f}  N(1) " + ", ".join(f"{f}{v:.0f}" for v, f in o["n_needed_1"])
                      + "  N(0.5) " + ", ".join(f"{f}{v:.0f}" for v, f in o["n_needed_05"])
                      + "  N0 " + ", ".join(f"{a:.0f}" for a, _ in o["fit"]) + "  slope " + ", ".join(f"{b:.2f}" for _, b in o["fit"])
                      + "  N0/D_eff %.1f" % (np.exp(np.mean([np.log(a) for a, _ in o["fit"]])) / o["d_eff"]))
        sys.exit()
    studies, post = load_studies(), load_posteriors()
    exact = json.loads((RES / "exact_grf.json").read_text()) if (RES / "exact_grf.json").exists() else {}
    print("== errors against N (median over the 3x3 grid) ==")
    for name in RANGES:
        for d in studies.get(name, []):
            rows = d["rows"]
            s = rows[0]["split"]
            print(f"{name:9s} set {s}  effect {rows[0]['effect_median']:.2f}  floor {rows[0]['floor_processing']:.2f}")
            for r in rows:
                extra = f"  I_diff {r['i_diff']:.1f}  I_loss {r['i_loss']:.1f}" if "i_diff" in r else ""
                print(f"   N {r['n']:4d}  total {r['error_median']:.2f}  texture {r['texture']:.2f}  processing {r['processing_median']:.2f} (max {r['processing_max']:.2f})  spread {r['spread_median']:.2f}{extra}")
    print("== data needed (processing error) ==")
    for name in RANGES:
        ds = studies.get(name, [])
        if not ds:
            continue
        for thr in (1.0, 0.5):
            vals = [n_needed(*curve(d, "processing_median"), thr) for d in ds]
            print(f"{name:9s} thr {thr}: " + ", ".join(f"{f}{v:.0f}" for v, f in vals))
    print("== information (nats) ==")
    for name, rows in post.items():
        for r in rows:
            print(f"{name:14s} N {r['n']:5d} rep {r['replicate']}  cross {r['i_cross']:.2f} +- {r['i_cross_se']:.2f}  test {r['i_test']:.2f}"
                  + (f"   exact {exact[name]['I']:.2f}" if name in exact else ""))

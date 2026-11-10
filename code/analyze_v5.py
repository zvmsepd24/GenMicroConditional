"""v5 analysis: scores the predictions of the historical v5 tests (V1-V8, W1-W4) and summarizes the geometry of
each range -> results/summary_v5.json, tests_v5.json.

Metrics are in the scaled parameters u. For a range, "exact" is the GRF metric, "desc" the real
descriptor metric (a lower bound on the image metric), "model" the metric read from a model's
denoiser (step 0.002 unless stated) and "gen" the descriptor metric of the model's generated images.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RES = HERE.parent / "results"
GRF = ["gA", "gB", "gC", "gD"]
ACC = ["acc_m0", "acc_m1", "acc_m2", "acc_m4", "acc_m8", "acc_m16"]
V3 = ["gA", "gB", "gC", "gD", "late", "late_narrow", "onset", "transition", "late_red", "potts_wide", "potts_narrow", "potts_early"]
RAWV = ["gA_raw", "gD_raw", "gC_raw", "late_raw", "onset_raw", "potts_wide_raw", "potts_narrow_raw", "late_red_raw"]
P_RED = np.array([[1, 0, 0], [0, 0.5, 0.5]])  # late (u1, u2) from late_red (u1, a, b)
NULL = np.array([0, 1, -1]) / math.sqrt(2)
SIGMA_STAR = 0.5


def load(kind, name):
    p = RES / f"geom_v5_{kind}_{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def base(name):
    n = name.removesuffix("_raw")
    return "late" if n == "late_red" else n


def rel(Gm, Gt):
    Gm, Gt = np.asarray(Gm), np.asarray(Gt)
    return np.linalg.norm(Gm - Gt, axis=(-2, -1)) / np.linalg.norm(Gt, axis=(-2, -1))


def tr(G):
    return np.trace(np.asarray(G), axis1=-2, axis2=-1)


def truth(name):
    """Reference metric on the grid in the model's coordinates: exact for GRF ranges, else the real
    descriptor metric (a lower bound). Returns (G, kind)."""
    r = load("real", base(name))
    if r is None:
        return None, None
    G, kind = (np.array(r["G_exact"]), "exact") if "G_exact" in r else (np.array(r["G_desc"]), "desc")
    if name.removesuffix("_raw") == "late_red":
        G = np.einsum("aj,gab,bk->gjk", P_RED, G, P_RED)
    return G, kind


def eig_summary(G):
    """Per setting: eigenvalues (ascending), angle of the stiff direction from the u1 axis (degrees),
    and anisotropy (largest / smallest)."""
    G = np.asarray(G)
    w, v = np.linalg.eigh(G)
    out = dict(eigvals=w.tolist())
    if G.shape[-1] == 2:
        out["stiff_angle"] = (np.degrees(np.arctan2(v[:, 1, -1], v[:, 0, -1])) % 180).tolist()
        out["anisotropy"] = (w[:, -1] / np.maximum(w[:, 0], 1e-12)).tolist()
    return out


def volume(G, theta_u):
    """Fisher volume int sqrt(det G) d^2u over [-1, 1]^2 from the 3 x 3 grid (mean times area 4), and
    the Clarke-Barron count of distinguishable settings V / (2 pi e)."""
    d = np.sqrt(np.clip(np.linalg.det(np.asarray(G)), 0, None))
    V = 4 * float(d.mean())
    return dict(volume=V, distinguishable=V / (2 * math.pi * math.e))


def per_range():
    out = {}
    for name in V3 + ACC + RAWV + ["acc_m0_raw", "acc_m8_raw", "acc_m16_raw"]:
        m = load("model", name)
        Gt, kind = truth(name)
        if m is None and Gt is None:
            continue
        o = dict(truth_kind=kind)
        if Gt is not None:
            o["truth"] = eig_summary(Gt)
            if Gt.shape[-1] == 2:
                o["truth_volume"] = volume(Gt, None)
        if m is not None:
            o["sets"] = m["sets"]
            Gm = [np.array(s["G"]) for s in m["per_set"]]
            o["model"] = eig_summary(Gm[0])
            o["model_tr"] = [tr(g).tolist() for g in Gm]
            if Gm[0].shape[-1] == 2:
                o["model_volume"] = volume(Gm[0], None)
            if Gt is not None:
                o["rel_error_by_set"] = [float(np.median(rel(g, Gt))) for g in Gm]
                o["tr_ratio_by_set"] = [float(np.median(tr(g) / tr(Gt))) for g in Gm]
            s0 = m["per_set"][0]
            if "G_by_h" in s0:
                o["tr_ratio_by_h"] = {"0.002": float(np.median(tr(s0["G"]) / tr(Gt)))} if Gt is not None else {}
                for h, g in s0["G_by_h"].items():
                    if Gt is not None:
                        o["tr_ratio_by_h"][h] = float(np.median(tr(g) / tr(Gt)))
                o["h_drop"] = float(np.median(tr(s0["G_by_h"]["0.2"]) / tr(s0["G"])))
            if "G_gen_desc" in s0:
                gd = np.array(s0["G_gen_desc"])
                r = load("real", base(name))
                if r is not None and gd.shape[-1] == 2:
                    o["gen_vs_desc_rel"] = float(np.median(rel(gd, r["G_desc"])))
                    o["gen_vs_desc_tr"] = float(np.median(tr(gd) / tr(r["G_desc"])))
            # share of the processing information carried by each noise scale (first set, mean over settings)
            integ = np.array(s0["integrand"])  # (G, S, k, k)
            ti = np.trace(integ, axis1=-2, axis2=-1).mean(0)
            o["integrand_tr"] = ti.tolist()
            o["sigmas"] = m["sigmas"]
        r = load("real", base(name))
        if r is not None and "G_exact" in r:
            o["desc_over_exact_tr"] = float(np.median(tr(r["G_desc"]) / tr(r["G_exact"])))
            o["integrand_exact_tr"] = np.trace(np.array(r["integrand_exact"]), axis1=-2, axis2=-1).mean(0).tolist()
        out[name] = o
    return out


def tests(s):
    t = {}
    held = lambda b: bool(b)
    # V1: sinusoidal models vs exact
    v1 = {nm: s[nm]["rel_error_by_set"] for nm in GRF + ACC[:4] if nm in s and "rel_error_by_set" in s[nm]}
    if v1:
        t["V1"] = dict(rel_error=v1, held=held(all(max(x) <= 0.25 for x in v1.values())))
    # W1: raw models vs exact
    w1 = {nm: s[nm]["rel_error_by_set"] for nm in ["gA_raw", "gD_raw", "acc_m0_raw", "gC_raw"] if nm in s and "rel_error_by_set" in s[nm]}
    if w1:
        t["W1"] = dict(rel_error=w1, held=held(all(max(w1[nm]) <= 0.25 for nm in ["gA_raw", "gD_raw", "acc_m0_raw"] if nm in w1)))
    # V2 / W2: redundant direction
    for key, nm in (("V2", "late_red"), ("W2", "late_red_raw")):
        m = load("model", nm)
        if m is None:
            continue
        res = []
        for st in m["per_set"]:
            G = np.array(st["G"])
            w, v = np.linalg.eigh(G)
            ratio = w[:, 0] / w[:, -1]
            ang = np.degrees(np.arccos(np.clip(np.abs(v[:, :, 0] @ NULL), 0, 1)))
            res.append(dict(max_ratio=float(ratio.max()), median_angle=float(np.median(ang)), ratio=ratio.tolist(), angle=ang.tolist()))
        t[key] = dict(per_set=res, held=held(all(r["max_ratio"] < 0.1 and r["median_angle"] < 20 for r in res)))
    # V3: descriptor lower bound respected by the model's image metric
    ok, total = 0, 0
    for nm in V3:
        m = load("model", nm)
        Gt, kind = truth(nm)
        r = load("real", base(nm))
        if m is None or r is None:
            continue
        Gd = np.array(r["G_desc"])
        if nm == "late_red":
            Gd = np.einsum("aj,gab,bk->gjk", P_RED, Gd, P_RED)
        c = tr(Gd) <= 1.1 * tr(m["per_set"][0]["G"])
        ok += int(c.sum())
        total += len(c)
    if total:
        t["V3"] = dict(share=ok / total, n=total, held=held(ok / total >= 0.9))
    # V4: generated descriptor dynamics vs real
    v4 = {nm: s[nm]["gen_vs_desc_rel"] for nm in V3 if nm in s and "gen_vs_desc_rel" in s[nm]}
    if v4:
        t["V4"] = dict(rel_error=v4, n_within=int(sum(x <= 0.35 for x in v4.values())), n=len(v4),
                       held=held(sum(x <= 0.35 for x in v4.values()) >= 10))
    # V5: folds the model cannot represent
    v5 = {}
    for nm in ["acc_m8", "acc_m16", "acc_m8_raw", "acc_m16_raw"]:
        m = load("model", nm)
        Gt, _ = truth(nm)
        if m is None or Gt is None:
            continue
        v5[nm] = [float(np.array(st["G"])[:, 0, 0].sum() / Gt[:, 0, 0].sum()) for st in m["per_set"]]
    if len(v5) == 4:
        t["V5"] = dict(g11_ratio=v5, held=held(max(v5["acc_m8_raw"] + v5["acc_m16_raw"]) < 0.5
                                              and all(0.5 <= x <= 2 for x in v5["acc_m8"] + v5["acc_m16"])))
    # W3: sinusoidal excess falls with the step
    w3 = {nm: s[nm]["h_drop"] for nm in GRF if nm in s and "h_drop" in s[nm]}
    if w3:
        t["W3"] = dict(ratio_02_over_0002=w3, held=held(all(x < 0.5 for x in w3.values())))
    # V6, W4, V7: ladder
    lad = {nm: load("ladder", nm) for nm in ["gC", "gA", "gC_raw", "gA_raw", "late_raw", "potts_wide_raw", "late_red_raw"]}
    ratios = {}
    for nm, L in lad.items():
        if L is None:
            continue
        Gt, kind = truth(nm)
        if "setting_index" in L:  # small ladder: one setting
            Gt = Gt[L["setting_index"]:L["setting_index"] + 1]
        ratios[nm] = dict(n=[r["n"] for r in L["rows"]], tr_ratio=[float(np.median(tr(r["G"]) / tr(Gt))) for r in L["rows"]], kind=kind,
                          g_ratio=[(np.diagonal(np.array(r["G"]), axis1=-2, axis2=-1) / np.diagonal(Gt, axis1=-2, axis2=-1)).mean(0).tolist() for r in L["rows"]])
    t["ladder_tr_ratio"] = ratios
    if "gC" in ratios and "gA" in ratios:
        rc, ra = ratios["gC"]["tr_ratio"], ratios["gA"]["tr_ratio"]
        t["V6"] = dict(gC=rc, gA=ra, held=held(rc[0] > 2 and abs(rc[-1] - 1) <= 0.25 and rc[0] > ra[0]))
    if "gC_raw" in ratios and "gA_raw" in ratios:
        t["W4"] = dict(gC_raw_16=ratios["gC_raw"]["tr_ratio"][0], gA_raw_16=ratios["gA_raw"]["tr_ratio"][0],
                       held=held(ratios["gC_raw"]["tr_ratio"][0] < 2 and ratios["gA_raw"]["tr_ratio"][0] < 2 and t.get("V6", {}).get("held", False)))
    v7 = {}
    for nm in ["gC", "gA", "gC_raw", "gA_raw"]:
        L = lad.get(nm)
        mm = RES / f"measure_v4_{base(nm)}.json"
        if L is None or not mm.exists():
            continue
        ex = json.loads(mm.read_text())["exact"]
        dsig = np.log(L["dsig"])
        e_star = float(np.interp(math.log(SIGMA_STAR), dsig, ex))
        dd = [float(np.interp(math.log(SIGMA_STAR), dsig, r["d_cond"])) / e_star for r in L["rows"]]
        v7[nm] = dd
    if v7:
        t["V7"] = dict(d_over_exact=v7, held=held(all(abs(x[0] - 1) <= 0.10 and abs(x[-1] - 1) <= 0.02 for x in v7.values())))
    # V8: stiff direction in late and potts_wide
    v8 = {}
    for nm, mnm in (("late", "late_raw"), ("potts_wide", "potts_wide_raw")):
        r = load("real", nm)
        m = load("model", mnm)
        if r is None or m is None:
            continue
        gd, gm = np.array(r["G_desc"])[4], np.array(m["per_set"][0]["G"])[4]
        v8[nm] = dict(desc=gd.tolist(), model=gm.tolist(), u2_stiff=bool(gd[1, 1] > gd[0, 0] and gm[1, 1] > gm[0, 0]))
    if v8:
        t["V8"] = v8
    return t


if __name__ == "__main__":
    s = per_range()
    t = tests(s)
    (RES / "summary_v5.json").write_text(json.dumps(s, indent=1))
    (RES / "tests_v5.json").write_text(json.dumps(t, indent=1))
    for k, v in t.items():
        print(k, json.dumps(v)[:500])

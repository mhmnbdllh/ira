#!/usr/bin/env python3
"""Data-based Monte Carlo evaluation of the confidence intervals reported by
the application. Each sample dataset is treated as a finite population;
R replicate samples of the same size (n = 86 units) are drawn with
replacement (units resampled as whole rows, so the rater structure is kept);
on each replicate the application's own engine (run_multi_irar, alpha=.05,
1,000 bootstrap resamples for Krippendorff's alpha) is run exactly as in the
web interface. Coverage = share of replicates whose interval contains the
coefficient's population value (its value on the full dataset); SE ratio =
mean reported SE / empirical SD of the replicate estimates.
Usage: coverage_sim.py <repo_dir> <out_json> <R> <seed>"""
import sys, json, time
import numpy as np, pandas as pd
from multiprocessing import Pool

repo, out_json, R, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
sys.path.insert(0, repo)
from utils.file_loader import load_sample_file
from modules.reliability import run_multi_irar

SC = {
 "S1": ("Dichotomous", ["R1", "R2"], "dichotomous", 2, None),
 "S2": ("Dichotomous", ["R1", "R2", "R3", "R4"], "dichotomous", 2, None),
 "S3": ("Nominal-Ordinal", ["R1", "R2"], "ordinal", 5, 1),
 "S4": ("Nominal-Ordinal", ["R1", "R2", "R3", "R4"], "ordinal", 5, 1),
 "S5": ("Interval-Ratio", ["R1", "R2"], "interval", None, None),
 "S6": ("Interval-Ratio", ["R1", "R2", "R3", "R4"], "interval", None, None),
}
DATA = {k: load_sample_file(f"{repo}/sample-resmeth_{k}.csv") for k in ("Dichotomous", "Nominal-Ordinal", "Interval-Ratio")}

def extract(c):
    """(estimate, se, lo_z, hi_z, lo_t, hi_t) per coefficient label; se/t may be None."""
    o = {}
    pa = c.get("percentage_agreement")
    if pa: o["PA"] = (pa["pa"], None, pa["ci_lo"], pa["ci_hi"], None, None)
    for key, name, est in (("scott_pi", "Scott pi", "pi"), ("brennan_prediger", "BP", "kappa_bp"), ("gwet_ac1", "AC1", "ac1")):
        r = c.get(key)
        if r: o[name] = (r[est], r["se"], r["ci_lo_z"], r["ci_hi_z"], r["ci_lo_t"], r["ci_hi_t"])
    ac2 = (c.get("gwet_ac2") or {}).get("schemes") or {}
    for sch, r in ac2.items():
        o[f"AC2 {sch}"] = (r["ac2"], r["se"], r["ci_lo_z"], r["ci_hi_z"], r["ci_lo_t"], r["ci_hi_t"])
    ka = c.get("krippendorff")
    if ka: o["Kripp alpha"] = (ka["alpha"], None, ka["ci_lo"], ka["ci_hi"], None, None)
    cc = c.get("lin_ccc")
    if cc: o["CCC"] = (cc["ccc"], None, cc["ci_lo"], cc["ci_hi"], None, None)
    ic = c.get("icc")
    if ic and "error" not in ic:
        for f in ("1_1", "2_1", "3_1", "1_k", "2_k", "3_k"):
            o[f"ICC {f}"] = (ic[f"icc_{f}"], None, ic[f"icc_{f}_ci_lo"], ic[f"icc_{f}_ci_hi"], None, None)
    return o

def analyse(sc, df):
    f, cols, dt, q, smin = SC[sc]
    g = {"name": sc, "items": cols, "data_type": dt, "q_declared": q, "scale_min": smin}
    return extract(run_multi_irar(df, [g], alpha=0.05, n_boot=1000)["constructs"][0])

def one(args):
    sc, r = args
    f, cols, *_ = SC[sc]
    full = DATA[f][cols]
    rng = np.random.default_rng([seed, r, list(SC).index(sc)])
    idx = rng.integers(0, len(full), len(full))
    return sc, analyse(sc, full.iloc[idx].reset_index(drop=True))

if __name__ == "__main__":
    t0 = time.time()
    truth = {sc: analyse(sc, DATA[SC[sc][0]][SC[sc][1]]) for sc in SC}
    jobs = [(sc, r) for sc in SC for r in range(R)]
    with Pool(2) as pool:
        res = pool.map(one, jobs, chunksize=10)
    summ = {}
    for sc in SC:
        reps = [o for s, o in res if s == sc]
        summ[sc] = {}
        for name, tv in truth[sc].items():
            pop = tv[0]
            est = np.array([o[name][0] for o in reps if name in o and o[name][0] is not None and not np.isnan(o[name][0])])
            covz = [o[name][2] <= pop <= o[name][3] for o in reps if name in o and not np.isnan(o[name][2])]
            covt = [o[name][4] <= pop <= o[name][5] for o in reps if name in o and o[name][4] is not None and not np.isnan(o[name][4])]
            ses = [o[name][1] for o in reps if name in o and o[name][1] is not None and not np.isnan(o[name][1])]
            summ[sc][name] = dict(pop=float(pop), n_rep=int(len(est)), mean_est=float(est.mean()), sd_est=float(est.std(ddof=1)),
                                  cover_main=float(np.mean(covz)) if covz else None,
                                  cover_t=float(np.mean(covt)) if covt else None,
                                  se_ratio=float(np.mean(ses) / est.std(ddof=1)) if ses else None)
    json.dump(dict(R=R, seed=seed, seconds=time.time() - t0, summary=summ), open(out_json, "w"), indent=1)
    print("done in", round(time.time() - t0), "s")

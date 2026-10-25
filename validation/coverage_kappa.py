#!/usr/bin/env python3
"""Data-based coverage of the kappa intervals (Cohen Eq. 8; weighted kappa
Eq. 8; Fleiss/Gwet 2021): resample units with replacement from each sample
dataset, run the application's engine, count how often the interval covers
the full-data value. Usage: coverage_kappa.py <repo_dir> <out_json> <R>"""
import sys, json
import numpy as np
repo, out, R = sys.argv[1], sys.argv[2], int(sys.argv[3])
sys.path.insert(0, repo)
from utils.file_loader import load_sample_file
from modules.reliability import run_multi_irar

cfg = {
    "S1": ("Dichotomous", ["R1", "R2"], "dichotomous", 2, None),
    "S2": ("Dichotomous", ["R1", "R2", "R3", "R4"], "dichotomous", 2, None),
    "S3": ("Nominal-Ordinal", ["R1", "R2"], "ordinal", 5, 1),
}
def grab(sc, c):
    o = {}
    if sc == "S1":
        v = c["cohen_kappa"]["variants"]["unweighted"]
        o["Cohen kappa"] = (v["kappa"], v["se_general"], v["se_null"], v["ci_lo_z"], v["ci_hi_z"], v["ci_lo_t"], v["ci_hi_t"])
    if sc == "S3":
        for s in ("unweighted", "linear", "quadratic"):
            v = c["weighted_kappa"]["variants"][s]
            o[f"kappa {s}"] = (v["kappa"], v["se_general"], v["se_null"], v["ci_lo_z"], v["ci_hi_z"], v["ci_lo_t"], v["ci_hi_t"])
    if sc == "S2":
        f = c["fleiss_kappa"]
        o["Fleiss kappa"] = (f["kappa"], f["kappa_se_general"], f["kappa_se"], f["ci_lo_z"], f["ci_hi_z"], f["ci_lo_t"], f["ci_hi_t"])
    return o
res = {}
for sc, (fn, cols, dt, q, smin) in cfg.items():
    full = load_sample_file(f"{repo}/sample-resmeth_{fn}.csv")[cols]
    g = {"name": sc, "items": cols, "data_type": dt, "q_declared": q, "scale_min": smin}
    truth = grab(sc, run_multi_irar(full, [g], n_boot=0)["constructs"][0])
    rng = np.random.default_rng(99)
    reps = []
    for r in range(R):
        d = full.iloc[rng.integers(0, len(full), len(full))].reset_index(drop=True)
        reps.append(grab(sc, run_multi_irar(d, [g], n_boot=0)["constructs"][0]))
    for name, tv in truth.items():
        pop = tv[0]
        est = np.array([o[name][0] for o in reps])
        sd = est.std(ddof=1)
        res[f"{sc} {name}"] = dict(pop=pop, sd=sd,
            se_general_ratio=float(np.mean([o[name][1] for o in reps]) / sd),
            se_null_ratio=float(np.mean([o[name][2] for o in reps]) / sd),
            cover_z=float(np.mean([o[name][3] <= pop <= o[name][4] for o in reps])),
            cover_t=float(np.mean([o[name][5] <= pop <= o[name][6] for o in reps])))
json.dump(res, open(out, "w"), indent=1)
for k, v in res.items():
    print(k, {a: round(b, 3) for a, b in v.items()})

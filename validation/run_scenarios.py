#!/usr/bin/env python3
"""Run the application's own engine (modules.reliability.run_multi_irar) on
the six data-type x rater-count configurations, exactly as the Streamlit view
would, and dump every reported quantity to JSON.
Usage: python3 run_scenarios.py <repo_dir> <out_json> [alpha] [n_boot]"""
import sys, json, time
import numpy as np

repo, out = sys.argv[1], sys.argv[2]
alpha = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
n_boot = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
sys.path.insert(0, repo)
from utils.file_loader import load_sample_file
from modules.reliability import run_multi_irar

SC = {
 "S1_dich_k2": ("sample-resmeth_Dichotomous.csv", ["R1", "R2"], "dichotomous", 2, None),
 "S2_dich_k4": ("sample-resmeth_Dichotomous.csv", ["R1", "R2", "R3", "R4"], "dichotomous", 2, None),
 "S3_ord_k2":  ("sample-resmeth_Nominal-Ordinal.csv", ["R1", "R2"], "ordinal", 5, 1),
 "S4_ord_k4":  ("sample-resmeth_Nominal-Ordinal.csv", ["R1", "R2", "R3", "R4"], "ordinal", 5, 1),
 "S5_int_k2":  ("sample-resmeth_Interval-Ratio.csv", ["R1", "R2"], "interval", None, None),
 "S6_int_k4":  ("sample-resmeth_Interval-Ratio.csv", ["R1", "R2", "R3", "R4"], "interval", None, None),
}

def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items() if k not in ("crosstab", "icc_df", "per_unit", "method_context")}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return [clean(v) for v in o.tolist()]
    if isinstance(o, (str, int, bool)) or o is None:
        return o
    return str(o)

res = {}
for key, (f, cols, dt, q, smin) in SC.items():
    df = load_sample_file(f"{repo}/{f}")
    grp = {"name": key, "items": cols, "data_type": dt, "q_declared": q, "scale_min": smin}
    t0 = time.perf_counter()
    M = run_multi_irar(df, [grp], alpha=alpha, n_boot=n_boot)
    dt_s = time.perf_counter() - t0
    c = M["constructs"][0]
    c["_runtime_s"] = dt_s
    res[key] = clean(c)
json.dump(res, open(out, "w"), indent=1)
print("wrote", out)
for k, v in res.items():
    print(k, "methods:", v["applicable_methods"], f"runtime {v['_runtime_s']:.2f}s")

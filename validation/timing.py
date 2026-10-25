#!/usr/bin/env python3
"""Server-side computation time of run_multi_irar for the six configurations
(median of 5 runs after one warm-up run), with 1,000 and 10,000 bootstrap
resamples. Usage: timing.py <repo> <out_json>"""
import sys, json, time, statistics, platform, os
repo, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, repo)
from utils.file_loader import load_sample_file
from modules.reliability import run_multi_irar
SC = {"S1": ("Dichotomous", ["R1", "R2"], "dichotomous", 2, None), "S2": ("Dichotomous", ["R1", "R2", "R3", "R4"], "dichotomous", 2, None),
      "S3": ("Nominal-Ordinal", ["R1", "R2"], "ordinal", 5, 1), "S4": ("Nominal-Ordinal", ["R1", "R2", "R3", "R4"], "ordinal", 5, 1),
      "S5": ("Interval-Ratio", ["R1", "R2"], "interval", None, None), "S6": ("Interval-Ratio", ["R1", "R2", "R3", "R4"], "interval", None, None)}
res = {"python": platform.python_version(), "processor": platform.processor() or platform.machine(), "cpus": os.cpu_count()}
for nb in (1000, 10000):
    for k, (f, cols, dt, q, smin) in SC.items():
        df = load_sample_file(f"{repo}/sample-resmeth_{f}.csv")
        g = {"name": k, "items": cols, "data_type": dt, "q_declared": q, "scale_min": smin}
        run_multi_irar(df, [g], n_boot=nb)
        ts = []
        for _ in range(5):
            t0 = time.perf_counter(); run_multi_irar(df, [g], n_boot=nb); ts.append(time.perf_counter() - t0)
        res[f"{k}_B{nb}"] = statistics.median(ts)
        print(k, nb, round(res[f"{k}_B{nb}"], 3), flush=True)
json.dump(res, open(out, "w"), indent=1)

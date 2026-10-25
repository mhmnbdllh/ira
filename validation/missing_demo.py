#!/usr/bin/env python3
"""Missing-data demonstration: remove 10% of rating cells at random (seed 7)
from each k=4 sample dataset, run the application's engine, and compare the
alpha it reports with the krippendorff package on the same incomplete data
and with alpha computed after listwise deletion.
Usage: missing_demo.py <repo> <out_json>"""
import sys, json, time
import numpy as np, krippendorff
repo, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, repo)
from utils.file_loader import load_sample_file
from modules.reliability import run_multi_irar
cfg = {"Dichotomous": ("dichotomous", 2, None, "nominal"),
       "Nominal-Ordinal": ("ordinal", 5, 1, "ordinal"),
       "Interval-Ratio": ("interval", None, None, "interval")}
res = {}
for fn, (dt, q, smin, lvl) in cfg.items():
    df = load_sample_file(f"{repo}/sample-resmeth_{fn}.csv")[["R1", "R2", "R3", "R4"]].astype(float)
    rng = np.random.default_rng(7)
    mask = rng.random(df.shape) < 0.10
    dm = df.mask(mask)
    g = {"name": fn, "items": ["R1", "R2", "R3", "R4"], "data_type": dt, "q_declared": q, "scale_min": smin}
    t0 = time.perf_counter()
    c = run_multi_irar(dm, [g], n_boot=1000)["constructs"][0]
    ka = c["krippendorff"]
    ref = krippendorff.alpha(reliability_data=dm.T.values, level_of_measurement=lvl)
    lw = dm.dropna()
    ref_lw = krippendorff.alpha(reliability_data=lw.T.values, level_of_measurement=lvl)
    res[fn] = dict(cells_removed=int(mask.sum()), cells_total=int(mask.size),
                   n_listwise=int(len(lw)), n_pairable=int(ka.get("n_units")),
                   alpha_app=float(ka["alpha"]), alpha_pkg=float(ref), alpha_listwise=float(ref_lw),
                   ci=[float(ka["ci_lo"]), float(ka["ci_hi"])], n_units_other=int(c["n_units"]))
    print(fn, res[fn])
json.dump(res, open(out, "w"), indent=1)

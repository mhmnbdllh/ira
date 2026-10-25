#!/usr/bin/env python3
"""Independent cross-validation of every ResMeth-IRA coefficient against
(a) independent third-party implementations (scikit-learn, statsmodels,
irrCAC, krippendorff, pingouin) and (b) formulas re-typed here from the
primary sources, never calling the application's own coefficient helpers
for the reference side. Usage: validate_independent.py <repo_dir> <out_json>"""
import sys, json
import numpy as np, pandas as pd
from scipy import stats

repo, out_json = sys.argv[1], sys.argv[2]
sys.path.insert(0, repo)
from utils.file_loader import load_sample_file
from modules.reliability import run_multi_irar, _krippendorff_alpha

from sklearn.metrics import cohen_kappa_score
import statsmodels.stats.inter_rater as smir
from irrCAC.raw import CAC
import krippendorff as kdp
import pingouin as pg

D = {n: load_sample_file(f"{repo}/sample-resmeth_{n}.csv") for n in ("Dichotomous", "Nominal-Ordinal", "Interval-Ratio")}
ROWS = []
def chk(name, app, ref, tol, kind):
    d = abs(float(app) - float(ref))
    ROWS.append(dict(check=name, app=float(app), ref=float(ref), absdiff=d, tol=tol, ok=bool(d <= tol), kind=kind))

def run(dfname, cols, dt, q=None, smin=None, alpha=0.05, nb=200):
    g = {"name": "g", "items": cols, "data_type": dt, "q_declared": q, "scale_min": smin}
    return run_multi_irar(D[dfname], [g], alpha=alpha, n_boot=nb)["constructs"][0]

S1 = run("Dichotomous", ["R1", "R2"], "dichotomous", 2)
S2 = run("Dichotomous", ["R1", "R2", "R3", "R4"], "dichotomous", 2)
S3 = run("Nominal-Ordinal", ["R1", "R2"], "ordinal", 5, 1)
S4 = run("Nominal-Ordinal", ["R1", "R2", "R3", "R4"], "ordinal", 5, 1)
S5 = run("Interval-Ratio", ["R1", "R2"], "interval")
S6 = run("Interval-Ratio", ["R1", "R2", "R3", "R4"], "interval")
X1 = D["Dichotomous"][["R1", "R2"]].to_numpy(float); X2 = D["Dichotomous"][["R1", "R2", "R3", "R4"]].to_numpy(float)
X3 = D["Nominal-Ordinal"][["R1", "R2"]].to_numpy(float); X4 = D["Nominal-Ordinal"][["R1", "R2", "R3", "R4"]].to_numpy(float)
X5 = D["Interval-Ratio"][["R1", "R2"]].to_numpy(float); X6 = D["Interval-Ratio"][["R1", "R2", "R3", "R4"]].to_numpy(float)

# ---------------- Percentage agreement (re-typed) ----------------
for lab, X, S in (("S1", X1, S1), ("S2", X2, S2), ("S3", X3, S3), ("S4", X4, S4), ("S5", X5, S5), ("S6", X6, S6)):
    n, k = X.shape
    per = np.array([np.mean([X[i, a] == X[i, b] for a in range(k) for b in range(a + 1, k)]) for i in range(n)])
    pa = per.mean(); se = per.std(ddof=1) / np.sqrt(n); t = stats.t.ppf(.975, n - 1)
    chk(f"PA {lab}", S["percentage_agreement"]["pa"], pa, 1e-12, "formula")
    chk(f"PA CI lo {lab}", S["percentage_agreement"]["ci_lo"], max(0, pa - t * se), 1e-12, "formula")
    chk(f"PA CI hi {lab}", S["percentage_agreement"]["ci_hi"], min(1, pa + t * se), 1e-12, "formula")

# ---------------- Cohen's kappa family ----------------
def cohen_block(lab, X, block, S):
    r1, r2 = X[:, 0].astype(int), X[:, 1].astype(int)
    cats = sorted(set(r1) | set(r2)); idx = {c: i for i, c in enumerate(cats)}
    T = np.zeros((len(cats), len(cats)))
    for a, b in zip(r1, r2): T[idx[a], idx[b]] += 1
    v = S[block]["variants"]
    for scheme, wt in (("unweighted", None), ("linear", "linear"), ("quadratic", "quadratic")):
        sk = cohen_kappa_score(r1, r2, weights=wt)
        chk(f"Cohen {scheme} kappa {lab} vs scikit-learn", v[scheme]["kappa"], sk, 1e-12, "package")
        if wt is None:
            sm = smir.cohens_kappa(T, return_results=True)
        else:
            sm = smir.cohens_kappa(T, weights=np.arange(len(cats)), wt=wt, return_results=True)
        chk(f"Cohen {scheme} kappa {lab} vs statsmodels", v[scheme]["kappa"], sm.kappa, 1e-12, "package")
        chk(f"Cohen {scheme} null SE (FCE Eq.9) {lab} vs statsmodels std_kappa0", v[scheme]["se_null"], sm.std_kappa0, 1e-10, "package")
        chk(f"Cohen {scheme} general SE (FCE Eq.8) {lab} vs statsmodels std_kappa", v[scheme]["se_general"], sm.std_kappa, 1e-10, "package")
cohen_block("S1", X1, "cohen_kappa", S1)
cohen_block("S3", X3, "weighted_kappa", S3)

# ---------------- Scott's pi (Fleiss k=2 identity) & SE re-typed ----------------
r1, r2 = X1[:, 0], X1[:, 1]
Po = np.mean(r1 == r2); cats = [0, 1]
Pe = sum(((np.mean(r1 == c) + np.mean(r2 == c)) / 2) ** 2 for c in cats)
pi = (Po - Pe) / (1 - Pe)
chk("Scott pi S1 vs statsmodels fleiss_kappa(k=2)", S1["scott_pi"]["pi"],
    smir.fleiss_kappa(smir.aggregate_raters(X1.astype(int))[0], method="fleiss"), 1e-12, "package")
chk("Scott pi S1 vs irrCAC fleiss(k=2)", S1["scott_pi"]["pi"],
    CAC(pd.DataFrame(X1), categories=[0, 1]).fleiss()["est"]["coefficient_value"], 1e-5, "package")
sp_ref = CAC(pd.DataFrame(X1), categories=[0, 1]).fleiss()["est"]
chk("Scott pi general SE S1 (Gwet 2021, k=2) vs irrCAC", S1["scott_pi"]["se"], sp_ref["se"], 1e-5, "package")
chk("Scott pi t-CI lower S1 vs irrCAC", S1["scott_pi"]["ci_lo_t"], sp_ref["confidence_interval"][0], 1e-5, "package")
chk("Scott pi t-CI upper S1 vs irrCAC", S1["scott_pi"]["ci_hi_t"], sp_ref["confidence_interval"][1], 1e-5, "package")
pb = np.array([(np.mean(r1 == c) + np.mean(r2 == c)) / 2 for c in cats]); S1_, S2_ = np.sum(pb * (1 - pb)), np.sum(pb * (1 - pb) * (1 - 2 * pb))
chk("Scott pi null SE S1 (Fleiss, Nee & Landis 1979, k=2) re-typed", S1["scott_pi"]["se_null"], np.sqrt((S1_ ** 2 - S2_) / (len(r1) * S1_ ** 2)), 1e-12, "formula")

# ---------------- Brennan-Prediger ----------------
bp_ref = CAC(pd.DataFrame(X1), categories=[0, 1]).bp()["est"]
chk("BP S1 vs irrCAC bp", S1["brennan_prediger"]["kappa_bp"], bp_ref["coefficient_value"], 1e-5, "package")
chk("BP SE S1 (Cohen 1960 form, Pe=1/q) re-typed", S1["brennan_prediger"]["se"], np.sqrt(Po * (1 - Po) / (len(r1) * 0.25)), 1e-12, "formula")
SE_COMPARE = {"BP_S1": (S1["brennan_prediger"]["se"], bp_ref["se"])}

# ---------------- Gwet AC1 ----------------
for lab, X, S, cats in (("S1", X1, S1, [0, 1]), ("S2", X2, S2, [0, 1])):
    ref = CAC(pd.DataFrame(X), categories=cats).gwet()["est"]
    chk(f"AC1 {lab} vs irrCAC gwet", S["gwet_ac1"]["ac1"], ref["coefficient_value"], 1e-5, "package")
    SE_COMPARE[f"AC1_{lab}"] = (S["gwet_ac1"]["se"], ref["se"])
    # jackknife SE re-typed
    def ac1(Xs):
        n, k = Xs.shape
        po = np.mean([np.mean([Xs[i, a] == Xs[i, b] for a in range(k) for b in range(a + 1, k)]) for i in range(n)])
        pic = np.array([np.mean(Xs == c) for c in cats])
        pe = np.sum(pic * (1 - pic)) / (len(cats) - 1)
        return (po - pe) / (1 - pe)
    jk = np.array([ac1(np.delete(X, i, 0)) for i in range(len(X))])
    se = np.sqrt((len(X) - 1) / len(X) * np.sum((jk - jk.mean()) ** 2))
    chk(f"AC1 jackknife SE {lab} re-typed", S["gwet_ac1"]["se"], se, 1e-12, "formula")

# ---------------- Gwet AC2 (five schemes, declared 1..5) ----------------
for lab, X, S in (("S3", X3, S3), ("S4", X4, S4)):
    for sch in ("identity", "quadratic", "linear", "radical", "ordinal"):
        ref = CAC(pd.DataFrame(X), weights=sch, categories=[1, 2, 3, 4, 5]).gwet()["est"]
        app = S["gwet_ac2"]["schemes"][sch]
        chk(f"AC2[{sch}] {lab} vs irrCAC gwet", app["ac2"], ref["coefficient_value"], 1e-5, "package")
        SE_COMPARE[f"AC2_{sch}_{lab}"] = (app["se"], ref["se"])
chk("AC2[identity] == AC1 (S3, q=5) identity", S3["gwet_ac2"]["schemes"]["identity"]["ac2"],
    CAC(pd.DataFrame(X3), categories=[1, 2, 3, 4, 5]).gwet()["est"]["coefficient_value"], 1e-5, "identity")

# ---------------- Fleiss' kappa ----------------
tab = smir.aggregate_raters(X2.astype(int))[0]
chk("Fleiss S2 vs statsmodels fleiss_kappa", S2["fleiss_kappa"]["kappa"], smir.fleiss_kappa(tab, method="fleiss"), 1e-12, "package")
chk("Fleiss S2 vs irrCAC fleiss", S2["fleiss_kappa"]["kappa"], CAC(pd.DataFrame(X2), categories=[0, 1]).fleiss()["est"]["coefficient_value"], 1e-5, "package")
fl_ref = CAC(pd.DataFrame(X2), categories=[0, 1]).fleiss()["est"]
chk("Fleiss general SE S2 (Gwet 2021) vs irrCAC", S2["fleiss_kappa"]["kappa_se_general"], fl_ref["se"], 1e-5, "package")
chk("Fleiss t-CI lower S2 vs irrCAC", S2["fleiss_kappa"]["ci_lo_t"], fl_ref["confidence_interval"][0], 1e-5, "package")
chk("Fleiss t-CI upper S2 vs irrCAC", S2["fleiss_kappa"]["ci_hi_t"], fl_ref["confidence_interval"][1], 1e-5, "package")
n, k = X2.shape; p = tab.sum(0) / (n * k); qv = 1 - p
var0 = 2 / (n * k * (k - 1)) * ((p * qv).sum() ** 2 - (p * qv * (qv - p)).sum()) / (p * qv).sum() ** 2
chk("Fleiss SE S2 (Fleiss-Nee-Landis 1979) re-typed", S2["fleiss_kappa"]["kappa_se"], np.sqrt(var0), 1e-12, "formula")

# ---------------- Krippendorff's alpha ----------------
lvl = {"S1": ("nominal", X1, S1), "S2": ("nominal", X2, S2), "S3": ("ordinal", X3, S3), "S4": ("ordinal", X4, S4),
       "S5": ("interval", X5, S5), "S6": ("interval", X6, S6)}
for lab, (lv, X, S) in lvl.items():
    chk(f"Kripp alpha {lab} ({lv}) vs krippendorff pkg", S["krippendorff"]["alpha"],
        kdp.alpha(reliability_data=X.T, level_of_measurement=lv), 1e-10, "package")
a_ratio, _ = _krippendorff_alpha(X6, "ratio", n_boot=0)
chk("Kripp alpha S6 (ratio) vs krippendorff pkg", a_ratio, kdp.alpha(reliability_data=X6.T, level_of_measurement="ratio"), 1e-10, "package")
chk("Kripp alpha S2 (nominal) vs irrCAC krippendorff", S2["krippendorff"]["alpha"],
    CAC(pd.DataFrame(X2), categories=[0, 1]).krippendorff()["est"]["coefficient_value"], 1e-5, "package")
# with missing data: delete ~10% of cells at random (seed 7) in the four-rater files
rng = np.random.default_rng(7)
MISSING = {}
for name, X, dt, lv in (("Dichotomous", X2, "dichotomous", "nominal"), ("Nominal-Ordinal", X4, "ordinal", "ordinal"),
                        ("Interval-Ratio", X6, "interval", "interval")):
    Xm = X.copy(); mask = rng.random(Xm.shape) < 0.10; Xm[mask] = np.nan
    dfm = pd.DataFrame(Xm, columns=["R1", "R2", "R3", "R4"])
    g = {"name": "m", "items": ["R1", "R2", "R3", "R4"], "data_type": dt,
         "q_declared": 2 if dt == "dichotomous" else (5 if dt == "ordinal" else None),
         "scale_min": 1 if dt == "ordinal" else None}
    Sm = run_multi_irar(dfm, [g], n_boot=200)["constructs"][0]
    chk(f"Kripp alpha {name} with 10% missing ({lv}) vs krippendorff pkg", Sm["krippendorff"]["alpha"],
        kdp.alpha(reliability_data=Xm.T, level_of_measurement=lv), 1e-10, "package")
    MISSING[name] = dict(cells_missing=int(mask.sum()), units_listwise=int(Sm["n_units"]),
                         units_alpha=int(Sm["krippendorff"]["n_units"]), alpha=Sm["krippendorff"]["alpha"])

# ---------------- ICC: independent two-way ANOVA (Shrout & Fleiss 1979; McGraw & Wong 1996) ----------------
def icc_forms(X):
    n, k = X.shape; gm = X.mean()
    msr = k * np.sum((X.mean(1) - gm) ** 2) / (n - 1)
    msc = n * np.sum((X.mean(0) - gm) ** 2) / (k - 1)
    sst = np.sum((X - gm) ** 2)
    mse = (sst - msr * (n - 1) - msc * (k - 1)) / ((n - 1) * (k - 1))
    msw = (sst - msr * (n - 1)) / (n * (k - 1))
    return {"icc_1_1": (msr - msw) / (msr + (k - 1) * msw),
            "icc_2_1": (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n),
            "icc_3_1": (msr - mse) / (msr + (k - 1) * mse),
            "icc_1_k": (msr - msw) / msr,
            "icc_2_k": (msr - mse) / (msr + (msc - mse) / n),
            "icc_3_k": (msr - mse) / msr}
for lab, X, S in (("S5", X5, S5), ("S6", X6, S6)):
    ref = icc_forms(X)
    for key, val in ref.items():
        chk(f"{key} {lab} vs ANOVA re-typed", S["icc"][key], val, 1e-10, "formula")

# ---------------- Lin's CCC ----------------
x, y = X5[:, 0], X5[:, 1]; n = len(x)
mx, my = x.mean(), y.mean(); sx2, sy2 = x.var(), y.var(); sxy = np.mean((x - mx) * (y - my))
ccc = 2 * sxy / (sx2 + sy2 + (mx - my) ** 2); rho = np.corrcoef(x, y)[0, 1]
u = (mx - my) / np.sqrt(np.sqrt(sx2) * np.sqrt(sy2))
vz = ((1 - rho ** 2) * ccc ** 2 / ((1 - ccc ** 2) * rho ** 2) + 2 * ccc ** 3 * (1 - ccc) * u ** 2 / (rho * (1 - ccc ** 2) ** 2)
      - ccc ** 4 * u ** 4 / (2 * rho ** 2 * (1 - ccc ** 2) ** 2)) / (n - 2)
z = np.arctanh(ccc); zc = stats.norm.ppf(.975)
chk("CCC S5 re-typed (Lin 1989)", S5["lin_ccc"]["ccc"], ccc, 1e-12, "formula")
chk("CCC CI lo S5 re-typed (Lin 1989 Fisher-z variance)", S5["lin_ccc"]["ci_lo"], np.tanh(z - zc * np.sqrt(vz)), 1e-12, "formula")
chk("CCC CI hi S5 re-typed", S5["lin_ccc"]["ci_hi"], np.tanh(z + zc * np.sqrt(vz)), 1e-12, "formula")

out = dict(rows=ROWS, se_compare={k: [float(a), float(b)] for k, (a, b) in SE_COMPARE.items()}, missing=MISSING)
json.dump(out, open(out_json, "w"), indent=1)
bad = [r for r in ROWS if not r["ok"]]
print(f"{len(ROWS)} checks, {len(ROWS) - len(bad)} within tolerance, {len(bad)} outside")
for r in bad: print("  OUTSIDE:", r)
for kk, (a, b) in SE_COMPARE.items(): print(f"  SE {kk}: app={a:.5f} irrCAC={b:.5f} ratio={a/b:.3f}")
print("missing-data:", MISSING)

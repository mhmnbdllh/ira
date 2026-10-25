# Validation scripts

These scripts reproduce the verification of the Inter-Rater Agreement module
(`modules/reliability.py`, entry point `run_multi_irar`). Run them from the
repository root. Every script writes one JSON file, and the JSON files in this
folder are the outputs that the published tables were built from.

| Script | What it does | Command | Output |
|---|---|---|---|
| `published_examples.py` | Reproduces numerical examples printed in Fleiss, Cohen & Everitt (1969), Fleiss (1971), Shrout & Fleiss (1979) and Krippendorff (2011) | `python3 validation/published_examples.py . validation/published_examples.json` | `published_examples.json` |
| `validate_independent.py` | Compares every coefficient with scikit-learn, statsmodels, irrCAC, krippendorff and with formulas recomputed from the primary sources | `python3 validation/validate_independent.py . validation/validate_independent.json` | `validate_independent.json` |
| `run_scenarios.py` | Runs the six configurations S1–S6 of the three sample files (alpha = 0.05, 1,000 bootstrap resamples) | `python3 validation/run_scenarios.py . validation/run_scenarios.json 0.05 1000` | `run_scenarios.json` |
| `coverage_sim.py` | Coverage of the 95% intervals in 1,000 resamples of 86 units per configuration (takes about 12 minutes) | `python3 validation/coverage_sim.py . validation/coverage_sim.json 1000 20261023` | `coverage_sim.json` |
| `coverage_kappa.py` | The same coverage check for Cohen's, weighted and Fleiss' kappa, with both standard errors | `python3 validation/coverage_kappa.py . validation/coverage_kappa.json 1000` | `coverage_kappa.json` |
| `missing_demo.py` | Deletes 10% of the ratings at random (seed 7) and compares Krippendorff's alpha with the krippendorff package and with listwise deletion | `python3 validation/missing_demo.py . validation/missing_demo.json` | `missing_demo.json` |
| `skewed_prevalence_mc.py` | Coverage of the two-rater intervals under skewed category prevalence and small samples | `python3 validation/skewed_prevalence_mc.py . validation/skewed_prevalence_mc.json 2000 20261022` | `skewed_prevalence_mc.json` |
| `timing.py` | Median computation time of one analysis with 1,000 and 10,000 bootstrap resamples | `python3 validation/timing.py . validation/timing.json` | `timing.json` |

## Requirements

The application's own requirements (`requirements.txt`) plus the reference
packages:

```
pip install -r tests/requirements-test.txt     # krippendorff, statsmodels
pip install --no-deps irrCAC==0.4.4            # used only by validate_independent.py
```

irrCAC 0.4.4 pins an old SciPy version; installing it with `--no-deps` keeps
the current SciPy and works for the functions used here.

The automated test suite is separate: `python3 tests/test_irar_formulas.py`.

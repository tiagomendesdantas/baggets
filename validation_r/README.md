# R validation pipeline

Everything here exists to prove the Python package faithfully reproduces the
original 2018 R implementation. All R interaction is manual/subprocess-based —
pytest never requires R; it reads the committed fixtures these scripts write.

Requires R 4.3+ with the `forecast`, `cluster`, and `Mcomp` packages
(`TSclust` deliberately NOT needed — its `diss(X, "EUCL")` is plain Euclidean
distance and is shimmed in 3 lines where the original code is sourced).

## Scripts, in dependency order

| script | writes | purpose |
|---|---|---|
| `export_m3_mcomp.R` | `data/m3/*.csv` | canonical M3 data (the paper's own source; the datasetsforecast URL is dead) |
| `00_write_input_series.py` | `tests/fixtures/input/` | 12 fixture series + ETS-reference ids + PAM distance matrix, shared by both sides |
| `01_export_component_fixtures.R` | `tests/fixtures/r_components/`, `r_mbb/` | Tier 1: Guerrero λ, Box-Cox, STL/loess components, MBB draw distributions |
| `01b_pam_reference.R` | `tests/fixtures/r_pam/` | Tier 3: `cluster::pam` partitions + silhouette widths on a fixed distance matrix |
| `02_ets_reference.R` | `tests/fixtures/r_ets/` | Tier 2: `forecast::ets` model picks + forecasts on 30 M3 monthly series |
| `03_run_reference_pipeline.R` | `tests/fixtures/r_e2e/` | Tier 4: the ORIGINAL `baggedClusterETS.R` run end-to-end (seed 42, B=1000) |
| `original/` | — | the 2018 implementation, verbatim from GitHub |

Tier gates and frozen tolerances: `tests/fixtures/TOLERANCES.md`.
Tier-4 comparison: `experiments/tier4_compare.py` (10 Python seeds vs the R run).

Regenerate everything with `make fixtures` (see Makefile at the repo root).
`03` takes ~15-25 minutes; the rest are seconds to ~2 minutes.

## Original code quirks (why some fixtures avoid certain series)

The original `baggedClusterETS.R` crashes on any series with
`length(y) - h_pseudo <= h_pseudo` (undefined variables in its short-series
branch), so Tier-4 uses only fixture series that survive it. The Python
implementation instead shrinks the validation window and warns (see
`baggets.validation.default_h_val`).

# M6 research findings — ensemble selection under a convergent engine

**Date**: 2026-08-28 · 1,428 M3 monthly series, B=1000 cache (run 8d1f4360f7), 3 eval seeds,
statsforecast AutoETS with the convergence patch. Companion to `validation_report.md`.

## Question

The 2018 paper's premise: bagged ETS ensembles are hurt by correlated members, so select a
decorrelated subset (via PAM clusters). Does *any* selection — cluster-based or the
covariance-direct strategies built here — actually beat not selecting, and where?

## Finding 1 — selection only matters at small ensemble sizes, and what matters is accuracy, not diversity

Mean sMAPE across ensemble sizes (median aggregation unless noted):

| strategy | n=10 | n=25 | n=50 | n=100 |
|---|---|---|---|---|
| none (first-n, mean agg) | **13.782** | **13.736** | 13.831 | 13.846 |
| greedy (covariance-direct) | 13.862 | 13.806 | **13.810** | 13.827 |
| topk (accuracy only) | 14.030 | 13.920 | 13.850 | 13.849 |
| cluster-k5 (paper) | 13.955 | 13.866 | 13.833 | 13.848 |
| portfolio | 13.994 | 13.923 | 13.868 | 13.833 |
| random | 13.936 | 13.850 | 13.825 | 13.797 |

At n=10 the differences are real (Friedman p=1e-5) and the *ranking contradicts the
diversity narrative*: accuracy-driven selection (greedy, topk by rank) and the plain
first-10 mean lead, while **cluster selection is significantly worse** (Hochberg p=0.022)
— its proportional allocation forces representation of bad clusters, which costs accuracy
exactly when the ensemble is small. By n=100 all selection variants are statistically
indistinguishable (p=0.72, see validation_report.md).

## Finding 2 — trimmed-mean aggregation dominates the whole design space

The mean aggregator is efficient but fragile (one runaway member on N2752 explodes its
MASE); the median is robust but wasteful. A 10%-trimmed mean gets both:

| configuration | mean sMAPE | mean MASE | relative cost* |
|---|---|---|---|
| **none100:trimmed** | **13.741** | **0.837** | 1.0 |
| none25:trimmed | 13.754 | 0.838 | 0.25 |
| greedy:trimmed (n=25) | 13.797 | 0.840 | ~11 |
| cluster-k5:trimmed (n=100) | 13.791 | 0.840 | ~11 |
| paper-auto (2018 configuration, reproduced) | 13.847 | 0.844 | ~20 |
| none100:mean (Bergmeir protocol) | 13.846 | 3.296 (N2752) | 1.0 |
| plain ETS | 14.253 | 0.860 | 0.01 |

*cost ≈ number of ETS fits: no-selection-at-n needs only n bootstrap fits; any
validation-based selection needs B=1000 validation fits first.

**A 25-member unselected trimmed bag (25 ETS fits) matches or beats every selection
strategy at any size — including the reproduced 2018 configuration (1,100 fits) — at
~2% of its compute.**

## Finding 3 — the 2018 edge does not survive engine convergence

Under the patched (properly converging) engine, the paper's −0.036pp advantage of cluster
selection over plain bagging compresses to ~0.000pp — while per-series engine effects are
±1–4pp. Combined with the paper's own non-significant Table 6 comparison, the parsimonious
conclusion is that member selection at n=100 was never doing measurable work on M3 monthly;
the robust and reproducible effects are (a) bagging itself (−0.4pp vs plain ETS) and
(b) robust-efficient aggregation (trimmed mean, −0.1pp more).

## Caveats / next validations

- Single dataset-frequency so far (M3 monthly). The pipeline supports quarterly/yearly
  (the original 2018 R code could not run them — its short-series branch crashes); running
  those is cheap and is the natural robustness check.
- 3 eval seeds; stage-A bootstrap seed fixed (seed replicates of stage A would add rigor
  for a paper).
- Findings are conditional on the convergent engine; part of the 2018 result likely
  reflected R `ets` optimizer noise (see TOLERANCES.md — R itself loses ~3pp to the
  converged optimizer on some series).

## Reproduce

```
uv run python experiments/run_m3.py evaluate --group monthly --n-select 25 \
  --strategies none25:trimmed,none25:mean,greedy:trimmed,greedy,topk,cluster-k5 \
  --eval-seeds 1,2,3 --run-id sweep25 --n-jobs 6
uv run python experiments/run_m3.py stats --run-id sweep25
```

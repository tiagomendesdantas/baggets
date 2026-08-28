# M5 validation report — full M3 monthly reproduction

**Date**: 2026-08-28 · **Data**: 1,428 M3 monthly series (Mcomp export) · **Config**: B=1000
bootstraps, n_select=100, validation MAPE, engine = statsforecast AutoETS 2.1.1 **with the
convergence patch** (tol=1e-8, maxit=4000; see `tests/fixtures/TOLERANCES.md`) · Stage-A root
seed 42, eval seed 1.

## Gate results

| Gate | Target (paper Table 5) | Result | Verdict |
|---|---|---|---|
| Plain ETS anchor, Mean sMAPE | 14.135 ± 0.15 | **14.253** (Δ +0.118) | ✅ PASS |
| Plain ETS, Median sMAPE / Mean MASE | 9.073 / 0.865 | 9.059 / 0.860 | ✅ (nearly exact) |
| Bagged.BLD.MBB.ETS protocol (`none100:mean`), Mean sMAPE | 13.653 ± 0.20 | **13.846** (Δ +0.193) | ✅ PASS |
| Proposed approach (`paper-auto`), Mean sMAPE | 13.617 ± 0.20 | **13.847** (Δ +0.230) | ❌ MISS by 0.03 |
| Proposed < Bagged.BLD.MBB ordering | 13.617 < 13.653 (Δ −0.036) | 13.847 ≈ 13.846 (Δ +0.001) | ❌ edge not reproduced |

Aggregation per row follows each method's own protocol: the proposed approach uses the
**median** of the member forecasts (paper Algorithm 2, line 20; also what Tier-4 compared
against in the original R output), while the Bagged.BLD.MBB.ETS row uses the **mean**, as
that method publishes. Every un-suffixed strategy in the tables below is median-aggregated;
`:mean` / `:trimmed` suffixes mark the exceptions.

## Full strategy table (Mean sMAPE / mean rank, 1,428 series)

| strategy | mean sMAPE | rank | note |
|---|---|---|---|
| random | 13.821 | 5.68 | best mean, significantly worst rank of the bagging set |
| greedy | 13.827 | 5.43 | covariance-direct (new) |
| portfolio | 13.833 | 5.38 | covariance-direct (new) |
| paper-k5 | 13.842 | 5.40 | 2018 constants, k=5 |
| none100:mean | 13.846 | 5.32 | Bergmeir protocol; best rank |
| paper-auto | 13.847 | 5.47 | **the 2018 headline configuration** |
| topk | 13.849 | 5.39 | accuracy-without-diversity ablation |
| cluster-k5 | 13.851 | 5.36 | |
| cluster-auto | 13.857 | 5.50 | |
| original (plain ETS) | 14.253 | 6.08 | |

Friedman p = 7.3e-12 overall; **all eight selection variants are statistically
indistinguishable** (Hochberg-adjusted p = 0.72 vs the best-ranked control); only `random`
(p = 0.015) and plain ETS (p ≈ 0) are significantly worse.

## Interpretation

1. **What reproduces robustly**: the bagging effect itself (−0.4pp Mean sMAPE, −0.017 MASE vs
   plain ETS, every variant), the plain-ETS aggregate anchor, and the Bergmeir-protocol
   aggregate — all inside their gates.
2. **What does not**: the paper's −0.036pp edge of cluster selection over plain bagging. Under
   the *converged* engine, every bagging strategy lands in a 0.04pp-wide band (13.82–13.86),
   and the paper's own Table 6 already found that edge not statistically significant. Our
   attribution chain (Tier-2/Tier-4, `TOLERANCES.md`) shows per-series engine effects of
   ±1–4pp — an order of magnitude larger than the strategy differences at n=100. The most
   parsimonious reading: **at ensemble size 100 on M3 monthly, which members you select is
   immaterial; that the published 0.036 edge does not survive engine convergence suggests it
   sat within optimizer noise.** (Both engines under-converge in places: R itself loses ~3pp
   to the patched Python on N2501.)
3. **The +0.19–0.23 aggregate offset vs 2018** is engine-attributable, moves *all* strategies
   together, and shrinks to +0.118 on the strategy-free anchor — consistent with the Tier-4
   finding that R behaves like "one more seed" only where the engines' optima coincide.
4. **Mean-aggregation fragility, quantified**: `none100:mean`'s Mean MASE explodes to 3.30
   (vs ~0.843 for all median strategies) because of a **single series** — N2752, where one
   of the first 100 members produces a runaway forecast (per-series MASE 3513 vs ~1.5 under
   any median strategy). One member in 142,800 destroys the mean; the median doesn't notice.
   A concrete, quotable argument for the paper's median choice.

## What this sets up (M6)

If selection is immaterial at n=100, its leverage — if any — is at small ensembles, where
picking *which* members matters most. The M6 experiment is the ensemble-size sweep
(n ∈ {10, 25, 50, 100}) across selection strategies from the same cache, with seed
replicates, plus the k-sweep (paper Fig. 4 shape) and aggregator comparisons.

## Reproduction commands

```
Rscript validation_r/export_m3_mcomp.R
uv run python experiments/run_m3.py precompute --group monthly --n-jobs 6   # ~2900 core-min
uv run python experiments/run_m3.py evaluate --group monthly \
  --strategies original,none100:mean,cluster-k5,cluster-auto,paper-k5,paper-auto,topk,random,greedy,portfolio \
  --run-id m5_full --n-jobs 6
uv run python experiments/run_m3.py stats --run-id m5_full
```

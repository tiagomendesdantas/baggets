# Frozen fixture tolerances (Tier 1 — components vs R)

Set empirically from the first fixture run (R 4.3.3, forecast 8.x, statsmodels
0.14.6, 2026-08-26). **Never widen a gate to make a failing test pass** — a
failure after this point means a regression on the Python side (or an
intentional, documented change regenerating the fixtures).

| Quantity | Gate | Observed on first run |
|---|---|---|
| Guerrero lambda | abs diff ≤ 1e-3 | ≤ ~7e-5 (R optimize tol near bounds) |
| BoxCox transform (given same lambda) | rtol/atol 1e-9 | exact (same formula) |
| STL trend/seasonal (m > 1) | max abs diff ≤ 1e-9 · sd(x_bc) | ≤ 9e-13 · sd — machine precision |
| STL remainder sd ratio | abs(ratio − 1) ≤ 1e-6 | ≤ 1e-7 |
| loess trend (m = 1) | max abs diff ≤ 1e-3 · sd; corr ≥ 0.999 | 1.45e-4 · sd |
| loess remainder sd ratio | abs(ratio − 1) ≤ 1e-3 | 6.9e-5 |
| MBB per-draw sd / pooled values | KS p > 0.01 (same source remainder) | p ≫ 0.01 |

# Tier 2 — ETS engine (statsforecast AutoETS vs R forecast::ets)

Same 30-model space, different optimizer/tie-breaks → statistical gates on the
30-series monthly reference set (`tests/fixtures/r_ets/reference.csv`).

**Amended 2026-08-26 (optimizer patch).** statsforecast ≤2.1.1 runs its
Nelder-Mead with hardcoded tol=1e-4, maxit=1000; on smooth (bootstrapped)
series it stops far from the optimum — up to ~70 AICc points worse than R —
and flips model selection away from trended families (discovered via Tier-4:
N2501 members' median validation MAPE 12.6 vs R's 3.3 on identical inputs).
`baggets.engine` patches the call to tol=1e-8, maxit=4000 (maxit is the
binding constraint; tighter buys nothing), restoring R-family picks at ~3x fit
cost. Under the patch, both engines converge properly and each finds its own
near-equal optimum, so the *median* inter-engine distance rises slightly while
the catastrophic tail collapses — gates re-frozen accordingly:

| Quantity | Gate | Loose optimizer (first run) | Patched optimizer (frozen) |
|---|---|---|---|
| Model-form agreement | ≥ 60% | 80% (24/30) | 70% (21/30) |
| Scaled forecast distance, median | ≤ 0.10 (was 0.05) | 0.043 | 0.062 |
| Scaled forecast distance, max | ≤ 1.5 (new tail gate) | 2.33 (would FAIL) | 0.87 |
| Aggregate: plain AutoETS, 1428 M3 monthly, Mean sMAPE | 14.135 ± 0.15 | 14.158 (Δ +0.023) | **14.253** (Δ +0.118) |
| Aggregate: Median sMAPE / Mean MASE / Median MASE | report | 9.164 / 0.863 / 0.711 | **9.059 / 0.860 / 0.713** (3 of 4 closer to paper) |

Why STL is exact: statsmodels STL and R's `stats::stl` descend from the same
netlib Fortran code. Reproducing R requires three non-default statsmodels
settings (see `src/baggets/decompose.py`): R's loess jump parameters
(`ceil(window/10)`), `inner_iter=2, outer_iter=0`, and R's `nextodd` window
formulas. The periodic post-averaging of the seasonal is replicated explicitly.

Why loess (non-seasonal) is not exact: R `loess` evaluates on a kd-tree and
interpolates (surface = "interpolate", cell = 0.2); statsmodels `lowess`
evaluates every point exactly. Same tricube local-linear regression, different
evaluation strategy. The bootstrap only consumes `trend + remainder`, which
both sides agree on to ~1e-4 of series sd.

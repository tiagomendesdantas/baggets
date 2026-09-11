# M7 — Exploration: what is bagging actually doing?

M5/M6 established that member *selection* does nothing measurable on M3 monthly. That left one
live, unexplained effect: **bagging itself beats plain ETS by ~0.4 pp, and nobody knows why.**

M7 tests a specific mechanism. All 1000 members share one Box-Cox λ and one STL `trend + seasonal`
bit-for-bit ([bootstrap.py:74-81](../src/baggets/bootstrap.py#L74-L81)); only the remainder is
resampled. That is a weak perturbation to be worth 0.4 pp — unless it is amplified through
something discrete. The discrete thing is `AutoETS`'s AICc argmax over the 15 admissible ETS
families, and the M3 optimizer finding (a tolerance change flipping families and moving per-series
accuracy by 1–4 pp) shows how violently the pipeline responds to it.

> **Hypothesis.** Bagged ETS is a 1000-fit Monte Carlo approximation to model-form averaging.

Verdict: **the substitution claim holds; the mechanism claim does not yet.** Details below.

---

## Finding 1 — A 15-fit family average is statistically indistinguishable from a 1000-member bag

`experiments/model_averaging.py families` fits all 15 admissible families to each *original*
series (1428 M3 monthly, 71 s at 6 workers), ranks them by in-sample AICc, and averages the top *k*.
No test data touches the ranking.

Mean sMAPE over the full k-sweep:

| k | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 12 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|
| mean sMAPE | 14.407 | 13.990 | 13.766 | 13.671 | 13.672 | 13.704 | 13.697 | **13.656** | 13.821 | 14.411 |

A clear interior optimum with a broad flat basin at k ≈ 4–11. Against the reference arms
(`none100:trimmed` 13.741, `none25:trimmed` 13.754, `paper-auto` 13.847, plain ETS 14.253):

- **Friedman + Hochberg** over 9 arms, 1428 series: Friedman p = 5.1e-19, but every combination arm
  is mutually indistinguishable (Hochberg p = 0.44 vs the best-ranked control). Only the
  single-model arms (`original`, `ets_best`) are significantly worse.
- **Paired Wilcoxon** vs `none100:trimmed`, per series, for every fixed k in 3..12: p = 0.37–0.83,
  win rates 48.5–51.1%, mean differences −0.085 to +0.080 pp.
- **k chosen out-of-sample** (split-half: pick k on one half of the series, score on the other):
  paired mean difference **+0.035 pp, Wilcoxon p = 0.249**.

**Conclusion: equivalence, not superiority.** The first-pass mean-sMAPE advantage (13.672 vs
13.741) sits inside the noise. The defensible claim is that ~15 ETS fits reproduce what a
1000-member bagged ensemble achieves — **~1.5% of the compute for statistically identical
accuracy**, and the same shape as M6's trimmed-25 result one level further down.

### Where the effect lives
`top5_mean` beats `none100:trimmed` on only 48.7% of series (hence the rank tie) but by
**−1.91 pp on the worst decile** and **+0.14 pp on the other 90%**. Model-form averaging pays off
exactly where forecasting is hard. Per-series gains correlate across the two methods:
corr(τ_bag, τ_ma) = **0.41 Spearman / 0.54 Pearson** — substantial, far from 1.

---

## Finding 2 — The mechanism test is inconclusive, and the pre-registered prediction was untestable

`experiments/model_averaging.py entropy` refit 100 members on each of 300 systematically sampled
series (30k fits, 24 min) recording the selected ETS family.

**Family choice is wildly unstable across bootstrap members** — the mechanism certainly exists:

- median **5 distinct families** per series (mean 5.47, max 13)
- modal-family share: median 0.55
- series where all 100 members agree: **1 of 300 (0.3%)**
- member 0's family is the modal family in only 55% of series

The pre-registered prediction was *τ ≈ 0 where all members select the same family*. It **could not
be tested**: the "stable" cell contains one series. Reported as a failed test design, not a result.

The continuous analogue is **weak and not significant**:

| predictor | vs τ_bag | vs τ_ma |
|---|---|---|
| family entropy | ρ = +0.055 (p = 0.34) | ρ = +0.092 (p = 0.11) |
| n distinct families | ρ = +0.113 (p = 0.051) | ρ = +0.084 (p = 0.15) |
| modal share | ρ = −0.022 (p = 0.71) | ρ = −0.056 (p = 0.33) |

Entropy quartiles show a suggestive but non-monotone trend (τ_bag: Q1 0.137, Q2 −0.286, Q3 0.497,
Q4 0.730). Direction is right; magnitude is not established at n = 300. Settling it needs the full
1428 series (~2 h).

**So: bagging can be *substituted* by model-form averaging (Finding 1, solid), but the per-series
evidence that model-form instability is *the mechanism* is not there yet.** Those are two distinct
claims and only the first is supported.

## Finding 3 — Likelihood ambiguity does not predict where bagging helps (negative)

Δ AICc₁₂ on the original series, Akaike-weight entropy, and effective model count all fail as
pre-forecast routing proxies: ρ = −0.026 (p = 0.32) against τ_bag; quartiles non-monotone. There is
no cheap "only bag the ambiguous series" rule from single-sample likelihood ambiguity. Consistent
with the mechanism being *selection instability under resampling* rather than likelihood
uncertainty — but see Finding 2 for why that remains unproven.

Note `aicc_weighted` (13.898) is much worse than `top5_mean`: Akaike weights are near-degenerate,
so weighting collapses toward plain ETS. **Equal weights over the top few beat likelihood weights.**

---

## Finding 4 — Engine finding #2: `AutoETS("ZZZ")` under-converges vs explicit per-family fits

Distinct from the M3 Nelder-Mead tolerance defect (already patched and active in all of the above).
Fitting the 15 families explicitly and taking argmin AICc reaches a **strictly better optimum than
AutoETS's own automatic search** on a systematic sample of 300 M3 monthly series:

- explicit fit of ZZZ's *own chosen form* is better on **4.0%** of series
- ZZZ picks a different family than the grid argmin on **6.7%**
- gap distribution: median 0.0, 99th pct 4.5, **max 47.5 AICc points** (N2649)
- concentrated in multiplicative-trend families `ETS(M,A,N)` / `ETS(M,Ad,N)`; on N1405 ZZZ pins
  β at its lower bound (AICc 932.11 vs 930.45 explicit)

**The twist:** the better-converged single model forecasts *worse* — `ets_best` 14.407 vs
`original` 14.253 mean sMAPE. Tighter convergence is not automatically better forecasting. This
sharpens the project's existing optimizer narrative rather than repeating it: the benchmark is
sensitive to numerical detail in both directions.

---

## Finding 5 — The member-0 confound is real, small, and M6's conclusion survives

[bootstrap.py:70](../src/baggets/bootstrap.py#L70) sets `series[0] = x` and
[selection.py:339](../src/baggets/selection.py#L339) returns `arange(n)`, so every `noneN` arm
contains the plain-ETS forecast of the *un-bootstrapped* series at 1/n weight.
`experiments/run_diagnostics.py ablate` re-scores against member-0-free random subsets
(20 permutations, all 1428 series):

| n | aggregator | first-n (as deployed) | random-n, member-0 free | difference |
|---|---|---|---|---|
| 10 | mean | 13.782 | 13.888 | −0.105 |
| 25 | trimmed | **13.754** | **13.795** | −0.041 |
| 25 | mean | 13.736 | 13.805 | −0.069 |
| 100 | trimmed | 13.741 | 13.754 | −0.013 |

The contamination is real and scales as ~1/n exactly as predicted. **M6's headline number should be
13.795, not 13.754** — still comfortably better than `paper-auto` (13.847), so Finding 2 of M6
stands with a corrected margin.

---

## What this changes

1. The honest headline is now one level below M6's: not "a 25-member trimmed bag suffices" but
   **"~15 ETS fits suffice"** — and the bag's remaining advantage over that is unmeasurable.
2. Two engine findings now exist, not one, and they point in opposite directions on accuracy.
3. The mechanism question is open. Finding 1 is consistent with model-form averaging but does not
   establish it; Findings 2 and 3 both failed to confirm it per-series.

## Open

- [ ] Extend the family-entropy test from 300 to all 1428 series (~2 h) — the only way to settle
      Finding 2 either way; at ρ ≈ 0.11 the n = 300 pilot is underpowered.
- [ ] Tier 1 threads T2–T8 (probabilistic calibration, conditional/heterogeneous, aggregation law)
      remain unrun; all are cache-only.
- [ ] Persist `FittedETS.method` / `sigma2` in stage A so this never needs a refit again.

## Reproduce

```
uv run python experiments/model_averaging.py families --n-jobs 6 --run-id m7_aicc      # 71 s
uv run python experiments/model_averaging.py entropy  --n-series 300 --n-members 100   # 24 min
uv run python experiments/run_diagnostics.py ablate --n-perm 20 --run-id m7_ablation   # 52 s
uv run python experiments/run_m3.py stats --run-id m7_tier0 --metric smape
```

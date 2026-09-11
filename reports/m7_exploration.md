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

Verdict: **Finding 1 carries the result.** ~15 ETS fits reproduce the bag. The per-series gain is
also associated with how much members disagree about model form (Finding 2, ρ = +0.0995,
p = 1.65e-04 at n = 1428), but that association explains only ~1% of per-series variance, points
in a direction that is close to mechanical, and was never separated from the simpler explanation
that bagging helps whenever member forecasts are spread apart. Read Finding 1 as the claim and
Finding 2 as supporting description.

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

## Finding 2 — Bagging's gain is associated with how much members disagree about model form (n = 1428)

**This variation is by design, not a defect.** BLD.MBB exists to generate genuinely different
series, and `AutoETS` re-selects a model on each one, so members choosing different ETS families is
the mechanism operating as intended. Note the vocabulary deliberately: elsewhere in this project
"instability" denotes a *numerical defect* — the optimizer flipping families through
under-convergence (Finding 4). Data-driven variation and optimizer-driven flipping are different
phenomena and should not share a word.

`experiments/model_averaging.py entropy` refit 100 **bootstrap** members on each of all 1428
series (142,800 fits, 125 min), recording the selected ETS family. Member 0 is excluded from these
statistics — it is the original series, not a bootstrap draw
([bootstrap.py:70](../src/baggets/bootstrap.py#L70)) — and recorded separately.

The analysis was **pre-registered before the data existed**
(`experiments/analyze_family_instability.py`, committed ahead of the run): primary test
Spearman(`n_distinct_families`, τ_bag), two-sided, α = 0.05. τ_bag is measured against the
benchmark bagging is always judged against — **plain ETS on the original series, with no bagging**
(14.253) — minus the bag's score.

**Members routinely disagree**: median **5 distinct families** per series across 100 bootstrap
members (mean 5.37, max 14); only **24 of 1428** series (1.7%) have all 100 members agreeing —
which is what a resampling scheme that perturbs the data is supposed to produce.

**Primary test: ρ = +0.0995, p = 1.65e-04.**

*A declared post-hoc correction.* The pre-registration also excluded the original series from the
**bag** side, which was wrong about the method — it is a designed member (Finding 5). Repeating the
identical test against bagging as specified (`none100:trimmed`) gives **ρ = +0.0985,
p = 1.94e-04**: the two definitions agree, so nothing here turns on the error. The pre-registered
figure remains primary and the corrected one is reported beside it, rather than substituted —
silently swapping a definition after seeing the result is the post-hoc flexibility this project
criticises in the 2018 paper.

Every secondary test agrees, all with the predicted sign:

| predictor | vs τ_bag | vs τ_ma |
|---|---|---|
| n distinct families | ρ = +0.0995 (p = 1.7e-4) | ρ = +0.0925 (p = 4.6e-4) |
| family entropy | ρ = +0.0965 (p = 2.6e-4) | ρ = +0.1007 (p = 1.4e-4) |
| modal share | ρ = −0.0793 (p = 2.7e-3) | ρ = −0.0913 (p = 5.5e-4) |

The quartile view is the legible one, and it is monotone where the pilot's was not:

| quartile | mean families | mean τ_bag | mean τ_ma |
|---|---|---|---|
| Q1 most agreement | 2.6 | **−0.100** | +0.006 |
| Q2 | 4.5 | +0.483 | +0.426 |
| Q3 | 6.0 | +0.431 | +0.699 |
| Q4 least agreement | 8.4 | **+1.184** | +1.193 |

**Bagging's entire average gain is concentrated in the half where members disagree.** On the
most-agreeing quartile (≤3 families, n = 347) it is **−0.104 pp, 95% CI [−0.390, +0.129]** —
bagging does not pay there at all. The M7 pilot's original prediction (τ ≈ 0 where all members
agree) was untestable at n = 1; at n = 24 it holds directionally — mean τ_bag −0.026 pp,
Wilcoxon p = 0.944, indistinguishable from zero — though that subgroup is far too small to
separate from the rest (Mann-Whitney p = 0.40).
The ≤3-family quartile is the better-powered form of the same statement.

**Three caveats, stated plainly, because they bound this finding hard.**

*It is weak.* ρ ≈ 0.10 explains ~1% of the variance in τ_bag. This is an association, not a
forecast of which series will benefit.

*Its direction is close to mechanical.* If every member selects the same family on near-identical
data, their forecasts are near-identical, so averaging them changes little and τ_bag ≈ 0 almost by
construction. The sign was never in doubt; only the magnitude (−0.100 pp vs +1.184 pp across
quartiles) carries information.

*It never ran the check that would make it a mechanism.* Bagging can only help when the member
forecasts differ — if all 100 are nearly identical, their average is nearly plain ETS. Members that
pick different families naturally produce forecasts that differ more, so family count and forecast
spread move together. This analysis measured only the family count, never the spread, so it cannot
tell whether the choice of model form is a mechanism in its own right or just a visible symptom of
forecasts being spread apart. **Read Finding 2 as an association, not as a demonstrated cause.**

Measuring the disagreement also costs 100 ETS fits per series — the very cost bagging was being
questioned for — so nothing here is an operational saving either.

## Finding 3 — Likelihood ambiguity does not predict where bagging helps (negative)

Δ AICc₁₂ on the original series, Akaike-weight entropy, and effective model count all fail as
pre-forecast routing proxies: ρ = −0.026 (p = 0.32) against τ_bag; quartiles non-monotone. There is
no cheap "only bag the ambiguous series" rule from single-sample likelihood ambiguity. Consistent
with what matters being the variation the *resampling* induces rather than ambiguity visible in a
single sample — but see Finding 2's caveats for why even that remains undemonstrated.

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

## Finding 5 — How much the original series contributes as an ensemble member

**The original series is a designed member of the pool, not a leak.**
[bootstrap.py:70](../src/baggets/bootstrap.py#L70) sets `series[0] = x` because the 2018 R source
does (`xs[[1]] <- x`), inherited from Bergmeir's `bld.mbb.bootstrap`, and every selection rule
there draws from the full pool — `bootstrapped_series[selecClus]` for the cluster arm,
`bootstrapped_series[sample(1:1000,100)]` for the random one. Including it is the method.

Separately, the *forecast* of that member is the benchmark bagging is judged against: plain ETS on
the original series, no bagging, 14.253 mean sMAPE. Those are two different roles for the same
object and should not be conflated.

What is worth measuring is how much that member contributes to the bag's accuracy.
`experiments/run_diagnostics.py ablate` re-scores each configuration against pools drawn from the
bootstrap replicates only — members 1..999, original excluded — over 20 permutations, all 1428
series:

| n | aggregator | as specified (first-n, incl. original) | replicates only (original excluded) | the original member's contribution |
|---|---|---|---|---|
| 10 | mean | 13.782 | 13.888 | −0.105 |
| 25 | trimmed | **13.754** | **13.795** | −0.041 |
| 25 | mean | 13.736 | 13.805 | −0.069 |
| 100 | trimmed | 13.741 | 13.754 | −0.013 |

Read the right-hand column as *"what bagging would score if the original series were removed from
the pool"* — a sensitivity variant, not a corrected baseline. **M6's headline 13.754 stands**: it
is the method as specified. The original member is worth ~0.04 pp at n = 25 and ~0.01 pp at
n = 100, shrinking with its ensemble weight exactly as a 1/n contribution should.

One genuine asymmetry does follow, and it is between *selection rules* rather than in the pool:
`NoSelection(n)` returns `arange(n)`, so the original always gets a slot, while `RandomSelection`
includes it only with probability n/B. At n = 10 that is worth ~0.1 pp, which matters when ranking
rules against each other at small n. Note `NoSelection` is the one behaving like the method here —
cluster, topk and greedy also draw from the full pool; `random` is this project's own ablation.

---

## What this changes

1. The honest headline is now one level below M6's: not "a 25-member trimmed bag suffices" but
   **"~15 ETS fits suffice"** — and the bag's remaining advantage over that is unmeasurable.
2. Two engine findings now exist, not one, and they point in opposite directions on accuracy.
3. **The substitution result is the one to carry forward.** A direct top-*k* family average stands
   in for the whole bagging apparatus (Finding 1) — that is measured, robust across k, and
   cross-fitted. The supporting story, that bagging is averaging over model-form uncertainty, is
   consistent with Finding 2 but not demonstrated by it: the association is weak (~1% of variance),
   its direction is close to mechanical, and it was never separated from the simpler account that
   averaging helps whenever member forecasts are spread apart. Treat it as a plausible reading,
   not a result.

## Open

- [ ] Tier 1 threads T2–T8 (probabilistic calibration, conditional/heterogeneous, aggregation law)
      remain unrun; all are cache-only.
- [ ] Persist `FittedETS.method` / `sigma2` in stage A so this never needs a refit again.

## Reproduce

```
uv run python experiments/model_averaging.py families --n-jobs 6 --run-id m7_aicc      # 71 s
uv run python experiments/model_averaging.py entropy --n-series 1428 --n-members 100 \
    --cache-key 8d1f4360f7 --run-id m7_entropy_full                                    # 125 min
uv run python experiments/analyze_family_instability.py                                # the test
uv run python experiments/run_diagnostics.py ablate --n-perm 20 --run-id m7_ablation   # 52 s
uv run python experiments/run_m3.py stats --run-id m7_tier0 --metric smape
```

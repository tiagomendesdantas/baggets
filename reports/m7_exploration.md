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

Verdict: **both halves hold.** ~15 ETS fits reproduce the bag (Finding 1), and the per-series gain
tracks how unstable the ETS family choice is across bootstrap members (Finding 2, ρ = +0.0995,
p = 1.65e-04 at n = 1428) — though the mechanism explains only ~1% of per-series variance.

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

## Finding 2 — Bagging's gain tracks ETS family instability (resolved, n = 1428)

`experiments/model_averaging.py entropy` refit 100 **bootstrap** members on each of all 1428
series (142,800 fits, 125 min), recording the selected ETS family. Member 0 is excluded from the
instability statistics — it is the original series, not a bootstrap draw
([bootstrap.py:70](../src/baggets/bootstrap.py#L70)) — and recorded separately.

The analysis was **pre-registered before the data existed**
(`experiments/analyze_family_instability.py`, committed ahead of the run): primary test
Spearman(`n_distinct_families`, τ_bag), two-sided, α = 0.05, with τ member-0-free on both sides
(plain ETS minus the ablation's member-0-free random-100 trimmed bag, never the contaminated
`none100:trimmed`).

**Family choice is wildly unstable**: median **5 distinct families** per series across 100
bootstrap members (mean 5.37, max 14); only **24 of 1428** series (1.7%) have all 100 members
agreeing.

**Primary test: ρ = +0.0995, p = 1.65e-04.** Every secondary test agrees, all with the predicted
sign:

| predictor | vs τ_bag | vs τ_ma |
|---|---|---|
| n distinct families | ρ = +0.0995 (p = 1.7e-4) | ρ = +0.0925 (p = 4.6e-4) |
| family entropy | ρ = +0.0965 (p = 2.6e-4) | ρ = +0.1007 (p = 1.4e-4) |
| modal share | ρ = −0.0793 (p = 2.7e-3) | ρ = −0.0913 (p = 5.5e-4) |

The quartile view is the legible one, and it is monotone where the pilot's was not:

| quartile | mean families | mean τ_bag | mean τ_ma |
|---|---|---|---|
| Q1 stable | 2.6 | **−0.100** | +0.006 |
| Q2 | 4.5 | +0.483 | +0.426 |
| Q3 | 6.0 | +0.431 | +0.699 |
| Q4 unstable | 8.4 | **+1.184** | +1.193 |

**Bagging's entire average gain is concentrated in the unstable half.** On the most-stable quartile
(≤3 families, n = 347) it is **−0.104 pp, 95% CI [−0.390, +0.129]** — bagging does not pay there at
all. The M7 pilot's original prediction (τ ≈ 0 where all members agree) was untestable at n = 1;
at n = 24 it holds directionally — mean τ_bag −0.026 pp, Wilcoxon p = 0.944, indistinguishable
from zero — though that subgroup is far too small to separate from the rest (Mann-Whitney p = 0.40).
The ≤3-family quartile is the better-powered form of the same statement.

**Two caveats, stated plainly.** First, the effect is *real but weak as a per-series predictor*:
ρ ≈ 0.10 explains ~1% of the variance in τ_bag. The mechanism is established; it is not a
forecast of which series will benefit. Second, measuring family instability costs 100 ETS fits per
series — the very cost bagging was being questioned for — so this is a **mechanistic explanation,
not an operational saving**. Whether a cheap pre-forecast proxy for instability exists is a
separate question, and one that should be pre-registered rather than mined from this data (see
Finding 3, where single-sample likelihood ambiguity already failed at that job).

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
3. **What bagging is doing in this pipeline is averaging over model-form uncertainty.** The two
   halves now agree: a direct family average substitutes for the bag (Finding 1), and the bag's
   gain appears exactly where the family choice is unstable and vanishes where it is not
   (Finding 2). The remaining honest limits are that the mechanism is a weak per-series predictor
   (~1% of variance) and that measuring it is not cheaper than the bag it explains.

## Open

- [ ] **A cheap way to predict family instability, decided in advance.** Finding 2 says bagging
      only pays on series whose ETS family choice is unstable, which sounds like a rule: skip
      bagging on the stable ~24% and lose nothing. It is not usable as it stands, because
      measuring instability takes 100 ETS fits per series — the same cost as the 100-member bag
      it would let you skip. You pay bagging's full price to learn you didn't need to bag.
      What would make it a real rule is some quantity computable from the original series at
      ~0–1 fits that predicts instability. Two warnings for whoever tries. The obvious candidate
      has already failed: single-sample likelihood ambiguity (ΔAICc₁₂, Akaike-weight entropy) gave
      ρ = −0.026, p = 0.32 in Finding 3. And with a true effect near ρ ≈ 0.10, testing ~10
      candidate features at α = 0.05 turns up a spurious winner about once by chance — the exact
      mistake this project is documenting in the 2018 paper. So fix the candidate list and the
      multiple-testing correction in writing *before* looking. Cheapest place to start, at **zero
      new fits**: the per-member family labels for all 1428 series are in
      `results/m7_entropy_full/series/*.json`, so "do the first 5/10/20 members rank series the
      same way as all 100?" is answerable from disk today.
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

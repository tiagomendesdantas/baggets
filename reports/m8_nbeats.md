# M8 — A neural base learner inside the bagging pipeline

**Date**: 2026-09-09 · M3 monthly · B=100 · plain bagging (`NoSelection` + mean)
**Status**: final for the interpretable arm, n=300 matched series. The generic arm is still running.

> **Note on a reading that changed.** An earlier pass over the first 55 series showed the
> two engines tied (p = 0.68). At n = 300 the ordering reversed and the difference became
> significant. The tie was a small-sample artifact, and it is recorded here rather than
> quietly overwritten — a result that moves when the sample grows is worth knowing about.

## The question

The pipeline's base learner was ETS because the paper's was. Swapping it for a
deep network — with the bootstrap, the selection strategies, the metrics and the
harness held fixed — asks one thing: **does a modern neural forecaster earn its
place as a bagging member on M3-scale series?**

N-BEATS (Oreshkin et al., ICLR 2020) is the right candidate rather than an LSTM
or a transformer. It is designed for exactly this shape of problem — univariate,
fixed lookback, point forecast — and it reported M3 and M4 results, so the
comparison is one its authors invited.

## Result

| engine | mean sMAPE | median sMAPE | mean MASE | median MASE |
|---|---|---|---|---|
| **bagged ETS** | **24.59** | **21.70** | **0.723** | **0.675** |
| bagged N-BEATS (interpretable) | 25.28 | 23.18 | 0.775 | 0.703 |

Wilcoxon signed-rank over the 300 matched series:

| metric | W | p | mean difference (N-BEATS − ETS) |
|---|---|---|---|
| sMAPE | 19558 | **0.045** | +0.69 |
| MASE | 18767 | **0.011** | +0.053 |

N-BEATS wins on **43% of series (128/300)**.

**Exponential smoothing wins, and the margin is real.** It is also small — 0.69
sMAPE — so the honest statement is "consistently a little worse", not "much
worse". Consistency is what the test detects; magnitude is what the table shows,
and the two should not be reported as one thing.

Friedman is the paper's procedure and is used elsewhere in this repo, but it is
a multi-group test. Two paired arms are a Wilcoxon signed-rank problem: same
series, two measurements each. The generic N-BEATS arm now running makes three,
at which point Friedman + Hochberg applies again.

This is the direction Makridakis, Spiliotis and Assimakopoulos (2018) reported
when they put statistical and machine-learning methods head to head on
M-competition data. Reproducing a known result on one's own pipeline is a weaker
claim than a new one, and a more useful one than a surprise that does not hold.

## Why this is the expected answer, not a disappointment

The pipeline hands each member **one series**. An M3 monthly series is 48–144
points, so after a lookback window a member trains on **tens of windows** —
N1402 gives 8. N-BEATS's published results come from *cross-learning*: one model
over 100,000 series. Per series is a different regime, and the honest reading is
that a deep network cannot show what it is for on this much data. That it lands
within 0.7 sMAPE of automatic ETS on eight training windows is arguably the more
surprising half of the result.

Two things fall out of that and are worth stating separately from the score:

- **Bagging is doing real work, and more than the base learner choice does.**
  On the same 300 series, plain bagging beats a single ETS fit (24.59 vs 25.37
  mean sMAPE) while cluster selection adds nothing at B=100 — consistent with
  [M6](m6_findings.md). Bagging buys 0.79 sMAPE; swapping the base learner for a
  deep network costs 0.69. The ensemble is where the accuracy lives.
- **The neural arm is where the training pathologies are.** Instrumenting one
  member with [torch-training-probes](https://github.com/tiagomendesdantas/torch-training-probes)
  on N1402 found a layer with 60% dead ReLU units and validation loss bottoming
  at **epoch 56 of 400**. A model that overfits by epoch 56 on 8 windows is not
  going to be rescued by tuning.

## What would actually test the idea

Cross-learning: one N-BEATS over all series, bootstrap replicates as extra
training data rather than as separate models. That is a different pipeline, not
a different engine, and it is the experiment this result points at.

## Reproduce

```bash
uv run python experiments/run_m3.py precompute --group monthly --limit 300 \
    --n-bootstraps 100 --engine ets --n-jobs 6
uv run python experiments/run_m3.py precompute --group monthly --limit 300 \
    --n-bootstraps 100 --engine nbeats-i --epochs 150 --n-jobs 6
uv run python experiments/compare_engines.py --engines ets nbeats-i \
    --n-bootstraps 100 --strategy none100:mean
```

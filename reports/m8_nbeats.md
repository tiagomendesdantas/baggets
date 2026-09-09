# M8 — A neural base learner inside the bagging pipeline

**Date**: 2026-09-09 · M3 monthly · B=100 · plain bagging (`NoSelection` + mean)
**Status**: preliminary, n=55 matched series. The 300-series run is still going.

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
| bagged N-BEATS (interpretable) | 32.85 | 29.73 | 0.658 | 0.638 |
| bagged ETS | 33.82 | 30.01 | 0.656 | 0.613 |

Wilcoxon signed-rank over the 55 matched series:

| metric | W | p | mean difference |
|---|---|---|---|
| sMAPE | 721 | 0.681 | −0.98 |
| MASE | 727 | 0.719 | +0.002 |

N-BEATS is ahead on sMAPE, behind on MASE, and wins on **45% of series (25/55)**.

**There is no detectable difference.** Not "the neural method loses" — the
sample says the two are indistinguishable, and it would be as wrong to claim a
win from −0.98 sMAPE as to claim a loss from +0.002 MASE.

Friedman is the paper's procedure and is used elsewhere in this repo, but it is
a multi-group test. Two paired arms are a Wilcoxon signed-rank problem: same
series, two measurements each.

## Why this is the expected answer, not a disappointment

The pipeline hands each member **one series**. An M3 monthly series is 48–144
points, so after a lookback window a member trains on **tens of windows** —
N1402 gives 8. N-BEATS's published results come from *cross-learning*: one model
over 100,000 series. Per series is a different regime, and the honest reading is
that a deep network cannot show what it is for on this much data.

Two things fall out of that and are worth stating separately from the score:

- **Bagging is doing real work regardless of base learner.** On the full
  300-series ETS arm, plain bagging beats a single ETS fit (24.59 vs 25.37 mean
  sMAPE) while cluster selection adds nothing at B=100 — consistent with
  [M6](m6_findings.md).
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

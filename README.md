# baggets

Python reimplementation of **Bagged.Cluster.ETS** — the forecasting method from:

> Dantas, T.M. & Cyrino Oliveira, F.L. (2018). *Improving time series forecasting: An approach
> combining bootstrap aggregation, clusters and exponential smoothing.* International Journal of
> Forecasting, 34(4), 748–761. https://doi.org/10.1016/j.ijforecast.2018.05.006

The method generates bootstrapped versions of a time series (Box-Cox + STL + moving block
bootstrap), evaluates them on a validation window, and builds a deliberately *less-correlated*
ensemble of ETS forecasts by selecting series across PAM clusters — attacking the covariance term
of the ensemble variance that plain bagging ignores. Final forecast is the pointwise median.

This package is both:

1. **A clean, fast implementation** of the published method (statsforecast AutoETS engine,
   ~100x faster than the original R), statistically validated against the original R code
   (see `validation_r/`).
2. **A research platform**: ensemble-selection strategies are pluggable, enabling direct
   comparison of the paper's cluster-based selection against covariance-direct alternatives
   and ablations:
   - `ClusterSelection` — the paper's method (`.paper()` pins the exact 2018 constants);
   - `GreedyCovarianceSelection` — Caruana-style forward selection minimizing the
     *aggregated* ensemble's validation error;
   - `PortfolioSelection` — greedy bias² + covariance minimization (Ledoit-Wolf-shrunk),
     the direct formalization of the paper's Eq. (4);
   - `TopKSelection` (accuracy without diversity — the ablation the paper lacked),
     `RandomSelection`, `NoSelection` (Bergmeir et al.'s protocol via `NoSelection(100)` + mean).

## Quick start

```python
import numpy as np
from baggets import BaggedETS
from baggets.selection import ClusterSelection

model = BaggedETS(season_length=12, selection=ClusterSelection(n_clusters="auto"),
                  random_state=42)
model.fit(y)                 # y: 1-D array-like
fc = model.predict(h=18)
fc.point                     # median ensemble forecast, shape (18,)
fc.ensemble                  # member forecasts, shape (n_members, 18)
```

## A neural base learner: N-BEATS

The base learner is pluggable (`baggets.engine.ForecastEngine`). `ETSEngine` is
the paper's and the default; `NBeatsEngine` swaps in a PyTorch implementation of
N-BEATS (Oreshkin et al., ICLR 2020) so the *same* bootstrap machinery, member
selection, metrics and M3 harness compare a statistical base learner against a
deep one on identical footing.

```python
from baggets import BaggedETS, NBeatsConfig, NBeatsEngine, NoSelection

model = BaggedETS(season_length=12, n_bootstraps=100, n_select=100,
                  selection=NoSelection(100), aggregator="mean",   # plain bagging
                  engine=NBeatsEngine(12, NBeatsConfig(interpretable=True)))
model.fit(y)
model.predict(h=18)
```

Two departures from the published N-BEATS, both because it is a base learner
inside a per-series bootstrap ensemble rather than the paper's setup, and both
worth stating plainly rather than discovering in the numbers:

1. **Per series, not cross-learning.** `fit(y)` receives one series, so a model
   sees only that series' windows. N-BEATS's published M3/M4 results come from
   cross-learning over all series — orders of magnitude more data. On an M3
   monthly series of 50 points this leaves **8 training windows**.
2. **Depth and width scaled to the data.** The paper's generic stack is 30
   blocks of width 512. The defaults here are far smaller; the paper's sizes are
   still reachable through `NBeatsConfig`.

The engine falls back to a naive forecast, and says so, on series too short to
yield training windows — the same discipline `ETSEngine` applies to a failed fit.

Training is instrumentable: pass `monitor_factory=` and each epoch is handed to
your monitor. The interface is duck-typed and baggets takes no dependency on any
particular one; a separate probe library (dead-ReLU fraction per layer, gradient
norm by depth and through time, validation divergence) satisfies it and finds, on a
real M3 series, a 60%-dead ReLU layer and validation loss bottoming at epoch 56 of 400.

```bash
uv sync --extra torch
uv run python experiments/run_m3.py precompute --group monthly --limit 300 \
    --n-bootstraps 100 --engine nbeats-i --epochs 150 --n-jobs 6
```

**Cost, and why neural runs are matched-subset.** An ETS fit is milliseconds; an
N-BEATS fit is seconds. At the paper's B=1000 over 1,428 series that is the
difference between an overnight run and an infeasible one. Neural runs therefore
use a smaller B on a series subset — and the ETS arm is rerun under the *same* B
and the *same* series, because otherwise the comparison means nothing.

## An S3 artifact store

Stage A is the expensive half, and its per-series cache is worth moving off one
laptop. `baggets.aws.S3ArtifactStore` (extra: `aws`, uses boto3 and the standard
credential chain) reads and writes those `.npz` tensors, and model checkpoints as
state dicts — weights only, because `torch.save` of a whole module pickles code
paths and an artifact store is the wrong place for something whose load is
arbitrary code execution.

## Development

```bash
uv sync --all-extras        # create venv, install deps
uv run pytest -n auto       # unit + fixture tests (no R required)
```

R fixture regeneration (requires R 4.3+ with the `forecast` and `cluster` packages) is manual —
see `validation_r/README.md`. The original 2018 R implementation is preserved verbatim in
`validation_r/original/`.

## Layout

- `src/baggets/` — the package (bootstrap core, ETS engine wrapper, selection strategies, estimator)
- `tests/` — unit tests + committed R-generated fixtures
- `validation_r/` — scripts that generate reference fixtures from the original R code
- `experiments/` — M3 benchmark runner (precompute / evaluate / report)

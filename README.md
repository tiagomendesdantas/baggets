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

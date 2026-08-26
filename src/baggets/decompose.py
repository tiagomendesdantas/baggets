"""Trend/seasonal/remainder decomposition matching the R reference.

Seasonal series: R's ``stl(x, s.window = "periodic")``, which internally sets
``s.window = 10*n + 1, s.degree = 0`` and then *post-averages* the seasonal
component by cycle position (``tapply(seasonal, cycle(x), mean)``) before
recomputing the remainder. statsmodels STL shares R's netlib Fortran lineage,
so passing R's exact settings reproduces R bit-for-bit. Three of them are NOT
statsmodels defaults and are required for that match:

- the loess "jump" parameters (R evaluates each loess at every ``ceil(window/10)``-th
  point and interpolates; statsmodels defaults to exact evaluation at every point);
- ``inner_iter=2, outer_iter=0`` (R's non-robust defaults; statsmodels'
  ``fit()`` defaults to 5 inner iterations);
- the trend/low-pass windows, computed here with R's ``nextodd`` formulas so we
  can derive their jumps.

Non-seasonal series: R's ``loess(x ~ t, span = 6/n, degree = 1)`` with the
default gaussian family, i.e. zero robustness iterations — hence ``it=0`` below
(statsmodels' default ``it=3`` would diverge from R on outlier-heavy series).
"""

from __future__ import annotations

import math

import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.seasonal import STL

__all__ = ["decompose"]


def _nextodd(x: float) -> int:
    """R stats:::nextodd — round, then bump to the next odd integer."""
    r = round(x)
    return int(r + 1 if r % 2 == 0 else r)


def decompose(
    x: np.ndarray, season_length: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose ``x`` into ``(trend, seasonal, remainder)`` summing exactly to ``x``.

    ``season_length > 1`` uses periodic STL; otherwise a loess trend with zero
    seasonal. Matches the decomposition step of ``bld.mbb.bootstrap`` in the
    original R code.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if season_length > 1:
        if n <= 2 * season_length:
            raise ValueError(
                f"STL needs more than two full seasonal cycles: n={n}, "
                f"season_length={season_length}"
            )
        s_window = 10 * n + 1
        t_window = _nextodd(math.ceil(1.5 * season_length / (1.0 - 1.5 / s_window)))
        l_window = _nextodd(season_length)
        res = STL(
            x,
            period=season_length,
            seasonal=s_window,
            trend=t_window,
            low_pass=l_window,
            seasonal_deg=0,
            robust=False,
            seasonal_jump=math.ceil(s_window / 10),
            trend_jump=math.ceil(t_window / 10),
            low_pass_jump=math.ceil(l_window / 10),
        ).fit(inner_iter=2, outer_iter=0)
        trend = np.asarray(res.trend, dtype=np.float64)
        # R's "periodic" post-step: average the seasonal by cycle position, then
        # recompute the remainder against the averaged seasonal.
        cycle_pos = np.arange(n) % season_length
        seasonal_raw = np.asarray(res.seasonal, dtype=np.float64)
        cycle_means = np.array(
            [seasonal_raw[cycle_pos == p].mean() for p in range(season_length)]
        )
        seasonal = cycle_means[cycle_pos]
    else:
        t = np.arange(1.0, n + 1.0)
        frac = min(1.0, 6.0 / n)
        trend = lowess(x, t, frac=frac, it=0, return_sorted=False)
        seasonal = np.zeros(n)
    remainder = x - trend - seasonal
    return trend, seasonal, remainder

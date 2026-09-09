"""Validation-window artifacts: the inputs every selection strategy consumes.

Mirrors the validation step of the original R code: truncate each bootstrapped
member by the last ``h_val`` observations, fit ETS, forecast ``h_val`` ahead,
and score against the *original* series' held-out tail (not the member's own).
The paper's Algorithm 2 says sMAPE; the R code that produced the published
numbers used plain MAPE — hence the pluggable metric with "mape" as default.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from joblib import Parallel, delayed

from .engine import ETSEngine, ForecastEngine
from .metrics import mape, smape

__all__ = ["ValidationArtifacts", "build_validation_artifacts", "default_h_val", "MIN_H_VAL"]

_METRICS: dict[str, Callable] = {"mape": mape, "smape": smape}


def default_h_val(n: int, season_length: int) -> int:
    """Validation-window length: R's rule, shrunk for short series.

    R used ``2*m`` (seasonal) or 10 and *crashed* on any series with
    ``n - h_val <= h_val`` (undefined-variable bug in the short-series branch).
    We instead shrink the window so the truncated series stays longer than the
    window. Callers should treat a result below ``max(4, m)`` (see MIN_H_VAL)
    as "too short for validation-based selection" and fall back to no selection.
    """
    h = 2 * season_length if season_length > 1 else 10
    return min(h, (n - 1) // 2)


def MIN_H_VAL(season_length: int) -> int:
    return max(4, season_length)


@dataclass(frozen=True)
class ValidationArtifacts:
    """Everything a selection strategy may need, for one series' bootstrap pool."""

    series: np.ndarray
    """(B, T) bootstrapped series, full length; member 0 is the original."""
    val_forecasts: np.ndarray
    """(B, h_val) forecasts from members truncated by h_val."""
    val_actuals: np.ndarray
    """(h_val,) the original series' held-out tail."""
    val_errors: np.ndarray
    """(B,) validation metric of each member's forecast vs val_actuals."""
    season_length: int
    meta: dict = field(default_factory=dict)

    @property
    def n_members(self) -> int:
        return self.series.shape[0]


def build_validation_artifacts(
    series: np.ndarray,
    y: np.ndarray,
    season_length: int,
    h_val: int,
    validation_metric: str = "mape",
    n_jobs: int = 1,
    meta: dict | None = None,
    engine: ForecastEngine | None = None,
) -> ValidationArtifacts:
    """Fit the base learner to each truncated member and score its forecast of the held-out tail.

    ``series`` is the (B, T) bootstrap pool (member 0 = original ``y``).
    ``engine`` defaults to ``ETSEngine`` — pass another (e.g. ``NBeatsEngine``)
    to validate members under a different base learner.
    """
    if validation_metric not in _METRICS:
        raise ValueError(f"unknown validation_metric {validation_metric!r}; use {set(_METRICS)}")
    metric = _METRICS[validation_metric]
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    if not 0 < h_val < n - h_val:
        raise ValueError(f"h_val={h_val} invalid for series of length {n}")

    val_actuals = y[n - h_val :]
    truncated = series[:, : n - h_val]
    engine = engine if engine is not None else ETSEngine(season_length)

    def _one(member: np.ndarray) -> tuple[np.ndarray, bool]:
        fc = engine.forecast(member, h_val)
        return fc.mean, fc.fallback

    if n_jobs == 1:
        results = [_one(m) for m in truncated]
    else:
        results = Parallel(n_jobs=n_jobs, batch_size=64)(delayed(_one)(m) for m in truncated)

    val_forecasts = np.vstack([r[0] for r in results])
    n_fallback = int(sum(r[1] for r in results))
    val_errors = np.array([metric(val_actuals, f) for f in val_forecasts])

    full_meta = {"h_val": h_val, "validation_metric": validation_metric,
                 "n_val_fallbacks": n_fallback}
    if meta:
        full_meta.update(meta)
    return ValidationArtifacts(
        series=series,
        val_forecasts=val_forecasts,
        val_actuals=val_actuals,
        val_errors=val_errors,
        season_length=season_length,
        meta=full_meta,
    )

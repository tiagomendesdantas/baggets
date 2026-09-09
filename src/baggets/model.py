"""The BaggedETS estimator: bootstrap -> validate -> select -> ensemble forecast.

Implements Bagged.Cluster.ETS (Dantas & Cyrino Oliveira 2018) with the paper's
defaults — 1000 bootstraps, ~100 selected members, cluster-based selection,
median aggregation — and deliberate fixes over the original R code, documented
in the README: a required ``h`` in ``predict`` (R silently defaulted to 10 via
a bug), a shrinking validation window instead of a crash on short series (R's
short-series branch referenced undefined variables), and honest ensemble
quantiles instead of min/max pseudo-intervals labeled "level=100".
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from joblib import Parallel, delayed

from .bootstrap import bld_mbb_bootstrap
from .engine import ETSEngine, ForecastEngine
from .selection import ClusterSelection, SelectionStrategy
from .validation import MIN_H_VAL, build_validation_artifacts, default_h_val

__all__ = ["BaggedETS", "Forecast"]

_AGGREGATORS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "median": lambda e: np.median(e, axis=0),
    "mean": lambda e: np.mean(e, axis=0),
}


@dataclass(frozen=True)
class Forecast:
    point: np.ndarray
    """(h,) aggregated forecast."""
    ensemble: np.ndarray
    """(n_members, h) member forecasts."""
    info: dict = field(default_factory=dict)

    def quantile(self, q: float | np.ndarray) -> np.ndarray:
        """Ensemble quantiles — honest spread of the member forecasts, NOT
        calibrated prediction intervals."""
        return np.quantile(self.ensemble, q, axis=0)


class BaggedETS:
    """Bagged bootstrap ETS ensemble with pluggable member selection.

    Parameters follow the paper's defaults; ``selection=ClusterSelection.paper()``
    pins the exact 2018 constants for validation runs.
    """

    def __init__(
        self,
        season_length: int,
        n_bootstraps: int = 1000,
        n_select: int = 100,
        selection: SelectionStrategy | None = None,
        aggregator: str | Callable[[np.ndarray], np.ndarray] = "median",
        validation_metric: str = "mape",
        h_val: int | None = None,
        block_size: int | None = None,
        random_state: int | None = None,
        n_jobs: int = 1,
        engine: ForecastEngine | None = None,
    ):
        self.season_length = int(season_length)
        self.n_bootstraps = int(n_bootstraps)
        self.n_select = int(n_select)
        self.selection = ClusterSelection() if selection is None else selection
        self.aggregator = aggregator
        self.validation_metric = validation_metric
        self.h_val = h_val
        self.block_size = block_size
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.engine = engine if engine is not None else ETSEngine(self.season_length)

    # ------------------------------------------------------------------ fit
    def fit(self, y: np.ndarray) -> BaggedETS:
        y = np.asarray(y, dtype=np.float64)
        rng = np.random.default_rng(self.random_state)

        boot = bld_mbb_bootstrap(
            y, self.n_bootstraps, self.season_length, rng, block_size=self.block_size
        )

        h_val = self.h_val if self.h_val is not None else default_h_val(y.size, self.season_length)
        selection = self.selection
        selection_fallback = False
        if h_val < MIN_H_VAL(self.season_length):
            warnings.warn(
                f"series too short for validation-based selection "
                f"(h_val={h_val} < {MIN_H_VAL(self.season_length)}); "
                f"falling back to a random subset of {self.n_select} members "
                "(the original R code crashed here)",
                UserWarning,
                stacklevel=2,
            )
            selection_fallback = True
            # no usable validation window: select randomly, skip validation fits
            selected = np.sort(
                rng.choice(boot.series.shape[0], size=min(self.n_select, boot.series.shape[0]),
                           replace=False)
            )
            self.artifacts_ = None
        else:
            self.artifacts_ = build_validation_artifacts(
                boot.series,
                y,
                self.season_length,
                h_val,
                validation_metric=self.validation_metric,
                n_jobs=self.n_jobs,
                meta={"lambda": boot.lam, "n_clipped": boot.n_clipped},
                engine=self.engine,
            )
            selected = selection.select(self.artifacts_, self.n_select, rng)

        engine = self.engine
        members = boot.series[selected]
        if self.n_jobs == 1:
            models: list[object] = [engine.fit(m) for m in members]
        else:
            models = Parallel(n_jobs=self.n_jobs, batch_size=16)(
                delayed(engine.fit)(m) for m in members
            )

        self.y_ = y
        self.selected_indices_ = selected
        self.models_ = models
        self.h_val_ = h_val
        self.info_ = {
            "lambda": boot.lam,
            "n_clipped": boot.n_clipped,
            "h_val": h_val,
            "engine": type(engine).__name__,
            "selection": selection.name if not selection_fallback else "fallback_random",
            "n_members": len(selected),
            "n_final_fallbacks": int(sum(getattr(m, "fallback", False) for m in models)),
        }
        if self.artifacts_ is not None:
            self.info_["n_val_fallbacks"] = self.artifacts_.meta["n_val_fallbacks"]
            if "k_chosen" in self.artifacts_.meta:
                self.info_["k_chosen"] = self.artifacts_.meta["k_chosen"]
        return self

    # -------------------------------------------------------------- predict
    def predict(self, h: int) -> Forecast:
        """Forecast ``h`` steps ahead. ``h`` is required by design — the original
        R method silently defaulted to 10 through a bug."""
        if not hasattr(self, "models_"):
            raise RuntimeError("call fit() before predict()")
        engine = self.engine
        ensemble = np.vstack([engine.predict(m, h) for m in self.models_])
        agg = _AGGREGATORS.get(self.aggregator) if isinstance(self.aggregator, str) \
            else self.aggregator
        if agg is None:
            raise ValueError(f"unknown aggregator {self.aggregator!r}")
        return Forecast(point=np.asarray(agg(ensemble)), ensemble=ensemble, info=dict(self.info_))

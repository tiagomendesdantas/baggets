"""ETS engine: statsforecast AutoETS behind a small, fault-tolerant wrapper.

statsforecast's AutoETS is a numba port of Hyndman's ``forecast::ets`` — same
30-model space, same restrictions (no multiplicative trend in automatic search,
multiplicative options dropped for non-positive series, no seasonality for
``season_length > 24``) — but orders of magnitude faster, which is what makes
1000-bootstrap experiments tractable. Individual fits can still fail on
degenerate bootstrapped series; the wrapper falls back (ANN exponential
smoothing, then naive last-value) and reports it, so experiments can assert the
fallback rate stays negligible.
"""

from __future__ import annotations

import os
import warnings
from typing import NamedTuple

import numpy as np

if "NUMBA_CACHE_DIR" not in os.environ:  # persist JIT compilation across runs
    os.environ["NUMBA_CACHE_DIR"] = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data",
        ".numba_cache",
    )

from statsforecast.models import AutoETS  # noqa: E402

__all__ = ["ETSEngine", "EngineForecast", "FittedETS"]

# --------------------------------------------------------------------------
# statsforecast optimizer patch (verified 2026-08-26 against R forecast::ets)
#
# statsforecast (<=2.1.1) calls its compiled Nelder-Mead with hardcoded
# tol=1e-4, maxit=1000 (see optimize_ets_target_fn). On smooth series — the
# bootstrapped members this package lives on — NM hits the iteration cap far
# from the optimum, producing likelihoods up to ~70 AICc points worse than R
# and flipping automatic model selection away from trended families
# (e.g. M3 N2501 members: median validation MAPE 12.6 vs R's 3.3). Raising
# maxit to 4000 with tol=1e-8 restores R-family model picks (12.6 -> 5.8) at
# ~3x the fit cost; tighter settings buy nothing more. maxit, not tol, is the
# binding constraint: 1e-8 with maxit=2000 is as bad as the default.
# --------------------------------------------------------------------------
_ETS_NM_TOL = 1e-8
_ETS_NM_MAXIT = 4000


def _install_optimizer_patch() -> None:
    import statsforecast.ets as _sfets

    if getattr(_sfets._ets.optimize, "_baggets_patched", False):
        return
    _original = _sfets._ets.optimize

    def _tight_optimize(x0, y, nstate, e, t, s, crit, nmse, m, oA, oB, oG, oP,
                        alpha, beta, gamma, phi, lowerb, upperb, _tol, _maxit, adaptive):
        return _original(x0, y, nstate, e, t, s, crit, nmse, m, oA, oB, oG, oP,
                         alpha, beta, gamma, phi, lowerb, upperb,
                         _ETS_NM_TOL, _ETS_NM_MAXIT, adaptive)

    _tight_optimize._baggets_patched = True
    _sfets._ets.optimize = _tight_optimize


_install_optimizer_patch()


class FittedETS(NamedTuple):
    model: object | None
    """Fitted statsforecast model, or None for the naive fallback."""
    last_value: float
    method: str
    """Selected model, e.g. "ETS(M,Ad,M)"; fallbacks report "ETS-ANN-fallback" / "naive-fallback"."""
    fallback: bool


class EngineForecast(NamedTuple):
    mean: np.ndarray
    """(h,) point forecasts."""
    fitted: np.ndarray | None
    """(n,) in-sample fitted values (None unless requested)."""
    method: str
    fallback: bool


class ETSEngine:
    """Automatic ETS fit + forecast for one seasonal frequency."""

    def __init__(self, season_length: int):
        self.season_length = int(season_length)

    def fit(self, y: np.ndarray) -> FittedETS:
        """Fit automatic ETS to ``y``, falling back (ANN, then naive) on failure."""
        y = np.ascontiguousarray(y, dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for season_length, model_spec, tag in (
                (self.season_length, "ZZZ", None),
                (1, "ANN", "ETS-ANN-fallback"),
            ):
                try:
                    fitted = AutoETS(season_length=season_length, model=model_spec).fit(y=y)
                    # a 1-step sanity check: reject fits that predict non-finite values
                    if np.all(np.isfinite(fitted.predict(h=1)["mean"])):
                        method = tag or str(fitted.model_.get("method", "ETS"))
                        return FittedETS(fitted, float(y[-1]), method, tag is not None)
                except Exception:
                    continue
        return FittedETS(None, float(y[-1]), "naive-fallback", True)

    def predict(self, fitted: FittedETS, h: int) -> np.ndarray:
        """Point forecasts from a ``fit()`` result (naive fallback repeats the last value)."""
        if fitted.model is None:
            return np.full(h, fitted.last_value, dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean = np.asarray(fitted.model.predict(h=h)["mean"], dtype=np.float64)
        if not np.all(np.isfinite(mean)):
            return np.full(h, fitted.last_value, dtype=np.float64)
        return mean

    def forecast(self, y: np.ndarray, h: int, with_fitted: bool = False) -> EngineForecast:
        """One-shot fit + ``h``-step forecast."""
        fitted = self.fit(y)
        mean = self.predict(fitted, h)
        fitted_values = None
        if with_fitted:
            if fitted.model is None:
                y = np.asarray(y, dtype=np.float64)
                fitted_values = np.concatenate([[np.nan], y[:-1]])
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fitted_values = np.asarray(
                        fitted.model.predict_in_sample()["fitted"], dtype=np.float64
                    )
        return EngineForecast(mean, fitted_values, fitted.method, fitted.fallback)

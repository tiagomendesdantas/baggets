"""Box-Cox transform with Guerrero lambda selection.

Ports the exact semantics of R's ``forecast::BoxCox.lambda(x, lower=0, upper=1)``
(Guerrero's method, the default), ``forecast::BoxCox`` and ``forecast::InvBoxCox``,
which the original Bagged.Cluster.ETS bootstrap relies on. scipy's MLE-based
``boxcox`` is deliberately not used: it is a different estimator and diverges from
the R results the 2018 paper was built on.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

__all__ = ["guerrero_lambda", "boxcox", "inv_boxcox"]

_LOG_LAMBDA_TOL = 1e-12


def guerrero_lambda(
    x: np.ndarray,
    season_length: int = 1,
    lower: float = 0.0,
    upper: float = 1.0,
) -> float:
    """Guerrero's (1993) coefficient-of-variation method for the Box-Cox lambda.

    Mirrors ``forecast:::guer.cv``: split the series into trailing non-overlapping
    blocks of ``max(2, season_length)`` consecutive observations (dropping leftover
    points from the *front*), compute each block's mean and sample standard
    deviation, and pick the lambda minimizing the coefficient of variation of
    ``sd / mean**(1 - lambda)`` across blocks.

    Mirrors ``forecast::BoxCox.lambda``'s guard: series with
    ``len(x) <= 2 * season_length`` return 1.0 (identity transform).
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n <= 2 * season_length:
        return 1.0

    period = int(round(max(2, season_length)))
    n_blocks = n // period
    if n_blocks < 2:
        return 1.0
    # R: matrix(x[(nobsf - nobst + 1):nobsf], period, nyr) — trailing obs, column-major,
    # so each column is one block of `period` consecutive observations.
    blocks = x[n - n_blocks * period :].reshape(n_blocks, period)
    block_mean = blocks.mean(axis=1)
    block_sd = blocks.std(axis=1, ddof=1)

    def cv(lam: float) -> float:
        ratio = block_sd / block_mean ** (1.0 - lam)
        return float(np.std(ratio, ddof=1) / np.mean(ratio))

    res = minimize_scalar(cv, bounds=(lower, upper), method="bounded", options={"xatol": 1e-6})
    return float(res.x)


def boxcox(x: np.ndarray, lam: float) -> np.ndarray:
    """Box-Cox transform, matching ``forecast::BoxCox``.

    Uses the Bickel-Doksum form ``(sign(x)|x|^lam - 1)/lam`` so negative values
    behave as in R (relevant only outside the usual positive-data case).
    """
    x = np.asarray(x, dtype=np.float64)
    if abs(lam) < _LOG_LAMBDA_TOL:
        return np.log(x)
    return (np.sign(x) * np.abs(x) ** lam - 1.0) / lam


def inv_boxcox(
    x: np.ndarray,
    lam: float,
    return_nclipped: bool = False,
) -> np.ndarray | tuple[np.ndarray, int]:
    """Inverse Box-Cox, matching ``forecast::InvBoxCox`` — with one deliberate fix.

    For ``lam > 0``, ``lam*x + 1`` can go negative when extreme bootstrap
    remainders are added to the transformed series. R produced NaN there and the
    downstream ``ets`` call crashed; we clip to a tiny positive value instead and
    (optionally) report how many points were clipped so callers can surface it.
    """
    x = np.asarray(x, dtype=np.float64)
    if abs(lam) < _LOG_LAMBDA_TOL:
        out = np.exp(x)
        n_clipped = 0
    else:
        xx = lam * x + 1.0
        n_clipped = int(np.count_nonzero(xx <= 0.0))
        if n_clipped:
            tiny = np.finfo(np.float64).tiny
            xx = np.clip(xx, tiny, None)
        out = xx ** (1.0 / lam)
    if return_nclipped:
        return out, n_clipped
    return out

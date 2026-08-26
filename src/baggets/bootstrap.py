"""Box-Cox + STL + Moving Block Bootstrap series generation.

Faithful port of ``bld.mbb.bootstrap`` from Bergmeir, Hyndman & Benitez (2016)
as used (verbatim) by the original Bagged.Cluster.ETS R code: member 0 is the
original series; every other member is ``InvBoxCox(trend + seasonal +
MBB(remainder))`` for one fixed decomposition of the Box-Cox-transformed series.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .boxcox import boxcox, guerrero_lambda, inv_boxcox
from .decompose import decompose

__all__ = ["moving_block_bootstrap", "bld_mbb_bootstrap", "BootstrapResult"]


def moving_block_bootstrap(
    x: np.ndarray, block_size: int, rng: np.random.Generator
) -> np.ndarray:
    """One MBB draw, matching the R ``MBB`` helper.

    Draws ``floor(n/block_size) + 2`` blocks of ``block_size`` consecutive values
    with uniformly random starts, concatenates them, and returns ``n`` values
    from a uniformly random offset in ``[0, block_size)``.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < block_size:
        raise ValueError(
            f"series too short for moving block bootstrap: n={n} < block_size={block_size}"
        )
    n_blocks = n // block_size + 2
    starts = rng.integers(0, n - block_size + 1, size=n_blocks)
    blocks = x[starts[:, None] + np.arange(block_size)[None, :]]
    offset = int(rng.integers(0, block_size))
    return blocks.ravel()[offset : offset + n]


class BootstrapResult(NamedTuple):
    series: np.ndarray
    """(n_bootstraps, len(x)) array; row 0 is the original series."""
    lam: float
    """Guerrero Box-Cox lambda used (1.0 means identity / not estimated)."""
    n_clipped: int
    """Points clipped in the inverse Box-Cox across all members (R produced NaN here)."""


def bld_mbb_bootstrap(
    x: np.ndarray,
    n_bootstraps: int,
    season_length: int,
    rng: np.random.Generator,
    block_size: int | None = None,
) -> BootstrapResult:
    """Generate ``n_bootstraps`` bootstrapped versions of ``x`` (member 0 = ``x``)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"expected a 1-D series, got shape {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("series contains NaN or infinite values")
    n = x.size
    if block_size is None:
        block_size = 2 * season_length if season_length > 1 else 8

    series = np.empty((n_bootstraps, n), dtype=np.float64)
    series[0] = x
    if n_bootstraps == 1:
        return BootstrapResult(series, 1.0, 0)

    lam = guerrero_lambda(x, season_length, lower=0.0, upper=1.0)
    x_bc = boxcox(x, lam)
    trend, seasonal, remainder = decompose(x_bc, season_length)
    structure = trend + seasonal

    n_clipped = 0
    for i in range(1, n_bootstraps):
        resampled = structure + moving_block_bootstrap(remainder, block_size, rng)
        series[i], clipped = inv_boxcox(resampled, lam, return_nclipped=True)
        n_clipped += clipped
    return BootstrapResult(series, lam, n_clipped)

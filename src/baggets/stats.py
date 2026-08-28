"""Statistical comparison of strategies, following the paper's protocol.

The 2018 paper (Tables 6/9/14) uses the Friedman rank-sum test across series,
then a post-hoc procedure comparing every method against a control (the
best-ranked one) with Hochberg's step-up adjustment of the pairwise p-values.
Pairwise comparisons use the sign-invariant normal approximation on average
rank differences (Demsar 2006 / Garcia et al. 2010 style), which is what the
`scmamp`-era R tooling used by that literature computes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, norm

__all__ = ["friedman_hochberg"]


def friedman_hochberg(
    scores: pd.DataFrame,
    metric: str = "smape",
    control: str | None = None,
) -> dict:
    """Friedman test + Hochberg-adjusted comparisons against a control strategy.

    ``scores``: tidy frame with columns (uid, strategy, <metric>), one row per
    (series, strategy) — average over seeds first if needed. ``control`` picks
    the reference strategy (default: best mean rank, like the paper).

    Returns {"friedman_pvalue", "control", "table"} where table has one row per
    non-control strategy: mean rank, z, raw and Hochberg-adjusted p-values.
    """
    wide = scores.pivot_table(index="uid", columns="strategy", values=metric)
    if wide.isna().any().any():
        missing = wide.columns[wide.isna().any()].tolist()
        raise ValueError(f"strategies with missing series: {missing}")

    ranks = wide.rank(axis=1)
    mean_ranks = ranks.mean().sort_values()
    n, k = wide.shape

    stat_args = [wide[c].to_numpy() for c in wide.columns]
    friedman_p = float(friedmanchisquare(*stat_args).pvalue)

    if control is None:
        control = str(mean_ranks.index[0])
    elif control not in wide.columns:
        raise ValueError(f"unknown control {control!r}")

    # normal approximation for average-rank differences vs the control
    se = np.sqrt(k * (k + 1) / (6.0 * n))
    others = [c for c in mean_ranks.index if c != control]
    z = np.array([(mean_ranks[c] - mean_ranks[control]) / se for c in others])
    p_raw = 2.0 * (1.0 - norm.cdf(np.abs(z)))

    # Hochberg step-up on the k-1 comparisons
    order = np.argsort(p_raw)[::-1]  # largest p first
    m = len(p_raw)
    adj = np.empty(m)
    running_min = 1.0
    for rank_pos, idx in enumerate(order):
        factor = rank_pos + 1  # 1 for the largest p, m for the smallest
        running_min = min(running_min, factor * p_raw[idx])
        adj[idx] = running_min
    adj = np.clip(adj, 0.0, 1.0)

    table = pd.DataFrame({
        "strategy": others,
        "mean_rank": [mean_ranks[c] for c in others],
        "z": z,
        "p_raw": p_raw,
        "p_hochberg": adj,
    }).sort_values("mean_rank", ignore_index=True)

    return {
        "friedman_pvalue": friedman_p,
        "control": control,
        "control_mean_rank": float(mean_ranks[control]),
        "n_series": int(n),
        "table": table,
    }

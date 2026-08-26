"""Tests for the covariance-direct strategies (the research surface)."""

import numpy as np

from baggets.selection import (
    GreedyCovarianceSelection,
    PortfolioSelection,
    TopKSelection,
)
from baggets.validation import ValidationArtifacts


def correlated_pool(seed=0, n_members=40, h=12):
    """Pool where the individually-best members share one bias direction (they
    are copies of the same over-forecast) while slightly-worse members carry
    independent, mean-zero errors. Top-k picks the correlated clones; a
    covariance-aware strategy should mix in the diverse members."""
    rng = np.random.default_rng(seed)
    actuals = np.full(h, 100.0)
    forecasts = np.empty((n_members, h))
    # members 0..9: identical +2.0 biased forecast, tiny independent jitter
    for i in range(10):
        forecasts[i] = actuals + 2.0 + rng.normal(0, 0.05, h)
    # members 10..39: unbiased but noisier (individually worse than the clones)
    for i in range(10, n_members):
        forecasts[i] = actuals + rng.normal(0, 3.0, h)
    errors = np.array([np.mean(np.abs((actuals - f) / actuals)) * 100 for f in forecasts])
    return ValidationArtifacts(
        series=rng.normal(size=(n_members, 30)),
        val_forecasts=forecasts,
        val_actuals=actuals,
        val_errors=errors,
        season_length=1,
    )


class TestGreedyCovariance:
    def test_basic_properties(self):
        art = correlated_pool()
        idx = GreedyCovarianceSelection().select(art, 12, np.random.default_rng(0))
        assert idx.size == 12
        assert np.unique(idx).size == 12
        assert idx.min() >= 0 and idx.max() < art.n_members

    def test_first_pick_is_best_single(self):
        art = correlated_pool()
        idx1 = GreedyCovarianceSelection().select(art, 1, np.random.default_rng(0))
        assert idx1[0] == int(np.argmin(art.val_errors))

    def test_beats_topk_on_validation_window(self):
        art = correlated_pool()
        rng = np.random.default_rng(0)
        greedy = GreedyCovarianceSelection(aggregator="mean").select(art, 12, rng)
        topk = TopKSelection().select(art, 12, rng)
        err = lambda idx: np.mean(  # noqa: E731
            np.abs(art.val_actuals - art.val_forecasts[idx].mean(axis=0))
        )
        # top-k picks the 10 identical biased clones -> ensemble inherits the bias;
        # greedy mixes in diverse members and must do at least as well
        assert err(greedy) < err(topk)

    def test_deterministic(self):
        art = correlated_pool()
        a = GreedyCovarianceSelection().select(art, 10, np.random.default_rng(1))
        b = GreedyCovarianceSelection().select(art, 10, np.random.default_rng(2))
        np.testing.assert_array_equal(a, b)  # rng is unused by the greedy path

    def test_patience_stops_early(self):
        art = correlated_pool()
        idx = GreedyCovarianceSelection(patience=0).select(art, 39, np.random.default_rng(0))
        assert idx.size < 39


class TestPortfolio:
    def test_basic_properties(self):
        art = correlated_pool()
        idx = PortfolioSelection().select(art, 12, np.random.default_rng(0))
        assert idx.size == 12
        assert np.unique(idx).size == 12

    def test_avoids_stacking_correlated_clones(self):
        art = correlated_pool()
        idx = PortfolioSelection().select(art, 12, np.random.default_rng(0))
        n_clones = int(np.sum(idx < 10))
        # top-k would take all 10 clones; portfolio must not
        assert n_clones < 10

    def test_deterministic(self):
        art = correlated_pool()
        a = PortfolioSelection().select(art, 12, np.random.default_rng(1))
        b = PortfolioSelection().select(art, 12, np.random.default_rng(2))
        np.testing.assert_array_equal(a, b)

import numpy as np
import pytest

from baggets import BaggedETS, ClusterSelection, NoSelection, TopKSelection


@pytest.fixture(scope="module")
def y_monthly():
    rng = np.random.default_rng(11)
    t = np.arange(96.0)
    return (140 + 1.2 * t) * (1 + 0.18 * np.sin(2 * np.pi * t / 12)) * np.exp(
        rng.normal(0, 0.03, 96)
    )


@pytest.fixture(scope="module")
def fitted_small(y_monthly):
    return BaggedETS(
        season_length=12, n_bootstraps=60, n_select=12,
        selection=ClusterSelection(n_clusters=3, n_prefilter=30),
        random_state=7,
    ).fit(y_monthly)


class TestBaggedETS:
    def test_fit_predict_shapes(self, fitted_small):
        fc = fitted_small.predict(h=18)
        assert fc.point.shape == (18,)
        assert fc.ensemble.shape == (12, 18)
        assert np.all(np.isfinite(fc.point))

    def test_info_contents(self, fitted_small):
        info = fitted_small.predict(h=6).info
        assert info["selection"] == "cluster(k=3)"
        assert info["n_members"] == 12
        assert info["k_chosen"] == 3
        assert 0.0 <= info["lambda"] <= 1.0
        assert info["h_val"] == 24

    def test_median_is_pointwise_ensemble_median(self, fitted_small):
        fc = fitted_small.predict(h=6)
        np.testing.assert_allclose(fc.point, np.median(fc.ensemble, axis=0))

    def test_mean_aggregator(self, y_monthly):
        m = BaggedETS(
            season_length=12, n_bootstraps=30, n_select=8, selection=TopKSelection(),
            aggregator="mean", random_state=1,
        ).fit(y_monthly)
        fc = m.predict(h=4)
        np.testing.assert_allclose(fc.point, fc.ensemble.mean(axis=0))

    def test_callable_aggregator(self, y_monthly):
        m = BaggedETS(
            season_length=12, n_bootstraps=30, n_select=8, selection=NoSelection(8),
            aggregator=lambda e: np.min(e, axis=0), random_state=1,
        ).fit(y_monthly)
        fc = m.predict(h=4)
        np.testing.assert_allclose(fc.point, fc.ensemble.min(axis=0))

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            BaggedETS(season_length=12).predict(h=5)

    def test_unknown_aggregator_raises(self, y_monthly):
        m = BaggedETS(
            season_length=12, n_bootstraps=20, n_select=5, selection=NoSelection(5),
            aggregator="bogus", random_state=0,
        ).fit(y_monthly)
        with pytest.raises(ValueError, match="aggregator"):
            m.predict(h=3)

    def test_quantiles_bracket_point(self, fitted_small):
        fc = fitted_small.predict(h=12)
        assert np.all(fc.quantile(0.0) <= fc.point)
        assert np.all(fc.point <= fc.quantile(1.0))

    def test_short_series_fallback_warns_but_works(self):
        rng = np.random.default_rng(2)
        y = 50 + np.arange(8.0) * 2 + rng.normal(0, 0.5, 8)  # m=1, n=8 -> h_val=3 < 4
        with pytest.warns(UserWarning, match="too short"):
            m = BaggedETS(season_length=1, n_bootstraps=20, n_select=6,
                          random_state=0).fit(y)
        fc = m.predict(h=4)
        assert fc.info["selection"] == "fallback_random"
        assert fc.point.shape == (4,)
        assert np.all(np.isfinite(fc.point))

    def test_reproducible_across_n_jobs(self, y_monthly):
        kw = dict(season_length=12, n_bootstraps=30, n_select=8,
                  selection=TopKSelection(), random_state=9)
        serial = BaggedETS(**kw, n_jobs=1).fit(y_monthly)
        parallel = BaggedETS(**kw, n_jobs=2).fit(y_monthly)
        np.testing.assert_array_equal(serial.selected_indices_, parallel.selected_indices_)
        np.testing.assert_allclose(
            serial.predict(h=6).point, parallel.predict(h=6).point, rtol=1e-12
        )

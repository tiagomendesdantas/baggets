import numpy as np
import pytest

from baggets.decompose import decompose


class TestSeasonalDecompose:
    def test_components_sum_to_series(self, monthly_series):
        trend, seasonal, remainder = decompose(monthly_series, 12)
        np.testing.assert_allclose(trend + seasonal + remainder, monthly_series, atol=1e-10)

    def test_seasonal_strictly_periodic(self, monthly_series):
        # R's "periodic" post-averaging makes the seasonal exactly repeat.
        _, seasonal, _ = decompose(monthly_series, 12)
        for start in range(12, len(seasonal), 12):
            np.testing.assert_array_equal(seasonal[start : start + 12], seasonal[:12])

    def test_seasonal_zero_mean_pattern(self, monthly_series):
        _, seasonal, _ = decompose(monthly_series, 12)
        # STL seasonal oscillates around zero
        assert abs(seasonal[:12].mean()) < 0.5 * np.abs(seasonal[:12]).max()

    def test_recovers_known_seasonal_shape(self):
        t = np.arange(240.0)
        true_seasonal = 10 * np.sin(2 * np.pi * t / 12)
        x = 100 + 0.5 * t + true_seasonal
        _, seasonal, remainder = decompose(x, 12)
        # Noise-free series: seasonal recovered closely, remainder near zero
        assert np.corrcoef(seasonal, true_seasonal)[0, 1] > 0.999
        assert remainder.std() < 0.5

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="two full seasonal cycles"):
            decompose(np.arange(24.0), 12)


class TestNonSeasonalDecompose:
    def test_components_sum_to_series(self, yearly_series):
        trend, seasonal, remainder = decompose(yearly_series, 1)
        np.testing.assert_array_equal(seasonal, np.zeros_like(yearly_series))
        np.testing.assert_allclose(trend + remainder, yearly_series, atol=1e-10)

    def test_trend_tracks_signal(self, yearly_series):
        trend, _, _ = decompose(yearly_series, 1)
        # Linear-trend series: loess trend explains most variance
        assert np.corrcoef(trend, yearly_series)[0, 1] > 0.99

    def test_short_series_frac_clamped(self):
        # n < 6 would give frac > 1; must not crash
        x = np.array([1.0, 2.0, 3.5, 3.0, 5.0])
        trend, _, remainder = decompose(x, 1)
        np.testing.assert_allclose(trend + remainder, x, atol=1e-10)

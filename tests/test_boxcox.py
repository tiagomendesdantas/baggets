import numpy as np
import pytest

from baggets.boxcox import boxcox, guerrero_lambda, inv_boxcox


class TestGuerreroLambda:
    def test_short_series_returns_identity(self):
        # R's BoxCox.lambda guard: n <= 2*frequency -> lambda = 1
        assert guerrero_lambda(np.arange(1.0, 25.0), season_length=12) == 1.0
        assert guerrero_lambda(np.array([1.0, 2.0]), season_length=1) == 1.0

    def test_bounds_respected(self, monthly_series):
        lam = guerrero_lambda(monthly_series, 12)
        assert 0.0 <= lam <= 1.0

    def test_multiplicative_series_prefers_log(self, monthly_series):
        # Variance grows with level -> lambda near 0
        assert guerrero_lambda(monthly_series, 12) < 0.1

    def test_additive_series_beats_multiplicative(self, monthly_series):
        rng = np.random.default_rng(3)
        t = np.arange(120.0)
        additive = 500 + 2.0 * t + 30 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 2.0, 120)
        # Constant-variance series wants lambda well above a multiplicative one's
        lam_add = guerrero_lambda(additive, 12)
        lam_mult = guerrero_lambda(monthly_series, 12)
        assert lam_add > 0.5
        assert lam_add > lam_mult + 0.3

    def test_deterministic(self, monthly_series):
        assert guerrero_lambda(monthly_series, 12) == guerrero_lambda(monthly_series, 12)


class TestBoxCoxRoundTrip:
    @pytest.mark.parametrize("lam", [0.0, 1e-7, 0.123, 0.5, 1.0])
    def test_roundtrip(self, monthly_series, lam):
        back = inv_boxcox(boxcox(monthly_series, lam), lam)
        np.testing.assert_allclose(back, monthly_series, rtol=1e-6)

    def test_lambda_zero_is_log(self, monthly_series):
        np.testing.assert_array_equal(boxcox(monthly_series, 0.0), np.log(monthly_series))
        np.testing.assert_allclose(inv_boxcox(np.log(monthly_series), 0.0), monthly_series, rtol=1e-12)

    def test_lambda_one_is_shift(self, monthly_series):
        np.testing.assert_allclose(boxcox(monthly_series, 1.0), monthly_series - 1.0)


class TestInvBoxCoxClipping:
    def test_negative_domain_clipped_and_counted(self):
        # lam*x + 1 <= 0 for x <= -2 at lam=0.5: R produced NaN, we clip.
        x = np.array([-5.0, -2.0, 0.0, 3.0])
        out, n_clipped = inv_boxcox(x, 0.5, return_nclipped=True)
        assert n_clipped == 2
        assert np.all(np.isfinite(out))
        assert np.all(out >= 0)

    def test_no_clip_in_normal_range(self, monthly_series):
        _, n_clipped = inv_boxcox(boxcox(monthly_series, 0.5), 0.5, return_nclipped=True)
        assert n_clipped == 0

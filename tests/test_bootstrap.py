import numpy as np
import pytest

from baggets.bootstrap import bld_mbb_bootstrap, moving_block_bootstrap


class TestMovingBlockBootstrap:
    def test_output_length(self):
        rng = np.random.default_rng(0)
        x = np.arange(100.0)
        assert moving_block_bootstrap(x, 24, rng).size == 100

    def test_values_come_from_source(self):
        rng = np.random.default_rng(1)
        x = np.random.default_rng(9).normal(size=50)
        out = moving_block_bootstrap(x, 8, rng)
        assert np.isin(out, x).all()

    def test_blocks_are_contiguous_source_slices(self):
        # With x = arange, contiguous source blocks are runs with diff == 1;
        # breaks can only occur at block boundaries (n//b + 2 blocks, plus offset).
        rng = np.random.default_rng(2)
        n, b = 96, 16
        x = np.arange(float(n))
        out = moving_block_bootstrap(x, b, rng)
        n_breaks = int(np.count_nonzero(np.diff(out) != 1.0))
        assert n_breaks <= n // b + 2

    def test_offset_and_starts_vary(self):
        x = np.arange(60.0)
        draws = {tuple(moving_block_bootstrap(x, 8, np.random.default_rng(s))) for s in range(20)}
        assert len(draws) == 20

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="too short"):
            moving_block_bootstrap(np.arange(5.0), 8, np.random.default_rng(0))

    def test_preserves_marginal_scale(self):
        rng = np.random.default_rng(3)
        x = np.random.default_rng(7).normal(0, 2.0, 200)
        pooled = np.concatenate(
            [moving_block_bootstrap(x, 16, rng) for _ in range(200)]
        )
        assert abs(pooled.std() - x.std()) / x.std() < 0.05


class TestBldMbbBootstrap:
    def test_shape_and_member_zero(self, monthly_series):
        res = bld_mbb_bootstrap(monthly_series, 25, 12, np.random.default_rng(0))
        assert res.series.shape == (25, monthly_series.size)
        np.testing.assert_array_equal(res.series[0], monthly_series)

    def test_deterministic_given_seed(self, monthly_series):
        a = bld_mbb_bootstrap(monthly_series, 10, 12, np.random.default_rng(42))
        b = bld_mbb_bootstrap(monthly_series, 10, 12, np.random.default_rng(42))
        np.testing.assert_array_equal(a.series, b.series)
        assert a.lam == b.lam

    def test_members_positive_for_positive_input(self, monthly_series):
        res = bld_mbb_bootstrap(monthly_series, 50, 12, np.random.default_rng(1))
        assert np.all(res.series > 0)

    def test_members_track_original(self, monthly_series):
        res = bld_mbb_bootstrap(monthly_series, 20, 12, np.random.default_rng(2))
        for member in res.series[1:]:
            assert np.corrcoef(member, monthly_series)[0, 1] > 0.9

    def test_nonseasonal_path(self, yearly_series):
        res = bld_mbb_bootstrap(yearly_series, 15, 1, np.random.default_rng(3))
        assert res.series.shape == (15, yearly_series.size)
        np.testing.assert_array_equal(res.series[0], yearly_series)
        assert np.all(np.isfinite(res.series))

    def test_single_bootstrap_is_original_only(self, monthly_series):
        res = bld_mbb_bootstrap(monthly_series, 1, 12, np.random.default_rng(4))
        assert res.series.shape == (1, monthly_series.size)
        assert res.lam == 1.0

    def test_rejects_nan(self):
        x = np.arange(100.0)
        x[3] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            bld_mbb_bootstrap(x, 5, 1, np.random.default_rng(0))

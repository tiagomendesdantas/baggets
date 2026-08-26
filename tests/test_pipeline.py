import numpy as np
import pytest

from baggets.datasets import SeriesRecord
from baggets.pipeline import (
    apply_strategy,
    load_series_artifacts,
    precompute_series,
    series_rng,
    strategy_rng,
)
from baggets.selection import NoSelection, TopKSelection


@pytest.fixture(scope="module")
def record():
    rng = np.random.default_rng(5)
    t = np.arange(90.0)
    y = (100 + t) * (1 + 0.15 * np.sin(2 * np.pi * t / 12)) * np.exp(rng.normal(0, 0.03, 90))
    return SeriesRecord(uid="TEST1", y_train=y[:72], y_test=y[72:], season_length=12)


@pytest.fixture(scope="module")
def cached(record, tmp_path_factory):
    path = tmp_path_factory.mktemp("cache") / "TEST1.npz"
    meta = precompute_series(record, n_bootstraps=30, root_seed=42, out_path=path)
    return path, meta


class TestPrecompute:
    def test_meta(self, cached):
        _, meta = cached
        assert meta["validated"] is True
        assert meta["h_val"] == 24
        assert meta["uid"] == "TEST1"

    def test_roundtrip(self, cached, record):
        path, _ = cached
        art = load_series_artifacts(path)
        assert art.final_forecasts.shape == (30, record.horizon)
        assert art.validation is not None
        assert art.validation.series.shape[0] == 30
        # float32 storage round-trip of the training data
        np.testing.assert_allclose(art.y_train, record.y_train, rtol=1e-6)
        # member 0 is the original series
        np.testing.assert_allclose(art.validation.series[0], record.y_train, rtol=1e-6)

    def test_deterministic_per_uid(self, record, tmp_path):
        p1, p2 = tmp_path / "a.npz", tmp_path / "b.npz"
        precompute_series(record, 10, 42, p1)
        precompute_series(record, 10, 42, p2)
        a, b = load_series_artifacts(p1), load_series_artifacts(p2)
        np.testing.assert_array_equal(a.final_forecasts, b.final_forecasts)

    def test_series_rng_differs_by_uid(self):
        a = series_rng(42, "N0001").integers(0, 1_000_000, 5)
        b = series_rng(42, "N0002").integers(0, 1_000_000, 5)
        assert not np.array_equal(a, b)


class TestApplyStrategy:
    def test_topk_median(self, cached):
        path, _ = cached
        art = load_series_artifacts(path)
        rng = strategy_rng(1, art.uid, "topk")
        point, info = apply_strategy(art, TopKSelection(), 8, lambda e: np.median(e, axis=0), rng)
        assert point.shape == art.y_test.shape
        assert info["n_members"] == 8
        # equals the median of the 8 lowest-validation-error members' forecasts
        idx = np.sort(np.argsort(art.validation.val_errors, kind="stable")[:8])
        np.testing.assert_allclose(point, np.median(art.final_forecasts[idx], axis=0))

    def test_none_all_members(self, cached):
        path, _ = cached
        art = load_series_artifacts(path)
        rng = strategy_rng(1, art.uid, "none")
        point, info = apply_strategy(art, NoSelection(), 8, lambda e: np.mean(e, axis=0), rng)
        assert info["n_members"] == 30
        np.testing.assert_allclose(point, art.final_forecasts.mean(axis=0))

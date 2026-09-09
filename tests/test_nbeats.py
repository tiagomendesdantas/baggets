"""N-BEATS engine: protocol conformance, training behaviour, honest fallback.

These tests assert the engine is a drop-in for ``ETSEngine`` and that its
failure mode is the documented one (naive fallback on series too short to yield
training windows) rather than a crash or a fabricated number.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from baggets.engine import ForecastEngine

torch = pytest.importorskip("torch", reason="N-BEATS needs the [torch] extra")

from baggets.nbeats import NBeatsConfig, NBeatsEngine  # noqa: E402

FAST = dict(max_epochs=12, patience=5, width=32, blocks=2, seed=7)


@pytest.fixture
def seasonal_series() -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.arange(140)
    return 100 + 0.4 * t + 12 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 2, t.size)


class TestProtocol:
    def test_satisfies_forecast_engine(self):
        assert isinstance(NBeatsEngine(12), ForecastEngine)

    def test_importing_baggets_does_not_import_torch(self):
        """torch is an optional extra: the package must not pay for it on import."""
        code = "import sys, baggets; print('torch' in sys.modules)"
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert out.stdout.strip() == "False"


class TestForecast:
    @pytest.mark.parametrize("interpretable", [False, True])
    def test_shape_and_finiteness(self, seasonal_series, interpretable):
        fc = NBeatsEngine(12, NBeatsConfig(interpretable=interpretable, **FAST)).forecast(
            seasonal_series, 18
        )
        assert fc.mean.shape == (18,)
        assert np.all(np.isfinite(fc.mean))
        assert not fc.fallback

    def test_seed_makes_it_reproducible(self, seasonal_series):
        cfg = NBeatsConfig(**FAST)
        a = NBeatsEngine(12, cfg).forecast(seasonal_series, 12).mean
        b = NBeatsEngine(12, cfg).forecast(seasonal_series, 12).mean
        np.testing.assert_allclose(a, b)

    def test_predict_reuses_the_trained_horizon(self, seasonal_series):
        engine = NBeatsEngine(12, NBeatsConfig(**FAST))
        fitted = engine.fit(seasonal_series)
        first = engine.predict(fitted, 12)
        again = engine.predict(fitted, 12)
        np.testing.assert_allclose(first, again)
        assert list(fitted.trained) == [12]

    def test_beats_a_flat_forecast_on_a_clean_seasonal_series(self, seasonal_series):
        """A trend+seasonality series is the easy case; losing it would mean the
        training loop is broken, not that the method is weak."""
        train, test = seasonal_series[:-18], seasonal_series[-18:]
        fc = NBeatsEngine(12, NBeatsConfig(interpretable=True, max_epochs=200, patience=30,
                                           seed=7)).forecast(train, 18)
        naive = np.full(18, train[-1])
        assert np.mean(np.abs(fc.mean - test)) < np.mean(np.abs(naive - test))


class TestFallback:
    def test_series_too_short_falls_back_to_naive(self):
        y = np.linspace(10, 20, 30)
        fc = NBeatsEngine(12, NBeatsConfig(**FAST)).forecast(y, 18)
        assert fc.fallback
        assert fc.method == "naive-fallback"
        np.testing.assert_allclose(fc.mean, y[-1])

    def test_fallback_records_why(self):
        engine = NBeatsEngine(12, NBeatsConfig(**FAST))
        fitted = engine.fit(np.linspace(10, 20, 30))
        engine.predict(fitted, 18)
        assert "too short" in fitted.history[18]["reason"]


class TestTrainingHistory:
    """The curves the training probes (project 2) read."""

    def test_history_carries_the_curves_and_split(self, seasonal_series):
        engine = NBeatsEngine(12, NBeatsConfig(**FAST))
        fitted = engine.fit(seasonal_series)
        engine.predict(fitted, 18)
        h = fitted.history[18]
        assert len(h["train_curve"]) == len(h["val_curve"]) > 0
        assert h["n_train_windows"] + h["n_val_windows"] == h["n_windows"]
        assert 0 <= h["best_epoch"] < len(h["val_curve"])

    def test_lookback_shrinks_rather_than_failing(self):
        """A series that cannot afford multiplier*h gets a shorter lookback, not a fallback."""
        rng = np.random.default_rng(1)
        y = 50 + np.arange(70) * 0.3 + rng.normal(0, 1, 70)
        engine = NBeatsEngine(12, NBeatsConfig(lookback_multiplier=7, **FAST))
        fitted = engine.fit(y)
        engine.predict(fitted, 18)
        assert fitted.history[18]["lookback"] < 7 * 18


class TestPlugsIntoBagging:
    def test_bagged_ensemble_runs_with_the_neural_engine(self, seasonal_series):
        from baggets import BaggedETS, NoSelection

        model = BaggedETS(
            season_length=12,
            n_bootstraps=6,
            n_select=3,
            selection=NoSelection(3),
            aggregator="mean",
            random_state=0,
            engine=NBeatsEngine(12, NBeatsConfig(**FAST)),
        ).fit(seasonal_series[:120])
        fc = model.predict(18)
        assert fc.ensemble.shape == (3, 18)
        assert fc.point.shape == (18,)
        assert model.info_["engine"] == "NBeatsEngine"

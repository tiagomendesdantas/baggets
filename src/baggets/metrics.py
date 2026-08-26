"""Forecast accuracy metrics with the M3-competition conventions used in the paper."""

from __future__ import annotations

import numpy as np

__all__ = ["smape", "mape", "mase"]


def smape(y: np.ndarray, f: np.ndarray) -> float:
    """Symmetric MAPE, M3 convention: mean of ``200 * |y - f| / (|y| + |f|)`` (percent)."""
    y = np.asarray(y, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    return float(np.mean(200.0 * np.abs(y - f) / (np.abs(y) + np.abs(f))))


def mape(y: np.ndarray, f: np.ndarray) -> float:
    """Plain MAPE (percent) — the validation metric of the original R code."""
    y = np.asarray(y, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    return float(np.mean(100.0 * np.abs((y - f) / y)))


def mase(y_test: np.ndarray, f: np.ndarray, y_train: np.ndarray, season_length: int) -> float:
    """Mean Absolute Scaled Error (Hyndman & Koehler 2006).

    Scales test MAE by the in-sample one-step MAE of the seasonal naive method
    (lag ``season_length``; lag 1 for non-seasonal series).
    """
    y_test = np.asarray(y_test, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    m = max(1, season_length)
    scale = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    return float(np.mean(np.abs(y_test - f)) / scale)

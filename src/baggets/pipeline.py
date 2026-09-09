"""Two-stage experiment pipeline with a per-series forecast-tensor cache.

Selection strategies only diverge AFTER validation artifacts exist. Stage A
(``precompute_series``) therefore does all the expensive work once per series —
bootstrap the pool, fit/forecast every member on the validation window, fit/
forecast every member full-length — and caches the tensors. Stage B
(``apply_strategy``) turns any (strategy, aggregator, n_select, seed) into pure
array slicing: a full-M3 strategy sweep reruns in seconds with zero refits.
"""

from __future__ import annotations

import json
import os
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bootstrap import bld_mbb_bootstrap
from .datasets import SeriesRecord
from .engine import ETSEngine, ForecastEngine
from .selection import SelectionStrategy
from .validation import (
    MIN_H_VAL,
    ValidationArtifacts,
    build_validation_artifacts,
    default_h_val,
)

__all__ = [
    "SeriesArtifacts",
    "precompute_series",
    "load_series_artifacts",
    "apply_strategy",
    "series_rng",
    "strategy_rng",
]


def series_rng(root_seed: int, uid: str) -> np.random.Generator:
    """Deterministic per-series generator, independent of processing order."""
    return np.random.default_rng((root_seed, zlib.crc32(uid.encode())))


def strategy_rng(eval_seed: int, uid: str, strategy_name: str) -> np.random.Generator:
    """Deterministic generator for one (strategy, seed, series) evaluation."""
    return np.random.default_rng(
        (eval_seed, zlib.crc32(uid.encode()), zlib.crc32(strategy_name.encode()))
    )


@dataclass(frozen=True)
class SeriesArtifacts:
    """Stage-A output for one series: validation artifacts + all-member final forecasts."""

    uid: str
    validation: ValidationArtifacts | None
    """None when the series was too short for a validation window."""
    final_forecasts: np.ndarray
    """(B, h) forecasts from every member fit on its full length."""
    y_test: np.ndarray
    y_train: np.ndarray
    season_length: int
    meta: dict


def precompute_series(
    rec: SeriesRecord,
    n_bootstraps: int,
    root_seed: int,
    out_path: Path,
    engine: ForecastEngine | None = None,
) -> dict:
    """Stage A for one series; writes ``out_path`` (.npz) and returns summary stats.

    ``engine`` defaults to ``ETSEngine``; pass another base learner to cache its
    forecast tensors instead. Stage B is engine-agnostic — it only slices the
    cached tensors — so a strategy sweep runs identically over either.
    """
    rng = series_rng(root_seed, rec.uid)
    m = rec.season_length
    y = rec.y_train
    h_final = rec.horizon

    boot = bld_mbb_bootstrap(y, n_bootstraps, m, rng)
    engine = engine if engine is not None else ETSEngine(m)

    h_val = default_h_val(y.size, m)
    validated = h_val >= MIN_H_VAL(m)
    if validated:
        artifacts = build_validation_artifacts(
            boot.series, y, m, h_val, validation_metric="mape", n_jobs=1,
            meta={"lambda": boot.lam, "n_clipped": boot.n_clipped},
            engine=engine,
        )
        val_forecasts = artifacts.val_forecasts
        val_errors = artifacts.val_errors
        n_val_fallbacks = artifacts.meta["n_val_fallbacks"]
    else:
        val_forecasts = np.zeros((n_bootstraps, 0))
        val_errors = np.full(n_bootstraps, np.nan)
        n_val_fallbacks = 0

    final_forecasts = np.empty((n_bootstraps, h_final))
    n_final_fallbacks = 0
    for i, member in enumerate(boot.series):
        fc = engine.forecast(member, h_final)
        final_forecasts[i] = fc.mean
        n_final_fallbacks += fc.fallback

    meta = {
        "uid": rec.uid,
        "season_length": m,
        "h_val": h_val if validated else None,
        "validated": validated,
        "lambda": boot.lam,
        "n_clipped": boot.n_clipped,
        "n_val_fallbacks": int(n_val_fallbacks),
        "n_final_fallbacks": int(n_final_fallbacks),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # atomic write: a kill mid-save must not leave a truncated file that the
    # resume logic would treat as a finished series
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    np.savez_compressed(
        tmp_path,
        series=boot.series.astype(np.float32),
        val_forecasts=val_forecasts.astype(np.float32),
        val_errors=val_errors.astype(np.float32),
        final_forecasts=final_forecasts.astype(np.float32),
        y_train=y.astype(np.float32),
        y_test=rec.y_test.astype(np.float32),
        meta=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8),
    )
    # np.savez appends .npz to names lacking it
    saved_tmp = tmp_path if tmp_path.exists() else tmp_path.with_name(tmp_path.name + ".npz")
    os.replace(saved_tmp, out_path)
    return meta


def load_series_artifacts(path: Path) -> SeriesArtifacts:
    """Load one Stage-A cache file back into memory (float64 views for math)."""
    with np.load(path) as z:
        meta = json.loads(bytes(z["meta"]).decode())
        series = z["series"].astype(np.float64)
        y_train = z["y_train"].astype(np.float64)
        y_test = z["y_test"].astype(np.float64)
        final_forecasts = z["final_forecasts"].astype(np.float64)
        val_forecasts = z["val_forecasts"].astype(np.float64)
        val_errors = z["val_errors"].astype(np.float64)

    m = int(meta["season_length"])
    validation = None
    if meta["validated"]:
        h_val = int(meta["h_val"])
        validation = ValidationArtifacts(
            series=series,
            val_forecasts=val_forecasts,
            val_actuals=y_train[y_train.size - h_val :],
            val_errors=val_errors,
            season_length=m,
            meta=dict(meta),
        )
    return SeriesArtifacts(
        uid=meta["uid"],
        validation=validation,
        final_forecasts=final_forecasts,
        y_test=y_test,
        y_train=y_train,
        season_length=m,
        meta=meta,
    )


def apply_strategy(
    art: SeriesArtifacts,
    strategy: SelectionStrategy,
    n_select: int,
    aggregator,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    """Stage B: select members, aggregate their cached final forecasts.

    Returns ``(point_forecast, info)``. Series without a validation window fall
    back to a random subset (mirrors BaggedETS.fit)."""
    B = art.final_forecasts.shape[0]
    if art.validation is None:
        idx = np.sort(rng.choice(B, size=min(n_select, B), replace=False))
        info = {"selection": "fallback_random", "n_members": int(idx.size)}
    else:
        idx = strategy.select(art.validation, n_select, rng)
        info = {"selection": strategy.name, "n_members": int(idx.size)}
        if "k_chosen" in art.validation.meta:
            info["k_chosen"] = art.validation.meta.pop("k_chosen")
    point = np.asarray(aggregator(art.final_forecasts[idx]))
    return point, info

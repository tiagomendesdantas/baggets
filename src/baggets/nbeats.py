"""N-BEATS engine: a deep residual forecaster behind the same wrapper as ETS.

Implements the architecture of Oreshkin, Carpov, Chapados & Bengio (2020),
*N-BEATS: Neural basis expansion analysis for interpretable time series
forecasting* (ICLR), in PyTorch, exposing the ``ETSEngine`` protocol so the
bagging machinery, selection strategies, metrics and M3 harness are reused
unchanged.

Two deliberate departures from the paper, both because this engine is a **base
learner inside a per-series bootstrap ensemble**, not the paper's setup:

1. **Per-series, not cross-learning.** ``fit(y)`` receives one series, so each
   model sees only that series' windows. The published M3/M4 results come from
   *cross-learning* — one model over all series — which is a different regime
   with orders of magnitude more data. Expect worse numbers here; that is the
   honest comparison this engine exists to make, not a bug to hide.
2. **Depth and width scaled to the data.** The paper's generic stack is 30
   blocks of width 512 trained on 100k series. M3 monthly series are 48-144
   points, which yields tens of training windows. Defaults here are much
   smaller; ``blocks``/``width`` expose the paper's sizes for anyone with the
   data to use them.

torch is an optional dependency (``pip install baggets[torch]``). Importing this
module without it raises with an actionable message; importing ``baggets`` does
not import this module.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .engine import EngineForecast

__all__ = ["NBeatsEngine", "FittedNBeats", "NBeatsConfig"]

_TORCH_HINT = (
    "N-BEATS needs PyTorch, which is an optional extra of baggets.\n"
    "Install it with:  uv sync --extra torch   (or: pip install 'baggets[torch]')"
)


def _torch():
    """Import torch lazily so the package works without it."""
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via message
        raise ModuleNotFoundError(_TORCH_HINT) from exc
    return torch


# ---------------------------------------------------------------- config


@dataclass(frozen=True)
class NBeatsConfig:
    """Architecture and training knobs.

    ``interpretable=True`` builds the paper's two-stack trend + seasonality model
    whose basis expansions are directly readable (that is what makes "what did
    the network learn" a plot rather than a metaphor). ``False`` builds the
    generic stack with a learned basis.
    """

    interpretable: bool = False
    blocks: int = 3
    """Blocks per stack (paper: 3 interpretable, 30 generic)."""
    width: int = 128
    """Hidden width of each block's 4-layer ReLU trunk (paper: 512 / 256)."""
    layers: int = 4
    """Fully-connected ReLU layers per block (paper: 4)."""
    trend_degree: int = 2
    """Polynomial degree of the trend basis (interpretable only)."""
    seasonality_harmonics: int | None = None
    """Fourier harmonics; defaults to ``season_length // 2`` (interpretable only)."""
    lookback_multiplier: int = 3
    """Target lookback as a multiple of the horizon (paper sweeps 2-7)."""
    min_windows: int = 8
    """Refuse to shrink the lookback below what yields this many training windows."""
    max_epochs: int = 300
    patience: int = 30
    """Early stopping patience, in epochs, on the held-out validation windows."""
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    loss: Literal["smape", "mape", "mse"] = "smape"
    val_fraction: float = 0.2
    """Fraction of the most recent windows held out for early stopping."""
    device: str = "cpu"
    seed: int | None = None


# ---------------------------------------------------------------- fitted state


@dataclass
class FittedNBeats:
    """Result of ``NBeatsEngine.fit``.

    The network is horizon-shaped, but the engine protocol only learns ``h`` at
    ``predict`` time, so training is deferred and memoised per horizon in
    ``trained``. ``method`` and ``fallback`` mirror ``FittedETS`` so callers that
    only read those two fields work with either engine.
    """

    y: np.ndarray
    config: NBeatsConfig
    season_length: int
    last_value: float
    method: str
    fallback: bool
    trained: dict[int, Any] = field(default_factory=dict)
    """h -> (module, lookback, scale_mode) for horizons already trained."""
    history: dict[int, dict] = field(default_factory=dict)
    """h -> training curves and stopping info; consumed by diagnostics/plots."""
    monitors: dict[int, Any] = field(default_factory=dict)
    """h -> the monitor that watched training, when a monitor_factory was given."""


# ---------------------------------------------------------------- basis blocks


def _build_modules():
    """Define the nn.Modules. Deferred so importing this file never needs torch."""
    torch = _torch()
    nn = torch.nn

    class _Trunk(nn.Module):
        """The 4-layer ReLU trunk shared by every block flavour."""

        def __init__(self, lookback: int, width: int, layers: int):
            super().__init__()
            dims = [lookback] + [width] * layers
            self.net = nn.Sequential(
                *[
                    layer
                    for i in range(layers)
                    for layer in (nn.Linear(dims[i], dims[i + 1]), nn.ReLU())
                ]
            )

        def forward(self, x):
            return self.net(x)

    class _GenericBlock(nn.Module):
        """Learned basis: theta straight to backcast/forecast through a linear map."""

        def __init__(self, lookback: int, horizon: int, width: int, layers: int, theta: int):
            super().__init__()
            self.trunk = _Trunk(lookback, width, layers)
            self.theta_b = nn.Linear(width, theta)
            self.theta_f = nn.Linear(width, theta)
            self.basis_b = nn.Linear(theta, lookback, bias=False)
            self.basis_f = nn.Linear(theta, horizon, bias=False)

        def forward(self, x):
            h = self.trunk(x)
            return self.basis_b(self.theta_b(h)), self.basis_f(self.theta_f(h))

    class _BasisBlock(nn.Module):
        """Fixed basis (polynomial trend or Fourier seasonality); theta are its coefficients."""

        def __init__(self, lookback: int, width: int, layers: int, basis_b, basis_f):
            super().__init__()
            self.trunk = _Trunk(lookback, width, layers)
            self.theta_b = nn.Linear(width, basis_b.shape[0])
            self.theta_f = nn.Linear(width, basis_f.shape[0])
            self.register_buffer("basis_b", basis_b)
            self.register_buffer("basis_f", basis_f)

        def forward(self, x):
            h = self.trunk(x)
            return self.theta_b(h) @ self.basis_b, self.theta_f(h) @ self.basis_f

    class NBeats(nn.Module):
        """Doubly residual stack: each block subtracts its backcast, forecasts sum."""

        def __init__(self, blocks):
            super().__init__()
            self.blocks = nn.ModuleList(blocks)

        def forward(self, x, per_block: bool = False):
            residual = x
            total = None
            parts = []
            for block in self.blocks:
                backcast, forecast = block(residual)
                residual = residual - backcast
                total = forecast if total is None else total + forecast
                if per_block:
                    parts.append(forecast)
            return (total, parts) if per_block else total

    return torch, nn, _GenericBlock, _BasisBlock, NBeats


def _trend_basis(torch, degree: int, lookback: int, horizon: int):
    """Polynomial basis on normalised time, as in the paper's trend stack."""
    tb = torch.stack([torch.linspace(-1, 0, lookback) ** p for p in range(degree + 1)])
    tf = torch.stack([torch.linspace(0, 1, horizon) ** p for p in range(degree + 1)])
    return tb.float(), tf.float()


def _seasonality_basis(torch, harmonics: int, lookback: int, horizon: int):
    """Fourier basis (cos and sin pairs), as in the paper's seasonality stack."""

    def _grid(n, start, stop):
        t = torch.linspace(start, stop, n)
        rows = [torch.ones(n)]
        for k in range(1, harmonics + 1):
            rows.append(torch.cos(2 * np.pi * k * t))
            rows.append(torch.sin(2 * np.pi * k * t))
        return torch.stack(rows).float()

    return _grid(lookback, -1, 0), _grid(horizon, 0, 1)


# ---------------------------------------------------------------- windowing


def _choose_lookback(n: int, horizon: int, cfg: NBeatsConfig) -> int | None:
    """Largest lookback <= multiplier*h that still leaves ``min_windows`` windows.

    Returns None when the series cannot yield enough windows at any lookback —
    the caller then falls back, exactly as ``ETSEngine`` does on a failed fit.
    Short M3 series hit this often, and that is information, not an error.
    """
    target = cfg.lookback_multiplier * horizon
    upper = n - horizon - cfg.min_windows + 1
    lookback = min(target, upper)
    return lookback if lookback >= horizon else None


def _windows(y: np.ndarray, lookback: int, horizon: int):
    """Sliding (x, target) pairs over one series, oldest first."""
    n_win = y.size - lookback - horizon + 1
    idx = np.arange(n_win)[:, None]
    x = y[idx + np.arange(lookback)]
    t = y[idx + lookback + np.arange(horizon)]
    return x, t


def _scale(x: np.ndarray) -> np.ndarray:
    """Per-window level scale: mean of the lookback, guarded away from zero.

    The paper divides each window by a statistic of its own history so one
    network sees comparable magnitudes. Here it also keeps a single series'
    windows on a common scale across a trend.
    """
    s = np.abs(x).mean(axis=-1, keepdims=True)
    return np.where(s < 1e-8, 1.0, s)


# ---------------------------------------------------------------- engine


class NBeatsEngine:
    """N-BEATS fit + forecast behind the ``ETSEngine`` protocol.

    ``fit`` is cheap (it stores the series); the network is horizon-shaped and
    the protocol supplies ``h`` at ``predict`` time, so training happens on first
    use of a horizon and is cached on the fitted object. ``forecast(y, h)`` — the
    call the validation and pipeline stages actually use — therefore does the
    full fit-and-train in one go.
    """

    def __init__(self, season_length: int, config: NBeatsConfig | None = None,
                 monitor_factory: Any = None, **kwargs):
        """``monitor_factory(module) -> monitor`` instruments the training loop.

        Duck-typed on purpose: the monitor only needs an ``epoch()`` context
        manager yielding an object with ``before_backward()`` and ``record()``.
        ``torch-training-probes`` satisfies it, and baggets does not depend on
        it — anything with that shape works, including a two-line stub.
        Monitors are kept on the fitted object under ``monitors[h]``.
        """
        self.season_length = int(season_length)
        base = config or NBeatsConfig()
        self.config = NBeatsConfig(**{**base.__dict__, **kwargs}) if kwargs else base
        self.monitor_factory = monitor_factory

    # -------------------------------------------------------------- protocol
    def fit(self, y: np.ndarray) -> FittedNBeats:
        y = np.ascontiguousarray(y, dtype=np.float64)
        return FittedNBeats(
            y=y,
            config=self.config,
            season_length=self.season_length,
            last_value=float(y[-1]),
            method=f"N-BEATS({'interpretable' if self.config.interpretable else 'generic'})",
            fallback=False,
        )

    def predict(self, fitted: FittedNBeats, h: int) -> np.ndarray:
        h = int(h)
        if h not in fitted.trained:
            self._train(fitted, h)
        entry = fitted.trained[h]
        if entry is None:
            fitted.fallback = True
            fitted.method = "naive-fallback"
            return np.full(h, fitted.last_value, dtype=np.float64)

        torch = _torch()
        module, lookback = entry
        x = fitted.y[-lookback:][None, :]
        scale = _scale(x)
        with torch.no_grad():
            out = module(torch.as_tensor(x / scale, dtype=torch.float32)).numpy()
        mean = (out * scale)[0].astype(np.float64)
        if not np.all(np.isfinite(mean)):
            fitted.fallback = True
            return np.full(h, fitted.last_value, dtype=np.float64)
        return mean

    def forecast(self, y: np.ndarray, h: int, with_fitted: bool = False) -> EngineForecast:
        """One-shot fit + train + ``h``-step forecast."""
        fitted = self.fit(y)
        mean = self.predict(fitted, h)
        fitted_values = None
        if with_fitted:
            fitted_values = self._in_sample(fitted, h)
        return EngineForecast(mean, fitted_values, fitted.method, fitted.fallback)

    # -------------------------------------------------------------- internals
    def _in_sample(self, fitted: FittedNBeats, h: int) -> np.ndarray:
        """In-sample values as the *backcast* of the final window.

        N-BEATS has no recursive one-step filter like ETS, so there is no
        equivalent of ``predict_in_sample``. The backcast is the model's
        reconstruction of what it looked at; everything earlier is NaN rather
        than fabricated.
        """
        out = np.full(fitted.y.size, np.nan)
        entry = fitted.trained.get(h)
        if entry is None:
            return out
        torch = _torch()
        module, lookback = entry
        x = fitted.y[-lookback:][None, :]
        scale = _scale(x)
        with torch.no_grad():
            residual = torch.as_tensor(x / scale, dtype=torch.float32)
            recon = torch.zeros_like(residual)
            for block in module.blocks:
                backcast, _ = block(residual)
                residual = residual - backcast
                recon = recon + backcast
        out[-lookback:] = (recon.numpy() * scale)[0]
        return out

    def _build(self, lookback: int, horizon: int):
        torch, nn, generic, basis_block, NBeats = _build_modules()
        cfg = self.config
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)

        if cfg.interpretable:
            harmonics = cfg.seasonality_harmonics or max(1, self.season_length // 2)
            tb, tf = _trend_basis(torch, cfg.trend_degree, lookback, horizon)
            sb, sf = _seasonality_basis(torch, harmonics, lookback, horizon)
            blocks = [
                basis_block(lookback, cfg.width, cfg.layers, tb, tf) for _ in range(cfg.blocks)
            ] + [
                basis_block(lookback, cfg.width, cfg.layers, sb, sf) for _ in range(cfg.blocks)
            ]
        else:
            theta = max(4, horizon // 2)
            blocks = [
                generic(lookback, horizon, cfg.width, cfg.layers, theta) for _ in range(cfg.blocks)
            ]
        return NBeats(blocks)

    def _train(self, fitted: FittedNBeats, h: int) -> None:
        """Train a network for horizon ``h``; store ``None`` when the series is too short."""
        cfg = self.config
        y = fitted.y
        lookback = _choose_lookback(y.size, h, cfg)
        if lookback is None:
            fitted.trained[h] = None
            fitted.history[h] = {"reason": "series too short for a training window"}
            return

        torch = _torch()
        x, t = _windows(y, lookback, h)
        scale = _scale(x)
        x, t = x / scale, t / scale

        n_val = max(1, int(round(x.shape[0] * cfg.val_fraction)))
        n_val = min(n_val, x.shape[0] - 1)
        x_tr, t_tr = x[:-n_val], t[:-n_val]
        x_va, t_va = x[-n_val:], t[-n_val:]

        device = torch.device(cfg.device)
        module = self._build(lookback, h).to(device)
        opt = torch.optim.Adam(
            module.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        loss_fn = _loss_fn(torch, cfg.loss)

        to_t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)  # noqa: E731
        x_tr_t, t_tr_t, x_va_t, t_va_t = to_t(x_tr), to_t(t_tr), to_t(x_va), to_t(t_va)

        best_state, best_val, best_epoch = None, float("inf"), -1
        train_curve, val_curve = [], []
        n = x_tr_t.shape[0]
        gen = torch.Generator().manual_seed(cfg.seed if cfg.seed is not None else 0)

        monitor = self.monitor_factory(module) if self.monitor_factory is not None else None

        for epoch in range(cfg.max_epochs):
            with _maybe_epoch(monitor) as ep:
                module.train()
                perm = torch.randperm(n, generator=gen)
                epoch_loss = 0.0
                for start in range(0, n, cfg.batch_size):
                    sel = perm[start : start + cfg.batch_size]
                    xb = ep.watch(x_tr_t[sel]) if ep is not None else x_tr_t[sel]
                    opt.zero_grad()
                    loss = loss_fn(module(xb), t_tr_t[sel])
                    if ep is not None:
                        ep.before_backward()
                    loss.backward()
                    opt.step()
                    epoch_loss += float(loss.detach()) * sel.numel()
                train_curve.append(epoch_loss / n)

                module.eval()
                with torch.no_grad():
                    val = float(loss_fn(module(x_va_t), t_va_t))
                val_curve.append(val)
                if ep is not None:
                    ep.record(train_loss=train_curve[-1], val_loss=val)

            if val < best_val - 1e-9:
                best_val, best_epoch = val, epoch
                best_state = {k: v.detach().clone() for k, v in module.state_dict().items()}
            elif epoch - best_epoch >= cfg.patience:
                break

        if best_state is not None:
            module.load_state_dict(best_state)
        module.eval()
        if monitor is not None:
            monitor.detach()
            fitted.monitors[h] = monitor
        fitted.trained[h] = (module, lookback)
        fitted.history[h] = {
            "lookback": lookback,
            "n_windows": int(x.shape[0]),
            "n_train_windows": int(x_tr.shape[0]),
            "n_val_windows": int(x_va.shape[0]),
            "train_curve": train_curve,
            "val_curve": val_curve,
            "best_epoch": best_epoch,
            "best_val": best_val,
            "stopped_early": len(train_curve) < cfg.max_epochs,
        }


@contextmanager
def _maybe_epoch(monitor):
    """Yield the monitor's epoch handle, or None when nothing is watching."""
    if monitor is None:
        yield None
    else:
        with monitor.epoch() as ep:
            yield ep


def _loss_fn(torch, name: str):
    """Differentiable training losses. sMAPE matches the M3/M4 evaluation metric."""
    if name == "mse":
        return torch.nn.functional.mse_loss

    def smape(pred, target):
        denom = (pred.abs() + target.abs()).clamp_min(1e-8)
        return (200.0 * (pred - target).abs() / denom).mean()

    def mape(pred, target):
        return (100.0 * (pred - target).abs() / target.abs().clamp_min(1e-8)).mean()

    if name == "smape":
        return smape
    if name == "mape":
        return mape
    raise ValueError(f"unknown loss {name!r}; use 'smape', 'mape' or 'mse'")

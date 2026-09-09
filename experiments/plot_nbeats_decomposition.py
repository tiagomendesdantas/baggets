"""Plot what the interpretable N-BEATS actually learned, per basis.

The interpretable configuration constrains each block to a fixed basis — a
polynomial for trend, a Fourier series for seasonality — so the forecast can be
read as a sum of named parts rather than described as one. That is the whole
claim behind "interpretable", and this makes it checkable on a real series.

    uv run --extra torch --extra experiments python \
        experiments/plot_nbeats_decomposition.py --uid N1402
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from baggets.datasets import load_m3  # noqa: E402
from baggets.nbeats import NBeatsConfig, NBeatsEngine, _scale  # noqa: E402


def block_contributions(fitted, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-block forecast contributions, on the original scale."""
    import torch

    module, lookback = fitted.trained[h]
    x = fitted.y[-lookback:][None, :]
    scale = _scale(x)
    with torch.no_grad():
        total, parts = module(torch.as_tensor(x / scale, dtype=torch.float32), per_block=True)
    return (total.numpy() * scale)[0], np.stack([(p.numpy() * scale)[0] for p in parts])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--uid", default=None)
    ap.add_argument("--horizon", type=int, default=18)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--out", default=str(ROOT / "results" / "nbeats-decomposition.png"))
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = load_m3("monthly", ROOT / "data" / "m3")
    rec = next((r for r in records if r.uid == args.uid), records[0]) if args.uid else records[0]

    cfg = NBeatsConfig(interpretable=True, max_epochs=args.epochs,
                       patience=args.epochs // 8, seed=7)
    engine = NBeatsEngine(rec.season_length, cfg)
    fitted = engine.fit(rec.y_train)
    point = engine.predict(fitted, args.horizon)
    if fitted.trained.get(args.horizon) is None:
        print(f"{rec.uid}: too short to train — nothing to decompose")
        return 1

    total, parts = block_contributions(fitted, args.horizon)
    n_trend = cfg.blocks  # first half of the stack is the trend basis
    trend = parts[:n_trend].sum(axis=0)
    seasonal = parts[n_trend:].sum(axis=0)

    truth = rec.y_test[: args.horizon]
    smape = float(np.mean(200 * np.abs(point - truth) / (np.abs(point) + np.abs(truth))))
    hist = fitted.history[args.horizon]

    t_hist = np.arange(-rec.y_train.size, 0)
    t_fc = np.arange(args.horizon)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6.4), height_ratios=[2, 1.2], sharex=True)
    ax = axes[0]
    ax.plot(t_hist, rec.y_train, color="0.45", lw=1, label="history")
    ax.plot(t_fc, truth, color="k", lw=1.6, marker=".", label="held-out truth")
    ax.plot(t_fc, point, color="C3", lw=1.8, label=f"N-BEATS forecast (sMAPE {smape:.1f})")
    ax.axvline(-0.5, color="0.8", lw=1)
    ax.set_title(f"M3 {rec.uid} — interpretable N-BEATS, trained on {hist['n_windows']} windows")
    ax.set_ylabel("value")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(t_fc, trend, color="C0", lw=1.8, label="trend stack (polynomial basis)")
    ax.plot(t_fc, seasonal, color="C1", lw=1.8, label="seasonality stack (Fourier basis)")
    ax.plot(t_fc, total, color="C3", lw=1, ls="--", label="sum = forecast")
    ax.axhline(0, color="0.8", lw=1)
    ax.set_xlabel("months ahead")
    ax.set_ylabel("contribution")
    ax.set_title("what each basis contributed")
    ax.legend(fontsize=8)

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")

    print(f"{rec.uid}: sMAPE {smape:.2f}, {hist['n_windows']} windows, "
          f"best epoch {hist['best_epoch']}/{len(hist['train_curve'])}")
    print(f"trend contributes {np.abs(trend).mean():.1f} on average, "
          f"seasonality {np.abs(seasonal).mean():.1f}")
    print(f"figure: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

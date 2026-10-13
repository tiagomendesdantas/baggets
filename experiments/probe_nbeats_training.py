"""Watch one N-BEATS member train: dead ReLU units per layer, and where validation bottoms.

``NBeatsEngine(monitor_factory=...)`` hands each training run to a monitor you
supply. ``DeadReLUMonitor`` below is a complete one, and it needs nothing beyond
baggets and torch. Its defaults reproduce the two training numbers cited in
``reports/m8_nbeats.md`` for M3 monthly series N1402:

    Rscript validation_r/export_m3_mcomp.R      # once: writes data/m3/
    uv run --extra torch python experiments/probe_nbeats_training.py

Setup of that run: generic N-BEATS (the engine default), horizon 18, seed 7,
400 epochs with early stopping turned off (patience = epochs), so the whole
validation curve is recorded instead of being cut at its minimum.

What is measured:

- **Dead ReLU fraction, per layer.** A ReLU unit counts as dead in an epoch if
  its output was never positive for any input the network saw during that
  epoch (every training batch and the validation pass). A dead unit gets zero
  gradient from every input, so it cannot recover by itself. The fractions
  reported are those of the final epoch.
- **Validation minimum.** The epoch with the lowest validation loss, read from
  the engine's own ``history`` (epochs counted from 0). No monitor is needed
  for this one.
"""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from baggets.datasets import load_m3  # noqa: E402
from baggets.nbeats import NBeatsConfig, NBeatsEngine  # noqa: E402


class DeadReLUMonitor:
    """A minimal monitor for ``NBeatsEngine(monitor_factory=...)``.

    The engine calls ``monitor.epoch()`` once per epoch as a context manager;
    on the handle it yields, ``watch(x)`` and ``before_backward()`` once per
    batch and ``record(train_loss=..., val_loss=...)`` once per epoch; and
    ``monitor.detach()`` when training ends.
    """

    def __init__(self, module):
        import torch

        self._names = {}
        self._hooks = []
        for name, layer in module.named_modules():
            if isinstance(layer, torch.nn.ReLU):
                self._names[layer] = name
                self._hooks.append(layer.register_forward_hook(self._seen))
        self._alive = {}
        self.dead_fraction = []
        """One ``{layer name: fraction of dead units}`` per epoch."""

    def _seen(self, layer, _inputs, output):
        alive = (output.detach().reshape(-1, output.shape[-1]) > 0).any(dim=0)
        name = self._names[layer]
        self._alive[name] = alive if name not in self._alive else self._alive[name] | alive

    @contextmanager
    def epoch(self):
        self._alive = {}
        yield self
        self.dead_fraction.append(
            {name: 1.0 - float(alive.float().mean()) for name, alive in self._alive.items()}
        )

    # per-batch and per-epoch calls this monitor has no use for
    def watch(self, x):
        return x

    def before_backward(self):
        pass

    def record(self, train_loss=None, val_loss=None):
        pass

    def detach(self):
        for hook in self._hooks:
            hook.remove()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uid", default="N1402", help="M3 monthly series id")
    ap.add_argument("--horizon", type=int, default=18)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--interpretable", action="store_true",
                    help="interpretable N-BEATS instead of the generic default")
    args = ap.parse_args()

    records = load_m3("monthly", ROOT / "data" / "m3")
    rec = next((r for r in records if r.uid == args.uid), None)
    if rec is None:
        print(f"no M3 monthly series {args.uid!r}")
        return 1

    monitors: list[DeadReLUMonitor] = []

    def factory(module):
        monitors.append(DeadReLUMonitor(module))
        return monitors[-1]

    cfg = NBeatsConfig(interpretable=args.interpretable, max_epochs=args.epochs,
                       patience=args.epochs, seed=args.seed)
    engine = NBeatsEngine(rec.season_length, cfg, monitor_factory=factory)
    fitted = engine.fit(rec.y_train)
    engine.predict(fitted, args.horizon)
    hist = fitted.history[args.horizon]
    if "reason" in hist:
        print(f"{rec.uid}: {hist['reason']}")
        return 1

    kind = "interpretable" if args.interpretable else "generic"
    print(f"{rec.uid}: {rec.y_train.size} points, {kind} N-BEATS, lookback {hist['lookback']}, "
          f"{hist['n_windows']} windows ({hist['n_train_windows']} train / "
          f"{hist['n_val_windows']} validation)")

    final = monitors[-1].dead_fraction[-1]
    print(f"\ndead ReLU units in the final epoch ({len(hist['train_curve'])}):")
    for name, fraction in final.items():
        print(f"  {name:28s} {fraction:5.0%}")
    worst = max(final, key=final.get)
    print(f"worst layer: {worst} at {final[worst]:.0%}")
    print(f"validation loss bottomed at epoch {hist['best_epoch']} of "
          f"{len(hist['val_curve'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

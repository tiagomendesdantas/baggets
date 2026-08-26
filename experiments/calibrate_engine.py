"""M2 engine-calibration gate: plain AutoETS on the full M3 monthly set.

The paper's Table 5 ETS row (Mean sMAPE 14.135, Mean MASE 0.865) is a free,
deterministic anchor: reproducing it validates the ETS engine AND the data
pipeline before any bagging enters the picture. Gate: Mean sMAPE within ±0.15.

Run from the repo root (~1-2 minutes):

    uv run python experiments/calibrate_engine.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from baggets.datasets import load_m3  # noqa: E402
from baggets.engine import ETSEngine  # noqa: E402
from baggets.metrics import mase, smape  # noqa: E402

PAPER = {"mean_smape": 14.135, "median_smape": 9.073, "mean_mase": 0.865, "median_mase": 0.716}
GATE = 0.15


def main() -> int:
    records = load_m3("monthly", ROOT / "data" / "m3")
    engine = ETSEngine(12)

    t0 = time.time()
    smapes, mases, n_fallback = [], [], 0
    for rec in records:
        fc = engine.forecast(rec.y_train, rec.horizon)
        smapes.append(smape(rec.y_test, fc.mean))
        mases.append(mase(rec.y_test, fc.mean, rec.y_train, rec.season_length))
        n_fallback += fc.fallback
    elapsed = time.time() - t0

    got = {
        "mean_smape": float(np.mean(smapes)),
        "median_smape": float(np.median(smapes)),
        "mean_mase": float(np.mean(mases)),
        "median_mase": float(np.median(mases)),
    }

    print(f"{len(records)} series in {elapsed:.0f}s, fallbacks={n_fallback}")
    for k, v in got.items():
        print(f"  {k:14} = {v:7.3f}   (paper {PAPER[k]:7.3f}, delta {v - PAPER[k]:+.3f})")

    delta = abs(got["mean_smape"] - PAPER["mean_smape"])
    ok = delta <= GATE and n_fallback / len(records) < 0.005
    print(f"\nGATE {'PASSED' if ok else 'FAILED'}: |mean sMAPE delta| = {delta:.3f} (limit {GATE})")

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    with open(out / "engine_calibration.md", "w") as f:
        f.write("# Engine calibration (M2 gate)\n\n")
        f.write(f"Plain AutoETS, {len(records)} M3 monthly series, {elapsed:.0f}s, "
                f"{n_fallback} fallbacks.\n\n")
        f.write("| metric | baggets | paper (Table 5, ETS) | delta |\n|---|---|---|---|\n")
        for k, v in got.items():
            f.write(f"| {k} | {v:.3f} | {PAPER[k]:.3f} | {v - PAPER[k]:+.3f} |\n")
        f.write(f"\nGate (mean sMAPE ± {GATE}): **{'PASSED' if ok else 'FAILED'}**\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

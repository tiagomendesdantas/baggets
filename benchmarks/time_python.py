"""Time baggets on the benchmark series, under the conditions of time_r.R.

Same series (committed fixture training parts), same B, k fixed at 5 with the
paper's constants (``ClusterSelection.paper``), 100 members, median, one core,
one warm-up ETS fit excluded from the timing (it pays numba's JIT compile or
cache load, which a long run pays once).

Run from the repo root (B=1000 takes about 7 minutes on one core):

    uv run python benchmarks/time_python.py 1000
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsforecast  # noqa: E402

import baggets  # noqa: E402
from baggets import BaggedETS, ClusterSelection, ETSEngine  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "tests" / "fixtures" / "input"
SERIES = ["monthly_N1603", "monthly_N2244", "monthly_N1908"]


def load_y(sid: str) -> np.ndarray:
    return pd.read_csv(INPUT / f"{sid}.csv")["y"].to_numpy(np.float64)


def main() -> int:
    n_bootstraps = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    manifest = pd.read_csv(INPUT / "manifest.csv").set_index("sid")

    ETSEngine(12).fit(load_y(SERIES[0]))  # warm-up, not timed

    rows = []
    for sid in SERIES:
        y = load_y(sid)
        m, h = int(manifest.loc[sid, "m"]), int(manifest.loc[sid, "h"])
        t0, c0 = time.perf_counter(), time.process_time()
        model = BaggedETS(season_length=m, n_bootstraps=n_bootstraps, n_select=100,
                          selection=ClusterSelection.paper(n_clusters=5),
                          aggregator="median", random_state=42, n_jobs=1).fit(y)
        fc = model.predict(h)
        wall, cpu = time.perf_counter() - t0, time.process_time() - c0
        rows.append({"sid": sid, "n_train": y.size, "B": n_bootstraps,
                     "n_members": fc.ensemble.shape[0],
                     "wall_seconds": round(wall, 3), "cpu_seconds": round(cpu, 3)})
        print(f"{sid}: n={y.size}, B={n_bootstraps}, {fc.ensemble.shape[0]} members, "
              f"wall {wall:.1f}s, cpu {cpu:.1f}s", flush=True)

    out = pd.DataFrame(rows)
    path = ROOT / "benchmarks" / f"timings_python_B{n_bootstraps}.csv"
    out.to_csv(path, index=False)
    print(f"total wall {out['wall_seconds'].sum():.1f}s, cpu {out['cpu_seconds'].sum():.1f}s "
          f"-> {path.relative_to(ROOT)}")
    print(f"Python {platform.python_version()}, baggets {baggets.__version__}, "
          f"statsforecast {statsforecast.__version__}, numpy {np.__version__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

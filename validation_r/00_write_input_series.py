"""Write the fixture input series shared by Python and R.

Selects a deterministic, length-diverse set of M3 series (from the local Mcomp
export — run ``Rscript validation_r/export_m3_mcomp.R`` first) plus one
synthetic series, and writes each TRAINING part to
``tests/fixtures/input/<sid>.csv`` with a ``manifest.json`` describing season
length and horizons. Both the R fixture scripts and the Python tests read these
files, so the two sides always see identical data.

Run from the repo root:

    uv run python validation_r/00_write_input_series.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from baggets.datasets import load_m3  # noqa: E402

INPUT_DIR = ROOT / "tests" / "fixtures" / "input"

GROUPS = {
    "monthly": {"n_pick": 8},
    "quarterly": {"n_pick": 2},
    "yearly": {"n_pick": 1},
}


def pick_length_diverse(records, n_pick: int):
    """Deterministically pick records spread across the length distribution."""
    ordered = sorted(records, key=lambda r: (r.y_train.size, r.uid))
    idx = np.linspace(0, len(ordered) - 1, n_pick).round().astype(int)
    return [ordered[i] for i in idx]


def main() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}

    for group, cfg in GROUPS.items():
        records = load_m3(group, ROOT / "data" / "m3")
        for rec in pick_length_diverse(records, cfg["n_pick"]):
            sid = f"{group}_{rec.uid}"
            np.savetxt(INPUT_DIR / f"{sid}.csv", rec.y_train, fmt="%.10g", header="y", comments="")
            manifest[sid] = {
                "source": f"M3 {group} {rec.uid} (training part)",
                "m": rec.season_length,
                "h": rec.horizon,
                "n_train": int(rec.y_train.size),
            }
            print(f"{sid}: n_train={rec.y_train.size}")

    # Synthetic monthly series with known structure (trend + seasonal + AR(1) noise)
    rng = np.random.default_rng(20180748)  # IJF 34(4):748
    n = 120
    noise = np.empty(n)
    noise[0] = rng.normal(0, 2.0)
    for i in range(1, n):
        noise[i] = 0.6 * noise[i - 1] + rng.normal(0, 2.0)
    t = np.arange(n, dtype=np.float64)
    y = 200 + 1.5 * t + 25 * np.sin(2 * np.pi * t / 12) + 10 * np.cos(2 * np.pi * t / 6) + noise
    sid = "synthetic_ar1"
    np.savetxt(INPUT_DIR / f"{sid}.csv", y, fmt="%.10g", header="y", comments="")
    manifest[sid] = {
        "source": "synthetic trend+seasonal+AR(1), seed 20180748",
        "m": 12,
        "h": 18,
        "n_train": n,
    }
    print(f"{sid}: n_train={n}")

    # 30 monthly ids for the Tier-2 ETS engine reference (data stays in data/m3;
    # only the id list is shared so R and Python fit the identical series)
    monthly = load_m3("monthly", ROOT / "data" / "m3")
    ref = pick_length_diverse(monthly, 30)
    with open(INPUT_DIR / "ets_reference_ids.csv", "w") as f:
        f.write("uid\n")
        for rec in ref:
            f.write(f"{rec.uid}\n")
    print(f"ets_reference_ids: {len(ref)} monthly ids")

    # Tier-3 fixture: a deterministic distance matrix over bootstrapped versions
    # of one M3 series, for the PAM-vs-cluster::pam equivalence test
    from baggets.bootstrap import bld_mbb_bootstrap
    from baggets.clustering import euclidean_distance_matrix

    rec = next(r for r in monthly if r.uid == "N1861")
    boot = bld_mbb_bootstrap(rec.y_train, 120, 12, np.random.default_rng(1861))
    dist = euclidean_distance_matrix(boot.series)
    np.savetxt(INPUT_DIR / "pam_distance.csv", dist, delimiter=",", fmt="%.12g")
    print(f"pam_distance: {dist.shape[0]}x{dist.shape[1]} matrix")

    with open(INPUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    # CSV twin of the manifest so the R scripts don't need a JSON parser
    with open(INPUT_DIR / "manifest.csv", "w") as f:
        f.write("sid,m,h,n_train\n")
        for sid in sorted(manifest):
            e = manifest[sid]
            f.write(f"{sid},{e['m']},{e['h']},{e['n_train']}\n")
    print(f"\nWrote {len(manifest)} series to {INPUT_DIR}")


if __name__ == "__main__":
    main()

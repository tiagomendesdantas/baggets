"""Loading the M3 competition data from the local Mcomp export.

The canonical M3 source is R's Mcomp package — the same data the 2018 paper
used. ``validation_r/export_m3_mcomp.R`` dumps it to ``data/m3/*.csv`` once;
this module reads those files. (The datasetsforecast M3 loader is not used: its
upstream forecasters.org URL is dead.)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["SeriesRecord", "load_m3", "M3_GROUPS"]

M3_GROUPS = ("yearly", "quarterly", "monthly", "other")


@dataclass(frozen=True)
class SeriesRecord:
    uid: str
    y_train: np.ndarray
    y_test: np.ndarray
    season_length: int

    @property
    def horizon(self) -> int:
        return self.y_test.size


def load_m3(group: str, data_dir: str | Path = "data/m3") -> list[SeriesRecord]:
    """Load one M3 group ("yearly", "quarterly", "monthly", "other").

    ``data_dir`` is the directory written by ``validation_r/export_m3_mcomp.R``
    (default assumes running from the repo root).
    """
    if group not in M3_GROUPS:
        raise ValueError(f"unknown M3 group {group!r}; expected one of {M3_GROUPS}")
    data_dir = Path(data_dir)
    values_path = data_dir / f"m3_{group}.csv"
    meta_path = data_dir / "meta.csv"
    if not values_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"M3 export not found under {data_dir.resolve()} — generate it with: "
            "Rscript validation_r/export_m3_mcomp.R"
        )

    meta = pd.read_csv(meta_path).set_index("unique_id")
    values = pd.read_csv(values_path).sort_values(["unique_id", "idx"])

    records = []
    for uid, sdf in values.groupby("unique_id", sort=True):
        y = sdf["y"].to_numpy(dtype=np.float64)
        n_train = int(meta.loc[uid, "n_train"])
        records.append(
            SeriesRecord(
                uid=str(uid),
                y_train=y[:n_train],
                y_test=y[n_train:],
                season_length=int(meta.loc[uid, "m"]),
            )
        )
    return records

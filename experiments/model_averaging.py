"""Tier 0: is bagged ETS a Monte Carlo approximation to model-form averaging?

Bagging beats plain ETS by ~0.4pp on M3 monthly and nothing in M5/M6 explains
why. All 1000 members share one Box-Cox lambda and one STL ``trend + seasonal``
(bootstrap.py:74-81) — a weak perturbation, unless it is amplified through
something discrete. The discrete thing is AutoETS's AICc argmax over the 15
admissible ETS families, and the optimizer finding (tol change -> family flips
-> 1-4pp swings) shows how violently the pipeline responds to it.

Two experiments:

``families``  fit all 15 families on each *original* series and combine them by
              AICc weight / top-k. If likelihood-based model averaging recovers
              the bagging gain, 1000 fits collapse to ~15.
``entropy``   refit members recording the selected family, and test whether the
              per-series bagging gain lives where the family choice is unstable
              across bootstrap resamples. This is selection instability, which
              AICc weighting on a single sample cannot see.

Both write results/<run-id>/scores.parquet in run_m3.py's schema, so `run_m3.py
report` and `run_m3.py stats` work on them unchanged.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from joblib import Parallel, delayed  # noqa: E402
from statsforecast.models import AutoETS  # noqa: E402

# importing the engine installs the statsforecast optimizer patch; every fit
# below must run under the same tolerances as the cached stage-A fits
from baggets.datasets import load_m3  # noqa: E402
from baggets.engine import ETSEngine  # noqa: E402
from baggets.metrics import mase, smape  # noqa: E402
from baggets.pipeline import load_series_artifacts  # noqa: E402

CACHE_ROOT = ROOT / "data" / "cache"
RESULTS_ROOT = ROOT / "results"

# The 15 families AutoETS("ZZZ") actually searches: error x trend x season with
# additive errors barred from multiplicative seasonality (R's ets restriction,
# confirmed empirically -- A,*,M raises ValueError). Multiplicative trend is
# excluded from automatic search by both R and statsforecast.
FAMILIES = tuple(
    (e, t, s) for e in "AM" for t in ("N", "A", "Ad") for s in "NAM"
    if not (e == "A" and s == "M")
)


def fit_family_table(y: np.ndarray, season_length: int) -> pd.DataFrame:
    """Fit every admissible family to ``y``; one row per family that converged."""
    y = np.ascontiguousarray(y, dtype=np.float64)
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for e, t, s in FAMILIES:
            try:
                mod = AutoETS(
                    season_length=season_length, model=f"{e}{t[0]}{s}", damped=(t == "Ad")
                ).fit(y=y)
                m_ = mod.model_
                aicc = float(m_["aicc"])
                if not np.isfinite(aicc):
                    continue
                rows.append({
                    "family": f"{e},{t},{s}",
                    "method": str(m_["method"]),
                    "aicc": aicc,
                    "n_params": int(m_["n_params"]),
                    "sigma2": float(m_["sigma2"]),
                    "model": mod,
                })
            except Exception:
                continue
    return pd.DataFrame(rows)


def aicc_weights(aicc: np.ndarray) -> np.ndarray:
    """Akaike weights w_i ∝ exp(-Δ_i/2), Δ_i = aicc_i - min(aicc)."""
    delta = aicc - aicc.min()
    w = np.exp(-0.5 * delta)
    return w / w.sum()


# --------------------------------------------------------------- T0.1 / T0.3
def _families_one(rec, h: int) -> dict | None:
    tab = fit_family_table(rec.y_train, rec.season_length)
    if tab.empty:
        return None
    tab = tab.sort_values("aicc").reset_index(drop=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc = np.vstack([
            np.asarray(m.predict(h=h)["mean"], dtype=np.float64) for m in tab["model"]
        ])
    ok = np.all(np.isfinite(fc), axis=1)
    tab, fc = tab[ok].reset_index(drop=True), fc[ok]
    if not len(tab):
        return None

    aicc = tab["aicc"].to_numpy()
    w = aicc_weights(aicc)
    k = len(tab)

    # Full k-sweep so the choice of k is a reported curve, never a number picked
    # after seeing test scores; k is selected out-of-sample in the analysis.
    combos = {f"top{j}_mean": fc[: min(j, k)].mean(axis=0) for j in range(1, len(FAMILIES) + 1)}
    combos.update({
        "ets_best": fc[0],
        "aicc_weighted": w @ fc,
        "top5_median": np.median(fc[: min(5, k)], axis=0),
        "all_median": np.median(fc, axis=0),
    })
    scores = [
        {
            "uid": rec.uid,
            "strategy": name,
            "seed": 0,
            "smape": smape(rec.y_test, point),
            "mase": mase(rec.y_test, point, rec.y_train, rec.season_length),
            "n_members": k,
        }
        for name, point in combos.items()
    ]

    # T0.3: the pre-forecast routing proxies. delta_aicc is how decisively the
    # data prefers one family; n_eff_models is exp(entropy of Akaike weights).
    ent = float(-(w * np.log(w)).sum())
    diag = {
        "uid": rec.uid,
        "n_families": k,
        "best_method": tab.loc[0, "method"],
        "delta_aicc_12": float(aicc[1] - aicc[0]) if k > 1 else np.nan,
        "w_best": float(w[0]),
        "aicc_entropy": ent,
        "n_eff_models": float(np.exp(ent)),
        "n_train": int(rec.y_train.size),
        "season_length": int(rec.season_length),
    }
    fam = tab[["family", "method", "aicc", "n_params"]].copy()
    fam.insert(0, "uid", rec.uid)
    fam["weight"] = w
    return {"scores": scores, "diag": diag, "families": fam}


def cmd_families(args) -> int:
    records = load_m3(args.group, ROOT / "data" / "m3")
    if args.limit:
        records = records[: args.limit]
    print(f"{len(records)} series x {len(FAMILIES)} families", flush=True)

    t0 = time.time()
    out = Parallel(n_jobs=args.n_jobs, return_as="generator")(
        delayed(_families_one)(r, r.y_test.size) for r in records
    )
    scores, diags, fams = [], [], []
    for i, res in enumerate(out):
        if res is not None:
            scores.extend(res["scores"])
            diags.append(res["diag"])
            fams.append(res["families"])
        if (i + 1) % 200 == 0:
            print(f"[{i+1}/{len(records)}] {time.time()-t0:.0f}s", flush=True)

    out_dir = RESULTS_ROOT / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(scores).to_parquet(out_dir / "scores.parquet", index=False)
    pd.DataFrame(diags).to_parquet(out_dir / "diagnostics.parquet", index=False)
    pd.concat(fams, ignore_index=True).to_parquet(out_dir / "families.parquet", index=False)
    print(f"\n{len(scores)} rows in {time.time()-t0:.0f}s -> {out_dir}")
    return 0


# --------------------------------------------------------------------- T0.2
def _entropy_one(path: Path, n_members: int) -> dict | None:
    """Refit the first ``n_members`` cached member series, recording family only."""
    art = load_series_artifacts(path)
    if art.validation is None:  # no validation window -> pool not exposed on the dataclass
        with np.load(path) as z:
            pool = z["series"].astype(np.float64)
    else:
        pool = art.validation.series
    engine = ETSEngine(art.season_length)
    methods = [engine.fit(m).method for m in pool[:n_members]]
    counts = pd.Series(methods).value_counts()
    p = (counts / counts.sum()).to_numpy()
    ent = float(-(p * np.log(p)).sum())
    return {
        "uid": art.uid,
        "n_members_refit": len(methods),
        "n_distinct_families": int(counts.size),
        "modal_family": str(counts.index[0]),
        "modal_share": float(p[0]),
        "family_entropy": ent,
        "n_eff_families": float(np.exp(ent)),
        "member0_family": methods[0],
        "member0_is_modal": bool(methods[0] == counts.index[0]),
    }


def cmd_entropy(args) -> int:
    from run_m3 import make_run_key  # same cache key derivation

    run_key, _ = make_run_key(args.group, args.n_bootstraps, args.seed)
    cache_dir = CACHE_ROOT / args.group / run_key
    records = load_m3(args.group, ROOT / "data" / "m3")
    paths = [cache_dir / f"{r.uid}.npz" for r in records]
    paths = [p for p in paths if p.exists()]
    if not paths:
        print(f"no cache at {cache_dir}; run precompute first")
        return 1

    # Systematic sample over the uid ordering: deterministic, and independent of
    # anything correlated with the bagging effect we are about to test against.
    if args.n_series and args.n_series < len(paths):
        step = len(paths) / args.n_series
        paths = [paths[int(i * step)] for i in range(args.n_series)]
    print(f"{len(paths)} series x {args.n_members} members "
          f"= {len(paths)*args.n_members} ETS fits", flush=True)

    t0 = time.time()
    out = Parallel(n_jobs=args.n_jobs, return_as="generator")(
        delayed(_entropy_one)(p, args.n_members) for p in paths
    )
    rows = []
    for i, r in enumerate(out):
        if r is not None:
            rows.append(r)
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f"[{i+1}/{len(paths)}] {el:.0f}s, ETA {el/(i+1)*(len(paths)-i-1):.0f}s",
                  flush=True)

    out_dir = RESULTS_ROOT / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_dir / "family_entropy.parquet", index=False)
    print(f"\n{len(rows)} series in {(time.time()-t0)/60:.1f} min -> {out_dir}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("families", help="T0.1/T0.3: AICc model averaging vs the bag")
    f.add_argument("--group", default="monthly")
    f.add_argument("--limit", type=int)
    f.add_argument("--n-jobs", type=int, default=6)
    f.add_argument("--run-id", default="m7_aicc")
    f.set_defaults(func=cmd_families)

    e = sub.add_parser("entropy", help="T0.2: family instability across bootstrap members")
    e.add_argument("--group", default="monthly")
    e.add_argument("--n-series", type=int, default=300)
    e.add_argument("--n-members", type=int, default=100)
    e.add_argument("--n-bootstraps", type=int, default=1000)
    e.add_argument("--seed", type=int, default=42)
    e.add_argument("--n-jobs", type=int, default=6)
    e.add_argument("--run-id", default="m7_entropy")
    e.set_defaults(func=cmd_entropy)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

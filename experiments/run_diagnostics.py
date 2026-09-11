"""Cache-only diagnostics over the stage-A tensors (M7 Tier 1).

Every diagnostic here is a pure function of what stage A already wrote, so a
full 1428-series pass costs one decompression sweep and no ETS fits.

``ablate``  T1, the member-0 gate. bootstrap.py:70 sets ``series[0] = x`` and
            NoSelection.select returns ``arange(n)`` (selection.py:339), so
            every ``noneN`` result -- including the M6 champions -- contains the
            plain-ETS forecast of the *un-bootstrapped* series at 1/n weight.
            This re-scores those configurations against member-0-free random
            subsets, so the reported effect is bagging and not that free anchor.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from joblib import Parallel, delayed  # noqa: E402
from run_m3 import AGGREGATORS  # noqa: E402

from baggets.metrics import mase, smape  # noqa: E402
from baggets.pipeline import load_series_artifacts  # noqa: E402

CACHE_ROOT = ROOT / "data" / "cache"
RESULTS_ROOT = ROOT / "results"

def resolve_cache_dir(group: str, cache_key: str | None, n_bootstraps: int, seed: int) -> Path:
    """Locate a stage-A cache, by explicit key or by deriving one.

    An explicit key is preferred for the historical caches: run_m3.make_run_key
    gained an ``engine`` field in M8, so keys derived today do not match caches
    written before it (the 1428-series B=1000 monthly cache is 8d1f4360f7, whose
    manifest has no ``engine``). Deriving is kept as the default for new runs.
    """
    from run_m3 import make_run_key

    root = CACHE_ROOT / group
    key = cache_key or make_run_key(group, n_bootstraps, seed)[0]
    cache_dir = root / key
    if cache_dir.exists():
        return cache_dir
    available = sorted(d.name for d in root.glob("*/") if any(d.glob("*.npz")))
    raise SystemExit(
        f"no cache at {cache_dir}\n"
        f"available under {root}: {', '.join(available) or '(none)'}\n"
        f"pass --cache-key explicitly (the M7 B=1000 monthly cache is 8d1f4360f7)"
    )



def _ablate_one(path: Path, sizes: tuple[int, ...], aggs: tuple[str, ...],
                n_perm: int, seed: int) -> list[dict]:
    art = load_series_artifacts(path)
    fc, y_test, y_train, m = art.final_forecasts, art.y_test, art.y_train, art.season_length
    B = fc.shape[0]
    # deterministic per-series stream, independent of processing order
    rng = np.random.default_rng((seed, abs(hash(art.uid)) % (2**31)))
    score = lambda p: (smape(y_test, p), mase(y_test, p, y_train, m))  # noqa: E731

    rows = []
    for n in sizes:
        n = min(n, B)
        # the three arms: as-deployed (first n, contains member 0), member-0-free
        # random, and random over the whole pool (member 0 with prob n/B)
        pools = {
            "first": [np.arange(n)],
            "random_no0": [rng.choice(np.arange(1, B), n, replace=False) for _ in range(n_perm)],
            "random_all": [rng.choice(B, n, replace=False) for _ in range(n_perm)],
        }
        for arm, idx_sets in pools.items():
            for agg in aggs:
                f = AGGREGATORS[agg]
                vals = np.array([score(np.asarray(f(fc[i]))) for i in idx_sets])
                rows.append({
                    "uid": art.uid, "n": n, "arm": arm, "aggregator": agg,
                    "smape": float(vals[:, 0].mean()), "mase": float(vals[:, 1].mean()),
                    "smape_sd": float(vals[:, 0].std(ddof=1)) if len(vals) > 1 else 0.0,
                    "n_draws": len(idx_sets),
                })
    # member 0 alone: the plain-ETS anchor that the 'first' arm smuggles in
    s0, m0 = score(fc[0])
    rows.append({"uid": art.uid, "n": 1, "arm": "member0", "aggregator": "none",
                 "smape": s0, "mase": m0, "smape_sd": 0.0, "n_draws": 1})
    return rows


def cmd_ablate(args) -> int:
    cache_dir = resolve_cache_dir(args.group, args.cache_key, args.n_bootstraps, args.seed)
    paths = sorted(cache_dir.glob("*.npz"))
    if args.limit:
        paths = paths[: args.limit]
    sizes = tuple(int(s) for s in args.sizes.split(","))
    aggs = tuple(args.aggregators.split(","))
    print(f"{len(paths)} series x sizes {sizes} x {aggs} x {args.n_perm} permutations",
          flush=True)

    t0 = time.time()
    out = Parallel(n_jobs=args.n_jobs, return_as="generator")(
        delayed(_ablate_one)(p, sizes, aggs, args.n_perm, args.seed) for p in paths
    )
    rows = []
    for i, chunk in enumerate(out):
        rows.extend(chunk)
        if (i + 1) % 200 == 0:
            print(f"[{i+1}/{len(paths)}] {time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    out_dir = RESULTS_ROOT / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "member0_ablation.parquet", index=False)
    print(f"\n{len(df)} rows in {time.time()-t0:.0f}s -> {out_dir}")

    piv = df[df.arm != "member0"].pivot_table(
        index=["n", "aggregator"], columns="arm", values="smape", aggfunc="mean")
    piv["first_minus_rand_no0"] = piv["first"] - piv["random_no0"]
    print(f"\nmember0 alone (plain ETS): {df[df.arm=='member0'].smape.mean():.3f}\n")
    print(piv.round(4).to_string())
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ablate", help="T1: member-0 contamination of the noneN arms")
    a.add_argument("--group", default="monthly")
    a.add_argument("--sizes", default="10,25,50,100")
    a.add_argument("--aggregators", default="mean,median,trimmed")
    a.add_argument("--n-perm", type=int, default=20)
    a.add_argument("--n-bootstraps", type=int, default=1000)
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--cache-key", help="stage-A cache key (M7 monthly B=1000 is 8d1f4360f7)")
    a.add_argument("--limit", type=int)
    a.add_argument("--n-jobs", type=int, default=6)
    a.add_argument("--run-id", default="m7_ablation")
    a.set_defaults(func=cmd_ablate)
    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

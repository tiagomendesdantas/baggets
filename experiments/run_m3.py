"""M3 experiment runner: precompute (stage A), evaluate (stage B), report.

Examples (repo root):

    # pilot on 100 series
    uv run python experiments/run_m3.py precompute --group monthly --limit 100
    uv run python experiments/run_m3.py evaluate --group monthly --limit 100 \
        --strategies original,none100:mean,cluster-k5,cluster-auto,topk,random
    uv run python experiments/run_m3.py report

    # full monthly (overnight)
    uv run python experiments/run_m3.py precompute --group monthly --n-jobs 6

Stage A caches per-series tensors under data/cache/<group>/<run_key>/ so it is
resumable (rerun skips finished series) and stage B never refits anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import statsforecast  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

import baggets  # noqa: E402
from baggets.datasets import load_m3  # noqa: E402
from baggets.metrics import mase, smape  # noqa: E402
from baggets.pipeline import (  # noqa: E402
    apply_strategy,
    load_series_artifacts,
    precompute_series,
    strategy_rng,
)
from baggets.selection import (  # noqa: E402
    ClusterSelection,
    GreedyCovarianceSelection,
    NoSelection,
    PortfolioSelection,
    RandomSelection,
    TopKSelection,
)

CACHE_ROOT = ROOT / "data" / "cache"
RESULTS_ROOT = ROOT / "results"

AGGREGATORS = {"median": lambda e: np.median(e, axis=0), "mean": lambda e: np.mean(e, axis=0)}


# --------------------------------------------------------------- strategies
def parse_strategy(spec: str):
    """'name[:aggregator]' -> (label, strategy_or_None, aggregator_fn).

    The pseudo-strategy 'original' scores member 0 (plain ETS on the original
    series) — the free calibration anchor.
    """
    name, _, agg_name = spec.partition(":")
    agg_name = agg_name or "median"
    agg = AGGREGATORS[agg_name]
    if name == "original":
        strategy = None
    elif name == "greedy":
        # the greedy objective aggregates the same way the deployment does
        strategy = GreedyCovarianceSelection(aggregator=agg_name)
    elif name == "portfolio":
        strategy = PortfolioSelection()
    elif name == "topk":
        strategy = TopKSelection()
    elif name == "random":
        strategy = RandomSelection()
    elif name == "none":
        strategy = NoSelection()
    elif name.startswith("none"):
        strategy = NoSelection(int(name[4:]))
    elif name.startswith(("cluster-", "paper-")):
        kind, arg = name.split("-", 1)
        if arg.startswith("k") and arg[1:].isdigit():  # accept cluster-k5 and cluster-5
            arg = arg[1:]
        n_clusters = "auto" if arg == "auto" else int(arg)
        strategy = (ClusterSelection.paper(n_clusters=n_clusters) if kind == "paper"
                    else ClusterSelection(n_clusters=n_clusters))
    else:
        raise ValueError(f"unknown strategy spec {spec!r}")
    label = spec if ":" in spec else f"{name}:{agg_name}" if agg_name != "median" else name
    return label, strategy, agg


# --------------------------------------------------------------- run key
def make_run_key(group: str, n_bootstraps: int, root_seed: int) -> str:
    params = {
        "group": group,
        "n_bootstraps": n_bootstraps,
        "root_seed": root_seed,
        "h_val_rule": "v1",
        "statsforecast": statsforecast.__version__,
        "baggets": baggets.__version__,
    }
    key = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:10]
    return key, params


def select_records(group: str, limit: int | None, uids: list[str] | None):
    records = load_m3(group, ROOT / "data" / "m3")
    if uids:
        wanted = set(uids)
        records = [r for r in records if r.uid in wanted]
    if limit:
        records = records[:limit]
    return records


# --------------------------------------------------------------- precompute
def cmd_precompute(args) -> int:
    run_key, params = make_run_key(args.group, args.n_bootstraps, args.seed)
    cache_dir = CACHE_ROOT / args.group / run_key
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_dir / "manifest.json", "w") as f:
        json.dump(params, f, indent=2)

    records = select_records(args.group, args.limit, args.uids)
    todo = [r for r in records if not (cache_dir / f"{r.uid}.npz").exists()]
    print(f"run_key={run_key}: {len(records)} series, {len(todo)} to compute "
          f"({len(records) - len(todo)} cached)")
    if not todo:
        return 0

    t0 = time.time()

    def _one(rec):
        t = time.time()
        meta = precompute_series(rec, args.n_bootstraps, args.seed, cache_dir / f"{rec.uid}.npz")
        return rec.uid, meta, time.time() - t

    results = Parallel(n_jobs=args.n_jobs, verbose=0, return_as="generator")(
        delayed(_one)(rec) for rec in todo
    )
    done = 0
    for uid, _meta, dt in results:
        done += 1
        if done % 10 == 0 or done == len(todo):
            elapsed = time.time() - t0
            eta = elapsed / done * (len(todo) - done)
            print(f"[{done}/{len(todo)}] {uid} ({dt:.0f}s)  "
                  f"elapsed {elapsed/60:.1f}m, ETA {eta/60:.1f}m", flush=True)
    print(f"stage A done in {(time.time()-t0)/60:.1f} min")
    return 0


# --------------------------------------------------------------- evaluate
def cmd_evaluate(args) -> int:
    run_key, _ = make_run_key(args.group, args.n_bootstraps, args.seed)
    cache_dir = CACHE_ROOT / args.group / run_key
    records = select_records(args.group, args.limit, args.uids)
    paths = [cache_dir / f"{r.uid}.npz" for r in records]
    missing = [p.stem for p in paths if not p.exists()]
    if missing:
        print(f"{len(missing)} series not precomputed (e.g. {missing[:5]}); "
              "run precompute first")
        return 1

    strategy_specs = args.strategies.split(",")
    eval_seeds = [int(s) for s in args.eval_seeds.split(",")]

    def _eval_one(path):
        strategies = [parse_strategy(s) for s in strategy_specs]
        art = load_series_artifacts(path)
        out = []
        for label, strategy, agg in strategies:
            for seed in eval_seeds:
                t1 = time.time()
                if strategy is None:  # 'original': plain ETS on member 0
                    point = art.final_forecasts[0]
                    info = {"n_members": 1}
                else:
                    rng = strategy_rng(seed, art.uid, label)
                    point, info = apply_strategy(art, strategy, args.n_select, agg, rng)
                out.append({
                    "uid": art.uid,
                    "strategy": label,
                    "seed": seed,
                    "smape": smape(art.y_test, point),
                    "mase": mase(art.y_test, point, art.y_train, art.season_length),
                    "n_members": info["n_members"],
                    "k_chosen": info.get("k_chosen"),
                    "ms": 1000 * (time.time() - t1),
                })
        return out

    t0 = time.time()
    rows = []
    if args.n_jobs == 1:
        for i, path in enumerate(paths):
            rows.extend(_eval_one(path))
            if (i + 1) % 100 == 0:
                print(f"[{i+1}/{len(paths)}] {(time.time()-t0):.0f}s", flush=True)
    else:
        chunks = Parallel(n_jobs=args.n_jobs, return_as="generator")(
            delayed(_eval_one)(p) for p in paths
        )
        for i, chunk in enumerate(chunks):
            rows.extend(chunk)
            if (i + 1) % 100 == 0:
                print(f"[{i+1}/{len(paths)}] {(time.time()-t0):.0f}s", flush=True)

    scores = pd.DataFrame(rows)
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(out_dir / "scores.parquet", index=False)

    summary = summarize(scores)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"\n{summary.to_string(index=False)}")
    print(f"\nwrote {out_dir}/scores.parquet and summary.csv "
          f"({len(scores)} rows, {(time.time()-t0):.0f}s)")
    return 0


def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    """The paper's six columns: Rank/Mean/Median x sMAPE/MASE (ranks within the
    evaluated strategy set, averaged over seeds first)."""
    per = scores.groupby(["uid", "strategy"])[["smape", "mase"]].mean().reset_index()
    per["rank_smape"] = per.groupby("uid")["smape"].rank()
    per["rank_mase"] = per.groupby("uid")["mase"].rank()
    out = per.groupby("strategy").agg(
        rank_smape=("rank_smape", "mean"),
        mean_smape=("smape", "mean"),
        median_smape=("smape", "median"),
        rank_mase=("rank_mase", "mean"),
        mean_mase=("mase", "mean"),
        median_mase=("mase", "median"),
    ).reset_index()
    return out.sort_values("rank_smape")


def cmd_stats(args) -> int:
    """Friedman + Hochberg post-hoc (the paper's Table 6 procedure) on a results run."""
    from baggets.stats import friedman_hochberg

    path = RESULTS_ROOT / args.run_id / "scores.parquet"
    if not path.exists():
        print(f"no scores at {path}")
        return 1
    scores = pd.read_parquet(path)
    per = scores.groupby(["uid", "strategy"], as_index=False)[args.metric].mean()
    res = friedman_hochberg(per, metric=args.metric, control=args.control)
    print(f"Friedman p-value: {res['friedman_pvalue']:.3e}  (n={res['n_series']} series)")
    print(f"Control: {res['control']} (mean rank {res['control_mean_rank']:.3f})\n")
    print(res["table"].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    out = RESULTS_ROOT / args.run_id / f"stats_{args.metric}.csv"
    res["table"].to_csv(out, index=False)
    print(f"\nwrote {out}")
    return 0


def cmd_report(args) -> int:
    runs = sorted(RESULTS_ROOT.glob("*/summary.csv"))
    if not runs:
        print("no results yet")
        return 1
    path = runs[-1] if not args.run_id else RESULTS_ROOT / args.run_id / "summary.csv"
    print(f"# {path.parent.name}\n")
    print(pd.read_csv(path).to_string(index=False))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--group", default="monthly")
    common.add_argument("--n-bootstraps", type=int, default=1000)
    common.add_argument("--seed", type=int, default=42, help="stage-A root seed")
    common.add_argument("--limit", type=int)
    common.add_argument("--uids", nargs="*")

    pp = sub.add_parser("precompute", parents=[common])
    pp.add_argument("--n-jobs", type=int, default=6)
    pp.set_defaults(fn=cmd_precompute)

    pe = sub.add_parser("evaluate", parents=[common])
    pe.add_argument("--strategies",
                    default="original,none100:mean,cluster-k5,cluster-auto,topk,random")
    pe.add_argument("--n-select", type=int, default=100)
    pe.add_argument("--eval-seeds", default="1", help="comma list")
    pe.add_argument("--run-id")
    pe.add_argument("--n-jobs", type=int, default=1)
    pe.set_defaults(fn=cmd_evaluate)

    pr = sub.add_parser("report")
    pr.add_argument("--run-id")
    pr.set_defaults(fn=cmd_report)

    ps = sub.add_parser("stats")
    ps.add_argument("--run-id", required=True)
    ps.add_argument("--metric", default="smape")
    ps.add_argument("--control", default=None)
    ps.set_defaults(fn=cmd_stats)

    args = p.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())

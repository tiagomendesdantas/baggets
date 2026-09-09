"""Compare base learners on the same series, with the same bootstrap budget.

The only comparison worth making is a matched one: same series, same B, same
selection strategy, same aggregation — only the base learner differs. This
loads two stage-A caches, intersects them on series id, scores both, and runs
the paper's Friedman + Hochberg procedure over the result.

    uv run python experiments/compare_engines.py --engines ets nbeats-i \
        --n-bootstraps 100 --strategy none100:mean
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_m3 import CACHE_ROOT, make_run_key, parse_strategy  # noqa: E402

from baggets.metrics import mase, smape  # noqa: E402
from baggets.pipeline import apply_strategy, load_series_artifacts, strategy_rng  # noqa: E402
from baggets.stats import friedman_hochberg  # noqa: E402


def cache_dir(group: str, n_bootstraps: int, seed: int, engine: str, epochs: int) -> Path:
    run_key, _ = make_run_key(group, n_bootstraps, seed, engine, epochs)
    return CACHE_ROOT / group / run_key


def score_engine(directory: Path, uids: list[str], strategy_spec: str, seed: int) -> pd.DataFrame:
    label, strategy, agg = parse_strategy(strategy_spec)
    rows = []
    for uid in uids:
        art = load_series_artifacts(directory / f"{uid}.npz")
        if strategy is None:
            point = art.final_forecasts[0]
        else:
            point, _ = apply_strategy(art, strategy, 10**9, agg,
                                      strategy_rng(seed, art.uid, label))
        rows.append({
            "uid": uid,
            "smape": smape(art.y_test, point),
            "mase": mase(art.y_test, point, art.y_train, art.season_length),
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--group", default="monthly")
    ap.add_argument("--engines", nargs="+", default=["ets", "nbeats-i"])
    ap.add_argument("--n-bootstraps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--strategy", default="none100:mean")
    ap.add_argument("--control", default=None, help="engine to test the others against")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dirs = {e: cache_dir(args.group, args.n_bootstraps, args.seed, e, args.epochs)
            for e in args.engines}
    available = {}
    for engine, d in dirs.items():
        if not d.exists():
            print(f"no cache for engine {engine!r} at {d}")
            return 1
        available[engine] = {p.stem for p in d.glob("*.npz")}
        print(f"{engine:10s} {len(available[engine]):4d} series cached  ({d.name})")

    uids = sorted(set.intersection(*available.values()))
    if not uids:
        print("no series in common — nothing to compare")
        return 1
    print(f"\nmatched on {len(uids)} series present in every arm "
          f"(strategy {args.strategy}, B={args.n_bootstraps})\n")

    frames = {e: score_engine(dirs[e], uids, args.strategy, args.seed) for e in args.engines}

    summary = pd.DataFrame([
        {"engine": e,
         "mean_smape": f["smape"].mean(), "median_smape": f["smape"].median(),
         "mean_mase": f["mase"].mean(), "median_mase": f["mase"].median()}
        for e, f in frames.items()
    ]).sort_values("mean_smape")
    print(summary.to_string(index=False))

    # friedman_hochberg takes a tidy frame keyed on "strategy"; here the thing
    # being compared is the engine, so it goes in that column.
    tidy = pd.concat([f.assign(strategy=e) for e, f in frames.items()], ignore_index=True)
    control = args.control or summary.iloc[0]["engine"]
    for metric in ("smape", "mase"):
        if len(args.engines) < 3:
            # Friedman is a multi-group test and needs 3+. Two paired arms call
            # for Wilcoxon signed-rank: same series, two measurements each.
            from scipy.stats import wilcoxon

            a = frames[args.engines[0]].set_index("uid")[metric]
            b = frames[args.engines[1]].set_index("uid")[metric].reindex(a.index)
            stat, pvalue = wilcoxon(a, b)
            diff = float((b - a).mean())
            print(f"\nWilcoxon signed-rank on {metric} "
                  f"({args.engines[1]} vs {args.engines[0]}, n={len(a)}): "
                  f"W={stat:.0f}, p={pvalue:.3g}, mean difference {diff:+.4f}")
            print("  -> " + ("a difference this consistent is unlikely by chance"
                             if pvalue < 0.05 else
                             "no difference detectable at this sample size"))
            continue
        try:
            stats = friedman_hochberg(tidy[["uid", "strategy", metric]], metric=metric,
                                      control=control)
        except Exception as exc:
            print(f"\n{metric}: statistics unavailable ({exc.__class__.__name__}: {exc})")
            continue
        print(f"\nFriedman + Hochberg on {metric} "
              f"(control = {stats['control']}, Friedman p = {stats['friedman_pvalue']:.3g}):")
        print(pd.DataFrame(stats["table"]).to_string(index=False))

    if args.out:
        wide = pd.DataFrame({e: f.set_index("uid")["smape"] for e, f in frames.items()})
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        wide.to_csv(args.out)
        print(f"\nper-series sMAPE: {args.out}")

    wins = {}
    base = args.engines[0]
    for e in args.engines[1:]:
        a = frames[base].set_index("uid")["smape"]
        b = frames[e].set_index("uid")["smape"]
        wins[e] = float((b < a).mean())
    for e, w in wins.items():
        print(f"\n{e} beats {base} on {w:.0%} of the matched series "
              f"({int(w*len(uids))}/{len(uids)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

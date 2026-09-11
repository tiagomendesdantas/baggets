"""Does bagging's per-series gain track ETS family instability across members?

M7's mechanism test at n=300 gave rho=+0.113, p=0.051 -- the right direction
with no significance. This is the pre-registered analysis for the full 1428
series. The tests, the alpha and the tau definitions are fixed HERE, before the
data exists, and the result is reported whichever way it comes out.

PRE-REGISTERED
  Primary    Spearman(n_distinct_families, tau_bag), two-sided, alpha = 0.05.
             The statistic that came closest in the pilot.
  Secondary  Spearman(family_entropy, tau_bag); both predictors against tau_ma.
  tau        Member-0-free on BOTH sides. Member 0 is the original series, not a
             bootstrap draw (bootstrap.py:70), and M7's T1 ablation showed it
             flatters every noneN arm by ~1/n. tau_bag therefore uses the
             ablation's member-0-free random-100 trimmed bag, never
             none100:trimmed, so the T1 confound stays out of the regression.
  Power      n=1428 gives 80% power at |rho| ~ 0.074. A null is therefore
             informative, and is reported as "refuted at effect sizes >= MDE"
             rather than as mere absence.
  No new predictors after seeing the data. A failed primary test is a refuted
  mechanism claim, not an invitation to hunt for a variable that works.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
RESULTS_ROOT = ROOT / "results"

ALPHA = 0.05
PRIMARY = "n_distinct_families"


def mde_rho(n: int, alpha: float = ALPHA, power: float = 0.80) -> float:
    """Smallest |rho| detectable at the given power (Fisher-z approximation)."""
    return (norm.ppf(1 - alpha / 2) + norm.ppf(power)) / np.sqrt(n - 3)


def load_taus(ablation_run: str, aicc_run: str) -> pd.DataFrame:
    ab = pd.read_parquet(RESULTS_ROOT / ablation_run / "member0_ablation.parquet")
    plain = ab[ab.arm == "member0"].set_index("uid").smape.rename("plain_ets")
    bag = (ab[(ab.arm == "random_no0") & (ab.n == 100) & (ab.aggregator == "trimmed")]
           .set_index("uid").smape.rename("bag_no0"))
    ma = (pd.read_parquet(RESULTS_ROOT / aicc_run / "scores.parquet")
          .query("strategy == 'top5_mean'").groupby("uid").smape.mean().rename("top5_mean"))
    d = pd.concat([plain, bag, ma], axis=1).dropna()
    d["tau_bag"] = d.plain_ets - d.bag_no0    # >0: bagging helps this series
    d["tau_ma"] = d.plain_ets - d.top5_mean   # >0: family averaging helps
    return d


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--entropy-run", default="m7_entropy_full")
    p.add_argument("--ablation-run", default="m7_ablation")
    p.add_argument("--aicc-run", default="m7_aicc")
    args = p.parse_args()

    ent = pd.read_parquet(RESULTS_ROOT / args.entropy_run / "family_entropy.parquet")
    d = load_taus(args.ablation_run, args.aicc_run).join(ent.set_index("uid"), how="inner")
    n = len(d)
    mde = mde_rho(n)

    print(f"n = {n} series  |  alpha = {ALPHA}  |  MDE at 80% power: |rho| = {mde:.4f}")
    print(f"mean tau_bag {d.tau_bag.mean():+.4f} pp   mean tau_ma {d.tau_ma.mean():+.4f} pp")
    print(f"family instability: median {d[PRIMARY].median():.0f} distinct families "
          f"per series, mean {d[PRIMARY].mean():.2f}, max {d[PRIMARY].max()}")
    print(f"series where all {int(d.n_members_refit.mode()[0])} members agree: "
          f"{(d[PRIMARY] == 1).sum()}\n")

    r = spearmanr(d[PRIMARY], d.tau_bag)
    verdict = "SUPPORTED" if r.pvalue < ALPHA and r.statistic > 0 else "NOT SUPPORTED"
    print(f"PRIMARY  Spearman({PRIMARY}, tau_bag) = {r.statistic:+.4f}  "
          f"p = {r.pvalue:.3e}   -> mechanism claim {verdict}")
    if r.pvalue >= ALPHA:
        print(f"         null is informative: any true |rho| >= {mde:.3f} would have been "
              f"detected at 80% power")
    print()

    print("SECONDARY")
    for pred in [PRIMARY, "family_entropy", "n_eff_families", "modal_share"]:
        for tau in ["tau_bag", "tau_ma"]:
            rr = spearmanr(d[pred], d[tau])
            flag = " *" if rr.pvalue < ALPHA else ""
            print(f"  {pred:20s} vs {tau:8s} rho = {rr.statistic:+.4f}  "
                  f"p = {rr.pvalue:.3e}{flag}")

    print("\nquartiles of the primary predictor:")
    d["q"] = pd.qcut(d[PRIMARY].rank(method="first"), 4,
                     labels=["Q1 stable", "Q2", "Q3", "Q4 unstable"])
    print(d.groupby("q", observed=True).agg(
        n=("tau_bag", "size"), n_families=(PRIMARY, "mean"),
        tau_bag=("tau_bag", "mean"), tau_ma=("tau_ma", "mean")).round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

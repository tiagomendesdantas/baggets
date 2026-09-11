"""Does bagging's per-series gain track how much members disagree about model form?

NOTE ON THE FILENAME. "instability" here means only that bootstrap members select
different ETS families, which is BY DESIGN -- BLD.MBB generates different series
and AutoETS re-selects on each. It is not a defect. Do not confuse it with the
optimizer-driven family flipping in M7 Finding 4, which IS a defect and which
this project also calls instability. The filename predates that distinction.

Read the result as an association, not a mechanism: bagging can only help when
member forecasts differ, and members picking different families naturally produce
forecasts that differ more, so family count and forecast spread move together.
Only the family count is measured here, so this cannot separate "model form is a
mechanism" from "forecasts were simply spread apart". See Finding 2's caveats.

M7's mechanism test at n=300 gave rho=+0.113, p=0.051 -- the right direction
with no significance. This is the pre-registered analysis for the full 1428
series. The tests, the alpha and the tau definitions are fixed HERE, before the
data exists, and the result is reported whichever way it comes out.

PRE-REGISTERED
  Primary    Spearman(n_distinct_families, tau_bag), two-sided, alpha = 0.05.
             The statistic that came closest in the pilot.
  Secondary  Spearman(family_entropy, tau_bag); both predictors against tau_ma.
  tau        Baseline is plain ETS on the ORIGINAL series with no bagging --
             final_forecasts[0], the 'original' arm. The bag side uses the
             member-0-free random-100 trimmed bag.
  Power      n=1428 gives 80% power at |rho| ~ 0.074. A null is therefore
             informative, and is reported as "refuted at effect sizes >= MDE"
             rather than as mere absence.
  No new predictors after seeing the data. A failed primary test is a refuted
  mechanism claim, not an invitation to hunt for a variable that works.

POST-HOC CORRECTION (added 2026-09-11, after the run)
  The pre-registered tau_bag excludes member 0 from the BAG as well as using it
  as the baseline. That was wrong about the method: the original series is a
  designed ensemble member, not contamination -- `xs[[1]] <- x` in the 2018 R
  source, and every selection rule there draws from the full pool. So the
  pre-registered tau_bag compares the benchmark against a modified estimator.
  Both are therefore computed and reported:
    preregistered  plain ETS - member-0-free random-100 trimmed bag
    as_specified   plain ETS - none100:trimmed, i.e. bagging as the method defines it
  The pre-registered one remains PRIMARY. Swapping the definition silently after
  seeing the result would be exactly the post-hoc flexibility this project
  criticises in the 2018 paper; the correction is declared, not substituted.
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


def load_taus(ablation_run: str, aicc_run: str, m5_run: str = "m5_full",
              bag_run: str = "m6_trimmed100") -> pd.DataFrame:
    """Per-series gains against the no-bagging benchmark, under both tau definitions."""
    ab = pd.read_parquet(RESULTS_ROOT / ablation_run / "member0_ablation.parquet")
    # the benchmark: plain ETS on the original series, no bagging
    plain = ab[ab.arm == "member0"].set_index("uid").smape.rename("plain_ets")
    bag_no0 = (ab[(ab.arm == "random_no0") & (ab.n == 100) & (ab.aggregator == "trimmed")]
               .set_index("uid").smape.rename("bag_no0"))
    bag_spec = (pd.read_parquet(RESULTS_ROOT / bag_run / "scores.parquet")
                .query("strategy == 'none100:trimmed'").groupby("uid").smape.mean()
                .rename("bag_as_specified"))
    ma = (pd.read_parquet(RESULTS_ROOT / aicc_run / "scores.parquet")
          .query("strategy == 'top5_mean'").groupby("uid").smape.mean().rename("top5_mean"))

    # consistency: the ablation's member-0 arm and m5_full's 'original' are both
    # final_forecasts[0], so they must agree up to the cache's float32 storage
    orig = (pd.read_parquet(RESULTS_ROOT / m5_run / "scores.parquet")
            .query("strategy == 'original'").groupby("uid").smape.mean())
    gap = (plain - orig.reindex(plain.index)).abs().max()
    assert gap < 1e-4, f"member0 arm disagrees with m5_full 'original' by {gap}"

    d = pd.concat([plain, bag_no0, bag_spec, ma], axis=1).dropna()
    d["tau_bag"] = d.plain_ets - d.bag_no0              # PRE-REGISTERED (member-0-free bag)
    d["tau_bag_spec"] = d.plain_ets - d.bag_as_specified  # post-hoc: bagging as specified
    d["tau_ma"] = d.plain_ets - d.top5_mean             # >0: family averaging helps
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
    print(f"benchmark (plain ETS on the original series, no bagging): "
          f"{d.plain_ets.mean():.4f} mean sMAPE")
    print(f"  bag, member-0-free (pre-registered): {d.bag_no0.mean():.4f}  "
          f"-> mean tau_bag {d.tau_bag.mean():+.4f} pp")
    print(f"  bag, as specified (none100:trimmed): {d.bag_as_specified.mean():.4f}  "
          f"-> mean tau_bag {d.tau_bag_spec.mean():+.4f} pp")
    print(f"  top5 family average:                 {d.top5_mean.mean():.4f}  "
          f"-> mean tau_ma  {d.tau_ma.mean():+.4f} pp")
    print(f"members disagree: median {d[PRIMARY].median():.0f} distinct families "
          f"per series, mean {d[PRIMARY].mean():.2f}, max {d[PRIMARY].max()}")
    print(f"series where all {int(d.n_members_refit.mode()[0])} members agree: "
          f"{(d[PRIMARY] == 1).sum()}\n")

    r = spearmanr(d[PRIMARY], d.tau_bag)
    verdict = "SUPPORTED" if r.pvalue < ALPHA and r.statistic > 0 else "NOT SUPPORTED"
    print(f"PRIMARY (pre-registered tau)  Spearman({PRIMARY}, tau_bag) = {r.statistic:+.4f}  "
          f"p = {r.pvalue:.3e}   -> mechanism claim {verdict}")
    if r.pvalue >= ALPHA:
        print(f"         null is informative: any true |rho| >= {mde:.3f} would have been "
              f"detected at 80% power")

    rs = spearmanr(d[PRIMARY], d.tau_bag_spec)
    agree = (rs.pvalue < ALPHA) == (r.pvalue < ALPHA) and np.sign(rs.statistic) == np.sign(r.statistic)
    print(f"SAME TEST, tau as specified   Spearman({PRIMARY}, tau_bag_spec) = {rs.statistic:+.4f}  "
          f"p = {rs.pvalue:.3e}")
    print(f"  post-hoc correction of a definitional error; the two definitions "
          f"{'AGREE' if agree else 'DISAGREE -- report this prominently'}\n")

    print("SECONDARY")
    for pred in [PRIMARY, "family_entropy", "n_eff_families", "modal_share"]:
        for tau in ["tau_bag", "tau_ma"]:
            rr = spearmanr(d[pred], d[tau])
            flag = " *" if rr.pvalue < ALPHA else ""
            print(f"  {pred:20s} vs {tau:8s} rho = {rr.statistic:+.4f}  "
                  f"p = {rr.pvalue:.3e}{flag}")

    print("\nquartiles of the primary predictor:")
    d["q"] = pd.qcut(d[PRIMARY].rank(method="first"), 4,
                     labels=["Q1 most agree", "Q2", "Q3", "Q4 least agree"])
    print(d.groupby("q", observed=True).agg(
        n=("tau_bag", "size"), n_families=(PRIMARY, "mean"),
        tau_bag=("tau_bag", "mean"), tau_ma=("tau_ma", "mean")).round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

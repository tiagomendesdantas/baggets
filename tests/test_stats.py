import numpy as np
import pandas as pd
import pytest

from baggets.stats import friedman_hochberg


def make_scores(n_series=60, seed=0, effect=0.0):
    """Three strategies; strategy 'c' is worse by `effect` on average."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_series):
        base = rng.normal(10, 2)
        rows.append({"uid": f"S{i}", "strategy": "a", "smape": base + rng.normal(0, 0.5)})
        rows.append({"uid": f"S{i}", "strategy": "b", "smape": base + rng.normal(0, 0.5)})
        rows.append({"uid": f"S{i}", "strategy": "c",
                     "smape": base + effect + rng.normal(0, 0.5)})
    return pd.DataFrame(rows)


class TestFriedmanHochberg:
    def test_null_case_no_significance(self):
        res = friedman_hochberg(make_scores(effect=0.0))
        assert res["friedman_pvalue"] > 0.01
        assert (res["table"]["p_hochberg"] > 0.05).all()

    def test_strong_effect_detected(self):
        res = friedman_hochberg(make_scores(effect=2.0))
        assert res["friedman_pvalue"] < 1e-6
        row_c = res["table"].set_index("strategy").loc["c"]
        assert row_c["p_hochberg"] < 0.01
        assert res["control"] in ("a", "b")

    def test_control_override(self):
        res = friedman_hochberg(make_scores(effect=2.0), control="c")
        assert res["control"] == "c"
        assert (res["table"]["p_hochberg"] < 0.05).all()

    def test_adjusted_monotone_in_raw(self):
        res = friedman_hochberg(make_scores(effect=0.5, seed=3))
        t = res["table"].sort_values("p_raw")
        assert (np.diff(t["p_hochberg"].to_numpy()) >= -1e-12).all()
        assert (t["p_hochberg"] >= t["p_raw"] - 1e-12).all()

    def test_missing_series_raises(self):
        df = make_scores().iloc[:-1]
        with pytest.raises(ValueError, match="missing"):
            friedman_hochberg(df)

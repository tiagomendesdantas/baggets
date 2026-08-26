import numpy as np

from baggets.selection import (
    ClusterSelection,
    NoSelection,
    RandomSelection,
    TopKSelection,
    _allocate_largest_remainder,
    _allocate_paper,
)
from baggets.validation import ValidationArtifacts


def make_artifacts(n_members=60, n_groups=3, seed=0):
    """Synthetic artifacts: members fall into distinct series-shape groups, with
    validation errors increasing with member index inside each group."""
    rng = np.random.default_rng(seed)
    t = 20
    levels = np.linspace(0, 400, n_groups)
    series, errors = [], []
    per = n_members // n_groups
    for g, level in enumerate(levels):
        for j in range(per):
            series.append(np.full(t, level) + rng.normal(0, 1.0, t))
            errors.append(1.0 + g * 0.1 + j)  # best members first within each group
    series = np.vstack(series)
    errors = np.array(errors)
    return ValidationArtifacts(
        series=series,
        val_forecasts=rng.normal(size=(n_members, 4)),
        val_actuals=np.zeros(4),
        val_errors=errors,
        season_length=1,
    )


class TestAllocations:
    def test_paper_matches_r_rounding(self):
        sizes = np.array([100, 100, 99])
        # round(100/299*100) = round(33.44) = 33; round(99/299*100) = 33
        np.testing.assert_array_equal(_allocate_paper(sizes, 299, 100), [33, 33, 33])

    def test_paper_min_one(self):
        sizes = np.array([297, 1, 1])
        n_h = _allocate_paper(sizes, 299, 100)
        assert n_h[1] == 1 and n_h[2] == 1

    def test_largest_remainder_sums_exactly(self):
        for sizes in ([100, 100, 99], [297, 1, 1], [150, 75, 50, 24], [299]):
            n_h = _allocate_largest_remainder(np.array(sizes), 299, 100)
            assert n_h.sum() == 100, sizes
            assert np.all(n_h >= 1)
            assert np.all(n_h <= np.array(sizes))

    def test_largest_remainder_proportionality(self):
        n_h = _allocate_largest_remainder(np.array([200, 60, 40]), 300, 100)
        np.testing.assert_array_equal(n_h, [67, 20, 13])


class TestTopK:
    def test_picks_lowest_errors(self):
        art = make_artifacts()
        idx = TopKSelection().select(art, 10, np.random.default_rng(0))
        expected = np.sort(np.argsort(art.val_errors, kind="stable")[:10])
        np.testing.assert_array_equal(idx, expected)


class TestRandom:
    def test_size_and_uniqueness(self):
        art = make_artifacts()
        idx = RandomSelection().select(art, 25, np.random.default_rng(1))
        assert idx.size == 25 == np.unique(idx).size

    def test_deterministic_given_rng(self):
        art = make_artifacts()
        a = RandomSelection().select(art, 25, np.random.default_rng(5))
        b = RandomSelection().select(art, 25, np.random.default_rng(5))
        np.testing.assert_array_equal(a, b)


class TestNoSelection:
    def test_all(self):
        art = make_artifacts()
        np.testing.assert_array_equal(
            NoSelection().select(art, 10, np.random.default_rng(0)), np.arange(60)
        )

    def test_first_n(self):
        art = make_artifacts()
        np.testing.assert_array_equal(
            NoSelection(15).select(art, 10, np.random.default_rng(0)), np.arange(15)
        )


class TestClusterSelection:
    def test_selects_across_all_groups(self):
        art = make_artifacts(n_members=60, n_groups=3)
        sel = ClusterSelection(n_clusters=3, n_prefilter=45)
        idx = sel.select(art, 15, np.random.default_rng(0))
        assert idx.size == 15
        # each shape-group contributes members (the whole point of the method)
        groups = idx // 20
        assert set(groups) == {0, 1, 2}
        assert art.meta["k_chosen"] == 3

    def test_picks_best_within_cluster(self):
        art = make_artifacts(n_members=60, n_groups=3)
        idx = ClusterSelection(n_clusters=3, n_prefilter=45).select(
            art, 15, np.random.default_rng(0)
        )
        # within each group, errors increase with index -> selected are the
        # low-index members of each group (among the prefiltered)
        for g in range(3):
            members = idx[idx // 20 == g]
            assert np.all(members % 20 < 15)

    def test_prefilter_smaller_than_target_degenerates_to_topk(self):
        art = make_artifacts()
        sel = ClusterSelection(n_clusters=3, n_prefilter=8)
        idx = sel.select(art, 10, np.random.default_rng(0))
        expected = TopKSelection().select(art, 10, np.random.default_rng(0))
        np.testing.assert_array_equal(idx, expected)

    def test_auto_k(self):
        art = make_artifacts(n_members=60, n_groups=3)
        sel = ClusterSelection(n_clusters="auto", n_prefilter=45, k_max=10)
        idx = sel.select(art, 15, np.random.default_rng(0))
        assert idx.size == 15
        assert art.meta["k_chosen"] == 3  # silhouette finds the 3 obvious groups

    def test_paper_preset(self):
        sel = ClusterSelection.paper()
        assert sel.n_prefilter == 299
        assert sel.allocation == "paper"
        assert sel.pam_variant == "pam"
        assert sel.name == "cluster(k=auto)"

    def test_deterministic(self):
        art = make_artifacts(n_members=60, n_groups=3)
        sel = ClusterSelection(n_clusters=3, n_prefilter=45)
        a = sel.select(art, 15, np.random.default_rng(3))
        b = sel.select(art, 15, np.random.default_rng(3))
        np.testing.assert_array_equal(a, b)

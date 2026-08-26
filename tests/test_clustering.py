import numpy as np
import pytest
from scipy.spatial.distance import cdist

from baggets.clustering import auto_k_silhouette, euclidean_distance_matrix, pam_cluster


@pytest.fixture(scope="module")
def three_blobs():
    """Three well-separated groups of series (rows)."""
    rng = np.random.default_rng(7)
    base = np.vstack(
        [np.full(20, 0.0), np.full(20, 50.0), np.full(20, 200.0)]
    )
    rows = np.repeat(base, 15, axis=0) + rng.normal(0, 1.0, (45, 20))
    labels_true = np.repeat([0, 1, 2], 15)
    return rows, labels_true


def partitions_equal(a, b):
    """Same partition up to label permutation."""
    return len({(x, y) for x, y in zip(a, b, strict=True)}) == len(set(a)) == len(set(b))


class TestDistanceMatrix:
    def test_matches_manual(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(10, 6))
        np.testing.assert_allclose(euclidean_distance_matrix(x), cdist(x, x), atol=1e-12)


class TestPam:
    def test_recovers_blobs(self, three_blobs):
        rows, labels_true = three_blobs
        d = euclidean_distance_matrix(rows)
        labels = pam_cluster(d, 3)
        assert partitions_equal(labels, labels_true)

    def test_variants_agree_on_easy_data(self, three_blobs):
        rows, _ = three_blobs
        d = euclidean_distance_matrix(rows)
        assert partitions_equal(pam_cluster(d, 3, "pam"), pam_cluster(d, 3, "fasterpam"))

    def test_unknown_variant(self, three_blobs):
        d = euclidean_distance_matrix(three_blobs[0])
        with pytest.raises(ValueError, match="variant"):
            pam_cluster(d, 3, "bogus")


class TestAutoK:
    def test_finds_three(self, three_blobs):
        rows, labels_true = three_blobs
        d = euclidean_distance_matrix(rows)
        k, labels = auto_k_silhouette(d, k_min=2, k_max=10)
        assert k == 3
        assert partitions_equal(labels, labels_true)

    def test_k_max_clamped_to_n_minus_one(self, three_blobs):
        rows, _ = three_blobs
        d = euclidean_distance_matrix(rows[:6])
        k, _ = auto_k_silhouette(d, k_min=2, k_max=100)
        assert 2 <= k <= 5

    def test_invalid_range_raises(self):
        d = euclidean_distance_matrix(np.eye(3))
        with pytest.raises(ValueError, match="cannot cluster"):
            auto_k_silhouette(d, k_min=5, k_max=100)

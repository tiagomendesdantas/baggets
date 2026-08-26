"""Tier-3 fixture tests: kmedoids.pam vs R cluster::pam on an identical
precomputed distance matrix (both implement classic BUILD+SWAP, so partitions
should agree exactly up to label permutation)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score, silhouette_score

from baggets.clustering import pam_cluster


@pytest.fixture(scope="module")
def dist(fixtures_dir):
    path = fixtures_dir / "input" / "pam_distance.csv"
    if not path.exists():
        pytest.skip("PAM distance fixture not generated")
    return np.loadtxt(path, delimiter=",")


@pytest.fixture(scope="module")
def r_labels(fixtures_dir):
    path = fixtures_dir / "r_pam" / "labels.csv"
    if not path.exists():
        pytest.skip("R PAM reference not generated")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def r_silhouette(fixtures_dir):
    path = fixtures_dir / "r_pam" / "silhouette.csv"
    if not path.exists():
        pytest.skip("R PAM silhouette reference not generated")
    return pd.read_csv(path)


class TestPamVsR:
    @pytest.mark.parametrize("k", [2, 5, 20, 50])
    def test_partitions_identical(self, dist, r_labels, k):
        ref = r_labels[r_labels["k"] == k].sort_values("point")["label"].to_numpy()
        labels = pam_cluster(dist, k, variant="pam")
        ari = adjusted_rand_score(ref, labels)
        assert ari == pytest.approx(1.0), f"k={k}: ARI={ari:.4f} (partitions differ)"

    @pytest.mark.parametrize("k", [2, 5, 20, 50])
    def test_silhouette_widths_match(self, dist, r_labels, r_silhouette, k):
        labels = pam_cluster(dist, k, variant="pam")
        width_py = silhouette_score(dist, labels, metric="precomputed")
        width_r = float(r_silhouette.loc[r_silhouette["k"] == k, "avg_width"].iloc[0])
        assert width_py == pytest.approx(width_r, abs=1e-9)

    def test_silhouette_argmax_k_matches(self, dist, r_silhouette):
        ks = r_silhouette["k"].to_numpy()
        widths_r = r_silhouette["avg_width"].to_numpy()
        widths_py = np.array([
            silhouette_score(dist, pam_cluster(dist, int(k), variant="pam"),
                             metric="precomputed")
            for k in ks
        ])
        assert ks[np.argmax(widths_py)] == ks[np.argmax(widths_r)]
        np.testing.assert_allclose(widths_py, widths_r, atol=1e-9)

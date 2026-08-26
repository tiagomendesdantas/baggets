"""PAM clustering + silhouette-based k selection, matching the R pipeline.

The R code's ``TSclust::diss(X, "EUCL")`` is plain Euclidean distance between
raw series values, and ``cluster::pam`` is the classic BUILD+SWAP algorithm —
the ``kmedoids`` package's ``pam`` implements the same algorithm and reproduces
R's partitions on identical distance matrices. For silhouette k-sweeps we use
``fasterpam`` (better optima, faster) since the automatic-k choice is a search,
not a fidelity-critical reproduction — except in ``pam_variant="paper"`` mode
where ``pam`` is used everywhere.
"""

from __future__ import annotations

import kmedoids
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score

__all__ = ["euclidean_distance_matrix", "pam_cluster", "auto_k_silhouette"]


def euclidean_distance_matrix(series: np.ndarray) -> np.ndarray:
    """(B, T) series rows -> (B, B) Euclidean distance matrix."""
    return squareform(pdist(np.asarray(series, dtype=np.float64)))


def pam_cluster(dist: np.ndarray, k: int, variant: str = "pam", seed: int = 0) -> np.ndarray:
    """Cluster a precomputed distance matrix into ``k`` groups; returns 0-based labels."""
    if variant == "pam":
        result = kmedoids.pam(dist, medoids=k)
    elif variant == "fasterpam":
        result = kmedoids.fasterpam(dist, medoids=k, random_state=seed)
    else:
        raise ValueError(f"unknown PAM variant {variant!r}")
    return np.asarray(result.labels, dtype=np.int64)


def auto_k_silhouette(
    dist: np.ndarray,
    k_min: int = 2,
    k_max: int = 100,
    variant: str = "fasterpam",
    seed: int = 0,
) -> tuple[int, np.ndarray]:
    """Choose k maximizing average silhouette width over ``k_min..k_max`` (R: 2..100).

    Returns ``(k_best, labels_at_k_best)``. Ties break toward the smallest k
    (argmax of the score array), like R's ``which.max``.
    """
    n = dist.shape[0]
    k_max = min(k_max, n - 1)
    if k_max < k_min:
        raise ValueError(f"cannot cluster {n} points with k in [{k_min}, {k_max}]")
    ks = range(k_min, k_max + 1)
    scores = np.empty(len(ks))
    labels_by_k = []
    for i, k in enumerate(ks):
        labels = pam_cluster(dist, k, variant=variant, seed=seed)
        labels_by_k.append(labels)
        scores[i] = silhouette_score(dist, labels, metric="precomputed")
    best = int(np.argmax(scores))
    return k_min + best, labels_by_k[best]

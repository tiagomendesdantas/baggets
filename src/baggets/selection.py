"""Ensemble-selection strategies over a pool of bootstrapped members.

Every strategy consumes :class:`~baggets.validation.ValidationArtifacts` and
returns member indices (into ``0..B-1``); everything downstream — final fits,
aggregation, metrics — is strategy-agnostic. That separation is the research
surface of this package: the paper's cluster-based selection and its ablations
(top-k, random, none) are interchangeable here, and covariance-direct
strategies plug into the same slot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from .clustering import auto_k_silhouette, euclidean_distance_matrix, pam_cluster
from .validation import ValidationArtifacts

__all__ = [
    "SelectionStrategy",
    "ClusterSelection",
    "GreedyCovarianceSelection",
    "PortfolioSelection",
    "TopKSelection",
    "RandomSelection",
    "NoSelection",
]


@runtime_checkable
class SelectionStrategy(Protocol):
    name: str

    def select(
        self, artifacts: ValidationArtifacts, n_select: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Return indices of the selected members (dtype int64, ascending)."""
        ...


def _rank_by_error(val_errors: np.ndarray) -> np.ndarray:
    """Ascending stable order — ties resolved by lower index, like R's ties.method="first"."""
    return np.argsort(val_errors, kind="stable")


def _allocate_paper(cluster_sizes: np.ndarray, n_prefilter: int, n_select: int) -> np.ndarray:
    """R's allocation: round(N_h / n_prefilter * n_select), minimum 1 per cluster.

    Faithful to the 2018 code (np.round is half-to-even like R's round): the
    total can deviate slightly from ``n_select``.
    """
    n_h = np.round(cluster_sizes / n_prefilter * n_select).astype(np.int64)
    return np.maximum(n_h, 1)


def _allocate_largest_remainder(
    cluster_sizes: np.ndarray, n_prefilter: int, n_select: int
) -> np.ndarray:
    """Proportional allocation summing exactly to ``n_select`` (min 1 per cluster)."""
    quotas = cluster_sizes / n_prefilter * n_select
    n_h = np.floor(quotas).astype(np.int64)
    remainder = n_select - int(n_h.sum())
    if remainder > 0:
        frac_order = np.argsort(-(quotas - n_h), kind="stable")
        n_h[frac_order[:remainder]] += 1
    n_h = np.minimum(np.maximum(n_h, 1), cluster_sizes)
    # min-1 / cap fixups can leave the total off target: rebalance from/to the
    # largest allocations, never dropping a cluster below 1 or above its size.
    while n_h.sum() > n_select:
        n_h[np.argmax(n_h)] -= 1
    while n_h.sum() < n_select:
        room = cluster_sizes - n_h
        candidates = np.where(room > 0)[0]
        if candidates.size == 0:
            break
        n_h[candidates[np.argmax(room[candidates])]] += 1
    return n_h


@dataclass(frozen=True)
class ClusterSelection:
    """The paper's strategy: prefilter by validation error, PAM-cluster the raw
    series, then pick the best members of each cluster proportionally.

    ``n_clusters="auto"`` chooses k by maximum average silhouette width
    (the paper's headline configuration); an int fixes k (R's default was 5).
    """

    n_clusters: int | str = "auto"
    n_prefilter: int = 300
    k_max: int = 100
    allocation: str = "largest_remainder"  # or "paper"
    pam_variant: str = "fasterpam"  # or "pam" (exact R algorithm, slower sweeps)

    @classmethod
    def paper(cls, n_clusters: int | str = "auto") -> ClusterSelection:
        """The exact 2018 constants: 299 prefiltered members (R's ``rank < 300``),
        R's rounding allocation, classic PAM everywhere."""
        return cls(
            n_clusters=n_clusters,
            n_prefilter=299,
            k_max=100,
            allocation="paper",
            pam_variant="pam",
        )

    @property
    def name(self) -> str:
        k = self.n_clusters if isinstance(self.n_clusters, int) else "auto"
        return f"cluster(k={k})"

    def select(
        self, artifacts: ValidationArtifacts, n_select: int, rng: np.random.Generator
    ) -> np.ndarray:
        errors = artifacts.val_errors
        n_prefilter = min(self.n_prefilter, artifacts.n_members)
        if n_prefilter <= n_select:
            order = _rank_by_error(errors)
            return np.sort(order[:n_select])

        keep = np.sort(_rank_by_error(errors)[:n_prefilter])  # ascending indices, like R's which()
        dist = euclidean_distance_matrix(artifacts.series[keep])

        seed = int(rng.integers(0, 2**31 - 1))
        if self.n_clusters == "auto":
            _, labels = auto_k_silhouette(
                dist, k_min=2, k_max=self.k_max, variant=self.pam_variant, seed=seed
            )
        else:
            labels = pam_cluster(dist, int(self.n_clusters), variant=self.pam_variant, seed=seed)

        k = int(labels.max()) + 1
        artifacts.meta["k_chosen"] = k
        sizes = np.bincount(labels, minlength=k)
        if self.allocation == "paper":
            n_h = _allocate_paper(sizes, n_prefilter, n_select)
        elif self.allocation == "largest_remainder":
            n_h = _allocate_largest_remainder(sizes, n_prefilter, n_select)
        else:
            raise ValueError(f"unknown allocation {self.allocation!r}")

        selected: list[int] = []
        for h in range(k):
            members = keep[labels == h]  # ascending index order
            best_within = _rank_by_error(errors[members])[: n_h[h]]
            selected.extend(members[best_within])
        return np.sort(np.asarray(selected, dtype=np.int64))


@dataclass(frozen=True)
class GreedyCovarianceSelection:
    """Caruana-style forward ensemble selection on the validation window.

    Starts from the best single member and repeatedly adds the candidate that
    minimizes the validation error of the *aggregated* ensemble — optimizing
    the deployed quantity directly instead of the cluster proxy. Correlated
    members stop helping the aggregate, so decorrelation emerges from the
    objective itself (covariance-direct selection).

    ``aggregator`` should match the deployment aggregator ("median" default).
    ``patience`` stops early after that many consecutive non-improving
    additions (None = always fill ``n_select``).
    """

    aggregator: str = "median"
    metric: str = "mape"
    patience: int | None = None

    @property
    def name(self) -> str:
        return f"greedy({self.aggregator})"

    def select(
        self, artifacts: ValidationArtifacts, n_select: int, rng: np.random.Generator
    ) -> np.ndarray:
        from .metrics import mape, smape

        metric = {"mape": mape, "smape": smape}[self.metric]
        agg = {"median": np.median, "mean": np.mean}[self.aggregator]
        forecasts = artifacts.val_forecasts
        actuals = artifacts.val_actuals
        n_members = artifacts.n_members
        n_select = min(n_select, n_members)

        chosen = [int(np.argmin(artifacts.val_errors))]
        best_obj = float(artifacts.val_errors[chosen[0]])
        stall = 0
        available = np.ones(n_members, dtype=bool)
        available[chosen[0]] = False

        while len(chosen) < n_select and available.any():
            cand_idx = np.flatnonzero(available)
            # (n_cand, s+1, h): current ensemble + each candidate, aggregated at once
            current = forecasts[chosen]
            stacked = np.concatenate(
                [np.broadcast_to(current, (cand_idx.size, *current.shape)),
                 forecasts[cand_idx][:, None, :]],
                axis=1,
            )
            agg_fc = agg(stacked, axis=1)
            objs = np.array([metric(actuals, f) for f in agg_fc])
            best = int(np.argmin(objs))
            if self.patience is not None:
                if objs[best] >= best_obj:
                    stall += 1
                    if stall > self.patience:
                        break
                else:
                    stall = 0
            best_obj = min(best_obj, float(objs[best]))
            pick = int(cand_idx[best])
            chosen.append(pick)
            available[pick] = False
        return np.sort(np.asarray(chosen, dtype=np.int64))


@dataclass(frozen=True)
class PortfolioSelection:
    """Greedy minimization of the ensemble's estimated squared bias plus
    variance — the direct formalization of the paper's Eq. (4) motivation.

    Member validation errors ``E = F - a`` give per-member biases and a
    Ledoit-Wolf-shrunk covariance (h_val samples for B members: shrinkage is
    doing heavy lifting — documented caveat). The objective for subset S with
    equal weights is ``(mean b_S)^2 + (1/|S|^2) * sum(Sigma_S)``; each greedy
    step is O(B) via running sums.
    """

    @property
    def name(self) -> str:
        return "portfolio"

    def select(
        self, artifacts: ValidationArtifacts, n_select: int, rng: np.random.Generator
    ) -> np.ndarray:
        from sklearn.covariance import LedoitWolf

        errors = artifacts.val_forecasts - artifacts.val_actuals[None, :]  # (B, h)
        bias = errors.mean(axis=1)
        # samples = validation horizons, features = members
        cov = LedoitWolf(assume_centered=False).fit(errors.T).covariance_

        n_members = artifacts.n_members
        n_select = min(n_select, n_members)

        obj_single = bias**2 + np.diag(cov)
        first = int(np.argmin(obj_single))
        chosen = [first]
        available = np.ones(n_members, dtype=bool)
        available[first] = False

        sum_bias = bias[first]
        sum_cov = cov[first, first]
        cross = cov[first].copy()  # sum of covariance rows over chosen members

        while len(chosen) < n_select and available.any():
            s = len(chosen)
            new_bias = (sum_bias + bias) / (s + 1)
            new_var = (sum_cov + 2 * cross + np.diag(cov)) / (s + 1) ** 2
            objs = np.where(available, new_bias**2 + new_var, np.inf)
            pick = int(np.argmin(objs))
            chosen.append(pick)
            available[pick] = False
            sum_bias += bias[pick]
            sum_cov += 2 * cross[pick] + cov[pick, pick]
            cross += cov[pick]
        return np.sort(np.asarray(chosen, dtype=np.int64))


@dataclass(frozen=True)
class TopKSelection:
    """Best ``n_select`` members by validation error — accuracy without diversity,
    the ablation the 2018 paper lacked."""

    @property
    def name(self) -> str:
        return "topk"

    def select(
        self, artifacts: ValidationArtifacts, n_select: int, rng: np.random.Generator
    ) -> np.ndarray:
        order = _rank_by_error(artifacts.val_errors)
        return np.sort(order[: min(n_select, artifacts.n_members)])


@dataclass(frozen=True)
class RandomSelection:
    """Uniform random subset — does *which* members you pick matter at all?"""

    @property
    def name(self) -> str:
        return "random"

    def select(
        self, artifacts: ValidationArtifacts, n_select: int, rng: np.random.Generator
    ) -> np.ndarray:
        n = min(n_select, artifacts.n_members)
        return np.sort(rng.choice(artifacts.n_members, size=n, replace=False))


@dataclass(frozen=True)
class NoSelection:
    """All members, or the first ``n`` — ``NoSelection(100)`` plus mean aggregation
    reproduces Bergmeir et al.'s Bagged.BLD.MBB.ETS protocol."""

    n: int | None = None

    @property
    def name(self) -> str:
        return "none" if self.n is None else f"first{self.n}"

    def select(
        self, artifacts: ValidationArtifacts, n_select: int, rng: np.random.Generator
    ) -> np.ndarray:
        n = artifacts.n_members if self.n is None else min(self.n, artifacts.n_members)
        return np.arange(n, dtype=np.int64)

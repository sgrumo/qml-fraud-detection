"""QUBO feature selection, solvable by simulated annealing or D-Wave hardware.

Minimises ``-sum(rel_j z_j) + alpha * sum(red_jl z_j z_l) + penalty * (sum(z_j) - k)^2``
over binary ``z``, where relevance is mutual information with the label and
redundancy is absolute feature correlation, both normalised to [0, 1].
"""

from typing import Self

import dimod
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif

from qfd.types import FloatArray, IntArray


def build_selection_bqm(
    relevance: FloatArray, redundancy: FloatArray, k: int, alpha: float, penalty: float
) -> dimod.BinaryQuadraticModel:
    """Build the binary quadratic model for selecting ``k`` features.

    Args:
        relevance: Per-feature relevance, shape (d,).
        redundancy: Pairwise redundancy, shape (d, d).
        k: Number of features to select.
        alpha: Weight of the redundancy term.
        penalty: Weight of the cardinality constraint.

    Returns:
        The BQM over binary variables ``0..d-1``.
    """
    d = len(relevance)
    linear = {j: -relevance[j] + penalty * (1 - 2 * k) for j in range(d)}
    quadratic = {
        (j, m): alpha * redundancy[j, m] + 2 * penalty
        for j in range(d)
        for m in range(j + 1, d)
    }
    return dimod.BinaryQuadraticModel(linear, quadratic, penalty * k**2, dimod.BINARY)


def _normalise(values: FloatArray) -> FloatArray:
    """Scale values to [0, 1]."""
    span = values.max() - values.min()
    return (values - values.min()) / span if span > 0 else np.zeros_like(values)


class QuboFeatureSelector(TransformerMixin, BaseEstimator):  # type: ignore[misc]
    """Scikit-learn transformer selecting features by solving a QUBO.

    Args:
        k: Number of features to keep.
        alpha: Weight of the redundancy penalty.
        sampler: Sampler name accepted by
            :func:`qfd.hardware.dwave.make_qubo_sampler`.
        num_reads: Samples requested from the sampler.
        seed: Random seed for mutual information.
    """

    def __init__(
        self,
        k: int = 4,
        alpha: float = 0.5,
        sampler: str = "simulated",
        num_reads: int = 100,
        seed: int = 0,
    ) -> None:
        self.k = k
        self.alpha = alpha
        self.sampler = sampler
        self.num_reads = num_reads
        self.seed = seed

    def fit(self, X: FloatArray, y: IntArray) -> Self:
        """Select features on training data.

        Args:
            X: Training features.
            y: Training labels.

        Returns:
            The fitted selector.
        """
        from qfd.hardware.dwave import make_qubo_sampler  # noqa: PLC0415

        relevance = _normalise(mutual_info_classif(X, y, random_state=self.seed))
        redundancy = np.abs(np.nan_to_num(np.corrcoef(X, rowvar=False)))
        bqm = build_selection_bqm(
            relevance, redundancy, self.k, self.alpha, penalty=2.0
        )
        sampler = make_qubo_sampler(self.sampler)
        read_kwargs = {} if self.sampler == "exact" else {"num_reads": self.num_reads}
        best = sampler.sample(bqm, **read_kwargs).first.sample
        chosen = [j for j in range(X.shape[1]) if best[j] == 1]
        self.support_ = self._repair(chosen, relevance)
        return self

    def _repair(self, chosen: list[int], relevance: FloatArray) -> IntArray:
        """Enforce exactly ``k`` features using relevance as a tie-breaker."""
        ranked = sorted(chosen, key=lambda j: -relevance[j])[: self.k]
        for j in np.argsort(-relevance):
            if len(ranked) == self.k:
                break
            if j not in ranked:
                ranked.append(int(j))
        return np.sort(np.asarray(ranked, dtype=np.int64))

    def transform(self, X: FloatArray) -> FloatArray:
        """Keep the selected features.

        Args:
            X: Feature matrix.

        Returns:
            The selected columns.
        """
        return X[:, self.support_]

import numpy as np
import pytest

from qfd.data import negative_subsample
from qfd.evaluation import (
    bootstrap_ci,
    evaluate,
    geometric_difference,
    kernel_offdiag_stats,
    paired_wilcoxon,
    precision_at_k,
    recall_at_fpr,
    target_alignment,
)


def test_perfect_ranking_scores_one() -> None:
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])

    metrics = evaluate(y, scores, k=2)

    assert metrics["auprc"] == pytest.approx(1.0)
    assert metrics["precision_at_2"] == pytest.approx(1.0)
    assert metrics["prevalence"] == pytest.approx(0.4)


def test_precision_at_k_counts_top_alerts() -> None:
    y = np.array([1, 0, 1, 0])
    scores = np.array([4.0, 3.0, 2.0, 1.0])

    assert precision_at_k(y, scores, k=2) == pytest.approx(0.5)


def test_recall_at_fpr() -> None:
    y = np.array([0, 0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.95, 0.9, 0.4])

    assert recall_at_fpr(y, scores, fpr=0.0) == pytest.approx(0.0)
    assert recall_at_fpr(y, scores, fpr=0.25) == pytest.approx(1.0)


def test_reweighting_recovers_full_test_auprc(rng: np.random.Generator) -> None:
    y = (rng.random(200_000) < 0.005).astype(np.int64)
    scores = rng.normal(size=len(y)) + 2.0 * y
    full = evaluate(y, scores)["auprc"]

    idx, w = negative_subsample(y, rate=0.2, rng=rng)
    reweighted = evaluate(y[idx], scores[idx], w)["auprc"]
    unweighted = evaluate(y[idx], scores[idx])["auprc"]

    assert reweighted == pytest.approx(full, abs=0.03)
    assert unweighted > full + 0.05


def test_kernel_diagnostics_on_ideal_kernel() -> None:
    y = np.array([0, 0, 1, 1])
    s = 2.0 * y - 1.0
    K = np.outer(s, s)

    mean, var = kernel_offdiag_stats(np.eye(4))

    assert (mean, var) == (0.0, 0.0)
    assert target_alignment(K, y) == pytest.approx(1.0)


def test_geometric_difference_of_kernel_with_itself(rng: np.random.Generator) -> None:
    X = rng.normal(size=(20, 3))
    K = np.exp(-np.sum((X[:, None] - X[None]) ** 2, axis=-1))

    assert geometric_difference(K, K) == pytest.approx(1.0, rel=1e-3)


def test_paired_statistics() -> None:
    a = [0.70, 0.72, 0.71, 0.69, 0.73, 0.74, 0.70, 0.72]
    b = [0.60, 0.61, 0.63, 0.58, 0.62, 0.64, 0.61, 0.60]

    result = paired_wilcoxon(a, b)
    low, high = bootstrap_ci(a)

    assert result["median_diff"] > 0
    assert result["p_value"] < 0.05
    assert low < np.mean(a) < high

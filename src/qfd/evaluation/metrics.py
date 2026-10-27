"""Precision-oriented metrics with optional reweighting to natural prevalence."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from qfd.types import FloatArray, IntArray


def _weights(y: IntArray, weights: FloatArray | None) -> FloatArray:
    """Return explicit unit weights when none are given."""
    return np.ones(len(y), dtype=np.float64) if weights is None else weights


def precision_at_k(
    y: IntArray, scores: FloatArray, k: int, weights: FloatArray | None = None
) -> float:
    """Precision among the ``k`` highest-scored samples.

    With weights, the top of the ranking is taken until it represents ``k``
    samples of the original (unsubsampled) population.

    Args:
        y: Labels.
        scores: Fraud scores.
        k: Alert budget.
        weights: Optional sample weights.

    Returns:
        Weighted precision among the top alerts.
    """
    w = _weights(y, weights)
    order = np.argsort(-scores, kind="stable")
    top = order[: max(1, int(np.searchsorted(np.cumsum(w[order]), k, side="right")))]
    return float(np.sum(w[top] * y[top]) / np.sum(w[top]))


def recall_at_fpr(
    y: IntArray, scores: FloatArray, fpr: float, weights: FloatArray | None = None
) -> float:
    """Highest recall achievable with a false-positive rate at most ``fpr``.

    Args:
        y: Labels.
        scores: Fraud scores.
        fpr: Maximum false-positive rate.
        weights: Optional sample weights.

    Returns:
        Recall at the operating point.
    """
    fprs, tprs, _ = roc_curve(y, scores, sample_weight=weights)
    return float(tprs[fprs <= fpr].max())


def evaluate(
    y: IntArray, scores: FloatArray, weights: FloatArray | None = None, k: int = 100
) -> dict[str, float]:
    """Compute the metric set of docs/06 section 8.

    Args:
        y: Labels.
        scores: Fraud scores.
        weights: Optional sample weights from negative subsampling.
        k: Alert budget for precision@k.

    Returns:
        Mapping of metric name to value. ``prevalence`` is the AUPRC of a
        random ranking.
    """
    return {
        "auprc": float(average_precision_score(y, scores, sample_weight=weights)),
        "roc_auc": float(roc_auc_score(y, scores, sample_weight=weights)),
        f"precision_at_{k}": precision_at_k(y, scores, k, weights),
        "recall_at_fpr_0.001": recall_at_fpr(y, scores, 1e-3, weights),
        "recall_at_fpr_0.01": recall_at_fpr(y, scores, 1e-2, weights),
        "prevalence": float(np.average(y, weights=weights)),
    }

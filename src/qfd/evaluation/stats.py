"""Statistical comparisons over repeated runs (docs/06 section 10)."""

from collections.abc import Sequence

import numpy as np
from scipy.stats import bootstrap, wilcoxon


def paired_wilcoxon(a: Sequence[float], b: Sequence[float]) -> dict[str, float]:
    """Wilcoxon signed-rank test on paired metric values.

    Args:
        a: Metric of model A per repetition.
        b: Metric of model B on the same repetitions.

    Returns:
        Median difference (A - B) and two-sided p-value.

    Raises:
        ValueError: If the sequences have different lengths.
    """
    if len(a) != len(b):
        raise ValueError("Paired samples must have the same length")
    diff = np.asarray(a) - np.asarray(b)
    result = wilcoxon(diff)
    return {"median_diff": float(np.median(diff)), "p_value": float(result.pvalue)}


def bootstrap_ci(
    values: Sequence[float], confidence: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    """Bootstrap confidence interval of the mean.

    Args:
        values: Metric values, one per repetition.
        confidence: Confidence level.
        seed: Random seed.

    Returns:
        Lower and upper bounds.
    """
    res = bootstrap(
        (np.asarray(values),),
        np.mean,
        confidence_level=confidence,
        rng=np.random.default_rng(seed),
    )
    return float(res.confidence_interval.low), float(res.confidence_interval.high)

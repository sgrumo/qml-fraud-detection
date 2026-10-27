"""Temporal splits and budget-aware subsampling (see docs/06 sections 3 and 5)."""

from dataclasses import dataclass

import numpy as np

from qfd.types import FloatArray, IntArray


@dataclass(frozen=True, slots=True)
class TemporalSplit:
    """Index arrays of a train / validation / test split ordered in time."""

    train: IntArray
    val: IntArray
    test: IntArray


def temporal_split(
    time: FloatArray, val_fraction: float, test_fraction: float
) -> TemporalSplit:
    """Split samples by time: past for training, future for testing.

    Args:
        time: Timestamp of each sample.
        val_fraction: Fraction of samples, just before the test period, used
            for validation.
        test_fraction: Fraction of the latest samples used for testing.

    Returns:
        The split indices.

    Raises:
        ValueError: If the fractions leave no training data.
    """
    if not 0 < val_fraction + test_fraction < 1:
        raise ValueError("val_fraction + test_fraction must be in (0, 1)")
    order = np.argsort(time, kind="stable").astype(np.int64)
    n = len(order)
    n_test = round(n * test_fraction)
    n_val = round(n * val_fraction)
    n_train = n - n_val - n_test
    return TemporalSplit(
        train=order[:n_train],
        val=order[n_train : n_train + n_val],
        test=order[n_train + n_val :],
    )


def enriched_subsample(
    y: IntArray, n: int, fraud_rate: float, rng: np.random.Generator
) -> IntArray:
    """Draw a training subsample with a requested fraud rate.

    Args:
        y: Labels of the pool to sample from.
        n: Subsample size.
        fraud_rate: Target fraction of frauds; capped by available frauds.
        rng: Random generator.

    Returns:
        Positions into ``y`` of the subsample.

    Raises:
        ValueError: If the pool is too small.
    """
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    n_pos = min(round(n * fraud_rate), len(pos))
    if n - n_pos > len(neg):
        raise ValueError(f"Not enough genuine samples for a subsample of {n}")
    idx = np.concatenate(
        [
            rng.choice(pos, n_pos, replace=False),
            rng.choice(neg, n - n_pos, replace=False),
        ]
    )
    return rng.permutation(idx).astype(np.int64)


def one_class_subsample(y: IntArray, n: int, rng: np.random.Generator) -> IntArray:
    """Draw a genuine-only training subsample for novelty detection.

    Args:
        y: Labels of the pool to sample from.
        n: Subsample size; capped by available genuine samples.
        rng: Random generator.

    Returns:
        Positions into ``y`` of the subsample.
    """
    neg = np.flatnonzero(y == 0)
    return rng.choice(neg, min(n, len(neg)), replace=False).astype(np.int64)


def negative_subsample(
    y: IntArray, rate: float, rng: np.random.Generator
) -> tuple[IntArray, FloatArray]:
    """Keep all frauds and a fraction of genuine samples, with reweighting.

    Weights restore the natural prevalence in precision-based metrics.

    Args:
        y: Test labels.
        rate: Fraction of genuine samples to keep, in (0, 1].
        rng: Random generator.

    Returns:
        Kept positions (sorted) and their sample weights.

    Raises:
        ValueError: If ``rate`` is outside (0, 1].
    """
    if not 0 < rate <= 1:
        raise ValueError("rate must be in (0, 1]")
    if rate == 1:
        return np.arange(len(y), dtype=np.int64), np.ones(len(y), dtype=np.float64)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    kept_neg = rng.choice(neg, round(rate * len(neg)), replace=False)
    idx = np.sort(np.concatenate([pos, kept_neg])).astype(np.int64)
    weights = np.where(y[idx] == 1, 1.0, 1.0 / rate).astype(np.float64)
    return idx, weights

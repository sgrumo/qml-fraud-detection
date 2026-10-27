import numpy as np
import pytest

from qfd.data import (
    Dataset,
    enriched_subsample,
    negative_subsample,
    one_class_subsample,
    temporal_split,
)


def test_temporal_split_orders_periods(dataset: Dataset) -> None:
    split = temporal_split(dataset.time, val_fraction=0.1, test_fraction=0.3)

    assert dataset.time[split.train].max() < dataset.time[split.val].min()
    assert dataset.time[split.val].max() < dataset.time[split.test].min()
    assert len(split.train) + len(split.val) + len(split.test) == len(dataset.time)


def test_temporal_split_rejects_bad_fractions(dataset: Dataset) -> None:
    with pytest.raises(ValueError):
        temporal_split(dataset.time, val_fraction=0.5, test_fraction=0.5)


def test_enriched_subsample_rate(dataset: Dataset, rng: np.random.Generator) -> None:
    idx = enriched_subsample(dataset.y, n=200, fraud_rate=0.25, rng=rng)

    assert len(idx) == 200
    assert len(np.unique(idx)) == 200
    assert dataset.y[idx].sum() == 50


def test_one_class_subsample_is_genuine(
    dataset: Dataset, rng: np.random.Generator
) -> None:
    idx = one_class_subsample(dataset.y, n=100, rng=rng)

    assert len(idx) == 100
    assert dataset.y[idx].sum() == 0


def test_negative_subsample_weights_restore_counts(
    dataset: Dataset, rng: np.random.Generator
) -> None:
    idx, w = negative_subsample(dataset.y, rate=0.1, rng=rng)

    assert dataset.y[idx].sum() == dataset.y.sum()
    assert w[dataset.y[idx] == 0].sum() == pytest.approx(
        (dataset.y == 0).sum(), rel=0.01
    )


def test_negative_subsample_full_rate_keeps_everything(
    dataset: Dataset, rng: np.random.Generator
) -> None:
    idx, w = negative_subsample(dataset.y, rate=1.0, rng=rng)

    assert len(idx) == len(dataset.y)
    assert np.all(w == 1.0)

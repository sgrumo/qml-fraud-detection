import numpy as np
import pytest

from qfd.config import PreprocessingConfig
from qfd.data import Dataset, build_preprocessor


@pytest.mark.parametrize("reduction", ["first", "pca", "mutual_info", "qubo"])
def test_reduces_to_angle_range(dataset: Dataset, reduction: str) -> None:
    config = PreprocessingConfig(n_features=3, reduction=reduction)  # type: ignore[arg-type]
    pre = build_preprocessor(config, seed=0)

    out = pre.fit_transform(dataset.X[:2000], dataset.y[:2000])

    assert out.shape == (2000, 3)
    assert out.min() >= 0.0
    assert out.max() <= np.pi + 1e-12


def test_test_values_are_clipped(dataset: Dataset) -> None:
    pre = build_preprocessor(PreprocessingConfig(n_features=2, reduction="pca"))
    pre.fit(dataset.X[:1000], dataset.y[:1000])

    out = pre.transform(dataset.X[1000:] * 50)

    assert out.min() >= 0.0
    assert out.max() <= np.pi + 1e-12

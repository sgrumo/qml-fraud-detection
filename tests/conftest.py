import numpy as np
import pytest

from qfd.data import Dataset, make_synthetic


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def dataset() -> Dataset:
    return make_synthetic(n_samples=3000, n_features=6, fraud_rate=0.05, seed=0)


@pytest.fixture
def angles(rng: np.random.Generator) -> np.ndarray:
    """Twelve 3-feature samples already scaled to [0, pi]."""
    return rng.uniform(0, np.pi, size=(12, 3))


@pytest.fixture
def labelled_angles(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Separable 3-feature samples in [0, pi] with ~20 % positives."""
    X = rng.uniform(0, np.pi / 2, size=(40, 3))
    y = np.zeros(40, dtype=np.int64)
    y[:8] = 1
    X[:8] += np.pi / 2
    return X, y

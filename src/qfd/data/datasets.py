"""Dataset loaders returning time-ordered feature matrices."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.datasets import make_classification

from qfd.config import DataConfig
from qfd.types import FloatArray, IntArray

ULB_FEATURES = (*(f"V{i}" for i in range(1, 29)), "Amount")


@dataclass(frozen=True, slots=True)
class Dataset:
    """Time-ordered samples with binary fraud labels."""

    X: FloatArray
    y: IntArray
    time: FloatArray
    feature_names: tuple[str, ...]

    @property
    def prevalence(self) -> float:
        """Fraction of fraudulent samples."""
        return float(self.y.mean())


def load_ulb(path: Path) -> Dataset:
    """Load the ULB credit-card fraud dataset (Kaggle ``creditcard.csv``).

    Duplicate rows are dropped, rows are sorted by ``Time`` and ``Amount`` is
    log-transformed. ``V1``-``V28`` keep their original PCA order.

    Args:
        path: Path to ``creditcard.csv``.

    Returns:
        The dataset.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Download creditcard.csv from "
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud into data/raw/."
        )
    df = (
        pl.read_csv(path)
        .unique(maintain_order=True)
        .sort("Time")
        .with_columns(pl.col("Amount").log1p())
    )
    return Dataset(
        X=df.select(ULB_FEATURES).to_numpy().astype(np.float64),
        y=df["Class"].to_numpy().astype(np.int64),
        time=df["Time"].to_numpy().astype(np.float64),
        feature_names=ULB_FEATURES,
    )


def make_synthetic(
    n_samples: int = 20_000,
    n_features: int = 10,
    fraud_rate: float = 0.01,
    seed: int = 0,
) -> Dataset:
    """Generate an imbalanced synthetic dataset for smoke tests.

    Args:
        n_samples: Number of samples.
        n_features: Number of features.
        fraud_rate: Fraction of positive samples.
        seed: Random seed.

    Returns:
        The dataset, with a monotone synthetic time column.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, n_features // 2),
        weights=[1.0 - fraud_rate],
        class_sep=1.5,
        random_state=seed,
    )
    return Dataset(
        X=X.astype(np.float64),
        y=y.astype(np.int64),
        time=np.arange(n_samples, dtype=np.float64),
        feature_names=tuple(f"x{i}" for i in range(n_features)),
    )


def load_dataset(config: DataConfig, seed: int = 0) -> Dataset:
    """Load the dataset described by a data configuration.

    Args:
        config: Data configuration.
        seed: Seed used by synthetic data.

    Returns:
        The dataset.

    Raises:
        ValueError: If the ULB source has no path.
    """
    if config.source == "synthetic":
        return make_synthetic(seed=seed)
    if config.path is None:
        raise ValueError("data.path is required for the ULB dataset")
    return load_ulb(config.path)

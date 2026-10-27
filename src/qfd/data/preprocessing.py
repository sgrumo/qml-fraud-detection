"""Leakage-safe preprocessing: standardise, reduce to q features, scale to angles."""

from functools import partial

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, StandardScaler

from qfd.config import PreprocessingConfig
from qfd.feature_selection import QuboFeatureSelector
from qfd.types import FloatArray

ANGLE_RANGE = (0.0, float(np.pi))


def _take_first(X: FloatArray, k: int) -> FloatArray:
    """Keep the first ``k`` columns."""
    return X[:, :k]


def _reducer(config: PreprocessingConfig, seed: int) -> object:
    """Build the reduction step selected in the configuration."""
    k = config.n_features
    match config.reduction:
        case "first":
            return FunctionTransformer(partial(_take_first, k=k))
        case "pca":
            return PCA(n_components=k, random_state=seed)
        case "mutual_info":
            return SelectKBest(partial(mutual_info_classif, random_state=seed), k=k)
        case "qubo":
            return QuboFeatureSelector(k=k, sampler=config.qubo_sampler, seed=seed)


def build_preprocessor(config: PreprocessingConfig, seed: int = 0) -> Pipeline:
    """Build the preprocessing pipeline shared by classical and quantum models.

    Fit it on the training period only, then transform every split. Scaled
    features lie in ``[0, pi]``; test values outside the training range are
    clipped.

    Args:
        config: Preprocessing configuration.
        seed: Random seed for stochastic reducers.

    Returns:
        An unfitted scikit-learn pipeline.
    """
    return Pipeline(
        [
            ("standardise", StandardScaler()),
            ("reduce", _reducer(config, seed)),
            ("to_angles", MinMaxScaler(feature_range=ANGLE_RANGE, clip=True)),
        ]
    )

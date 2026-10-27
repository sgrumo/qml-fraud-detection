"""Classical baselines wrapped behind the :class:`qfd.types.Detector` protocol."""

from collections.abc import Callable
from typing import Protocol, Self

import numpy as np
from lightgbm import LGBMClassifier
from pyod.models.ecod import ECOD
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, OneClassSVM
from xgboost import XGBClassifier

from qfd.types import FloatArray, IntArray, Params


class Estimator(Protocol):
    """Minimal scikit-learn estimator interface."""

    def fit(self, X: FloatArray, y: IntArray | None = None) -> object:
        """Fit the estimator."""
        ...

    def get_params(self) -> dict[str, object]:
        """Return the estimator parameters."""
        ...

    def set_params(self, **params: object) -> object:
        """Set estimator parameters."""
        ...

    def decision_function(self, X: FloatArray) -> FloatArray:
        """Signed margin or anomaly score (not every estimator has it)."""
        ...

    def predict_proba(self, X: FloatArray) -> FloatArray:
        """Class probabilities (not every estimator has it)."""
        ...


class SupervisedDetector:
    """Classifier trained on both classes; scores with probabilities or margins."""

    is_one_class = False

    def __init__(self, estimator: Estimator) -> None:
        self.estimator = estimator

    def fit(self, X: FloatArray, y: IntArray) -> Self:
        """Fit the wrapped classifier.

        Args:
            X: Training features.
            y: Training labels.

        Returns:
            The fitted detector.
        """
        self.estimator.fit(X, y)
        return self

    def fraud_score(self, X: FloatArray) -> FloatArray:
        """Probability of fraud, or the decision margin when unavailable.

        Args:
            X: Features to score.

        Returns:
            Fraud scores.
        """
        if hasattr(self.estimator, "predict_proba"):
            scores = self.estimator.predict_proba(X)[:, 1]
        else:
            scores = self.estimator.decision_function(X)
        return np.asarray(scores, dtype=np.float64)


class OneClassDetector:
    """Novelty detector trained on genuine samples only."""

    is_one_class = True

    def __init__(self, estimator: Estimator, *, higher_is_normal: bool) -> None:
        self.estimator = estimator
        self.higher_is_normal = higher_is_normal

    def fit(self, X: FloatArray, y: IntArray) -> Self:
        """Fit on the genuine samples of ``X``.

        Args:
            X: Training features.
            y: Training labels; only rows with ``y == 0`` are used.

        Returns:
            The fitted detector.
        """
        self.estimator.fit(X[y == 0])
        return self

    def fraud_score(self, X: FloatArray) -> FloatArray:
        """Anomaly score; higher means more anomalous.

        Args:
            X: Features to score.

        Returns:
            Fraud scores.
        """
        scores = np.asarray(self.estimator.decision_function(X), dtype=np.float64)
        return -scores if self.higher_is_normal else scores


type EstimatorFactory = Callable[..., Estimator]

_SUPERVISED: dict[str, tuple[EstimatorFactory, Params]] = {
    "logistic_regression": (
        LogisticRegression,
        {"max_iter": 2000, "class_weight": "balanced"},
    ),
    "svm_rbf": (SVC, {"kernel": "rbf", "class_weight": "balanced"}),
    "random_forest": (
        RandomForestClassifier,
        {"n_estimators": 300, "class_weight": "balanced_subsample", "n_jobs": -1},
    ),
    "xgboost": (
        XGBClassifier,
        {"n_estimators": 300, "max_depth": 6, "eval_metric": "aucpr"},
    ),
    "lightgbm": (
        LGBMClassifier,
        {"n_estimators": 300, "learning_rate": 0.05, "verbose": -1},
    ),
}

_ONE_CLASS: dict[str, tuple[EstimatorFactory, Params, bool]] = {
    "ocsvm_rbf": (OneClassSVM, {"kernel": "rbf", "nu": 0.05}, True),
    "isolation_forest": (IsolationForest, {"n_estimators": 300}, True),
    "ecod": (ECOD, {}, False),
}

CLASSICAL_MODELS = tuple(sorted(_SUPERVISED | _ONE_CLASS))


def _instantiate(
    factory: EstimatorFactory, defaults: Params, params: Params, seed: int
) -> Estimator:
    """Create an estimator, seeding it when it accepts ``random_state``."""
    estimator = factory(**(defaults | params))
    if "random_state" in estimator.get_params():
        estimator.set_params(random_state=seed)
    return estimator


def build_classical(
    name: str, params: Params, seed: int = 0
) -> SupervisedDetector | OneClassDetector:
    """Build a classical detector from the registry.

    Args:
        name: One of :data:`CLASSICAL_MODELS`.
        params: Hyperparameters overriding the defaults.
        seed: Random seed.

    Returns:
        The detector.

    Raises:
        KeyError: If the model name is unknown.
    """
    if name in _SUPERVISED:
        cls, defaults = _SUPERVISED[name]
        return SupervisedDetector(_instantiate(cls, defaults, params, seed))
    if name in _ONE_CLASS:
        cls, defaults, higher_is_normal = _ONE_CLASS[name]
        return OneClassDetector(
            _instantiate(cls, defaults, params, seed), higher_is_normal=higher_is_normal
        )
    raise KeyError(f"Unknown classical model '{name}'; available: {CLASSICAL_MODELS}")

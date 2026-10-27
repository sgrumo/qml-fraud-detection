"""Shared type aliases and the detector protocol implemented by every model."""

from typing import Protocol, Self

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
ComplexArray = npt.NDArray[np.complex128]

type ParamValue = str | int | float | bool | None
type Params = dict[str, ParamValue]


class Detector(Protocol):
    """A fraud detector that ranks samples by how likely they are to be fraud."""

    @property
    def is_one_class(self) -> bool:
        """Whether the detector is trained on genuine samples only."""
        ...

    def fit(self, X: FloatArray, y: IntArray) -> Self:
        """Train the detector.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Labels, 1 for fraud and 0 for genuine.

        Returns:
            The fitted detector.
        """
        ...

    def fraud_score(self, X: FloatArray) -> FloatArray:
        """Score samples; higher values mean more likely fraud.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            One score per sample.
        """
        ...

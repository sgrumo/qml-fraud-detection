"""Quantum detectors: kernel SVMs and a variational classifier."""

from dataclasses import dataclass
from typing import Self

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.svm import SVC, OneClassSVM

from qfd.quantum.kernels import QuantumKernel
from qfd.quantum.qnodes import build_qnode
from qfd.types import FloatArray, IntArray


class _PrecomputedKernelSVM:
    """Shared logic for SVMs on a precomputed quantum Gram matrix.

    At scoring time only kernel columns of support vectors are evaluated; the
    other columns have zero dual coefficients, so they are left at zero.
    """

    def __init__(self, kernel: QuantumKernel, svm: SVC | OneClassSVM) -> None:
        self.kernel = kernel
        self.svm = svm

    def _fit_gram(self, X: FloatArray, y: IntArray | None) -> None:
        """Compute the Gram matrix and fit the SVM."""
        self.X_train_ = X
        self.gram_ = self.kernel.gram(X)
        self.svm.fit(self.gram_, y)

    def _decision(self, X: FloatArray) -> FloatArray:
        """Decision function evaluated against support vectors only."""
        support = self.svm.support_
        K = np.zeros((len(X), len(self.X_train_)), dtype=np.float64)
        K[:, support] = self.kernel(X, self.X_train_[support])
        return np.asarray(self.svm.decision_function(K), dtype=np.float64)


class QuantumKernelSVC(_PrecomputedKernelSVM):
    """Supervised SVM with a quantum kernel (QSVM).

    Args:
        kernel: Quantum kernel.
        C: Regularisation strength.
    """

    is_one_class = False

    def __init__(self, kernel: QuantumKernel, C: float = 1.0) -> None:
        super().__init__(
            kernel, SVC(kernel="precomputed", C=C, class_weight="balanced")
        )

    def fit(self, X: FloatArray, y: IntArray) -> Self:
        """Fit on labelled samples.

        Args:
            X: Training features scaled to rotation angles.
            y: Training labels.

        Returns:
            The fitted detector.
        """
        self._fit_gram(X, y)
        return self

    def fraud_score(self, X: FloatArray) -> FloatArray:
        """Signed distance to the decision boundary; higher means fraud.

        Args:
            X: Features to score.

        Returns:
            Fraud scores.
        """
        return self._decision(X)


class QuantumOneClassSVM(_PrecomputedKernelSVM):
    """One-class SVM with a quantum kernel, trained on genuine samples only.

    Args:
        kernel: Quantum kernel.
        nu: Upper bound on the fraction of training outliers.
    """

    is_one_class = True

    def __init__(self, kernel: QuantumKernel, nu: float = 0.05) -> None:
        super().__init__(kernel, OneClassSVM(kernel="precomputed", nu=nu))

    def fit(self, X: FloatArray, y: IntArray) -> Self:
        """Fit on the genuine samples of ``X``.

        Args:
            X: Training features scaled to rotation angles.
            y: Training labels; only rows with ``y == 0`` are used.

        Returns:
            The fitted detector.
        """
        self._fit_gram(X[y == 0], None)
        return self

    def fraud_score(self, X: FloatArray) -> FloatArray:
        """Negated one-class decision function; higher means more anomalous.

        Args:
            X: Features to score.

        Returns:
            Fraud scores.
        """
        return -self._decision(X)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Optimiser settings for variational models."""

    learning_rate: float = 0.05
    epochs: int = 20
    batch_size: int = 32
    seed: int = 0


class VariationalClassifier:
    """Re-uploading variational quantum classifier with a local observable.

    Each layer re-encodes the input with ``RY`` rotations followed by trainable
    ``StronglyEntanglingLayers``. The output is ``<Z_0> + bias``, trained with a
    class-balanced squared loss on targets in {-1, +1}.

    Args:
        n_qubits: Number of qubits (= features).
        n_layers: Re-uploading layers.
        training: Optimiser settings; defaults to :class:`TrainingConfig`.
        device: PennyLane device; defaults to ``default.qubit``.
        shots: Shots per circuit; ``None`` for analytic simulation.
    """

    is_one_class = False

    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 3,
        training: TrainingConfig | None = None,
        device: qml.devices.Device | None = None,
        shots: int | None = None,
    ) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.training = training or TrainingConfig()
        wires = list(range(n_qubits))

        def circuit(weights: FloatArray, x: FloatArray) -> object:
            for layer in range(n_layers):
                qml.AngleEmbedding(x, wires=wires, rotation="Y")
                qml.StronglyEntanglingLayers(weights[layer : layer + 1], wires=wires)
            return qml.expval(qml.PauliZ(0))

        self._circuit = build_qnode(
            circuit, device or qml.device("default.qubit", wires=n_qubits), shots=shots
        )

    def _loss(
        self,
        weights: FloatArray,
        bias: float,
        batch: tuple[FloatArray, FloatArray, FloatArray],
    ) -> float:
        """Class-weighted squared loss on one batch of (x, target, weight)."""
        X, target, w = batch
        preds = self._circuit(weights, X) + bias
        return pnp.sum(w * (preds - target) ** 2) / pnp.sum(w)  # type: ignore[no-any-return]

    def fit(self, X: FloatArray, y: IntArray) -> Self:
        """Train with Adam on mini-batches.

        Args:
            X: Training features scaled to rotation angles.
            y: Training labels.

        Returns:
            The fitted detector.
        """
        cfg = self.training
        rng = np.random.default_rng(cfg.seed)
        shape = qml.StronglyEntanglingLayers.shape(
            n_layers=self.n_layers, n_wires=self.n_qubits
        )
        weights = pnp.array(rng.uniform(0, 2 * np.pi, size=shape), requires_grad=True)
        bias = pnp.array(0.0, requires_grad=True)
        target = 2.0 * y - 1.0
        counts = np.bincount(y, minlength=2)
        sample_w = (len(y) / (2.0 * np.maximum(counts, 1)))[y]
        opt = qml.AdamOptimizer(stepsize=cfg.learning_rate)
        for _ in range(cfg.epochs):
            order = rng.permutation(len(y))
            for s in range(0, len(y), cfg.batch_size):
                idx = order[s : s + cfg.batch_size]
                batch = tuple(
                    pnp.array(a[idx], requires_grad=False)
                    for a in (X, target, sample_w)
                )
                (weights, bias), _ = opt.step_and_cost(
                    lambda w, b, batch=batch: self._loss(w, b, batch), weights, bias
                )
        self.weights_, self.bias_ = weights, bias
        return self

    def fraud_score(self, X: FloatArray) -> FloatArray:
        """Model output ``<Z_0> + bias``; higher means fraud.

        Args:
            X: Features to score.

        Returns:
            Fraud scores.
        """
        chunks = [
            np.atleast_1d(np.asarray(self._circuit(self.weights_, X[s : s + 4096])))
            for s in range(0, len(X), 4096)
        ]
        return np.concatenate(chunks).astype(np.float64) + float(self.bias_)

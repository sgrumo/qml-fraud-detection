"""Fidelity quantum kernels ``k(x, x') = |<psi(x)|psi(x')>|^2``.

:class:`StatevectorKernel` computes exact kernels from simulated states with one
matrix product. :class:`DeviceKernel` estimates each entry with the
compute-uncompute circuit on any PennyLane device: shot-based or noisy
simulators, AWS Braket, or Cirq.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from typing import Protocol

import numpy as np
import pennylane as qml

from qfd.quantum.embeddings import FeatureMap
from qfd.quantum.qnodes import build_qnode
from qfd.types import ComplexArray, FloatArray, IntArray


class QuantumKernel(Protocol):
    """A kernel function evaluated on batches of samples."""

    def __call__(self, XA: FloatArray, XB: FloatArray) -> FloatArray:
        """Kernel matrix of shape (len(XA), len(XB))."""
        ...

    def gram(self, X: FloatArray) -> FloatArray:
        """Symmetric Gram matrix of ``X``."""
        ...


@dataclass
class StatevectorKernel:
    """Exact fidelity kernel from statevectors (simulation only).

    Each sample is simulated once; the kernel is ``|S_A^* S_B^T|^2``.

    Args:
        feature_map: Encoding circuit.
        n_qubits: Number of qubits (= features).
        bandwidth: Factor multiplying inputs before encoding.
        batch_size: Samples simulated per broadcast batch.
    """

    feature_map: FeatureMap
    n_qubits: int
    bandwidth: float = 1.0
    batch_size: int = 1024

    @cached_property
    def _state_fn(self) -> qml.QNode:
        """QNode returning the encoded statevector."""
        wires = list(range(self.n_qubits))

        def circuit(x: FloatArray) -> object:
            self.feature_map(x, wires)
            return qml.state()

        return build_qnode(circuit, qml.device("default.qubit", wires=self.n_qubits))

    def states(self, X: FloatArray) -> ComplexArray:
        """Simulate the encoded state of every sample.

        Args:
            X: Samples, shape (n, q).

        Returns:
            Statevectors, shape (n, 2**q).
        """
        dim = 2**self.n_qubits
        chunks = [
            np.asarray(self._state_fn(self.bandwidth * X[s : s + self.batch_size]))
            for s in range(0, len(X), self.batch_size)
        ]
        return np.concatenate([c.reshape(-1, dim) for c in chunks]).astype(
            np.complex128
        )

    def __call__(self, XA: FloatArray, XB: FloatArray) -> FloatArray:
        """Kernel matrix between two sample sets."""
        overlaps = self.states(XA).conj() @ self.states(XB).T
        return np.abs(overlaps).astype(np.float64) ** 2

    def gram(self, X: FloatArray) -> FloatArray:
        """Gram matrix of one sample set."""
        S = self.states(X)
        return np.abs(S.conj() @ S.T).astype(np.float64) ** 2


class PairwiseKernel(ABC):
    """Kernel estimated pair by pair, in batches, from circuit executions."""

    bandwidth: float
    batch_size: int

    @abstractmethod
    def _batch_values(self, A: FloatArray, B: FloatArray) -> FloatArray:
        """Kernel values ``k(A[t], B[t])`` for already-scaled row pairs."""

    def _evaluate_pairs(
        self, XA: FloatArray, XB: FloatArray, ia: IntArray, ib: IntArray
    ) -> FloatArray:
        """Kernel values for the index pairs ``(ia[t], ib[t])``."""
        values = np.empty(len(ia), dtype=np.float64)
        for s in range(0, len(ia), self.batch_size):
            a = self.bandwidth * XA[ia[s : s + self.batch_size]]
            b = self.bandwidth * XB[ib[s : s + self.batch_size]]
            values[s : s + len(a)] = self._batch_values(a, b)
        return values

    def __call__(self, XA: FloatArray, XB: FloatArray) -> FloatArray:
        """Kernel matrix between two sample sets."""
        ia, ib = np.indices((len(XA), len(XB))).reshape(2, -1).astype(np.int64)
        return self._evaluate_pairs(XA, XB, ia, ib).reshape(len(XA), len(XB))

    def gram(self, X: FloatArray) -> FloatArray:
        """Gram matrix using only the strict upper triangle; the diagonal is 1."""
        ia, ib = np.triu_indices(len(X), k=1)
        K = np.eye(len(X), dtype=np.float64)
        K[ia, ib] = K[ib, ia] = self._evaluate_pairs(
            X, X, ia.astype(np.int64), ib.astype(np.int64)
        )
        return K


@dataclass
class DeviceKernel(PairwiseKernel):
    """Fidelity kernel estimated with compute-uncompute circuits on a device.

    ``k(x, x')`` is the probability of measuring ``|0...0>`` after
    ``U(x)`` then ``U(x')^dagger``. Pairs are evaluated in broadcast batches.

    Args:
        feature_map: Encoding circuit.
        device: PennyLane device with at least ``n_qubits`` wires.
        n_qubits: Number of qubits (= features).
        bandwidth: Factor multiplying inputs before encoding.
        shots: Shots per circuit; ``None`` for analytic simulators.
    """

    feature_map: FeatureMap
    device: qml.devices.Device
    n_qubits: int
    bandwidth: float = 1.0
    shots: int | None = None
    batch_size: int = field(default=256, kw_only=True)

    @cached_property
    def _overlap_fn(self) -> qml.QNode:
        """QNode returning computational-basis probabilities."""
        wires = list(range(self.n_qubits))

        def circuit(x1: FloatArray, x2: FloatArray) -> object:
            self.feature_map(x1, wires)
            qml.adjoint(self.feature_map)(x2, wires)
            return qml.probs(wires=wires)

        return build_qnode(circuit, self.device, shots=self.shots)

    def _batch_values(self, A: FloatArray, B: FloatArray) -> FloatArray:
        """Probability of the all-zero outcome for each pair."""
        probs = np.asarray(self._overlap_fn(A, B)).reshape(len(A), -1)
        return probs[:, 0].astype(np.float64)

"""Kernel diagnostics explaining quantum kernel behaviour (docs/06 section 9)."""

import numpy as np

from qfd.types import FloatArray, IntArray


def kernel_offdiag_stats(K: FloatArray) -> tuple[float, float]:
    """Mean and variance of off-diagonal Gram entries.

    Exponentially shrinking variance with qubit count signals concentration.

    Args:
        K: Square Gram matrix.

    Returns:
        Mean and variance of the off-diagonal entries.
    """
    off = K[~np.eye(len(K), dtype=bool)]
    return float(off.mean()), float(off.var())


def target_alignment(K: FloatArray, y: IntArray) -> float:
    """Kernel-target alignment with labels mapped to {-1, +1}.

    Args:
        K: Square Gram matrix.
        y: Labels in {0, 1}.

    Returns:
        Alignment in [-1, 1].
    """
    s = 2.0 * y - 1.0
    Y = np.outer(s, s)
    return float(np.sum(K * Y) / (np.linalg.norm(K) * np.linalg.norm(Y)))


def _sqrt_psd(K: FloatArray) -> FloatArray:
    """Matrix square root of a symmetric positive semi-definite matrix."""
    vals, vecs = np.linalg.eigh((K + K.T) / 2)
    return np.asarray(
        (vecs * np.sqrt(np.clip(vals, 0, None))) @ vecs.T, dtype=np.float64
    )


def geometric_difference(
    K_classical: FloatArray, K_quantum: FloatArray, reg: float = 1e-6
) -> float:
    """Geometric difference g(K_C || K_Q) of Huang et al. (2021).

    A small value means a classical model with ``K_classical`` can match the
    quantum kernel on this data.

    Args:
        K_classical: Classical Gram matrix.
        K_quantum: Quantum Gram matrix on the same samples.
        reg: Ridge added to ``K_classical`` before inversion.

    Returns:
        The geometric difference.
    """
    n = len(K_classical)
    Kc = K_classical * n / np.trace(K_classical)
    sq = _sqrt_psd(K_quantum * n / np.trace(K_quantum))
    M = sq @ np.linalg.solve(Kc + reg * np.eye(n), sq)
    return float(np.sqrt(np.max(np.abs(np.linalg.eigvalsh((M + M.T) / 2)))))

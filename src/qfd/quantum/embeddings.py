"""Feature maps that encode classical vectors into quantum states."""

from collections.abc import Callable, Sequence
from functools import partial
from itertools import pairwise

import pennylane as qml

from qfd.types import FloatArray

type FeatureMap = Callable[[FloatArray, Sequence[int]], None]


def angle_map(x: FloatArray, wires: Sequence[int], reps: int = 1) -> None:
    """Product-state angle encoding with ``RY`` rotations.

    Its kernel is classically trivial; use it as the control model.

    Args:
        x: Features, shape (q,) or (batch, q).
        wires: Wires to encode on.
        reps: Ignored; repeated ``RY`` rotations only rescale the angle.
    """
    del reps
    qml.AngleEmbedding(x, wires=wires, rotation="Y")


def iqp_map(x: FloatArray, wires: Sequence[int], reps: int = 2) -> None:
    """IQP / ZZ-style entangling encoding (Havlicek et al., 2019).

    Args:
        x: Features, shape (q,) or (batch, q).
        wires: Wires to encode on.
        reps: Number of encoding repetitions.
    """
    qml.IQPEmbedding(x, wires=wires, n_repeats=reps)


def angle_entangled_map(x: FloatArray, wires: Sequence[int], reps: int = 2) -> None:
    """Data re-uploading: ``RY`` encoding followed by a ring of ``CZ`` gates.

    Args:
        x: Features, shape (q,) or (batch, q).
        wires: Wires to encode on.
        reps: Number of encoding repetitions.
    """
    ring = list(wires)
    pairs = list(pairwise(ring))
    if len(ring) > 2:
        pairs.append((ring[-1], ring[0]))
    for _ in range(reps):
        qml.AngleEmbedding(x, wires=ring, rotation="Y")
        for a, b in pairs:
            qml.CZ(wires=[a, b])


FEATURE_MAPS: dict[str, Callable[..., None]] = {
    "angle": angle_map,
    "iqp": iqp_map,
    "angle_entangled": angle_entangled_map,
}


def make_feature_map(name: str, reps: int = 2) -> FeatureMap:
    """Look up a feature map and fix its number of repetitions.

    Args:
        name: One of ``angle``, ``iqp``, ``angle_entangled``.
        reps: Encoding repetitions.

    Returns:
        A callable ``(x, wires) -> None`` applying the encoding.

    Raises:
        KeyError: If the name is unknown.
    """
    if name not in FEATURE_MAPS:
        raise KeyError(
            f"Unknown feature map '{name}'; available: {sorted(FEATURE_MAPS)}"
        )
    return partial(FEATURE_MAPS[name], reps=reps)

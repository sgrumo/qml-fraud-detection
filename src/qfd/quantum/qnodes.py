"""QNode construction with optional finite shots."""

from collections.abc import Callable

import pennylane as qml


def build_qnode(
    fn: Callable[..., object],
    device: qml.devices.Device,
    shots: int | None = None,
    diff_method: str = "best",
) -> qml.QNode:
    """Wrap a quantum function as a QNode.

    Args:
        fn: Quantum function.
        device: Execution device.
        shots: Shots per execution; ``None`` keeps the device default
            (analytic for simulators).
        diff_method: PennyLane differentiation method.

    Returns:
        The QNode.
    """
    qnode = qml.QNode(fn, device, diff_method=diff_method)
    if shots is not None:
        qnode = qml.set_shots(qnode, shots=shots)
    return qnode

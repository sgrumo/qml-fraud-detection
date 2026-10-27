"""Local PennyLane simulators, optionally with depolarising noise."""

import pennylane as qml

SIMULATORS = (
    "default.qubit",
    "lightning.qubit",
    "lightning.gpu",
    "default.mixed",
    "cirq.simulator",
    "cirq.qsim",
    "braket.local.qubit",
)


def make_simulator(
    name: str, n_qubits: int, depolarizing: float | None = None
) -> qml.devices.Device:
    """Create a local simulator device.

    Depolarising noise is applied after every gate and requires the
    density-matrix simulator ``default.mixed``. ``cirq.*`` devices need the
    ``google`` extra and ``braket.local.qubit`` the ``braket`` extra.

    Args:
        name: One of :data:`SIMULATORS`.
        n_qubits: Number of wires.
        depolarizing: Depolarising probability per gate, or ``None``.

    Returns:
        The device, wrapped with a noise transform when requested.

    Raises:
        ValueError: If the name is unknown or noise is requested on a
            statevector simulator.
    """
    if name not in SIMULATORS:
        raise ValueError(f"Unknown simulator '{name}'; available: {SIMULATORS}")
    device = qml.device(name, wires=n_qubits)
    if depolarizing is None:
        return device
    if name != "default.mixed":
        raise ValueError("Depolarising noise requires the 'default.mixed' simulator")
    noise = qml.NoiseModel(
        {
            qml.noise.wires_in(range(n_qubits)): qml.noise.partial_wires(
                qml.DepolarizingChannel, depolarizing
            )
        }
    )
    return qml.add_noise(device, noise)

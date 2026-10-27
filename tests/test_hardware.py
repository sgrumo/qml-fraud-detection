import os

import numpy as np
import pennylane as qml
import pytest

from qfd.hardware import (
    HardwareBudgetError,
    HardwareConfigurationError,
    check_hardware_budget,
    estimate_kernel_circuits,
)
from qfd.hardware.dwave import make_qubo_sampler
from qfd.quantum import StatevectorKernel, make_feature_map


def test_circuit_estimate() -> None:
    assert estimate_kernel_circuits(n_train=10, n_test=5) == 45 + 50


def test_budget_requires_explicit_permission() -> None:
    with pytest.raises(HardwareBudgetError, match="--allow-hardware"):
        check_hardware_budget(10, 100, allowed=False)


def test_budget_rejects_oversized_runs() -> None:
    with pytest.raises(HardwareBudgetError, match="exceed"):
        check_hardware_budget(1_000, 100, allowed=True)


def test_remote_dwave_sampler_needs_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DWAVE_API_TOKEN", raising=False)

    with pytest.raises(HardwareConfigurationError, match="DWAVE_API_TOKEN"):
        make_qubo_sampler("qpu")


def test_braket_device_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    from qfd.hardware.braket import braket_device

    captured: dict[str, object] = {}

    def fake_device(name: str, **kwargs: object) -> str:
        captured.update(name=name, **kwargs)
        return "device"

    monkeypatch.setattr(qml, "device", fake_device)

    braket_device("arn:sv1", n_qubits=4, shots=100, s3_bucket="bucket")

    assert captured["name"] == "braket.aws.qubit"
    assert captured["device_arn"] == "arn:sv1"
    assert captured["s3_destination_folder"] == ("bucket", "qfd")


def test_braket_rejects_analytic_mode() -> None:
    from qfd.hardware.braket import braket_device

    with pytest.raises(HardwareConfigurationError, match="shots"):
        braket_device("arn:sv1", n_qubits=4, shots=0)


def test_cirq_sampler_kernel_matches_statevector(angles: np.ndarray) -> None:
    cirq = pytest.importorskip("cirq")
    from qfd.hardware.google import CirqSamplerKernel

    feature_map = make_feature_map("iqp", reps=1)
    exact = StatevectorKernel(feature_map, n_qubits=3, bandwidth=0.5)
    sampled = CirqSamplerKernel(
        feature_map, cirq.Simulator(seed=0), n_qubits=3, bandwidth=0.5, shots=4000
    )

    np.testing.assert_allclose(
        sampled(angles[:2], angles[2:4]), exact(angles[:2], angles[2:4]), atol=0.05
    )


def test_cirq_kernel_routes_onto_device_graph(angles: np.ndarray) -> None:
    cirq = pytest.importorskip("cirq")
    import networkx as nx

    from qfd.hardware.google import CirqSamplerKernel

    qubits = cirq.GridQubit.rect(1, 3)
    graph = nx.Graph([(qubits[0], qubits[1]), (qubits[1], qubits[2])])
    kernel = CirqSamplerKernel(
        make_feature_map("iqp", reps=1),
        cirq.Simulator(seed=0),
        n_qubits=3,
        device_graph=graph,
    )

    circuit = kernel.to_cirq(angles[0], angles[1])

    assert circuit.all_qubits() <= set(qubits)


@pytest.mark.hardware
def test_braket_sv1_round_trip() -> None:
    """Submits real tasks to AWS Braket SV1 (paid). Run with: pytest -m hardware."""
    from qfd.hardware.braket import braket_device
    from qfd.quantum import DeviceKernel

    if not os.environ.get("AWS_PROFILE") and not os.environ.get("AWS_ACCESS_KEY_ID"):
        pytest.skip("AWS credentials not configured")
    device = braket_device(
        "arn:aws:braket:::device/quantum-simulator/amazon/sv1", 2, shots=100
    )
    kernel = DeviceKernel(make_feature_map("angle"), device, n_qubits=2, shots=100)

    K = kernel.gram(np.array([[0.1, 0.2], [0.3, 0.4]]))

    assert K.shape == (2, 2)

import numpy as np
import pennylane as qml
import pytest

from qfd.quantum import DeviceKernel, StatevectorKernel, make_feature_map


@pytest.mark.parametrize("name", ["angle", "iqp", "angle_entangled"])
def test_statevector_gram_is_valid_kernel(angles: np.ndarray, name: str) -> None:
    kernel = StatevectorKernel(make_feature_map(name, reps=2), n_qubits=3)

    K = kernel.gram(angles)

    np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-10)
    np.testing.assert_allclose(K, K.T, atol=1e-12)
    assert np.linalg.eigvalsh(K).min() > -1e-9


def test_angle_kernel_matches_closed_form(angles: np.ndarray) -> None:
    kernel = StatevectorKernel(make_feature_map("angle"), n_qubits=3, bandwidth=0.7)

    K = kernel(angles[:4], angles[4:])

    diff = 0.7 * (angles[:4, None, :] - angles[None, 4:, :])
    np.testing.assert_allclose(K, np.prod(np.cos(diff / 2) ** 2, axis=-1), atol=1e-10)


@pytest.mark.parametrize("name", ["iqp", "angle_entangled"])
def test_device_kernel_matches_statevector(angles: np.ndarray, name: str) -> None:
    feature_map = make_feature_map(name, reps=2)
    exact = StatevectorKernel(feature_map, n_qubits=3, bandwidth=0.5)
    circuit = DeviceKernel(
        feature_map, qml.device("default.qubit", wires=3), n_qubits=3, bandwidth=0.5
    )

    np.testing.assert_allclose(circuit.gram(angles), exact.gram(angles), atol=1e-8)
    np.testing.assert_allclose(
        circuit(angles[:5], angles[5:]), exact(angles[:5], angles[5:]), atol=1e-8
    )


def test_shot_kernel_is_close_to_exact(angles: np.ndarray) -> None:
    feature_map = make_feature_map("iqp", reps=1)
    exact = StatevectorKernel(feature_map, n_qubits=3, bandwidth=0.5)
    shots = DeviceKernel(
        feature_map,
        qml.device("default.qubit", wires=3, seed=1),
        n_qubits=3,
        bandwidth=0.5,
        shots=4000,
    )

    np.testing.assert_allclose(
        shots(angles[:3], angles[3:6]), exact(angles[:3], angles[3:6]), atol=0.05
    )

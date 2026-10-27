import numpy as np
import pytest

from qfd.quantum import DeviceKernel, make_feature_map
from qfd.simulation import make_simulator


def test_unknown_simulator_raises() -> None:
    with pytest.raises(ValueError, match="Unknown simulator"):
        make_simulator("not.a.device", 2)


def test_noise_requires_density_matrix_simulator() -> None:
    with pytest.raises(ValueError, match=r"default\.mixed"):
        make_simulator("default.qubit", 2, depolarizing=0.01)


def test_depolarising_noise_shrinks_self_overlap(angles: np.ndarray) -> None:
    device = make_simulator("default.mixed", 3, depolarizing=0.05)
    kernel = DeviceKernel(make_feature_map("iqp", reps=1), device, n_qubits=3)

    self_overlap = np.diag(kernel(angles[:3], angles[:3]))

    assert np.all(self_overlap < 0.99)

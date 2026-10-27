"""Build any detector from its configuration, for the configured stage."""

import pennylane as qml

from qfd.classical import CLASSICAL_MODELS, build_classical
from qfd.config import DeviceConfig, ModelConfig, Stage
from qfd.params import get_float, get_int, get_str
from qfd.quantum import (
    DeviceKernel,
    QuantumKernel,
    QuantumKernelSVC,
    QuantumOneClassSVM,
    StatevectorKernel,
    TrainingConfig,
    VariationalClassifier,
    make_feature_map,
)
from qfd.simulation import make_simulator
from qfd.types import Detector, Params

QUANTUM_MODELS = ("qsvc", "qocsvm", "vqc")
HARDWARE_BACKENDS = ("braket", "google_engine")


def _option(options: Params, key: str) -> str:
    """Read a required string option of a hardware backend."""
    value = options.get(key)
    if not isinstance(value, str):
        raise ValueError(f"device.options.{key} is required for this backend")
    return value


def _pennylane_device(device: DeviceConfig, n_qubits: int) -> qml.devices.Device:
    """Device for the ``braket`` backend or a local simulator."""
    if device.backend == "braket":
        from qfd.hardware.braket import braket_device  # noqa: PLC0415

        bucket = device.options.get("s3_bucket")
        return braket_device(
            _option(device.options, "device_arn"),
            n_qubits,
            shots=device.shots or 1000,
            s3_bucket=bucket if isinstance(bucket, str) else None,
        )
    return make_simulator(device.backend, n_qubits, device.depolarizing)


def build_kernel(params: Params, device: DeviceConfig, n_qubits: int) -> QuantumKernel:
    """Build the quantum kernel described by model parameters and device.

    Args:
        params: Model parameters (``feature_map``, ``reps``, ``bandwidth``).
        device: Execution backend.
        n_qubits: Number of qubits.

    Returns:
        The kernel.
    """
    feature_map = make_feature_map(
        get_str(params, "feature_map", "iqp"), get_int(params, "reps", 2)
    )
    bandwidth = get_float(params, "bandwidth", 1.0)
    if device.backend == "statevector":
        return StatevectorKernel(feature_map, n_qubits, bandwidth)
    if device.backend == "google_engine":
        from qfd.hardware.google import (  # noqa: PLC0415
            CirqSamplerKernel,
            google_engine_sampler,
        )

        sampler, graph = google_engine_sampler(
            _option(device.options, "project_id"),
            _option(device.options, "processor_id"),
        )
        return CirqSamplerKernel(
            feature_map,
            sampler,
            n_qubits,
            bandwidth,
            device.shots or 1000,
            device_graph=graph,
        )
    return DeviceKernel(
        feature_map,
        _pennylane_device(device, n_qubits),
        n_qubits,
        bandwidth,
        device.shots,
    )


def build_detector(
    model: ModelConfig, stage: Stage, device: DeviceConfig, n_qubits: int, seed: int
) -> Detector:
    """Build a detector, checking that it belongs to the configured stage.

    Args:
        model: Model name and parameters.
        stage: Experiment stage.
        device: Execution backend for quantum models.
        n_qubits: Number of qubits (= reduced features).
        seed: Random seed.

    Returns:
        The detector.

    Raises:
        ValueError: If a quantum model is used in the classical stage.
        KeyError: If the model name is unknown.
    """
    if model.name in CLASSICAL_MODELS:
        return build_classical(model.name, model.params, seed)
    if model.name not in QUANTUM_MODELS:
        raise KeyError(f"Unknown model '{model.name}'")
    if stage is Stage.CLASSICAL:
        raise ValueError(
            f"Quantum model '{model.name}' is not allowed in the classical stage"
        )
    p = model.params
    if model.name == "vqc":
        training = TrainingConfig(
            learning_rate=get_float(p, "learning_rate", 0.05),
            epochs=get_int(p, "epochs", 20),
            batch_size=get_int(p, "batch_size", 32),
            seed=seed,
        )
        backend = (
            None
            if device.backend == "statevector"
            else _pennylane_device(device, n_qubits)
        )
        return VariationalClassifier(
            n_qubits, get_int(p, "layers", 3), training, backend, device.shots
        )
    kernel = build_kernel(p, device, n_qubits)
    if model.name == "qsvc":
        return QuantumKernelSVC(kernel, C=get_float(p, "C", 1.0))
    return QuantumOneClassSVM(kernel, nu=get_float(p, "nu", 0.05))

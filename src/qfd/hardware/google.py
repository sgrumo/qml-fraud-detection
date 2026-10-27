"""Google Quantum AI stack: Cirq samplers, local or on Quantum Engine.

PennyLane circuits are exported to OpenQASM and imported into Cirq, so the same
feature maps run on any ``cirq.Sampler``: ``cirq.Simulator`` locally, or a
Quantum Engine processor sampler. Requires ``pip install -e ".[google]"``.

Quantum Engine hardware is available only to Google Cloud projects approved by
Google Quantum AI. Authenticate with ``gcloud auth application-default login``
and set ``GOOGLE_CLOUD_PROJECT``.
"""

from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import pennylane as qml

from qfd.hardware.errors import HardwareConfigurationError
from qfd.quantum.embeddings import FeatureMap
from qfd.quantum.kernels import PairwiseKernel
from qfd.types import FloatArray

try:
    import cirq
    from cirq.contrib.qasm_import import circuit_from_qasm
except ImportError as exc:  # pragma: no cover - exercised without the extra
    raise HardwareConfigurationError(
        'Install the Google extra: pip install -e ".[google]"'
    ) from exc


@dataclass
class CirqSamplerKernel(PairwiseKernel):
    """Fidelity kernel estimated on a Cirq sampler.

    Args:
        feature_map: Encoding circuit.
        sampler: Any ``cirq.Sampler`` (``cirq.Simulator`` or an Engine sampler).
        n_qubits: Number of qubits (= features).
        bandwidth: Factor multiplying inputs before encoding.
        shots: Repetitions per circuit.
        device_graph: Hardware connectivity; when given, circuits are compiled
            to the CZ gate set and routed onto it.
    """

    feature_map: FeatureMap
    sampler: cirq.Sampler
    n_qubits: int
    bandwidth: float = 1.0
    shots: int = 1000
    device_graph: nx.Graph | None = field(default=None, kw_only=True)
    batch_size: int = field(default=64, kw_only=True)

    def to_cirq(self, x1: FloatArray, x2: FloatArray) -> cirq.Circuit:
        """Build the compute-uncompute circuit for one pair as a Cirq circuit.

        Args:
            x1: First (scaled) sample.
            x2: Second (scaled) sample.

        Returns:
            The measured Cirq circuit, compiled and routed if a device graph
            is set.
        """
        wires = list(range(self.n_qubits))

        def circuit(a: FloatArray, b: FloatArray) -> object:
            self.feature_map(a, wires)
            qml.adjoint(self.feature_map)(b, wires)
            return qml.probs(wires=wires)

        qnode = qml.QNode(circuit, qml.device("default.qubit", wires=self.n_qubits))
        cirq_circuit = circuit_from_qasm(
            qml.to_openqasm(qnode, measure_all=True)(x1, x2)
        )
        if self.device_graph is None:
            return cirq_circuit
        compiled = cirq.optimize_for_target_gateset(
            cirq_circuit, gateset=cirq.CZTargetGateset()
        )
        routed, _, _ = cirq.RouteCQC(self.device_graph).route_circuit(compiled)
        return routed.unfreeze(copy=False)

    def _batch_values(self, A: FloatArray, B: FloatArray) -> FloatArray:
        """Fraction of all-zero outcomes for each pair."""
        circuits = [self.to_cirq(a, b) for a, b in zip(A, B, strict=True)]
        results = self.sampler.run_batch(circuits, repetitions=self.shots)
        values = np.empty(len(circuits), dtype=np.float64)
        for t, (result,) in enumerate(results):
            bits = np.hstack(list(result.measurements.values()))
            values[t] = np.mean(~bits.any(axis=1))
        return values


def google_engine_sampler(
    project_id: str, processor_id: str
) -> tuple[cirq.Sampler, nx.Graph]:
    """Sampler and connectivity graph of a Quantum Engine processor.

    Args:
        project_id: Approved Google Cloud project.
        processor_id: Engine processor name.

    Returns:
        The processor sampler and its qubit connectivity graph.

    Raises:
        HardwareConfigurationError: If ``cirq-google`` is not installed.
    """
    try:
        import cirq_google  # noqa: PLC0415
    except ImportError as exc:
        raise HardwareConfigurationError(
            'Install the Google extra: pip install -e ".[google]"'
        ) from exc
    processor = cirq_google.get_engine(project_id).get_processor(processor_id)
    metadata = processor.get_device().metadata
    if metadata is None:
        raise HardwareConfigurationError(
            f"Processor '{processor_id}' does not expose its qubit connectivity"
        )
    return processor.get_sampler(), metadata.nx_graph

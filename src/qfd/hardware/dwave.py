"""D-Wave Ocean samplers for QUBO problems, local or on Leap hardware.

Local samplers (``exact``, ``simulated``) need only the core install. The
``qpu`` and ``hybrid`` samplers need ``pip install -e ".[dwave]"`` and a
``DWAVE_API_TOKEN`` environment variable (see ``.env.example``).
"""

import os

import dimod
from dwave.samplers import SimulatedAnnealingSampler

from qfd.hardware.errors import HardwareConfigurationError

LOCAL_SAMPLERS = ("exact", "simulated")
REMOTE_SAMPLERS = ("qpu", "hybrid")


def make_qubo_sampler(kind: str) -> dimod.Sampler:
    """Create a QUBO sampler.

    Args:
        kind: ``exact`` (brute force, tiny problems), ``simulated`` (simulated
            annealing), ``qpu`` (D-Wave annealer with automatic embedding) or
            ``hybrid`` (Leap hybrid solver).

    Returns:
        A dimod sampler.

    Raises:
        HardwareConfigurationError: If a remote sampler is requested without the
            ``dwave`` extra or an API token.
        ValueError: If ``kind`` is unknown.
    """
    if kind == "exact":
        return dimod.ExactSolver()
    if kind == "simulated":
        return SimulatedAnnealingSampler()
    if kind not in REMOTE_SAMPLERS:
        raise ValueError(
            f"Unknown sampler '{kind}'; use one of {LOCAL_SAMPLERS + REMOTE_SAMPLERS}"
        )
    if not os.environ.get("DWAVE_API_TOKEN"):
        raise HardwareConfigurationError(
            "Set DWAVE_API_TOKEN to use D-Wave Leap samplers"
        )
    try:
        from dwave.system import (  # noqa: PLC0415
            DWaveSampler,
            EmbeddingComposite,
            LeapHybridSampler,
        )
    except ImportError as exc:
        raise HardwareConfigurationError(
            'Install the D-Wave extra: pip install -e ".[dwave]"'
        ) from exc
    if kind == "qpu":
        return EmbeddingComposite(DWaveSampler())
    return LeapHybridSampler()

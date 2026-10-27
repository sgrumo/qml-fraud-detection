"""Stage 2: quantum models on local simulators (exact, shot-based or noisy)."""

from qfd.simulation.devices import SIMULATORS, make_simulator

__all__ = ["SIMULATORS", "make_simulator"]

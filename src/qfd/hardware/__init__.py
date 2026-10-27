"""Stage 3: real quantum hardware (AWS Braket, Google Quantum Engine, D-Wave).

Provider modules are imported lazily because each needs its optional extra.
"""

from qfd.hardware.budget import check_hardware_budget, estimate_kernel_circuits
from qfd.hardware.errors import HardwareBudgetError, HardwareConfigurationError

__all__ = [
    "HardwareBudgetError",
    "HardwareConfigurationError",
    "check_hardware_budget",
    "estimate_kernel_circuits",
]

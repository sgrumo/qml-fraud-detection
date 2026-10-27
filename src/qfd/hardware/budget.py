"""Guards that stop accidental, expensive hardware submissions."""

import logging

from qfd.hardware.errors import HardwareBudgetError

logger = logging.getLogger(__name__)


def estimate_kernel_circuits(n_train: int, n_test: int) -> int:
    """Upper bound on circuits for training and scoring a quantum kernel SVM.

    Args:
        n_train: Training samples (Gram matrix upper triangle).
        n_test: Test samples scored against every training sample.

    Returns:
        Number of circuit executions.
    """
    return n_train * (n_train - 1) // 2 + n_test * n_train


def check_hardware_budget(n_circuits: int, max_circuits: int, *, allowed: bool) -> None:
    """Refuse hardware runs that were not explicitly allowed or are too large.

    Args:
        n_circuits: Estimated circuit executions.
        max_circuits: Budget from the experiment configuration.
        allowed: Whether the user passed ``--allow-hardware``.

    Raises:
        HardwareBudgetError: If the run is not allowed or exceeds the budget.
    """
    if not allowed:
        raise HardwareBudgetError(
            f"This run submits about {n_circuits} circuits to paid hardware. "
            "Re-run with --allow-hardware to confirm."
        )
    if n_circuits > max_circuits:
        raise HardwareBudgetError(
            f"Estimated {n_circuits} circuits exceed "
            f"max_hardware_circuits={max_circuits}. "
            "Reduce n_train or test_negative_rate, or raise the budget."
        )
    logger.warning("Submitting about %d circuits to quantum hardware", n_circuits)

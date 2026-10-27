"""Errors raised when using real quantum hardware."""


class HardwareConfigurationError(RuntimeError):
    """A hardware provider is missing its extra, credentials or settings."""


class HardwareBudgetError(RuntimeError):
    """A hardware run was not allowed or exceeds the circuit budget."""

"""Typed accessors for heterogeneous model parameters read from YAML."""

from qfd.types import Params


def get_float(params: Params, key: str, default: float) -> float:
    """Read a float parameter.

    Args:
        params: Parameter mapping.
        key: Parameter name.
        default: Value used when the key is missing.

    Returns:
        The parameter as a float.

    Raises:
        TypeError: If the value is not numeric.
    """
    value = params.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"Parameter '{key}' must be a number, got {value!r}")
    return float(value)


def get_int(params: Params, key: str, default: int) -> int:
    """Read an integer parameter.

    Args:
        params: Parameter mapping.
        key: Parameter name.
        default: Value used when the key is missing.

    Returns:
        The parameter as an int.

    Raises:
        TypeError: If the value is not an integer.
    """
    value = params.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Parameter '{key}' must be an integer, got {value!r}")
    return value


def get_str(params: Params, key: str, default: str) -> str:
    """Read a string parameter.

    Args:
        params: Parameter mapping.
        key: Parameter name.
        default: Value used when the key is missing.

    Returns:
        The parameter as a string.

    Raises:
        TypeError: If the value is not a string.
    """
    value = params.get(key, default)
    if not isinstance(value, str):
        raise TypeError(f"Parameter '{key}' must be a string, got {value!r}")
    return value

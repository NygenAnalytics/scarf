import math
from numbers import Real

import numpy as np


def float_argument(value: object, name: str) -> float:
    """Return ``value`` as a finite Python float.

    Python and NumPy integers and floats are accepted, so ``1`` and ``1.0``
    give the same value, and booleans are rejected.

    Args:
        value: The argument to validate.
        name: Argument name used in error messages.

    Raises:
        TypeError: If ``value`` is not a real number.
        ValueError: If ``value`` is not finite.
    """
    if isinstance(value, bool | np.bool_) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite")
    return resolved


def integer_argument(
    value: object,
    name: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Return ``value`` as a Python integer within inclusive bounds.

    Python and NumPy integers are accepted and booleans are rejected.

    Args:
        value: The argument to validate.
        name: Argument name used in error messages.
        minimum: Smallest accepted value, if any.
        maximum: Largest accepted value, if any.

    Raises:
        TypeError: If ``value`` is not an integer.
        ValueError: If ``value`` lies outside the bounds.
    """
    if isinstance(value, bool | np.bool_) or not isinstance(value, int | np.integer):
        raise TypeError(f"{name} must be an integer")
    resolved = int(value)
    if minimum is not None and resolved < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and resolved > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return resolved

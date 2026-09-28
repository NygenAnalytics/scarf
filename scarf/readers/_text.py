from typing import Any

import numpy as np

from ..utils.arrays import has_duplicates


def as_text(value: Any) -> str:
    """Return ``value`` as text, decoding byte strings as UTF-8."""
    if isinstance(value, bytes | np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def require_unique_identifiers(values: Any, label: str) -> None:
    """Reject identifiers that are missing, blank, or repeated.

    Args:
        values: One-dimensional identifiers. Byte strings are decoded as UTF-8.
        label: Name of the identifiers used in error messages, such as
            ``Cell IDs``.

    Raises:
        ValueError: If a value is missing or blank, or a value repeats.
    """
    array = np.asarray(values, dtype=object)
    if array.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional")
    texts = np.asarray(
        ["" if value is None else as_text(value) for value in array],
        dtype=str,
    )
    if texts.size and bool(np.any(np.char.strip(texts) == "")):
        raise ValueError(f"{label} must contain non-empty values")
    if has_duplicates(texts):
        raise ValueError(f"{label} must contain unique values")

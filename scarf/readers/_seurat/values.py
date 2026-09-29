from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import DTypeLike, NDArray

from .errors import MatrixSourceError, ResourceLimitError


_TEXT_BLOCK = 4096


def vector_length(value: Any, object_path: str) -> int:
    try:
        return len(value)
    except TypeError:
        shape = getattr(value, "shape", None)
        if shape is None:
            raise TypeError(f"vector at {object_path} has no bounded length") from None
        normalized = tuple(int(item) for item in shape)
        if len(normalized) != 1:
            raise MatrixSourceError(
                f"vector at {object_path} must be one-dimensional"
            ) from None
        return normalized[0]


def _raw_window(value: Any, start: int, stop: int, object_path: str) -> Any:
    read_block = getattr(value, "read_block", None)
    try:
        return read_block(start, stop) if callable(read_block) else value[start:stop]
    except (IndexError, TypeError, ValueError) as error:
        raise MatrixSourceError(
            f"vector at {object_path} does not support bounded slicing"
        ) from error


def read_window(
    value: Any,
    start: int,
    stop: int,
    *,
    object_path: str = "array",
    dtype: DTypeLike | None = None,
) -> NDArray[Any]:
    result = np.asarray(_raw_window(value, start, stop, object_path))
    if result.ndim != 1:
        result = result.reshape(-1)
    if result.size != stop - start:
        raise MatrixSourceError(
            f"vector at {object_path} returned {result.size} values; "
            f"expected {stop - start}"
        )
    return result if dtype is None else result.astype(dtype, copy=False)


def read_bounded(
    value: Any,
    *,
    max_length: int,
    object_path: str,
    dtype: DTypeLike | None = None,
) -> NDArray[Any]:
    """Read a complete vector after checking its serialized length."""
    length = vector_length(value, object_path)
    if length > max_length:
        raise MatrixSourceError(
            f"vector at {object_path} has {length} values; at most {max_length} "
            "are allowed"
        )
    return read_window(value, 0, length, object_path=object_path, dtype=dtype)


def scalar_value(value: Any, object_path: str) -> Any:
    """Return the only element of a scalar or a length-one vector."""
    if isinstance(
        value,
        str
        | bytes
        | bool
        | int
        | float
        | complex
        | np.str_
        | np.bytes_
        | np.bool_
        | np.number,
    ):
        return value
    if vector_length(value, object_path) != 1:
        raise MatrixSourceError(f"value at {object_path} must be scalar")
    window = _raw_window(value, 0, 1, object_path)
    if isinstance(window, np.ndarray):
        if window.size != 1:
            raise MatrixSourceError(f"value at {object_path} must be scalar")
        return window.reshape(-1)[0].item()
    item = window[0]
    return item.item() if isinstance(item, np.generic) else item


def logical_scalar(value: Any, object_path: str) -> bool:
    """Return one R logical value; missing and non-logical values are rejected."""
    try:
        scalar = scalar_value(value, object_path)
    except MatrixSourceError as error:
        raise MatrixSourceError(
            f"{object_path} must contain one logical value"
        ) from error
    if isinstance(scalar, bool | np.bool_):
        return bool(scalar)
    if isinstance(scalar, int | np.integer) and int(scalar) in {0, 1}:
        return bool(scalar)
    raise MatrixSourceError(f"{object_path} must contain one logical value")


def shape_value(value: Any, label: str) -> tuple[int, int]:
    """Read a two-integer dimension vector; ``label`` names it in errors."""
    try:
        values = read_bounded(value, max_length=2, object_path=label)
    except MatrixSourceError as error:
        raise MatrixSourceError(f"{label} must contain two integers") from error
    if (
        values.size != 2
        or not np.issubdtype(values.dtype, np.number)
        or np.issubdtype(values.dtype, np.bool_)
        or np.any(~np.isfinite(values))
        or np.any(values != np.floor(values))
    ):
        raise MatrixSourceError(f"{label} must contain two integers")
    shape = (int(values[0]), int(values[1]))
    if min(shape) < 0:
        raise MatrixSourceError(f"{label} cannot contain negative values")
    return shape


def class_names(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, bytes):
        result = tuple(str(item) for item in value)
        if not result:
            raise MatrixSourceError("class vector cannot be empty")
        return result
    return (str(value),)


def decode_text(value: Any, object_path: str) -> str:
    """Decode one text value as UTF-8 and reject NUL characters."""
    if isinstance(value, bytes | np.bytes_):
        try:
            result = bytes(value).decode("utf-8")
        except UnicodeDecodeError as error:
            raise MatrixSourceError(
                f"text at {object_path} is not valid UTF-8"
            ) from error
    elif isinstance(value, str | np.str_):
        result = str(value)
        try:
            result.encode("utf-8")
        except UnicodeEncodeError as error:
            raise MatrixSourceError(
                f"text at {object_path} is not valid UTF-8"
            ) from error
    else:
        raise TypeError(f"text at {object_path} must contain strings")
    if "\x00" in result:
        raise MatrixSourceError(f"text at {object_path} contains NUL")
    return result


def decode_text_values(
    value: Any,
    *,
    object_path: str,
    max_bytes: int,
) -> tuple[str, ...]:
    """Decode a text vector blockwise within a metadata byte budget."""
    if isinstance(value, str | bytes):
        raise TypeError(f"text at {object_path} must be a sequence")
    length = vector_length(value, object_path)
    if length * 8 > max_bytes:
        raise ResourceLimitError(
            f"text at {object_path} exceeds maxMetadataBytes={max_bytes}"
        )
    output: list[str] = []
    used = 0
    for start in range(0, length, _TEXT_BLOCK):
        stop = min(length, start + _TEXT_BLOCK)
        block = _raw_window(value, start, stop, object_path)
        items = (
            np.asarray(block, dtype=object).reshape(-1)
            if isinstance(block, np.ndarray)
            else block
        )
        for offset, item in enumerate(items):
            text = decode_text(item, f"{object_path}[{start + offset}]")
            used += len(text.encode("utf-8")) + 8
            if used > max_bytes:
                raise ResourceLimitError(
                    f"text at {object_path} exceeds maxMetadataBytes={max_bytes}"
                )
            output.append(text)
    if len(output) != length:
        raise MatrixSourceError(
            f"text at {object_path} returned {len(output)} values; expected {length}"
        )
    return tuple(output)

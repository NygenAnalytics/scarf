"""The nullable value contract of metadata columns.

A metadata column stores one value of one dtype per row. A row whose value is
missing holds a placeholder of that dtype and is flagged in the column's
linked missing mask, ``__scarf_missing__<name>``, which readers honor:
``MetaData.to_pandas_dataframe``, ``head``, plots, and exports show the row as
missing, filters never pass it, and ``fetch`` returns the placeholder.

Supplied values are typed as imports type them. NumPy values keep their
dtype, and byte strings are decoded as UTF-8. Object values, as NumPy reads a
list that mixes types or holds ``None``, and pandas categorical, nullable, and
string values take the dtype that their present values share: bool, int64,
float64 for real numbers, complex128 for other numbers, and otherwise text.
Numbers that the shared dtype cannot hold, such as an integer outside the
int64 range, are refused rather than wrapped. ``None``, NaN, ``pd.NA``, and
``NaT`` among them are missing.
"""

import datetime
from typing import Any

import numpy as np
import pandas as pd

from ..storage.arrays import encode_metadata_values, text_value

__all__ = [
    "checked_fill_value",
    "missing_placeholder",
    "missing_rows",
    "nullable_values",
    "stored_values",
]


def missing_rows(values: np.ndarray) -> np.ndarray:
    """Return which object values are missing: None, NaN, ``pd.NA``, or NaT.

    Values of other dtypes are never missing.
    """
    array = np.asarray(values)
    if array.dtype.kind != "O":
        return np.zeros(array.shape, dtype=bool)
    return np.asarray(pd.isna(array), dtype=bool)


def missing_placeholder(dtype: np.dtype[Any]) -> Any:
    """Return the value that a masked row holds in a column of ``dtype``."""
    kind = dtype.kind
    if kind == "b":
        return False
    if kind in "iu":
        return 0
    if kind in "fc":
        return np.nan
    if kind in "Mm":
        return np.array("NaT", dtype=dtype)[()]
    if kind == "U":
        return ""
    if kind == "S":
        return b""
    raise TypeError(f"A metadata column of dtype {dtype} cannot hold missing values")


def _text(values: np.ndarray, missing: np.ndarray, *, name: str) -> np.ndarray:
    try:
        text = [
            "" if absent else text_value(value)
            for value, absent in zip(values.tolist(), missing.tolist(), strict=True)
        ]
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Column {name!r} holds text that is not valid UTF-8; pass str values "
            "or UTF-8 encoded bytes"
        ) from exc
    width = max((len(value) for value in text), default=1)
    return np.asarray(text, dtype=f"U{max(width, 1)}")


def _is_integer(value: Any) -> bool:
    # NumPy derives bool from int and timedelta64 from its signed integers.
    return isinstance(value, int | np.integer) and not isinstance(
        value, bool | np.bool_ | np.timedelta64
    )


def _is_real(value: Any) -> bool:
    return _is_integer(value) or isinstance(value, float | np.floating)


def _is_number(value: Any) -> bool:
    return _is_real(value) or isinstance(value, complex | np.complexfloating)


def _is_finite(value: Any) -> bool:
    # An integer is finite, and NumPy cannot test one beyond int64.
    return _is_integer(value) or bool(np.isfinite(value))


def _shared_dtype(present: np.ndarray) -> np.dtype[Any] | None:
    """Return the dtype that object values share, or None for text."""
    if not present.size:
        return None
    if all(isinstance(value, bool | np.bool_) for value in present):
        return np.dtype(bool)
    if all(_is_integer(value) for value in present):
        return np.dtype(np.int64)
    if all(_is_real(value) for value in present):
        return np.dtype(np.float64)
    if all(_is_number(value) for value in present):
        return np.dtype(np.complex128)
    return None


_INT64 = np.iinfo(np.int64)


def _typed_numbers(
    present: np.ndarray, dtype: np.dtype[Any], *, name: str
) -> np.ndarray:
    """Return the present object values as ``dtype``, which must hold each.

    Raises:
        ValueError: If an integer lies outside the int64 range, or a number
            is too large for float64 or complex128.
    """
    values = present.tolist()
    if dtype == np.int64:
        # Compare Python integers, so no value wraps on its way to int64.
        values = [int(value) for value in values]
        outside = [value for value in values if not _INT64.min <= value <= _INT64.max]
        if outside:
            raise ValueError(
                f"Column {name!r} holds integers outside the int64 range, such as "
                f"{outside[0]}; pass them as a NumPy array of a dtype that holds "
                "them, such as uint64, or as text"
            )
    message = f"Column {name!r} holds numbers too large for {dtype}"
    try:
        with np.errstate(over="ignore"):
            typed = np.asarray(values, dtype=dtype)
    except OverflowError as error:
        raise ValueError(message) from error
    # A finite number of higher precision can become infinite.
    if dtype.kind in "fc" and any(
        _is_finite(value) and not np.isfinite(cast)
        for value, cast in zip(values, typed, strict=True)
    ):
        raise ValueError(message)
    return typed


def stored_values(
    values: np.ndarray,
    missing: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    """Return a typed column whose masked rows hold a placeholder."""
    kind = values.dtype.kind
    if kind == "O":
        present = values[~missing]
        dtype = _shared_dtype(present)
        if dtype is None:
            return _text(values, missing, name=name)
        stored = np.full(values.shape, missing_placeholder(dtype), dtype=dtype)
        stored[~missing] = _typed_numbers(present, dtype, name=name)
        return stored
    if kind in "SU":
        return _text(values, missing, name=name)
    stored = values.copy()
    if missing.any():
        stored[missing] = missing_placeholder(values.dtype)
    return stored


def _is_pandas_values(values: Any) -> bool:
    return isinstance(
        values, pd.Series | pd.Index | pd.api.extensions.ExtensionArray
    ) and not (isinstance(values.dtype, np.dtype) and values.dtype.kind != "O")


def nullable_values(values: Any, *, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return supplied column values as a typed array and their missing rows."""
    array = np.asarray(values, dtype=object) if _is_pandas_values(values) else values
    array = np.asarray(array)
    if array.ndim != 1:
        raise ValueError(
            f"Column {name!r} needs one value per row; found values of shape "
            f"{array.shape}"
        )
    if array.dtype.kind == "T":
        array = array.astype(object)
    if array.dtype.kind == "O":
        missing = missing_rows(array)
        return stored_values(array, missing, name=name), missing
    if array.dtype.kind == "S":
        array = encode_metadata_values(array, None, name=name)
    return array, np.zeros(array.shape, dtype=bool)


_FILL_KINDS = {
    "U": "text",
    "b": "a bool",
    "i": "an integer that is not a bool, within the range of the dtype",
    "u": "an integer that is not a bool, within the range of the dtype",
    "f": "a real number that is not a bool, within the range of the dtype",
    "c": "a number that is not a bool, within the range of the dtype",
    "M": "a datetime without a time zone that the unit of the dtype holds exactly",
    "m": "a timedelta that the unit of the dtype holds exactly",
}


def _exact_time_fill(fill_value: Any, dtype: np.dtype[Any]) -> Any | None:
    """Return a datetime or timedelta fill as a value of ``dtype``, or None.

    The fill is first taken as its own datetime64 or timedelta64, at its own
    resolution, and is accepted only when the cast to ``dtype`` gives that
    value back, so a fill that the unit would truncate, or that lies outside
    the unit's range, is rejected rather than stored as another value. NaT
    is a value of either kind. A datetime with a time zone has no exact
    value in a column without one.
    """
    if fill_value is pd.NaT:
        return missing_placeholder(dtype)
    if dtype.kind == "M":
        if not isinstance(
            fill_value, np.datetime64 | datetime.datetime | datetime.date
        ):
            return None
        if getattr(fill_value, "tzinfo", None) is not None:
            return None
        own = (
            fill_value.to_datetime64()
            if isinstance(fill_value, pd.Timestamp)
            else np.datetime64(fill_value)
        )
    else:
        if not isinstance(fill_value, np.timedelta64 | datetime.timedelta):
            return None
        own = (
            fill_value.to_timedelta64()
            if isinstance(fill_value, pd.Timedelta)
            else np.timedelta64(fill_value)
        )
    if np.isnat(own):
        return missing_placeholder(dtype)
    try:
        cast = own.astype(dtype)
        exact = bool(cast.astype(own.dtype) == own)
    except OverflowError:
        return None
    return cast if exact else None


def _fitted_fill(fill_value: Any, dtype: np.dtype[Any]) -> Any | None:
    """Return ``fill_value`` as a value of ``dtype``, or None if it does not fit."""
    kind = dtype.kind
    if kind == "U":
        return str(fill_value) if isinstance(fill_value, str) else None
    if isinstance(fill_value, bool | np.bool_):
        return bool(fill_value) if kind == "b" else None
    if kind in "iu" and _is_integer(fill_value):
        limits = np.iinfo(dtype)
        return int(fill_value) if limits.min <= int(fill_value) <= limits.max else None
    if (kind == "f" and _is_real(fill_value)) or (
        kind == "c" and _is_number(fill_value)
    ):
        try:
            with np.errstate(over="ignore"):
                cast = dtype.type(fill_value)
            # A finite value must not overflow; NaN and infinity are values.
            overflowed = _is_finite(fill_value) and not bool(np.isfinite(cast))
        except OverflowError:
            return None
        return None if overflowed else cast
    if kind in "Mm":
        return _exact_time_fill(fill_value, dtype)
    return None


def checked_fill_value(
    fill_value: Any, dtype: np.dtype[Any]
) -> tuple[Any, np.dtype[Any]]:
    """Return an explicit fill value and the dtype of the column that holds it."""
    fitted = _fitted_fill(fill_value, dtype)
    if fitted is None:
        expected = _FILL_KINDS.get(dtype.kind, "no explicit value")
        raise ValueError(
            f"fill_value {fill_value!r} does not fit values of dtype {dtype}: pass "
            f"{expected}, or None to flag the rows as missing"
        )
    if dtype.kind == "U":
        width = max(dtype.itemsize // np.dtype("U1").itemsize, len(fitted), 1)
        return fitted, np.dtype(f"U{width}")
    return fitted, dtype

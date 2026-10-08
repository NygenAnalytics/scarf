"""Write-time checks of artifact arrays that must hold only finite values.

A checked writer casts each block to the dtype of its array and checks it
before writing it, so NaN or infinity, including a value that overflows the
dtype, raises :class:`NonFiniteArtifactError` before the block is written.
Producers write the arrays that ``FINITE_ARTIFACT_ARRAYS`` names with a
checked writer before they publish the artifact, so a non-finite value
publishes nothing.
"""

from collections.abc import Mapping
from types import MappingProxyType, TracebackType
from typing import Any

import numpy as np
import zarr

# Arrays of each artifact kind that must hold only finite values; None names
# every floating-point array of the artifact. Producers write such an array
# with a checked writer. Kinds that are not listed carry no such contract:
# scores where NaN marks an undefined value, labels, ANN index bytes, and
# selection masks. Integer arrays are finite by their type and need no check.
FINITE_ARTIFACT_ARRAYS: Mapping[str, frozenset[str] | None] = MappingProxyType(
    {
        # Normalized values are graph coordinates of their own, without a
        # reduction, so they are held to the contract of coordinates.
        "normalized": frozenset({"data"}),
        "reduction": frozenset({"data", "loadings", "center"}),
        "batch_correction": None,
        "embedding_initialization": frozenset({"cluster_centers"}),
        "embedding": frozenset({"values"}),
        "neighbors": frozenset({"distances"}),
        "connectivity_map": frozenset({"weights"}),
        "integrated_graph": frozenset({"weights", "modality_weights"}),
    }
)

# A check scans at most this many values at once, so its temporaries stay
# bounded whatever the size of the block.
_SCAN_VALUES = 1 << 22


class NonFiniteArtifactError(ValueError):
    """A checked writer refused NaN or infinity in an artifact array."""

    def __init__(self, operation: str, array: str, row: int) -> None:
        self.operation = operation
        self.array = array
        self.row = int(row)
        super().__init__(
            f"{operation} result holds a non-finite value (NaN or infinity) in "
            f"array {array!r} at row {self.row}, where only finite values are allowed"
        )

    def __reduce__(self) -> tuple[Any, tuple[str, str, int]]:
        return (type(self), (self.operation, self.array, self.row))


def first_nonfinite_row(values: Any) -> int | None:
    """Return the first row of ``values`` that holds NaN or infinity, or None."""
    array = np.asarray(values)
    kind = array.dtype.kind
    if kind in "biu":
        return None
    if kind not in "fc":
        raise TypeError(f"Finite checks need numeric values, not {array.dtype}")
    if array.ndim == 0:
        array = array.reshape(1)
    row_values = max(1, int(np.prod(array.shape[1:], dtype=np.int64)))
    band = max(1, _SCAN_VALUES // row_values)
    for start in range(0, int(array.shape[0]), band):
        finite = np.isfinite(array[start : start + band])
        if finite.all():
            continue
        rows = finite.reshape(finite.shape[0], -1).all(axis=1)
        return start + int(np.argmin(rows))
    return None


def requires_finite_values(kind: str, name: str, dtype: Any) -> bool:
    """Return whether array ``name`` of a ``kind`` artifact needs a checked writer."""
    if np.dtype(dtype).kind not in "fc":
        return False
    names = FINITE_ARTIFACT_ARRAYS.get(kind, frozenset())
    return names is None or name in names


class FiniteRowWriter:
    """Write a floating-point array row by row, refusing non-finite values.

    Call ``close`` after the last row, or use the writer as a context manager.
    """

    def __init__(self, array: zarr.Array, *, operation: str) -> None:
        if not isinstance(operation, str) or not operation:
            raise ValueError("operation must name the producer of the values")
        if np.dtype(array.dtype).kind not in "fc":
            raise TypeError(
                f"Array {array.basename!r} holds {array.dtype} values; checked "
                "writers write floating-point arrays"
            )
        if array.ndim < 1:
            raise ValueError(f"Array {array.basename!r} has no rows")
        self._array = array
        self._operation = operation
        self._rows = int(array.shape[0])
        self._next = 0
        self._closed = False

    def write(self, values: Any) -> None:
        """Check ``values`` and write them as the next rows of the array."""
        if self._closed:
            raise RuntimeError(f"The writer of {self._array.basename!r} is closed")
        array = self._array
        with np.errstate(over="ignore", invalid="ignore"):
            block = np.asarray(values).astype(array.dtype, copy=False)
        if block.ndim != array.ndim or block.shape[1:] != tuple(array.shape[1:]):
            raise ValueError(
                f"A block of shape {block.shape} does not fit array "
                f"{array.basename!r} of shape {tuple(array.shape)}"
            )
        start = self._next
        stop = start + int(block.shape[0])
        if stop > self._rows:
            raise ValueError(
                f"Array {array.basename!r} has {self._rows} rows, so rows "
                f"{start} to {stop} do not fit"
            )
        row = first_nonfinite_row(block)
        if row is not None:
            raise NonFiniteArtifactError(self._operation, array.basename, start + row)
        if stop > start:
            array[start:stop] = block
        self._next = stop

    def close(self) -> None:
        """Check that every row of the array is written."""
        if self._closed:
            return
        if self._next != self._rows:
            raise ValueError(
                f"Array {self._array.basename!r} has {self._next} of its "
                f"{self._rows} rows written"
            )
        self._closed = True

    def __enter__(self) -> "FiniteRowWriter":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self.close()


def write_finite_array(array: zarr.Array, values: Any, *, operation: str) -> None:
    """Write every row of ``array`` from ``values`` with one checked writer."""
    with FiniteRowWriter(array, operation=operation) as writer:
        writer.write(values)

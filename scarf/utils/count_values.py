"""Value ranges of count matrices, which decide their storage dtype.

Readers and writers summarize the canonical (duplicate-summed) values of a
count matrix into a :class:`CountValueRange`, one for each group of features
that an import stores as one assay;
:func:`scarf.storage.count_dtype.count_storage_dtype` turns a range into the
storage dtype of its counts. Count matrices hold finite values, so the scans
read every value and reject NaN and infinity.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .arrays import canonicalize_sparse

# A scan window holds its values and indices, a 64-bit copy of narrow integers,
# the sort and sum buffers of its canonical values, the values of one group,
# and the integrality checks, within 64 bytes per value. A block of up to one
# pointer per window value adds 8 more.
_SCAN_BYTES_PER_VALUE = 72
_SCAN_WINDOW_VALUES = 1 << 20


@dataclass(slots=True)
class CountValueRange:
    """Running summary of canonical count values.

    Attributes:
        integral: Whether every value seen is a non-negative integer.
        maximum: The largest value seen while ``integral`` holds.
    """

    integral: bool = True
    maximum: int = 0

    def update(self, values: Any) -> None:
        """Add a block of canonical values, in any shape.

        A fractional or negative value settles the range; later blocks are
        only checked for NaN and infinity.

        Raises:
            TypeError: If the values are not real numbers.
            ValueError: If a value is NaN or infinite.
        """
        block = np.asarray(values)
        if block.size == 0:
            return
        kind = block.dtype.kind
        if kind not in "biuf":
            raise TypeError(f"Count values must be real numbers, not {block.dtype}")
        if kind == "f" and not np.isfinite(block).all():
            raise ValueError(
                "Count matrices must hold finite values; found NaN or infinity"
            )
        if not self.integral:
            return
        if kind == "f" and not np.equal(block, np.trunc(block)).all():
            self.integral = False
            return
        if kind in "if" and block.min() < 0:
            self.integral = False
            return
        self.maximum = max(self.maximum, int(block.max()))


def _summable(values: np.ndarray) -> np.ndarray:
    """Return values in the dtype in which their duplicates are summed.

    Narrow integers widen to 64 bits so that sums cannot wrap. float16, which
    SciPy sparse matrices cannot hold, is read as float32, as readers do.
    """
    kind = values.dtype.kind
    if kind in "bu":
        return values.astype(np.uint64, copy=False)
    if kind == "i":
        return values.astype(np.int64, copy=False)
    if kind == "f":
        dtype = values.dtype.newbyteorder("=")
        return values.astype(np.float32 if dtype == np.float16 else dtype, copy=False)
    raise TypeError(f"Count values must be real numbers, not {values.dtype}")


def _canonical(block: Any) -> Any:
    """Return a CSR window with its duplicate coordinates summed as writers sum them."""
    if block.dtype.kind == "f":
        # Float sums depend on the order of their terms, so they are summed
        # by the writers' function, which leaves the caller's arrays unchanged.
        return canonicalize_sparse(block.tocoo(copy=False)).tocsr()
    # Integer sums are exact in 64 bits. SciPy sorts and sums in place, and
    # the window may be a view of the caller's arrays, so it works on a copy.
    block = block.copy()
    block.sum_duplicates()
    return block


def new_count_ranges(groups: np.ndarray | None) -> list[CountValueRange]:
    """Return one range that has seen no value for each group in ``groups``.

    Groups are numbered from zero; None stands for one group.
    """
    count = 1 if groups is None else int(np.max(groups, initial=0)) + 1
    return [CountValueRange() for _ in range(count)]


def compressed_count_ranges(
    indptr: Any,
    indices: Any,
    data: Any,
    *,
    minorSize: int,
    maxBytes: int,
    groups: np.ndarray | None = None,
    groupAxis: int = 1,
    valueRanges: list[CountValueRange] | None = None,
) -> list[CountValueRange]:
    """Return the ranges of the canonical values of a compressed sparse matrix.

    The arrays are those of a CSR or CSC matrix, or objects that slice like
    them, such as HDF5 datasets; they are not modified. The scan reads every
    value, in windows of whole compressed vectors, and sums duplicate
    coordinates as writers store them: integers exactly in 64 bits and floats
    with :func:`~scarf.utils.arrays.canonicalize_sparse`, in source order. The
    ranges therefore depend neither on the index order nor on ``maxBytes``.

    Args:
        indptr: Compressed vector pointers into ``indices`` and ``data``. A
            slice of a matrix's pointers scans only those vectors.
        indices: Minor-axis index of each stored value.
        data: Stored values.
        minorSize: Length of the minor axis.
        maxBytes: Memory available to the scan. A window holds at most about
            ``maxBytes // 72`` values.
        groups: Group, numbered from zero, of each minor index when
            ``groupAxis`` is 1, or of each compressed vector of ``indptr``
            when it is 0. None puts every value in one group.
        groupAxis: The axis that ``groups`` indexes.
        valueRanges: The range of each group that the scan extends, so that a
            matrix can be scanned in parts. None starts new ranges.

    Returns:
        The range of each group.

    Raises:
        MemoryError: If one compressed vector holds more values than fit in
            ``maxBytes``.
        ValueError: If a value is NaN or infinite.
    """
    from scipy.sparse import csr_matrix

    value_ranges = new_count_ranges(groups) if valueRanges is None else valueRanges
    limit = max(0, int(maxBytes)) // _SCAN_BYTES_PER_VALUE
    window = min(_SCAN_WINDOW_VALUES, limit)
    n_vectors = int(indptr.shape[0]) - 1
    start = 0
    while start < n_vectors:
        # Each pointer is read once; windows of whole vectors walk the block.
        pointers = np.asarray(
            indptr[start : min(n_vectors, start + max(1, window)) + 1],
            dtype=np.int64,
        )
        position = 0
        while position < len(pointers) - 1:
            target = pointers[position] + window
            end = int(np.searchsorted(pointers, target, side="right")) - 1
            if end <= position:
                # A vector longer than a window is read whole when it fits.
                length = int(pointers[position + 1] - pointers[position])
                if length > limit:
                    raise MemoryError(
                        f"One compressed vector holds {length} values; scanning "
                        f"it needs about {length * _SCAN_BYTES_PER_VALUE} bytes, "
                        f"more than the {max(0, int(maxBytes))}-byte limit. "
                        "Increase mem_budget."
                    )
                end = position + 1
            first, last = int(pointers[position]), int(pointers[end])
            block = csr_matrix(
                (
                    _summable(np.asarray(data[first:last])),
                    np.asarray(indices[first:last]),
                    pointers[position : end + 1] - first,
                ),
                shape=(end - position, int(minorSize)),
            )
            if not block.has_canonical_format:
                block = _canonical(block)
            if groups is None:
                value_ranges[0].update(block.data)
            else:
                codes = (
                    groups[block.indices]
                    if groupAxis == 1
                    else np.repeat(
                        groups[start + position : start + end], np.diff(block.indptr)
                    )
                )
                for code, value_range in enumerate(value_ranges):
                    value_range.update(block.data[codes == code])
            position = end
        start += position
    return value_ranges


def dense_count_ranges(
    read: Callable[[int, int], np.ndarray],
    nRows: int,
    nColumns: int,
    *,
    maxBytes: int,
    groups: np.ndarray | None = None,
) -> list[CountValueRange]:
    """Return the ranges of the values of a dense matrix read in row blocks.

    The scan reads every row.

    Args:
        read: Returns rows ``[start, stop)`` as a NumPy array.
        nRows: Number of rows.
        nColumns: Number of columns.
        maxBytes: Memory available to the scan.
        groups: Group, numbered from zero, of each column. None puts every
            value in one group.

    Returns:
        The range of each group.

    Raises:
        MemoryError: If one row does not fit in ``maxBytes``.
        ValueError: If a value is NaN or infinite.
    """
    row_values = max(1, int(nColumns))
    rows = min(
        max(1, _SCAN_WINDOW_VALUES // row_values),
        max(0, int(maxBytes)) // (_SCAN_BYTES_PER_VALUE * row_values),
    )
    value_ranges = new_count_ranges(groups)
    if int(nRows) and rows < 1:
        raise MemoryError(
            f"One dense row holds {row_values} values; scanning it needs about "
            f"{row_values * _SCAN_BYTES_PER_VALUE} bytes, more than the "
            f"{max(0, int(maxBytes))}-byte limit. Increase mem_budget."
        )
    columns = (
        None
        if groups is None
        else [np.flatnonzero(groups == code) for code in range(len(value_ranges))]
    )
    for start in range(0, int(nRows), max(1, rows)):
        block = read(start, min(int(nRows), start + rows))
        if columns is None:
            value_ranges[0].update(block)
        else:
            for value_range, selected in zip(value_ranges, columns, strict=True):
                value_range.update(block[:, selected])
    return value_ranges

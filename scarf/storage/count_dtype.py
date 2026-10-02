"""The storage dtype of count matrices, one policy for every count writer.

A count matrix is stored unsigned, in the narrowest of uint8, uint16, uint32,
and uint64 that holds its largest value, exactly when every canonical
(duplicate-summed) value is a non-negative integer. Any other matrix keeps its
source dtype, with float16 read as float32. Readers and writers summarize the
values with :class:`scarf.utils.count_values.CountValueRange`.
"""

from typing import Any

import numpy as np

from ..utils.count_values import CountValueRange

_UNSIGNED = tuple(np.dtype(name) for name in ("uint8", "uint16", "uint32", "uint64"))


def count_storage_dtype(sourceDtype: Any, valueRange: CountValueRange) -> np.dtype[Any]:
    """Return the dtype that stores counts with ``valueRange``.

    Args:
        sourceDtype: Dtype in which the source holds the values.
        valueRange: Range of every canonical value of the matrix.

    Returns:
        The narrowest unsigned dtype that holds every value when all are
        non-negative integers; otherwise the source dtype in native byte
        order, with float16 read as float32.
    """
    if valueRange.integral:
        for dtype in _UNSIGNED:
            if valueRange.maximum <= np.iinfo(dtype).max:
                return dtype
    source = np.dtype(sourceDtype).newbyteorder("=")
    return np.dtype(np.float32) if source == np.float16 else source

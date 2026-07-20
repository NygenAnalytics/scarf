import inspect
import pickle
from typing import get_type_hints

import numpy as np

import scarf.matrix as matrix
from scarf.matrix.blocks import Block as implementation_block
from scarf.matrix.chunked import ChunkedArray as implementation_chunked_array


def test_matrix_facade_exports_canonical_classes():
    assert matrix.__all__ == ["Block", "ChunkedArray"]
    assert matrix.Block is implementation_block
    assert matrix.ChunkedArray is implementation_chunked_array
    assert matrix.Block.__module__ == "scarf.matrix"
    assert matrix.ChunkedArray.__module__ == "scarf.matrix"


def test_chunked_array_signatures_remain_stable():
    assert list(inspect.signature(matrix.ChunkedArray).parameters) == [
        "backing",
        "rows",
        "cols",
        "ops",
        "out_cols",
        "block_size",
        "nthreads",
        "is_numpy",
    ]
    assert list(inspect.signature(matrix.ChunkedArray.from_numpy).parameters) == [
        "arr",
        "block_size",
        "nthreads",
    ]
    assert list(inspect.signature(matrix.ChunkedArray.stream_blocks).parameters) == [
        "self",
        "nthreads",
        "msg",
        "prefetch",
    ]
    assert list(inspect.signature(matrix.ChunkedArray.dot).parameters) == [
        "self",
        "b",
    ]


def test_matrix_type_hints_and_pickle_paths_resolve():
    assert get_type_hints(matrix.Block.__init__)["parent"] is matrix.ChunkedArray
    original = matrix.ChunkedArray.from_numpy(
        np.arange(6, dtype=np.float64).reshape(3, 2),
        block_size=2,
    )

    restored = pickle.loads(pickle.dumps(original))

    assert type(restored) is matrix.ChunkedArray
    np.testing.assert_array_equal(restored.compute(), original.compute())

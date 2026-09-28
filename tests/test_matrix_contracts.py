import pickle

import numpy as np

import scarf.matrix as matrix
from scarf.matrix.chunked import ChunkedArray as implementation_chunked_array
from tests.signature_contracts import signature_digest


def test_matrix_facade_exports_canonical_classes():
    assert matrix.__all__ == ["ChunkedArray"]
    assert matrix.ChunkedArray is implementation_chunked_array
    assert matrix.ChunkedArray.__module__ == "scarf.matrix"
    assert not hasattr(matrix, "Block")
    for removed in ("blocks", "chunks", "dot", "map_blocks", "nthreads", "std"):
        assert not hasattr(matrix.ChunkedArray, removed)


def test_chunked_array_signatures_remain_stable():
    methods = {
        "ChunkedArray.__init__": matrix.ChunkedArray.__init__,
        "ChunkedArray.from_numpy": matrix.ChunkedArray.from_numpy,
        "ChunkedArray.stream_blocks": matrix.ChunkedArray.stream_blocks,
    }

    assert signature_digest(methods) == (
        "ecfef6d543fe26b9c5e406576cf1d19168d3ba214a4d925301cf7f64c6612e05"
    )


def test_matrix_pickle_paths_resolve():
    original = matrix.ChunkedArray.from_numpy(
        np.arange(6, dtype=np.float64).reshape(3, 2),
        block_size=2,
    )

    restored = pickle.loads(pickle.dumps(original))

    assert type(restored) is matrix.ChunkedArray
    np.testing.assert_array_equal(restored.compute(), original.compute())

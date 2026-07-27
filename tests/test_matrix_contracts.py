import pickle
from typing import get_type_hints

import numpy as np

import scarf.matrix as matrix
from scarf.matrix.blocks import Block as implementation_block
from scarf.matrix.chunked import ChunkedArray as implementation_chunked_array
from tests.signature_contracts import signature_digest


def test_matrix_facade_exports_canonical_classes():
    assert matrix.__all__ == ["Block", "ChunkedArray"]
    assert matrix.Block is implementation_block
    assert matrix.ChunkedArray is implementation_chunked_array
    assert matrix.Block.__module__ == "scarf.matrix"
    assert matrix.ChunkedArray.__module__ == "scarf.matrix"


def test_chunked_array_signatures_remain_stable():
    methods = {
        "ChunkedArray.__init__": matrix.ChunkedArray.__init__,
        "ChunkedArray.from_numpy": matrix.ChunkedArray.from_numpy,
        "ChunkedArray.stream_blocks": matrix.ChunkedArray.stream_blocks,
        "ChunkedArray.dot": matrix.ChunkedArray.dot,
    }

    assert signature_digest(methods) == (
        "0ac8b1f50482e55a8697f2fba39d0fa51c3b588c81923a0c70f4ef5a8ae2fee7"
    )


def test_matrix_type_hints_and_pickle_paths_resolve():
    assert get_type_hints(matrix.Block.__init__)["parent"] is matrix.ChunkedArray
    original = matrix.ChunkedArray.from_numpy(
        np.arange(6, dtype=np.float64).reshape(3, 2),
        block_size=2,
    )

    restored = pickle.loads(pickle.dumps(original))

    assert type(restored) is matrix.ChunkedArray
    np.testing.assert_array_equal(restored.compute(), original.compute())

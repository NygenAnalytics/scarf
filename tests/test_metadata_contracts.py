import pickle

import numpy as np
import zarr
from zarr.storage import MemoryStore

import scarf.metadata as metadata
from scarf.metadata.rows import MetaDataRowBlock as implementation_row_block
from scarf.metadata.rows import (
    array_row_selection_peak_bytes,
    iter_metadata_column_blocks,
    metadata_missing_mask,
    read_metadata_missing_rows,
    read_metadata_rows,
    read_metadata_rows_chunkwise,
)
from scarf.metadata.table import MetaData as implementation_metadata
from tests.signature_contracts import signature_digest


_METHODS = {
    "__init__",
    "__repr__",
    "_col_renamer",
    "_column_map",
    "_fill_to_index",
    "_get_array",
    "_get_loc",
    "_get_size",
    "_save",
    "_verify_bool",
    "active_index",
    "default_block_rows",
    "drop",
    "fetch",
    "fetch_all",
    "get_dtype",
    "get_index_by",
    "grep",
    "head",
    "index_to_bool",
    "insert",
    "iter_row_blocks",
    "mount_location",
    "multi_sift",
    "remove_trend",
    "reset_key",
    "sift",
    "to_pandas_dataframe",
    "unmount_location",
    "update_key",
}


def _metadata_fixture() -> metadata.MetaData:
    group = zarr.open_group(store=MemoryStore(), mode="w")
    group.create_array(
        "I",
        data=np.array([True, False, True, True]),
        chunks=(2,),
    )
    group.create_array(
        "ids",
        data=np.array(["a", "b", "c", "d"]),
        chunks=(2,),
    )
    group.create_array(
        "names",
        data=np.array(["Alpha", "Beta", "Alpine", "Delta"]),
        chunks=(2,),
    )
    group.create_array(
        "score",
        data=np.array([0.5, 2.0, 3.5, 5.0]),
        chunks=(2,),
    )
    return metadata.MetaData(group)


def test_metadata_facade_exports_canonical_objects():
    assert metadata.__all__ == ["MetaData", "MetaDataRowBlock"]
    assert metadata.MetaData is implementation_metadata
    assert metadata.MetaDataRowBlock is implementation_row_block
    assert metadata.MetaData.__module__ == "scarf.metadata"
    assert metadata.MetaDataRowBlock.__module__ == "scarf.metadata"


def test_metadata_method_ownership_and_signatures_remain_stable():
    assert _METHODS <= set(metadata.MetaData.__dict__)
    methods = {name: getattr(metadata.MetaData, name) for name in _METHODS}

    assert signature_digest(methods) == (
        "f882e8fb00453f282ab117edfa27a4433ee2e46446b71631f8c4c450f0327d5d"
    )


def test_metadata_rows_and_queries_match_table_contract():
    table = _metadata_fixture()

    assert table.default_block_rows() == 2
    blocks = list(table.iter_row_blocks(columns=["score"], block_rows=2))
    np.testing.assert_array_equal(
        np.concatenate([block.active_global_indices for block in blocks]),
        [0, 2, 3],
    )
    np.testing.assert_allclose(
        np.concatenate([block.values["score"] for block in blocks]),
        [0.5, 3.5, 5.0],
    )
    np.testing.assert_array_equal(
        table.sift("score", 1.0, 4.0), [False, True, True, False]
    )
    assert table.grep("^AL") == ["ALPHA", "ALPINE"]
    assert table.head(2)["score"].tolist() == [0.5, 2.0]


def test_metadata_row_helpers_read_permutations_and_missing_masks():
    table = _metadata_fixture()
    group = table.locations["primary"]
    group.create_array(
        "__scarf_missing__score",
        data=np.array([False, True, False, True]),
        chunks=(2,),
    )
    group["score"].attrs["missing_mask"] = "__scarf_missing__score"

    np.testing.assert_allclose(
        read_metadata_rows(table, "score", np.array([3, 2])),
        [5.0, 3.5],
    )
    assert metadata_missing_mask(table, "score") is not None
    np.testing.assert_array_equal(
        read_metadata_missing_rows(table, "score", np.array([3, 2])),
        [True, False],
    )
    assert "__scarf_missing__score" not in table.columns


def test_metadata_row_helpers_preserve_noncontiguous_order_without_span():
    class TrackingArray:
        def __init__(self, values):
            self._values = np.asarray(values)
            self.shape = self._values.shape
            self.requests: list[tuple[str, object]] = []

        def __getitem__(self, key):
            self.requests.append(("getitem", key))
            return self._values[key]

        def get_orthogonal_selection(self, selection):
            self.requests.append(("orthogonal", selection))
            (indices,) = selection
            return self._values[np.asarray(indices)]

    class TrackingMeta:
        N = 6
        columns = ["score"]

        def __init__(self, array):
            self._array = array

        def _get_array(self, column):
            assert column == "score"
            return self._array

        def _verify_bool(self, key):
            return False

        def default_block_rows(self, column="I"):
            return self.N

    array = TrackingArray([10, 20, 30, 40, 50, 60])
    table = TrackingMeta(array)
    rows = np.array([5, 1, 4], dtype=np.int64)
    np.testing.assert_array_equal(
        read_metadata_rows(table, "score", rows), [60, 20, 50]
    )
    assert array.requests[0][0] == "orthogonal"
    assert not any(kind == "getitem" for kind, _ in array.requests)

    contiguous = TrackingArray([10, 20, 30, 40])
    contiguous_table = TrackingMeta(contiguous)
    np.testing.assert_array_equal(
        read_metadata_rows(contiguous_table, "score", np.array([1, 2, 3])),
        [20, 30, 40],
    )
    assert contiguous.requests == [("getitem", slice(1, 4))]


def test_chunkwise_metadata_rows_preserve_order_and_decode_one_chunk():
    class ArrayMetadata:
        shards = None

    class TrackingArray:
        def __init__(self, values, chunk_rows):
            self._values = np.asarray(values)
            self.shape = self._values.shape
            self.dtype = self._values.dtype
            self.chunks = (chunk_rows,)
            self.metadata = ArrayMetadata()
            self.requests: list[tuple[str, object]] = []

        def __getitem__(self, key):
            self.requests.append(("getitem", key))
            return self._values[key]

        def get_orthogonal_selection(self, selection):
            self.requests.append(("orthogonal", selection))
            (indices,) = selection
            return self._values[np.asarray(indices)]

    class TrackingMeta:
        columns = ["score"]

        def __init__(self, array):
            self._array = array
            self.N = int(array.shape[0])

        def _get_array(self, column):
            assert column == "score"
            return self._array

        def default_block_rows(self, column="I"):
            _ = column
            return int(self._array.chunks[0])

    array = TrackingArray([10, 20, 30, 40, 50, 60], chunk_rows=2)
    table = TrackingMeta(array)
    rows = np.array([5, 0, 4, 1, 2], dtype=np.int64)
    np.testing.assert_array_equal(
        read_metadata_rows_chunkwise(table, "score", rows),
        [60, 10, 50, 20, 30],
    )

    for kind, request in array.requests:
        if kind == "getitem":
            assert isinstance(request, slice)
            selected = np.arange(request.start, request.stop)
        else:
            assert isinstance(request, tuple)
            selected = np.asarray(request[0])
        assert np.unique(selected // array.chunks[0]).size == 1

    one_row = array_row_selection_peak_bytes(array, 1)
    five_rows = array_row_selection_peak_bytes(array, 5)
    assert one_row > array.chunks[0] * array.dtype.itemsize
    assert five_rows > one_row


def test_metadata_column_blocks_respect_source_chunk_boundaries():
    class ArrayMetadata:
        shards = None

    class TrackingArray:
        def __init__(self):
            self._values = np.arange(7)
            self.shape = self._values.shape
            self.dtype = self._values.dtype
            self.chunks = (3,)
            self.metadata = ArrayMetadata()
            self.requests: list[slice] = []

        def __getitem__(self, key):
            assert isinstance(key, slice)
            self.requests.append(key)
            return self._values[key]

    class TrackingMeta:
        N = 7
        columns = ["score"]

        def __init__(self, array):
            self._array = array

        def _get_array(self, column):
            assert column == "score"
            return self._array

        def default_block_rows(self, column="I"):
            _ = column
            return int(self._array.chunks[0])

    array = TrackingArray()
    table = TrackingMeta(array)
    values = list(iter_metadata_column_blocks(table, "score", block_rows=2))
    np.testing.assert_array_equal(np.concatenate(values), np.arange(7))
    assert max(value.size for value in values) <= 2
    for request in array.requests:
        first_bin = request.start // array.chunks[0]
        last_bin = (request.stop - 1) // array.chunks[0]
        assert first_bin == last_bin


def test_metadata_remove_trend_preserves_fixed_strategy():
    from scarf.features.variability import fit_lowess

    rng = np.random.default_rng(5)
    means = rng.uniform(0.5, 20.0, 80)
    variances = np.clip(means**1.5 + rng.normal(0, 0.05, len(means)), 0.1, None)
    group = zarr.open_group(store=MemoryStore(), mode="w")
    group.create_array("I", data=np.ones(len(means), dtype=bool))
    group.create_array("means", data=means)
    group.create_array("variances", data=variances)
    table = metadata.MetaData(group)

    observed = table.remove_trend("means", "variances", n_bins=8, lowess_frac=0.6)
    expected = fit_lowess(
        means,
        variances,
        n_bins=8,
        lowess_frac=0.6,
        bin_strategy="fixed",
    )

    np.testing.assert_array_equal(observed, expected)


def test_metadata_row_blocks_remain_pickle_resolvable():
    block = metadata.MetaDataRowBlock(
        start=0,
        stop=2,
        active_global_indices=np.array([0]),
        values={"score": np.array([0.5])},
    )

    restored = pickle.loads(pickle.dumps(block))

    assert type(restored) is metadata.MetaDataRowBlock
    np.testing.assert_array_equal(restored.active_global_indices, [0])

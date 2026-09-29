import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import zarr
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync
from zarr.storage import MemoryStore

import scarf.metadata as metadata
import scarf.metadata.selection as selection
from scarf.metadata.selection import (
    CellField,
    FeatureRef,
    NamedCellArtifact,
    NormalizationSpec,
    grouping_value_name,
    resolve_cell_aligned_artifact,
    resolve_grouping,
    valid_category_mask,
)
from scarf.metadata.queries import missing_frame_values
from scarf.metadata.rows import MetaDataRowBlock as implementation_row_block
from scarf.metadata.rows import (
    apply_missing_mask,
    iter_metadata_column_blocks,
    metadata_missing_mask,
    read_metadata_missing_rows,
    read_metadata_rows,
    read_metadata_rows_chunkwise,
)
from scarf.metadata.table import MetaData as implementation_metadata
from scarf.storage import ArtifactRef
from tests.signature_contracts import signature_digest


_METHODS = {
    "__init__",
    "__repr__",
    "_column_names",
    "_fill_to_index",
    "_get_array",
    "_get_size",
    "_save",
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
    "multi_sift",
    "reset_key",
    "sift",
    "to_pandas_dataframe",
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
        "7a60ae131135bc959cbc4443f91307a33616740779a0a5d50d3f9b2341152508"
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


def test_metadata_fetch_reads_only_the_requested_column():
    from tests.store_probes import RecordingStore

    store = RecordingStore()
    group = zarr.open_group(store=store, mode="w")
    for index in range(20):
        group.create_array(f"value{index}", data=np.arange(3))
    table = metadata.MetaData(group)
    store.reset()
    np.testing.assert_array_equal(table.fetch_all("value4"), np.arange(3))
    metadata_reads = [
        key
        for operation, key in store.ops
        if operation == "get" and key.endswith("zarr.json")
    ]
    assert metadata_reads and set(metadata_reads) == {"value4/zarr.json"}
    group.create_array("added", data=np.ones(3))
    np.testing.assert_array_equal(table.fetch_all("added"), np.ones(3))
    del group["added"]
    with pytest.raises(KeyError):
        table.fetch_all("added")
    group.create_array("nested/hidden", data=np.ones(3))
    with pytest.raises(KeyError):
        table.fetch_all("nested/hidden")


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


def test_apply_missing_mask_shows_masked_rows_as_missing():
    missing = np.array([False, True, False])
    counts = np.array([3, 0, 5], dtype=np.int64)

    assert apply_missing_mask(counts, None) is counts
    unmasked = apply_missing_mask(counts, np.zeros(3, dtype=bool))
    assert unmasked.dtype == np.int64
    numeric = apply_missing_mask(counts, missing)
    assert numeric.dtype == np.float64
    np.testing.assert_array_equal(numeric, [3.0, np.nan, 5.0])
    np.testing.assert_array_equal(counts, [3, 0, 5])
    np.testing.assert_array_equal(
        apply_missing_mask(np.array([True, True, True]), missing),
        [True, False, True],
    )
    assert apply_missing_mask(np.array(["a", "", "b"]), missing).tolist() == [
        "a",
        None,
        "b",
    ]
    assert apply_missing_mask(counts, missing, labels=True).tolist() == [3, None, 5]
    assert apply_missing_mask(
        np.array([True, False, True]), missing, labels=True
    ).tolist() == [True, None, True]
    with pytest.raises(ValueError, match="does not align"):
        apply_missing_mask(counts, np.array([True]))


def test_metadata_frames_show_masked_rows_as_missing_and_fetch_stays_raw():
    table = _metadata_fixture()
    group = table.locations["primary"]
    missing = np.array([False, True, True, False])
    for name, values in (
        ("donor", np.array([1, 0, 0, 2], dtype=np.int64)),
        ("site", np.array(["A", "", "", "B"])),
        ("flag", np.array([True, False, False, False])),
    ):
        group.create_array(name, data=values, chunks=(2,))
        group.create_array(f"__scarf_missing__{name}", data=missing, chunks=(2,))
        group[name].attrs["missing_mask"] = f"__scarf_missing__{name}"

    np.testing.assert_array_equal(table.fetch_all("donor"), [1, 0, 0, 2])
    np.testing.assert_array_equal(table.fetch("site"), ["A", "", "B"])
    frame = table.to_pandas_dataframe(["donor", "site", "flag", "score"], key="I")
    assert frame["donor"].dtype == np.float64
    assert frame["flag"].dtype == "boolean"
    np.testing.assert_array_equal(
        frame.isna().to_numpy(),
        [[False, False, False, False], [True, True, True, False], [False] * 4],
    )
    assert frame["donor"].tolist()[::2] == [1.0, 2.0]
    assert frame["site"].tolist()[::2] == ["A", "B"]
    assert frame["flag"].tolist()[::2] == [True, False]
    head = table.head(2)
    assert head["donor"].isna().tolist() == [False, True]
    assert head["site"].isna().tolist() == [False, True]
    assert head["score"].tolist() == [0.5, 2.0]
    boolean = missing_frame_values(np.array([True, True]), np.array([True, False]))
    assert boolean.dtype == "boolean"
    assert boolean.isna().tolist() == [True, False]


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


def _artifact(kind: str, token: str) -> ArtifactRef:
    return ArtifactRef(
        scope="datastore",
        kind=kind,
        artifact_id=token * 64,
    )


def test_metadata_selection_value_contracts_reject_ambiguous_inputs():
    np.testing.assert_array_equal(
        valid_category_mask(np.array([b"", b"a", np.bytes_(b" ")], dtype=object)),
        [False, True, False],
    )
    with pytest.raises(ValueError, match="one-dimensional"):
        valid_category_mask(np.ones((2, 2)))
    with pytest.raises(ValueError, match="missing mask must align"):
        valid_category_mask(["a", "b"], missing_mask=[True])
    with pytest.raises(ValueError, match="by must be"):
        FeatureRef("gene", by="label")
    with pytest.raises(ValueError, match="reduction must be"):
        FeatureRef("gene", reduction="median")
    with pytest.raises(ValueError, match="non-empty name"):
        NamedCellArtifact("", _artifact("cluster_labels", "a"))
    with pytest.raises(ValueError, match="surrounding whitespace"):
        NamedCellArtifact(" labels ", _artifact("cluster_labels", "a"))
    with pytest.raises(TypeError, match="ArtifactRef"):
        NamedCellArtifact("labels", object())
    with pytest.raises(ValueError, match="categorical cell labels"):
        grouping_value_name("embedding")
    with pytest.raises(ValueError, match="transform"):
        NormalizationSpec(transform="sqrt")


def test_cell_aligned_artifact_validation_reports_each_broken_contract(monkeypatch):
    artifact = _artifact("cluster_labels", "b")
    source = _artifact("cell_selection", "c")
    target = _artifact("cell_selection", "d")

    with pytest.raises(TypeError, match="artifact must be"):
        resolve_cell_aligned_artifact(None, object())
    with pytest.raises(ValueError, match="Expected a 'cell_cycle'"):
        resolve_cell_aligned_artifact(None, artifact, expected_kind="cell_cycle")
    with pytest.raises(ValueError, match="value_name must be"):
        resolve_cell_aligned_artifact(None, artifact, value_name="")

    monkeypatch.setattr(
        selection,
        "inspect_artifact",
        lambda *_: SimpleNamespace(exists=False, complete=False, inputs={}),
    )
    with pytest.raises(ValueError, match="unavailable or incomplete"):
        resolve_cell_aligned_artifact(None, artifact)

    status = SimpleNamespace(exists=True, complete=True, inputs={})
    monkeypatch.setattr(selection, "inspect_artifact", lambda *_: status)
    with pytest.raises(ValueError, match="no cell-selection input"):
        resolve_cell_aligned_artifact(None, artifact)

    status.inputs = {"cell_selection": {"type": "wrong"}}
    with pytest.raises(ValueError, match="cell selection is malformed"):
        resolve_cell_aligned_artifact(None, artifact)

    status.inputs = {"cell_selection": source.to_dict()}
    monkeypatch.setattr(selection, "_selection_indices", lambda *_: np.array([1, 3]))
    with pytest.raises(TypeError, match="cell_selection must be"):
        resolve_cell_aligned_artifact(None, artifact, cell_selection=object())

    def out_of_bounds(_root, ref):
        return np.array([1, 3]) if ref == source else np.array([5])

    monkeypatch.setattr(selection, "_selection_indices", out_of_bounds)
    with pytest.raises(ValueError, match="must be a subset"):
        resolve_cell_aligned_artifact(None, artifact, cell_selection=target)

    def not_a_member(_root, ref):
        return np.array([1, 3]) if ref == source else np.array([2])

    monkeypatch.setattr(selection, "_selection_indices", not_a_member)
    with pytest.raises(ValueError, match="must be a subset"):
        resolve_cell_aligned_artifact(None, artifact, cell_selection=target)

    monkeypatch.setattr(selection, "_selection_indices", lambda *_: np.array([1, 3]))
    monkeypatch.setattr(selection, "artifact_group", lambda *_: {})
    with pytest.raises(ValueError, match="no 'values' value array"):
        resolve_cell_aligned_artifact(None, artifact)

    bad_array = SimpleNamespace(ndim=1, shape=(3,))
    monkeypatch.setattr(selection, "artifact_group", lambda *_: {"values": bad_array})
    monkeypatch.setattr(selection, "as_zarr_array", lambda value, **_: value)
    with pytest.raises(ValueError, match="one value per source-selected cell"):
        resolve_cell_aligned_artifact(None, artifact)

    good_array = SimpleNamespace(ndim=1, shape=(2,))
    monkeypatch.setattr(selection, "artifact_group", lambda *_: {"values": good_array})
    monkeypatch.setattr(
        selection,
        "read_array_rows_chunkwise",
        lambda *_: np.array([1]),
    )
    with pytest.raises(ValueError, match="values do not match"):
        resolve_cell_aligned_artifact(None, artifact)


def test_resolve_grouping_validates_field_kind_and_missing_mask(monkeypatch):
    cells = SimpleNamespace(N=2)
    with pytest.raises(ValueError, match="must be categorical"):
        resolve_grouping(None, cells, CellField("score", kind="continuous"))

    monkeypatch.setattr(
        selection, "read_metadata_rows", lambda *_: np.array(["a", "b"])
    )
    monkeypatch.setattr(
        selection, "read_metadata_missing_rows", lambda *_: np.array([True])
    )
    with pytest.raises(ValueError, match="missing mask does not align"):
        resolve_grouping(None, cells, CellField("group", kind="categorical"))


def test_case_insensitive_index_matches_the_text_of_any_value():
    from scarf.metadata.table import CaseInsensitiveIndex

    index = CaseInsensitiveIndex(["Gene", "GENE", None, 7, np.nan, "other"])

    assert index.positions("gene") == [0, 1]
    assert index.positions(7) == [3]
    assert index.positions("missing") == []
    index.positions("gene").append(9)
    assert index.positions("gene") == [0, 1]


def test_metadata_table_fill_and_error_contracts():
    empty = zarr.open_group(store=MemoryStore(), mode="w")
    with pytest.raises(ValueError, match="empty zarr group"):
        metadata.MetaData(empty)

    primary = zarr.open_group(store=MemoryStore(), mode="w")
    primary.create_array("I", data=np.array([True, False, True, False]), chunks=(2,))
    primary.create_array("ids", data=np.array(["a", "b", "c", "d"]), chunks=(2,))
    primary.create_array("score", data=np.arange(4.0), chunks=(2,))
    primary.create_array("x_value", data=np.arange(4), chunks=(2,))
    table = metadata.MetaData(primary)

    # Listing a group reads every column's metadata, so a corrupt column fails loudly.
    corrupt = zarr.open_group(store=MemoryStore(), mode="w")
    corrupt.create_array("value", data=np.arange(4), chunks=(2,))
    sync(
        corrupt.store.set(
            "value/zarr.json", default_buffer_prototype().buffer.from_bytes(b"{")
        )
    )
    with pytest.raises(ValueError):
        table._get_size(corrupt)
    with pytest.raises(ValueError, match="empty zarr group"):
        table._get_size(empty)
    assert table._has_column("I") and not table._has_column("unknown")
    with pytest.raises(KeyError, match="does not exist"):
        table._get_array("unknown")
    with pytest.raises(KeyError, match="does not exist"):
        table.drop("unknown")
    with pytest.raises(TypeError, match="boolean type column"):
        table._bool_array("score")
    assert list(table.locations) == ["primary"]

    with pytest.raises(ValueError, match="Expected shape"):
        table._save("value", np.arange(2))

    np.testing.assert_array_equal(
        table._fill_to_index([5, 6], np.nan, "I"), [5, 0, 6, 0]
    )
    with pytest.raises(ValueError, match="integer value"):
        table._fill_to_index(np.array([-2, -1]), np.nan, "I")
    with pytest.raises(ValueError, match="integer value"):
        table._fill_to_index(np.array([1, 2]), "bad", "I")
    with pytest.raises(ValueError, match="incorrect length"):
        table._fill_to_index(np.array([1]), 0, "I")

    with pytest.raises(TypeError, match="value_targets"):
        table.get_index_by("a", "ids")
    np.testing.assert_array_equal(table.get_index_by(["A"], "ids", key="I"), [0])
    # With a key, indices are positions among the selected rows.
    np.testing.assert_array_equal(
        table.get_index_by(["C", "a"], "ids", key="I"), [1, 0]
    )
    # Values and targets that are not text match as text instead of raising.
    numeric = table.get_index_by([2, "missing"], "x_value")
    assert numeric.dtype == np.int64
    assert numeric.tolist() == [2]
    assert table.get_index_by(["missing"], "ids").dtype == np.int64
    np.testing.assert_array_equal(
        table.index_to_bool(np.array([1]), invert=True), [True, False, True, True]
    )

    with pytest.raises(ValueError, match="protected column"):
        table.insert("I", np.ones(4, dtype=bool))
    with pytest.raises(ValueError, match="already exists"):
        table.insert("score", np.arange(4.0))
    table.insert("list_values", [1, 2, 3, 4])
    with pytest.raises(ValueError, match="protected name"):
        table.drop("ids")
    np.testing.assert_array_equal(
        table.multi_sift(["score"], [0.0], [3.0]),
        [False, True, True, False],
    )
    assert repr(table) == "MetaData of 2(4) elements"

    with pytest.raises(TypeError, match="boolean type column"):
        table.active_index("score")


def _masked_metadata() -> metadata.MetaData:
    group = zarr.open_group(store=MemoryStore(), mode="w")
    group.create_array("I", data=np.ones(4, dtype=bool))
    group.create_array("ids", data=np.array(["a", "b", "c", "d"]))
    group.create_array("names", data=np.array(["a", "b", "c", "d"]))
    for name, values in (
        ("zeta", np.array([1, 0, 7, 2], dtype=np.int64)),
        ("count", np.array([5, 0, 7, 2], dtype=np.int64)),
        ("label", np.array(["a", "", "b", "a"])),
        ("donor", np.array(["x", "", "y", "x"])),
    ):
        array = group.create_array(name, data=values)
        group.create_array(
            f"__scarf_missing__{name}", data=np.array([False, True, False, False])
        )
        array.attrs["missing_mask"] = f"__scarf_missing__{name}"
    return metadata.MetaData(group)


def test_metadata_columns_are_listed_in_a_stable_order():
    table = _masked_metadata()

    assert table.columns == ["I", "ids", "names", "count", "donor", "label", "zeta"]


def test_sift_never_passes_rows_with_a_linked_missing_mask():
    table = _masked_metadata()

    np.testing.assert_array_equal(
        table.sift("count", -1, 10), [True, False, True, True]
    )
    np.testing.assert_array_equal(
        table.multi_sift(["count", "zeta"], [-1, -1], [10, 10]),
        [True, False, True, True],
    )


def test_partition_helpers_treat_masked_rows_as_missing():
    from scarf.metadata.queries import (
        column_constant_within,
        column_partition_digest,
        columns_same_partition,
        reduce_observation_units,
    )

    table = _masked_metadata()
    digest = column_partition_digest(table, "label")

    assert (digest.nMissing, digest.nLevels, digest.nRows) == (1, 3, 4)
    assert columns_same_partition(table, "label", "donor")[0]
    assert column_constant_within(table, "donor", "label")
    units = reduce_observation_units(table, "donor", ["label"])
    assert units["donor"].isna().tolist() == [False, True, False]
    assert units["label"].isna().tolist() == [False, True, False]
    assert units.dropna()["label"].tolist() == ["a", "b"]


def test_insert_keeps_an_explicit_boolean_fill_value():
    table = _metadata_fixture()

    table.insert("keep", np.array([True, False, False]), fill_value=True)
    table.insert("drop_rows", np.array([True, False, False]))

    np.testing.assert_array_equal(table.fetch_all("keep"), [True, True, False, False])
    np.testing.assert_array_equal(
        table.fetch_all("drop_rows"), [True, False, False, False]
    )


def test_head_reads_only_the_requested_rows(monkeypatch):
    table = _metadata_fixture()

    def fail(_column):
        raise AssertionError("head must not read whole columns")

    monkeypatch.setattr(table, "fetch_all", fail)
    assert table.head(2)["score"].tolist() == [0.5, 2.0]


@pytest.mark.parametrize("name", ["a/b", "a\\b"], ids=["slash", "backslash"])
def test_insert_rejects_zarr_path_separators_before_writing(name):
    table = _metadata_fixture()

    with pytest.raises(ValueError, match="path separators; use 'a_b' instead"):
        table.insert(name, np.arange(4.0))

    assert set(table._group.keys()) == {"I", "ids", "names", "score"}
    assert not table._is_public(name)
    with pytest.raises(KeyError, match="look for 'a_b' or a numbered variant"):
        table.fetch_all(name)
    with pytest.raises(KeyError, match="'a_b_2'"):
        table.drop(name)


@pytest.mark.parametrize("name", ["r/k", "r\\k"], ids=["slash", "backslash"])
def test_reset_key_rejects_zarr_path_separators_before_writing(name):
    table = _metadata_fixture()

    with pytest.raises(ValueError, match="path separators; use 'r_k' instead"):
        table.reset_key(name)

    assert set(table._group.keys()) == {"I", "ids", "names", "score"}


def test_lookups_of_names_that_are_not_text_raise_key_errors():
    table = _metadata_fixture()

    for column in (123, None):
        with pytest.raises(KeyError, match="does not exist in the metadata columns"):
            table.fetch_all(column)  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["__scarf_missing__score", "__scarf_missing__new"])
def test_writes_reject_the_missing_mask_prefix(name):
    table = _metadata_fixture()

    with pytest.raises(ValueError, match="reserves for missing-value masks"):
        table.insert(name, np.ones(4, dtype=bool))
    with pytest.raises(ValueError, match="reserves for missing-value masks"):
        table.reset_key(name)

    assert set(table._group.keys()) == {"I", "ids", "names", "score"}


@pytest.mark.parametrize("name", ["", ".", ".."])
def test_lookups_of_names_zarr_cannot_store_raise_key_errors(name):
    table = _metadata_fixture()

    with pytest.raises(KeyError, match="does not exist in the metadata columns"):
        table.fetch_all(name)
    with pytest.raises(KeyError, match="does not exist in the metadata columns"):
        table.drop(name)


def test_writes_over_a_nested_group_ask_for_a_new_import():
    table = _metadata_fixture()
    table._group.create_array("Baseline (ml/min)", data=np.ones(4))

    for write in (
        lambda: table.drop("Baseline (ml"),
        lambda: table.insert("Baseline (ml", np.zeros(4), overwrite=True),
        lambda: table.reset_key("Baseline (ml"),
    ):
        with pytest.raises(TypeError, match="import the source again"):
            write()
    assert list(table._group["Baseline (ml"].array_keys()) == ["min)"]


def test_nested_group_from_an_older_import_asks_for_a_new_import():
    table = _metadata_fixture()
    # Older imports let Zarr nest a source column named with '/' into groups.
    table._group.create_array("Baseline (ml/min)", data=np.ones(4))

    assert "Baseline (ml" in table.columns
    with pytest.raises(TypeError, match="nested group named 'Baseline \\(ml'"):
        table.fetch_all("Baseline (ml")
    with pytest.raises(TypeError, match="import the source again"):
        table.head()

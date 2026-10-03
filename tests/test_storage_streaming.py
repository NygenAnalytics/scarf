import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.metadata import MetaData
from scarf.storage.arrays import MetadataBlock, create_streamed_metadata_column
from scarf.storage.artifacts import (
    artifact_group,
    fingerprint_strings,
)
from scarf.storage.schema import (
    create_empty_cell_data,
    create_empty_zarr_count_assay,
)
from scarf.storage.selections import (
    fingerprint_selected_stored_strings,
    resolve_stored_selection_artifact,
)


def test_streamed_metadata_column_writes_contiguous_blocks_and_mask():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    group = root.create_group("cellData")
    values = create_streamed_metadata_column(
        group,
        "cluster",
        shape=4,
        dtype="U2",
        blocks=(
            MetadataBlock(0, np.array(["a", ""], dtype="U2"), np.array([0, 1])),
            MetadataBlock(2, np.array(["bb", "a"], dtype="U2"), np.array([0, 0])),
        ),
        hasMissing=True,
    )

    assert values[:].tolist() == ["a", "", "bb", "a"]
    missing_name = values.attrs["missing_mask"]
    assert group[missing_name][:].tolist() == [False, True, False, False]


def test_streamed_metadata_column_casts_blocks_and_marks_unflagged_rows_present():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    values = create_streamed_metadata_column(
        root,
        "count",
        shape=5,
        dtype=np.int16,
        blocks=(
            MetadataBlock(0, np.array([1, 2], dtype=np.int64), np.array([0, 1])),
            # A block without a mask holds only present values.
            MetadataBlock(2, np.array([3.0, 4.0, 5.0])),
        ),
        chunkSize=2,
        hasMissing=True,
    )

    assert values.dtype == np.dtype(np.int16)
    assert values.chunks == (2,)
    np.testing.assert_array_equal(values[:], [1, 2, 3, 4, 5])
    mask = root[values.attrs["missing_mask"]]
    np.testing.assert_array_equal(mask[:], [False, True, False, False, False])


@pytest.mark.parametrize(
    ("options", "blocks", "message"),
    [
        ({"shape": -1}, (), "shape must be non-negative"),
        ({"chunkSize": 0}, (), "chunkSize must be positive"),
        (
            {},
            (MetadataBlock(1, np.array([], dtype=np.int32)),),
            "Metadata blocks must be contiguous; expected 0, received 1",
        ),
        (
            {},
            (MetadataBlock(0, np.zeros((1, 1), dtype=np.int32)),),
            "Metadata blocks must be one-dimensional",
        ),
        (
            {},
            (MetadataBlock(0, np.arange(4, dtype=np.int32)),),
            "Metadata block exceeds declared shape",
        ),
        (
            {},
            (MetadataBlock(0, np.arange(3, dtype=np.int32), np.zeros(3)),),
            "A missing mask was supplied but hasMissing is false",
        ),
        (
            {"hasMissing": True},
            (MetadataBlock(0, np.arange(3, dtype=np.int32), np.zeros(2)),),
            "Missing mask must align with metadata values",
        ),
        (
            {},
            (MetadataBlock(0, np.arange(2, dtype=np.int32)),),
            "Metadata column is incomplete: wrote 2 of 3 rows",
        ),
    ],
    ids=[
        "negative-shape",
        "zero-chunks",
        "gap",
        "two-dimensional",
        "overflow",
        "unexpected-mask",
        "misaligned-mask",
        "incomplete",
    ],
)
def test_streamed_metadata_column_rejects_malformed_blocks(options, blocks, message):
    root = zarr.open_group(store=MemoryStore(), mode="w")
    arguments = {"shape": 3, "dtype": np.int32, **options}
    with pytest.raises(ValueError, match=f"^{message}"):
        create_streamed_metadata_column(root, "bad", blocks=blocks, **arguments)


def test_internal_missing_columns_are_not_public_metadata():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    group = create_empty_cell_data(root, None, 2, "U2", "U2")
    group["ids"][:] = np.array(["c1", "c2"])
    group["names"][:] = np.array(["c1", "c2"])
    create_streamed_metadata_column(
        group,
        "label",
        shape=2,
        dtype="U1",
        blocks=(MetadataBlock(0, np.array(["a", ""], dtype="U1"), np.array([0, 1])),),
        hasMissing=True,
    )

    metadata = MetaData(group)
    assert "label" in metadata.columns
    assert "__scarf_missing__label" not in metadata.columns


def test_empty_assay_schema_accepts_blockwise_feature_metadata():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    counts, feature_data = create_empty_zarr_count_assay(
        root,
        "RNA",
        None,
        3,
        2,
        "U2",
        "U6",
        dtype=np.uint32,
    )
    feature_data["ids"][:] = np.array(["g1", "g2"])
    feature_data["names"][:] = np.array(["Gene 1", "Gene 2"])

    assert counts.shape == (3, 2)
    assert feature_data["ids"][:].tolist() == ["g1", "g2"]
    assert feature_data["I"][:].tolist() == [True, True]


def test_stored_selection_artifact_copies_values_blockwise():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    cells = create_empty_cell_data(root, None, 4, "U2", "U2")
    cells["ids"][:] = np.array(["c1", "c2", "c3", "c4"])
    cells["names"][:] = cells["ids"][:]
    cells["I"][:] = np.array([True, False, True, True])

    ref = resolve_stored_selection_artifact(
        root,
        table_path="cellData",
        id_column="ids",
        source_column="I",
        scope="datastore",
        kind="cell_selection",
        operation="import_cell_selection",
        parameters={"source": "seurat"},
        inputs={},
    )

    assert artifact_group(root, ref)["values"][:].tolist() == [
        True,
        False,
        True,
        True,
    ]


def test_selected_stored_string_fingerprint_matches_materialized_values():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    ids = root.create_array(
        "ids",
        data=np.array(["cell-1", "cell-2", "cell-3", "cell-4"], dtype="U6"),
        chunks=(2,),
    )
    selection = root.create_array(
        "selection",
        data=np.array([True, False, True, True]),
        chunks=(2,),
    )

    digest, count = fingerprint_selected_stored_strings(ids, selection)

    assert count == 3
    assert digest == fingerprint_strings(ids[:][selection[:]])

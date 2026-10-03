import numpy as np
import pytest
import zarr

from scarf import DataStore
from scarf.datastore.datastore import mount_datastore
from scarf.features.gene_families import DEFAULT_PERCENT_PATTERNS
from scarf.metadata import MetaData
from scarf.metadata.rows import read_metadata_missing_rows
from scarf.metadata.selection import CellField, resolve_grouping, valid_category_mask
from scarf.storage.copy import copy_zarr_group_tree
from scarf.storage.identity import read_dataset_fingerprint
from scarf.storage.schema import create_zarr_count_assay
from scarf.tools.repack_zarr import repack_store
from scarf.writers.counts_t import finalize_writer_counts_t
from scarf.utils.logging import logger
from scarf.writers.subset import SubsetZarr
from tests.storage_helpers import finalize_test_counts
from tests.qc_helpers import (
    QC_VALUES,
    create_labelled_qc_store,
    open_qc_store,
    fresh_qc_store,
)


def test_first_preparation_replaces_incorrect_summaries_and_column_attributes():
    storage, _ = fresh_qc_store()
    root = zarr.open_group(store=storage, mode="r+")
    cells = MetaData(root["cellData"])
    cells.insert("RNA_nCounts", np.full(6, 999.0))
    cells.insert("RNA_nFeatures", np.full(6, 999.0))
    root["cellData/RNA_nCounts"].attrs["source_artifact"] = "omitted"
    store = open_qc_store(storage)
    np.testing.assert_array_equal(
        store.cells.fetch_all("RNA_nCounts"), QC_VALUES.sum(axis=1)
    )
    np.testing.assert_array_equal(
        store.cells.fetch_all("RNA_nFeatures"), (QC_VALUES > 0).sum(axis=1)
    )
    assert "source_artifact" not in store.zw["cellData/RNA_nCounts"].attrs
    assert read_dataset_fingerprint(store.RNA.z)


def test_construction_retry_finishes_interrupted_transpose():
    storage, _ = fresh_qc_store()
    root = zarr.open_group(store=storage, mode="r+")
    root["RNA/countsT"].attrs["complete"] = False
    finalize_writer_counts_t(root, "RNA", None, nthreads=1)
    assert root["RNA/countsT"].attrs["complete"] is True
    np.testing.assert_array_equal(root["RNA/countsT"][:], QC_VALUES.T)
    open_qc_store(storage)
    with pytest.raises(ValueError, match="Prepared counts"):
        finalize_writer_counts_t(root, "RNA", None, nthreads=1)


def test_merge_preserves_feature_annotations_and_missing_flags(tmp_path):
    from scarf.merge import DataStoreMerge

    left = create_labelled_qc_store(tmp_path / "left")
    right = create_labelled_qc_store(tmp_path / "right")
    for store in (left, right):
        features = store.RNA.z["featureData"]
        features.create_array("label", data=np.array(["a", "", "c", "d", "e", "f"]))
        features.create_array(
            "__scarf_missing__label",
            data=np.array([False, True, False, False, False, False]),
        )
        features["label"].attrs.update(
            {"missing_mask": "__scarf_missing__label", "description": "Feature label"}
        )
    destination = str(tmp_path / "merged")
    DataStoreMerge([left, right], destination, ["left", "right"], nthreads=1).dump()
    merged = DataStore(destination, nthreads=1, min_features_per_cell=0)
    np.testing.assert_array_equal(
        merged.RNA.feats.fetch_all("label"), ["a", "", "c", "d", "e", "f"]
    )
    np.testing.assert_array_equal(
        read_metadata_missing_rows(merged.RNA.feats, "label", np.arange(6)),
        [False, True, False, False, False, False],
    )
    assert merged.RNA.feats._get_array("label").attrs["description"] == "Feature label"
    right.RNA.feats._get_array("label")[0] = "conflict"
    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]),
        level="WARNING",
    )
    try:
        DataStoreMerge(
            [left, right], str(tmp_path / "conflict"), ["left", "right"], nthreads=1
        ).dump()
    finally:
        logger.remove(sink)
    conflicted = DataStore(
        str(tmp_path / "conflict"), nthreads=1, min_features_per_cell=0
    )
    assert "label" not in conflicted.RNA.feats.columns
    assert "__scarf_missing__label" not in conflicted.RNA.z["featureData"]
    assert any("were not merged: label" in message for message in messages)


@pytest.mark.parametrize(
    ("table", "column"),
    [
        ("cellData", "ids"),
        ("cellData", "RNA_nCounts"),
        ("cellData", "RNA_nFeatures"),
        ("cellData", "RNA_percentMito"),
        ("RNA/featureData", "ids"),
        ("RNA/featureData", "nCells"),
        ("RNA/featureData", "dropOuts"),
    ],
)
def test_separate_metadata_handle_cannot_replace_protected_columns(table, column):
    storage, _ = fresh_qc_store()
    open_qc_store(storage)
    other = MetaData(zarr.open_group(store=storage, mode="r+")[table])
    before = other.fetch_all(column)
    prepared = f"^Column {column!r} belongs to prepared data and cannot be changed"
    with pytest.raises(ValueError, match=prepared):
        other.insert(column, before[::-1], force=True, overwrite=True)
    # IDs are refused by name before the prepared-data check.
    with pytest.raises(
        ValueError,
        match=(
            "^ERROR: ids is a protected name in MetaData class"
            if column == "ids"
            else prepared
        ),
    ):
        other.drop(column)
    np.testing.assert_array_equal(other.fetch_all(column), before)


@pytest.mark.parametrize("operation", ["mount", "repack"])
def test_unchanged_copy_preserves_percentage_proof_and_missing_values(
    tmp_path, operation
):
    source = tmp_path / "source.zarr"
    target = tmp_path / "target.zarr"
    original = create_labelled_qc_store(source)
    if operation == "mount":
        mount_datastore(str(source), str(target))
    else:
        repack_store(str(source), str(target), nthreads=1)
    copied = DataStore(
        str(target),
        zarr_mode="r",
        mito_pattern=DEFAULT_PERCENT_PATTERNS["percentMito"],
        ribo_pattern=DEFAULT_PERCENT_PATTERNS["percentRibo"],
        nthreads=1,
    )
    assert read_dataset_fingerprint(copied.RNA.z) == read_dataset_fingerprint(
        original.RNA.z
    )
    np.testing.assert_array_equal(
        read_metadata_missing_rows(copied.cells, "label", np.arange(6)),
        [False, True, False, False, True, False],
    )
    grouping = resolve_grouping(copied.zw, copied.cells, CellField("label"))
    np.testing.assert_array_equal(
        valid_category_mask(grouping.labels, missing_mask=grouping.missing_mask),
        [True, False, True, True, False, True],
    )


def test_subset_maps_missing_flags_and_recomputes_statistics(tmp_path):
    source = create_labelled_qc_store(tmp_path / "source.zarr")
    target = tmp_path / "subset.zarr"
    rows = np.array([5, 1, 3])
    writer = SubsetZarr(str(target), [source.RNA], cell_idx=rows, nthreads=1)
    writer.dump()
    copied = DataStore(
        str(target), default_assay="RNA", min_features_per_cell=0, nthreads=1
    )
    np.testing.assert_array_equal(copied.RNA.rawData.compute(), QC_VALUES[rows])
    np.testing.assert_array_equal(
        read_metadata_missing_rows(copied.cells, "label", np.arange(3)),
        [False, True, False],
    )
    np.testing.assert_array_equal(
        copied.RNA.feats.fetch_all("nCells"), (QC_VALUES[rows] > 0).sum(axis=0)
    )
    assert read_dataset_fingerprint(copied.RNA.z) != read_dataset_fingerprint(
        source.RNA.z
    )


def test_data_only_rebuild_discards_broken_generated_metadata(tmp_path):
    source = tmp_path / "source.zarr"
    target = tmp_path / "rebuilt.zarr"
    original = create_labelled_qc_store(source)
    root = zarr.open_group(str(source), mode="r+")
    root["cellData/RNA_nCounts"][:] = 999
    root["cellData/RNA_nCounts"].attrs["missing_mask"] = "missing-array"
    del root["RNA"].attrs["prepared"]
    root["RNA"].create_group("artifacts").create_group("old-result")
    repack_store(str(source), str(target), data_only=True, nthreads=1)
    rebuilt = DataStore(str(target), zarr_mode="r", nthreads=1)
    np.testing.assert_array_equal(
        rebuilt.cells.fetch_all("RNA_nCounts"), QC_VALUES.sum(axis=1)
    )
    assert "artifacts" not in rebuilt.RNA.z
    np.testing.assert_array_equal(
        rebuilt.cells.fetch_all("label"), original.cells.fetch_all("label")
    )


def test_metadata_copy_rejects_missing_dependency_before_writes():
    source = zarr.group()
    source.create_array("value", data=np.array(["a", "b"]))
    source["value"].attrs["missing_mask"] = "absent"
    destination = zarr.group()
    with pytest.raises(ValueError, match="malformed missing-mask link"):
        copy_zarr_group_tree(source, destination)
    assert list(destination.members()) == []


def test_count_identity_detects_changes_with_equal_summaries():
    first = np.array([[1, 2], [2, 1]], dtype=np.uint32)
    second = np.array([[2, 1], [1, 2]], dtype=np.uint32)
    root = zarr.group()
    a = root.create_array("a", data=first, chunks=(1, 2))
    b = root.create_array("b", data=second, chunks=(2, 1))
    c = root.create_array("c", data=first.astype(">u4"), chunks=(2, 1))
    assert finalize_test_counts(a) != finalize_test_counts(b)
    assert finalize_test_counts(a) == finalize_test_counts(c)


def test_shared_physical_matrix_collision_fails_before_writes():
    root = zarr.group()
    counts = create_zarr_count_assay(
        root,
        "RNA",
        "first",
        2,
        feat_ids=np.array(["a", "b"]),
        feat_names=np.array(["a", "b"]),
        dtype="uint32",
    )
    counts[:] = [[1, 2], [3, 4]]
    with pytest.raises(
        ValueError,
        match="^Assay 'RNA' already has metadata or a count matrix; choose a new name$",
    ):
        create_zarr_count_assay(
            root,
            "RNA",
            "second",
            2,
            feat_ids=np.array(["a", "b"]),
            feat_names=np.array(["a", "b"]),
            dtype="uint32",
        )
    assert "second" not in root
    np.testing.assert_array_equal(counts[:], [[1, 2], [3, 4]])


def test_first_preparation_discards_imported_percentages_with_warning():
    storage, _ = fresh_qc_store()
    cells = MetaData(zarr.open_group(store=storage, mode="r+")["cellData"])
    cells.insert("RNA_percentMito", np.full(6, 42.0))

    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]),
        level="WARNING",
    )
    try:
        store = open_qc_store(storage)
    finally:
        logger.remove(sink)

    discarded = [message for message in messages if "Discarding existing" in message]
    assert len(discarded) == 1
    assert "'RNA_percentMito'" in discarded[0]
    mito = store.cells.fetch_all("RNA_percentMito")
    assert not np.any(mito == 42.0)
    assert np.isnan(mito[4])


def test_prepared_store_explains_how_to_add_a_missing_percentage():
    storage, _ = fresh_qc_store()
    open_qc_store(storage, mito_pattern="^NO_SUCH_GENE$")

    with pytest.raises(ValueError) as caught:
        open_qc_store(storage, mito_pattern="^MT-")
    message = str(caught.value)
    assert "'RNA_percentMito' was not computed" in message
    assert "fresh store" in message
    assert "run_feature_percentage" in message
    assert "repack" not in message


def test_clear_column_fails_closed_on_a_noncanonical_mask_link():
    from scarf.storage.identity import clear_column

    cells = zarr.group().create_group("cellData")
    cells.create_array("label", data=np.array(["a", "b"]))
    cells.create_array("mask", data=np.array([False, True]))
    cells["label"].attrs["missing_mask"] = "mask"

    with pytest.raises(ValueError, match="malformed missing-mask link"):
        clear_column(cells, "label")
    assert {"label", "mask"} <= set(cells.array_keys())


def test_first_preparation_publishes_identity_with_the_prepared_flag(monkeypatch):
    import json

    storage, _ = fresh_qc_store()
    documents = []
    store_type = type(storage)
    original_set = store_type.set

    async def recording_set(self, key, value, *args, **kwargs):
        if key == "RNA/zarr.json":
            documents.append(json.loads(value.to_bytes())["attributes"])
        return await original_set(self, key, value, *args, **kwargs)

    monkeypatch.setattr(store_type, "set", recording_set)
    open_qc_store(storage)

    prepared = [attrs for attrs in documents if attrs.get("prepared") is True]
    assert prepared
    assert all(
        attrs.get("dataset_fingerprint") and attrs.get("counts_fingerprint")
        for attrs in prepared
    )

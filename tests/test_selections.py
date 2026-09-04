import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.storage.artifacts import (
    ArtifactRef,
    artifact_group,
    artifact_path,
    fingerprint_strings,
    inspect_artifact,
)
from scarf.storage.arrays import create_metadata_column
from scarf.storage.errors import ArtifactResolutionError
from scarf.storage.selections import (
    fingerprint_selected_stored_strings,
    iter_full_axis_selection_blocks,
    iter_selected_axis_selection_blocks,
    iter_stored_selection_blocks,
    read_stored_selection_indices,
    read_stored_selection_mask,
    resolve_generated_selection_artifact,
    resolve_metadata_snapshot,
    resolve_selection_artifact,
    resolve_stored_selection_artifact,
    snapshot_run_metadata,
    validate_run_metadata_snapshot,
    validate_stored_selection_integrity,
    validate_stored_selection_live_alias,
)


def test_create_metadata_column_accepts_utf8_byte_strings() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    values = np.array([b"1000 cells/\xce\xbcl", b"unknown"], dtype=object)
    column = create_metadata_column(root, "cell_number_loaded", data=values)
    assert column[0] == "1000 cells/\u03bcl"
    assert column[1] == "unknown"


def test_fingerprint_selected_stored_strings_rejects_and_hashes() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    ids = create_metadata_column(
        root,
        "ids",
        data=np.array(["a", "b", "c", "d"]),
        dtype=str,
    )
    selection = create_metadata_column(
        root,
        "selection",
        data=np.array([True, False, True, False]),
        dtype=bool,
    )
    digest, count = fingerprint_selected_stored_strings(ids, selection)
    assert count == 2
    assert digest == fingerprint_strings(np.array(["a", "c"]))
    assert digest == fingerprint_selected_stored_strings(ids, selection)[0]

    other = create_metadata_column(
        root,
        "other",
        data=np.array([False, True, False, True]),
        dtype=bool,
    )
    other_digest, other_count = fingerprint_selected_stored_strings(ids, other)
    assert other_count == 2
    assert other_digest == fingerprint_strings(np.array(["b", "d"]))
    assert other_digest != digest

    short_ids = create_metadata_column(
        root,
        "short_ids",
        data=np.array(["a", "b"]),
        dtype=str,
    )
    with pytest.raises(ValueError, match="aligned vectors"):
        fingerprint_selected_stored_strings(short_ids, selection)

    not_bool = create_metadata_column(
        root,
        "not_bool",
        data=np.array([1, 0, 1, 0], dtype=np.int8),
        dtype=np.int8,
    )
    with pytest.raises(TypeError, match="booleans"):
        fingerprint_selected_stored_strings(ids, not_bool)

    numeric_ids = create_metadata_column(
        root,
        "numeric_ids",
        data=np.arange(4, dtype=np.int32),
        dtype=np.int32,
    )
    with pytest.raises(TypeError, match="must contain strings"):
        fingerprint_selected_stored_strings(numeric_ids, selection)


def _create_stored_cell_selection(
    root: zarr.Group,
    values: np.ndarray,
) -> ArtifactRef:
    table = root.create_group("cellData")
    create_metadata_column(
        table,
        "ids",
        data=np.asarray([f"cell_{index}" for index in range(len(values))]),
        dtype=str,
        chunkSize=2,
    )
    create_metadata_column(
        table,
        "I",
        data=values,
        dtype=bool,
        chunkSize=2,
    )
    return resolve_stored_selection_artifact(
        root,
        table_path="cellData",
        id_column="ids",
        source_column="I",
        scope="datastore",
        kind="cell_selection",
        operation="manual_selection",
        parameters={},
        inputs={},
    )


def test_selection_integrity_is_independent_of_live_alias() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    values = np.array([True, False, True, False, True])
    ref = _create_stored_cell_selection(root, values)

    validated = validate_stored_selection_integrity(
        root,
        ref,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )
    assert validated.selected_count == 3
    assert isinstance(validated.values, zarr.Array)

    root["cellData/I"][0] = False
    validate_stored_selection_integrity(
        root,
        ref,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )
    with pytest.raises(ArtifactResolutionError) as live_error:
        validate_stored_selection_live_alias(
            root,
            ref,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
            column="I",
        )
    assert live_error.value.code == "selection_values_changed"
    del root["cellData"]["I"]
    validate_stored_selection_integrity(
        root,
        ref,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )
    with pytest.raises(ArtifactResolutionError) as missing_error:
        validate_stored_selection_live_alias(
            root,
            ref,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
            column="I",
        )
    assert missing_error.value.code == "selection_column_missing"

    root["cellData/ids"][0] = "other"
    with pytest.raises(ArtifactResolutionError) as row_error:
        validate_stored_selection_integrity(
            root,
            ref,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )
    assert row_error.value.code == "row_identity_mismatch"


def test_selection_block_helpers_preserve_compact_and_full_axis_alignment() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    mask = np.array([False, True, True, False, False, True, False])
    ref = _create_stored_cell_selection(root, mask)
    common = {
        "kind": "cell_selection",
        "scope": "datastore",
        "assay": None,
        "table_path": "cellData",
        "block_rows": 2,
    }

    blocks = list(iter_stored_selection_blocks(root, ref, **common))
    assert [(block.start, block.stop) for block in blocks] == [
        (0, 2),
        (2, 4),
        (4, 6),
        (6, 7),
    ]
    assert [(block.compact_start, block.compact_stop) for block in blocks] == [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 3),
    ]
    np.testing.assert_array_equal(
        np.concatenate([block.selected_indices for block in blocks]),
        np.array([1, 2, 5]),
    )
    np.testing.assert_array_equal(
        read_stored_selection_mask(root, ref, **common),
        mask,
    )
    np.testing.assert_array_equal(
        read_stored_selection_indices(root, ref, **common),
        np.array([1, 2, 5]),
    )

    compact = np.array([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]])
    aligned_blocks = list(
        iter_full_axis_selection_blocks(
            root,
            ref,
            compact,
            fill_value=np.nan,
            **common,
        )
    )
    aligned = np.concatenate([block.values for block in aligned_blocks])
    expected = np.full((len(mask), 2), np.nan)
    expected[mask] = compact
    np.testing.assert_equal(aligned, expected)

    full = np.arange(len(mask) * 2).reshape(len(mask), 2)
    selected_blocks = list(
        iter_selected_axis_selection_blocks(root, ref, full, **common)
    )
    assert [(block.start, block.stop) for block in selected_blocks] == [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 3),
    ]
    np.testing.assert_array_equal(
        np.concatenate([block.values for block in selected_blocks]),
        full[mask],
    )


def test_selection_integrity_rejects_tampered_stored_values() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    ref = _create_stored_cell_selection(
        root,
        np.array([True, False, True, False]),
    )
    root[artifact_path(ref)]["values"][1] = True

    with pytest.raises(ArtifactResolutionError) as error:
        validate_stored_selection_integrity(
            root,
            ref,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )
    assert error.value.code == "selection_values_changed"


def test_run_metadata_snapshot_preserves_named_full_axis_columns() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    table = root.create_group("cellData")
    table.create_array(
        "ids",
        data=np.array(
            ["cell_a", "cell_b", "cell_c", "cell_d"],
            dtype=np.dtypes.StringDType(),
        ),
        chunks=(2,),
    )
    table.create_array(
        "names",
        data=np.array(["A", "B", "C", "D"], dtype=np.dtypes.StringDType()),
        chunks=(2,),
    )
    score = create_metadata_column(
        table,
        "score",
        data=np.array([1.5, np.nan, 3.5, 4.5], dtype=np.float32),
        dtype=np.float32,
        chunkSize=2,
    )
    missing = create_metadata_column(
        table,
        "__scarf_missing__score",
        data=np.array([False, True, False, False]),
        dtype=bool,
        chunkSize=2,
    )
    score.attrs["missing_mask"] = "__scarf_missing__score"
    score.attrs["source_artifact"] = {"diagnostic": "must not be copied"}
    create_metadata_column(
        table,
        "batch",
        data=np.array([1, 1, 2, 2], dtype=np.int16),
        dtype=np.int16,
        chunkSize=2,
    )

    first = snapshot_run_metadata(
        root,
        table_path="cellData",
        id_column="ids",
        columns=["names", "score", "batch"],
        axis="cell",
    )
    reused = snapshot_run_metadata(
        root,
        table_path="cellData",
        id_column="ids",
        columns=["names", "score", "batch"],
        axis="cell",
    )

    assert reused == first
    assert first.scope == "datastore"
    assert first.assay is None
    assert first.kind == "metadata_snapshot"
    snapshot = root[artifact_path(first)]
    assert set(snapshot.array_keys()) == {
        "names",
        "score",
        "__scarf_missing__score",
        "batch",
    }
    assert "values" not in snapshot
    np.testing.assert_array_equal(snapshot["names"][:], table["names"][:])
    np.testing.assert_array_equal(snapshot["score"][:], score[:])
    np.testing.assert_array_equal(
        snapshot["__scarf_missing__score"][:],
        missing[:],
    )
    np.testing.assert_array_equal(snapshot["batch"][:], table["batch"][:])
    assert snapshot["score"].dtype == score.dtype
    assert snapshot["batch"].dtype == table["batch"].dtype
    assert np.dtype(snapshot["names"].dtype).kind == "U"
    assert snapshot["score"].attrs.asdict() == {
        "missing_mask": "__scarf_missing__score"
    }
    assert snapshot["names"].attrs.asdict() == {}
    status = inspect_artifact(root, first)
    assert status.operation == "snapshot_run_metadata"
    assert status.parameters == {
        "axis": "cell",
        "assay": None,
        "ordered_columns": ["names", "score", "batch"],
    }
    assert status.inputs is not None
    assert isinstance(status.inputs["ordered_row_ids_fingerprint"], str)
    assert list(status.inputs["column_fingerprints"]) == [
        "names",
        "score",
        "batch",
    ]
    assert all(
        isinstance(value, str)
        for value in status.inputs["column_fingerprints"].values()
    )
    assert (
        validate_run_metadata_snapshot(
            root,
            first,
            axis="cell",
            assay=None,
            table_path="cellData",
            ordered_columns=["names", "score", "batch"],
        )
        == snapshot
    )

    score[0] = np.float32(9.5)
    validate_run_metadata_snapshot(
        root,
        first,
        axis="cell",
        assay=None,
        table_path="cellData",
        ordered_columns=["names", "score", "batch"],
    )
    changed = snapshot_run_metadata(
        root,
        table_path="cellData",
        id_column="ids",
        columns=["names", "score", "batch"],
        axis="cell",
    )
    assert changed != first
    assert snapshot["score"][0] == np.float32(1.5)
    assert root[artifact_path(changed)]["score"][0] == np.float32(9.5)

    snapshot["score"][1] = np.float32(8.5)
    with pytest.raises(ArtifactResolutionError) as payload_error:
        validate_run_metadata_snapshot(
            root,
            first,
            axis="cell",
            assay=None,
            table_path="cellData",
        )
    assert payload_error.value.code == "snapshot_values_changed"


def test_feature_run_metadata_snapshot_is_assay_scoped() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    table = root.create_group("RNA/featureData")
    create_metadata_column(
        table,
        "ids",
        data=np.array(["gene_a", "gene_b", "gene_c"]),
        dtype=str,
    )
    create_metadata_column(
        table,
        "names",
        data=np.array(["A", "B", "C"]),
        dtype=str,
    )
    create_metadata_column(
        table,
        "I",
        data=np.array([True, False, True]),
        dtype=bool,
    )

    ref = snapshot_run_metadata(
        root,
        table_path="RNA/featureData",
        id_column="ids",
        columns=["names", "I"],
        axis="feature",
        assay="RNA",
    )

    assert ref.scope == "assay"
    assert ref.assay == "RNA"
    group = root[artifact_path(ref)]
    assert set(group.array_keys()) == {"names", "I"}
    assert inspect_artifact(root, ref).parameters == {
        "axis": "feature",
        "assay": "RNA",
        "ordered_columns": ["names", "I"],
    }


def test_run_metadata_snapshot_rejects_ambiguous_or_misaligned_inputs() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    table = root.create_group("cellData")
    create_metadata_column(table, "ids", data=np.array(["a", "b"]), dtype=str)
    create_metadata_column(table, "names", data=np.array(["A", "B"]), dtype=str)
    create_metadata_column(table, "short", data=np.array([1]), dtype=np.int8)

    common = {
        "root": root,
        "table_path": "cellData",
        "id_column": "ids",
        "axis": "cell",
    }
    with pytest.raises(ValueError, match="must be unique"):
        snapshot_run_metadata(**common, columns=["names", "names"])
    with pytest.raises(ValueError, match="full metadata axis"):
        snapshot_run_metadata(**common, columns=["short"])
    with pytest.raises(KeyError, match="unavailable"):
        snapshot_run_metadata(**common, columns=["missing"])
    with pytest.raises(ValueError, match="cannot set an assay"):
        snapshot_run_metadata(**common, columns=["names"], assay="RNA")


def test_resolve_selection_artifact_rejects_bad_masks_and_ids() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    row_ids = np.array(["a", "b", "c"])
    common = dict(
        root=root,
        scope="datastore",
        kind="cell_selection",
        row_ids=row_ids,
        operation="manual_selection",
        parameters={},
        inputs={},
        source_column="I",
    )
    with pytest.raises(TypeError, match="one-dimensional boolean"):
        resolve_selection_artifact(**common, values=np.array([1, 0, 1]))
    with pytest.raises(ValueError, match="must align"):
        resolve_selection_artifact(
            **common,
            values=np.array([True, False]),
        )
    with pytest.raises(TypeError, match="one-dimensional boolean"):
        resolve_generated_selection_artifact(
            **common,
            values=np.array([[True, False, True]]),
            assay="RNA",
        )
    with pytest.raises(ValueError, match="must align"):
        resolve_generated_selection_artifact(
            root=root,
            scope="assay",
            assay="RNA",
            kind="feature_selection",
            values=np.array([True, False, True]),
            row_ids=np.array(["a", "b"]),
            operation="select_hvgs",
            parameters={},
            inputs={},
            source_column="hvgs",
        )


def test_selection_artifacts_snapshot_values_and_reuse_by_provenance() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    values = np.array([True, False, True, False])
    row_ids = np.array(["a", "b", "c", "d"])

    first = resolve_selection_artifact(
        root,
        scope="datastore",
        kind="cell_selection",
        values=values,
        row_ids=row_ids,
        operation="manual_selection",
        parameters={},
        inputs={},
        source_column="I",
    )
    renamed = resolve_selection_artifact(
        root,
        scope="datastore",
        kind="cell_selection",
        values=values.copy(),
        row_ids=row_ids.copy(),
        operation="manual_selection",
        parameters={},
        inputs={},
        source_column="renamed_mask",
    )

    assert renamed == first
    group = root[artifact_path(first)]
    np.testing.assert_array_equal(group["values"][:], values)
    status = inspect_artifact(root, first)
    assert status.complete
    assert status.execution_options == {"source_column": "I"}


def test_changed_or_invalidated_selection_creates_another_random_artifact() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    values = np.array([True, False, True, False])
    changed = np.array([True, True, False, False])
    row_ids = np.arange(4).astype(str)

    first = resolve_selection_artifact(
        root,
        scope="assay",
        assay="RNA",
        kind="feature_selection",
        values=values,
        row_ids=row_ids,
        operation="select_hvgs",
        parameters={"top_n": 2},
        inputs={},
        source_column="hvgs",
    )
    changed_ref = resolve_selection_artifact(
        root,
        scope="assay",
        assay="RNA",
        kind="feature_selection",
        values=changed,
        row_ids=row_ids,
        operation="select_hvgs",
        parameters={"top_n": 2},
        inputs={},
        source_column="hvgs",
    )
    invalidated = resolve_selection_artifact(
        root,
        scope="assay",
        assay="RNA",
        kind="feature_selection",
        values=values,
        row_ids=row_ids,
        operation="select_hvgs",
        parameters={"top_n": 2},
        inputs={},
        source_column="hvgs",
        invalidate_cache=True,
    )

    assert (
        len({first.artifact_id, changed_ref.artifact_id, invalidated.artifact_id}) == 3
    )


def test_integer_selection_payload_is_not_reused() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    values = np.array([True, False, True])
    row_ids = np.array(["a", "b", "c"])
    first = resolve_selection_artifact(
        root,
        scope="datastore",
        kind="cell_selection",
        values=values,
        row_ids=row_ids,
        operation="manual_selection",
        parameters={},
        inputs={},
        source_column="I",
    )
    group = root[artifact_path(first)]
    del group["values"]
    create_metadata_column(
        group,
        "values",
        data=values.astype(np.int8),
        dtype=np.int8,
    )

    replacement = resolve_selection_artifact(
        root,
        scope="datastore",
        kind="cell_selection",
        values=values,
        row_ids=row_ids,
        operation="manual_selection",
        parameters={},
        inputs={},
        source_column="I",
    )

    assert replacement != first
    assert root[artifact_path(replacement)]["values"].dtype == np.dtype(bool)


def test_generated_selection_identity_and_reuse_include_output_values() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    values = np.array([True, False, True])
    row_ids = np.array(["a", "b", "c"])
    first, stored = resolve_generated_selection_artifact(
        root,
        scope="assay",
        assay="ATAC",
        kind="feature_selection",
        values=values,
        row_ids=row_ids,
        operation="select_prevalent_peaks",
        parameters={"top_n": 2},
        inputs={"feature_selection": {"artifact_id": "input"}},
        source_column="prevalent_peaks",
    )
    reused, reused_values = resolve_generated_selection_artifact(
        root,
        scope="assay",
        assay="ATAC",
        kind="feature_selection",
        values=values.copy(),
        row_ids=row_ids,
        operation="select_prevalent_peaks",
        parameters={"top_n": 2},
        inputs={"feature_selection": {"artifact_id": "input"}},
        source_column="renamed",
    )
    root[artifact_path(first)]["values"][0] = False
    replacement, replacement_values = resolve_generated_selection_artifact(
        root,
        scope="assay",
        assay="ATAC",
        kind="feature_selection",
        values=values.copy(),
        row_ids=row_ids,
        operation="select_prevalent_peaks",
        parameters={"top_n": 2},
        inputs={"feature_selection": {"artifact_id": "input"}},
        source_column="prevalent_peaks",
    )
    changed, changed_values = resolve_generated_selection_artifact(
        root,
        scope="assay",
        assay="ATAC",
        kind="feature_selection",
        values=~values,
        row_ids=row_ids,
        operation="select_prevalent_peaks",
        parameters={"top_n": 2},
        inputs={"feature_selection": {"artifact_id": "input"}},
        source_column="prevalent_peaks",
    )

    assert reused == first
    assert replacement != first
    assert changed != first
    np.testing.assert_array_equal(stored, values)
    np.testing.assert_array_equal(reused_values, values)
    np.testing.assert_array_equal(replacement_values, values)
    np.testing.assert_array_equal(changed_values, ~values)
    assert isinstance(inspect_artifact(root, first).inputs["values_fingerprint"], str)


@pytest.mark.parametrize(
    "values",
    (
        np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        np.asarray([["a", "b"], ["c", "d"]]),
    ),
)
def test_metadata_snapshot_reuse_validates_flattened_payload(
    values: np.ndarray,
) -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    kwargs = {
        "values": values,
        "row_ids": np.asarray(["r1", "r2"]),
        "operation": "snapshot_fixture_metadata",
        "parameters": {"column": "fixture"},
        "inputs": {},
        "source_columns": ["fixture"],
    }
    first = resolve_metadata_snapshot(root, **kwargs)
    assert resolve_metadata_snapshot(root, **kwargs) == first

    artifact_group(root, first)["values"][0] = (
        "changed" if values.dtype.kind in {"O", "S", "U"} else -1
    )
    replacement = resolve_metadata_snapshot(root, **kwargs)
    assert replacement != first
    np.testing.assert_array_equal(
        artifact_group(root, replacement)["values"][:],
        values.reshape(-1).astype(str)
        if values.dtype.kind in {"O", "S", "U"}
        else values.reshape(-1),
    )


def test_filter_and_hvg_return_artifacts_without_metadata_aliases(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    cell_column = datastore.zw["cellData"]["I"]
    cell_values_before = np.asarray(cell_column[:], dtype=bool).copy()
    cell_attrs_before = dict(cell_column.attrs)
    feature_columns_before = set(datastore.RNA.feats.columns)
    cell_ref = datastore.filter_cells(
        attrs=["RNA_nCounts"],
        lows=[0],
        highs=[None],
    )
    cell_status = datastore.inspect_artifact(cell_ref)
    assert cell_status.operation == "filter_cells"
    assert cell_status.parameters["attrs"] == ["RNA_nCounts"]
    np.testing.assert_array_equal(cell_column[:], cell_values_before)
    assert dict(cell_column.attrs) == cell_attrs_before

    feature_ref = datastore.select_hvgs(
        cell_ref,
        from_assay="RNA",
        top_n=50,
        show_plot=False,
    )
    assert set(datastore.RNA.feats.columns) == feature_columns_before
    feature_status = datastore.inspect_artifact(feature_ref)
    assert feature_status.operation == "select_hvgs"
    assert feature_status.parameters["top_n"] == 50
    summary_ref = ArtifactRef.from_dict(feature_status.inputs["feature_summary"])
    summary_status = datastore.inspect_artifact(summary_ref)
    assert (
        summary_status.inputs["cell_selection"]["artifact_id"] == cell_ref.artifact_id
    )

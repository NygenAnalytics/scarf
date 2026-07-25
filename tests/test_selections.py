import numpy as np
import zarr
from zarr.storage import MemoryStore

from scarf.storage.artifacts import ArtifactRef, artifact_path, inspect_artifact
from scarf.storage.arrays import create_metadata_column
from scarf.storage.selections import (
    resolve_generated_selection_artifact,
    resolve_selection_artifact,
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
        source_column="I__hvgs",
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
        source_column="I__hvgs",
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
        source_column="I__hvgs",
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


def test_generated_selection_identity_excludes_output_values() -> None:
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
        operation="mark_prevalent_peaks",
        parameters={"top_n": 2},
        inputs={"feature_selection": {"artifact_id": "input"}},
        source_column="I__prevalent_peaks",
    )
    reused, reused_values = resolve_generated_selection_artifact(
        root,
        scope="assay",
        assay="ATAC",
        kind="feature_selection",
        values=~values,
        row_ids=row_ids,
        operation="mark_prevalent_peaks",
        parameters={"top_n": 2},
        inputs={"feature_selection": {"artifact_id": "input"}},
        source_column="renamed",
    )

    assert reused == first
    np.testing.assert_array_equal(stored, values)
    np.testing.assert_array_equal(reused_values, values)
    assert "values_fingerprint" not in inspect_artifact(root, first).inputs


def test_filter_and_hvg_columns_link_to_provenance_artifacts(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    datastore.filter_cells(
        attrs=["RNA_nCounts"],
        lows=[0],
        highs=[None],
        reset_previous=True,
    )
    cell_column = datastore.zw["cellData"]["I"]
    cell_ref = ArtifactRef.from_dict(cell_column.attrs["source_artifact"])
    cell_status = datastore.inspect_artifact(cell_ref)
    assert cell_status.operation == "filter_cells"
    assert cell_status.parameters["attrs"] == ["RNA_nCounts"]

    datastore.mark_hvgs(
        from_assay="RNA",
        cell_key="I",
        top_n=50,
        hvg_key_name="selection_test_hvgs",
        show_plot=False,
    )
    feature_column = datastore.get_assay("RNA").z["featureData"][
        "I__selection_test_hvgs"
    ]
    feature_ref = ArtifactRef.from_dict(feature_column.attrs["source_artifact"])
    feature_status = datastore.inspect_artifact(feature_ref)
    assert feature_status.operation == "mark_hvgs"
    assert feature_status.parameters["top_n"] == 50
    assert (
        feature_status.inputs["cell_selection"]["artifact_id"] == cell_ref.artifact_id
    )

from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

import scarf.datastore._operations.mapping as mapping_operations
from scarf.datastore.datastore import DataStore, mount_datastore
from scarf.mapping.features import AlignedFeatureStream
from scarf.mapping.projection import load_projection, resolve_projection
from scarf.storage.artifacts import (
    ArtifactRef,
    ExternalArtifactRef,
    artifact_group,
    list_artifacts,
)


def _snapshot_store(path: str) -> dict[str, bytes]:
    root = Path(path)
    return {
        str(file.relative_to(root)): file.read_bytes()
        for file in root.rglob("*")
        if file.is_file()
    }


def _plain_reference(datastore):
    state = datastore.get_assay_state("RNA")
    assert state is not None
    assert state.neighbors is not None
    return datastore.build_mapping_reference(state.neighbors)


def _symphony_reference(datastore):
    state = datastore.get_assay_state("RNA")
    assert state is not None
    assert state.reduction is not None
    datastore.cells.insert(
        "mapping_batch",
        np.where(np.arange(datastore.cells.N) % 2, "a", "b"),
        overwrite=True,
    )
    correction = datastore.run_harmony(
        ["mapping_batch"],
        state.reduction,
        harmony_params={"nclust": 5},
        update_state=False,
    )
    ann_index = datastore.build_ann_index(correction, update_state=False)
    neighbors = datastore.query_neighbors(
        ann_index,
        coordinates=correction,
        k=3,
        update_state=False,
    )
    return datastore.build_mapping_reference(neighbors)


def _copied_query(datastore, path: Path, *, zarr_mode: str = "r+") -> DataStore:
    shutil.copytree(datastore.zarr_loc, path)
    return DataStore(
        str(path),
        default_assay="RNA",
        zarr_mode=zarr_mode,
    )


def _changed_files(
    before: dict[str, bytes],
    after: dict[str, bytes],
) -> set[str]:
    return {
        path
        for path in before.keys() | after.keys()
        if before.get(path) != after.get(path)
    }


def test_plain_mapping_is_query_owned_and_reuses_exact_projection(
    analyzed_datastore_ephemeral,
    tmp_path,
    monkeypatch,
):
    reference_store = analyzed_datastore_ephemeral
    reference = _plain_reference(reference_store)
    query = _copied_query(reference_store, tmp_path / "query.zarr")
    reference_before = _snapshot_store(reference_store.zarr_loc)
    query_before = _snapshot_store(query.zarr_loc)
    query_sf = query.RNA.sf
    query_scalar = query.RNA.scalar
    query_attrs = dict(query.RNA.attrs)
    query_ncounts = np.array(query.cells.fetch_all("RNA_nCounts"), copy=True)
    query_feature_columns = set(query.RNA.feats.columns)
    available_k = int(
        artifact_group(reference_store.zw, reference.neighbors)["indices"].shape[1]
    )
    project_pca = mapping_operations.project_pca

    def project_with_zero_row(values, model):
        projected = project_pca(values, model)
        projected[0] = 0
        return projected

    monkeypatch.setattr(mapping_operations, "project_pca", project_with_zero_row)
    warning_messages = []
    monkeypatch.setattr(
        "scarf.datastore._operations.mapping.logger.warning",
        warning_messages.append,
    )

    result = query.run_mapping(
        reference,
        "plain",
        save_k=available_k + 10,
    )

    assert _snapshot_store(reference_store.zarr_loc) == reference_before
    assert len(warning_messages) == 1
    assert "save_k" in warning_messages[0]
    assert result.reference is reference
    assert result.mapping_name == "plain"
    assert result.correction_method == "none"
    assert result.indices is None
    assert result.distances is None
    assert result.uninformative is None
    assert set(result.diagnostics) == {
        "featureCoverage",
        "queryBatchCount",
        "algorithmVariant",
        "zeroNormCellCount",
        "queryScaledDispersion",
    }
    assert result.diagnostics["algorithmVariant"] == "scaled_pca"
    assert result.diagnostics["queryBatchCount"] == 1
    assert result.diagnostics["zeroNormCellCount"] > 0
    assert query.RNA.sf == query_sf
    assert query.RNA.scalar is query_scalar
    assert dict(query.RNA.attrs) == query_attrs
    np.testing.assert_array_equal(
        query.cells.fetch_all("RNA_nCounts"),
        query_ncounts,
    )
    assert set(query.RNA.feats.columns) == query_feature_columns

    status = query.inspect_artifact(result.ref)
    assert status.parameters == {
        "mapping_name": "plain",
        "save_k": available_k,
        "missing_feature_policy": "reference_mean",
        "correction_method": "none",
    }
    assert set(status.inputs or {}) == {
        "cell_selection",
        "feature_selection",
        "selected_expression_fingerprint",
        "query_batch_fingerprint",
        "mapping_reference",
    }
    assert (
        ExternalArtifactRef.from_dict((status.inputs or {})["mapping_reference"])
        == reference.external_ref
    )
    lineage = query.lineage(result.ref, references=reference)
    assert reference.external_ref in lineage.graph
    assert all(
        node["status"] is not None and node["status"].complete
        for _, node in lineage.graph.nodes(data=True)
    )
    cell_selection = ArtifactRef.from_dict((status.inputs or {})["cell_selection"])
    feature_selection = ArtifactRef.from_dict(
        (status.inputs or {})["feature_selection"]
    )
    np.testing.assert_array_equal(
        artifact_group(query.zw, cell_selection)["values"][:],
        query.cells.fetch_all("I"),
    )
    query_feature_ids = np.asarray(query.RNA.feats.fetch_all("ids")).astype(str)
    np.testing.assert_array_equal(
        artifact_group(query.zw, feature_selection)["values"][:],
        np.isin(query_feature_ids, reference.feature_ids),
    )
    feature_status = query.inspect_artifact(feature_selection)
    assert isinstance(
        (feature_status.inputs or {}).get("alignment_map_hash"),
        str,
    )
    group = artifact_group(query.zw, result.ref)
    assert set(group.array_keys()) == {"indices", "distances", "uninformative"}
    assert set(group.group_keys()) == set()
    assert set(group.attrs) == {
        "artifact_id",
        "kind",
        "provenance",
        "execution_options",
        "created_at_ns",
        "complete",
        "diagnostics",
    }
    loaded = load_projection(
        query.zw,
        result.ref,
        load_arrays=True,
        reference=reference,
    )
    assert loaded.indices is not None
    assert loaded.indices.shape == (result.n_cells, available_k)
    assert loaded.distances is not None
    assert loaded.distances.shape == loaded.indices.shape
    assert loaded.uninformative is not None
    assert (
        int(np.count_nonzero(loaded.uninformative))
        == result.diagnostics["zeroNormCellCount"]
    )

    changed = _changed_files(query_before, _snapshot_store(query.zarr_loc))
    assert not any("aligned" in path or "normed" in path for path in changed)

    reuse_before = _snapshot_store(query.zarr_loc)

    def fail_ann_load(*args, **kwargs):
        raise AssertionError("A reused mapping must not load or query the ANN")

    monkeypatch.setattr(
        "scarf.datastore._operations.mapping._load_reference_neighbor_query",
        fail_ann_load,
    )
    reused = query.run_mapping(reference, "plain", save_k=available_k + 10)
    assert reused.ref == result.ref
    assert reused.reference is reference
    assert reused.diagnostics == result.diagnostics
    assert _snapshot_store(query.zarr_loc) == reuse_before


def test_mapping_failure_leaves_projection_incomplete(
    analyzed_datastore_ephemeral,
    tmp_path,
    monkeypatch,
):
    reference_store = analyzed_datastore_ephemeral
    reference = _plain_reference(reference_store)
    query = _copied_query(reference_store, tmp_path / "query.zarr")
    before = set(
        list_artifacts(
            query.zw,
            scope="assay",
            assay="RNA",
            kind="projection",
        )
    )

    class FailingNeighborQuery:
        @staticmethod
        def query(_values):
            raise RuntimeError("injected ANN failure")

    monkeypatch.setattr(
        mapping_operations,
        "_load_reference_neighbor_query",
        lambda *_args, **_kwargs: FailingNeighborQuery(),
    )
    with pytest.raises(RuntimeError, match="injected ANN failure"):
        query.run_mapping(reference, "failure")

    created = (
        set(
            list_artifacts(
                query.zw,
                scope="assay",
                assay="RNA",
                kind="projection",
            )
        )
        - before
    )
    assert len(created) == 1
    failed = query.inspect_artifact(created.pop())
    assert failed.exists
    assert not failed.complete


def test_mapping_load_failure_after_finish_keeps_projection_complete(
    analyzed_datastore_ephemeral,
    tmp_path,
    monkeypatch,
):
    reference_store = analyzed_datastore_ephemeral
    reference = _plain_reference(reference_store)
    query = _copied_query(reference_store, tmp_path / "query-load-fail.zarr")
    before = set(
        list_artifacts(
            query.zw,
            scope="assay",
            assay="RNA",
            kind="projection",
        )
    )

    def fail_load(*_args, **_kwargs):
        raise RuntimeError("injected load_projection failure")

    monkeypatch.setattr(mapping_operations, "load_projection", fail_load)
    with pytest.raises(RuntimeError, match="injected load_projection failure"):
        query.run_mapping(reference, "load_failure")

    created = (
        set(
            list_artifacts(
                query.zw,
                scope="assay",
                assay="RNA",
                kind="projection",
            )
        )
        - before
    )
    assert len(created) == 1
    finished = created.pop()
    status = query.inspect_artifact(finished)
    assert status.exists
    assert status.complete

    monkeypatch.undo()
    loaded = query.get_mapping_result(finished, reference=reference)
    assert loaded.ref == finished
    reused = query.run_mapping(reference, "load_failure")
    assert reused.ref == finished


def test_mapping_guards_precede_query_writes(
    analyzed_datastore_ephemeral,
    tmp_path,
):
    reference_store = analyzed_datastore_ephemeral
    reference = _plain_reference(reference_store)
    reference_before = _snapshot_store(reference_store.zarr_loc)

    with pytest.raises(ValueError, match="same physical Zarr store"):
        reference_store.run_mapping(reference, "same_store")
    assert _snapshot_store(reference_store.zarr_loc) == reference_before

    reopened = DataStore(reference_store.zarr_loc, default_assay="RNA")
    with pytest.raises(ValueError, match="same physical Zarr store"):
        reopened.run_mapping(reference, "reopened_same_store")
    assert _snapshot_store(reference_store.zarr_loc) == reference_before

    read_only = _copied_query(
        reference_store,
        tmp_path / "read-only.zarr",
        zarr_mode="r",
    )
    read_only_before = _snapshot_store(read_only.zarr_loc)
    with pytest.raises(ValueError, match="read-write query datastore"):
        read_only.run_mapping(reference, "read_only")
    assert _snapshot_store(read_only.zarr_loc) == read_only_before

    writable = _copied_query(reference_store, tmp_path / "writable.zarr")
    writable_before = _snapshot_store(writable.zarr_loc)
    with pytest.raises(ValueError, match="only supported by Symphony"):
        writable.run_mapping(
            reference,
            "plain_batches",
            query_batches=pd.DataFrame({"batch": ["a"]}),
        )
    assert _snapshot_store(writable.zarr_loc) == writable_before

    with pytest.raises(TypeError, match="RNA query assays"):
        writable.run_mapping(reference, "non_rna", query_assay="assay2")
    assert _snapshot_store(writable.zarr_loc) == writable_before

    writable.cells.insert(
        "empty_mapping_selection",
        np.zeros(writable.cells.N, dtype=bool),
        overwrite=True,
    )
    empty_before = _snapshot_store(writable.zarr_loc)
    with pytest.raises(ValueError, match="at least one query cell"):
        writable.run_mapping(
            reference,
            "empty",
            cell_key="empty_mapping_selection",
        )
    assert _snapshot_store(writable.zarr_loc) == empty_before

    original_fingerprint = reference_store.RNA.attrs["dataset_fingerprint"]
    reference_store.RNA.attrs["dataset_fingerprint"] = "changed"
    fingerprint_before = _snapshot_store(writable.zarr_loc)
    try:
        with pytest.raises(ValueError, match="dataset fingerprint mismatch"):
            writable.run_mapping(reference, "stale_reference")
    finally:
        reference_store.RNA.attrs["dataset_fingerprint"] = original_fingerprint
    assert _snapshot_store(writable.zarr_loc) == fingerprint_before


def test_separately_mounted_query_can_map_its_source_reference(
    analyzed_datastore_ephemeral,
    tmp_path,
):
    reference_store = analyzed_datastore_ephemeral
    reference = _plain_reference(reference_store)
    reference_before = _snapshot_store(reference_store.zarr_loc)
    read_only_reference_store = DataStore(
        reference_store.zarr_loc,
        default_assay="RNA",
        zarr_mode="r",
    )
    read_only_reference = read_only_reference_store.get_mapping_reference(reference.ref)
    query = mount_datastore(
        reference_store.zarr_loc,
        at=str(tmp_path / "mounted-query.zarr"),
        default_assay="RNA",
    )

    result = query.run_mapping(read_only_reference, "mounted")

    assert result.reference is read_only_reference
    assert result.n_cells == len(query.cells.active_index("I"))
    assert _snapshot_store(reference_store.zarr_loc) == reference_before


def test_query_projection_reproduces_stored_reference_coordinates(
    analyzed_datastore_ephemeral,
):
    reference_store = analyzed_datastore_ephemeral
    reference = _plain_reference(reference_store)
    stored = np.asarray(
        artifact_group(reference_store.zw, reference.reduction)["data"][:],
        dtype=np.float64,
    )

    stream = AlignedFeatureStream(
        query_assay=reference_store.RNA,
        query_cell_indices=reference_store.cells.active_index(reference.cell_key),
        reference_feature_ids=reference.feature_ids,
        reference_normalized_means=reference.model.feature_means,
        reference_normalization_parameters=reference.normalization_parameters,
        missing_feature_policy="error",
        resources=reference_store.resources,
    )
    projected = np.vstack(
        [
            mapping_operations.project_pca(block.values, reference.model)
            for block in stream
        ]
    )

    assert stream.feature_coverage == 1.0
    assert projected.shape == stored.shape
    np.testing.assert_allclose(
        projected,
        stored,
        rtol=0,
        atol=1e-4 * float(np.abs(stored).max()),
    )


def test_self_mapping_recovers_the_reference_graph_and_labels(
    analyzed_datastore_ephemeral,
    tmp_path,
):
    reference_store = analyzed_datastore_ephemeral
    reference_store.cells.insert(
        "mapping_labels",
        np.array([f"c{index % 7}" for index in range(reference_store.cells.N)]),
        overwrite=True,
    )
    reference = _plain_reference(reference_store)
    graph = artifact_group(reference_store.zw, reference.neighbors)
    reference_indices = np.asarray(graph["indices"][:])
    reference_distances = np.asarray(graph["distances"][:], dtype=np.float64)
    available_k = int(reference_indices.shape[1])
    assert not (reference_indices[:, 0] == np.arange(len(reference_indices))).all()

    query = mount_datastore(
        reference_store.zarr_loc,
        at=str(tmp_path / "self-query.zarr"),
        default_assay="RNA",
    )
    result = query.run_mapping(reference, "self", save_k=available_k)
    loaded = query.get_mapping_result(result, reference=reference, load_arrays=True)
    indices = loaded.indices
    distances = loaded.distances
    assert indices is not None and distances is not None

    positions = np.arange(result.n_cells)
    assert result.n_cells == reference.selected_cell_count
    assert (indices[:, 0] == positions).mean() > 0.95
    assert float(np.median(distances[:, 0])) < 1e-4
    # The reference PCA is fitted on z-scored features, so a query that is the
    # reference has to disperse exactly like it. This anchors the diagnostic:
    # values well below 1 mean the query occupies a narrower region.
    assert result.diagnostics["queryScaledDispersion"] == pytest.approx(1.0, abs=0.02)

    agreement = indices[:, 1:] == reference_indices[:, : available_k - 1]
    assert agreement.mean() > 0.9
    np.testing.assert_allclose(
        distances[:, 1:][agreement],
        reference_distances[:, : available_k - 1][agreement],
        rtol=1e-4,
        atol=1e-4,
    )
    assert len(np.unique(indices)) > 0.9 * reference.selected_cell_count

    transferred = query.get_target_classes(
        loaded,
        reference_class_group="mapping_labels",
        reference=reference,
        threshold_fraction=0.6,
    ).to_numpy()
    known = np.asarray(query.cells.fetch("mapping_labels"))
    assert (transferred == known).mean() > 0.95

    evidence = query.get_target_label_evidence(
        loaded,
        reference_class_group="mapping_labels",
        reference=reference,
    )
    assert float(np.median(evidence["referenceDistancePercentile"])) == 0.0
    assert float(np.median(evidence["voteFraction"])) > 0.99
    assert not evidence["isUnknown"].mean() > 0.05


def test_symphony_mapping_validates_batches_and_persists_diagnostics(
    analyzed_datastore_ephemeral,
    tmp_path,
    monkeypatch,
):
    reference_store = analyzed_datastore_ephemeral
    reference = _symphony_reference(reference_store)
    query = _copied_query(reference_store, tmp_path / "query.zarr")
    n_cells = len(query.cells.active_index("I"))

    invalid_frames = (
        pd.DataFrame(index=np.arange(n_cells)),
        pd.DataFrame({"batch": ["a"] * (n_cells - 1)}),
        pd.DataFrame(
            np.column_stack((np.zeros(n_cells), np.ones(n_cells))),
            columns=["batch", "batch"],
        ),
        pd.DataFrame({"batch": [None] + ["a"] * (n_cells - 1)}),
    )
    for batches in invalid_frames:
        before = _snapshot_store(query.zarr_loc)
        with pytest.raises(ValueError):
            query.run_mapping(
                reference,
                "invalid_batches",
                query_batches=batches,
            )
        assert _snapshot_store(query.zarr_loc) == before

    batches = pd.DataFrame(
        {
            "donor": np.where(np.arange(n_cells) % 2, "a", "b"),
            "library": np.arange(n_cells) % 3,
        }
    )
    accumulated_rows = 0
    original_accumulate = mapping_operations.accumulate_sufficient_statistics

    def observe_accumulation(
        counts,
        sums,
        coordinates,
        assignments,
        batch_codes,
    ):
        nonlocal accumulated_rows
        assert not mapping_operations.zero_norm_rows(coordinates).any()
        accumulated_rows += len(coordinates)
        return original_accumulate(
            counts,
            sums,
            coordinates,
            assignments,
            batch_codes,
        )

    monkeypatch.setattr(
        mapping_operations,
        "accumulate_sufficient_statistics",
        observe_accumulation,
    )
    reference_before = _snapshot_store(reference_store.zarr_loc)
    result = query.run_mapping(
        reference,
        "symphony",
        query_batches=batches,
    )

    assert _snapshot_store(reference_store.zarr_loc) == reference_before
    assert result.reference is reference
    assert result.correction_method == "symphony"
    assert result.diagnostics == {
        "featureCoverage": 1.0,
        "queryBatchCount": 6,
        "algorithmVariant": "symphony",
        "zeroNormCellCount": result.diagnostics["zeroNormCellCount"],
        "queryScaledDispersion": result.diagnostics["queryScaledDispersion"],
    }
    assert accumulated_rows == n_cells - result.diagnostics["zeroNormCellCount"]
    loaded = load_projection(query.zw, result.ref, reference=reference)
    assert loaded.diagnostics == result.diagnostics
    reused = query.run_mapping(
        reference,
        "symphony",
        query_batches=batches.copy(),
    )
    assert reused.ref == result.ref
    assert reused.diagnostics == result.diagnostics

    omitted = query.run_mapping(reference, "symphony_omitted")
    assert omitted.diagnostics["queryBatchCount"] == 1


def test_projection_cache_tracks_counts_references_and_newest_pair(
    analyzed_datastore_ephemeral,
    tmp_path,
):
    reference_store = analyzed_datastore_ephemeral
    first_reference = _plain_reference(reference_store)
    state = reference_store.get_assay_state("RNA")
    assert state is not None
    assert state.neighbors is not None
    second_reference = reference_store.build_mapping_reference(
        state.neighbors,
        invalidate_cache=True,
    )
    query = _copied_query(reference_store, tmp_path / "query.zarr")

    first = query.run_mapping(first_reference, "shared")
    second = query.run_mapping(second_reference, "shared")
    assert first.ref != second.ref
    assert (
        resolve_projection(
            query.zw,
            query_assay="RNA",
            mapping_name="shared",
            mapping_reference=first_reference.external_ref,
        )
        == first.ref
    )
    assert (
        resolve_projection(
            query.zw,
            query_assay="RNA",
            mapping_name="shared",
            mapping_reference=second_reference.external_ref,
        )
        == second.ref
    )

    newest = query.run_mapping(
        first_reference,
        "shared",
        invalidate_cache=True,
    )
    assert newest.ref != first.ref
    assert (
        resolve_projection(
            query.zw,
            query_assay="RNA",
            mapping_name="shared",
            mapping_reference=first_reference.external_ref,
        )
        == newest.ref
    )

    backing = query.RNA.rawData._backing
    selected_row = int(query.cells.active_index("I")[0])
    original = int(backing[selected_row, 0])
    backing[selected_row, 0] = original + 1
    changed_counts = query.run_mapping(first_reference, "raw_edit")
    backing[selected_row, 0] = original + 2
    changed_again = query.run_mapping(first_reference, "raw_edit")
    assert changed_again.ref != changed_counts.ref

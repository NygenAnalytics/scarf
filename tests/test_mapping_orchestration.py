import gc
from pathlib import Path
import shutil
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import zarr

import scarf.datastore._operations.mapping as mapping_operations
from scarf.datastore.datastore import DataStore, mount_datastore
from scarf.graph.feature_projection import resolve_native_graph_inputs
from scarf.mapping.features import AlignedFeatureStream
from scarf.mapping.projection import load_projection
from scarf.storage.artifacts import (
    ArtifactRef,
    ExternalArtifactRef,
    artifact_group,
    list_artifacts,
)
from scarf.storage.selections import (
    read_stored_selection_indices,
    resolve_generated_selection_artifact,
)


def _snapshot_store(path: str) -> dict[str, bytes]:
    root = Path(path)
    return {
        str(file.relative_to(root)): file.read_bytes()
        for file in root.rglob("*")
        if file.is_file()
    }


def _fixture_graph(datastore) -> ArtifactRef:
    refs = datastore.list_artifacts(
        kind="connectivity_map",
        from_assay="RNA",
        scope="assay",
        complete_only=True,
    )
    assert len(refs) == 1
    return refs[0]


def _plain_reference(datastore):
    graph = _fixture_graph(datastore)
    raw_neighbors = datastore.inspect_artifact(graph).inputs["neighbors"]
    neighbors = ArtifactRef.from_dict(raw_neighbors)
    reference_ref = datastore.build_mapping_reference(neighbors)
    return datastore.get_mapping_reference(reference_ref)


def _symphony_reference(datastore):
    reduction = resolve_native_graph_inputs(
        datastore.zw,
        _fixture_graph(datastore),
    ).coordinates
    datastore.cells.insert(
        "mapping_batch",
        np.where(np.arange(datastore.cells.N) % 2, "a", "b"),
        overwrite=True,
    )
    correction = datastore.run_harmony(
        reduction,
        ["mapping_batch"],
        harmony_params={"nclust": 5},
    )
    ann_index = datastore.build_ann_index(correction)
    neighbors = datastore.query_neighbors(
        ann_index,
        coordinates=correction,
        k=3,
    )
    reference_ref = datastore.build_mapping_reference(neighbors)
    return datastore.get_mapping_reference(reference_ref)


def _copied_query(datastore, path: Path, *, zarr_mode: str = "r+") -> DataStore:
    shutil.copytree(datastore.zarr_loc, path)
    return DataStore(
        str(path),
        default_assay="RNA",
        zarr_mode=zarr_mode,
    )


def _query_selection_matching_reference(query, reference) -> ArtifactRef:
    values = np.asarray(
        artifact_group(reference.datastore.zw, reference.cell_selection)["values"][:],
        dtype=bool,
    )
    return resolve_generated_selection_artifact(
        query.zw,
        scope="datastore",
        kind="cell_selection",
        values=values,
        row_ids=np.asarray(query.cells.fetch_all("ids")),
        operation="select_mapping_query",
        parameters={},
        inputs={"mapping_reference": reference.external_ref},
        source_column="mapping_reference",
    )[0]


def _selected_rows(datastore, cell_selection: ArtifactRef) -> np.ndarray:
    return read_stored_selection_indices(
        datastore.zw,
        cell_selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )


def _overlap_observed(query, reference, cell_selection: ArtifactRef) -> np.ndarray:
    """Mark selected query cells with a raw count in any shared feature."""
    query_ids = np.asarray(query.RNA.feats.fetch_all("ids")).astype(str)
    columns = np.flatnonzero(np.isin(query_ids, reference.feature_ids))
    counts = query.RNA.rawData._backing.oindex[
        _selected_rows(query, cell_selection), columns
    ]
    return np.count_nonzero(counts, axis=1) > 0


def _panel_query(
    reference_store,
    reference,
    path: Path,
    *,
    n_overlap: int = 40,
    n_informative: int = 60,
) -> tuple[DataStore, np.ndarray]:
    """Write a targeted-panel query of real reference cells and empty cells.

    The panel measures part of the reference features plus features the
    reference lacks. Every seventh query cell has counts only in the features
    the reference lacks.
    """
    from scipy.sparse import csr_matrix

    from scarf.writers import SparseToZarr

    rng = np.random.default_rng(7)
    reference_ids = np.asarray(reference_store.RNA.feats.fetch_all("ids")).astype(str)
    columns = np.flatnonzero(np.isin(reference_ids, reference.feature_ids))[:n_overlap]
    rows = _selected_rows(reference_store, reference.cell_selection)[:n_informative]
    measured = np.asarray(
        reference_store.RNA.rawData._backing.oindex[rows, columns],
        dtype=np.uint32,
    )
    n_cells = n_informative + n_informative // 6
    empty = np.arange(n_cells) % 7 == 3
    counts = np.zeros((n_cells, n_overlap + 5), dtype=np.uint32)
    counts[~empty, :n_overlap] = measured[: int(np.count_nonzero(~empty))]
    counts[:, n_overlap:] = rng.integers(1, 6, size=(n_cells, 5))
    SparseToZarr(
        csr_matrix(counts),
        str(path),
        cell_ids=[f"query-{index}" for index in range(n_cells)],
        feature_ids=[
            *reference_ids[columns],
            *(f"panel-only-{index}" for index in range(5)),
        ],
        nthreads=1,
    ).dump()
    query = DataStore(
        str(path),
        default_assay="RNA",
        min_features_per_cell=0,
        nthreads=1,
    )
    return query, empty


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
    warning_messages = []
    monkeypatch.setattr(
        "scarf.datastore._operations.mapping.logger.warning",
        warning_messages.append,
    )

    projection_ref = query.run_mapping(
        reference,
        reference.cell_selection,
        save_k=available_k + 10,
    )
    assert isinstance(projection_ref, ArtifactRef)
    result = query.get_mapping_result(projection_ref, reference=reference)

    assert _snapshot_store(reference_store.zarr_loc) == reference_before
    assert len(warning_messages) == 1
    assert "save_k" in warning_messages[0]
    assert result.reference is reference
    assert result.correction_method == "none"
    assert result.indices is None
    assert result.distances is None
    assert result.uninformative is None
    assert set(result.diagnostics) == {
        "featureCoverage",
        "queryBatchCount",
        "algorithmVariant",
        "uninformativeCellCount",
        "queryScaledDispersion",
    }
    assert result.diagnostics["algorithmVariant"] == "scaled_pca"
    assert result.diagnostics["queryBatchCount"] == 1
    assert result.diagnostics["uninformativeCellCount"] == int(
        np.count_nonzero(~_overlap_observed(query, reference, reference.cell_selection))
    )
    assert query.RNA.sf == query_sf
    assert query.RNA.scalar is query_scalar
    assert dict(query.RNA.attrs) == query_attrs
    np.testing.assert_array_equal(
        query.cells.fetch_all("RNA_nCounts"),
        query_ncounts,
    )
    assert set(query.RNA.feats.columns) == query_feature_columns

    status = query.inspect_artifact(projection_ref)
    assert status.parameters == {
        "save_k": available_k,
        "missing_feature_policy": "reference_mean",
        "correction_method": "none",
    }
    assert set(status.inputs or {}) == {
        "cell_selection",
        "feature_selection",
        "query_dataset_fingerprint",
        "query_batch_fingerprint",
        "query_batch_count",
        "mapping_reference",
    }
    assert (
        ExternalArtifactRef.from_dict((status.inputs or {})["mapping_reference"])
        == reference.external_ref
    )
    lineage = query.lineage(projection_ref, references=reference)
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
        artifact_group(query.zw, reference.cell_selection)["values"][:],
    )
    query_feature_ids = np.asarray(query.RNA.feats.fetch_all("ids")).astype(str)
    np.testing.assert_array_equal(
        artifact_group(query.zw, feature_selection)["values"][:],
        np.isin(query_feature_ids, reference.feature_ids),
    )
    feature_status = query.inspect_artifact(feature_selection)
    assert feature_status.operation == "select_mapping_overlap"
    assert feature_status.parameters == {}
    assert set(feature_status.inputs or {}) == {
        "mapping_reference",
        "all_features",
    }
    assert (
        ExternalArtifactRef.from_dict(feature_status.inputs["mapping_reference"])
        == reference.external_ref
    )
    all_features = ArtifactRef.from_dict(feature_status.inputs["all_features"])
    all_features_status = query.inspect_artifact(all_features)
    assert all_features_status.operation == "create_all_features"
    assert all_features_status.inputs == {}
    group = artifact_group(query.zw, projection_ref)
    assert set(group.array_keys()) == {"indices", "distances", "uninformative"}
    assert set(group.group_keys()) == set()
    assert set(group.attrs) == {
        "artifact_id",
        "kind",
        "provenance",
        "execution_options",
        "created_at_ns",
        "scarf_version",
        "complete",
        "diagnostics",
        "payload_fingerprint",
    }
    loaded = load_projection(
        query.zw,
        projection_ref,
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
        == result.diagnostics["uninformativeCellCount"]
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
    reused = query.run_mapping(
        reference,
        reference.cell_selection,
        save_k=available_k + 10,
    )
    assert reused == projection_ref
    assert _snapshot_store(query.zarr_loc) == reuse_before


@pytest.mark.parametrize("method", ["pca", "symphony"])
def test_mapping_failure_leaves_projection_incomplete(
    analyzed_datastore_ephemeral,
    tmp_path,
    monkeypatch,
    method,
):
    reference_store = analyzed_datastore_ephemeral
    reference = (_plain_reference if method == "pca" else _symphony_reference)(
        reference_store
    )
    query = _copied_query(reference_store, tmp_path / "query.zarr")
    files = []
    original_temporary_file = mapping_operations.TemporaryFile

    def observe_file():
        file = original_temporary_file()
        files.append(file)
        return file

    monkeypatch.setattr(mapping_operations, "TemporaryFile", observe_file)
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
    # Without a collection to rescue it, a stream left suspended by the failure
    # would keep Zarr's process-wide I/O limit lowered for later work.
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        with zarr.config.set({"async.concurrency": 37}):
            with pytest.raises(RuntimeError, match="injected ANN failure"):
                query.run_mapping(reference, reference.cell_selection)
            assert zarr.config.get("async.concurrency") == 37
    finally:
        if gc_enabled:
            gc.enable()
    assert len(files) == (1 if method == "symphony" else 0)
    assert all(file.closed for file in files)

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


def test_mapping_rejects_reference_handles_forged_from_a_stored_reference(
    analyzed_datastore_ephemeral,
    tmp_path,
):
    reference_store = analyzed_datastore_ephemeral
    reference = _plain_reference(reference_store)
    query = _copied_query(reference_store, tmp_path / "query-forged-reference.zarr")
    projection = query.run_mapping(reference, reference.cell_selection)
    forged = replace(
        reference,
        model=object(),  # type: ignore[arg-type]
        cell_selection=ArtifactRef(
            scope="datastore",
            assay=None,
            kind="cell_selection",
            artifact_id="f" * 64,
        ),
    )

    with pytest.raises(ValueError, match="does not match its stored artifact"):
        query.run_mapping(forged, reference.cell_selection)
    with pytest.raises(ValueError, match="does not match its stored artifact"):
        query.get_mapping_result(projection, reference=forged)
    with pytest.raises(ValueError, match="does not match its stored artifact"):
        forged.fetch_cell_column("ids")
    with pytest.raises(ValueError, match="does not match its stored artifact"):
        query.lineage(projection, references=forged)


def test_mapping_rejects_feature_axis_changes_during_stream_setup(
    analyzed_datastore_ephemeral,
    tmp_path,
    monkeypatch,
):
    reference_store = analyzed_datastore_ephemeral
    reference = _plain_reference(reference_store)
    query = _copied_query(reference_store, tmp_path / "query-axis-change.zarr")
    before = set(
        query.list_artifacts(
            kind="projection",
            from_assay="RNA",
        )
    )
    live_ids = query.RNA.feats._get_array("ids")
    original_ids = np.asarray(live_ids[:]).copy()
    changed_ids = original_ids.copy()
    changed_ids[[0, 1]] = changed_ids[[1, 0]]
    original_init = AlignedFeatureStream.__init__

    def initialize_then_change_ids(stream, *args, **kwargs):
        original_init(stream, *args, **kwargs)
        live_ids[:] = changed_ids

    monkeypatch.setattr(
        AlignedFeatureStream,
        "__init__",
        initialize_then_change_ids,
    )
    try:
        with pytest.raises(ValueError, match="identities changed during mapping setup"):
            query.run_mapping(reference, reference.cell_selection)
    finally:
        live_ids[:] = original_ids

    assert (
        set(
            query.list_artifacts(
                kind="projection",
                from_assay="RNA",
            )
        )
        == before
    )


def test_mapping_producer_returns_ref_without_loading_projection(
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
    finished = query.run_mapping(reference, reference.cell_selection)

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
    assert created == {finished}
    status = query.inspect_artifact(finished)
    assert status.exists
    assert status.complete

    monkeypatch.undo()
    loaded = query.get_mapping_result(finished, reference=reference)
    assert loaded.ref == finished
    reused = query.run_mapping(reference, reference.cell_selection)
    assert reused == finished


def test_mapping_guards_precede_query_writes(
    analyzed_datastore_ephemeral,
    tmp_path,
):
    reference_store = analyzed_datastore_ephemeral
    reference = _plain_reference(reference_store)
    reference_before = _snapshot_store(reference_store.zarr_loc)

    with pytest.raises(ValueError, match="same physical Zarr store"):
        reference_store.run_mapping(reference, reference.cell_selection)
    assert _snapshot_store(reference_store.zarr_loc) == reference_before

    reopened = DataStore(reference_store.zarr_loc, default_assay="RNA")
    with pytest.raises(ValueError, match="same physical Zarr store"):
        reopened.run_mapping(reference, reference.cell_selection)
    assert _snapshot_store(reference_store.zarr_loc) == reference_before

    read_only = _copied_query(
        reference_store,
        tmp_path / "read-only.zarr",
        zarr_mode="r",
    )
    read_only_before = _snapshot_store(read_only.zarr_loc)
    with pytest.raises(ValueError, match="read-write query datastore"):
        read_only.run_mapping(reference, reference.cell_selection)
    assert _snapshot_store(read_only.zarr_loc) == read_only_before

    writable = _copied_query(reference_store, tmp_path / "writable.zarr")
    writable_before = _snapshot_store(writable.zarr_loc)
    with pytest.raises(ValueError, match="only supported by Symphony"):
        writable.run_mapping(
            reference,
            reference.cell_selection,
            query_batches=pd.DataFrame({"batch": ["a"]}),
        )
    assert _snapshot_store(writable.zarr_loc) == writable_before

    with pytest.raises(TypeError, match="RNA query assays"):
        writable.run_mapping(
            reference,
            reference.cell_selection,
            query_assay="assay2",
        )
    assert _snapshot_store(writable.zarr_loc) == writable_before

    writable.cells.insert(
        "empty_mapping_selection",
        np.zeros(writable.cells.N, dtype=bool),
        overwrite=True,
    )
    empty_selection = writable.snapshot_cell_selection("empty_mapping_selection")
    empty_before = _snapshot_store(writable.zarr_loc)
    with pytest.raises(ValueError, match="at least one query cell"):
        writable.run_mapping(reference, empty_selection)
    assert _snapshot_store(writable.zarr_loc) == empty_before

    had_stored_fingerprint = "dataset_fingerprint" in reference_store.RNA.attrs
    original_fingerprint = reference_store.RNA.attrs.get("dataset_fingerprint")
    reference_store.RNA.attrs["dataset_fingerprint"] = "changed"
    fingerprint_before = _snapshot_store(writable.zarr_loc)
    try:
        with pytest.raises(ValueError, match="dataset fingerprint mismatch"):
            writable.run_mapping(reference, reference.cell_selection)
    finally:
        if had_stored_fingerprint:
            reference_store.RNA.attrs["dataset_fingerprint"] = original_fingerprint
        else:
            del reference_store.RNA.attrs["dataset_fingerprint"]
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
    query_selection = _query_selection_matching_reference(
        query,
        read_only_reference,
    )

    result_ref = query.run_mapping(
        read_only_reference,
        query_selection,
    )
    result = query.get_mapping_result(
        result_ref,
        reference=read_only_reference,
    )

    assert result.reference is read_only_reference
    assert result.n_cells == read_only_reference.selected_cell_count
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
        query_cell_indices=read_stored_selection_indices(
            reference_store.zw,
            reference.cell_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        ),
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


@pytest.mark.parametrize("method", ["pca", "symphony"])
def test_subset_fitted_pca_center_survives_mapping_reference_reload(tmp_path, method):
    from scipy.sparse import csr_matrix

    from scarf.writers import SparseToZarr

    rng = np.random.default_rng(42)
    counts = rng.integers(10, 80, size=(30, 6), dtype=np.uint16)
    counts[:15, :3] += 300
    path = str(tmp_path / "reference.zarr")
    SparseToZarr(
        csr_matrix(counts),
        path,
        cell_ids=[f"c{i}" for i in range(len(counts))],
        feature_ids=[f"g{i}" for i in range(counts.shape[1])],
        nthreads=1,
    ).dump()
    datastore = DataStore(
        path, default_assay="RNA", min_features_per_cell=0, nthreads=1
    )
    cells = datastore.snapshot_cell_selection("I")
    datastore.cells.insert("pca_fit", np.arange(len(counts)) < 15)
    fit_cells = datastore.snapshot_cell_selection("pca_fit")
    features = datastore.select_all_features(from_assay="RNA")
    normalized = datastore.run_normalization(cells, features)
    reduction = datastore.run_pca(
        normalized, dims=3, pca_cell_selection=fit_cells, local_cache=False
    )
    reduction_group = artifact_group(datastore.zw, reduction)
    expected = np.asarray(reduction_group["data"][:])
    center = np.asarray(reduction_group["center"][:])
    assert np.linalg.norm(center) > 0.5
    coordinates = reduction
    if method == "symphony":
        datastore.cells.insert("batch", np.where(np.arange(len(counts)) % 2, "a", "b"))
        coordinates = datastore.run_harmony(
            reduction, ["batch"], harmony_params={"nclust": 2}
        )
    ann_index = datastore.build_ann_index(coordinates)
    neighbors = datastore.query_neighbors(ann_index, coordinates=coordinates, k=3)
    reference_ref = datastore.build_mapping_reference(neighbors)

    for reference_store in (
        datastore,
        DataStore(path, default_assay="RNA", nthreads=1, zarr_mode="r"),
    ):
        reference = reference_store.get_mapping_reference(reference_ref)
        np.testing.assert_array_equal(reference.model.center, center)
        stream = AlignedFeatureStream(
            query_assay=reference_store.RNA,
            query_cell_indices=np.arange(len(counts)),
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
        np.testing.assert_allclose(projected, expected, rtol=1e-6, atol=1e-6)

    if method == "pca":
        query = mount_datastore(
            path, at=str(tmp_path / "query.zarr"), default_assay="RNA", nthreads=1
        )
        query_selection = _query_selection_matching_reference(query, reference)
        result_ref = query.run_mapping(reference, query_selection, save_k=1)
        result = query.get_mapping_result(
            result_ref, reference=reference, load_arrays=True
        )
        np.testing.assert_array_equal(result.indices[:, 0], np.arange(len(counts)))
        np.testing.assert_allclose(result.distances[:, 0], 0, atol=1e-5)


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
    query_selection = _query_selection_matching_reference(query, reference)
    result = query.run_mapping(
        reference,
        query_selection,
        save_k=available_k,
    )
    loaded = query.get_mapping_result(result, reference=reference, load_arrays=True)
    indices = loaded.indices
    distances = loaded.distances
    assert indices is not None and distances is not None

    positions = np.arange(loaded.n_cells)
    assert loaded.n_cells == reference.selected_cell_count
    assert (indices[:, 0] == positions).mean() > 0.95
    assert float(np.median(distances[:, 0])) < 1e-4
    # The reference PCA is fitted on z-scored features, so a query that is the
    # reference has to disperse exactly like it. This anchors the diagnostic:
    # values well below 1 mean the query occupies a narrower region.
    assert loaded.diagnostics["queryScaledDispersion"] == pytest.approx(
        1.0,
        abs=0.02,
    )

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
        result,
        reference_class_group="mapping_labels",
        reference=reference,
        threshold_fraction=0.6,
    ).to_numpy()
    query_rows = read_stored_selection_indices(
        query.zw,
        query_selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )
    known = np.asarray(query.cells.fetch_all("mapping_labels"))[query_rows]
    assert (transferred == known).mean() > 0.95

    evidence = query.get_target_label_evidence(
        result,
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
    n_cells = reference.selected_cell_count

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
                reference.cell_selection,
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
    projected_rows = 0
    files = []
    original_project = mapping_operations.project_pca
    original_temporary_file = mapping_operations.TemporaryFile

    def observe_projection(values, model):
        nonlocal projected_rows
        projected_rows += len(values)
        return original_project(values, model)

    def observe_file():
        file = original_temporary_file()
        files.append(file)
        return file

    monkeypatch.setattr(mapping_operations, "project_pca", observe_projection)
    monkeypatch.setattr(mapping_operations, "TemporaryFile", observe_file)

    def observe_accumulation(
        counts,
        sums,
        coordinates,
        assignments,
        batch_codes,
    ):
        nonlocal accumulated_rows
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
    result_ref = query.run_mapping(
        reference,
        reference.cell_selection,
        query_batches=batches,
    )
    result = query.get_mapping_result(result_ref, reference=reference)

    assert _snapshot_store(reference_store.zarr_loc) == reference_before
    assert result.reference is reference
    assert result.correction_method == "symphony"
    assert (
        query.inspect_artifact(result_ref).parameters["query_batch_model"] == "additive"
    )
    assert result.diagnostics == {
        "featureCoverage": 1.0,
        "queryBatchCount": 6,
        "algorithmVariant": "symphony",
        "uninformativeCellCount": result.diagnostics["uninformativeCellCount"],
        "queryScaledDispersion": result.diagnostics["queryScaledDispersion"],
    }
    assert accumulated_rows == n_cells - result.diagnostics["uninformativeCellCount"]
    assert projected_rows == n_cells
    assert len(files) == 1 and files[0].closed
    loaded = load_projection(query.zw, result_ref, reference=reference)
    assert loaded.diagnostics == result.diagnostics
    reused = query.run_mapping(
        reference,
        reference.cell_selection,
        query_batches=batches.copy(),
    )
    assert reused == result_ref
    assert projected_rows == n_cells
    assert len(files) == 1

    omitted_ref = query.run_mapping(reference, reference.cell_selection)
    omitted = query.get_mapping_result(omitted_ref, reference=reference)
    assert omitted.diagnostics["queryBatchCount"] == 1
    assert len(files) == 2 and all(file.closed for file in files)


def test_projection_cache_tracks_exact_references(
    analyzed_datastore_ephemeral,
    tmp_path,
):
    reference_store = analyzed_datastore_ephemeral
    first_reference = _plain_reference(reference_store)
    second_reference = reference_store.get_mapping_reference(
        reference_store.build_mapping_reference(
            first_reference.neighbors,
            invalidate_cache=True,
        )
    )
    query = _copied_query(reference_store, tmp_path / "query.zarr")

    first = query.run_mapping(first_reference, first_reference.cell_selection)
    second = query.run_mapping(second_reference, second_reference.cell_selection)
    assert first != second
    assert load_projection(query.zw, first, reference=first_reference).ref == first
    assert load_projection(query.zw, second, reference=second_reference).ref == second

    newest = query.run_mapping(
        first_reference,
        first_reference.cell_selection,
        invalidate_cache=True,
    )
    assert newest != first
    assert load_projection(query.zw, newest, reference=first_reference).ref == (newest)
    assert load_projection(query.zw, first, reference=first_reference).ref == first


def test_zero_overlap_query_cells_are_uninformative(
    analyzed_datastore_ephemeral,
    tmp_path,
):
    reference_store = analyzed_datastore_ephemeral
    reference_store.cells.insert(
        "mapping_labels",
        np.array([f"c{index % 3}" for index in range(reference_store.cells.N)]),
        overwrite=True,
    )
    reference = _plain_reference(reference_store)
    query, empty = _panel_query(reference_store, reference, tmp_path / "panel.zarr")
    cells = query.snapshot_cell_selection("I")
    assert len(_selected_rows(query, cells)) == len(empty)
    expected = ~_overlap_observed(query, reference, cells)
    assert expected[empty].all()
    assert not expected.all()

    result_ref = query.run_mapping(reference, cells, save_k=5)
    result = query.get_mapping_result(result_ref, reference=reference, load_arrays=True)
    np.testing.assert_array_equal(result.uninformative, expected)
    assert result.diagnostics["uninformativeCellCount"] == int(expected.sum())
    assert result.diagnostics["featureCoverage"] < 1

    labels = query.get_target_classes(
        result_ref,
        "mapping_labels",
        reference=reference,
        threshold_fraction=0.0,
    ).to_numpy()
    assert (labels[expected] == "NA").all()
    evidence = query.get_target_label_evidence(
        result_ref,
        "mapping_labels",
        reference=reference,
        threshold_fraction=0.0,
    )
    assert evidence["isUnknown"].to_numpy()[expected].all()
    assert evidence["voteFraction"].isna().to_numpy()[expected].all()
    assert evidence["voteFraction"].notna().to_numpy()[~expected].all()
    assert (labels[~expected] != "NA").any()

    groups = np.where(expected, "empty", "measured")
    scores = dict(
        query.get_mapping_score(
            result_ref,
            target_groups=groups,
            reference=reference,
            log_transform=False,
        )
    )
    np.testing.assert_array_equal(scores["empty"], 0.0)
    assert scores["measured"].sum() > 0

    # Symphony statistics use only informative cells, so mapping the
    # informative cells alone reproduces their corrected neighbors.
    symphony_reference = _symphony_reference(reference_store)
    query.cells.insert("measured_cells", ~expected, overwrite=True)
    measured_cells = query.snapshot_cell_selection("measured_cells")
    with_empty = query.get_mapping_result(
        query.run_mapping(symphony_reference, cells, save_k=3),
        reference=symphony_reference,
        load_arrays=True,
    )
    measured_only = query.get_mapping_result(
        query.run_mapping(symphony_reference, measured_cells, save_k=3),
        reference=symphony_reference,
        load_arrays=True,
    )
    np.testing.assert_array_equal(with_empty.uninformative, expected)
    assert not measured_only.uninformative.any()
    np.testing.assert_array_equal(with_empty.indices[~expected], measured_only.indices)
    np.testing.assert_allclose(
        with_empty.distances[~expected],
        measured_only.distances,
        rtol=1e-9,
        atol=1e-12,
    )


def test_query_scaled_dispersion_ignores_missing_feature_fill(
    analyzed_datastore_ephemeral,
    tmp_path,
):
    reference_store = analyzed_datastore_ephemeral
    reference = _plain_reference(reference_store)
    query, _ = _panel_query(reference_store, reference, tmp_path / "panel.zarr")
    cells = query.snapshot_cell_selection("I")
    dispersion = {
        policy: query.get_mapping_result(
            query.run_mapping(reference, cells, missing_feature_policy=policy),
            reference=reference,
        ).diagnostics["queryScaledDispersion"]
        for policy in ("reference_mean", "zero")
    }
    assert dispersion["zero"] == pytest.approx(dispersion["reference_mean"], rel=1e-12)


def test_mapping_reuse_does_not_read_query_counts(
    analyzed_datastore_ephemeral,
    tmp_path,
    monkeypatch,
):
    reference_store = analyzed_datastore_ephemeral
    reference = _plain_reference(reference_store)
    query = _copied_query(reference_store, tmp_path / "query.zarr")
    projection = query.run_mapping(reference, reference.cell_selection)

    matrix = query.RNA.matrixGroup
    count_paths = tuple(f"{matrix[name].path}/" for name in matrix.array_keys())
    assert any(path.endswith("/counts/") for path in count_paths)
    matrix_store = matrix.store
    keys: list[str] = []
    store_type = type(matrix_store)
    original_get = store_type.get

    async def recording_get(store, key, prototype, byte_range=None):
        if store is matrix_store:
            keys.append(key)
        return await original_get(store, key, prototype, byte_range)

    monkeypatch.setattr(store_type, "get", recording_get)
    reused = query.run_mapping(reference, reference.cell_selection)

    assert reused == projection
    assert any(key.startswith("RNA/") for key in keys)
    count_chunks = [
        key
        for key in keys
        if key.startswith(count_paths) and not key.endswith("zarr.json")
    ]
    assert count_chunks == []

    # A fresh projection streams the counts, so the recorder does see them.
    keys.clear()
    query.run_mapping(reference, reference.cell_selection, invalidate_cache=True)
    assert any(
        key.startswith(count_paths) and not key.endswith("zarr.json") for key in keys
    )


def test_projection_inputs_record_query_dataset_fingerprint(
    analyzed_datastore_ephemeral,
    tmp_path,
):
    reference_store = analyzed_datastore_ephemeral
    reference = _plain_reference(reference_store)
    query = _copied_query(reference_store, tmp_path / "query.zarr")
    projection = query.run_mapping(reference, reference.cell_selection)

    inputs = query.inspect_artifact(projection).inputs or {}
    assert inputs["query_dataset_fingerprint"] == query._ensure_dataset_fingerprint(
        "RNA"
    )
    assert "selected_expression_fingerprint" not in inputs

    # Stand in for a rebuilt query dataset at the same location.
    query.RNA.z.attrs["dataset_fingerprint"] = "rebuilt-query-dataset"
    with pytest.raises(ValueError, match="prepared query assay 'RNA'.*run_mapping"):
        query.get_mapping_result(projection, reference=reference)
    remapped = query.run_mapping(reference, reference.cell_selection)
    assert remapped != projection
    remapped_inputs = query.inspect_artifact(remapped).inputs or {}
    assert remapped_inputs["query_dataset_fingerprint"] == "rebuilt-query-dataset"
    assert query.get_mapping_result(remapped, reference=reference).ref == remapped

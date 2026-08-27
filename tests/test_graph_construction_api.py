import numpy as np
import pandas as pd
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.datastore._operations import graph as graph_operations
from scarf.embeddings.harmony import fit_harmony
from scarf.graph.feature_projection import resolve_native_graph_inputs
from scarf.storage.artifacts import (
    ArtifactRef,
    artifact_group,
    artifact_path,
    list_artifacts,
)
from scarf.storage.budget import ResourceBudget
from scarf.storage.errors import ArtifactResolutionError
from scarf.utils import logger
from tests import full_path

pytestmark = pytest.mark.slow

_RELEASED_KNN_FEATURE_INDICES = (
    57,
    1363,
    2059,
    2060,
    2061,
    2176,
    2279,
    2344,
    2663,
    3545,
    4334,
    4377,
    4667,
    5639,
    7096,
    7880,
    8072,
    8357,
    8446,
    8473,
    8510,
    9685,
    9828,
    9831,
    10108,
    10153,
    10398,
    10557,
    11134,
    11280,
    11381,
    11689,
    13034,
    13347,
    13430,
    14081,
    14254,
    15065,
    16512,
    17128,
    17465,
    17834,
    18148,
    18216,
    18447,
    18735,
    18927,
    18970,
    19037,
    19793,
    19928,
    19989,
    20349,
    20837,
    21083,
    21106,
    21202,
    21209,
    21227,
    22897,
    23170,
    23856,
    24023,
    24024,
    24440,
    24559,
    24791,
    24990,
    24992,
    24994,
    24995,
    26633,
    26651,
    27766,
    28904,
    28917,
    29157,
    29440,
    29478,
    29726,
    30726,
    31204,
    31413,
    31655,
    32134,
    32546,
    32925,
    33201,
    33524,
    34021,
    34397,
    34657,
    34659,
    34661,
    35044,
    35464,
    35985,
    36085,
    36233,
    36424,
)


def test_streaming_lsi_block_rows_respect_memory_budget() -> None:
    group = zarr.open_group(store=MemoryStore(), mode="w")
    array = group.create_array(
        "normalized",
        shape=(100, 20),
        chunks=(10, 20),
        dtype=np.float32,
    )

    constrained = graph_operations._streaming_lsi_block_rows(
        array,
        ResourceBudget(memoryBytes=5_000, workers=1),
        n_components=3,
        n_oversamples=2,
    )
    roomy = graph_operations._streaming_lsi_block_rows(
        array,
        ResourceBudget(memoryBytes=1_000_000, workers=1),
        n_components=3,
        n_oversamples=2,
    )

    assert 1 <= constrained < roomy
    assert roomy == array.shape[0]
    with pytest.raises(MemoryError, match="Streaming LSI needs about"):
        graph_operations._streaming_lsi_block_rows(
            array,
            ResourceBudget(memoryBytes=1_000, workers=1),
            n_components=3,
            n_oversamples=2,
        )


def _prepare_graph_features(datastore) -> tuple[ArtifactRef, ArtifactRef]:
    cell_selection = datastore.auto_filter_cells()
    features = datastore.select_hvgs(
        cell_selection,
        from_assay="RNA",
        top_n=100,
        show_plot=False,
    )
    return cell_selection, features


def _single_artifact(datastore, kind: str) -> ArtifactRef:
    refs = list_artifacts(
        datastore.zw,
        scope="assay",
        assay="RNA",
        kind=kind,
    )
    assert len(refs) == 1
    return refs[0]


def _selection_mask(datastore, selection: ArtifactRef) -> np.ndarray:
    return np.asarray(
        artifact_group(datastore.zw, selection)["values"][:],
        dtype=bool,
    )


def test_graph_construction_methods_chain_explicit_refs_and_persist_artifacts(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    cell_selection, features = _prepare_graph_features(datastore)

    normalized = datastore.run_normalization(cell_selection, features)
    pca = datastore.run_pca(normalized, dims=5, batch_size=100)
    ann = datastore.build_ann_index(pca, batch_size=100)
    neighbors = datastore.query_neighbors(ann, k=3, batch_size=100)
    graph = datastore.build_connectivity_map(neighbors)

    assert all(
        isinstance(ref, ArtifactRef) for ref in (normalized, pca, ann, neighbors, graph)
    )
    normalized_group = datastore.zw[artifact_path(normalized)]
    assert normalized_group["feature_sum"].dtype == np.dtype(np.float64)
    assert normalized_group["feature_squared_sum"].dtype == np.dtype(np.float64)
    normalized_values = normalized_group["data"][:]
    np.testing.assert_allclose(
        normalized_group["feature_sum"][:],
        normalized_values.sum(axis=0, dtype=np.float64),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        normalized_group["feature_squared_sum"][:],
        np.square(normalized_values, dtype=np.float64).sum(axis=0),
        rtol=1e-6,
    )
    reduction_group = datastore.zw[artifact_path(pca)]
    assert reduction_group["loadings"].dtype == np.dtype(np.float64)
    assert reduction_group["data"].dtype == np.dtype(np.float32)
    assert reduction_group["data"].shape == (
        int(np.count_nonzero(_selection_mask(datastore, cell_selection))),
        5,
    )
    stored_scores = reduction_group["data"][:]
    _, legacy_projection = datastore._load_reduction_stream(
        pca,
        batch_size=100,
    )
    legacy_scores = np.concatenate(
        tuple(legacy_projection.iter_coordinate_blocks("")),
        axis=0,
    )
    np.testing.assert_allclose(
        stored_scores,
        legacy_scores,
        rtol=2e-5,
        atol=2e-6,
    )
    reduction_inputs = datastore.inspect_artifact(pca).inputs
    assert reduction_inputs is not None
    scaling = ArtifactRef.from_dict(reduction_inputs["feature_scaling"])
    scaling_group = datastore.load_artifact(scaling)
    assert scaling_group["mean"].dtype == np.dtype(np.float64)
    assert scaling_group["scale"].dtype == np.dtype(np.float64)
    neighbors_group = datastore.zw[artifact_path(neighbors)]
    assert neighbors_group["indices"].dtype == np.dtype(np.uint32)
    assert neighbors_group["distances"].dtype == np.dtype(np.float32)
    squared_distances = np.square(
        stored_scores[:, np.newaxis, :] - stored_scores[np.newaxis, :, :],
        dtype=np.float64,
    ).sum(axis=2)
    np.fill_diagonal(squared_distances, np.inf)
    exact_neighbors = np.argpartition(squared_distances, kth=2, axis=1)[:, :3]
    approximate_neighbors = neighbors_group["indices"][:]
    expected_neighbor_distances = np.sqrt(
        squared_distances[
            np.arange(len(stored_scores))[:, np.newaxis],
            approximate_neighbors,
        ]
    )
    np.testing.assert_allclose(
        neighbors_group["distances"][:],
        expected_neighbor_distances,
        rtol=2e-5,
        atol=2e-6,
    )
    recall = np.mean(
        [
            len(set(exact) & set(approximate)) / 3
            for exact, approximate in zip(
                exact_neighbors,
                approximate_neighbors,
                strict=True,
            )
        ]
    )
    assert recall >= 0.95
    graph_group = datastore.zw[artifact_path(graph)]
    assert graph_group["edges"].dtype == np.dtype(np.uint32)
    assert graph_group["weights"].dtype == np.dtype(np.float32)
    lineage = resolve_native_graph_inputs(datastore.zw, graph)
    assert lineage.normalized == normalized
    assert lineage.coordinates == pca
    assert lineage.ann_index == ann
    assert lineage.neighbors == neighbors
    assert lineage.cell_selection == cell_selection
    loaded = datastore.load_graph(graph)
    assert loaded.shape[0] == int(
        np.count_nonzero(_selection_mask(datastore, cell_selection))
    )
    assert np.isfinite(loaded.data).all()


def test_ann_index_logs_rebuild_and_reuse_accurately(datastore_ephemeral) -> None:
    datastore = datastore_ephemeral
    cell_selection, features = _prepare_graph_features(datastore)
    normalized = datastore.run_normalization(cell_selection, features)
    reduction = datastore.run_pca(normalized, dims=3)
    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]),
        level="INFO",
    )
    try:
        first = datastore.build_ann_index(reduction)
        second = datastore.build_ann_index(reduction)
    finally:
        logger.remove(sink)

    assert first == second
    assert any(message.startswith("Stored ANN index") for message in messages)
    assert any(message.startswith("Reused ANN index") for message in messages)
    assert all("Loaded existing ANN stream" not in message for message in messages)


@pytest.mark.parametrize("dims", [0, -1, 1.5, True])
def test_reduction_rejects_invalid_dimensions(datastore_ephemeral, dims) -> None:
    datastore = datastore_ephemeral
    cell_selection, features = _prepare_graph_features(datastore)
    normalized = datastore.run_normalization(cell_selection, features)

    with pytest.raises((TypeError, ValueError), match="dims"):
        datastore.run_pca(normalized, dims=dims)


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_reduction_rejects_invalid_batch_sizes(
    datastore_ephemeral,
    batch_size,
) -> None:
    datastore = datastore_ephemeral
    cell_selection, features = _prepare_graph_features(datastore)
    normalized = datastore.run_normalization(cell_selection, features)

    with pytest.raises((TypeError, ValueError), match="batch_size"):
        datastore.run_pca(
            normalized,
            dims=3,
            batch_size=batch_size,
        )


def test_row_block_expands_to_aligned_minimum(monkeypatch) -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    data = root.create_array(
        "data",
        shape=(20, 3),
        chunks=(2, 3),
        dtype=np.float32,
    )
    warnings: list[str] = []
    monkeypatch.setattr(graph_operations.logger, "warning", warnings.append)

    assert graph_operations._row_block(data, None, minimum=5) == 6
    assert graph_operations._row_block(data, 3, minimum=5) == 6
    assert any("below the required minimum of 5" in message for message in warnings)


def test_pca_rejects_empty_fit_selection(datastore_ephemeral) -> None:
    datastore = datastore_ephemeral
    cell_selection, features = _prepare_graph_features(datastore)
    normalized = datastore.run_normalization(cell_selection, features)
    datastore.cells.insert(
        "no_pca_cells",
        np.zeros(datastore.cells.N, dtype=bool),
        overwrite=True,
    )
    empty_selection = datastore.snapshot_cell_selection(cell_key="no_pca_cells")

    with pytest.raises(ValueError, match="dims \\+ 1 selected cells"):
        datastore.run_pca(
            normalized,
            dims=3,
            pca_cell_selection=empty_selection,
        )


def test_pca_rejects_fit_selection_outside_normalized_cells(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    wider_selection = datastore.auto_filter_cells()
    normalized_mask = _selection_mask(datastore, wider_selection)
    normalized_mask[np.flatnonzero(normalized_mask)[0]] = False
    datastore.cells.insert("normalized_cells", normalized_mask, overwrite=True)
    normalized_selection = datastore.snapshot_cell_selection(
        cell_key="normalized_cells"
    )
    features = datastore.select_hvgs(
        normalized_selection,
        from_assay="RNA",
        top_n=100,
        show_plot=False,
    )
    normalized = datastore.run_normalization(normalized_selection, features)

    with pytest.raises(
        ArtifactResolutionError,
        match="PCA cell selection must be a subset of normalized cells",
    ) as caught:
        datastore.run_pca(
            normalized,
            dims=3,
            pca_cell_selection=wider_selection,
        )

    assert caught.value.code == "row_mismatch"


def test_custom_reduction_rejects_invalid_loadings(datastore_ephemeral) -> None:
    datastore = datastore_ephemeral
    cell_selection, features = _prepare_graph_features(datastore)
    normalized = datastore.run_normalization(cell_selection, features)

    with pytest.raises(ValueError, match="two-dimensional"):
        datastore.run_custom_reduction(
            np.ones(4),
            normalized,
        )


def test_graph_construction_operations_reuse_persistent_local_cache(
    datastore_ephemeral,
    monkeypatch,
    tmp_path,
) -> None:
    datastore = datastore_ephemeral
    cell_selection, features = _prepare_graph_features(datastore)
    cache_path = tmp_path / "normalized_cache"
    monkeypatch.setattr(
        graph_operations,
        "is_remote_datastore",
        lambda *_args: True,
    )
    normalized = datastore.run_normalization(cell_selection, features)
    normalized_data = datastore.load_artifact(normalized)["data"]
    assert normalized_data.chunks[0] == normalized_data.shape[0]
    reduction = datastore.run_pca(
        normalized,
        dims=4,
        batch_size=100,
        local_cache=str(cache_path),
    )
    ann = datastore.build_ann_index(
        reduction,
        batch_size=100,
    )
    neighbors = datastore.query_neighbors(
        ann,
        k=3,
        batch_size=100,
    )

    staged = cache_path / normalized.artifact_id / "normed.zarr"
    assert staged.is_dir()
    assert datastore.inspect_artifact(normalized).execution_options == {
        "invalidate_cache": False
    }
    reduction_execution = datastore.inspect_artifact(reduction).execution_options or {}
    assert reduction_execution["local_cache"] == str(cache_path)
    for ref in (ann, neighbors):
        execution = datastore.inspect_artifact(ref).execution_options or {}
        assert "local_cache" not in execution


@pytest.mark.parametrize("local_cache", [True, "auto"])
def test_temporary_local_cache_is_removed_after_success(
    datastore_ephemeral,
    monkeypatch,
    tmp_path,
    local_cache,
) -> None:
    datastore = datastore_ephemeral
    cell_selection, features = _prepare_graph_features(datastore)
    normalized = datastore.run_normalization(cell_selection, features)
    cache_root = tmp_path / str(local_cache).lower()

    def make_cache_dir(*_args, **_kwargs):
        cache_root.mkdir()
        return str(cache_root)

    monkeypatch.setattr(
        graph_operations,
        "is_remote_datastore",
        lambda *_args: True,
    )
    monkeypatch.setattr(graph_operations.tempfile, "mkdtemp", make_cache_dir)

    with datastore._cache_normalized_artifact(
        normalized,
        local_cache,
        100,
    ):
        assert cache_root.is_dir()

    assert not cache_root.exists()


def test_temporary_local_cache_is_removed_after_failure(
    datastore_ephemeral,
    monkeypatch,
    tmp_path,
) -> None:
    datastore = datastore_ephemeral
    cell_selection, features = _prepare_graph_features(datastore)
    normalized = datastore.run_normalization(cell_selection, features)
    cache_root = tmp_path / "failed"

    def make_cache_dir(*_args, **_kwargs):
        cache_root.mkdir()
        return str(cache_root)

    monkeypatch.setattr(
        graph_operations,
        "is_remote_datastore",
        lambda *_args: True,
    )
    monkeypatch.setattr(graph_operations.tempfile, "mkdtemp", make_cache_dir)

    with pytest.raises(RuntimeError, match="stop after staging"):
        with datastore._cache_normalized_artifact(normalized, "auto", 100):
            raise RuntimeError("stop after staging")

    assert not cache_root.exists()


def test_embedding_initialization_persists_expected_payload(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    cell_selection, features = _prepare_graph_features(datastore)
    normalized = datastore.run_normalization(cell_selection, features)
    reduction = datastore.run_pca(normalized, dims=4)

    initialization = datastore.build_embedding_initialization(
        reduction,
        n_centroids=5,
        batch_size=2,
        kmeans_batch_size=5,
    )

    initialization_group = datastore.load_artifact(initialization)
    assert initialization_group["cluster_centers"].shape == (5, 4)
    assert initialization_group["cluster_labels"].dtype == np.uint32
    inputs = datastore.inspect_artifact(initialization).inputs
    assert inputs is not None
    assert ArtifactRef.from_dict(inputs["coordinates"]) == reduction


def test_connectivity_rejects_invalid_kernel_parameters(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    cell_selection, features = _prepare_graph_features(datastore)
    normalized = datastore.run_normalization(cell_selection, features)
    reduction = datastore.run_pca(normalized, dims=4)
    ann = datastore.build_ann_index(reduction)
    neighbors = datastore.query_neighbors(ann, k=3)

    for values in (
        {"local_connectivity": -1.0},
        {"local_connectivity": np.nan},
        {"local_connectivity": True},
        {"bandwidth": 0.0},
        {"bandwidth": -1.0},
        {"bandwidth": np.nan},
        {"bandwidth": True},
    ):
        with pytest.raises((TypeError, ValueError)):
            datastore.build_connectivity_map(
                neighbors,
                **values,
            )


def test_ann_index_rejects_invalid_runtime_parameters(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    cell_selection, features = _prepare_graph_features(datastore)
    normalized = datastore.run_normalization(cell_selection, features)
    reduction = datastore.run_pca(normalized, dims=4)

    for values, error, match in (
        ({"ann_metric": "ip"}, ValueError, "l2, cosine"),
        ({"ann_efc": 1.5}, TypeError, "positive integer"),
        ({"ann_ef": True}, TypeError, "positive integer"),
        ({"ann_m": 1}, ValueError, "at least two"),
        ({"rand_state": 0}, ValueError, "greater than zero"),
        ({"batch_size": 0}, ValueError, "greater than zero"),
    ):
        with pytest.raises(error, match=match):
            datastore.build_ann_index(
                reduction,
                **values,
            )


def test_neighbor_count_changes_only_neighbor_and_connectivity_artifacts(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    cell_selection, features = _prepare_graph_features(datastore)
    normalized = datastore.run_normalization(cell_selection, features)
    reduction = datastore.run_pca(normalized, dims=4)
    ann = datastore.build_ann_index(reduction)
    neighbors_three = datastore.query_neighbors(
        ann,
        k=3,
    )
    connectivity_three = datastore.build_connectivity_map(neighbors_three)
    neighbors_four = datastore.query_neighbors(
        ann,
        k=4,
    )
    connectivity_four = datastore.build_connectivity_map(neighbors_four)

    assert datastore.run_normalization(cell_selection, features) == normalized
    assert (
        datastore.run_pca(
            normalized,
            dims=4,
        )
        == reduction
    )
    assert datastore.build_ann_index(reduction) == ann
    assert neighbors_three != neighbors_four
    assert connectivity_three != connectivity_four


def test_cache_identity_distinguishes_parameters_from_execution_options(
    analyzed_datastore_ephemeral,
) -> None:
    datastore = analyzed_datastore_ephemeral
    graph = _single_artifact(datastore, "connectivity_map")
    lineage = resolve_native_graph_inputs(datastore.zw, graph)
    assert lineage.normalized is not None

    reused_reduction = datastore.run_pca(
        lineage.normalized,
        dims=11,
        local_cache="auto",
    )
    reused_ann = datastore.build_ann_index(lineage.coordinates)
    reused_neighbors = datastore.query_neighbors(
        lineage.ann_index,
        k=11,
    )
    changed_neighbors = datastore.query_neighbors(
        lineage.ann_index,
        k=3,
    )
    invalidated_reduction = datastore.run_pca(
        lineage.normalized,
        dims=11,
        local_cache=False,
        invalidate_cache=True,
    )

    assert reused_reduction == lineage.coordinates
    assert reused_ann == lineage.ann_index
    assert reused_neighbors == lineage.neighbors
    assert changed_neighbors != lineage.neighbors
    assert invalidated_reduction != lineage.coordinates
    assert datastore.inspect_artifact(changed_neighbors).parameters == {
        "k": 3,
        "distance_metric": "l2",
    }


@pytest.mark.slow
def test_seeded_graph_rebuild_is_deterministic(
    analyzed_datastore_ephemeral,
) -> None:
    datastore = analyzed_datastore_ephemeral
    graph = _single_artifact(datastore, "connectivity_map")
    reduction = resolve_native_graph_inputs(datastore.zw, graph).coordinates

    def rebuild():
        ann = datastore.build_ann_index(
            reduction,
            rand_state=4466,
            invalidate_cache=True,
        )
        neighbors = datastore.query_neighbors(
            ann,
            k=11,
            invalidate_cache=True,
        )
        connectivity = datastore.build_connectivity_map(
            neighbors,
            invalidate_cache=True,
        )
        group = datastore.load_artifact(connectivity)
        return connectivity, group["edges"][:], group["weights"][:]

    first_ref, first_edges, first_weights = rebuild()
    second_ref, second_edges, second_weights = rebuild()

    assert first_ref != second_ref
    np.testing.assert_array_equal(first_edges, second_edges)
    np.testing.assert_allclose(first_weights, second_weights, rtol=0, atol=0)


def test_explicit_graph_preserves_exact_feature_selection_ref(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    cell_selection, features = _prepare_graph_features(datastore)
    normalized = datastore.run_normalization(cell_selection, features)
    reduction = datastore.run_pca(normalized, dims=3)
    ann = datastore.build_ann_index(reduction)
    neighbors = datastore.query_neighbors(ann, k=3)
    connectivity = datastore.build_connectivity_map(neighbors)
    ancestry = resolve_native_graph_inputs(datastore.zw, connectivity)

    assert ancestry.feature_selection == features
    cell_status = datastore.inspect_artifact(ancestry.cell_selection)
    assert cell_status.operation == "auto_filter_cells"
    assert cell_status.execution_options == {"source_column": "artifact"}


def test_historical_neighbors_preserve_named_lineage_inputs(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    original_selection, features = _prepare_graph_features(datastore)
    mask = _selection_mask(datastore, original_selection)
    datastore.cells.insert("selection_a", mask, overwrite=True)
    datastore.cells.insert("selection_b", mask, overwrite=True)
    selection_b = datastore.snapshot_cell_selection(cell_key="selection_b")

    normalized = datastore.run_normalization(selection_b, features)
    reduction = datastore.run_pca(normalized, dims=4)
    ann = datastore.build_ann_index(reduction)
    neighbors = datastore.query_neighbors(ann, k=3)
    datastore.run_normalization(original_selection, features)

    ancestry = resolve_native_graph_inputs(datastore.zw, neighbors)
    assert ancestry.feature_selection == features
    assert ancestry.cell_selection == selection_b


def test_reduction_and_harmony_keep_immutable_selection_after_live_alias_change(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    cell_selection, features = _prepare_graph_features(datastore)
    normalized = datastore.run_normalization(cell_selection, features)
    reduction = datastore.run_pca(normalized, dims=4)
    datastore.cells.insert(
        "graph_batch",
        np.where(np.arange(datastore.cells.N) % 2, "a", "b"),
        overwrite=True,
    )
    mask = _selection_mask(datastore, cell_selection)
    selected = np.flatnonzero(mask)
    excluded = np.flatnonzero(~mask)
    assert len(selected) > 0 and len(excluded) > 0
    mask[selected[0]] = False
    mask[excluded[0]] = True
    datastore.cells.insert("I", mask, overwrite=True, force=True)

    new_reduction = datastore.run_pca(
        normalized,
        dims=5,
        invalidate_cache=True,
    )
    corrected = datastore.run_harmony(
        reduction,
        ["graph_batch"],
        harmony_params={"nclust": 5},
        invalidate_cache=True,
    )
    assert datastore.inspect_artifact(new_reduction).complete
    assert datastore.inspect_artifact(corrected).complete


def test_datastore_inspects_and_loads_artifact_read_only(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    cell_selection, features = _prepare_graph_features(datastore)
    ref = datastore.run_normalization(cell_selection, features)

    status = datastore.inspect_artifact(ref)
    group = datastore.load_artifact(ref)

    assert status.complete
    assert status.operation == "run_normalization"
    assert status.parameters is not None
    assert "data" in group
    with pytest.raises(ValueError):
        group.attrs["invalid"] = True


def test_graph_harmony_is_an_explicit_ann_coordinate_source(
    datastore_ephemeral,
    monkeypatch,
) -> None:
    datastore = datastore_ephemeral
    cell_selection, features = _prepare_graph_features(datastore)
    batches = np.where(np.arange(datastore.cells.N) % 2, "a", "b")
    datastore.cells.insert("graph_batch", batches, overwrite=True)
    normalized = datastore.run_normalization(cell_selection, features)
    pca = datastore.run_pca(normalized, dims=5)
    pca_scores = datastore.load_artifact(pca)["data"][:]
    active_batches = pd.DataFrame(
        {"graph_batch": batches[_selection_mask(datastore, cell_selection)]}
    ).astype(object)
    expected_correction = fit_harmony(
        np.asarray(pca_scores.T, dtype=np.float64),
        active_batches,
        nclust=5,
    )

    def fail_legacy_projection(*_args, **_kwargs):
        raise AssertionError("persisted coordinates should be used")

    monkeypatch.setattr(
        datastore,
        "_load_reduction_stream",
        fail_legacy_projection,
    )

    corrected = datastore.run_harmony(
        pca,
        ["graph_batch"],
        harmony_params={"nclust": 5},
    )
    ann = datastore.build_ann_index(corrected, batch_size=100)
    datastore.query_neighbors(ann, k=3)
    datastore.build_embedding_initialization(
        pca,
        n_centroids=5,
    )

    ann_inputs = datastore.inspect_artifact(ann).inputs
    assert ann_inputs is not None
    assert ann_inputs["coordinates"] == corrected.to_dict()
    correction_group = datastore.load_artifact(corrected)
    assert correction_group["data"].dtype == np.dtype(np.float32)
    np.testing.assert_allclose(
        correction_group["data"][:],
        expected_correction.corrected.T,
        rtol=2e-5,
        atol=2e-6,
    )
    assert "assignments" not in correction_group
    assert {
        "cluster_mass",
        "raw_centroids",
        "corrected_centroids",
    } <= set(correction_group.array_keys())


def test_lsi_and_custom_reduction_have_distinct_public_methods(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    cell_selection, features = _prepare_graph_features(datastore)
    normalized = datastore.run_normalization(cell_selection, features)
    lsi = datastore.run_lsi(
        normalized,
        dims=3,
        n_iter=1,
        n_oversamples=2,
    )
    materialized_lsi = datastore.run_lsi(
        normalized,
        dims=3,
        solver="materialized",
        n_iter=1,
        n_oversamples=2,
    )
    n_features = datastore.load_artifact(normalized)["data"].shape[1]
    loadings = np.eye(n_features, 2, dtype=np.float64)
    custom = datastore.run_custom_reduction(
        loadings,
        normalized,
    )

    lsi_status = datastore.inspect_artifact(lsi)
    assert lsi_status.operation == "run_lsi"
    assert lsi_status.parameters["solver"] == "streaming"
    assert lsi_status.parameters["n_iter"] == 1
    assert lsi_status.parameters["n_oversamples"] == 2
    assert datastore.inspect_artifact(materialized_lsi).parameters["solver"] == (
        "materialized"
    )
    assert materialized_lsi != lsi
    assert datastore.inspect_artifact(custom).operation == "run_custom_reduction"
    ann = datastore.build_ann_index(custom)
    neighbors = datastore.query_neighbors(ann, k=3)
    connectivity = datastore.build_connectivity_map(neighbors)
    ancestry = resolve_native_graph_inputs(datastore.zw, connectivity)
    assert ancestry.reduction == custom
    custom_status = datastore.inspect_artifact(custom)
    assert custom_status.operation == "run_custom_reduction"
    assert custom_status.parameters["dims"] == 2
    assert custom_status.parameters["feat_scaling"] is False


def test_graph_chain_matches_released_knn_golden(
    analyzed_datastore_ephemeral,
) -> None:
    datastore = analyzed_datastore_ephemeral
    cell_selection = datastore.auto_filter_cells()
    features = datastore.set_feature_selection(
        from_assay="RNA",
        feature_indexes=_RELEASED_KNN_FEATURE_INDICES,
        invalidate_cache=True,
    )
    normalized = datastore.run_normalization(
        cell_selection,
        features,
        invalidate_cache=True,
    )
    reduction = datastore.run_pca(
        normalized,
        dims=11,
        invalidate_cache=True,
    )
    ann = datastore.build_ann_index(
        reduction,
        invalidate_cache=True,
    )
    neighbors = datastore.query_neighbors(
        ann,
        coordinates=reduction,
        k=11,
        invalidate_cache=True,
    )
    group = datastore.load_artifact(neighbors)

    np.testing.assert_array_equal(
        group["indices"][:],
        np.load(full_path("knn_indices.npy")),
    )
    np.testing.assert_allclose(
        group["distances"][:],
        np.sqrt(np.load(full_path("knn_distances.npy"))),
        rtol=0,
        atol=1e-3,
    )


def test_connectivity_rebuild_requires_named_distance_metric(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    cell_selection, features = _prepare_graph_features(datastore)
    normalized = datastore.run_normalization(cell_selection, features)
    reduction = datastore.run_pca(normalized, dims=4)
    ann = datastore.build_ann_index(reduction)
    neighbors = datastore.query_neighbors(ann, k=3)

    neighbor_group = datastore.zw[artifact_path(neighbors)]
    provenance = dict(neighbor_group.attrs["provenance"])
    parameters = dict(provenance["parameters"])
    assert parameters["distance_metric"] == "l2"

    provenance["parameters"] = {"k": 3}
    neighbor_group.attrs["provenance"] = provenance
    with pytest.raises(ValueError, match="does not name the metric"):
        datastore.build_connectivity_map(
            neighbors,
            invalidate_cache=True,
        )

    provenance["parameters"] = {**parameters, "distance_metric": "cosine"}
    neighbor_group.attrs["provenance"] = provenance
    with pytest.raises(ValueError, match="does not match its ANN index input"):
        datastore.build_connectivity_map(
            neighbors,
            invalidate_cache=True,
        )

    provenance["parameters"] = parameters
    neighbor_group.attrs["provenance"] = provenance
    rebuilt = datastore.build_connectivity_map(
        neighbors,
        invalidate_cache=True,
    )
    assert datastore.inspect_artifact(rebuilt).complete


def test_corrupt_ann_bytes_are_not_reused(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    cell_selection, features = _prepare_graph_features(datastore)
    normalized = datastore.run_normalization(cell_selection, features)
    reduction = datastore.run_pca(normalized, dims=3)
    current = datastore.build_ann_index(reduction)
    for attribute, invalid_value in (
        ("metric", "cosine"),
        ("dimensions", 2),
        ("element_count", 1),
    ):
        ann_group = datastore.zw[artifact_path(current)]
        ann_group["ann_idx_bytes"].attrs[attribute] = invalid_value
        repaired = datastore.build_ann_index(reduction)
        assert repaired != current
        current = repaired

    ann_group = datastore.zw[artifact_path(current)]
    ann_group["ann_idx_bytes"][:] = 0
    repaired = datastore.build_ann_index(reduction)
    assert repaired != current
    assert datastore.build_ann_index(reduction) == repaired
    assert datastore.inspect_artifact(repaired).complete

    legacy_group = datastore.zw[artifact_path(repaired)]["ann_idx_bytes"]
    for attribute in (
        "ann_index_format_version",
        "metric",
        "dimensions",
        "element_count",
        "payload_sha256",
    ):
        del legacy_group.attrs[attribute]
    assert datastore.build_ann_index(reduction) == repaired

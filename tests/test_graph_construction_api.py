import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf import DataStore
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


@pytest.mark.parametrize(
    "center", [np.zeros(2), np.zeros(3, dtype=np.float32), np.full(3, np.nan)]
)
def test_read_pca_center_rejects_corrupt_values(center):
    group = zarr.open_group(store=MemoryStore(), mode="w")
    group.create_array("loadings", data=np.eye(3, 2))
    group.create_array("center", data=center)
    with pytest.raises(ValueError, match="PCA center.*Re-run run_pca"):
        graph_operations._read_pca_center(group)


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


def _batch_labels(n_cells: int) -> np.ndarray:
    """The ``graph_batch`` column the template stores: alternating ``b`` and ``a``."""
    return np.where(np.arange(n_cells) % 2, "a", "b")


def _open_store(zarr_loc: Path, **options) -> DataStore:
    return DataStore(str(zarr_loc), default_assay="RNA", **options)


@pytest.fixture(scope="module")
def graph_template(
    datastore_zarr_root,
    tmp_path_factory,
) -> tuple[Path, dict[str, ArtifactRef]]:
    """The 1K PBMC store with filtered cells, 100 HVGs, and their normalized data.

    Selecting HVGs dominates the cost of every graph test, so tests copy this
    store instead of repeating the selection. It also holds the alternating
    ``graph_batch`` column the Harmony tests correct for.
    """
    zarr_loc = tmp_path_factory.mktemp("graph_construction") / "store.zarr"
    shutil.copytree(datastore_zarr_root, zarr_loc)
    datastore = _open_store(zarr_loc)
    cell_selection = datastore.auto_filter_cells()
    features = datastore.select_hvgs(
        cell_selection,
        from_assay="RNA",
        top_n=100,
        show_plot=False,
    )
    datastore.cells.insert("graph_batch", _batch_labels(datastore.cells.N))
    normalized = datastore.run_normalization(cell_selection, features)
    return zarr_loc, {
        "cells": cell_selection,
        "features": features,
        "normalized": normalized,
    }


@pytest.fixture
def graph_store(graph_template, tmp_path) -> tuple[DataStore, dict[str, ArtifactRef]]:
    """A private writable copy of the template."""
    zarr_loc, refs = graph_template
    target = tmp_path / "store.zarr"
    shutil.copytree(zarr_loc, target)
    return _open_store(target), refs


@pytest.fixture(scope="module")
def read_only_store(graph_template) -> tuple[DataStore, dict[str, ArtifactRef]]:
    """The template opened read only, for calls that must fail before writing."""
    zarr_loc, refs = graph_template
    return _open_store(zarr_loc, zarr_mode="r"), refs


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


@pytest.mark.parametrize("feat_scaling", [False, True])
def test_pca_reopens_with_center_and_rebuilds_legacy_artifacts(
    tmp_path, feat_scaling: bool
) -> None:
    from scipy.sparse import csr_matrix

    from scarf.writers import SparseToZarr

    values = np.random.default_rng(31).integers(10, 100, size=(30, 6))
    values[15:, :3] += 200
    path = str(tmp_path / "pca.zarr")
    SparseToZarr(
        csr_matrix(values),
        path,
        cell_ids=[f"c{i}" for i in range(30)],
        feature_ids=[f"g{i}" for i in range(6)],
        nthreads=1,
    ).dump()
    datastore = DataStore(
        path, default_assay="RNA", min_features_per_cell=0, nthreads=1
    )
    cells = datastore.snapshot_cell_selection("I")
    datastore.cells.insert("pca_fit", np.arange(30) < 15)
    fit_cells = datastore.snapshot_cell_selection("pca_fit")
    features = datastore.select_all_features(from_assay="RNA")
    normalized = datastore.run_normalization(cells, features)
    arguments = dict(
        dims=2,
        pca_cell_selection=fit_cells,
        feat_scaling=feat_scaling,
        batch_size=10,
        local_cache=False,
    )
    reduction = datastore.run_pca(normalized, **arguments)
    group = artifact_group(datastore.zw, reduction)
    center = np.asarray(group["center"][:])
    assert center.shape == (6,)
    assert center.dtype == np.dtype(np.float64)
    assert np.linalg.norm(center) > 0.1
    expected = np.asarray(group["data"][:])
    np.testing.assert_allclose(expected[:15].mean(axis=0), 0, atol=1e-6)

    reopened = DataStore(path, default_assay="RNA", nthreads=1)
    stream, n_rows, n_dims = reopened._coordinate_source(reduction, batch_size=10)
    assert (n_rows, n_dims) == expected.shape
    np.testing.assert_allclose(
        np.vstack(list(stream.iter_coordinate_blocks(""))),
        expected,
        rtol=1e-6,
        atol=1e-6,
    )
    assert reopened.run_pca(normalized, **arguments) == reduction

    del group["center"]
    with pytest.raises(ValueError, match="PCA artifact has no fitted center"):
        reopened.build_ann_index(reduction)
    with pytest.raises(ValueError, match="PCA artifact has no fitted center"):
        reopened._coordinate_source(reduction, batch_size=10)
    rebuilt = reopened.run_pca(normalized, **arguments)
    assert rebuilt != reduction
    assert "center" not in group
    np.testing.assert_allclose(
        artifact_group(reopened.zw, rebuilt)["data"][:], expected
    )
    reopened.build_ann_index(rebuilt)


def test_graph_construction_methods_chain_explicit_refs_and_persist_artifacts(
    graph_store,
) -> None:
    datastore, refs = graph_store
    cell_selection, features = refs["cells"], refs["features"]

    normalized = datastore.run_normalization(cell_selection, features)
    pca = datastore.run_pca(normalized, dims=5, batch_size=100)
    ann = datastore.build_ann_index(pca, batch_size=100)
    neighbors = datastore.query_neighbors(ann, k=3, batch_size=100)
    graph = datastore.build_connectivity_map(neighbors)

    assert normalized == refs["normalized"]
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
    assert reduction_group["center"].dtype == np.dtype(np.float64)
    assert reduction_group["center"].shape == (normalized_values.shape[1],)
    assert reduction_group["data"].dtype == np.dtype(np.float32)
    assert reduction_group["data"].shape == (
        int(np.count_nonzero(_selection_mask(datastore, cell_selection))),
        5,
    )
    stored_scores = reduction_group["data"][:]
    reduction_inputs = datastore.inspect_artifact(pca).inputs
    assert reduction_inputs is not None
    scaling = ArtifactRef.from_dict(reduction_inputs["feature_scaling"])
    scaling_group = datastore.load_artifact(scaling)
    assert scaling_group["mean"].dtype == np.dtype(np.float64)
    assert scaling_group["scale"].dtype == np.dtype(np.float64)
    standardized = (normalized_values - scaling_group["mean"][:]) / scaling_group[
        "scale"
    ][:]
    expected_scores = (standardized - reduction_group["center"][:]) @ reduction_group[
        "loadings"
    ][:]
    np.testing.assert_allclose(
        stored_scores,
        expected_scores,
        rtol=2e-5,
        atol=2e-6,
    )
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


def test_ann_index_logs_rebuild_and_reuse_accurately(graph_store) -> None:
    datastore, refs = graph_store
    reduction = datastore.run_pca(refs["normalized"], dims=3)
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


@pytest.mark.parametrize(
    ("dims", "error", "message"),
    [
        (0, ValueError, "dims must be at least 1"),
        (-1, ValueError, "dims must be at least 1"),
        (1.5, TypeError, "dims must be an integer"),
        (True, TypeError, "dims must be an integer"),
    ],
)
def test_reduction_rejects_invalid_dimensions(
    read_only_store, dims, error, message
) -> None:
    datastore, refs = read_only_store

    with pytest.raises(error, match=message):
        datastore.run_pca(refs["normalized"], dims=dims)


@pytest.mark.parametrize(
    ("batch_size", "error", "message"),
    [
        (0, ValueError, "batch_size must be at least 1"),
        (-1, ValueError, "batch_size must be at least 1"),
        (1.5, TypeError, "batch_size must be an integer"),
        (True, TypeError, "batch_size must be an integer"),
    ],
)
def test_reduction_rejects_invalid_batch_sizes(
    read_only_store,
    batch_size,
    error,
    message,
) -> None:
    datastore, refs = read_only_store

    with pytest.raises(error, match=message):
        datastore.run_pca(
            refs["normalized"],
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


def test_pca_rejects_empty_fit_selection(graph_store) -> None:
    datastore, refs = graph_store
    datastore.cells.insert(
        "no_pca_cells",
        np.zeros(datastore.cells.N, dtype=bool),
        overwrite=True,
    )
    empty_selection = datastore.snapshot_cell_selection(cell_key="no_pca_cells")

    with pytest.raises(ValueError, match="dims \\+ 1 selected cells"):
        datastore.run_pca(
            refs["normalized"],
            dims=3,
            pca_cell_selection=empty_selection,
        )


def test_pca_rejects_fit_selection_outside_normalized_cells(graph_store) -> None:
    datastore, refs = graph_store
    wider_selection = refs["cells"]
    normalized_mask = _selection_mask(datastore, wider_selection)
    normalized_mask[np.flatnonzero(normalized_mask)[0]] = False
    datastore.cells.insert("normalized_cells", normalized_mask, overwrite=True)
    normalized_selection = datastore.snapshot_cell_selection(
        cell_key="normalized_cells"
    )
    normalized = datastore.run_normalization(normalized_selection, refs["features"])

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


@pytest.mark.parametrize(
    ("loadings", "error", "message"),
    [
        (np.ones(4), ValueError, "two-dimensional matrix with columns"),
        (np.ones((100, 0)), ValueError, "two-dimensional matrix with columns"),
        (np.ones((100, 2), dtype=complex), TypeError, "must contain real numbers"),
        (np.full((100, 2), np.nan), ValueError, "must contain only finite values"),
    ],
    ids=["one_dimensional", "no_columns", "complex", "non_finite"],
)
def test_custom_reduction_rejects_invalid_loadings(
    read_only_store, loadings, error, message
) -> None:
    datastore, refs = read_only_store

    with pytest.raises(error, match=message):
        datastore.run_custom_reduction(loadings, refs["normalized"])


def test_graph_construction_operations_reuse_persistent_local_cache(
    graph_store,
    monkeypatch,
    tmp_path,
) -> None:
    datastore, refs = graph_store
    cache_path = tmp_path / "normalized_cache"
    monkeypatch.setattr(
        graph_operations,
        "is_remote_datastore",
        lambda *_args: True,
    )
    normalized = datastore.run_normalization(refs["cells"], refs["features"])
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
    graph_template,
    monkeypatch,
    tmp_path,
    local_cache,
) -> None:
    # Staging only reads the store, so a read-only view of the template serves.
    zarr_loc, refs = graph_template
    datastore = _open_store(zarr_loc, zarr_mode="r")
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
        refs["normalized"],
        local_cache,
        100,
    ):
        assert cache_root.is_dir()
        staged = datastore._normalizedArtifactCache[refs["normalized"]]
        np.testing.assert_array_equal(
            staged.compute(),
            datastore.load_artifact(refs["normalized"])["data"][:],
        )

    assert not cache_root.exists()
    assert refs["normalized"] not in datastore._normalizedArtifactCache


def test_temporary_local_cache_is_removed_after_failure(
    graph_template,
    monkeypatch,
    tmp_path,
) -> None:
    zarr_loc, refs = graph_template
    datastore = _open_store(zarr_loc, zarr_mode="r")
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
        with datastore._cache_normalized_artifact(refs["normalized"], "auto", 100):
            raise RuntimeError("stop after staging")

    assert not cache_root.exists()


def test_embedding_initialization_persists_expected_payload(graph_store) -> None:
    datastore, refs = graph_store
    reduction = datastore.run_pca(refs["normalized"], dims=4)
    coordinates = datastore.load_artifact(reduction)["data"][:].astype(np.float64)

    # Blocks of 100 rows stream in nine reads, each coalesced into K-means
    # updates of five rows.
    initialization = datastore.build_embedding_initialization(
        reduction,
        n_centroids=5,
        batch_size=100,
        kmeans_batch_size=5,
    )

    initialization_group = datastore.load_artifact(initialization)
    centers = initialization_group["cluster_centers"][:]
    labels = initialization_group["cluster_labels"][:]
    assert centers.shape == (5, 4)
    assert initialization_group["cluster_labels"].dtype == np.uint32
    assert labels.shape == (len(coordinates),)
    assert set(np.unique(labels)) <= set(range(5))
    # Each stored label names the stored centroid nearest to its cell.
    squared = np.square(coordinates[:, np.newaxis, :] - centers[np.newaxis]).sum(axis=2)
    labelled = squared[np.arange(len(labels)), labels]
    np.testing.assert_allclose(labelled, squared.min(axis=1), rtol=1e-5, atol=1e-6)
    # Centroids are averages of cells, so they lie inside the coordinate range.
    assert np.all(centers >= coordinates.min(axis=0) - 1e-6)
    assert np.all(centers <= coordinates.max(axis=0) + 1e-6)
    inputs = datastore.inspect_artifact(initialization).inputs
    assert inputs is not None
    assert ArtifactRef.from_dict(inputs["coordinates"]) == reduction


@pytest.fixture(scope="module")
def neighbors_store(graph_template, tmp_path_factory) -> tuple[DataStore, ArtifactRef]:
    """A read-only copy of the template with 3-neighbor graph inputs."""
    zarr_loc, refs = graph_template
    target = tmp_path_factory.mktemp("graph_neighbors") / "store.zarr"
    shutil.copytree(zarr_loc, target)
    writable = _open_store(target)
    reduction = writable.run_pca(refs["normalized"], dims=4)
    neighbors = writable.query_neighbors(writable.build_ann_index(reduction), k=3)
    return _open_store(target, zarr_mode="r"), neighbors


@pytest.mark.parametrize(
    ("values", "error", "message"),
    [
        (
            {"local_connectivity": -1.0},
            ValueError,
            "local_connectivity must be finite and non-negative",
        ),
        (
            {"local_connectivity": np.nan},
            ValueError,
            "local_connectivity must be finite and non-negative",
        ),
        (
            {"local_connectivity": True},
            TypeError,
            "local_connectivity must be a real number",
        ),
        (
            {"bandwidth": 0.0},
            ValueError,
            "bandwidth must be finite and greater than zero",
        ),
        (
            {"bandwidth": -1.0},
            ValueError,
            "bandwidth must be finite and greater than zero",
        ),
        (
            {"bandwidth": np.nan},
            ValueError,
            "bandwidth must be finite and greater than zero",
        ),
        ({"bandwidth": True}, TypeError, "bandwidth must be a real number"),
    ],
)
def test_connectivity_rejects_invalid_kernel_parameters(
    neighbors_store,
    values,
    error,
    message,
) -> None:
    datastore, neighbors = neighbors_store

    with pytest.raises(error, match=message):
        datastore.build_connectivity_map(neighbors, **values)


def test_ann_index_rejects_invalid_runtime_parameters(graph_store) -> None:
    datastore, refs = graph_store
    reduction = datastore.run_pca(refs["normalized"], dims=4)

    for values, error, match in (
        ({"ann_metric": "ip"}, ValueError, "l2, cosine"),
        ({"ann_efc": 1.5}, TypeError, "ann_efc must be an integer"),
        ({"ann_ef": True}, TypeError, "ann_ef must be an integer"),
        ({"ann_m": 1}, ValueError, "at least two"),
        ({"rand_state": 0}, ValueError, "rand_state must be at least 1"),
        ({"batch_size": 0}, ValueError, "batch_size must be at least 1"),
    ):
        with pytest.raises(error, match=match):
            datastore.build_ann_index(
                reduction,
                **values,
            )
    assert not datastore.list_artifacts(kind="ann_index", from_assay="RNA")


def test_neighbor_count_changes_only_neighbor_and_connectivity_artifacts(
    graph_store,
) -> None:
    datastore, refs = graph_store
    cell_selection, features = refs["cells"], refs["features"]
    normalized = refs["normalized"]
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
    assert datastore.load_artifact(neighbors_three)["indices"].shape[1] == 3
    assert datastore.load_artifact(neighbors_four)["indices"].shape[1] == 4


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


def test_explicit_graph_preserves_exact_feature_selection_ref(graph_store) -> None:
    datastore, refs = graph_store
    reduction = datastore.run_pca(refs["normalized"], dims=3)
    ann = datastore.build_ann_index(reduction)
    neighbors = datastore.query_neighbors(ann, k=3)
    connectivity = datastore.build_connectivity_map(neighbors)
    ancestry = resolve_native_graph_inputs(datastore.zw, connectivity)

    assert ancestry.feature_selection == refs["features"]
    assert ancestry.cell_selection == refs["cells"]
    cell_status = datastore.inspect_artifact(ancestry.cell_selection)
    assert cell_status.operation == "auto_filter_cells"
    assert cell_status.execution_options == {"source_column": "artifact"}


def test_historical_neighbors_preserve_named_lineage_inputs(graph_store) -> None:
    datastore, refs = graph_store
    original_selection, features = refs["cells"], refs["features"]
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
    graph_store,
) -> None:
    datastore, refs = graph_store
    cell_selection, normalized = refs["cells"], refs["normalized"]
    reduction = datastore.run_pca(normalized, dims=4)
    mask = _selection_mask(datastore, cell_selection)
    frozen_mask = mask.copy()
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
    # Both fits keep the normalized artifact's frozen cells, not the live ``I``.
    n_frozen = int(frozen_mask.sum())
    assert datastore.load_artifact(new_reduction)["data"].shape == (n_frozen, 5)
    scores = datastore.load_artifact(reduction)["data"][:]
    frozen_batches = pd.DataFrame(
        {"graph_batch": _batch_labels(datastore.cells.N)[frozen_mask]}
    ).astype(object)
    expected = fit_harmony(
        np.asarray(scores.T, dtype=np.float64), frozen_batches, nclust=5
    )
    np.testing.assert_allclose(
        datastore.load_artifact(corrected)["data"][:],
        expected.corrected.T,
        rtol=2e-5,
        atol=2e-6,
    )


def test_run_harmony_refuses_read_only_store_before_writing_a_snapshot(
    graph_store,
) -> None:
    datastore, refs = graph_store
    pca = datastore.run_pca(refs["normalized"], dims=5)
    snapshots = datastore.list_artifacts(kind="metadata_snapshot", scope="datastore")

    read_only = DataStore(datastore.zarr_loc, default_assay="RNA", zarr_mode="r")
    with pytest.raises(PermissionError, match="snapshot_run_metadata"):
        read_only.run_harmony(pca, ["graph_batch"], harmony_params={"nclust": 5})
    assert (
        datastore.list_artifacts(kind="metadata_snapshot", scope="datastore")
        == snapshots
    )


def test_datastore_inspects_and_loads_artifact_read_only(graph_store) -> None:
    datastore, refs = graph_store
    ref = refs["normalized"]

    status = datastore.inspect_artifact(ref)
    group = datastore.load_artifact(ref)

    assert status.complete
    assert status.operation == "run_normalization"
    assert status.parameters == {
        "normalization_method": {
            "external_hook": True,
            "module": "scarf.assay",
            "qualname": "norm_lib_size",
        },
        "size_factor": 1000.0,
        "log_transform": True,
        "renormalize_subset": True,
    }
    assert status.input_ref("cell_selection") == refs["cells"]
    assert status.input_ref("feature_selection") == refs["features"]
    assert group["data"].shape == (
        int(_selection_mask(datastore, refs["cells"]).sum()),
        100,
    )
    # The store itself is writable; the loaded view is not.
    with pytest.raises(ValueError, match="read-only mode"):
        group.attrs["invalid"] = True
    assert "invalid" not in datastore.zw[artifact_path(ref)].attrs


def test_graph_harmony_is_an_explicit_ann_coordinate_source(
    graph_store,
    monkeypatch,
) -> None:
    datastore, refs = graph_store
    cell_selection = refs["cells"]
    batches = _batch_labels(datastore.cells.N)
    pca = datastore.run_pca(refs["normalized"], dims=5)
    pca_scores = datastore.load_artifact(pca)["data"][:]
    active_batches = pd.DataFrame(
        {"graph_batch": batches[_selection_mask(datastore, cell_selection)]}
    ).astype(object)
    expected_correction = fit_harmony(
        np.asarray(pca_scores.T, dtype=np.float64),
        active_batches,
        nclust=5,
    )

    def fail_projection(*_args, **_kwargs):
        raise AssertionError("persisted coordinates should be used")

    monkeypatch.setattr(
        graph_operations.ReductionTransform,
        "transform",
        fail_projection,
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


def test_lsi_and_custom_reduction_have_distinct_public_methods(graph_store) -> None:
    datastore, refs = graph_store
    normalized = refs["normalized"]
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
    normalized_values = datastore.load_artifact(normalized)["data"][:]
    n_features = normalized_values.shape[1]
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
    # Unscaled custom loadings project the normalized values directly.
    np.testing.assert_allclose(
        datastore.load_artifact(custom)["data"][:],
        normalized_values[:, :2],
        rtol=1e-6,
    )
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
    cell_selection = datastore.auto_filter_cells(method="gaussian")
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


def test_connectivity_rebuild_requires_named_distance_metric(graph_store) -> None:
    datastore, refs = graph_store
    reduction = datastore.run_pca(refs["normalized"], dims=4)
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


def test_ann_reuse_checks_metadata_and_explicit_validation_checks_bytes(
    graph_store,
) -> None:
    datastore, refs = graph_store
    reduction = datastore.run_pca(refs["normalized"], dims=3)
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
    assert datastore.build_ann_index(reduction) == current
    from scarf.storage.ann_index import validate_ann_index_payload

    with pytest.raises(ValueError, match="payload digest"):
        validate_ann_index_payload(ann_group, "l2", 3)
    repaired = datastore.build_ann_index(reduction, invalidate_cache=True)
    assert repaired != current
    assert datastore.build_ann_index(reduction) == repaired
    assert datastore.inspect_artifact(repaired).complete

    # An index without its metadata record is not reused; a new one is built.
    legacy_group = datastore.zw[artifact_path(repaired)]["ann_idx_bytes"]
    for attribute in (
        "ann_index_format_version",
        "metric",
        "dimensions",
        "element_count",
        "payload_sha256",
    ):
        del legacy_group.attrs[attribute]
    rebuilt = datastore.build_ann_index(reduction)
    assert rebuilt != repaired
    assert datastore.inspect_artifact(rebuilt).complete
    with pytest.raises(ValueError, match="metadata is missing"):
        validate_ann_index_payload(datastore.zw[artifact_path(repaired)], "l2", 3)

"""No complete artifact holds non-finite numbers where its contract needs finite ones.

Checked writers refuse NaN and infinity before they write a block, so a
producer that meets one raises ``NonFiniteArtifactError`` and publishes nothing.
"""

import dataclasses
import pickle
import shutil
from pathlib import Path

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf import DataStore
from scarf.neighbors.stages import (
    KMeansInitializationStage,
    NeighborQueryStage,
    ReductionTransform,
)
from scarf.storage.artifacts import ArtifactRef, list_artifacts
from scarf.storage.finite_values import (
    FiniteRowWriter,
    NonFiniteArtifactError,
    first_nonfinite_row,
    requires_finite_values,
    write_finite_array,
)
from scarf.storage.sharding import write_dense_from_row_batches
from tests.storage_helpers import write_count_store

N_CELLS = 40


def _root() -> zarr.Group:
    return zarr.open_group(store=MemoryStore(), mode="w")


# Checked writers


def test_first_nonfinite_row_scans_bounded_bands(monkeypatch) -> None:
    import scarf.storage.finite_values as finite_values

    values = np.zeros((9, 3))
    assert first_nonfinite_row(values) is None
    values[7, 2] = np.inf
    values[8, 0] = np.nan
    # Bands of one row cover the matrix piecewise and find the same row.
    monkeypatch.setattr(finite_values, "_SCAN_VALUES", 3)
    assert first_nonfinite_row(values) == 7
    assert first_nonfinite_row(np.array([1.0, np.nan])) == 1
    assert first_nonfinite_row(np.arange(4)) is None
    assert first_nonfinite_row(np.array([True, False])) is None
    assert first_nonfinite_row(np.float64(np.nan)) == 0
    with pytest.raises(TypeError, match="numeric"):
        first_nonfinite_row(np.array(["a"]))


def test_non_finite_error_names_operation_array_and_row() -> None:
    error = NonFiniteArtifactError("run_pca", "data", 7)

    assert isinstance(error, ValueError)
    assert (error.operation, error.array, error.row) == ("run_pca", "data", 7)
    assert "run_pca" in str(error)
    assert "'data'" in str(error)
    assert "row 7" in str(error)
    assert "a non-finite value (NaN or infinity)" in str(error)
    restored = pickle.loads(pickle.dumps(error))
    assert (restored.operation, restored.array, restored.row) == ("run_pca", "data", 7)


def test_row_writer_refuses_a_block_before_writing_it() -> None:
    array = _root().create_array("data", shape=(4, 2), dtype=np.float32, fill_value=0)
    block = np.ones((2, 2))
    block[1, 0] = np.nan

    with pytest.raises(NonFiniteArtifactError) as caught:
        with FiniteRowWriter(array, operation="run_test") as writer:
            writer.write(np.ones((2, 2)))
            writer.write(block)

    assert (caught.value.array, caught.value.row) == ("data", 3)
    np.testing.assert_array_equal(array[:2], 1)
    np.testing.assert_array_equal(array[2:], 0)


def test_row_writer_refuses_values_that_overflow_its_dtype() -> None:
    root = _root()
    array = root.create_array("data", shape=(2,), dtype=np.float32)

    # A finite value that the dtype cannot hold is infinite once cast.
    with pytest.raises(NonFiniteArtifactError, match="row 1"):
        write_finite_array(array, np.array([1.0, 1e39]), operation="run_test")

    # Integers that the floating-point dtype cannot hold overflow too.
    half = root.create_array("half", shape=(2, 2), dtype=np.float16)
    with pytest.raises(NonFiniteArtifactError) as caught:
        write_finite_array(
            half, np.array([[1, 2], [3, 70_000]], dtype=np.int64), operation="run_test"
        )
    assert caught.value.row == 1


def test_row_writer_requires_every_row_and_a_floating_array() -> None:
    root = _root()
    array = root.create_array("data", shape=(3,), dtype=np.float64)
    writer = FiniteRowWriter(array, operation="run_test")
    writer.write(np.ones(2))
    with pytest.raises(ValueError, match="2 of its 3 rows"):
        writer.close()
    with pytest.raises(ValueError, match="do not fit"):
        writer.write(np.ones(2))
    with pytest.raises(ValueError, match="does not fit array 'data' of shape"):
        writer.write(np.ones((1, 2)))
    writer.write(np.ones(1))
    writer.close()
    writer.close()
    with pytest.raises(RuntimeError, match="is closed"):
        writer.write(np.ones(1))
    with pytest.raises(ValueError, match="has no rows"):
        FiniteRowWriter(
            root.create_array("scalar", shape=(), dtype=np.float64),
            operation="run_test",
        )

    with pytest.raises(TypeError, match="floating-point"):
        FiniteRowWriter(
            root.create_array("labels", shape=(3,), dtype=np.uint32),
            operation="run_test",
        )
    with pytest.raises(ValueError, match="operation"):
        FiniteRowWriter(array, operation="")


def test_dense_batch_writer_checks_cast_rows() -> None:
    root = _root()
    finite = root.create_array("finite", shape=(5, 2), dtype=np.float32, chunks=(2, 2))
    batches = [np.ones((3, 2)), np.full((2, 2), 2.0)]

    assert (
        write_dense_from_row_batches(
            finite,
            iter(batches),
            requireFinite=True,
            operation="run_test",
        )
        == 5
    )

    overflow = root.create_array("overflow", shape=(5, 2), dtype=np.float32)
    # Finite float64 values that float32 cannot hold become infinite.
    bad = [np.ones((3, 2)), np.array([[1.0, 1.0], [1e300, 1.0]])]
    with pytest.raises(NonFiniteArtifactError) as caught:
        write_dense_from_row_batches(
            overflow,
            iter(bad),
            dtype=np.float32,
            requireFinite=True,
            operation="run_test",
        )
    assert (caught.value.operation, caught.value.array) == ("run_test", "overflow")
    assert caught.value.row == 4

    # float64 bands that the write casts to float32 are checked as stored.
    narrowing = root.create_array("narrowing", shape=(5, 2), dtype=np.float32)
    with pytest.raises(NonFiniteArtifactError, match="row 2"):
        write_dense_from_row_batches(
            narrowing,
            iter([np.array([[0.0, 0.0], [1.0, 1.0], [1e300, 0.0]]), np.ones((2, 2))]),
            dtype=np.float64,
            requireFinite=True,
            operation="run_test",
        )

    # So are bands of a dtype narrower than the destination.
    wide = root.create_array("wide", shape=(5, 2), dtype=np.float32)
    with pytest.raises(NonFiniteArtifactError, match="row 1"):
        write_dense_from_row_batches(
            wide,
            iter([np.array([[0.0, 0.0], [7e4, 0.0]]), np.ones((3, 2))]),
            dtype=np.float16,
            requireFinite=True,
            operation="run_test",
        )

    with pytest.raises(ValueError, match="operation"):
        write_dense_from_row_batches(finite, iter(batches), requireFinite=True)
    labels = root.create_array("labels", shape=(5, 2), dtype=np.uint32)
    with pytest.raises(TypeError, match="floating-point destination"):
        write_dense_from_row_batches(
            labels, iter(batches), requireFinite=True, operation="run_test"
        )


# Producers


def _counts() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(31)
    rna = rng.poisson(rng.gamma(1.0, 3.0, size=30), size=(N_CELLS, 30))
    # Every cell has RNA counts, so the live ``I`` column selects every cell.
    rna[:, 0] += 5
    return {"RNA": rna, "ADT": rng.poisson(20.0, size=(N_CELLS, 4)) + 1}


def _open(zarr_loc: Path) -> DataStore:
    return DataStore(
        str(zarr_loc),
        default_assay="RNA",
        min_features_per_cell=0,
        nthreads=1,
    )


def _write_template(zarr_loc: Path) -> dict[str, ArtifactRef]:
    write_count_store(str(zarr_loc), _counts(), "uint16")
    store = _open(zarr_loc)
    store.cells.insert("batch", np.where(np.arange(N_CELLS) % 2, "a", "b"))
    cells = store.snapshot_cell_selection()
    normalized = store.run_normalization(
        cells, store.select_all_features(from_assay="RNA")
    )
    pca = store.run_pca(normalized, dims=3)
    initialization = store.build_embedding_initialization(pca, n_centroids=5)
    neighbors = store.query_neighbors(store.build_ann_index(pca), k=5)
    graph = store.build_connectivity_map(neighbors)
    adt = store.run_normalization(cells, store.select_all_features(from_assay="ADT"))
    adt_neighbors = store.query_neighbors(
        store.build_ann_index(store.run_pca(adt, dims=2)), k=5
    )
    return {
        "cells": cells,
        "normalized": normalized,
        "pca": pca,
        "initialization": initialization,
        "neighbors": neighbors,
        "graph": graph,
        "adt_neighbors": adt_neighbors,
    }


@pytest.fixture(scope="module")
def template(tmp_path_factory) -> tuple[Path, dict[str, ArtifactRef]]:
    zarr_loc = tmp_path_factory.mktemp("finite_template") / "store.zarr"
    return zarr_loc, _write_template(zarr_loc)


@pytest.fixture
def finite_store(template, tmp_path) -> tuple[DataStore, dict[str, ArtifactRef]]:
    zarr_loc, refs = template
    target = tmp_path / "store.zarr"
    shutil.copytree(zarr_loc, target)
    return _open(target), refs


def _refs(store: DataStore, kind: str) -> set[ArtifactRef]:
    scope = "datastore" if kind == "integrated_graph" else "assay"
    return set(
        list_artifacts(
            store.zw,
            scope=scope,
            assay=None if scope == "datastore" else "RNA",
            kind=kind,
        )
    )


def _poison(values: np.ndarray, row: int) -> np.ndarray:
    poisoned = np.array(values, copy=True)
    poisoned.reshape(poisoned.shape[0], -1)[row, -1] = np.nan
    return poisoned


def _poison_transform(monkeypatch) -> None:
    transform = ReductionTransform.transform

    def poisoned(self, values):
        return _poison(transform(self, values), 6)

    monkeypatch.setattr(ReductionTransform, "transform", poisoned)


def _run_pca(store, refs, monkeypatch):
    _poison_transform(monkeypatch)
    return store.run_pca(refs["normalized"], dims=3, invalidate_cache=True)


def _run_pca_loadings(store, refs, monkeypatch):
    import scarf.embeddings.reduction as reduction

    fit = reduction.fit_incremental_pca

    def poisoned_fit(*args, **kwargs):
        loadings, model = fit(*args, **kwargs)
        return _poison(loadings, 2), model

    monkeypatch.setattr(reduction, "fit_incremental_pca", poisoned_fit)
    return store.run_pca(refs["normalized"], dims=3, invalidate_cache=True)


def _run_harmony(array: str):
    def run(store, refs, monkeypatch):
        import scarf.neighbors.stages as stages

        fit = stages.fit_harmony

        def poisoned_fit(*args, **kwargs):
            result = fit(*args, **kwargs)
            if array == "data":
                return dataclasses.replace(
                    result, corrected=_poison(result.corrected.T, 6).T
                )
            return dataclasses.replace(result, sigma=_poison(result.sigma, 1))

        monkeypatch.setattr(stages, "fit_harmony", poisoned_fit)
        return store.run_harmony(
            refs["pca"],
            ["batch"],
            harmony_params={"nclust": 3},
            invalidate_cache=True,
        )

    return run


def _build_embedding_initialization(store, refs, monkeypatch):
    fit = KMeansInitializationStage.fit

    def poisoned_fit(**kwargs):
        result = fit(**kwargs)
        result.model.cluster_centers_[2, 0] = np.inf
        return result

    monkeypatch.setattr(KMeansInitializationStage, "fit", staticmethod(poisoned_fit))
    return store.build_embedding_initialization(
        refs["pca"], n_centroids=5, invalidate_cache=True
    )


def _query_neighbors(store, refs, monkeypatch):
    query = NeighborQueryStage.query

    def poisoned_query(self, values, *, self_indices=None):
        indices, distances, missed = query(self, values, self_indices=self_indices)
        return indices, _poison(distances, 6), missed

    monkeypatch.setattr(NeighborQueryStage, "query", poisoned_query)
    ann_index = store.inspect_artifact(refs["neighbors"]).input_ref("ann_index")
    return store.query_neighbors(ann_index, k=5, invalidate_cache=True)


def _build_connectivity_map(store, refs, monkeypatch):
    import scarf.neighbors.graph as graph

    build = graph.build_connectivity_arrays

    def poisoned_build(*args, **kwargs):
        edges, weights = build(*args, **kwargs)
        return edges, _poison(weights, 6)

    monkeypatch.setattr(graph, "build_connectivity_arrays", poisoned_build)
    return store.build_connectivity_map(refs["neighbors"], invalidate_cache=True)


def _run_umap(store, refs, monkeypatch):
    import scarf.embeddings.umap as umap

    fit = umap.fit_transform

    def poisoned_fit(*args, **kwargs):
        coordinates, a, b = fit(*args, **kwargs)
        return _poison(coordinates, 6), a, b

    monkeypatch.setattr(umap, "fit_transform", poisoned_fit)
    return store.run_umap(
        refs["graph"], refs["initialization"], n_epochs=5, invalidate_cache=True
    )


def _integrate_wnn(store, refs, monkeypatch):
    import scarf.neighbors.integration as integration

    integrate = integration._wnn_integration_many

    def poisoned_integration(*args, **kwargs):
        graph, modality_weights = integrate(*args, **kwargs)
        graph.data = _poison(graph.data, 6)
        return graph, modality_weights

    monkeypatch.setattr(integration, "_wnn_integration_many", poisoned_integration)
    return store.integrate_assays(
        [refs["neighbors"], refs["adt_neighbors"]], invalidate_cache=True
    )


def _integrate_wnn_modality_weights(store, refs, monkeypatch):
    import scarf.neighbors.integration as integration

    integrate = integration._wnn_integration_many

    def poisoned_integration(*args, **kwargs):
        graph, modality_weights = integrate(*args, **kwargs)
        return graph, _poison(modality_weights, 6)

    monkeypatch.setattr(integration, "_wnn_integration_many", poisoned_integration)
    return store.integrate_assays(
        [refs["neighbors"], refs["adt_neighbors"]], invalidate_cache=True
    )


@pytest.mark.parametrize(
    ("produce", "kind", "operation", "array", "row"),
    [
        pytest.param(_run_pca, "reduction", "run_pca", "data", 6, id="pca"),
        pytest.param(
            _run_pca_loadings, "reduction", "run_pca", "loadings", 2, id="pca-loadings"
        ),
        pytest.param(
            _run_harmony("data"),
            "batch_correction",
            "run_harmony",
            "data",
            6,
            id="harmony-coordinates",
        ),
        pytest.param(
            _run_harmony("sigma"),
            "batch_correction",
            "run_harmony",
            "sigma",
            1,
            id="harmony-fit-state",
        ),
        pytest.param(
            _build_embedding_initialization,
            "embedding_initialization",
            "build_embedding_initialization",
            "cluster_centers",
            2,
            id="embedding-initialization",
        ),
        pytest.param(
            _query_neighbors, "neighbors", "query_neighbors", "distances", 6, id="knn"
        ),
        pytest.param(
            _build_connectivity_map,
            "connectivity_map",
            "build_connectivity_map",
            "weights",
            6,
            id="connectivity",
        ),
        pytest.param(_run_umap, "embedding", "run_umap", "values", 6, id="umap"),
        pytest.param(
            _integrate_wnn,
            "integrated_graph",
            "integrate_assays",
            "weights",
            6,
            id="wnn",
        ),
        pytest.param(
            _integrate_wnn_modality_weights,
            "integrated_graph",
            "integrate_assays",
            "modality_weights",
            6,
            id="wnn-modality-weights",
        ),
    ],
)
def test_non_finite_producer_output_raises_and_leaves_no_artifact(
    finite_store, monkeypatch, produce, kind, operation, array, row
) -> None:
    store, refs = finite_store
    before = _refs(store, kind)

    with pytest.raises(NonFiniteArtifactError) as caught:
        produce(store, refs, monkeypatch)

    assert (caught.value.operation, caught.value.array) == (operation, array)
    assert caught.value.row == row
    # The started slot was discarded, so no artifact of the kind was added,
    # complete or not.
    assert _refs(store, kind) == before


def test_cell_data_artifacts_check_the_arrays_their_kind_requires_finite(
    finite_store,
) -> None:
    from scarf.metadata.artifacts import (
        plan_cell_data_artifact,
        write_cell_data_artifact,
    )

    store, refs = finite_store
    cells = refs["cells"]

    def write(kind: str, name: str, values: np.ndarray) -> zarr.Group:
        planned = plan_cell_data_artifact(
            store.zw,
            scope="assay",
            assay="RNA",
            kind=kind,
            operation=f"manual_{kind}",
            parameters={"name": name},
            inputs={},
            execution_options={},
            cell_selection=cells,
            arrays={"values": (values.shape, "f")},
        )
        return write_cell_data_artifact(store.zw, planned, {"values": values})

    layout = np.ones((N_CELLS, 2))
    # The kind decides which arrays a checked writer writes.
    write("embedding", "layout", layout)
    layout[3, 1] = np.nan
    before = _refs(store, "embedding")
    with pytest.raises(NonFiniteArtifactError) as caught:
        write("embedding", "poisoned", layout)
    assert (caught.value.operation, caught.value.array, caught.value.row) == (
        "manual_embedding",
        "values",
        3,
    )
    assert _refs(store, "embedding") == before
    # A kind listed with None checks every floating-point array.
    assert requires_finite_values("batch_correction", "sigma", np.float32)
    assert not requires_finite_values("batch_correction", "codes", np.int64)
    # A quality metric may hold NaN for an undefined value.
    metric = write("quality_metric", "undefined", layout[:, 1])
    assert np.isnan(metric["values"][3])

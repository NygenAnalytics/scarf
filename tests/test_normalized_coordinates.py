"""Normalized artifacts as graph coordinates, and their finite-value contract.

A normalized artifact holds one float32 column per selected feature, so a
graph can be built on its values without a reduction, as ``pca_dims=0`` does
in the pipeline. Its values are then coordinates, so their finiteness is part
of their contract: the writers of ``run_normalization`` are checked writers,
and a non-finite value raises before the artifact is complete. Harmony, WNN,
mapping references, and doublet scoring need reduced coordinates and reject a
graph on normalized values with a clear error.
"""

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr

from scarf import DataStore
from scarf.storage.artifacts import ArtifactRef, artifact_group
from scarf.storage.errors import ArtifactResolutionError
from scarf.storage.finite_values import NonFiniteArtifactError
from tests.storage_helpers import write_count_store

N_CELLS = 40
SUBSET = np.arange(12)
# The selected cell whose values the poisoned normalizer makes NaN.
POISONED_CELL = 23


def _counts() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(17)
    rna = rng.poisson(rng.gamma(1.0, 3.0, size=30), size=(N_CELLS, 30))
    # Two groups of cells that differ in their first features.
    rna[: N_CELLS // 2, :6] += 6
    rna[:, 0] += 1
    return {"RNA": rna, "ADT": rng.poisson(20.0, size=(N_CELLS, 4)) + 1}


def _open(zarr_loc: Path) -> DataStore:
    return DataStore(
        str(zarr_loc),
        default_assay="RNA",
        min_features_per_cell=0,
        nthreads=1,
    )


def poisoned_by_cell(assay: Any, counts: Any) -> Any:
    """A custom RNA normalizer whose values are NaN for one selected cell."""
    poison = np.ones((counts.shape[0], 1))
    poison[POISONED_CELL] = np.nan
    return 50.0 * counts * poison / assay.scalar.reshape(-1, 1)


def _write_template(zarr_loc: Path) -> dict[str, ArtifactRef]:
    write_count_store(str(zarr_loc), _counts(), "uint16")
    store = _open(zarr_loc)
    store.cells.insert("batch", np.where(np.arange(N_CELLS) % 2, "a", "b"))
    cells = store.snapshot_cell_selection()
    features = store.set_feature_selection(from_assay="RNA", feature_indexes=SUBSET)
    adt = store.run_normalization(cells, store.select_all_features(from_assay="ADT"))
    adt_neighbors = store.query_neighbors(
        store.build_ann_index(store.run_pca(adt, dims=2)), k=5
    )
    return {
        "cells": cells,
        "features": features,
        "normalized": store.run_normalization(cells, features),
        "adt_neighbors": adt_neighbors,
    }


@pytest.fixture(scope="module")
def template(tmp_path_factory) -> tuple[Path, dict[str, ArtifactRef]]:
    zarr_loc = tmp_path_factory.mktemp("normalized_coordinates") / "store.zarr"
    return zarr_loc, _write_template(zarr_loc)


@pytest.fixture
def coordinate_store(template, tmp_path) -> tuple[DataStore, dict[str, ArtifactRef]]:
    zarr_loc, refs = template
    target = tmp_path / "store.zarr"
    shutil.copytree(zarr_loc, target)
    return _open(target), refs


def _normalized_graph(
    store: DataStore, refs: dict[str, ArtifactRef]
) -> dict[str, ArtifactRef]:
    """Build an ANN index, neighbors, and a graph on the normalized values."""
    ann_index = store.build_ann_index(refs["normalized"])
    neighbors = store.query_neighbors(ann_index, k=5)
    return {
        **refs,
        "ann_index": ann_index,
        "neighbors": neighbors,
        "graph": store.build_connectivity_map(neighbors),
    }


def _normalized_values(store: DataStore, ref: ArtifactRef) -> np.ndarray:
    return np.asarray(artifact_group(store.zw, ref)["data"][:])


# Graphs on normalized values


def test_ann_index_and_neighbors_are_built_on_the_normalized_values(
    coordinate_store,
) -> None:
    store, refs = coordinate_store
    refs = _normalized_graph(store, refs)
    values = _normalized_values(store, refs["normalized"])
    assert values.shape == (N_CELLS, len(SUBSET))

    for key in ("ann_index", "neighbors"):
        status = store.inspect_artifact(refs[key])
        assert status.input_ref("coordinates") == refs["normalized"]
    # Forty cells are few enough for the index search to be exact, so the
    # neighbors are the nearest cells in the space of the selected features.
    indices = np.asarray(artifact_group(store.zw, refs["neighbors"])["indices"][:])
    exact = np.linalg.norm(
        values[:, None].astype(np.float64) - values[None].astype(np.float64), axis=2
    )
    np.fill_diagonal(exact, np.inf)
    distances = np.asarray(artifact_group(store.zw, refs["neighbors"])["distances"][:])
    np.testing.assert_allclose(
        distances,
        np.take_along_axis(exact, indices.astype(np.intp), axis=1),
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        distances, np.sort(exact, axis=1)[:, :5], rtol=1e-4, atol=1e-6
    )
    # The same request reuses the index; an embedding initialization accepts
    # the normalized coordinates too.
    assert store.build_ann_index(refs["normalized"]) == refs["ann_index"]
    initialization = store.build_embedding_initialization(
        refs["normalized"], n_centroids=4
    )
    centers = artifact_group(store.zw, initialization)["cluster_centers"]
    assert centers.shape == (4, len(SUBSET))


def test_harmony_rejects_normalized_coordinates_before_any_write(
    coordinate_store,
) -> None:
    store, refs = coordinate_store
    before = store.list_artifacts(scope="datastore")

    with pytest.raises(ValueError) as caught:
        store.run_harmony(refs["normalized"], ["batch"])

    assert str(caught.value) == (
        "Harmony corrects reduction coordinates, such as those of run_pca, but "
        "got a normalized artifact; normalized values cannot be corrected, so "
        "run run_pca on them first"
    )
    assert store.list_artifacts(scope="datastore") == before
    assert store.list_artifacts(kind="batch_correction", from_assay="RNA") == []


def test_wnn_rejects_neighbors_built_on_normalized_values(coordinate_store) -> None:
    store, refs = coordinate_store
    refs = _normalized_graph(store, refs)

    with pytest.raises(ArtifactResolutionError) as caught:
        store.integrate_assays([refs["neighbors"], refs["adt_neighbors"]])

    assert str(caught.value) == (
        "WNN coordinates must be reduction or batch_correction, but the RNA "
        "neighbors were built on normalized values; build them on a reduction "
        "such as run_pca, or integrate connectivity maps with method='snn'"
    )
    assert caught.value.code == "wrong_kind"
    assert caught.value.context["actual_kind"] == "normalized"
    assert store.list_artifacts(scope="datastore", kind="integrated_graph") == []


def test_mapping_references_and_doublets_reject_normalized_graphs(
    coordinate_store,
) -> None:
    store, refs = coordinate_store
    refs = _normalized_graph(store, refs)

    with pytest.raises(
        ValueError, match="Neighbor coordinates must be a PCA reduction"
    ):
        store.build_mapping_reference(refs["neighbors"])
    clusters = store.run_leiden_clustering(refs["graph"], resolution=1.0)
    with pytest.raises(
        ValueError, match="Doublet detection requires an uncorrected PCA graph"
    ):
        store.run_doublet_detection(clusters, refs["graph"], heterotypic_fraction=0)


# The finite-value contract of normalized data


def _mirrored_destination() -> tuple[zarr.Group, zarr.Array]:
    root = zarr.open_group(store=zarr.storage.MemoryStore(), mode="w")
    mirror = root.create_array(
        "mirror",
        shape=(N_CELLS, len(SUBSET)),
        chunks=(N_CELLS, len(SUBSET)),
        dtype=np.float32,
    )
    return root, mirror


def test_checked_subset_writer_with_a_mirror_writes_both_arrays(
    coordinate_store,
) -> None:
    from scarf.assay.normalization import write_renorm_subset_to_zarr

    store, _refs = coordinate_store
    root, mirror = _mirrored_destination()

    # A mirror selects the writer that reads row bands of the counts.
    write_renorm_subset_to_zarr(
        store.RNA,
        np.arange(N_CELLS),
        SUBSET,
        root,
        "data",
        1,
        mirror=mirror,
        requireFinite=True,
        operation="run_normalization",
    )

    np.testing.assert_array_equal(mirror[:], root["data"][:])


def test_custom_normalizer_nan_raises_at_write_time(
    coordinate_store, monkeypatch
) -> None:
    store, refs = coordinate_store
    before = store.list_artifacts(kind="normalized", from_assay="RNA")
    monkeypatch.setattr(store.RNA, "normMethod", poisoned_by_cell)

    with pytest.raises(NonFiniteArtifactError) as caught:
        store.run_normalization(refs["cells"], refs["features"])

    assert (caught.value.operation, caught.value.array, caught.value.row) == (
        "run_normalization",
        "data",
        POISONED_CELL,
    )
    # The started slot was discarded, so no artifact was added, complete or
    # not.
    assert store.list_artifacts(kind="normalized", from_assay="RNA") == before


def test_log_of_a_negative_library_size_value_raises_at_write_time(
    tmp_path,
) -> None:
    from scarf.assay.normalization import (
        write_renorm_subset_to_zarr as checked_subset_writer,
    )
    from scarf.writers import write_renorm_subset_to_zarr

    counts = _counts()
    rna = counts["RNA"].astype(np.float32)
    # A negative count in a cell whose subset total stays positive normalizes
    # below -1, so its log1p is NaN.
    rna[POISONED_CELL, :2] = (-1.0, 3.0)
    rna[POISONED_CELL, 2 : len(SUBSET)] = 0.0
    write_count_store(str(tmp_path / "store.zarr"), {"RNA": rna}, "float32")
    store = _open(tmp_path / "store.zarr")
    cells = store.snapshot_cell_selection()
    features = store.set_feature_selection(from_assay="RNA", feature_indexes=SUBSET)

    # The library-size subset writer streams the counts from countsT.
    with pytest.raises(NonFiniteArtifactError) as caught:
        store.run_normalization(cells, features)

    assert (caught.value.operation, caught.value.array, caught.value.row) == (
        "run_normalization",
        "data",
        POISONED_CELL,
    )
    assert store.list_artifacts(kind="normalized", from_assay="RNA") == []

    # With a mirror, the checked writer reads row bands of the counts, and
    # checks them too, naming the operation it writes for.
    root, mirror = _mirrored_destination()
    with pytest.raises(NonFiniteArtifactError) as caught:
        checked_subset_writer(
            store.RNA,
            np.arange(N_CELLS),
            SUBSET,
            root,
            "data",
            1,
            log_transform=True,
            mirror=mirror,
            requireFinite=True,
            operation="custom_subset",
        )
    assert (caught.value.operation, caught.value.row) == (
        "custom_subset",
        POISONED_CELL,
    )

    # The public writer writes the values as computed: the NaN is stored.
    root, mirror = _mirrored_destination()
    write_renorm_subset_to_zarr(
        store.RNA,
        np.arange(N_CELLS),
        SUBSET,
        root,
        "data",
        1,
        log_transform=True,
        mirror=mirror,
    )
    values = np.asarray(root["data"][:])
    assert np.isnan(values[POISONED_CELL, 0])
    assert np.isfinite(np.delete(values, POISONED_CELL, axis=0)).all()

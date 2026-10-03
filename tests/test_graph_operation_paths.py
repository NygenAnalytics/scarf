"""Argument, reuse, staging, and integrity paths of the explicit graph operations."""

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import Mock

import numpy as np
import pytest

from scarf import DataStore
from scarf.datastore._operations import graph as graph_operations
from scarf.graph.arguments import OperationArguments
from scarf.storage.artifact_writer import artifact_transaction, plan_artifact
from scarf.storage.artifacts import (
    ArtifactRef,
    artifact_group,
    artifact_path,
    new_artifact_id,
)
from scarf.storage.errors import ArtifactResolutionError
from scarf.utils.logging import logger
from tests.storage_helpers import write_count_store
from tests.storage_helpers import insert_nullable_cell_column

N_CELLS = 40


def synthetic_counts() -> dict[str, np.ndarray]:
    """RNA, ADT, HTO, and ATAC counts over the same cells."""
    rng = np.random.default_rng(29)
    rna = rng.poisson(rng.gamma(1.0, 3.0, size=30), size=(N_CELLS, 30))
    # Every cell has RNA counts, so the live ``I`` column selects every cell.
    rna[:, 0] += 5
    return {
        "RNA": rna,
        "ADT": rng.poisson(20.0, size=(N_CELLS, 4)) + 1,
        "HTO": rng.poisson(5.0, size=(N_CELLS, 3)),
        "ATAC": rng.poisson(0.8, size=(N_CELLS, 25)),
    }


def open_store(zarr_loc: Path, **options: Any) -> DataStore:
    return DataStore(
        str(zarr_loc),
        default_assay="RNA",
        min_features_per_cell=0,
        nthreads=1,
        **options,
    )


def neighbors_for(
    store: DataStore,
    assay: str,
    cells: ArtifactRef,
    *,
    dims: int = 2,
) -> ArtifactRef:
    """Build ``assay`` neighbors over ``cells`` through the public stages."""
    features = store.select_all_features(from_assay=assay)
    normalized = store.run_normalization(cells, features)
    reduction = store.run_pca(normalized, dims=dims)
    return store.query_neighbors(store.build_ann_index(reduction), k=5)


def write_graph_template(zarr_loc: Path) -> dict[str, ArtifactRef]:
    """Write a store with RNA and ADT neighbors over every cell and their WNN graph.

    WNN integration takes neighbors, so no connectivity map is built.
    """
    write_count_store(str(zarr_loc), synthetic_counts(), "uint16")
    store = open_store(zarr_loc)
    cells = store.snapshot_cell_selection()
    rna_features = store.select_all_features(from_assay="RNA")
    rna_normalized = store.run_normalization(cells, rna_features)
    rna_pca = store.run_pca(rna_normalized, dims=3)
    rna_neighbors = store.query_neighbors(store.build_ann_index(rna_pca), k=5)
    adt_neighbors = neighbors_for(store, "ADT", cells)
    return {
        "cells": cells,
        "rna_features": rna_features,
        "rna_normalized": rna_normalized,
        "rna_pca": rna_pca,
        "rna_neighbors": rna_neighbors,
        "adt_neighbors": adt_neighbors,
        "atac_features": store.select_all_features(from_assay="ATAC"),
        "wnn": store.integrate_assays([rna_neighbors, adt_neighbors]),
    }


@pytest.fixture(scope="module")
def graph_template(tmp_path_factory) -> tuple[Path, dict[str, ArtifactRef]]:
    zarr_loc = tmp_path_factory.mktemp("graph_template") / "store.zarr"
    return zarr_loc, write_graph_template(zarr_loc)


@pytest.fixture(scope="module")
def read_only_store(graph_template) -> tuple[DataStore, dict[str, ArtifactRef]]:
    """The template opened read only, for calls that must fail before writing."""
    zarr_loc, refs = graph_template
    return open_store(zarr_loc, zarr_mode="r"), refs


@pytest.fixture
def graph_store(graph_template, tmp_path) -> tuple[DataStore, dict[str, ArtifactRef]]:
    zarr_loc, refs = graph_template
    target = tmp_path / "store.zarr"
    shutil.copytree(zarr_loc, target)
    return open_store(target), refs


def _artifacts(
    store: DataStore,
    kind: str,
    assay: str | None = "RNA",
) -> list[ArtifactRef]:
    if assay is None:
        return store.list_artifacts(kind=kind, scope="datastore")
    return store.list_artifacts(kind=kind, from_assay=assay)


def datastore_scoped_record(
    store: DataStore,
    kind: str,
    arrays: dict[str, np.ndarray],
) -> ArtifactRef:
    """Store a complete datastore-scoped record of a kind scarf keeps per assay.

    No scarf producer writes one, but a foreign or hand-edited store can.
    """
    planned = plan_artifact(
        store.zw,
        scope="datastore",
        kind=kind,
        operation=f"import_{kind}",
        parameters={},
        inputs={},
        execution_options={},
    )
    with artifact_transaction(store.zw, planned) as group:
        for name, values in arrays.items():
            group.create_array(name, data=values)
    return planned.ref


def _ann_index(store: DataStore, neighbors: ArtifactRef) -> ArtifactRef:
    return ArtifactRef.from_dict(store.inspect_artifact(neighbors).inputs["ann_index"])


def test_operation_arguments_reject_a_field_without_an_argument_role() -> None:
    @dataclass(frozen=True, slots=True)
    class UnroledArguments(OperationArguments):
        operation: ClassVar[str] = "run_pca"
        artifact_kind: ClassVar[str] = "reduction"

        dims: int = 2

    # A field without a role would otherwise leave the artifact identity
    # silently.
    with pytest.raises(TypeError, match=r"UnroledArguments\.dims has no argument role"):
        UnroledArguments().to_record()


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda store: store.run_pca("normalized"), "normalized must be"),
        (lambda store: store.run_harmony("reduction", ["batch"]), "reduction must be"),
        (
            lambda store: store.build_embedding_initialization("reduction"),
            "coordinates must be",
        ),
        (lambda store: store.build_ann_index("reduction"), "coordinates must be"),
        (lambda store: store.query_neighbors("ann_index"), "ann_index must be"),
        (lambda store: store.build_connectivity_map("neighbors"), "neighbors must be"),
        (lambda store: store.load_graph("graph"), "graph must be"),
    ],
    ids=[
        "run_pca",
        "run_harmony",
        "build_embedding_initialization",
        "build_ann_index",
        "query_neighbors",
        "build_connectivity_map",
        "load_graph",
    ],
)
def test_graph_operations_take_artifact_refs_rather_than_names(
    read_only_store,
    call,
    message,
) -> None:
    store, _refs = read_only_store

    with pytest.raises(TypeError, match=f"{message} an ArtifactRef"):
        call(store)


def test_normalization_requires_an_assay_scoped_feature_selection(
    read_only_store,
) -> None:
    store, refs = read_only_store
    detached = ArtifactRef(
        scope="datastore",
        kind="feature_selection",
        artifact_id=new_artifact_id(),
    )

    with pytest.raises(ValueError, match="Feature-selection artifact has no assay"):
        store.run_normalization(refs["cells"], detached)


@pytest.mark.parametrize(
    ("kind", "arrays", "call", "message"),
    [
        (
            "normalized",
            {"data": np.ones((N_CELLS, 4), dtype=np.float32)},
            lambda store, record: store.run_pca(record, dims=2),
            "Normalized artifact has no assay",
        ),
        (
            "ann_index",
            {},
            lambda store, record: store.query_neighbors(record),
            "ANN artifact has no assay",
        ),
    ],
    ids=["normalized", "ann_index"],
)
def test_assay_operations_refuse_a_datastore_scoped_record(
    graph_store,
    kind,
    arrays,
    call,
    message,
) -> None:
    store, _refs = graph_store
    record = datastore_scoped_record(store, kind, arrays)

    with pytest.raises(ValueError, match=message):
        call(store, record)


@pytest.mark.parametrize("flag", ["log_transform", "renormalize_subset"])
@pytest.mark.parametrize("assay", ["RNA", "ATAC"])
def test_normalization_flags_must_be_booleans(read_only_store, assay, flag) -> None:
    store, refs = read_only_store
    features = refs[f"{assay.lower()}_features"]

    with pytest.raises(TypeError, match=f"{flag} must be a boolean"):
        store.run_normalization(refs["cells"], features, **{flag: "no"})


def test_normalization_refuses_a_dynamic_method_without_an_identity(
    read_only_store,
    monkeypatch,
) -> None:
    store, refs = read_only_store
    # A lambda has no importable name, so provenance could not identify it.
    monkeypatch.setattr(store.RNA, "normMethod", lambda assay, counts: counts)

    with pytest.raises(ValueError, match="must define artifact_identity"):
        store.run_normalization(refs["cells"], refs["rna_features"])


def test_custom_reduction_loadings_must_cover_the_normalized_features(
    read_only_store,
) -> None:
    store, refs = read_only_store
    n_features = store.RNA.feats.N

    with pytest.raises(ValueError, match="rows must match normalized features"):
        store.run_custom_reduction(
            np.ones((n_features - 1, 2)),
            refs["rna_normalized"],
        )


def test_pca_shows_the_elbow_plot_only_after_a_new_fit(
    graph_store,
    monkeypatch,
) -> None:
    store, refs = graph_store
    shown: list[dict[str, Any]] = []
    monkeypatch.setattr("scarf.plotting.elbow", lambda **kwargs: shown.append(kwargs))

    store.run_pca(refs["rna_normalized"], dims=2, show_elbow_plot=True)

    (plot,) = shown
    variance = np.asarray(plot["variance_explained"])
    assert plot["show"] is True
    # The percentage of variance explained by each fitted component.
    assert np.all(variance > 0)
    assert np.all(np.diff(variance) <= 0)
    assert variance.sum() <= 100
    store.run_pca(refs["rna_normalized"], dims=2, show_elbow_plot=True)
    assert len(shown) == 1


def test_a_local_cache_directory_reuses_its_completed_staging_copy(
    graph_store,
    tmp_path,
    monkeypatch,
) -> None:
    store, refs = graph_store
    normalized = refs["rna_normalized"]
    # Stage the normalized data as if the store were remote.
    monkeypatch.setattr(graph_operations, "is_remote_datastore", lambda *_: True)
    copy_array = Mock(side_effect=graph_operations.copy_zarr_array)
    monkeypatch.setattr(graph_operations, "copy_zarr_array", copy_array)
    cache = tmp_path / "normalized_cache"

    first = store.run_pca(normalized, dims=2, local_cache=str(cache))
    second = store.run_pca(
        normalized,
        dims=2,
        local_cache=str(cache),
        invalidate_cache=True,
    )

    assert second != first
    copy_array.assert_called_once()
    # An explicit cache directory outlives the calls that staged into it.
    assert any(cache.iterdir())
    np.testing.assert_array_equal(
        artifact_group(store.zw, second)["data"][:],
        artifact_group(store.zw, first)["data"][:],
    )


def test_streaming_lsi_shrinks_its_blocks_to_fit_a_tight_memory_budget(
    tmp_path,
) -> None:
    rng = np.random.default_rng(3)
    zarr_loc = tmp_path / "atac.zarr"
    write_count_store(
        str(zarr_loc), {"ATAC": rng.poisson(0.3, size=(40, 400))}, "uint16"
    )
    roomy_store = DataStore(
        str(zarr_loc),
        default_assay="ATAC",
        min_features_per_cell=0,
        nthreads=1,
    )
    normalized = roomy_store.run_normalization(
        roomy_store.snapshot_cell_selection(),
        roomy_store.select_all_features(from_assay="ATAC"),
    )
    roomy = roomy_store.run_lsi(normalized, dims=2)
    # This budget holds the LSI accumulators but not the whole 40-row band.
    # The coordinate write holds three blocks at once, so it needs smaller
    # blocks than the fit.
    tight_store = DataStore(
        str(zarr_loc),
        default_assay="ATAC",
        min_features_per_cell=0,
        nthreads=1,
        mem_budget=300_000,
    )
    warnings: list[str] = []
    sink = logger.add(
        lambda message: warnings.append(message.record["message"]),
        level="WARNING",
    )
    try:
        tight = tight_store.run_lsi(normalized, dims=2, invalidate_cache=True)
    finally:
        logger.remove(sink)

    assert roomy_store.inspect_artifact(roomy).execution_options["batch_size"] == 40
    assert tight_store.inspect_artifact(tight).execution_options["batch_size"] < 40
    assert any("to honor the memory budget" in message for message in warnings)
    # batch_size was left unset, so no warning suggests unsetting it.
    assert not any("Leave batch_size unset" in message for message in warnings)
    np.testing.assert_allclose(
        artifact_group(tight_store.zw, tight)["data"][:],
        artifact_group(roomy_store.zw, roomy)["data"][:],
        rtol=0,
        atol=1e-5,
    )


def test_harmony_reads_masked_batch_labels_as_missing(graph_store) -> None:
    store, refs = graph_store
    missing = np.zeros(N_CELLS, dtype=bool)
    missing[:4] = True
    labels = np.where(np.arange(N_CELLS) % 2, "a", "b")
    # Unmasked, the placeholder would be read as a third batch.
    labels[missing] = ""
    insert_nullable_cell_column(store, "batch", labels, missing)

    with pytest.raises(ValueError, match="cannot contain missing values"):
        store.run_harmony(refs["rna_pca"], ["batch"], harmony_params={"nclust": 3})
    assert not _artifacts(store, "batch_correction")

    store.cells.insert("labelled", ~missing, overwrite=True)
    normalized = store.run_normalization(
        store.snapshot_cell_selection("labelled"),
        refs["rna_features"],
    )
    corrected = store.run_harmony(
        store.run_pca(normalized, dims=3),
        ["batch"],
        harmony_params={"nclust": 3},
    )
    assert store.inspect_artifact(corrected).complete


def test_harmony_reuses_a_matching_correction(graph_store, monkeypatch) -> None:
    store, refs = graph_store
    store.cells.insert("batch", np.where(np.arange(N_CELLS) % 2, "a", "b"))
    corrected = store.run_harmony(
        refs["rna_pca"],
        ["batch"],
        harmony_params={"nclust": 3},
    )
    monkeypatch.setattr(
        "scarf.neighbors.stages.fit_harmony",
        Mock(side_effect=AssertionError("a matching correction must be reused")),
    )

    assert (
        store.run_harmony(refs["rna_pca"], ["batch"], harmony_params={"nclust": 3})
        == corrected
    )


def test_neighbor_queries_refuse_an_ann_record_without_a_supported_metric(
    graph_store,
) -> None:
    store, refs = graph_store
    ann = _ann_index(store, refs["rna_neighbors"])
    group = artifact_group(store.zw, ann)
    provenance = group.attrs["provenance"]
    # build_ann_index records only l2 or cosine, so an older or foreign writer
    # stored this inner-product index.
    group.attrs["provenance"] = {
        **provenance,
        "parameters": {**provenance["parameters"], "ann_metric": "ip"},
    }

    with pytest.raises(ValueError, match="no supported distance metric"):
        store.query_neighbors(ann, k=3)


def test_neighbor_queries_refuse_more_cells_than_graph_records_hold(
    read_only_store,
    monkeypatch,
) -> None:
    store, refs = read_only_store
    stream_coordinates = store._coordinate_source

    def uint32_overflow(coordinates: ArtifactRef, *, batch_size: int | None):
        source, _n_cells, dims = stream_coordinates(coordinates, batch_size=batch_size)
        return source, 2**32, dims

    # Indices of 2**32 cells fit uint32, but stored graph payloads must also
    # record n_cells within the uint32 range.
    monkeypatch.setattr(store, "_coordinate_source", uint32_overflow)

    with pytest.raises(ValueError, match=r"fewer than 2\*\*32 cells"):
        store.query_neighbors(_ann_index(store, refs["rna_neighbors"]), k=3)


def test_load_graph_rejects_a_graph_without_its_dimensions(graph_store) -> None:
    store, refs = graph_store
    graph = refs["wnn"]
    assert store.load_graph(graph).shape == (N_CELLS, N_CELLS)

    del store.zw[artifact_path(graph)].attrs["n_neighbors"]

    for use_k in (None, 1):
        with pytest.raises(ValueError, match="missing n_cells or n_neighbors"):
            store.load_graph(graph, use_k=use_k)


def test_integration_sources_must_belong_to_an_assay(read_only_store) -> None:
    store, refs = read_only_store
    detached = ArtifactRef(
        scope="datastore",
        kind="neighbors",
        artifact_id=new_artifact_id(),
    )

    with pytest.raises(ArtifactResolutionError, match="has no assay") as caught:
        store.integrate_assays([detached, refs["adt_neighbors"]])
    assert caught.value.code == "wrong_scope"


@pytest.mark.parametrize(
    "adt_cells",
    [np.arange(N_CELLS) >= 10, np.arange(N_CELLS) < 20],
    ids=["as_many_cells", "fewer_cells"],
)
def test_integration_requires_one_shared_cell_selection(
    graph_store,
    adt_cells,
) -> None:
    store, _refs = graph_store
    store.cells.insert("first", np.arange(N_CELLS) < 30)
    store.cells.insert("adt_cells", adt_cells)
    rna = neighbors_for(store, "RNA", store.snapshot_cell_selection("first"), dims=3)
    adt = neighbors_for(store, "ADT", store.snapshot_cell_selection("adt_cells"))
    before = _artifacts(store, "integrated_graph", assay=None)

    with pytest.raises(ValueError, match="one exact shared cell selection") as caught:
        store.integrate_assays([rna, adt])
    # Sound sources over other cells are not reported as a corrupt payload.
    assert not isinstance(caught.value, ArtifactResolutionError)
    assert _artifacts(store, "integrated_graph", assay=None) == before


def test_integration_rejects_a_source_payload_over_other_cells(graph_store) -> None:
    store, refs = graph_store
    adt = refs["adt_neighbors"]
    payload = artifact_group(store.zw, adt)
    # Replace the payload with a well-formed one over one cell fewer than its
    # cell selection.
    n_cells = N_CELLS - 1
    n_neighbors = int(payload["indices"].shape[1])
    offsets = np.arange(1, n_neighbors + 1)
    indices = (np.arange(n_cells)[:, None] + offsets) % n_cells
    payload.create_array("indices", data=indices.astype(np.uint32), overwrite=True)
    payload.create_array(
        "distances",
        data=np.ones((n_cells, n_neighbors), dtype=np.float32),
        overwrite=True,
    )
    payload.attrs["n_cells"] = n_cells

    with pytest.raises(
        ArtifactResolutionError, match="different cell counts"
    ) as caught:
        store.integrate_assays([refs["rna_neighbors"], adt])
    assert caught.value.code == "corrupt_payload"
    assert caught.value.context["artifact_id"] == adt.artifact_id


def test_wnn_integration_assembles_coordinates_streamed_in_several_blocks(
    graph_store,
    monkeypatch,
) -> None:
    store, refs = graph_store
    stream_coordinates = store._coordinate_source

    class SplitBlocks:
        """Stream each stored block in three parts, as several bands stream."""

        def __init__(self, source: Any) -> None:
            self.source = source

        def iter_coordinate_blocks(self, message: str):
            for block in self.source.iter_coordinate_blocks(message):
                yield from np.array_split(np.asarray(block), 3)

    def split_coordinate_source(coordinates: ArtifactRef, *, batch_size: int | None):
        source, n_cells, dims = stream_coordinates(coordinates, batch_size=batch_size)
        return SplitBlocks(source), n_cells, dims

    monkeypatch.setattr(store, "_coordinate_source", split_coordinate_source)

    split = store.integrate_assays(
        [refs["rna_neighbors"], refs["adt_neighbors"]],
        invalidate_cache=True,
    )

    assert split != refs["wnn"]
    for name in ("edges", "weights", "modality_weights"):
        np.testing.assert_array_equal(
            artifact_group(store.zw, split)[name][:],
            artifact_group(store.zw, refs["wnn"])[name][:],
        )


def test_densmap_on_an_integrated_graph_runs_and_records_standard_umap(
    graph_store,
) -> None:
    store, refs = graph_store
    initial = np.random.default_rng(0).normal(size=(N_CELLS, 2)).astype(np.float32)
    warnings: list[str] = []
    sink = logger.add(
        lambda message: warnings.append(message.record["message"]),
        level="WARNING",
    )
    try:
        requested = store.run_umap(
            refs["wnn"], initial, n_epochs=5, use_density_map=True
        )
    finally:
        logger.remove(sink)

    assert "DensMap is not available for integrated graphs" in " ".join(warnings)
    parameters = store.inspect_artifact(requested).parameters
    assert parameters["use_density_map"] is False
    assert "densmap_algorithm_version" not in parameters
    # The request is the standard embedding, so a standard call reuses it.
    assert store.run_umap(refs["wnn"], initial, n_epochs=5) == requested

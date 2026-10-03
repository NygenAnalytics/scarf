"""Benchmarks of DataStore stages over synthetic local Zarr stores.

The stages mirror rows of the stage table in
``docs/source/concepts/benchmarks.md``, so a projected delay reads directly
against a recorded production stage. Every stage runs with one thread and
``invalidate_cache=True``, so each repeat recomputes and rewrites its
artifact.

At these sizes a stage's fixed overhead (artifact transactions, validation,
provenance, and Zarr metadata) is comparable to its per-cell work, so the
stages fit a fixed overhead plus a per-cell rate instead of a power law, and
each part is compared with the baseline on its own.
"""

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from scarf.storage.artifacts import ArtifactRef
from scarf.storage.feature_selection import read_feature_selection_indices
from scarf.storage.selections import read_stored_selection_indices
from tests.storage_helpers import write_count_store

from . import inputs
from .harness import Ladder, Thresholds

pytestmark = pytest.mark.benchmark

STAGES = Ladder(sizes=(4_000, 8_000, 16_000, 32_000), smoke=600, model="linear")
# Stages whose per-cell work stays below timing noise up to 32k cells are
# timed per call at the largest size, as overhead benchmarks.
OVERHEAD = Ladder(sizes=(32_000,), smoke=600, unit="call", targets=(1,), model="fixed")
OVERHEAD_THRESHOLDS = Thresholds(meaningful_delay=0.25)
# The smallest stores sample too few cells for the production funnel's 1,000
# K-means centroids.
N_CENTROIDS = 50
# The production funnel runs 300 UMAP epochs.
UMAP_EPOCHS = 30


@dataclass(frozen=True)
class Pipeline:
    """One synthetic store and the artifacts of its stages, built once."""

    path: Path
    store: object
    cells: ArtifactRef
    hvgs: ArtifactRef
    normalized: ArtifactRef
    pca: ArtifactRef
    initialization: ArtifactRef
    ann: ArtifactRef
    neighbors: ArtifactRef
    graph: ArtifactRef
    clusters: ArtifactRef


def _open(path: Path):
    from scarf import DataStore

    return DataStore(str(path), default_assay="RNA", nthreads=1)


def _build(path: Path, n_cells: int) -> Pipeline:
    counts, _labels = inputs.labelled_counts(n_cells)
    write_count_store(str(path), {"RNA": counts}, "uint16")
    store = _open(path)
    cells = store.auto_filter_cells()
    hvgs = store.select_hvgs(
        cells, top_n=inputs.N_HVGS, show_plot=False, bin_strategy="fixed"
    )
    normalized = store.run_normalization(cells, store.resolve_features("RNA", hvgs))
    pca = store.run_pca(normalized, dims=inputs.N_DIMS)
    initialization = store.build_embedding_initialization(pca, n_centroids=N_CENTROIDS)
    ann = store.build_ann_index(pca)
    neighbors = store.query_neighbors(ann, coordinates=pca, k=inputs.N_NEIGHBORS)
    graph = store.build_connectivity_map(neighbors)
    clusters = store.run_leiden_clustering(graph)
    return Pipeline(
        path,
        store,
        cells,
        hvgs,
        normalized,
        pca,
        initialization,
        ann,
        neighbors,
        graph,
        clusters,
    )


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory) -> Callable[[int], Pipeline]:
    """Return the pipeline of a size, building its store on first use."""
    root = tmp_path_factory.mktemp("stage_benchmarks")
    built: dict[int, Pipeline] = {}

    def get(n_cells: int) -> Pipeline:
        if n_cells not in built:
            built[n_cells] = _build(root / f"cells_{n_cells}.zarr", n_cells)
        return built[n_cells]

    return get


def _cell_indices(store, selection: ArtifactRef) -> np.ndarray:
    return read_stored_selection_indices(
        store.zw,
        selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )


def _feature_indices(store, selection: ArtifactRef) -> np.ndarray:
    return read_feature_selection_indices(store.zw, "RNA", selection)


def test_stage_filter_cells(bench, pipeline) -> None:
    def make(n_cells: int):
        store = pipeline(n_cells).store
        return lambda: store.auto_filter_cells(invalidate_cache=True)

    def check(n_cells: int, cells: ArtifactRef) -> None:
        kept = _cell_indices(pipeline(n_cells).store, cells).size
        # MAD bounds trim the tails of the drawn library sizes, not the bulk.
        assert 0.8 * n_cells < kept < n_cells

    bench(
        "stage.filter_cells",
        make,
        OVERHEAD,
        check=check,
        thresholds=OVERHEAD_THRESHOLDS,
    )


def test_stage_highly_variable_features(bench, pipeline) -> None:
    def make(n_cells: int):
        built = pipeline(n_cells)
        return lambda: built.store.select_hvgs(
            built.cells,
            top_n=inputs.N_HVGS,
            show_plot=False,
            bin_strategy="fixed",
            invalidate_cache=True,
        )

    def check(n_cells: int, hvgs: ArtifactRef) -> None:
        store = pipeline(n_cells).store
        indices = _feature_indices(store, store.resolve_features("RNA", hvgs))
        counts, _labels = inputs.labelled_counts(n_cells)
        assert indices.size == inputs.N_HVGS
        # Selected genes vary more across the drawn programs than the rest.
        dispersion = counts.var(axis=0) / np.maximum(counts.mean(axis=0), 1e-12)
        rest = np.setdiff1d(np.arange(inputs.N_GENES), indices)
        assert np.median(dispersion[indices]) > np.median(dispersion[rest])

    bench("stage.highly_variable_features", make, STAGES, check=check)


def test_stage_normalization(bench, pipeline) -> None:
    def make(n_cells: int):
        built = pipeline(n_cells)
        features = built.store.resolve_features("RNA", built.hvgs)
        return lambda: built.store.run_normalization(
            built.cells, features, invalidate_cache=True
        )

    def check(n_cells: int, normalized: ArtifactRef) -> None:
        assert normalized.kind == pipeline(n_cells).normalized.kind

    bench("stage.normalization", make, STAGES, check=check)


def test_stage_pca(bench, pipeline) -> None:
    def make(n_cells: int):
        built = pipeline(n_cells)
        return lambda: built.store.run_pca(
            built.normalized, dims=inputs.N_DIMS, invalidate_cache=True
        )

    def check(n_cells: int, pca: ArtifactRef) -> None:
        assert pca.kind == "reduction"

    bench("stage.pca", make, STAGES, check=check)


def test_stage_embedding_initialization(bench, pipeline) -> None:
    def make(n_cells: int):
        built = pipeline(n_cells)
        return lambda: built.store.build_embedding_initialization(
            built.pca, n_centroids=N_CENTROIDS, invalidate_cache=True
        )

    bench(
        "stage.embedding_initialization",
        make,
        OVERHEAD,
        thresholds=OVERHEAD_THRESHOLDS,
    )


def test_stage_ann_index(bench, pipeline) -> None:
    def make(n_cells: int):
        built = pipeline(n_cells)
        return lambda: built.store.build_ann_index(built.pca, invalidate_cache=True)

    bench("stage.ann_index", make, STAGES)


def test_stage_query_neighbors(bench, pipeline) -> None:
    def make(n_cells: int):
        built = pipeline(n_cells)
        return lambda: built.store.query_neighbors(
            built.ann,
            coordinates=built.pca,
            k=inputs.N_NEIGHBORS,
            invalidate_cache=True,
        )

    bench("stage.query_neighbors", make, STAGES)


def test_stage_connectivity(bench, pipeline) -> None:
    def make(n_cells: int):
        built = pipeline(n_cells)
        return lambda: built.store.build_connectivity_map(
            built.neighbors, invalidate_cache=True
        )

    def check(n_cells: int, graph: ArtifactRef) -> None:
        built = pipeline(n_cells)
        matrix = built.store.load_graph(graph)
        kept = _cell_indices(built.store, built.cells).size
        assert matrix.shape == (kept, kept)
        np.testing.assert_array_equal(
            np.diff(matrix.indptr), np.full(kept, inputs.N_NEIGHBORS)
        )

    bench("stage.connectivity", make, STAGES, check=check)


def test_stage_umap(bench, pipeline) -> None:
    def make(n_cells: int):
        built = pipeline(n_cells)
        return lambda: built.store.run_umap(
            built.graph,
            built.initialization,
            n_epochs=UMAP_EPOCHS,
            invalidate_cache=True,
        )

    ladder = Ladder(
        sizes=STAGES.sizes, smoke=STAGES.smoke, work=300 / UMAP_EPOCHS, model="linear"
    )
    bench("stage.umap", make, ladder)


def test_stage_leiden(bench, pipeline) -> None:
    def make(n_cells: int):
        built = pipeline(n_cells)
        return lambda: built.store.run_leiden_clustering(
            built.graph, invalidate_cache=True
        )

    bench("stage.leiden", make, STAGES)


def test_stage_marker_search(bench, pipeline) -> None:
    def make(n_cells: int):
        built = pipeline(n_cells)
        features = built.store.resolve_features("RNA", built.hvgs)
        return lambda: built.store.run_marker_search(
            built.clusters, features=features, invalidate_cache=True
        )

    bench("stage.marker_search", make, STAGES)


def _write_h5ad(path: Path, n_cells: int) -> Path:
    """Write CSR float32 counts with the H5AD layout of a CELLxGENE download."""
    import h5py
    from scipy.sparse import csr_matrix

    counts, _labels = inputs.labelled_counts(n_cells)
    matrix = csr_matrix(counts.astype(np.float32))
    names = np.array([f"gene{index}".encode() for index in range(inputs.N_GENES)])
    with h5py.File(path, "w") as h5:
        group = h5.create_group("X")
        group.attrs["encoding-type"] = "csr_matrix"
        group.attrs["shape"] = counts.shape
        group.create_dataset("data", data=matrix.data)
        group.create_dataset("indices", data=matrix.indices)
        group.create_dataset("indptr", data=matrix.indptr.astype(np.int64))
        h5.create_group("obs").create_dataset(
            "_index",
            data=np.array([f"cell{index}".encode() for index in range(n_cells)]),
        )
        var = h5.create_group("var")
        var.create_dataset("_index", data=names)
        var.create_dataset("feature_name", data=names)
        h5.create_group("obsm")
    return path


def test_stage_create_count_store(bench, tmp_path) -> None:
    """Import H5AD counts into a new store, as the funnel's first stage does."""
    from scarf.readers import H5adReader
    from scarf.writers import H5adToZarr

    sources: dict[int, Path] = {}
    created: list[Path] = []

    def make(n_cells: int):
        if n_cells not in sources:
            sources[n_cells] = _write_h5ad(tmp_path / f"cells_{n_cells}.h5ad", n_cells)
        for previous in created:
            shutil.rmtree(previous, ignore_errors=True)
        target = tmp_path / f"store_{len(created)}.zarr"
        created.append(target)

        def call() -> Path:
            reader = H5adReader(str(sources[n_cells]))
            try:
                H5adToZarr(reader, str(target), nthreads=1).dump()
            finally:
                reader.close()
            return target

        return call

    def check(n_cells: int, target: Path) -> None:
        import zarr

        counts, _labels = inputs.labelled_counts(n_cells)
        stored = zarr.open_group(str(target), mode="r")["RNA/counts"]
        np.testing.assert_array_equal(stored[:], counts)

    bench("stage.create_count_store", make, STAGES, check=check, fresh=True)


def test_stage_write_counts_t(bench, pipeline) -> None:
    """Rewrite the feature-major countsT copy of an imported store."""
    import zarr

    from scarf.assay.classification import default_feature_sets
    from scarf.storage.budget import resolve_budget
    from scarf.storage.sharding import write_counts_t

    def make(n_cells: int):
        group = zarr.open_group(str(pipeline(n_cells).path), mode="r+")["RNA"]
        counts = group["counts"]
        return lambda: write_counts_t(
            counts,
            group,
            resources=resolve_budget(workers=1),
            overwrite=True,
            featureSets=default_feature_sets(group),
        )

    def check(n_cells: int, counts_t) -> None:
        counts, _labels = inputs.labelled_counts(n_cells)
        np.testing.assert_array_equal(counts_t[:], counts.T)

    bench("stage.write_counts_t", make, STAGES, check=check)

"""Resident memory of Harmony, WNN, SNN, and the ANN index, admitted before use.

Harmony holds several float64 copies of the coordinates and matrices of
assignments per cell, WNN holds every assay's neighbors and coordinates, the
SNN merge holds every graph with a float64 matrix of shared-neighbor
fractions per graph, and an hnswlib index holds every cell. Pure estimators in
the domain packages bound what each holds, calibrated against tracemalloc
peaks or, for hnswlib, glibc's count of allocated bytes, and the datastore
operations compare them with the memory budget before a coordinate or graph
is read or an index is created or loaded. Coordinate streams reserve only the
blocks in flight, so their consumers hold no block while the next is read.
"""

import ctypes
import inspect
import re
import shutil
import sys
import threading
from collections.abc import Callable
from contextlib import nullcontext
import tracemalloc
import weakref
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits

from scarf import DataStore
from scarf.datastore._operations.graph import _ann_index_transfer_bytes
from scarf.embeddings.harmony import fit_harmony
from scarf.embeddings.harmony.api import harmony_cluster_count, harmony_peak_bytes
from scarf.neighbors.graph import merge_graphs, snn_merge_peak_bytes
from scarf.neighbors.index import (
    ann_index_file_bytes,
    ann_index_peak_bytes,
    ann_query_block_bytes,
    instantiate_knn_index,
)
from scarf.neighbors.integration import _wnn_integration_many, wnn_peak_bytes
from scarf.neighbors.stages import (
    AnnIndexStage,
    ChunkedCoordinateStream,
    NeighborQueryStage,
)
from scarf.storage.ann_index import load_ann_index, save_ann_index
from scarf.storage.artifacts import artifact_group, fingerprint_text_blocks
from scarf.storage.budget import ResourceBudget
from scarf.storage.execution import ExecutionReport, execution_report_scope
from scarf.storage.profiles import resolve_storage_profile
from tests.storage_helpers import write_count_store

N_CELLS = 120


# Estimators


def _traced_peak(run) -> int:
    tracemalloc.start()
    try:
        run()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak


def _batches(n_cells: int, levels: int, columns: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            f"batch{column}": np.asarray(
                [f"b{(cell // (column + 1)) % levels}" for cell in range(n_cells)],
                dtype=object,
            )
            for column in range(columns)
        }
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    ("n_cells", "dims", "nclust", "levels", "columns", "block_size"),
    [
        (4_000, 10, 20, 2, 1, 0.05),
        (4_000, 5, 10, 40, 1, 1.0),
        (4_000, 20, 30, 6, 2, 0.3),
        # Batch levels beyond the cells make the square ridge matrices dominate.
        (400, 5, 2, 400, 4, 0.05),
    ],
    ids=["few-levels", "one-update-block", "two-columns", "many-levels"],
)
def test_harmony_peak_bytes_bounds_the_traced_peak(
    n_cells, dims, nclust, levels, columns, block_size
) -> None:
    rng = np.random.default_rng(11)
    source = rng.normal(size=(n_cells, dims)).astype(np.float32)
    batches = _batches(n_cells, levels, columns)
    n_levels = sum(int(batches[column].nunique()) for column in batches.columns)
    options = {
        "nclust": nclust,
        "block_size": block_size,
        "max_iter_harmony": 1,
        "max_iter_kmeans": 2,
    }

    def fit() -> None:
        # The stage fills a float64 input matrix from the coordinate stream.
        uncorrected = np.empty((dims, n_cells), dtype=np.float64)
        uncorrected[:] = source.T
        fit_harmony(uncorrected, batches, **options)

    with threadpool_limits(limits=1):
        # Import and warm scikit-learn and the optimizer outside the trace.
        fit_harmony(
            rng.normal(size=(dims, 300)),
            batches.iloc[:300].reset_index(drop=True),
            nclust=3,
            max_iter_harmony=1,
        )
        peak = _traced_peak(fit)

    estimate = harmony_peak_bytes(
        n_cells,
        dims,
        harmony_cluster_count(n_cells, nclust),
        n_levels,
        block_size=block_size,
        nthreads=1,
    )
    # The estimate also counts scikit-learn's per-thread buffers, which the
    # trace misses.
    assert peak <= estimate <= 2 * peak


def test_harmony_cluster_count_is_the_default_of_fit_harmony() -> None:
    # The cell count over 30, rounded half to even, kept between 1 and 100.
    assert [harmony_cluster_count(n) for n in (10, 45, 75, 105, 3_000, 9_000)] == [
        1,
        2,
        2,
        4,
        100,
        100,
    ]
    assert harmony_cluster_count(10, 7) == 7
    for nclust in (0, 11):
        with pytest.raises(ValueError, match="between one and the cell count"):
            harmony_cluster_count(10, nclust)
    values = np.random.default_rng(2).normal(size=(3, 75))
    result = fit_harmony(values, _batches(75, 2), max_iter_harmony=1)
    assert result.parameters["nclust"] == harmony_cluster_count(75) == 2
    # The estimate assumes the update blocks of fit_harmony's default.
    assert (
        inspect.signature(harmony_peak_bytes).parameters["block_size"].default
        == inspect.signature(fit_harmony).parameters["block_size"].default
    )


def test_estimators_refuse_shapes_they_cannot_size() -> None:
    # hnswlib spreads each cell's upper-level links over m - 1 levels.
    with pytest.raises(ValueError, match="m must be at least 2"):
        ann_index_peak_bytes(N_CELLS, 5, 1)
    with pytest.raises(ValueError, match="two or more modalities"):
        wnn_peak_bytes(N_CELLS, [6], [5])
    for block_size, error in (
        (0, ValueError),
        (-0.5, ValueError),
        (np.inf, ValueError),
        (True, TypeError),
    ):
        with pytest.raises(error, match="block_size must be"):
            harmony_peak_bytes(N_CELLS, 5, 4, 2, block_size=block_size)


def _knn(n_cells: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Return ``k`` distinct neighbors of every cell, without the cell itself."""
    indices = np.empty((n_cells, k), dtype=np.uint32)
    for cell in range(n_cells):
        choice = rng.choice(n_cells - 1, size=k, replace=False)
        choice[choice >= cell] += 1
        indices[cell] = choice
    return indices


@pytest.mark.slow
@pytest.mark.parametrize(
    ("n_cells", "neighbor_counts", "dims"),
    [
        (3_000, (15, 10, 8), (30, 5, 12)),
        (2_000, (40, 5), (10, 10)),
    ],
    ids=["three-assays", "unequal-neighbors"],
)
def test_wnn_peak_bytes_bounds_the_traced_peak(
    monkeypatch, n_cells, neighbor_counts, dims
) -> None:
    # Setting a thread limit scans the process's loaded libraries, a
    # transient that grows with them, not with the data, and that the
    # estimate leaves out; these shapes are too small to outgrow it.
    monkeypatch.setattr(
        "threadpoolctl.threadpool_limits", lambda **_options: nullcontext()
    )
    rng = np.random.default_rng(5)
    stored = [
        (_knn(n_cells, k, rng), rng.normal(size=(n_cells, d)).astype(np.float32))
        for k, d in zip(neighbor_counts, dims, strict=True)
    ]
    _wnn_integration_many(
        [
            (f"warm{index}", _knn(200, k, rng), rng.normal(size=(200, d)))
            for index, (k, d) in enumerate(zip(neighbor_counts, dims, strict=True))
        ],
        1,
    )

    def integrate() -> None:
        # The operation loads every assay's neighbors and coordinates.
        modalities = [
            (f"assay{index}", indices.copy(), coordinates.copy())
            for index, (indices, coordinates) in enumerate(stored)
        ]
        _wnn_integration_many(modalities, 1)

    peak = _traced_peak(integrate)

    estimate = wnn_peak_bytes(
        n_cells, neighbor_counts, dims, index_itemsize=4, coordinate_itemsize=4
    )
    assert peak <= estimate <= 2 * peak


def test_wnn_row_norms_sum_bounded_bands_of_every_row(monkeypatch) -> None:
    from scarf.neighbors import integration

    values = np.random.default_rng(4).normal(size=(50_000, 10)).astype(np.float32)
    expected = 1 / np.linalg.norm(values.astype(np.float64), axis=1)
    # Bands of 1,000 rows, whose float64 copies are small beside the norms.
    monkeypatch.setattr(integration, "_NORM_BAND_VALUES", 10_000)
    integration._inverse_row_norms(values[:10])

    peak = _traced_peak(lambda: integration._inverse_row_norms(values))

    # The norms and their inverses, beside two float64 copies of one band.
    assert peak < 3 * values.shape[0] * 8
    np.testing.assert_allclose(
        integration._inverse_row_norms(values), expected, rtol=1e-12
    )


def _stored_graph(
    n_cells: int, k: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.empty((n_cells * k, 2), dtype=np.uint32)
    edges[:, 0] = np.repeat(np.arange(n_cells, dtype=np.uint32), k)
    edges[:, 1] = _knn(n_cells, k, rng).ravel()
    return rng.random(n_cells * k).astype(np.float32), edges


def _loaded_graph(weights: np.ndarray, edges: np.ndarray, n_cells: int) -> csr_matrix:
    # As DataStore._store_to_sparse loads a stored connectivity map.
    weights, edges = weights.copy(), edges.copy()
    return csr_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(n_cells, n_cells))


@pytest.mark.slow
@pytest.mark.parametrize(
    ("n_cells", "k", "n_graphs"),
    [(2_000, 10, 2), (3_000, 5, 4), (20_000, 2, 4)],
    ids=["two-graphs", "four-graphs", "two-neighbors"],
)
def test_snn_merge_peak_bytes_bounds_the_traced_peak(n_cells, k, n_graphs) -> None:
    rng = np.random.default_rng(9)
    stored = [_stored_graph(n_cells, k, rng) for _ in range(n_graphs)]
    merge_graphs([_loaded_graph(*_stored_graph(200, k, rng), 200) for _ in range(2)])

    def merge() -> None:
        merge_graphs([_loaded_graph(*graph, n_cells) for graph in stored])

    peak = _traced_peak(merge)

    estimate = snn_merge_peak_bytes(n_cells, k, n_graphs)
    assert peak <= estimate <= 2 * peak


def _glibc_allocated() -> Callable[[], int] | None:
    """Return a reader of the bytes glibc's malloc has allocated, if any.

    hnswlib allocates its index with malloc, which tracemalloc does not see.
    """
    if not sys.platform.startswith("linux"):
        return None
    try:
        mallinfo2 = ctypes.CDLL("libc.so.6").mallinfo2
    except (OSError, AttributeError):
        return None

    class MallInfo2(ctypes.Structure):
        _fields_ = [
            (name, ctypes.c_size_t)
            for name in (
                "arena",
                "ordblks",
                "smblks",
                "hblks",
                "hblkhd",
                "usmblks",
                "fsmblks",
                "uordblks",
                "fordblks",
                "keepcost",
            )
        ]

    mallinfo2.restype = MallInfo2

    def allocated() -> int:
        info = mallinfo2()
        # Bytes in use in the heap and in chunks mapped on their own.
        return int(info.uordblks + info.hblkhd)

    return allocated


def _allocated_peak(allocated: Callable[[], int], run: Callable[[], Any]) -> int:
    """Return the most bytes allocated beyond the start while ``run`` runs.

    A thread samples the count while ``run`` releases the GIL, and the count
    after it returns covers what it still holds.
    """
    start = allocated()
    peak = start
    done = threading.Event()

    def sample() -> None:
        nonlocal peak
        while not done.wait(0.0002):
            peak = max(peak, allocated())

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        run()
    finally:
        done.set()
        sampler.join()
    return max(peak, allocated()) - start


@pytest.mark.slow
@pytest.mark.parametrize(
    ("n_cells", "dims", "m", "nthreads", "metric"),
    [
        (8_000, 120, 48, 1, "cosine"),
        (20_000, 8, 4, 2, "l2"),
        (5_000, 30, 2, 1, "l2"),
    ],
    ids=["wide-cosine", "two-threads", "minimal-m"],
)
def test_ann_index_peak_bytes_bounds_the_allocated_peak(
    tmp_path, n_cells, dims, m, nthreads, metric
) -> None:
    import hnswlib

    allocated = _glibc_allocated()
    if allocated is None:
        pytest.skip("needs glibc's count of allocated bytes")
    values = np.random.default_rng(7).normal(size=(n_cells, dims))
    values = values.astype(np.float32)
    held: list[Any] = []

    def build() -> None:
        index = instantiate_knn_index(metric, dims, n_cells, 50, m, 4466, 50, nthreads)
        held.append(index)
        for start in range(0, n_cells, 4_096):
            index.add_items(values[start : start + 4_096])

    built = _allocated_peak(allocated, build)
    path = str(tmp_path / "index.bin")
    held.pop().save_index(path)

    def load() -> None:
        index = hnswlib.Index(space=metric, dim=dims)
        held.append(index)
        index.load_index(path)

    loaded = _allocated_peak(allocated, load)

    # The estimate counts the index as built or loaded on this platform.
    estimate = ann_index_peak_bytes(n_cells, dims, m, nthreads=nthreads)
    assert max(built, loaded) <= estimate <= 2 * min(built, loaded)
    # The saved file holds the level-0 block and the upper levels' links,
    # whose sizes vary with the random levels of the cells.
    saved = (tmp_path / "index.bin").stat().st_size
    assert abs(ann_index_file_bytes(n_cells, dims, m) - saved) <= 0.01 * saved


@pytest.mark.slow
def test_ann_query_block_bytes_bounds_the_allocated_peak() -> None:
    allocated = _glibc_allocated()
    if allocated is None:
        pytest.skip("needs glibc's count of allocated bytes")
    rng = np.random.default_rng(2)
    values = rng.normal(size=(30_000, 16)).astype(np.float32)
    index = instantiate_knn_index("l2", 16, 30_000, 40, 16, 1, 40, 1)
    index.add_items(values)
    # A deep search on several threads holds candidate queues per thread.
    for k, rows, ef, threads in (
        (10, 20_000, 40, 1),
        (30, 12_000, 40, 1),
        (10, 2_000, 4_000, 8),
    ):
        index.set_ef(ef)
        index.set_num_threads(threads)
        query = NeighborQueryStage(index, k, "l2")
        block = values[:rows].copy()

        def run() -> None:
            # As query_neighbors queries and checks one block.
            indices, _distances, _missed = query.query(
                block, self_indices=np.arange(rows)
            )
            assert not (np.any(indices < 0) or np.any(indices >= 30_000))

        # A thread samples the count, so under load one run can miss the
        # short peak of a query; the largest of three is kept.
        peak = max(_allocated_peak(allocated, run) for _ in range(3))
        estimate = ann_query_block_bytes(rows, k, ef=ef, nthreads=threads)
        assert peak <= estimate <= 2 * peak, (k, rows, ef, threads, peak, estimate)


@pytest.mark.slow
def test_ann_index_transfer_bytes_bound_the_traced_copies(tmp_path) -> None:
    import zarr

    n_cells, dims = 10_000, 32
    values = np.random.default_rng(3).normal(size=(n_cells, dims)).astype(np.float32)
    index = instantiate_knn_index("l2", dims, n_cells, 20, 16, 1, 20, 1)
    index.add_items(values)
    root = zarr.open_group(str(tmp_path / "store.zarr"), mode="w")
    profile = resolve_storage_profile(root.store)
    small = instantiate_knn_index("l2", dims, 50, 20, 16, 1, 20, 1)
    small.add_items(values[:50])
    # Import and warm the codecs outside the trace.
    save_ann_index(
        root.create_group("warm"),
        small,
        profile=profile,
        metric="l2",
        dimensions=dims,
        element_count=50,
    )
    load_ann_index(root["warm"], "l2", dims, expected_count=50)
    group = root.create_group("index")

    def save() -> None:
        save_ann_index(
            group,
            index,
            profile=profile,
            metric="l2",
            dimensions=dims,
            element_count=n_cells,
        )

    saving = _traced_peak(save)
    payload = int(group["ann_idx_bytes"].shape[0])
    loading = _traced_peak(lambda: load_ann_index(group, "l2", dims))

    # The payload fits one Zarr chunk, so a save holds about five copies of
    # it (seven with Zarr 3.2) and a load about three.
    bound = _ann_index_transfer_bytes(payload)
    assert max(saving, loading) <= bound <= 2 * max(saving, loading)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("n_cells", "dims", "clusters", "working_mib"),
    [(10_000, 50, 200, 4), (5_000, 20, 10, 1)],
    ids=["many-clusters", "small-chunks"],
)
def test_silhouette_working_memory_factor_bounds_the_traced_peak(
    n_cells, dims, clusters, working_mib
) -> None:
    from sklearn import config_context
    from sklearn.metrics import silhouette_score

    from scarf.datastore._pipeline_cluster_selection import (
        _SILHOUETTE_WORKING_MEMORY_FACTOR,
    )

    rng = np.random.default_rng(1)
    values = rng.normal(size=(n_cells, dims))
    labels = rng.integers(0, clusters, size=n_cells)
    with config_context(working_memory=working_mib):
        silhouette_score(values[:200], labels[:200])

        # Cluster selection scores its float64 sample, which it counts on its
        # own, in distance chunks of its working memory.
        peak = _traced_peak(
            lambda: silhouette_score(values, labels, metric="euclidean")
        )

    allowance = _SILHOUETTE_WORKING_MEMORY_FACTOR * working_mib * 1024**2
    assert peak <= allowance <= 2 * peak


# Admission


def _counts() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(23)
    rna = rng.poisson(rng.gamma(1.0, 3.0, size=30), size=(N_CELLS, 30))
    rna[: N_CELLS // 2, :5] += 4
    rna[:, 0] += 1
    return {"RNA": rna, "ADT": rng.poisson(20.0, size=(N_CELLS, 4)) + 1}


def _open(zarr_loc: Path, **options: Any) -> DataStore:
    return DataStore(
        str(zarr_loc),
        default_assay="RNA",
        min_features_per_cell=0,
        nthreads=1,
        **options,
    )


def _write_template(zarr_loc: Path) -> dict[str, Any]:
    write_count_store(str(zarr_loc), _counts(), "uint16")
    store = _open(zarr_loc)
    store.cells.insert("batch", np.where(np.arange(N_CELLS) % 2, "a", "b"))
    cells = store.snapshot_cell_selection()
    refs: dict[str, Any] = {"cells": cells}
    for assay, dims in (("RNA", 5), ("ADT", 3)):
        normalized = store.run_normalization(
            cells, store.select_all_features(from_assay=assay)
        )
        pca = store.run_pca(normalized, dims=dims)
        neighbors = store.query_neighbors(store.build_ann_index(pca), k=6)
        refs[assay] = {
            "pca": pca,
            "neighbors": neighbors,
            "graph": store.build_connectivity_map(neighbors),
        }
    return refs


@pytest.fixture(scope="module")
def template(tmp_path_factory) -> tuple[Path, dict[str, Any]]:
    zarr_loc = tmp_path_factory.mktemp("resident_memory") / "store.zarr"
    return zarr_loc, _write_template(zarr_loc)


@pytest.fixture
def memory_store(template, tmp_path) -> tuple[DataStore, dict[str, Any]]:
    zarr_loc, refs = template
    target = tmp_path / "store.zarr"
    shutil.copytree(zarr_loc, target)
    return _open(target), refs


def _record_coordinate_reads(monkeypatch) -> list[str]:
    reads: list[str] = []
    original = ChunkedCoordinateStream.iter_coordinate_blocks

    def recorded(self, message: str):
        reads.append(message)
        yield from original(self, message)

    monkeypatch.setattr(ChunkedCoordinateStream, "iter_coordinate_blocks", recorded)
    return reads


def _split_and_check_blocks(monkeypatch, parts: int = 3) -> list[bool]:
    """Stream each stored block as fresh parts and record held parts.

    Before each part after the first, it records whether the consumer still
    holds the part before it. A read plan reserves only the blocks in flight,
    so a consumer that still holds the previous part exceeds it.
    """
    held: list[bool] = []
    original = ChunkedCoordinateStream.iter_coordinate_blocks

    def checked(self, message: str):
        previous = None
        for block in original(self, message):
            for part in np.array_split(np.asarray(block), parts):
                if previous is not None:
                    held.append(previous() is not None)
                fresh = part.copy()
                previous = weakref.ref(fresh)
                yield fresh
                del fresh

    monkeypatch.setattr(ChunkedCoordinateStream, "iter_coordinate_blocks", checked)
    return held


def _required_bytes(message: str) -> int:
    found = re.search(r"needs about (\d+) bytes", message)
    assert found is not None, message
    return int(found.group(1))


def _reserved_bytes(reports: list[ExecutionReport]) -> list[int]:
    """Return the bytes each coordinate read reserved for what its caller holds."""
    return [
        report.plan.residentBytes
        for report in reports
        if report.unitKind == "countsRowBlock"
    ]


def _batch_label_bytes() -> int:
    """Bytes of the batch labels that run_harmony holds: one text object each."""
    labels = np.where(np.arange(N_CELLS) % 2, "a", "b").astype(object)
    return int(pd.DataFrame({"batch": labels}).memory_usage(deep=True).sum())


def test_harmony_admission_counts_the_batch_labels(memory_store, monkeypatch) -> None:
    store, refs = memory_store
    reads = _record_coordinate_reads(monkeypatch)
    options = {"harmony_params": {"nclust": 10}}
    peak = harmony_peak_bytes(N_CELLS, 5, 10, 2)
    # The fit's own estimate fits, even beside a pointer per label, but not
    # beside the labels the operation reads first and holds through the fit,
    # about 58 bytes per cell here.
    store.resources = ResourceBudget(peak + 16 * N_CELLS, 1)

    with pytest.raises(MemoryError, match="for the batch labels") as caught:
        store.run_harmony(refs["RNA"]["pca"], ["batch"], **options)

    assert _batch_label_bytes() > 50 * N_CELLS
    assert reads == []
    assert store.list_artifacts(kind="batch_correction", from_assay="RNA") == []

    # The admitted bytes are enough, and the coordinate read reserves the
    # float64 matrix that it fills beside the labels.
    store.resources = ResourceBudget(_required_bytes(str(caught.value)), 1)
    with execution_report_scope() as reports:
        corrected = store.run_harmony(refs["RNA"]["pca"], ["batch"], **options)
    assert store.inspect_artifact(corrected).complete
    assert reads == ["Loading uncorrected latent dimensions"]
    assert _reserved_bytes(reports) == [N_CELLS * 5 * 8 + _batch_label_bytes()]


def test_snn_integration_raises_before_loading_graphs_over_budget(
    memory_store, monkeypatch
) -> None:
    store, refs = memory_store
    loads: list[str] = []
    monkeypatch.setattr(
        store, "_store_to_sparse", lambda location, _use_k: loads.append(location)
    )
    peak = snn_merge_peak_bytes(N_CELLS, 6, 2)
    store.resources = ResourceBudget(peak - 1, 1)

    with pytest.raises(MemoryError) as caught:
        store.integrate_assays(
            [refs["RNA"]["graph"], refs["ADT"]["graph"]], method="snn"
        )

    message = str(caught.value)
    assert _required_bytes(message) == peak
    assert f"2 graphs over {N_CELLS} cells with 6 neighbors" in message
    assert f"operation limit is {peak - 1} bytes" in message
    assert loads == []
    assert store.list_artifacts(scope="datastore", kind="integrated_graph") == []


def test_wnn_integration_raises_before_loading_coordinates_over_budget(
    memory_store, monkeypatch
) -> None:
    store, refs = memory_store
    reads = _record_coordinate_reads(monkeypatch)
    peak = wnn_peak_bytes(N_CELLS, (6, 6), (5, 3))
    store.resources = ResourceBudget(peak - 1, 1)

    with pytest.raises(MemoryError) as caught:
        store.integrate_assays([refs["RNA"]["neighbors"], refs["ADT"]["neighbors"]])

    message = str(caught.value)
    assert _required_bytes(message) == peak
    assert (
        f"2 assays over {N_CELLS} cells with [6, 6] neighbors and [5, 3] dimensions"
        in message
    )
    assert f"operation limit is {peak - 1} bytes" in message
    assert reads == []
    assert store.list_artifacts(scope="datastore", kind="integrated_graph") == []

    # Within the budget it loads each assay's coordinates once, and holds no
    # block while it reads the next.
    store.resources = ResourceBudget(peak, 1)
    held = _split_and_check_blocks(monkeypatch)
    with execution_report_scope() as reports:
        integrated = store.integrate_assays(
            [refs["RNA"]["neighbors"], refs["ADT"]["neighbors"]]
        )
    assert store.inspect_artifact(integrated).complete
    assert reads == ["Loading RNA coordinates", "Loading ADT coordinates"]
    assert held and not any(held)
    # Each read reserves the uint32 neighbors and float32 coordinates of its
    # assay beside those of the assays loaded before it.
    rna = N_CELLS * (6 * 4 + 5 * 4)
    assert _reserved_bytes(reports) == [rna, rna + N_CELLS * (6 * 4 + 3 * 4)]


def _record_index_creations(monkeypatch) -> list[int]:
    created: list[int] = []
    create = AnnIndexStage.create

    def recorded(**options):
        created.append(options["n_cells"])
        return create(**options)

    monkeypatch.setattr(AnnIndexStage, "create", staticmethod(recorded))
    return created


def test_build_ann_index_raises_before_creating_the_index_over_budget(
    memory_store, monkeypatch
) -> None:
    store, refs = memory_store
    pca = refs["RNA"]["pca"]
    created = _record_index_creations(monkeypatch)
    reads = _record_coordinate_reads(monkeypatch)
    before = store.list_artifacts(kind="ann_index", from_assay="RNA")
    index_bytes = ann_index_peak_bytes(N_CELLS, 5, 8, nthreads=1)
    store.resources = ResourceBudget(index_bytes, 1)

    with pytest.raises(MemoryError) as caught:
        store.build_ann_index(pca, ann_m=8)

    message = str(caught.value)
    required = _required_bytes(message)
    # The index with one block of coordinates, or with the copy that saves
    # it through a temporary file.
    saving = _ann_index_transfer_bytes(ann_index_file_bytes(N_CELLS, 5, 8))
    assert required >= index_bytes + saving
    for size in (
        f"ANN index of {N_CELLS} cells with 5 dimensions",
        f"{index_bytes} of them for the hnswlib index with ann_m=8",
        f"operation limit is {index_bytes} bytes",
    ):
        assert size in message
    assert created == [] and reads == []
    assert store.list_artifacts(kind="ann_index", from_assay="RNA") == before

    # The admitted bytes are enough: the reads plan their blocks beside the
    # index, and the index is saved.
    store.resources = ResourceBudget(required, 1)
    with execution_report_scope() as reports:
        built = store.build_ann_index(pca, ann_m=8)
    assert store.inspect_artifact(built).complete
    assert created == [N_CELLS]
    assert reads == ["Fitting ANN"]
    assert _reserved_bytes(reports) == [index_bytes]


def test_query_neighbors_raises_before_loading_the_index_over_budget(
    memory_store, monkeypatch
) -> None:
    import scarf.datastore._operations.graph as graph_operations

    store, refs = memory_store
    ann_index = store.inspect_artifact(refs["RNA"]["neighbors"]).input_ref("ann_index")
    loads: list[str] = []
    load = graph_operations.load_ann_index

    def recorded(group, *args, **kwargs):
        loads.append(group.path)
        return load(group, *args, **kwargs)

    monkeypatch.setattr(graph_operations, "load_ann_index", recorded)
    reads = _record_coordinate_reads(monkeypatch)
    # build_ann_index's default ann_m.
    index_bytes = ann_index_peak_bytes(N_CELLS, 5, 48, nthreads=1)
    store.resources = ResourceBudget(index_bytes, 1)

    with pytest.raises(MemoryError) as caught:
        store.query_neighbors(ann_index, k=7)

    message = str(caught.value)
    required = _required_bytes(message)
    # The index and the neighbor results, beside one block with its query
    # results or the copy that loads the index through a temporary file.
    payload = int(artifact_group(store.zw, ann_index)["ann_idx_bytes"].shape[0])
    assert required >= index_bytes + N_CELLS * 7 * 8 + max(
        ann_query_block_bytes(N_CELLS, 7, ef=50, nthreads=1),
        _ann_index_transfer_bytes(payload),
    )
    for size in (
        f"Querying 7 neighbors of {N_CELLS} cells with 5 dimensions",
        f"{index_bytes} of them for the hnswlib index with ann_m=48",
        f"operation limit is {index_bytes} bytes",
    ):
        assert size in message
    assert loads == [] and reads == []
    assert len(store.list_artifacts(kind="neighbors", from_assay="RNA")) == 1

    # Within the budget it holds no block while it reads the next.
    store.resources = ResourceBudget(required, 1)
    held = _split_and_check_blocks(monkeypatch)
    with execution_report_scope() as reports:
        neighbors = store.query_neighbors(ann_index, k=7)
    assert store.inspect_artifact(neighbors).complete
    assert len(loads) == 1
    assert reads == ["Identifying neighbors"]
    assert held and not any(held)
    scratch = ann_query_block_bytes(N_CELLS, 7, ef=50, nthreads=1)
    assert _reserved_bytes(reports) == [index_bytes + N_CELLS * 7 * 8 + scratch]


# Identity checks of the cell IDs


def test_text_fingerprints_hash_fixed_width_bands_without_a_copy() -> None:
    ids = np.asarray([f"cell{index:07d}" for index in range(200_000)])
    assert ids.dtype == np.dtype("<U11")

    peak = _traced_peak(
        lambda: fingerprint_text_blocks(ids.size, ids.dtype, lambda: iter((ids,)))
    )

    # Before, each band was copied once more to its own fixed-width dtype.
    assert peak < ids.nbytes // 8

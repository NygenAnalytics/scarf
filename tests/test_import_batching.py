from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import zarr
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
from zarr.storage import MemoryStore

from scarf.readers import CrH5Reader, H5adReader
from scarf.storage.budget import ResourceBudget
from scarf.storage.layout import array_shard_rows
from scarf.storage.sharding import (
    resolve_sparse_import_batch,
    sparse_write_task_count,
)
from scarf.writers import CrToZarr, H5adToZarr, SparseToZarr


def _planner_destinations() -> tuple[zarr.Array, zarr.Array]:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    first = root.create_array(
        "first",
        shape=(12, 4),
        chunks=(2, 4),
        shards=(4, 4),
        dtype=np.uint32,
    )
    second = root.create_array(
        "second",
        shape=(12, 2),
        chunks=(3, 2),
        shards=(6, 2),
        dtype=np.uint32,
    )
    return first, second


def test_sparse_import_planner_prefers_geometry_and_shrinks_monotonically() -> None:
    destinations = _planner_destinations()
    large = resolve_sparse_import_batch(
        destinations,
        nRows=12,
        resources=ResourceBudget(15_000, 2),
        maxWindowNnz=lambda rows: min(rows, 12) * 4,
        sourceDtype=np.uint32,
        producerStagingBytes=lambda rows: rows * 1_000,
    )
    small = resolve_sparse_import_batch(
        destinations,
        nRows=12,
        resources=ResourceBudget(10_000, 2),
        maxWindowNnz=lambda rows: min(rows, 12) * 4,
        sourceDtype=np.uint32,
        producerStagingBytes=lambda rows: rows * 1_000,
    )
    wider = resolve_sparse_import_batch(
        destinations,
        nRows=12,
        resources=ResourceBudget(15_000, 2),
        maxWindowNnz=lambda rows: min(rows, 12) * 8,
        sourceDtype=np.uint32,
        producerStagingBytes=lambda rows: rows * 1_000,
    )

    assert large.batchRows == 4
    assert large.producerReserveBytes == 8_272
    assert large.writeTasks == 5
    assert small.batchRows == 1
    assert small.producerReserveBytes < large.producerReserveBytes
    assert wider.batchRows < large.batchRows


def test_sparse_import_planner_preserves_override_and_fails_one_band() -> None:
    destinations = _planner_destinations()
    explicit = resolve_sparse_import_batch(
        destinations,
        nRows=12,
        resources=ResourceBudget(1_000_000, 2),
        maxWindowNnz=lambda rows: min(rows, 12) * 4,
        sourceDtype=np.uint32,
        batchRows=7,
    )
    assert explicit.batchRows == 7
    with pytest.raises(ValueError, match="batch_size must be positive"):
        resolve_sparse_import_batch(
            destinations,
            nRows=12,
            resources=ResourceBudget(1_000_000, 2),
            maxWindowNnz=lambda rows: min(rows, 12) * 4,
            sourceDtype=np.uint32,
            batchRows=0,
        )
    with pytest.raises(MemoryError, match="one source row"):
        resolve_sparse_import_batch(
            destinations,
            nRows=12,
            resources=ResourceBudget(1_000, 1),
            maxWindowNnz=lambda rows: min(rows, 12) * 4,
            sourceDtype=np.uint32,
        )


class _PlanningReader:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values
        self.nCells, self.nFeatures = values.shape
        self.matrix_dtype = values.dtype
        self.assayFeats = pd.DataFrame(
            {
                "RNA": [
                    "Gene Expression",
                    0,
                    self.nFeatures,
                    self.nFeatures,
                ]
            },
            index=["type", "start", "end", "nFeatures"],
        )
        self.consumed: list[int] = []
        self.windowRequests: list[int] = []

    def cell_names(self) -> list[str]:
        return [f"cell-{index}" for index in range(self.nCells)]

    def feature_ids(self, assay: str) -> list[str]:
        return [f"feature-{index}" for index in range(self.nFeatures)]

    def feature_names(self, assay: str) -> list[str]:
        return [f"gene-{index}" for index in range(self.nFeatures)]

    def consume(self, batch_size: int, lines_in_mem: int):
        self.consumed.append(batch_size)
        for start in range(0, self.nCells, batch_size):
            yield coo_matrix(self.values[start : start + batch_size])

    def max_window_nnz(self, window_rows: int) -> int:
        self.windowRequests.append(window_rows)
        width = min(window_rows, self.nCells)
        return max(
            np.count_nonzero(self.values[start : start + width])
            for start in range(self.nCells - width + 1)
        )

    def producer_staging_bytes(self, batch_size: int, lines_in_mem: int) -> int:
        return 0


def test_crtozarr_automatic_rows_and_preflight_before_consume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scarf.storage import sharding

    original_write = sharding.write_sparse_bands
    observed: dict[str, int] = {}

    def recording_write(writes, **kwargs):
        count = 0

        def recording_source():
            nonlocal count
            for write in writes:
                count += 1
                yield write

        original_write(recording_source(), **kwargs)
        observed["writes"] = count
        observed["total"] = int(kwargs["total"])

    monkeypatch.setattr(sharding, "write_sparse_bands", recording_write)
    values = np.eye(8, 4, dtype=np.uint16)
    reader = _PlanningReader(values)
    writer = CrToZarr(
        reader,
        MemoryStore(),
        dtype="uint16",
        mem_budget="64M",
        targetChunkBytes=16,
        targetShardBytes=32,
    )
    writer.dump()

    assert reader.consumed == [4]
    assert writer._lastImportPlan.batchRows == 4
    assert 4 in reader.windowRequests
    assert observed == {"writes": 2, "total": 2}
    assert writer._lastImportPlan.writeTasks == 2

    rejected = _PlanningReader(values)
    writer = CrToZarr(
        rejected,
        MemoryStore(),
        dtype="uint16",
        mem_budget=1,
        targetChunkBytes=16,
        targetShardBytes=32,
    )
    with pytest.raises(MemoryError, match="one source row"):
        writer.dump()
    assert rejected.consumed == []


class _TrackingCsr(csr_matrix):
    def __init__(self, values: np.ndarray) -> None:
        super().__init__(values)
        self.sliceCount = 0

    def __getitem__(self, key):
        self.sliceCount += 1
        return super().__getitem__(key)


def test_h5ad_and_sparse_preflight_before_source_iteration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = np.eye(8, 4, dtype=np.uint16)
    path = tmp_path / "preflight.h5ad"
    _write_h5ad(path, values, "csr")
    reader = H5adReader(str(path), feature_name_key="feature_name")
    h5ad_writer = H5adToZarr(
        reader,
        MemoryStore(),
        mem_budget="64M",
        targetChunkBytes=16,
        targetShardBytes=32,
    )
    consumed = False

    def unexpected_consume(batch_size: int):
        nonlocal consumed
        consumed = True
        yield coo_matrix(values[:batch_size])

    monkeypatch.setattr(reader, "consume", unexpected_consume)
    h5ad_writer.resources = ResourceBudget(1, 1)
    try:
        with pytest.raises(MemoryError, match="one source row"):
            h5ad_writer.dump()
    finally:
        reader.h5.close()
    assert not consumed

    tracked = _TrackingCsr(values)
    sparse_writer = SparseToZarr(
        tracked,
        MemoryStore(),
        cell_ids=[f"cell-{index}" for index in range(values.shape[0])],
        feature_ids=[f"feature-{index}" for index in range(values.shape[1])],
        mem_budget="64M",
        targetChunkBytes=16,
        targetShardBytes=32,
    )
    sparse_writer.resources = ResourceBudget(1, 1)
    with pytest.raises(MemoryError, match="one source row"):
        sparse_writer.dump()
    assert sparse_writer.mat.sliceCount == 0


def _write_cr_h5(path: Path, values: np.ndarray) -> None:
    matrix = csr_matrix(values)
    with h5py.File(path, mode="w") as handle:
        group = handle.create_group("matrix")
        group.create_dataset("data", data=matrix.data)
        group.create_dataset("indices", data=matrix.indices)
        group.create_dataset("indptr", data=matrix.indptr)
        group.create_dataset(
            "barcodes",
            data=np.array(
                [f"cell-{index}".encode() for index in range(values.shape[0])]
            ),
        )
        features = group.create_group("features")
        features.create_dataset(
            "id",
            data=np.array(
                [f"feature-{index}".encode() for index in range(values.shape[1])]
            ),
        )
        features.create_dataset(
            "name",
            data=np.array(
                [f"gene-{index}".encode() for index in range(values.shape[1])]
            ),
        )
        features.create_dataset(
            "feature_type",
            data=np.array(
                [
                    (
                        "Gene Expression" if index % 2 == 0 else "Antibody Capture"
                    ).encode()
                    for index in range(values.shape[1])
                ]
            ),
        )


def _write_h5ad(path: Path, values: np.ndarray, encoding: str) -> None:
    matrix = csr_matrix(values) if encoding == "csr" else csc_matrix(values)
    with h5py.File(path, mode="w") as handle:
        group = handle.create_group("X")
        group.attrs["encoding-type"] = f"{encoding}_matrix"
        group.attrs["shape"] = values.shape
        group.create_dataset("data", data=matrix.data)
        group.create_dataset("indices", data=matrix.indices)
        group.create_dataset("indptr", data=matrix.indptr)
        obs = handle.create_group("obs")
        obs.create_dataset(
            "_index",
            data=np.array(
                [f"cell-{index}".encode() for index in range(values.shape[0])]
            ),
        )
        var = handle.create_group("var")
        var.create_dataset(
            "_index",
            data=np.array(
                [f"feature-{index}".encode() for index in range(values.shape[1])]
            ),
        )
        var.create_dataset(
            "feature_name",
            data=np.array(
                [f"gene-{index}".encode() for index in range(values.shape[1])]
            ),
        )
        var.create_dataset(
            "feature_types",
            data=np.array(
                [
                    (
                        "Gene Expression" if index % 2 == 0 else "Antibody Capture"
                    ).encode()
                    for index in range(values.shape[1])
                ]
            ),
        )
        handle.create_group("obsm")


def _assay_counts(store: MemoryStore) -> dict[str, np.ndarray]:
    root = zarr.open_group(store=store, mode="r")
    return {name: np.asarray(root[f"{name}/counts"][:]) for name in ("RNA", "ADT")}


def _count_arrays(store: MemoryStore) -> tuple[zarr.Array, zarr.Array]:
    root = zarr.open_group(store=store, mode="r")
    return root["RNA/counts"], root["ADT/counts"]


def test_cellranger_h5_automatic_matches_explicit_and_caches_planning(
    tmp_path: Path,
) -> None:
    values = np.arange(48, dtype=np.uint16).reshape(8, 6) % 5
    path = tmp_path / "counts.h5"
    _write_cr_h5(path, values)
    stores = [MemoryStore(), MemoryStore()]
    readers = [CrH5Reader(str(path)), CrH5Reader(str(path))]
    try:
        automatic = CrToZarr(
            readers[0],
            stores[0],
            dtype="uint16",
            mem_budget="64M",
            targetChunkBytes=24,
            targetShardBytes=48,
        )
        explicit = CrToZarr(
            readers[1],
            stores[1],
            dtype="uint16",
            mem_budget="64M",
            targetChunkBytes=24,
            targetShardBytes=48,
        )
        automatic.dump()
        explicit.dump(batch_size=3)
        cached = readers[0]._indptrCache
        assert cached is not None
        readers[0].max_window_nnz(2)
        assert readers[0]._indptrCache is cached
        automatic_arrays = _count_arrays(stores[0])
        explicit_arrays = _count_arrays(stores[1])
        assert automatic._lastImportPlan.batchRows == min(
            array_shard_rows(array) for array in automatic_arrays
        )
        assert automatic._lastImportPlan.writeTasks == sparse_write_task_count(
            automatic_arrays,
            values.shape[0],
        )
        without_projection = resolve_sparse_import_batch(
            automatic_arrays,
            nRows=values.shape[0],
            resources=automatic.resources,
            maxWindowNnz=readers[0].max_window_nnz,
            sourceDtype=readers[0].matrix_dtype,
            batchRows=automatic._lastImportPlan.batchRows,
            producerStagingBytes=lambda rows: readers[0].producer_staging_bytes(
                rows,
                100_000,
            ),
        )
        assert (
            automatic._lastImportPlan.producerReserveBytes
            > without_projection.producerReserveBytes
        )
        assert explicit._lastImportPlan.batchRows == 3
        for automatic_array, explicit_array in zip(
            automatic_arrays,
            explicit_arrays,
            strict=True,
        ):
            assert automatic_array.dtype == explicit_array.dtype == np.dtype(np.uint16)
            assert automatic_array.chunks == explicit_array.chunks
            assert array_shard_rows(automatic_array) == array_shard_rows(explicit_array)
    finally:
        for reader in readers:
            reader.close()

    for name, expected in _assay_counts(stores[1]).items():
        np.testing.assert_array_equal(_assay_counts(stores[0])[name], expected)


@pytest.mark.parametrize("encoding", ["csr", "csc"])
def test_h5ad_automatic_matches_explicit_for_split_assays(
    tmp_path: Path,
    encoding: str,
) -> None:
    values = np.arange(48, dtype=np.uint16).reshape(8, 6) % 7
    path = tmp_path / f"{encoding}.h5ad"
    _write_h5ad(path, values, encoding)
    stores = [MemoryStore(), MemoryStore()]
    readers = [
        H5adReader(str(path), feature_name_key="feature_name"),
        H5adReader(str(path), feature_name_key="feature_name"),
    ]
    try:
        automatic = H5adToZarr(
            readers[0],
            stores[0],
            assay_split_key="feature_types",
            mem_budget="64M",
            targetChunkBytes=24,
            targetShardBytes=48,
        )
        explicit = H5adToZarr(
            readers[1],
            stores[1],
            assay_split_key="feature_types",
            mem_budget="64M",
            targetChunkBytes=24,
            targetShardBytes=48,
        )
        automatic.dump()
        explicit.dump(batch_size=3)
        automatic_arrays = _count_arrays(stores[0])
        explicit_arrays = _count_arrays(stores[1])
        assert automatic._lastImportPlan.batchRows == min(
            array_shard_rows(array) for array in automatic_arrays
        )
        assert automatic._lastImportPlan.writeTasks == sparse_write_task_count(
            automatic_arrays,
            values.shape[0],
        )
        without_projection = resolve_sparse_import_batch(
            automatic_arrays,
            nRows=values.shape[0],
            resources=automatic.resources,
            maxWindowNnz=readers[0].max_batch_nnz,
            sourceDtype=readers[0].sourceMatrixDtype,
            batchRows=automatic._lastImportPlan.batchRows,
            producerStagingBytes=readers[0].producer_batch_staging_bytes,
        )
        assert (
            automatic._lastImportPlan.producerReserveBytes
            > without_projection.producerReserveBytes
        )
        assert explicit._lastImportPlan.batchRows == 3
        for automatic_array, explicit_array in zip(
            automatic_arrays,
            explicit_arrays,
            strict=True,
        ):
            assert automatic_array.dtype == explicit_array.dtype == np.dtype(np.uint16)
            assert automatic_array.chunks == explicit_array.chunks
            assert array_shard_rows(automatic_array) == array_shard_rows(explicit_array)
    finally:
        for reader in readers:
            reader.h5.close()

    for name, expected in _assay_counts(stores[1]).items():
        np.testing.assert_array_equal(_assay_counts(stores[0])[name], expected)


def test_sparse_automatic_matches_explicit() -> None:
    values = csr_matrix(np.arange(48, dtype=np.uint16).reshape(8, 6) % 5)
    stores = [MemoryStore(), MemoryStore()]
    writers = [
        SparseToZarr(
            values,
            store,
            cell_ids=[f"cell-{index}" for index in range(values.shape[0])],
            feature_ids=[f"feature-{index}" for index in range(values.shape[1])],
            mem_budget="64M",
            targetChunkBytes=24,
            targetShardBytes=48,
        )
        for store in stores
    ]
    writers[0].dump()
    writers[1].dump(batch_size=3)

    automatic_array = zarr.open_group(store=stores[0], mode="r")["RNA/counts"]
    explicit_array = zarr.open_group(store=stores[1], mode="r")["RNA/counts"]
    assert writers[0]._lastImportPlan.batchRows == array_shard_rows(automatic_array)
    assert writers[0]._lastImportPlan.writeTasks == sparse_write_task_count(
        (automatic_array,),
        values.shape[0],
    )
    assert writers[1]._lastImportPlan.batchRows == 3
    assert automatic_array.dtype == explicit_array.dtype == np.dtype(np.uint16)
    assert automatic_array.chunks == explicit_array.chunks
    assert array_shard_rows(automatic_array) == array_shard_rows(explicit_array)
    np.testing.assert_array_equal(automatic_array[:], explicit_array[:])

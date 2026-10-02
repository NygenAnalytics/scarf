"""Fitting the count-matrix layout to the import memory budget."""

from pathlib import Path

import h5py
import numpy as np
import pytest
import zarr
from scipy.sparse import csr_matrix

from scarf.readers import H5adReader
from scarf.storage.budget import ResourceBudget
from scarf.storage.count_matrix import (
    DEFAULT_COUNT_MATRIX_POLICY,
    CountMatrixPolicy,
    load_count_matrix_plan,
    policy_from_payload,
)
from scarf.storage.sharding import fit_count_layout
from scarf.writers import H5adToZarr


def _fit(admitCounts, *, transposed=(), memoryBytes=1 << 40, requested=None):
    return fit_count_layout(
        {"RNA": (1_000, np.uint16)},
        nCells=1_000_000,
        profile="fast_local",
        memoryBytes=memoryBytes,
        transposed=transposed,
        admitCounts=admitCounts,
        requested=requested,
    )


def test_fit_uses_an_explicit_policy_exactly():
    explicit = CountMatrixPolicy(unitBytes=64_000, chunkBytes=16_000)
    seen: list[int] = []

    def admit(specs, resources: ResourceBudget) -> None:
        seen.append(specs[0].shards[0])

    assert _fit(admit, requested=explicit) == explicit
    assert seen == [32]

    def refuse(specs, resources: ResourceBudget) -> None:
        raise MemoryError("does not fit")

    with pytest.raises(MemoryError, match="requested count-matrix policy"):
        _fit(refuse, requested=explicit)


def test_fit_halves_the_default_policy_until_the_counts_write_fits():
    seen: list[int] = []
    workers: set[int] = set()

    def admit(specs, resources: ResourceBudget) -> None:
        workers.add(resources.workers)
        rows = specs[0].shards[0]
        seen.append(rows)
        if rows > 100_000:
            raise MemoryError("band too tall")

    fitted = _fit(admit)
    assert fitted == CountMatrixPolicy(
        unitBytes=DEFAULT_COUNT_MATRIX_POLICY.unitBytes // 8,
        chunkBytes=DEFAULT_COUNT_MATRIX_POLICY.chunkBytes // 8,
    )
    assert fitted.chunksPerShard == DEFAULT_COUNT_MATRIX_POLICY.chunksPerShard
    assert seen == [500_000, 250_000, 125_000, 62_500]
    # Admission plans one worker, so the layout never depends on the worker count.
    assert workers == {1}


def test_fit_also_admits_the_counts_t_transpose():
    def admit(specs, resources: ResourceBudget) -> None:
        pass

    untransposed = _fit(admit, memoryBytes=64 * 1024**2)
    transposed = _fit(admit, transposed=("RNA",), memoryBytes=64 * 1024**2)
    assert untransposed == DEFAULT_COUNT_MATRIX_POLICY
    assert transposed.unitBytes < DEFAULT_COUNT_MATRIX_POLICY.unitBytes


def test_fit_stops_at_count_shards_of_one_row():
    seen: list[int] = []

    def refuse(specs, resources: ResourceBudget) -> None:
        seen.append(specs[0].shards[0])
        raise MemoryError("does not fit")

    with pytest.raises(MemoryError, match="count shards of one row"):
        _fit(refuse)
    assert seen[-1] == 1
    assert seen.count(1) == 1


def _write_h5ad(path: Path, values: np.ndarray) -> Path:
    matrix = csr_matrix(values)
    with h5py.File(path, "w") as h5:
        group = h5.create_group("X")
        group.attrs["encoding-type"] = "csr_matrix"
        group.attrs["shape"] = values.shape
        group.create_dataset("data", data=matrix.data)
        group.create_dataset("indices", data=matrix.indices)
        group.create_dataset("indptr", data=matrix.indptr)
        h5.create_group("obs").create_dataset(
            "_index", data=np.array([f"c{i}".encode() for i in range(values.shape[0])])
        )
        var = h5.create_group("var")
        names = np.array([f"g{i}".encode() for i in range(values.shape[1])])
        var.create_dataset("_index", data=names)
        var.create_dataset("feature_name", data=names)
        h5.create_group("obsm")
    return path


@pytest.fixture(scope="module")
def wide_counts(tmp_path_factory) -> tuple[Path, np.ndarray]:
    values = np.random.default_rng(0).poisson(0.3, size=(4_000, 600))
    # One count past uint16 keeps uint32 storage and wide count rows.
    values[0, 0] = 70_000
    path = tmp_path_factory.mktemp("layout") / "wide.h5ad"
    return _write_h5ad(path, values.astype(np.float32)), values


def _build(path: Path, destination: Path, **options) -> zarr.Group:
    reader = H5adReader(str(path), feature_name_key="feature_name")
    try:
        H5adToZarr(reader, str(destination), **options).dump()
    finally:
        reader.close()
    return zarr.open_group(str(destination), mode="r")


def _policy(root: zarr.Group) -> CountMatrixPolicy:
    return policy_from_payload(load_count_matrix_plan(root["RNA/counts"]))


def test_h5ad_import_fits_the_count_layout_to_its_budget(wide_counts, tmp_path):
    path, values = wide_counts
    budget = 24 * 1024**2
    # The default layout writes the whole matrix as one band, which does not
    # fit this budget, and it fails before the destination exists.
    with pytest.raises(MemoryError, match="requested count-matrix policy"):
        _build(
            path,
            tmp_path / "default.zarr",
            mem_budget=budget,
            policy=DEFAULT_COUNT_MATRIX_POLICY,
        )
    assert not (tmp_path / "default.zarr").exists()

    roomy = _build(path, tmp_path / "roomy.zarr", mem_budget="1G", nthreads=1)
    fitted = _build(path, tmp_path / "fitted.zarr", mem_budget=budget, nthreads=1)
    assert _policy(roomy) == DEFAULT_COUNT_MATRIX_POLICY
    policy = _policy(fitted)
    assert policy.unitBytes < DEFAULT_COUNT_MATRIX_POLICY.unitBytes
    assert policy.chunksPerShard == DEFAULT_COUNT_MATRIX_POLICY.chunksPerShard
    assert fitted["RNA/counts"].dtype == np.uint32
    assert fitted["RNA/countsT"].attrs["complete"] is True
    np.testing.assert_array_equal(fitted["RNA/counts"][:], values)
    np.testing.assert_array_equal(fitted["RNA/countsT"][:], values.T)
    # Identity does not depend on the layout.
    assert (
        fitted["RNA/counts"].attrs["content_fingerprint"]
        == roomy["RNA/counts"].attrs["content_fingerprint"]
    )

    # The fitted layout does not depend on the worker count.
    parallel = _build(path, tmp_path / "parallel.zarr", mem_budget=budget, nthreads=4)
    assert _policy(parallel) == policy


def test_h5ad_import_below_one_row_shards_fails_before_the_destination_exists(
    wide_counts, tmp_path
):
    path, _values = wide_counts
    destination = tmp_path / "tiny.zarr"
    # The count summaries alone need more than this budget.
    with pytest.raises(MemoryError, match="count shards of one row"):
        _build(path, destination, mem_budget=128 * 1024)
    assert not destination.exists()

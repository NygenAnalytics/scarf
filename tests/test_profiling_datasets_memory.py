from pathlib import Path

import h5py
import numpy as np

from profiling.config import load_profiling_config
from profiling.datasets import (
    SourceSpec,
    load_csr_source_into_memory,
    prepare_local_datasets,
    select_nested_rows,
    write_fixture_h5ad,
    write_h5ad_sample,
    write_h5ad_sample_from_memory,
)


def _fixture_spec(path: Path, *, nRows: int, nColumns: int, nnz: int) -> SourceSpec:
    return SourceSpec(
        datasetId="fixture",
        versionId="fixture-v1",
        url="file://fixture",
        nRows=nRows,
        nColumns=nColumns,
        nnz=nnz,
        sourceBytes=path.stat().st_size,
    )


def test_load_csr_downcasts_indices_to_int32(tmp_path: Path) -> None:
    source = tmp_path / "source.h5ad"
    artifact = write_fixture_h5ad(source, nRows=40, nColumns=25, seed=3)
    spec = _fixture_spec(
        source,
        nRows=40,
        nColumns=25,
        nnz=artifact.nnz,
    )
    memory = load_csr_source_into_memory(source, spec=spec)
    assert memory.indices.dtype == np.dtype(np.int32)
    assert memory.indicesDtype == np.dtype(np.int64)
    assert memory.data.dtype == np.dtype(np.float32)
    assert int(memory.indptr[-1]) == artifact.nnz


def test_write_from_memory_matches_disk_sample(tmp_path: Path) -> None:
    source = tmp_path / "source.h5ad"
    artifact = write_fixture_h5ad(source, nRows=60, nColumns=20, seed=4)
    spec = _fixture_spec(
        source,
        nRows=60,
        nColumns=20,
        nnz=artifact.nnz,
    )
    rows = select_nested_rows(60, (15,), seed=1, sourceVersion=spec.versionId)[15]
    memory = load_csr_source_into_memory(source, spec=spec)

    disk_path = tmp_path / "disk.h5ad"
    memory_path = tmp_path / "memory.h5ad"
    disk = write_h5ad_sample(source, disk_path, rows, spec=spec)
    mem = write_h5ad_sample_from_memory(memory, memory_path, rows)

    assert disk.nnz == mem.nnz
    assert disk.sourceRowsSha256 == mem.sourceRowsSha256
    assert disk.finalSourceRow == mem.finalSourceRow

    with h5py.File(disk_path, "r") as left, h5py.File(memory_path, "r") as right:
        assert np.array_equal(left["X/data"][:], right["X/data"][:])
        assert np.array_equal(left["X/indices"][:], right["X/indices"][:])
        assert np.array_equal(left["X/indptr"][:], right["X/indptr"][:])


def test_prepare_local_datasets_uses_in_memory_path(tmp_path: Path) -> None:
    source = tmp_path / "source.h5ad"
    artifact = write_fixture_h5ad(source, nRows=80, nColumns=30, seed=5)
    spec = _fixture_spec(
        source,
        nRows=80,
        nColumns=30,
        nnz=artifact.nnz,
    )
    prepared = prepare_local_datasets(
        source,
        tmp_path / "subsets",
        targetRows=(10, 25),
        seed=0,
        spec=spec,
    )
    assert [item.targetRows for item in prepared.artifacts] == [10, 25]
    assert (tmp_path / "subsets" / "10.h5ad").is_file()
    assert (tmp_path / "subsets" / "25.h5ad").is_file()

    selections = select_nested_rows(
        80,
        (10, 25),
        seed=0,
        sourceVersion=spec.versionId,
    )
    assert set(selections[10].tolist()).issubset(set(selections[25].tolist()))


def test_example_config_loads_prepare_resources() -> None:
    config = load_profiling_config(
        Path(__file__).parents[1] / "profiling" / "config.example.toml"
    )
    assert config.prepareResources.modalMemoryRequestMb == 196_608
    assert config.prepareResources.modalMemoryLimitMb == 212_992

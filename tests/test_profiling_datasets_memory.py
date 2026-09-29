from pathlib import Path

import h5py
import numpy as np

from profiling.config import load_profiling_config
from profiling.datasets import (
    SourceSpec,
    download_source,
    load_csr_source_into_memory,
    ordered_source_row_digest,
    prepare_local_datasets,
    select_nested_rows,
    write_fixture_h5ad,
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


def test_write_from_memory_selects_exact_source_rows(tmp_path: Path) -> None:
    from scipy.sparse import csr_matrix

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

    sample_path = tmp_path / "memory.h5ad"
    written = write_h5ad_sample_from_memory(memory, sample_path, rows)

    with h5py.File(source, "r") as h5:
        expected = csr_matrix(
            (h5["X/data"][:], h5["X/indices"][:], h5["X/indptr"][:]),
            shape=(60, 20),
        )[rows]
        expected_ids = h5["obs/_index"][:][rows]
    with h5py.File(sample_path, "r") as h5:
        assert np.array_equal(h5["X/data"][:], expected.data)
        assert np.array_equal(h5["X/indices"][:], expected.indices)
        assert np.array_equal(h5["X/indptr"][:], expected.indptr)
        assert np.array_equal(h5["obs/_index"][:], expected_ids)
    assert written.nnz == expected.nnz
    assert written.sourceRowsSha256 == ordered_source_row_digest(rows)
    assert written.finalSourceRow == int(rows[-1])


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


class _Response:
    def __init__(self, payload: bytes, *, status: int, headers: dict[str, str]):
        self.payload = payload
        self.status = status
        self.headers = headers
        self.stall = False

    def read(self, size: int) -> bytes:
        if self.stall and not self.payload:
            raise TimeoutError("read timed out")
        chunk, self.payload = self.payload[:size], self.payload[size:]
        return chunk

    def close(self) -> None:
        pass


def test_download_source_resumes_a_stalled_transfer(tmp_path: Path) -> None:
    payload = bytes(range(256)) * 40
    requests: list[int] = []

    def opener(_url: str, offset: int) -> _Response:
        requests.append(offset)
        if offset == 0:
            # The first connection stalls after 3000 bytes.
            response = _Response(
                payload[:3000],
                status=200,
                headers={"Content-Length": str(len(payload))},
            )
            response.stall = True
            return response
        return _Response(
            payload[offset:],
            status=206,
            headers={
                "Content-Range": f"bytes {offset}-{len(payload) - 1}/{len(payload)}"
            },
        )

    result = download_source(
        tmp_path / "source.h5ad",
        url="https://example.invalid/source.h5ad",
        expectedBytes=len(payload),
        chunkBytes=1000,
        opener=opener,
        retryDelaySeconds=0.0,
    )

    assert requests == [0, 3000]
    assert (tmp_path / "source.h5ad").read_bytes() == payload
    assert result.fileBytes == len(payload)
    assert [path.name for path in tmp_path.iterdir()] == ["source.h5ad"]


def test_download_source_rejects_a_server_that_ignores_the_range(
    tmp_path: Path,
) -> None:
    import pytest

    payload = b"x" * 5000

    def opener(_url: str, offset: int) -> _Response:
        response = _Response(
            payload,
            status=200,
            headers={"Content-Length": str(len(payload))},
        )
        if offset == 0:
            response.payload = payload[:2000]
            response.stall = True
        return response

    with pytest.raises(ValueError, match="did not resume at byte 2000"):
        download_source(
            tmp_path / "source.h5ad",
            url="https://example.invalid/source.h5ad",
            expectedBytes=len(payload),
            chunkBytes=1000,
            opener=opener,
            retryDelaySeconds=0.0,
        )
    assert list(tmp_path.iterdir()) == []

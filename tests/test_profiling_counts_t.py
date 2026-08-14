import numpy as np
import pytest
import zarr

from profiling import stages as profiling_stages
from profiling.config import StageResources, WorkflowParameters
from profiling.stages import run_stage
from scarf.storage.sharding import write_counts_t


def _resources() -> StageResources:
    return StageResources(
        modalMemoryRequestMb=4096,
        modalMemoryLimitMb=4096,
        modalCpuRequest=1.0,
        modalCpuLimit=1.0,
        scarfMemoryBudget=2 * 1024**3,
        workers=1,
        timeoutSeconds=600,
        ephemeralDiskMb=524288,
    )


def test_write_counts_t_rewrites_incomplete_array(tmp_path):
    root_path = tmp_path / "store.zarr"
    root = zarr.open_group(str(root_path), mode="w")
    group = root.create_group("RNA")
    values = np.arange(24, dtype=np.uint32).reshape(6, 4)
    counts = group.create_array("counts", shape=values.shape, dtype=values.dtype)
    counts[:] = values
    counts_t = write_counts_t(counts, group)
    assert counts_t is not None
    counts_t.attrs["complete"] = False
    counts_t[0, 0] = 999

    result = run_stage(
        "writeCountsT",
        nRows=6,
        storeUri=str(root_path),
        workflow=WorkflowParameters(),
        resources=_resources(),
        sampleIntervalSeconds=0.01,
    )

    assert result.status == "ok"
    assert result.details is not None
    assert result.details["complete"] is True
    assert result.details["beforeComplete"] is False
    reopened = zarr.open_group(str(root_path), mode="r")
    fixed = reopened["RNA/countsT"]
    assert fixed.attrs["complete"] is True
    np.testing.assert_array_equal(fixed[:], values.T)


def test_write_counts_t_runs_as_standard_profile_stage(tmp_path):
    root_path = tmp_path / "store.zarr"
    root = zarr.open_group(str(root_path), mode="w")
    group = root.create_group("RNA")
    values = np.arange(24, dtype=np.uint32).reshape(6, 4)
    counts = group.create_array("counts", shape=values.shape, dtype=values.dtype)
    counts[:] = values

    result = run_stage(
        "writeCountsT",
        nRows=6,
        storeUri=str(root_path),
        workflow=WorkflowParameters(),
        resources=_resources(),
        sampleIntervalSeconds=0.01,
    )

    assert result.status == "ok"
    assert result.inputSetupSeconds is not None
    assert result.seconds is not None
    assert result.validationPersistenceSeconds is not None
    assert result.details is not None
    assert result.details["complete"] is True
    assert result.details["shape"] == [4, 6]
    assert result.details["beforeComplete"] is None
    reopened = zarr.open_group(str(root_path), mode="r")
    np.testing.assert_array_equal(reopened["RNA/countsT"][:], values.T)


def test_create_store_defers_counts_t_to_write_stage(tmp_path):
    import h5py
    from scipy.sparse import csr_matrix

    values = np.arange(24, dtype=np.uint32).reshape(6, 4)
    h5ad_path = tmp_path / "cells.h5ad"
    with h5py.File(h5ad_path, "w") as h5:
        group = h5.create_group("X")
        matrix = csr_matrix(values)
        group.create_dataset("data", data=matrix.data)
        group.create_dataset("indices", data=matrix.indices.astype(np.int32))
        group.create_dataset("indptr", data=matrix.indptr.astype(np.int32))
        group.attrs["encoding-type"] = "csr_matrix"
        group.attrs["encoding-version"] = "0.1.0"
        group.attrs["shape"] = values.shape
        obs = h5.create_group("obs")
        obs.create_dataset(
            "_index",
            data=np.array([f"c{i}".encode() for i in range(values.shape[0])]),
        )
        var = h5.create_group("var")
        var.create_dataset(
            "_index",
            data=np.array([f"f{i}".encode() for i in range(values.shape[1])]),
        )
        var.create_dataset(
            "feature_name",
            data=np.array([f"g{i}".encode() for i in range(values.shape[1])]),
        )

    store_path = tmp_path / "store.zarr"
    create = run_stage(
        "createStore",
        nRows=values.shape[0],
        storeUri=str(store_path),
        workflow=WorkflowParameters(),
        resources=_resources(),
        localH5adPath=h5ad_path,
        sampleIntervalSeconds=0.01,
    )
    assert create.status == "ok"
    assert create.details is not None
    assert create.details["countsOnly"] is True
    assert create.details["countsTPresent"] is False
    after_create = zarr.open_group(str(store_path), mode="r")
    assert "countsT" not in after_create["RNA"]
    np.testing.assert_array_equal(after_create["RNA/counts"][:], values)

    write = run_stage(
        "writeCountsT",
        nRows=values.shape[0],
        storeUri=str(store_path),
        workflow=WorkflowParameters(),
        resources=_resources(),
        sampleIntervalSeconds=0.01,
    )
    assert write.status == "ok"
    assert write.details is not None
    assert write.details["beforeComplete"] is None
    assert write.details["complete"] is True
    after_write = zarr.open_group(str(store_path), mode="r")
    np.testing.assert_array_equal(after_write["RNA/countsT"][:], values.T)


def test_write_counts_t_clears_complete_when_validation_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    root_path = tmp_path / "store.zarr"
    root = zarr.open_group(str(root_path), mode="w")
    group = root.create_group("RNA")
    values = np.arange(24, dtype=np.uint32).reshape(6, 4)
    counts = group.create_array("counts", shape=values.shape, dtype=values.dtype)
    counts[:] = values
    original_write = profiling_stages._write_counts_t

    def corrupt_counts_t(*args, **kwargs):
        counts_t, details = original_write(*args, **kwargs)
        counts_t[0, 0] = 999
        return counts_t, details

    monkeypatch.setattr(profiling_stages, "_write_counts_t", corrupt_counts_t)

    result = run_stage(
        "writeCountsT",
        nRows=6,
        storeUri=str(root_path),
        workflow=WorkflowParameters(),
        resources=_resources(),
        sampleIntervalSeconds=0.01,
    )

    assert result.status == "error"
    assert result.error is not None
    assert "tile mismatch" in result.error
    reopened = zarr.open_group(str(root_path), mode="r")
    assert reopened["RNA/countsT"].attrs["complete"] is False

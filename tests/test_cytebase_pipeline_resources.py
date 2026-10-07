"""Offline tests for choosing the resource tier of a Cytebase dataset worker."""

from pathlib import Path

import h5py
import numpy as np
import pytest

import scarf
from scarf.cytebase.pipeline.models import DatasetRecord
from scarf.cytebase.pipeline.resources import (
    LARGE_SOURCE_BYTES,
    default_layout_fits,
    initial_tier,
    tier_with_memory,
)
from scarf.storage.sharding import CountLayoutMemoryError
from tests.fixtures_cytebase import (
    NEW_VERSION_ID,
    VERSION_ID,
    dataset_record,
    write_h5ad,
)

N_CELLS, N_GENES, GENES_PER_CELL = 3_000, 400, 40


def _record(**overrides) -> DatasetRecord:
    return DatasetRecord.model_validate(dataset_record(**overrides))


def _smallest_admitted_budget(**shape) -> int:
    low, high = 1, 1 << 40
    assert not default_layout_fits(low, **shape)
    while high - low > 1:
        middle = (low + high) // 2
        if default_layout_fits(middle, **shape):
            high = middle
        else:
            low = middle
    return high


def _uniform_csr_source(path: Path) -> Path:
    """Write float32 integer counts, ``GENES_PER_CELL`` per cell, as int64 CSR."""
    rows = np.arange(N_CELLS)[:, None]
    columns = (rows + np.arange(GENES_PER_CELL)[None, :] * 10) % N_GENES
    counts = np.zeros((N_CELLS, N_GENES), dtype=np.float32)
    counts[rows, columns] = 1 + (rows + columns) % 999
    write_h5ad(path, counts, annotations=False, umap=False, uns=False)
    with h5py.File(path, "r+") as h5:
        pointers = h5["X/indptr"][:].astype(np.int64)
        del h5["X/indptr"]
        h5["X"].create_dataset("indptr", data=pointers)
    return path


def _h5ad_import_admits(source: Path, destination: Path, budget: int) -> bool:
    reader = scarf.H5adReader(str(source))
    try:
        writer = scarf.H5adToZarr(
            reader,
            zarr_loc=str(destination),
            assay_name="RNA",
            profile="cloud",
            nthreads=1,
            mem_budget=budget,
        )
    except CountLayoutMemoryError:
        return False
    finally:
        reader.h5.close()
    assert writer.storageDtypes == {"RNA": np.dtype(np.uint16)}
    writer.z.store.close()
    return True


def test_default_layout_fits_matches_the_h5ad_import_admission(tmp_path):
    # The estimate copies what H5adToZarr derives from a CSR source. If Scarf
    # changes its admission, this test shows the estimate no longer agrees.
    source = _uniform_csr_source(tmp_path / "source.h5ad")
    budget = _smallest_admitted_budget(
        nCells=N_CELLS, nFeatures=N_GENES, windowGenesPerCell=GENES_PER_CELL
    )

    assert _h5ad_import_admits(source, tmp_path / "fits.zarr", budget)
    assert not _h5ad_import_admits(source, tmp_path / "refused.zarr", budget - 1)


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        # Measured with this estimate on the CELLxGENE shapes in the plan:
        # a 10x 3' v3 dataset needs about 9.6 GiB, a richer one 15.4 GiB.
        ({"cellCount": 100_000, "nGenes": 36_601, "meanGenesPerCell": 3_000.0}, 0),
        ({"cellCount": 100_000, "nGenes": 36_601, "meanGenesPerCell": 5_000.0}, 1),
        ({"cellCount": 79_631, "nGenes": 18_736, "meanGenesPerCell": 5_519.0}, 2),
        # Cells barely move the estimate; density does.
        ({"cellCount": 11_441_407, "nGenes": 45_525, "meanGenesPerCell": 1_706.0}, 0),
        ({"cellCount": 100_000, "nGenes": 36_601, "meanGenesPerCell": 36_601.0}, 2),
    ],
)
def test_initial_tier_starts_in_the_smallest_tier_that_admits_the_estimate(
    shape, expected
):
    assert initial_tier(_record(**shape)) == expected


@pytest.mark.parametrize(
    ("source_bytes", "expected"), [(LARGE_SOURCE_BYTES, 0), (LARGE_SOURCE_BYTES + 1, 1)]
)
def test_initial_tier_adds_a_margin_for_large_sources(source_bytes, expected):
    record = _record(
        cellCount=100_000,
        nGenes=36_601,
        meanGenesPerCell=3_000.0,
        sourceBytes=source_bytes,
    )
    assert initial_tier(record) == expected


@pytest.mark.parametrize(
    "missing", [{"meanGenesPerCell": None}, {"cellCount": None}, {"nGenes": 0}]
)
def test_initial_tier_starts_at_the_smallest_tier_without_sizes(missing):
    shape = {"cellCount": 100_000, "nGenes": 36_601, "meanGenesPerCell": 5_000.0}
    assert initial_tier(_record(**shape | missing)) == 0


@pytest.mark.parametrize(
    ("version", "memory", "expected"),
    [
        (VERSION_ID, 32_768, 1),
        (VERSION_ID, 65_536, 2),
        (VERSION_ID, None, 0),
        # A receipt from an older version says nothing about the latest one.
        (NEW_VERSION_ID, 65_536, 0),
    ],
)
def test_initial_tier_starts_at_least_in_the_tier_that_built_the_latest_version(
    version, memory, expected
):
    resources = None if memory is None else {"cpu": 8, "memoryMiB": memory}
    record = _record(buildReceipt={"datasetVersionId": version, "resources": resources})
    assert initial_tier(record) == expected


@pytest.mark.parametrize(
    ("memory", "expected"),
    [(1, 0), (16_384, 0), (16_385, 1), (65_536, 2), (1_000_000, 2)],
)
def test_tier_with_memory_rounds_up_to_a_tier(memory, expected):
    assert tier_with_memory(memory) == expected

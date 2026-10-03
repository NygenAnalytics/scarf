"""Benchmarks of the merge and H5AD export paths whose cost grows with the data.

A dataset merge prefixes the ID of every source cell, regenerates and compares
every merged ID each time it plans a resume, measures the widest text value of
each source metadata column while it plans the schema, and streams every
source count block into the merged layout. An H5AD export converts each dense
block of raw counts to CSR, so its cost follows cells times features rather
than the stored nonzeros. Each benchmark calls the function its path calls, on
Scarf-written stores and tables where the path reads them, and checks the
result against an independent oracle, so it doubles as a test at the smoke
size.
"""

from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import zarr
from scipy import sparse
from zarr.storage import MemoryStore

from tests.storage_helpers import write_count_store

from . import inputs
from .harness import Ladder

pytestmark = pytest.mark.benchmark

SOURCES = ("left", "right")
# Row-plan blocks follow the row chunks of the source counts. The default
# layout chunks uint16 counts of 25,000 features in 2,000 rows.
PLAN_BLOCK_ROWS = 2_048
# A merge budgets the memory of its sources, which defaults to the machine's
# memory and holds whole source chunks; 8 GiB does so at every size here.
MERGE_BUDGET = 8 * 1024**3
# The feature ladder exports this many cells, each with this many nonzero
# counts, so its stored nonzeros stay constant while the features grow.
EXPORT_CELLS = 2_000
EXPORT_NONZEROS = 64


def _barcode_letters(n_cells: int) -> np.ndarray:
    """Return ``n_cells`` distinct 16-letter 10x barcodes in shuffled order.

    Each barcode spells a distinct integer in base four.
    """
    rng = np.random.default_rng(inputs.SEED)
    codes = rng.permutation(n_cells).astype(np.int64)
    alphabet = np.frombuffer(b"ACGT", dtype=np.uint8)
    letters = np.empty((n_cells, 16), dtype=np.uint8)
    for position in range(16):
        letters[:, position] = alphabet[(codes >> (2 * (15 - position))) & 3]
    return letters.view("S16").reshape(-1).astype(str)


@lru_cache(maxsize=4)
def _barcodes(n_cells: int) -> np.ndarray:
    """Return 10x cell IDs such as ``ACGT...-1`` as Scarf stores them."""
    return np.char.add(_barcode_letters(n_cells), "-1")


@lru_cache(maxsize=4)
def _lane_barcodes(n_cells: int) -> np.ndarray:
    """Return barcodes suffixed by one of twelve lanes, of 18 or 19 letters."""
    lanes = (np.arange(n_cells) % 12 + 1).astype(str)
    return np.char.add(np.char.add(_barcode_letters(n_cells), "-"), lanes)


def _cell_table(ids: np.ndarray):
    """Return a Zarr-backed cell table with the columns a source DataStore has."""
    from scarf.metadata import MetaData
    from scarf.storage.arrays import create_metadata_column
    from scarf.storage.layout import PROFILE_METADATA_CHUNK

    group = zarr.open_group(store=MemoryStore(), mode="w")
    for name, values in (
        ("ids", ids),
        ("names", ids),
        ("I", np.ones(ids.size, dtype=bool)),
    ):
        create_metadata_column(
            group, name, data=values, chunkSize=PROFILE_METADATA_CHUNK
        )
    return MetaData(group)


def _merged_id_dtype(ids: np.ndarray) -> np.dtype:
    """Return the dtype a merge gives its prefixed cell IDs."""
    longest = max(len(name) for name in SOURCES)
    return np.dtype(f"U{longest + 2 + ids.dtype.itemsize // 4}")


def test_prefixed_cell_ids(bench) -> None:
    from scarf.merge.row_plan import prefixed_cell_ids

    def make(n_cells: int):
        ids = _barcodes(n_cells)
        dtype = _merged_id_dtype(ids)
        return lambda: prefixed_cell_ids("right", ids, dtype)

    def check(n_cells: int, merged) -> None:
        np.testing.assert_array_equal(
            merged, np.char.add("right__", _barcodes(n_cells))
        )

    # A merge writes the prefixed ID of every cell of every source once.
    ladder = Ladder(sizes=(100_000, 200_000, 400_000, 800_000), smoke=2_000)
    bench("merge.prefixed_cell_ids", make, ladder, check=check)


@lru_cache(maxsize=4)
def _identity_inputs(n_cells: int):
    """Return the row plan, source tables, and stored IDs of a merge of two sources.

    The stored IDs are built with NumPy string concatenation in the order the
    row plan fixes, independently of the merge's own ID generation.
    """
    from scarf.merge.metadata import metadata_chunk_rows
    from scarf.merge.row_plan import build_row_plan, iter_row_plan_segments
    from scarf.storage.arrays import create_metadata_column

    ids = _barcodes(n_cells)
    half = n_cells // 2
    sources = (ids[:half], ids[half:])
    tables = [_cell_table(values) for values in sources]
    plan = build_row_plan(
        [values.size for values in sources],
        [PLAN_BLOCK_ROWS] * len(sources),
        list(SOURCES),
        seed=0,
    )
    merged = np.empty(n_cells, dtype=_merged_id_dtype(ids))
    for segment in iter_row_plan_segments(plan):
        name = SOURCES[segment.sourceIdx]
        merged[segment.destStart : segment.destStart + segment.localRows.size] = (
            np.char.add(f"{name}__", sources[segment.sourceIdx][segment.localRows])
        )
    group = zarr.open_group(store=MemoryStore(), mode="w")
    stored = create_metadata_column(
        group, "ids", data=merged, chunkSize=metadata_chunk_rows(plan)
    )
    return plan, tables, stored


def test_verify_merged_cell_ids(bench) -> None:
    from scarf.merge.row_plan import verify_merged_cell_ids

    def make(n_cells: int):
        plan, tables, stored = _identity_inputs(n_cells)
        # A resume plan validates identity in segments of one row-plan block.
        return lambda: verify_merged_cell_ids(
            stored, plan, tables, block_rows=PLAN_BLOCK_ROWS
        )

    def check(n_cells: int, value) -> None:
        # The IDs built by NumPy concatenation verified; two swapped IDs do not.
        assert value is None
        plan, tables, stored = _identity_inputs(n_cells)
        swapped = np.asarray(stored[:])
        swapped[[0, n_cells - 1]] = swapped[[n_cells - 1, 0]]
        tampered = zarr.open_group(store=MemoryStore(), mode="w").create_array(
            "ids", data=swapped, chunks=stored.chunks
        )
        with pytest.raises(ValueError, match="order of cells does not match"):
            verify_merged_cell_ids(tampered, plan, tables, block_rows=PLAN_BLOCK_ROWS)

    # Every resume of a merge plans this pass over all merged cells.
    ladder = Ladder(sizes=(50_000, 100_000, 200_000, 400_000), smoke=2_000)
    bench("merge.verify_merged_cell_ids", make, ladder, check=check)


def test_metadata_text_width(bench) -> None:
    from scarf.merge.metadata import _max_text_width
    from scarf.storage.layout import PROFILE_METADATA_CHUNK

    @lru_cache(maxsize=4)
    def table(n_cells: int):
        return _cell_table(_lane_barcodes(n_cells))

    def make(n_cells: int):
        source = table(n_cells)
        return lambda: _max_text_width(source, "ids", block_rows=PROFILE_METADATA_CHUNK)

    def check(n_cells: int, width) -> None:
        assert width == max(len(text) for text in _lane_barcodes(n_cells).tolist())

    # The schema scan measures the IDs, names, and text columns of every
    # source once; this times one column of one source.
    ladder = Ladder(sizes=(250_000, 500_000, 1_000_000, 2_000_000), smoke=5_000)
    bench("merge.metadata_text_width", make, ladder, check=check)


def _open(path: Path):
    from scarf import DataStore

    return DataStore(str(path), default_assay="RNA", nthreads=1)


@pytest.fixture(scope="module")
def stores(tmp_path_factory) -> Callable[[str, int], object]:
    """Return the store of a kind and size, building it on first use.

    The kinds are ``cells`` and ``features`` for the export ladders, and the
    source names for the merge ladder, whose size counts the merged cells.
    """
    root = tmp_path_factory.mktemp("merge_export_benchmarks")
    built: dict[tuple[str, int], object] = {}

    def get(kind: str, size: int):
        key = (kind, size)
        if key not in built:
            path = root / f"{kind}_{size}.zarr"
            write_count_store(str(path), {"RNA": _counts(kind, size)}, "uint16")
            built[key] = _open(path)
        return built[key]

    return get


@lru_cache(maxsize=4)
def _sparse_rows(n_features: int) -> np.ndarray:
    """Return ``EXPORT_CELLS`` rows with ``EXPORT_NONZEROS`` counts of 1 to 9 each."""
    rng = np.random.default_rng(inputs.SEED)
    counts = np.zeros((EXPORT_CELLS, n_features), dtype=np.uint16)
    for row in counts:
        columns = rng.choice(n_features, size=EXPORT_NONZEROS, replace=False)
        row[columns] = rng.integers(1, 10, size=EXPORT_NONZEROS)
    return counts


def _counts(kind: str, size: int) -> np.ndarray:
    """Return the counts of a store kind; see the ``stores`` fixture."""
    if kind == "features":
        return _sparse_rows(size)
    counts, _labels = inputs.labelled_counts(size)
    if kind == "cells":
        return counts
    # A merge source holds one half of the cells of the merged size.
    half = size // 2
    return counts[:half] if kind == SOURCES[0] else counts[half:]


def _check_export(path: Path, counts: np.ndarray) -> None:
    """Compare an exported H5AD file with the counts of the store it came from."""
    import h5py

    expected = sparse.csr_matrix(counts)
    with h5py.File(path, mode="r") as h5:
        shape = tuple(int(value) for value in h5["X"].attrs["shape"])
        exported = sparse.csr_matrix(
            (h5["X/data"][:], h5["X/indices"][:], h5["X/indptr"][:]), shape=shape
        )
        cell_ids = h5["obs/_index"].asstr()[:]
        feature_ids = h5["var/_index"].asstr()[:]
    assert exported.shape == expected.shape
    np.testing.assert_array_equal(exported.indptr, expected.indptr)
    np.testing.assert_array_equal(exported.indices, expected.indices)
    np.testing.assert_array_equal(exported.data, expected.data)
    np.testing.assert_array_equal(
        cell_ids, [f"cell{index}" for index in range(counts.shape[0])]
    )
    np.testing.assert_array_equal(
        feature_ids, [f"RNA{index}" for index in range(counts.shape[1])]
    )


def test_h5ad_export_cells(bench, stores, tmp_path) -> None:
    from scarf.writers import to_h5ad

    def make(n_cells: int):
        assay = stores("cells", n_cells).RNA
        output = tmp_path / f"cells_{n_cells}.h5ad"

        def call() -> Path:
            to_h5ad(assay, str(output), nthreads=1)
            return output

        return call

    def check(n_cells: int, output: Path) -> None:
        _check_export(output, _counts("cells", n_cells))

    ladder = Ladder(sizes=(2_000, 4_000, 8_000, 16_000), smoke=600)
    bench("export.h5ad_cells", make, ladder, check=check)


def test_h5ad_export_features(bench, stores, tmp_path) -> None:
    from scarf.writers import to_h5ad

    def make(n_features: int):
        assay = stores("features", n_features).RNA
        output = tmp_path / f"features_{n_features}.h5ad"

        def call() -> Path:
            to_h5ad(assay, str(output), nthreads=1)
            return output

        return call

    def check(n_features: int, output: Path) -> None:
        _check_export(output, _counts("features", n_features))

    # The stored nonzeros stay at EXPORT_NONZEROS per cell while the features
    # grow, so any growth comes from converting dense blocks. Export time is
    # linear in cells, so the projections scale the 2,000 cells to 1M cells.
    ladder = Ladder(
        sizes=(8_000, 16_000, 32_000, 64_000),
        smoke=2_000,
        unit="features",
        targets=(30_000, 60_000),
        work=1_000_000 / EXPORT_CELLS,
    )
    bench("export.h5ad_features", make, ladder, check=check)


def test_merge_counts(bench, stores, tmp_path) -> None:
    from scarf.merge.features import align_features
    from scarf.merge.row_plan import build_row_plan, iter_row_plan_segments
    from scarf.merge.writer import create_assay_counts, write_assay_counts
    from scarf.storage.budget import ResourceBudget
    from scarf.storage.count_matrix import DEFAULT_COUNT_MATRIX_POLICY

    @lru_cache(maxsize=4)
    def prepared(n_cells: int):
        # The two sources hold the two halves of the cells, under the IDs and
        # features that write_count_store gives them.
        assays = [stores(name, n_cells).RNA for name in SOURCES]
        alignment = align_features(assays, list(SOURCES))
        plan = build_row_plan(
            [int(assay.rawData.shape[0]) for assay in assays],
            [int(assay.rawData.chunksize[0]) for assay in assays],
            list(SOURCES),
            seed=0,
        )
        root = zarr.open_group(str(tmp_path / f"merged_{n_cells}.zarr"), mode="w")
        create_assay_counts(
            root,
            "RNA",
            None,
            plan.nCells,
            alignment,
            "uint16",
            profile="fast_local",
            policy=DEFAULT_COUNT_MATRIX_POLICY,
        )
        return assays, alignment, plan, root

    def make(n_cells: int):
        assays, alignment, plan, root = prepared(n_cells)
        resources = ResourceBudget(MERGE_BUDGET, 1)
        return lambda: write_assay_counts(
            root, "RNA", None, assays, plan, alignment, resources=resources
        )

    def check(n_cells: int, rows) -> None:
        _assays, _alignment, plan, root = prepared(n_cells)
        sources = [_counts(name, n_cells) for name in SOURCES]
        expected = np.empty((n_cells, inputs.N_GENES), dtype=np.uint16)
        for segment in iter_row_plan_segments(plan):
            stop = segment.destStart + segment.localRows.size
            expected[segment.destStart : stop] = sources[segment.sourceIdx][
                segment.localRows
            ]
        assert rows == n_cells
        np.testing.assert_array_equal(root["RNA/counts"][:], expected)

    # Sizes count the merged cells, half from each source.
    ladder = Ladder(sizes=(4_000, 8_000, 16_000, 32_000), smoke=600)
    bench("merge.write_assay_counts", make, ladder, check=check)

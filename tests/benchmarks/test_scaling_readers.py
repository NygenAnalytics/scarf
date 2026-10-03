"""Benchmarks of the Seurat reader paths whose cost grows with the cell count.

Each benchmark times the function a Seurat import calls on inputs shaped like
an RDS file's contents: 10x barcodes as cell identifiers, and Poisson counts
stored as a column-compressed double matrix. Each one checks its result
against an independent oracle, so it doubles as a test at the smoke size.
"""

import gzip
import struct
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse
from scipy.stats import rankdata

from . import inputs
from .harness import Ladder

pytestmark = pytest.mark.benchmark

# Mean counts of the ranked features, from nearly empty to dense, so ranks
# cover implicit zeros, explicit ties, and distinct values.
RANK_FEATURE_MEANS = np.asarray([0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0])
# The default SourceLimits.tileCells, so the ladder spans 4 to 32 tiles.
RANK_TILE_CELLS = 1_024


def _barcodes(n_cells: int) -> tuple[str, ...]:
    """Return ``n_cells`` distinct 10x barcodes such as ``ACGT...-1`` in shuffled order.

    Each barcode spells a distinct integer in base four, so they are unique
    by construction.
    """
    rng = np.random.default_rng(inputs.SEED)
    codes = rng.permutation(n_cells).astype(np.int64)
    digits = (codes[:, None] >> (2 * np.arange(15, -1, -1))) & 3
    letters = np.frombuffer(b"ACGT", dtype=np.uint8)[digits]
    return tuple(row.tobytes().decode("ascii") + "-1" for row in letters)


@lru_cache(maxsize=4)
def _rank_counts(n_cells: int) -> sparse.csc_matrix:
    """Return feature-by-cell Poisson counts as a dgCMatrix stores them."""
    rng = np.random.default_rng(inputs.SEED)
    counts = rng.poisson(
        RANK_FEATURE_MEANS[:, None], size=(RANK_FEATURE_MEANS.size, n_cells)
    )
    return sparse.csc_matrix(counts.astype(np.float64))


def _expected_row_ranks(counts: sparse.csc_matrix) -> np.ndarray:
    """Rank each feature across cells as BPCells does, with SciPy's ``rankdata``.

    Ties share their average rank, ranks are shifted so the zeros' rank is
    zero, and zeros stay zero.
    """
    values = counts.toarray()
    expected = np.zeros_like(values)
    for feature, row in enumerate(values):
        ranks = rankdata(row, method="average")
        zeros = row == 0
        zero_rank = ranks[zeros].mean() if zeros.any() else 0.0
        expected[feature] = np.where(zeros, 0.0, ranks - zero_rank)
    return expected.T


def test_row_rank_full_pass(bench) -> None:
    from scarf.readers._seurat import CscMatrixSource, RankMatrixSource, SourceLimits

    def make(n_cells: int):
        counts = _rank_counts(n_cells)
        ranked = RankMatrixSource(
            CscMatrixSource(counts.data, counts.indices, counts.indptr, counts.shape),
            axis="row",
            limits=SourceLimits(tileCells=RANK_TILE_CELLS),
        )

        # An import reads every cell once, one window at a time.
        def call():
            return [
                ranked.read_cells(start, min(n_cells, start + RANK_TILE_CELLS))
                for start in range(0, n_cells, RANK_TILE_CELLS)
            ]

        return call

    def check(n_cells: int, blocks) -> None:
        ranks = sparse.vstack(blocks).toarray()
        np.testing.assert_allclose(
            ranks, _expected_row_ranks(_rank_counts(n_cells)), rtol=0, atol=1e-9
        )

    # Each window rescans every tile of the source, so time grows with the
    # square of the cell count. Cost is linear in features; this ranks eight.
    ladder = Ladder(sizes=(4_096, 8_192, 16_384, 32_768), smoke=1_500)
    bench("seurat.row_rank_full_pass", make, ladder, check=check)


def test_cell_identifier_uniqueness_index(bench, tmp_path: Path) -> None:
    from scarf.readers._seurat import DEFAULT_LIMITS
    from scarf.readers.seurat import SeuratImportError, _validate_unique_ids

    def validate(identifiers: tuple[str, ...]) -> None:
        return _validate_unique_ids(
            identifiers,
            object_path="meta.data/row.names",
            scratch_dir=tmp_path,
            maximum_bytes=DEFAULT_LIMITS.maxMetadataBytes,
        )

    def make(n_cells: int):
        identifiers = _barcodes(n_cells)
        return lambda: validate(identifiers)

    def check(n_cells: int, value) -> None:
        identifiers = _barcodes(n_cells)
        # The set oracle agrees: these barcodes are unique and are accepted.
        assert len(set(identifiers)) == n_cells
        assert value is None
        # Repeating one barcode at the end makes the index reject the axis.
        repeated = identifiers[:-1] + (identifiers[n_cells // 2],)
        with pytest.raises(SeuratImportError) as error:
            validate(repeated)
        assert error.value.code == "duplicate_id"
        assert list(tmp_path.iterdir()) == []

    # meta.data, one Assay5 cell LogMap, and the PCA and UMAP embeddings each
    # validate the cell identifiers once when a Seurat object is opened.
    ladder = Ladder(sizes=(16_384, 32_768, 65_536, 131_072), smoke=2_000, work=4.0)
    bench("seurat.cell_id_uniqueness", make, ladder, check=check)


def test_cell_identifier_alignment_index(bench, tmp_path: Path) -> None:
    from scarf.readers._seurat import DEFAULT_LIMITS
    from scarf.readers.seurat import _identifier_positions

    def order(n_cells: int) -> np.ndarray:
        return np.random.default_rng(inputs.SEED + 1).permutation(n_cells)

    def make(n_cells: int):
        identifiers = _barcodes(n_cells)
        reordered = tuple(identifiers[index] for index in order(n_cells))
        # An axis stored in another order, such as an Assay5 cell LogMap of a
        # merged object, is mapped onto the global cells through sqlite.
        return lambda: _identifier_positions(
            reordered,
            identifiers,
            object_path="assays/RNA/cells/Dimnames/0",
            scratch_dir=tmp_path,
            maximum_bytes=DEFAULT_LIMITS.maxMetadataBytes,
            bijective=True,
        )

    def check(n_cells: int, positions) -> None:
        # The reordered axis holds the global cell at each permuted position.
        np.testing.assert_array_equal(positions, order(n_cells))
        assert positions.dtype == np.int64
        assert list(tmp_path.iterdir()) == []

    ladder = Ladder(sizes=(8_192, 16_384, 32_768, 65_536), smoke=2_000)
    bench("seurat.cell_id_alignment", make, ladder, check=check)


@lru_cache(maxsize=4)
def _barcode_rds(n_cells: int) -> bytes:
    """Return a gzip RDS of one character vector of barcodes, as saveRDS writes it.

    The XDR header names UTF-8 as the native encoding, and every CHARSXP
    carries R's ASCII flag.
    """
    header = (
        b"X\n"
        + struct.pack(">iii", 3, 4 * 65_536 + 4 * 256, 3 * 65_536 + 5 * 256)
        + struct.pack(">i", 5)
        + b"UTF-8"
    )
    char_flags = struct.pack(">i", 9 | ((1 << 6) << 12))
    body = b"".join(
        char_flags + struct.pack(">i", len(barcode)) + barcode.encode("ascii")
        for barcode in _barcodes(n_cells)
    )
    payload = header + struct.pack(">ii", 16, n_cells) + body
    return gzip.compress(payload, compresslevel=6, mtime=0)


def test_rds_barcode_parse(bench, tmp_path: Path) -> None:
    from scarf.readers._rds import LazyStringVector, open_rds

    def path_for(n_cells: int) -> Path:
        path = tmp_path / f"barcodes-{n_cells}.rds"
        if not path.exists():
            path.write_bytes(_barcode_rds(n_cells))
        return path

    def make(n_cells: int):
        path = path_for(n_cells)

        def call():
            with open_rds(path, temp_dir=tmp_path) as document:
                strings = document.root.value
                return type(strings), len(strings)

        return call

    def check(n_cells: int, value) -> None:
        assert value == (LazyStringVector, n_cells)
        # Every barcode reads back as written, in order.
        with open_rds(path_for(n_cells), temp_dir=tmp_path) as document:
            assert document.root.value.read_block(0, n_cells) == list(
                _barcodes(n_cells)
            )
        assert sorted(path.name for path in tmp_path.iterdir()) == [
            f"barcodes-{n_cells}.rds"
        ]

    # Opening a Seurat object parses every cell barcode about five times: the
    # meta.data row names, the active.ident names, one Assay5 cell LogMap, and
    # the PCA and UMAP embeddings.
    ladder = Ladder(sizes=(4_096, 8_192, 16_384, 32_768), smoke=2_000, work=5.0)
    bench("seurat.rds_barcode_parse", make, ladder, check=check)

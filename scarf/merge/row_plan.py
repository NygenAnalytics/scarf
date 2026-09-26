from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..metadata.rows import read_metadata_rows_chunkwise
from ..utils.arrays import permute_into_chunks


@dataclass(frozen=True, slots=True)
class RowPlan:
    """Shared merged cell order across every assay in a DataStoreMerge."""

    permutationsRows: dict[int, dict[int, np.ndarray]]
    coordinatesPermutations: np.ndarray
    nCells: int
    sourceNames: tuple[str, ...]

    def resident_bytes(self) -> int:
        return int(self.coordinatesPermutations.nbytes) + sum(
            int(rows.nbytes)
            for chunks in self.permutationsRows.values()
            for rows in chunks.values()
        )


@dataclass(frozen=True, slots=True)
class RowPlanSegment:
    sourceIdx: int
    blockIdx: int
    destStart: int
    localRows: np.ndarray


def build_row_plan(
    n_cells_per_source: list[int],
    row_chunk_sizes: list[int],
    source_names: list[str],
    seed: int | None = 42,
) -> RowPlan:
    """Build a deterministic shared row order for concatenated sources."""
    if len(n_cells_per_source) != len(row_chunk_sizes):
        raise ValueError("Row chunk sizes must match the number of sources")
    if len(n_cells_per_source) != len(source_names):
        raise ValueError("Source names must match the number of sources")
    if any(rows <= 0 for rows in row_chunk_sizes):
        raise ValueError("Row chunk sizes must be positive")
    if len(source_names) != len(set(source_names)):
        raise ValueError("A unique name must be provided for each source DataStore")

    rng = np.random.default_rng(seed=seed)
    chunk_size = np.asarray(row_chunk_sizes, dtype=int)
    n_cells = np.asarray(n_cells_per_source, dtype=int)
    # Within-block permutation keeps the historical fixed seed used by
    # permute_into_chunks; the caller seed only reorders source blocks.
    permutations = {
        i: permute_into_chunks(int(n_cells[i]), int(chunk_size[i]))
        for i in range(len(n_cells_per_source))
    }
    permutations_rows = {
        key: {i: x for i, x in enumerate(arrays)}
        for key, arrays in permutations.items()
    }

    coordinates: list[list[int]] = []
    extra: list[list[int]] = []
    for i in range(len(n_cells_per_source)):
        for j in range(len(permutations[i])):
            if j == len(permutations[i]) - 1:
                extra.append([i, j])
                continue
            coordinates.append([i, j])
    coordinates_permutations = rng.permutation(coordinates)
    if len(coordinates_permutations) > 0:
        coordinates_permutations = np.concatenate(
            [coordinates_permutations, extra],
            axis=0,
        )
    else:
        coordinates_permutations = np.array(extra, dtype=np.int64)

    return RowPlan(
        permutationsRows=permutations_rows,
        coordinatesPermutations=np.asarray(coordinates_permutations, dtype=np.int64),
        nCells=int(n_cells.sum()),
        sourceNames=tuple(source_names),
    )


def max_row_plan_block_rows(row_plan: RowPlan) -> int:
    return max(
        (
            int(rows.size)
            for blocks in row_plan.permutationsRows.values()
            for rows in blocks.values()
        ),
        default=0,
    )


def iter_row_plan_segments(
    row_plan: RowPlan,
    *,
    segment_rows: int | None = None,
) -> Iterator[RowPlanSegment]:
    """Yield destination-ordered source row segments from a row plan.

    Blocks fill the destination contiguously in ``coordinatesPermutations``
    order, so each block starts where the previous one ended.
    """
    if segment_rows is not None and int(segment_rows) < 1:
        raise ValueError("segment_rows must be positive")

    dest_start = 0
    for source_value, block_value in row_plan.coordinatesPermutations:
        source_idx = int(source_value)
        block_idx = int(block_value)
        local_rows = row_plan.permutationsRows[source_idx][block_idx]
        if local_rows.size:
            width = local_rows.size if segment_rows is None else int(segment_rows)
            for offset in range(0, local_rows.size, width):
                yield RowPlanSegment(
                    sourceIdx=source_idx,
                    blockIdx=block_idx,
                    destStart=dest_start + offset,
                    localRows=local_rows[offset : offset + width],
                )
        dest_start += int(local_rows.size)

    if dest_start != row_plan.nCells:
        raise AssertionError("Merged row plan does not cover every planned cell")


def iter_merged_cell_ids(
    row_plan: RowPlan,
    source_cell_tables: list[Any],
    *,
    dtype: Any,
    block_rows: int | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield contiguous merged cell-id blocks in destination row order.

    Each yield is ``(start, ids)`` where ``ids`` are already prefixed with
    ``{source_name}__``. Blocks follow ``coordinatesPermutations`` so a single
    generator can both write and verify cell identity.
    """
    if len(source_cell_tables) != len(row_plan.sourceNames):
        raise ValueError("Source cell tables must match the row plan")
    for segment in iter_row_plan_segments(row_plan, segment_rows=block_rows):
        source_idx = segment.sourceIdx
        source_ids = read_metadata_rows_chunkwise(
            source_cell_tables[source_idx],
            "ids",
            segment.localRows,
        )
        name = row_plan.sourceNames[source_idx]
        merged = np.empty(source_ids.size, dtype=np.dtype(dtype))
        for index, value in enumerate(source_ids):
            merged[index] = f"{name}__{value}"
        del source_ids
        yield segment.destStart, merged


def verify_merged_cell_ids(
    stored_ids: Any,
    row_plan: RowPlan,
    source_cell_tables: list[Any],
    *,
    block_rows: int = 100_000,
) -> None:
    """Compare stored cell ids against the row-plan identity generator.

    Stored ids are read one whole chunk band at a time. Segments arrive in
    destination order, so each band is read once.
    """
    n_cells = int(stored_ids.shape[0])
    if n_cells != row_plan.nCells:
        raise ValueError(
            "ERROR: order of cells does not match the one in existing file"
        )
    chunk_rows = max(1, int(stored_ids.chunks[0]))
    band_start = 0
    band = np.asarray(stored_ids[0:0])
    for start, expected in iter_merged_cell_ids(
        row_plan,
        source_cell_tables,
        dtype=stored_ids.dtype,
        block_rows=block_rows,
    ):
        offset = 0
        while offset < expected.size:
            row = start + offset
            if row >= band_start + band.size:
                band_start = row - row % chunk_rows
                band = np.asarray(
                    stored_ids[band_start : min(band_start + chunk_rows, n_cells)]
                )
            width = min(expected.size - offset, band_start + band.size - row)
            actual = band[row - band_start : row - band_start + width]
            if not np.array_equal(actual, expected[offset : offset + width]):
                raise ValueError(
                    "ERROR: order of cells does not match the one in existing file"
                )
            offset += width
